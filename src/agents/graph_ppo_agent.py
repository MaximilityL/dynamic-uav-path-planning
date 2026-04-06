"""Minimal PPO-style agent for dense graph observations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from .buffers import EpisodeBuffer, RolloutBuffer
from .policy import GraphActorCritic
from .tensor_ops import (
    GraphObservation,
    batch_slice,
    single_graph_observation,
    stack_graph_observations,
)


class GraphPPOAgent:
    """Minimal PPO agent with a pluggable graph encoder."""

    def __init__(
        self,
        *,
        encoder_type: str,
        max_nodes: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        global_feature_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int,
        message_passing_steps: int,
        lr_actor: float,
        lr_critic: float,
        gamma: float,
        gae_lambda: float,
        clip_epsilon: float,
        entropy_coef: float,
        value_coef: float,
        ppo_epochs: int,
        mini_batch_size: int,
        max_grad_norm: float,
        action_std_init: float,
        device: str,
    ) -> None:
        self.device = torch.device(device)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_epsilon = float(clip_epsilon)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.ppo_epochs = int(ppo_epochs)
        self.mini_batch_size = int(mini_batch_size)
        self.max_grad_norm = float(max_grad_norm)
        self._default_actor_lr = float(lr_actor)
        self._default_critic_lr = float(lr_critic)

        self.model = GraphActorCritic(
            encoder_type=encoder_type,
            max_nodes=max_nodes,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=hidden_dim,
            message_passing_steps=message_passing_steps,
            action_dim=action_dim,
            action_low=torch.as_tensor(action_low, dtype=torch.float32),
            action_high=torch.as_tensor(action_high, dtype=torch.float32),
            action_std_init=action_std_init,
        ).to(self.device)
        self.optimizer = self._build_optimizer(actor_lr=self._default_actor_lr, critic_lr=self._default_critic_lr)

        self._episode_buffer = EpisodeBuffer()
        self._rollout_buffer = RolloutBuffer()

    def _build_optimizer(self, *, actor_lr: float, critic_lr: float) -> torch.optim.Adam:
        """Create a fresh optimizer for the current model parameters."""
        actor_parameters = list(self.model.encoder.parameters()) + list(self.model.policy_head.parameters()) + [self.model.log_std]
        critic_parameters = list(self.model.value_head.parameters())
        return torch.optim.Adam(
            [
                {"params": actor_parameters, "lr": float(actor_lr)},
                {"params": critic_parameters, "lr": float(critic_lr)},
            ]
        )

    def learning_rates(self) -> Dict[str, float]:
        """Return the actor/critic learning rates currently in use."""
        return {
            "actor": float(self.optimizer.param_groups[0]["lr"]),
            "critic": float(self.optimizer.param_groups[1]["lr"]),
        }

    def default_learning_rates(self) -> Dict[str, float]:
        """Return the base actor/critic learning rates configured for fresh resets."""
        return {
            "actor": float(self._default_actor_lr),
            "critic": float(self._default_critic_lr),
        }

    def set_learning_rates(self, *, actor_lr: float | None = None, critic_lr: float | None = None) -> Dict[str, float]:
        """Update the optimizer learning rates in place."""
        current = self.learning_rates()
        self.optimizer.param_groups[0]["lr"] = float(current["actor"] if actor_lr is None else actor_lr)
        self.optimizer.param_groups[1]["lr"] = float(current["critic"] if critic_lr is None else critic_lr)
        return self.learning_rates()

    def scale_learning_rates(self, multiplier: float) -> Dict[str, float]:
        """Scale the actor and critic learning rates together."""
        factor = float(multiplier)
        current = self.learning_rates()
        return self.set_learning_rates(
            actor_lr=current["actor"] * factor,
            critic_lr=current["critic"] * factor,
        )

    def reset_optimizer(self, *, actor_lr: float | None = None, critic_lr: float | None = None) -> Dict[str, float]:
        """Discard optimizer momentum/state while keeping the current model weights."""
        current = self.learning_rates()
        self.optimizer = self._build_optimizer(
            actor_lr=current["actor"] if actor_lr is None else actor_lr,
            critic_lr=current["critic"] if critic_lr is None else critic_lr,
        )
        return self.learning_rates()

    def action_std(self) -> np.ndarray:
        """Return the current policy action standard deviation."""
        with torch.no_grad():
            return torch.exp(self.model.log_std).detach().cpu().numpy().astype(np.float32)

    def set_action_std(self, action_std: float | np.ndarray) -> np.ndarray:
        """Overwrite the policy action standard deviation in-place."""
        std_array = np.asarray(action_std, dtype=np.float32)
        if std_array.ndim == 0:
            std_array = np.full(self.model.log_std.shape, float(std_array), dtype=np.float32)
        if tuple(std_array.shape) != tuple(self.model.log_std.shape):
            raise ValueError(
                f"action_std shape {tuple(std_array.shape)} does not match policy shape {tuple(self.model.log_std.shape)}"
            )
        clamped = np.clip(std_array, 1e-4, None)
        log_std_tensor = torch.as_tensor(np.log(clamped), dtype=self.model.log_std.dtype, device=self.model.log_std.device)
        with torch.no_grad():
            self.model.log_std.copy_(log_std_tensor)
        return self.action_std()

    def clear_rollout_buffers(self) -> None:
        """Drop any partially collected rollout data."""
        self._episode_buffer.clear()
        self._rollout_buffer.clear()

    def select_action(self, observation: GraphObservation, deterministic: bool = False) -> tuple[np.ndarray, Dict[str, float]]:
        """Choose an action and return rollout metadata."""
        observation_batch = single_graph_observation(observation, self.device)
        with torch.no_grad():
            action, log_prob, value = self.model.act(observation_batch, deterministic=deterministic)
        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            {
                "log_prob": float(log_prob.item()),
                "value": float(value.item()),
            },
        )

    def estimate_value(self, observation: GraphObservation) -> float:
        """Estimate the state value without sampling an action."""
        observation_batch = single_graph_observation(observation, self.device)
        with torch.no_grad():
            _, value = self.model.distribution_and_value(observation_batch)
        return float(value.item())

    def evaluate_action(self, observation: GraphObservation, action: np.ndarray) -> Dict[str, float]:
        """Score an externally provided action under the current policy."""
        observation_batch = single_graph_observation(observation, self.device)
        action_tensor = torch.as_tensor(np.asarray(action, dtype=np.float32), dtype=torch.float32, device=self.device)
        action_tensor = action_tensor.unsqueeze(0)
        with torch.no_grad():
            log_prob, _, value = self.model.evaluate_actions(observation_batch, action_tensor)
        return {
            "log_prob": float(log_prob.item()),
            "value": float(value.item()),
        }

    def store_transition(
        self,
        *,
        observation: GraphObservation,
        action: np.ndarray,
        teacher_action: np.ndarray | None = None,
        bc_mask: float = 0.0,
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Append one transition to the current episode buffer."""
        self._episode_buffer.observations.append({key: np.asarray(val, dtype=np.float32).copy() for key, val in observation.items()})
        self._episode_buffer.actions.append(np.asarray(action, dtype=np.float32).copy())
        if teacher_action is None:
            teacher_action = np.zeros_like(action, dtype=np.float32)
        self._episode_buffer.teacher_actions.append(np.asarray(teacher_action, dtype=np.float32).copy())
        self._episode_buffer.bc_masks.append(float(bc_mask))
        self._episode_buffer.rewards.append(float(reward))
        self._episode_buffer.dones.append(float(done))
        self._episode_buffer.log_probs.append(float(log_prob))
        self._episode_buffer.values.append(float(value))

    def finish_rollout(self, last_value: float) -> None:
        """Finalize GAE targets for the current episode and move them into the rollout buffer."""
        rewards = self._episode_buffer.rewards
        if not rewards:
            return

        values = self._episode_buffer.values + [float(last_value)]
        advantages = [0.0] * len(rewards)
        gae = 0.0
        for timestep in reversed(range(len(rewards))):
            mask = 1.0 - self._episode_buffer.dones[timestep]
            delta = rewards[timestep] + self.gamma * values[timestep + 1] * mask - values[timestep]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[timestep] = gae
        returns = [advantage + value for advantage, value in zip(advantages, self._episode_buffer.values)]

        self._rollout_buffer.observations.extend(self._episode_buffer.observations)
        self._rollout_buffer.actions.extend(self._episode_buffer.actions)
        self._rollout_buffer.teacher_actions.extend(self._episode_buffer.teacher_actions)
        self._rollout_buffer.bc_masks.extend(self._episode_buffer.bc_masks)
        self._rollout_buffer.log_probs.extend(self._episode_buffer.log_probs)
        self._rollout_buffer.returns.extend(returns)
        self._rollout_buffer.advantages.extend(advantages)
        self._episode_buffer.clear()

    def has_pending_rollout(self) -> bool:
        """Report whether there is data waiting for an update."""
        return len(self._rollout_buffer) > 0

    def _bc_action_representation(self, actions: torch.Tensor, *, normalize: bool) -> torch.Tensor:
        """Project actions into the representation used by BC loss."""
        if not normalize:
            return actions

        normalized = actions.clone()
        direction_scale = torch.maximum(self.model.action_high[:-1].abs(), self.model.action_low[:-1].abs()).clamp_min(1e-6)
        normalized[..., :-1] = normalized[..., :-1] / direction_scale
        normalized[..., -1] = normalized[..., -1] / self.model.speed_scale.clamp_min(1e-6)
        return normalized

    def update(
        self,
        *,
        bc_coef: float = 0.0,
        normalize_bc_target_action: bool = False,
    ) -> Dict[str, float]:
        """Run a PPO update over the accumulated rollout data."""
        if not self.has_pending_rollout():
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "bc_loss": 0.0,
                "bc_nonzero": 0.0,
                "bc_coef": float(bc_coef),
                "bc_active_fraction": 0.0,
                "bc_active_samples": 0.0,
                "bc_total_samples": 0.0,
            }

        observations = stack_graph_observations(self._rollout_buffer.observations, self.device)
        actions = torch.as_tensor(np.asarray(self._rollout_buffer.actions), dtype=torch.float32, device=self.device)
        teacher_actions = torch.as_tensor(np.asarray(self._rollout_buffer.teacher_actions), dtype=torch.float32, device=self.device)
        bc_masks = torch.as_tensor(np.asarray(self._rollout_buffer.bc_masks), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(self._rollout_buffer.log_probs), dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(np.asarray(self._rollout_buffer.returns), dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(np.asarray(self._rollout_buffer.advantages), dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)

        batch_size = int(actions.shape[0])
        bc_coef = float(max(bc_coef, 0.0))
        bc_active_fraction = float(bc_masks.mean().item()) if batch_size > 0 else 0.0
        bc_active_samples = float(bc_masks.sum().item()) if batch_size > 0 else 0.0
        actor_losses: List[float] = []
        critic_losses: List[float] = []
        entropies: List[float] = []
        bc_losses: List[float] = []

        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, self.mini_batch_size):
                indices = permutation[start : start + self.mini_batch_size]
                batch_obs = batch_slice(observations, indices)
                batch_actions = actions.index_select(0, indices)
                batch_teacher_actions = teacher_actions.index_select(0, indices)
                batch_bc_masks = bc_masks.index_select(0, indices)
                batch_old_log_probs = old_log_probs.index_select(0, indices)
                batch_returns = returns.index_select(0, indices)
                batch_advantages = advantages.index_select(0, indices)

                distribution, values = self.model.distribution_and_value(batch_obs)
                new_log_probs = distribution.log_prob(batch_actions).sum(dim=-1)
                entropy = distribution.entropy().sum(dim=-1)
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                unclipped = ratios * batch_advantages
                clipped = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = torch.nn.functional.mse_loss(values, batch_returns)
                entropy_mean = entropy.mean()
                bc_loss = torch.zeros((), device=self.device)
                if bc_coef > 0.0 and float(batch_bc_masks.sum().item()) > 0.0:
                    policy_mean = self._bc_action_representation(distribution.mean, normalize=normalize_bc_target_action)
                    teacher_target = self._bc_action_representation(
                        batch_teacher_actions,
                        normalize=normalize_bc_target_action,
                    )
                    per_sample_bc_loss = torch.mean((policy_mean - teacher_target) ** 2, dim=-1)
                    bc_loss = (per_sample_bc_loss * batch_bc_masks).sum() / batch_bc_masks.sum().clamp_min(1.0)
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_mean + bc_coef * bc_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy_mean.item()))
                bc_losses.append(float(bc_loss.item()))

        self._rollout_buffer.clear()
        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "bc_loss": float(np.mean(bc_losses)) if bc_losses else 0.0,
            "bc_nonzero": float(any(loss > 0.0 for loss in bc_losses)),
            "bc_coef": float(bc_coef),
            "bc_active_fraction": bc_active_fraction,
            "bc_active_samples": bc_active_samples,
            "bc_total_samples": float(batch_size),
        }

    def behavior_clone_pretrain(
        self,
        *,
        observations: List[GraphObservation],
        target_actions: List[np.ndarray],
        epochs: int,
        batch_size: int,
        normalize_target_action: bool = False,
    ) -> Dict[str, float]:
        """Run supervised actor pretraining against teacher actions."""
        if not observations or not target_actions:
            return {
                "bc_pretrain_loss": 0.0,
                "bc_pretrain_epochs": int(max(epochs, 0)),
                "bc_pretrain_samples": 0.0,
            }

        observations_tensor = stack_graph_observations(observations, self.device)
        target_actions_tensor = torch.as_tensor(np.asarray(target_actions), dtype=torch.float32, device=self.device)
        resolved_batch_size = max(int(batch_size), 1)
        resolved_epochs = max(int(epochs), 1)
        dataset_size = int(target_actions_tensor.shape[0])
        losses: List[float] = []

        for _ in range(resolved_epochs):
            permutation = torch.randperm(dataset_size, device=self.device)
            for start in range(0, dataset_size, resolved_batch_size):
                indices = permutation[start : start + resolved_batch_size]
                batch_obs = batch_slice(observations_tensor, indices)
                batch_targets = target_actions_tensor.index_select(0, indices)
                distribution, _ = self.model.distribution_and_value(batch_obs)
                predicted_actions = self._bc_action_representation(
                    distribution.mean,
                    normalize=normalize_target_action,
                )
                target_actions_batch = self._bc_action_representation(
                    batch_targets,
                    normalize=normalize_target_action,
                )
                loss = torch.mean((predicted_actions - target_actions_batch) ** 2)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses.append(float(loss.item()))

        return {
            "bc_pretrain_loss": float(np.mean(losses)) if losses else 0.0,
            "bc_pretrain_epochs": float(resolved_epochs),
            "bc_pretrain_samples": float(dataset_size),
        }

    def save(self, path: str | Path, metadata: Dict[str, object] | None = None) -> Path:
        """Persist the policy and optimizer state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": None,
                "metadata": metadata or {},
            },
            path,
        )
        return path

    def load(
        self,
        path: str | Path,
        *,
        load_optimizer_state: bool = True,
        load_scheduler_state: bool = True,
        reset_optimizer_if_skipped: bool = False,
    ) -> Dict[str, object]:
        """Restore the policy and optionally the optimizer state."""
        del load_scheduler_state
        payload = torch.load(Path(path), map_location=self.device)
        try:
            self.model.load_state_dict(payload["model_state"])
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load checkpoint because the checkpoint architecture does not match the current model. "
                "When resuming training, make sure settings like `agent.hidden_dim` match the checkpoint you are loading.\n"
                f"Original error:\n{exc}"
            ) from exc
        optimizer_state = payload.get("optimizer_state")
        if load_optimizer_state and optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        elif reset_optimizer_if_skipped:
            self.reset_optimizer()
        return payload.get("metadata", {})
