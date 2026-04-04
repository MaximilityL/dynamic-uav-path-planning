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

        actor_parameters = list(self.model.encoder.parameters()) + list(self.model.policy_head.parameters()) + [self.model.log_std]
        critic_parameters = list(self.model.value_head.parameters())
        self.optimizer = torch.optim.Adam(
            [
                {"params": actor_parameters, "lr": lr_actor},
                {"params": critic_parameters, "lr": lr_critic},
            ]
        )

        self._episode_buffer = EpisodeBuffer()
        self._rollout_buffer = RolloutBuffer()

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
        reward: float,
        done: bool,
        log_prob: float,
        value: float,
    ) -> None:
        """Append one transition to the current episode buffer."""
        self._episode_buffer.observations.append({key: np.asarray(val, dtype=np.float32).copy() for key, val in observation.items()})
        self._episode_buffer.actions.append(np.asarray(action, dtype=np.float32).copy())
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
        self._rollout_buffer.log_probs.extend(self._episode_buffer.log_probs)
        self._rollout_buffer.returns.extend(returns)
        self._rollout_buffer.advantages.extend(advantages)
        self._episode_buffer.clear()

    def has_pending_rollout(self) -> bool:
        """Report whether there is data waiting for an update."""
        return len(self._rollout_buffer) > 0

    def update(self) -> Dict[str, float]:
        """Run a PPO update over the accumulated rollout data."""
        if not self.has_pending_rollout():
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        observations = stack_graph_observations(self._rollout_buffer.observations, self.device)
        actions = torch.as_tensor(np.asarray(self._rollout_buffer.actions), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(self._rollout_buffer.log_probs), dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(np.asarray(self._rollout_buffer.returns), dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(np.asarray(self._rollout_buffer.advantages), dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)

        batch_size = int(actions.shape[0])
        actor_losses: List[float] = []
        critic_losses: List[float] = []
        entropies: List[float] = []

        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, self.mini_batch_size):
                indices = permutation[start : start + self.mini_batch_size]
                batch_obs = batch_slice(observations, indices)
                batch_actions = actions.index_select(0, indices)
                batch_old_log_probs = old_log_probs.index_select(0, indices)
                batch_returns = returns.index_select(0, indices)
                batch_advantages = advantages.index_select(0, indices)

                new_log_probs, entropy, values = self.model.evaluate_actions(batch_obs, batch_actions)
                ratios = torch.exp(new_log_probs - batch_old_log_probs)
                unclipped = ratios * batch_advantages
                clipped = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = torch.nn.functional.mse_loss(values, batch_returns)
                entropy_mean = entropy.mean()
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_mean

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                actor_losses.append(float(actor_loss.item()))
                critic_losses.append(float(critic_loss.item()))
                entropies.append(float(entropy_mean.item()))

        self._rollout_buffer.clear()
        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    def save(self, path: str | Path, metadata: Dict[str, object] | None = None) -> Path:
        """Persist the policy and optimizer state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )
        return path

    def load(self, path: str | Path) -> Dict[str, object]:
        """Restore the policy and optimizer state."""
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
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        return payload.get("metadata", {})
