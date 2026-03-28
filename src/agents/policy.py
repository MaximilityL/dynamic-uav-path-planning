"""Actor-critic policy modules for dense graph observations."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal

from .encoders import build_encoder
from .tensor_ops import GraphTensorBatch


class GraphActorCritic(nn.Module):
    """Actor-critic policy that consumes dense graph observations."""

    def __init__(
        self,
        *,
        encoder_type: str,
        max_nodes: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int,
        message_passing_steps: int,
        action_dim: int,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        action_std_init: float,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(
            encoder_type=encoder_type,
            max_nodes=max_nodes,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=hidden_dim,
            message_passing_steps=message_passing_steps,
        )
        self.policy_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(action_std_init)))

        self.register_buffer("action_low", action_low.float())
        self.register_buffer("action_high", action_high.float())
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2.0)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2.0)

    def distribution_and_value(self, observation: GraphTensorBatch) -> tuple[Normal, torch.Tensor]:
        """Build the action distribution and value estimate."""
        embedding = self.encoder(observation)
        mean = torch.tanh(self.policy_head(embedding)) * self.action_scale + self.action_bias
        std = torch.exp(self.log_std).unsqueeze(0).expand_as(mean)
        distribution = Normal(mean, std)
        value = self.value_head(embedding).squeeze(-1)
        return distribution, value

    def act(self, observation: GraphTensorBatch, deterministic: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or choose an action."""
        distribution, value = self.distribution_and_value(observation)
        action = distribution.mean if deterministic else distribution.rsample()
        action = torch.clamp(action, self.action_low, self.action_high)
        log_prob = distribution.log_prob(action).sum(dim=-1)
        return action, log_prob, value

    def evaluate_actions(self, observation: GraphTensorBatch, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given actions under the current policy."""
        distribution, value = self.distribution_and_value(observation)
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy, value
