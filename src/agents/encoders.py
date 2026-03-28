"""Graph encoder modules."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn

GraphTensorBatch = Dict[str, torch.Tensor]


def _masked_mean(node_embeddings: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Average node embeddings while ignoring masked-out nodes."""
    mask = node_mask.unsqueeze(-1)
    total = (node_embeddings * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return total / denom


class FlattenEncoder(nn.Module):
    """Simple fallback encoder that flattens the dense graph observation."""

    def __init__(
        self,
        max_nodes: int,
        node_feature_dim: int,
        edge_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        flat_dim = (
            max_nodes * node_feature_dim
            + max_nodes
            + max_nodes * max_nodes
            + max_nodes * max_nodes * edge_feature_dim
            + global_feature_dim
        )
        self.output_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, observation: GraphTensorBatch) -> torch.Tensor:
        """Encode a dense graph observation with a flat MLP."""
        flat = torch.cat(
            [
                observation["node_features"].flatten(start_dim=1),
                observation["node_mask"],
                observation["adjacency"].flatten(start_dim=1),
                observation["edge_features"].flatten(start_dim=1),
                observation["global_features"],
            ],
            dim=1,
        )
        return self.network(flat)


class DenseGNNEncoder(nn.Module):
    """Lightweight dense message-passing encoder implemented in plain PyTorch."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        global_feature_dim: int,
        hidden_dim: int,
        message_passing_steps: int,
    ) -> None:
        super().__init__()
        self.message_passing_steps = int(message_passing_steps)
        self.output_dim = hidden_dim

        self.node_input = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.Tanh(),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2 + global_feature_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(self, observation: GraphTensorBatch) -> torch.Tensor:
        """Encode a graph observation into a fixed-size embedding."""
        node_features = observation["node_features"]
        node_mask = observation["node_mask"]
        adjacency = observation["adjacency"]
        edge_features = observation["edge_features"]
        global_features = observation["global_features"]

        hidden = self.node_input(node_features) * node_mask.unsqueeze(-1)
        num_nodes = hidden.shape[1]

        for _ in range(self.message_passing_steps):
            receiver = hidden.unsqueeze(2).expand(-1, num_nodes, num_nodes, -1)
            sender = hidden.unsqueeze(1).expand(-1, num_nodes, num_nodes, -1)
            pair_features = torch.cat([receiver, sender, edge_features], dim=-1)
            messages = self.message_mlp(pair_features) * adjacency.unsqueeze(-1)
            degrees = adjacency.sum(dim=2, keepdim=True).clamp_min(1.0)
            aggregated = messages.sum(dim=2) / degrees
            updated = self.update_mlp(torch.cat([hidden, aggregated], dim=-1))
            hidden = updated * node_mask.unsqueeze(-1)

        pooled = _masked_mean(hidden, node_mask)
        ego_embedding = hidden[:, 0, :]
        return self.output_projection(torch.cat([pooled, ego_embedding, global_features], dim=-1))


def build_encoder(
    encoder_type: str,
    max_nodes: int,
    node_feature_dim: int,
    edge_feature_dim: int,
    global_feature_dim: int,
    hidden_dim: int,
    message_passing_steps: int,
) -> nn.Module:
    """Build an encoder from a small registry."""
    normalized = encoder_type.strip().lower()
    if normalized == "gnn":
        return DenseGNNEncoder(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=hidden_dim,
            message_passing_steps=message_passing_steps,
        )
    if normalized == "mlp":
        return FlattenEncoder(
            max_nodes=max_nodes,
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            global_feature_dim=global_feature_dim,
            hidden_dim=hidden_dim,
        )
    raise ValueError(f"Unsupported encoder type: {encoder_type}")
