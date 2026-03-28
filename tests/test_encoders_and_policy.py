"""Tests for encoder and policy helper modules."""

from __future__ import annotations

import pytest
import torch

from src.agents.encoders import DenseGNNEncoder, FlattenEncoder, build_encoder
from src.agents.modules import GraphActorCritic as CompatGraphActorCritic
from src.agents.policy import GraphActorCritic


def _graph_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    """Build a small dense graph batch for model tests."""
    return {
        "node_features": torch.randn(batch_size, 4, 10),
        "node_mask": torch.ones(batch_size, 4),
        "adjacency": torch.ones(batch_size, 4, 4) - torch.eye(4).unsqueeze(0),
        "edge_features": torch.randn(batch_size, 4, 4, 4),
        "global_features": torch.randn(batch_size, 4),
    }


def test_build_encoder_returns_expected_module_types() -> None:
    """Encoder registry should return the configured implementation."""
    gnn = build_encoder("gnn", max_nodes=4, node_feature_dim=10, edge_feature_dim=4, global_feature_dim=4, hidden_dim=16, message_passing_steps=2)
    mlp = build_encoder("mlp", max_nodes=4, node_feature_dim=10, edge_feature_dim=4, global_feature_dim=4, hidden_dim=16, message_passing_steps=2)

    assert isinstance(gnn, DenseGNNEncoder)
    assert isinstance(mlp, FlattenEncoder)


def test_build_encoder_rejects_unknown_type() -> None:
    """Unknown encoder keys should fail fast."""
    with pytest.raises(ValueError, match="Unsupported encoder type"):
        build_encoder(
            "bad",
            max_nodes=4,
            node_feature_dim=10,
            edge_feature_dim=4,
            global_feature_dim=4,
            hidden_dim=16,
            message_passing_steps=2,
        )


def test_encoders_and_policy_produce_expected_shapes() -> None:
    """Encoders and policy heads should produce finite tensors with stable shapes."""
    batch = _graph_batch()

    encoder = DenseGNNEncoder(
        node_feature_dim=10,
        edge_feature_dim=4,
        global_feature_dim=4,
        hidden_dim=16,
        message_passing_steps=2,
    )
    embedding = encoder(batch)
    assert embedding.shape == (2, 16)

    policy = GraphActorCritic(
        encoder_type="gnn",
        max_nodes=4,
        node_feature_dim=10,
        edge_feature_dim=4,
        global_feature_dim=4,
        hidden_dim=16,
        message_passing_steps=2,
        action_dim=4,
        action_low=torch.full((4,), -1.0),
        action_high=torch.full((4,), 1.0),
        action_std_init=0.3,
    )
    action, log_prob, value = policy.act(batch, deterministic=False)
    assert action.shape == (2, 4)
    assert log_prob.shape == (2,)
    assert value.shape == (2,)
    assert torch.isfinite(action).all()

    compat_policy = CompatGraphActorCritic(
        encoder_type="mlp",
        max_nodes=4,
        node_feature_dim=10,
        edge_feature_dim=4,
        global_feature_dim=4,
        hidden_dim=16,
        message_passing_steps=2,
        action_dim=4,
        action_low=torch.full((4,), -1.0),
        action_high=torch.full((4,), 1.0),
        action_std_init=0.3,
    )
    compat_distribution, compat_value = compat_policy.distribution_and_value(batch)
    assert compat_distribution.mean.shape == (2, 4)
    assert compat_value.shape == (2,)
