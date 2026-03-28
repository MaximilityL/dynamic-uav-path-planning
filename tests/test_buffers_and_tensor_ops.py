"""Tests for agent buffer and tensor helper modules."""

from __future__ import annotations

import numpy as np
import torch

from src.agents.buffers import EpisodeBuffer, RolloutBuffer
from src.agents.tensor_ops import batch_slice, single_graph_observation, stack_graph_observations


def _sample_observation(scale: float = 1.0) -> dict[str, np.ndarray]:
    """Build a tiny graph observation payload for helper tests."""
    return {
        "node_features": np.full((3, 2), scale, dtype=np.float32),
        "node_mask": np.ones((3,), dtype=np.float32),
        "adjacency": np.eye(3, dtype=np.float32),
        "edge_features": np.full((3, 3, 1), scale, dtype=np.float32),
        "global_features": np.asarray([scale, scale + 1.0], dtype=np.float32),
    }


def test_episode_and_rollout_buffers_clear_state() -> None:
    """Buffers should drop all retained rollout state when cleared."""
    episode = EpisodeBuffer(
        observations=[_sample_observation()],
        actions=[np.zeros(4, dtype=np.float32)],
        rewards=[1.0],
        dones=[0.0],
        log_probs=[-0.1],
        values=[0.2],
    )
    rollout = RolloutBuffer(
        observations=[_sample_observation()],
        actions=[np.ones(4, dtype=np.float32)],
        log_probs=[-0.2],
        returns=[1.5],
        advantages=[0.5],
    )

    assert len(rollout) == 1
    episode.clear()
    rollout.clear()

    assert episode.observations == []
    assert rollout.actions == []
    assert len(rollout) == 0


def test_tensor_helpers_stack_slice_and_single_wrap() -> None:
    """Tensor helpers should preserve graph keys and expected batch shapes."""
    observations = [_sample_observation(1.0), _sample_observation(2.0)]
    device = torch.device("cpu")

    single = single_graph_observation(observations[0], device)
    assert single["node_features"].shape == (1, 3, 2)

    stacked = stack_graph_observations(observations, device)
    assert stacked["node_features"].shape == (2, 3, 2)
    assert stacked["global_features"].shape == (2, 2)

    subset = batch_slice(stacked, torch.tensor([1], dtype=torch.long))
    assert subset["node_features"].shape == (1, 3, 2)
    assert torch.allclose(subset["global_features"][0], torch.tensor([2.0, 3.0]))
