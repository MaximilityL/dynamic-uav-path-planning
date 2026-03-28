"""Tests for the dynamic airspace environment."""

from __future__ import annotations

import numpy as np

from src.training.runner import create_environment
from src.utils.config import load_config


def test_environment_reset_and_step_shapes() -> None:
    """Environment reset and step should produce stable graph shapes."""
    config = load_config("configs/default.yaml")
    config.environment.num_dynamic_obstacles = 3
    config.environment.max_episode_steps = 10

    env = create_environment(config, gui=False, seed=5)
    try:
        observation, info = env.reset(seed=5)
        assert observation["node_features"].shape == (env.max_nodes, env.node_feature_dim)
        assert observation["adjacency"].shape == (env.max_nodes, env.max_nodes)
        assert observation["edge_features"].shape == (env.max_nodes, env.max_nodes, env.edge_feature_dim)
        assert observation["global_features"].shape == (env.global_feature_dim,)
        assert "distance_to_goal" in info

        next_observation, reward, terminated, truncated, step_info = env.step(env.action_space.sample().astype(np.float32))
        assert next_observation["node_mask"].shape == (env.max_nodes,)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "reward_components" in step_info
    finally:
        env.close()
