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


def test_environment_can_expand_time_budget_from_distance() -> None:
    """Automatic time budgeting should expand short configured horizons when enabled."""
    config = load_config("configs/default.yaml")
    config.environment.max_episode_steps = 5
    config.environment.auto_time_budget_steps_per_meter = 200.0
    config.environment.auto_time_budget_padding = 10

    env = create_environment(config, gui=False, seed=7)
    try:
        env.reset(seed=7)
        assert env.max_episode_steps > 5
    finally:
        env.close()


def test_teacher_bonus_budget_caps_dense_guidance_reward() -> None:
    """Teacher shaping should stay within the configured per-episode budget."""
    config = load_config("configs/default.yaml")
    config.environment.max_episode_steps = 12
    config.environment.num_dynamic_obstacles = 2
    config.environment.seed = 9
    config.environment.teacher_config = {
        "enabled": True,
        "reward_scale": 0.5,
        "episode_bonus_budget": 0.2,
        "repulsion_radius": 1.0,
        "repulsion_gain": 0.5,
        "far_speed": 0.9,
        "near_speed": 0.5,
        "near_goal_distance": 1.0,
    }

    env = create_environment(config, gui=False, seed=9)
    try:
        env.reset(seed=9)
        total_teacher_bonus = 0.0
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = env.teacher_action_for_current_state()
            _, _, terminated, truncated, info = env.step(action)
            total_teacher_bonus += abs(float(info["reward_components"].get("teacher", 0.0)))
        assert total_teacher_bonus <= 0.200001
    finally:
        env.close()
