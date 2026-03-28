"""Tests for environment helper modules outside the main env class."""

from __future__ import annotations

import numpy as np

from src.environments.observation import build_dense_graph_observation, minimum_obstacle_distance
from src.environments.reward import RewardWeights, compute_reward
from src.environments.scenario import (
    advance_obstacles,
    sample_dynamic_obstacles,
    sample_goal_position,
    sample_start_position,
)


def test_scenario_sampling_and_bounce_dynamics() -> None:
    """Scenario helpers should respect shapes and bounce obstacles at bounds."""
    rng = np.random.default_rng(7)
    workspace_bounds = np.asarray([[-3.0, 3.0], [-3.0, 3.0], [0.5, 2.5]], dtype=np.float32)

    start = sample_start_position(rng, workspace_bounds)
    goal = sample_goal_position(rng, workspace_bounds, start)
    positions, velocities = sample_dynamic_obstacles(
        rng,
        workspace_bounds,
        num_dynamic_obstacles=3,
        obstacle_radius=0.2,
        goal_tolerance=0.3,
        collision_distance=0.15,
        obstacle_speed_range=(0.15, 0.45),
        start_position=start,
        goal_position=goal,
    )

    assert start.shape == (3,)
    assert goal.shape == (3,)
    assert positions.shape == (3, 3)
    assert velocities.shape == (3, 3)

    next_positions, next_velocities = advance_obstacles(
        obstacle_positions=np.asarray([[2.75, 0.0, 1.0]], dtype=np.float32),
        obstacle_velocities=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        workspace_bounds=workspace_bounds,
        obstacle_radius=0.2,
        dt=0.5,
    )
    assert np.isclose(next_positions[0, 0], 2.8)
    assert next_velocities[0, 0] < 0.0


def test_observation_and_reward_helpers_handle_edge_cases() -> None:
    """Observation and reward helpers should stay finite in common edge cases."""
    workspace_bounds = np.asarray([[-3.0, 3.0], [-3.0, 3.0], [0.5, 2.5]], dtype=np.float32)
    drone_position = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    drone_velocity = np.asarray([0.1, 0.0, 0.0], dtype=np.float32)
    goal_position = np.asarray([1.0, 1.0, 1.2], dtype=np.float32)
    obstacle_positions = np.asarray([[0.4, 0.0, 1.0], [1.5, 0.2, 1.1]], dtype=np.float32)
    obstacle_velocities = np.asarray([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]], dtype=np.float32)

    observation = build_dense_graph_observation(
        drone_position=drone_position,
        drone_velocity=drone_velocity,
        goal_position=goal_position,
        obstacle_positions=obstacle_positions,
        obstacle_velocities=obstacle_velocities,
        obstacle_radius=0.2,
        goal_tolerance=0.3,
        workspace_bounds=workspace_bounds,
        obstacle_speed_range=(0.15, 0.45),
        connect_radius=4.0,
        current_step=5,
        max_episode_steps=100,
    )
    assert observation["node_features"].shape == (4, 10)
    assert observation["adjacency"][0, 1] == 1.0
    assert observation["global_features"].shape == (4,)

    fallback_distance = minimum_obstacle_distance(
        drone_position=drone_position,
        obstacle_positions=np.zeros((0, 3), dtype=np.float32),
        default_distance=4.0,
    )
    assert fallback_distance == 4.0

    reward, components = compute_reward(
        reward_weights=RewardWeights(),
        goal_now=True,
        collision_now=False,
        progress_delta=0.5,
        clearance_margin=3.0,
        action_array=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert components["goal"] > 0.0
    assert components["clearance"] <= RewardWeights().clearance
    assert np.isfinite(reward)
