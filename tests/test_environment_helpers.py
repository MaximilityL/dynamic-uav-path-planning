"""Tests for environment helper modules outside the main env class."""

from __future__ import annotations

import numpy as np

from src.environments.observation import build_dense_graph_observation, minimum_obstacle_distance
from src.environments.dynamic_airspace_env import DynamicAirspaceEnv
from src.environments.reward import RewardWeights, compute_reward
from src.environments.scenario import (
    advance_obstacles,
    sample_dynamic_obstacles,
    sample_goal_position,
    sample_start_position,
)
from src.environments.teacher import heuristic_teacher_action, teacher_alignment_bonus, teacher_guided_action


def _distance_to_segment(point: np.ndarray, segment_start: np.ndarray, segment_end: np.ndarray) -> float:
    """Compute the Euclidean distance from a point to a line segment."""
    segment = segment_end - segment_start
    segment_norm = float(np.dot(segment, segment))
    if segment_norm <= 0.0:
        return float(np.linalg.norm(point - segment_start))
    projection = float(np.clip(np.dot(point - segment_start, segment) / segment_norm, 0.0, 1.0))
    closest = segment_start + projection * segment
    return float(np.linalg.norm(point - closest))


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


def test_goal_sampling_can_stay_close_in_lateral_axes() -> None:
    """Relative goal windows should keep easier-stage goals near the start laterally."""
    rng = np.random.default_rng(17)
    workspace_bounds = np.asarray([[-3.0, 3.0], [-3.0, 3.0], [0.5, 2.5]], dtype=np.float32)
    scenario_config = {
        "goal_relative_y_range": (-0.4, 0.4),
        "goal_relative_z_range": (-0.2, 0.2),
    }

    start = sample_start_position(rng, workspace_bounds, scenario_config)
    goal = sample_goal_position(rng, workspace_bounds, start, scenario_config)
    assert abs(float(goal[1] - start[1])) <= 0.41
    assert abs(float(goal[2] - start[2])) <= 0.21


def test_route_biased_obstacle_sampling_can_force_bridge_interactions() -> None:
    """Route-biased obstacle placement should stay near the nominal path when configured."""
    rng = np.random.default_rng(23)
    workspace_bounds = np.asarray([[-3.0, 3.0], [-3.0, 3.0], [0.5, 2.5]], dtype=np.float32)
    start = np.asarray([-2.0, 0.0, 1.0], dtype=np.float32)
    goal = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)

    positions, _ = sample_dynamic_obstacles(
        rng,
        workspace_bounds,
        num_dynamic_obstacles=2,
        obstacle_radius=0.2,
        goal_tolerance=0.95,
        collision_distance=0.15,
        obstacle_speed_range=(0.15, 0.15),
        start_position=start,
        goal_position=goal,
        scenario_config={
            "route_obstacle_count": 2,
            "route_obstacle_progress_range": (0.35, 0.65),
            "route_obstacle_lateral_offset_range": (-0.20, 0.20),
            "route_obstacle_vertical_offset_range": (-0.10, 0.10),
            "route_obstacle_longitudinal_jitter": 0.04,
            "obstacle_start_goal_exclusion_radius": 0.7,
        },
    )

    path_distances = [_distance_to_segment(position, start, goal) for position in positions]
    assert max(path_distances) < 0.35
    assert min(float(np.linalg.norm(position - start)) for position in positions) >= 0.7
    assert min(float(np.linalg.norm(position - goal)) for position in positions) >= 0.7


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
        collision_distance=0.15,
        workspace_bounds=workspace_bounds,
        obstacle_speed_range=(0.15, 0.45),
        connect_radius=4.0,
        current_step=5,
        max_episode_steps=100,
    )
    assert observation["node_features"].shape == (4, 10)
    assert observation["adjacency"][0, 1] == 1.0
    assert observation["global_features"].shape == (4,)
    assert np.isfinite(observation["edge_features"][0, 2, 3])

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

    backtrack_reward, backtrack_components = compute_reward(
        reward_weights=RewardWeights(progress=8.0, progress_negative_scale=2.0),
        goal_now=False,
        collision_now=False,
        progress_delta=-0.25,
        clearance_margin=0.0,
        action_array=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert np.isclose(backtrack_components["progress"], -4.0)
    assert np.isfinite(backtrack_reward)

    penalty_only_reward, penalty_only_components = compute_reward(
        reward_weights=RewardWeights(clearance=0.5, clearance_positive_scale=0.0),
        goal_now=False,
        collision_now=False,
        progress_delta=0.0,
        clearance_margin=3.0,
        action_array=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert penalty_only_components["clearance"] == 0.0
    assert np.isfinite(penalty_only_reward)

    distance_reward, distance_components = compute_reward(
        reward_weights=RewardWeights(remaining_distance=-0.2),
        goal_now=False,
        collision_now=False,
        progress_delta=0.0,
        clearance_margin=0.0,
        action_array=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        remaining_distance_ratio=0.5,
    )
    assert np.isclose(distance_components["remaining_distance"], -0.1)
    assert np.isfinite(distance_reward)

    danger_reward, danger_components = compute_reward(
        reward_weights=RewardWeights(danger_clearance_threshold=0.3, danger_clearance_penalty=2.0, danger_clearance_power=1.0),
        goal_now=False,
        collision_now=False,
        progress_delta=0.0,
        clearance_margin=0.1,
        action_array=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert np.isclose(danger_components["danger_clearance"], -1.3333333333333333)
    assert np.isfinite(danger_reward)


def test_episode_budget_cap_and_stall_penalty_are_configurable() -> None:
    """Late-stage anti-stall controls should be easy to reason about in isolation."""
    env = DynamicAirspaceEnv.__new__(DynamicAirspaceEnv)
    env.configured_max_episode_steps = 180
    env.auto_time_budget_steps_per_meter = 120.0
    env.auto_time_budget_padding = 30
    env.auto_time_budget_max_steps = 240
    assert env._resolve_episode_step_budget(3.0) == 240

    env.reward_weights = RewardWeights(
        stall=-0.6,
        stall_window_steps=4,
        stall_progress_threshold=0.2,
        stall_grace_steps=2,
        stall_remaining_ratio_threshold=0.15,
    )
    env.current_step = 10
    env.start_to_goal_distance = 4.0
    env.distance_to_goal_history = [4.0, 3.95, 3.9, 3.82, 3.75, 3.7, 3.65, 3.62, 3.6]

    stall_term = env._stall_penalty(distance_to_goal=3.6, goal_now=False)
    assert stall_term < 0.0

    near_goal_term = env._stall_penalty(distance_to_goal=0.4, goal_now=False)
    assert near_goal_term == 0.0


def test_progress_milestone_bonus_is_awarded_once() -> None:
    """Progress milestones should create one-time bonuses as the agent commits further."""
    env = DynamicAirspaceEnv.__new__(DynamicAirspaceEnv)
    env.reward_weights = RewardWeights(
        progress_milestone_bonus=1.5,
        progress_milestone_thresholds=(0.25, 0.5, 0.75),
        progress_milestone_bonus_weights=(1.0, 2.0, 4.0),
    )
    env.start_to_goal_distance = 4.0
    env.progress_milestones_hit = set()

    first_bonus = env._progress_milestone_bonus(distance_to_goal=2.7, clearance_margin=0.3)
    assert np.isclose(first_bonus, 1.0)

    repeated_bonus = env._progress_milestone_bonus(distance_to_goal=2.6, clearance_margin=0.3)
    assert repeated_bonus == 0.0

    later_bonus = env._progress_milestone_bonus(distance_to_goal=0.8, clearance_margin=0.3)
    assert np.isclose(later_bonus, 6.0)


def test_frontier_progress_bonus_only_rewards_new_best_progress() -> None:
    """Frontier bonus should only pay when the episode reaches a new best progress ratio."""
    env = DynamicAirspaceEnv.__new__(DynamicAirspaceEnv)
    env.reward_weights = RewardWeights(frontier_progress=20.0)
    env.start_to_goal_distance = 4.0
    env.best_progress_ratio = 0.0

    first_bonus = env._frontier_progress_bonus(distance_to_goal=3.0, clearance_margin=0.3)
    assert np.isclose(first_bonus, 5.0)

    repeated_bonus = env._frontier_progress_bonus(distance_to_goal=3.1, clearance_margin=0.3)
    assert repeated_bonus == 0.0

    later_bonus = env._frontier_progress_bonus(distance_to_goal=2.0, clearance_margin=0.3)
    assert np.isclose(later_bonus, 5.0)


def test_commit_bonus_gating_requires_safe_clearance() -> None:
    """Unsafe near-obstacle progress should not consume or award forward bonuses."""
    env = DynamicAirspaceEnv.__new__(DynamicAirspaceEnv)
    env.reward_weights = RewardWeights(
        frontier_progress=20.0,
        commit_bonus_min_clearance=0.2,
        progress_milestone_thresholds=(0.25,),
        progress_milestone_bonus_weights=(2.0,),
    )
    env.start_to_goal_distance = 4.0
    env.best_progress_ratio = 0.0
    env.progress_milestones_hit = set()

    unsafe_frontier = env._frontier_progress_bonus(distance_to_goal=3.0, clearance_margin=0.05)
    unsafe_milestone = env._progress_milestone_bonus(distance_to_goal=3.0, clearance_margin=0.05)
    assert unsafe_frontier == 0.0
    assert unsafe_milestone == 0.0
    assert env.best_progress_ratio == 0.0
    assert env.progress_milestones_hit == set()

    safe_frontier = env._frontier_progress_bonus(distance_to_goal=3.0, clearance_margin=0.3)
    safe_milestone = env._progress_milestone_bonus(distance_to_goal=3.0, clearance_margin=0.3)
    assert np.isclose(safe_frontier, 5.0)
    assert np.isclose(safe_milestone, 2.0)


def test_goal_proximity_bonus_only_rewards_new_near_goal_closeness() -> None:
    """Near-goal shaping should reward improved closeness, not stationary hovering."""
    env = DynamicAirspaceEnv.__new__(DynamicAirspaceEnv)
    env.reward_weights = RewardWeights(
        goal_proximity_bonus=10.0,
        goal_proximity_radius=1.0,
        goal_proximity_power=2.0,
        commit_bonus_min_clearance=0.2,
    )
    env.best_goal_proximity_score = 0.0

    outside_bonus = env._goal_proximity_bonus(distance_to_goal=1.2, clearance_margin=0.3)
    assert outside_bonus == 0.0

    unsafe_bonus = env._goal_proximity_bonus(distance_to_goal=0.8, clearance_margin=0.05)
    assert unsafe_bonus == 0.0

    first_bonus = env._goal_proximity_bonus(distance_to_goal=0.8, clearance_margin=0.3)
    assert np.isclose(first_bonus, 0.4)

    repeated_bonus = env._goal_proximity_bonus(distance_to_goal=0.82, clearance_margin=0.3)
    assert repeated_bonus == 0.0

    later_bonus = env._goal_proximity_bonus(distance_to_goal=0.5, clearance_margin=0.3)
    assert np.isclose(later_bonus, 2.1)


def test_teacher_helpers_produce_finite_guidance() -> None:
    """The heuristic teacher should emit bounded actions and finite bonuses."""
    action_low = np.asarray([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    action_high = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    teacher_action = heuristic_teacher_action(
        drone_position=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        goal_position=np.asarray([1.0, 0.0, 1.0], dtype=np.float32),
        obstacle_positions=np.asarray([[0.4, 0.2, 1.0]], dtype=np.float32),
        action_low=action_low,
        action_high=action_high,
        teacher_config={"enabled": True, "reward_scale": 0.1},
    )
    assert teacher_action.shape == (4,)
    assert np.all(teacher_action[:-1] <= action_high[:-1])
    assert np.all(teacher_action[:-1] >= action_low[:-1])
    assert 0.0 <= teacher_action[-1] <= action_high[-1]

    bonus = teacher_alignment_bonus(
        action_array=np.asarray([1.0, 0.0, 0.0, 0.9], dtype=np.float32),
        teacher_action=teacher_action,
        action_low=action_low,
        action_high=action_high,
        teacher_config={"enabled": True, "reward_scale": 0.1},
    )
    assert np.isfinite(bonus)
    assert bonus > 0.0

    negative_bonus = teacher_alignment_bonus(
        action_array=np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        teacher_action=teacher_action,
        action_low=action_low,
        action_high=action_high,
        teacher_config={"enabled": True, "reward_scale": 0.1},
    )
    assert negative_bonus < 0.0

    guided_action = teacher_guided_action(
        policy_action=np.asarray([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        teacher_action=np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        action_low=action_low,
        action_high=action_high,
        teacher_config={"enabled": True, "action_mix": 0.5},
    )
    assert np.allclose(guided_action, np.asarray([0.0, 0.0, 0.0, 0.5], dtype=np.float32))

    unchanged_action = teacher_guided_action(
        policy_action=np.asarray([-0.3, 0.0, 0.0, 0.2], dtype=np.float32),
        teacher_action=np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        action_low=action_low,
        action_high=action_high,
        teacher_config={"enabled": False, "action_mix": 0.8},
    )
    assert np.allclose(unchanged_action, np.asarray([-0.3, 0.0, 0.0, 0.2], dtype=np.float32))


def test_teacher_can_generate_lateral_bypass_for_obstacles_ahead() -> None:
    """Teacher guidance should include a clear lateral component when the goal ray is blocked."""
    action_low = np.asarray([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    action_high = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    teacher_action = heuristic_teacher_action(
        drone_position=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        goal_position=np.asarray([2.0, 0.0, 1.0], dtype=np.float32),
        obstacle_positions=np.asarray([[0.8, 0.0, 1.0]], dtype=np.float32),
        action_low=action_low,
        action_high=action_high,
        teacher_config={
            "enabled": True,
            "repulsion_radius": 1.0,
            "repulsion_gain": 0.35,
            "lateral_avoidance_gain": 1.1,
            "lateral_avoidance_radius": 0.75,
            "forward_lookahead": 2.2,
        },
    )

    assert abs(float(teacher_action[1])) > 0.4
    assert float(teacher_action[0]) > 0.0
    assert 0.0 <= float(teacher_action[-1]) <= 1.0
