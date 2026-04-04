"""Lightweight heuristic teacher helpers for early-stage training guidance."""

from __future__ import annotations

from typing import Dict

import numpy as np

DEFAULT_TEACHER_CONFIG: Dict[str, float | bool] = {
    "enabled": False,
    "reward_scale": 0.0,
    "episode_bonus_budget": 0.0,
    "action_mix": 0.0,
    "repulsion_radius": 1.0,
    "repulsion_gain": 0.5,
    "far_speed": 0.9,
    "near_speed": 0.5,
    "near_goal_distance": 1.2,
    "direction_weight": 0.75,
    "speed_weight": 0.25,
}


def _teacher_float(config: Dict[str, object], key: str) -> float:
    """Read a float-like teacher configuration value."""
    return float(config.get(key, DEFAULT_TEACHER_CONFIG[key]))


def _teacher_bool(config: Dict[str, object], key: str) -> bool:
    """Read a boolean-like teacher configuration value."""
    return bool(config.get(key, DEFAULT_TEACHER_CONFIG[key]))


def heuristic_teacher_action(
    *,
    drone_position: np.ndarray,
    goal_position: np.ndarray,
    obstacle_positions: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    teacher_config: Dict[str, object] | None,
) -> np.ndarray:
    """Build a simple goal-seeking action with local obstacle repulsion."""
    config = dict(DEFAULT_TEACHER_CONFIG)
    if teacher_config:
        config.update(teacher_config)

    to_goal = np.asarray(goal_position - drone_position, dtype=np.float32)
    goal_distance = float(np.linalg.norm(to_goal))
    goal_direction = to_goal / max(goal_distance, 1e-6)

    repulsion = np.zeros(3, dtype=np.float32)
    repulsion_radius = _teacher_float(config, "repulsion_radius")
    repulsion_gain = _teacher_float(config, "repulsion_gain")
    for obstacle_position in np.asarray(obstacle_positions, dtype=np.float32):
        delta = np.asarray(drone_position - obstacle_position, dtype=np.float32)
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6 or distance >= repulsion_radius:
            continue
        repulsion += delta / max(distance * distance, 1e-3)

    steering = goal_direction + repulsion_gain * repulsion
    steering_norm = float(np.linalg.norm(steering))
    if steering_norm > 1e-6:
        steering = steering / steering_norm
    else:
        steering = goal_direction

    far_speed = _teacher_float(config, "far_speed")
    near_speed = _teacher_float(config, "near_speed")
    near_goal_distance = _teacher_float(config, "near_goal_distance")
    speed_cap = float(max(abs(action_high[-1]), abs(action_low[-1]), 1e-6))
    speed = far_speed if goal_distance > near_goal_distance else near_speed
    speed = float(np.clip(speed, 0.0, speed_cap))

    action = np.zeros_like(action_high, dtype=np.float32)
    action[:-1] = np.clip(steering, action_low[:-1], action_high[:-1])
    action[-1] = speed
    return action.astype(np.float32)


def teacher_alignment_bonus(
    *,
    action_array: np.ndarray,
    teacher_action: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    teacher_config: Dict[str, object] | None,
) -> float:
    """Return a small dense bonus for matching the heuristic teacher."""
    config = dict(DEFAULT_TEACHER_CONFIG)
    if teacher_config:
        config.update(teacher_config)
    if not _teacher_bool(config, "enabled"):
        return 0.0

    reward_scale = _teacher_float(config, "reward_scale")
    if reward_scale <= 0.0:
        return 0.0

    direction_weight = _teacher_float(config, "direction_weight")
    speed_weight = _teacher_float(config, "speed_weight")

    action_direction = np.asarray(action_array[:-1], dtype=np.float32)
    teacher_direction = np.asarray(teacher_action[:-1], dtype=np.float32)
    action_norm = float(np.linalg.norm(action_direction))
    teacher_norm = float(np.linalg.norm(teacher_direction))
    if action_norm > 1e-6 and teacher_norm > 1e-6:
        direction_score = float(np.dot(action_direction, teacher_direction) / (action_norm * teacher_norm))
    else:
        direction_score = 0.0

    speed_cap = float(max(abs(action_high[-1]), abs(action_low[-1]), 1e-6))
    speed_score = 1.0 - abs(float(action_array[-1] - teacher_action[-1])) / speed_cap
    speed_score = float(np.clip(2.0 * speed_score - 1.0, -1.0, 1.0))

    return float(reward_scale * (direction_weight * direction_score + speed_weight * speed_score))


def teacher_guided_action(
    *,
    policy_action: np.ndarray,
    teacher_action: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    teacher_config: Dict[str, object] | None,
) -> np.ndarray:
    """Blend a policy action with the heuristic teacher when configured."""
    config = dict(DEFAULT_TEACHER_CONFIG)
    if teacher_config:
        config.update(teacher_config)
    if not _teacher_bool(config, "enabled"):
        return np.asarray(policy_action, dtype=np.float32)

    action_mix = float(np.clip(_teacher_float(config, "action_mix"), 0.0, 1.0))
    if action_mix <= 0.0:
        return np.asarray(policy_action, dtype=np.float32)

    policy_action_array = np.asarray(policy_action, dtype=np.float32)
    teacher_action_array = np.asarray(teacher_action, dtype=np.float32)
    mixed_action = (1.0 - action_mix) * policy_action_array + action_mix * teacher_action_array
    return np.clip(mixed_action, action_low, action_high).astype(np.float32)
