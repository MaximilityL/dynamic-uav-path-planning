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
    "lateral_avoidance_gain": 0.0,
    "lateral_avoidance_radius": 0.6,
    "forward_lookahead": 1.8,
    "far_speed": 0.9,
    "near_speed": 0.5,
    "near_goal_distance": 1.2,
    "rejoin_gain": 0.0,
    "rejoin_clearance_threshold": 1.0,
    "rejoin_progress_ratio_threshold": 0.0,
    "rejoin_min_lateral_error": 0.0,
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
    route_start_position: np.ndarray | None = None,
    teacher_config: Dict[str, object] | None,
) -> np.ndarray:
    """Build a simple goal-seeking action with local obstacle repulsion."""
    return np.asarray(
        heuristic_teacher_guidance(
            drone_position=drone_position,
            goal_position=goal_position,
            obstacle_positions=obstacle_positions,
            action_low=action_low,
            action_high=action_high,
            route_start_position=route_start_position,
            teacher_config=teacher_config,
        )["action"],
        dtype=np.float32,
    )


def heuristic_teacher_guidance(
    *,
    drone_position: np.ndarray,
    goal_position: np.ndarray,
    obstacle_positions: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    route_start_position: np.ndarray | None = None,
    teacher_config: Dict[str, object] | None,
) -> Dict[str, object]:
    """Build the heuristic teacher action together with gating-friendly metadata."""
    config = dict(DEFAULT_TEACHER_CONFIG)
    if teacher_config:
        config.update(teacher_config)

    to_goal = np.asarray(goal_position - drone_position, dtype=np.float32)
    goal_distance = float(np.linalg.norm(to_goal))
    goal_direction = to_goal / max(goal_distance, 1e-6)
    route_progress_ratio = 0.0
    route_line_lateral_error = 0.0
    route_rejoin_vector = np.zeros(3, dtype=np.float32)
    if route_start_position is not None:
        route_start = np.asarray(route_start_position, dtype=np.float32)
        route_vector = np.asarray(goal_position - route_start, dtype=np.float32)
        route_length = float(np.linalg.norm(route_vector))
        if route_length > 1e-6:
            route_direction = route_vector / route_length
            from_start = np.asarray(drone_position - route_start, dtype=np.float32)
            along_track = float(np.dot(from_start, route_direction))
            along_track = float(np.clip(along_track, 0.0, route_length))
            closest_point = route_start + along_track * route_direction
            route_rejoin_vector = np.asarray(closest_point - drone_position, dtype=np.float32)
            route_line_lateral_error = float(np.linalg.norm(route_rejoin_vector))
            route_progress_ratio = float(np.clip(along_track / max(route_length, 1e-6), 0.0, 1.5))

    repulsion = np.zeros(3, dtype=np.float32)
    repulsion_radius = _teacher_float(config, "repulsion_radius")
    repulsion_gain = _teacher_float(config, "repulsion_gain")
    repulsion_obstacle_count = 0
    lateral_avoidance = np.zeros(3, dtype=np.float32)
    lateral_avoidance_gain = _teacher_float(config, "lateral_avoidance_gain")
    lateral_avoidance_radius = _teacher_float(config, "lateral_avoidance_radius")
    forward_lookahead = _teacher_float(config, "forward_lookahead")
    blocking_obstacle_count = 0
    blocking_min_lateral_distance = float(max(lateral_avoidance_radius, 0.0))
    for obstacle_position in np.asarray(obstacle_positions, dtype=np.float32):
        delta = np.asarray(drone_position - obstacle_position, dtype=np.float32)
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-6 or distance >= repulsion_radius:
            within_repulsion_radius = False
        else:
            within_repulsion_radius = True
            repulsion += delta / max(distance * distance, 1e-3)
            repulsion_obstacle_count += 1

        if lateral_avoidance_gain <= 0.0 or lateral_avoidance_radius <= 0.0 or forward_lookahead <= 0.0:
            continue

        obstacle_vector = np.asarray(obstacle_position - drone_position, dtype=np.float32)
        forward_distance = float(np.dot(obstacle_vector, goal_direction))
        if forward_distance <= 0.0 or forward_distance >= forward_lookahead:
            continue

        lateral_vector = obstacle_vector - forward_distance * goal_direction
        lateral_distance = float(np.linalg.norm(lateral_vector))
        if lateral_distance >= lateral_avoidance_radius:
            continue
        if within_repulsion_radius or forward_distance <= forward_lookahead:
            blocking_obstacle_count += 1
            blocking_min_lateral_distance = min(blocking_min_lateral_distance, lateral_distance)

        if lateral_distance > 1e-6:
            bypass_direction = -lateral_vector / lateral_distance
        else:
            reference_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(float(np.dot(goal_direction, reference_axis))) > 0.9:
                reference_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            bypass_direction = np.cross(goal_direction, reference_axis)
            bypass_norm = float(np.linalg.norm(bypass_direction))
            if bypass_norm <= 1e-6:
                reference_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
                bypass_direction = np.cross(goal_direction, reference_axis)
                bypass_norm = float(np.linalg.norm(bypass_direction))
            bypass_direction = bypass_direction / max(bypass_norm, 1e-6)

        forward_factor = 1.0 - forward_distance / max(forward_lookahead, 1e-6)
        lateral_factor = 1.0 - lateral_distance / max(lateral_avoidance_radius, 1e-6)
        lateral_avoidance += bypass_direction * max(forward_factor, 0.0) * max(lateral_factor, 0.0)

    far_speed = _teacher_float(config, "far_speed")
    near_speed = _teacher_float(config, "near_speed")
    near_goal_distance = _teacher_float(config, "near_goal_distance")
    speed_cap = float(max(abs(action_high[-1]), abs(action_low[-1]), 1e-6))
    speed = far_speed if goal_distance > near_goal_distance else near_speed
    speed = float(np.clip(speed, 0.0, speed_cap))
    repulsion_magnitude = float(np.linalg.norm(repulsion))
    bypass_magnitude = float(np.linalg.norm(lateral_avoidance))
    if lateral_avoidance_radius > 1e-6:
        blocking_clearance_score = float(
            np.clip(blocking_min_lateral_distance / max(lateral_avoidance_radius, 1e-6), 0.0, 1.0)
        )
    else:
        blocking_clearance_score = 1.0
    rejoin_gain = _teacher_float(config, "rejoin_gain")
    rejoin_clearance_threshold = _teacher_float(config, "rejoin_clearance_threshold")
    rejoin_progress_ratio_threshold = _teacher_float(config, "rejoin_progress_ratio_threshold")
    rejoin_min_lateral_error = _teacher_float(config, "rejoin_min_lateral_error")
    rejoin_active = False
    rejoin_direction = np.zeros(3, dtype=np.float32)
    if (
        rejoin_gain > 0.0
        and route_line_lateral_error >= rejoin_min_lateral_error
        and route_progress_ratio >= rejoin_progress_ratio_threshold
        and blocking_clearance_score >= rejoin_clearance_threshold
    ):
        rejoin_active = True
        rejoin_direction = route_rejoin_vector / max(route_line_lateral_error, 1e-6)

    steering = goal_direction + repulsion_gain * repulsion + lateral_avoidance_gain * lateral_avoidance + rejoin_gain * rejoin_direction
    steering_norm = float(np.linalg.norm(steering))
    if steering_norm > 1e-6:
        steering = steering / steering_norm
    else:
        steering = goal_direction
    action = np.zeros_like(action_high, dtype=np.float32)
    action[:-1] = np.clip(steering, action_low[:-1], action_high[:-1])
    action[-1] = speed

    return {
        "action": action.astype(np.float32),
        "goal_distance": goal_distance,
        "repulsion_active": bool(repulsion_obstacle_count > 0),
        "bypass_active": bool(blocking_obstacle_count > 0 and lateral_avoidance_gain > 0.0),
        "teacher_active": bool(repulsion_obstacle_count > 0 or blocking_obstacle_count > 0),
        "rejoin_active": bool(rejoin_active),
        "repulsion_obstacle_count": int(repulsion_obstacle_count),
        "blocking_obstacle_count": int(blocking_obstacle_count),
        "blocking_min_lateral_distance": float(blocking_min_lateral_distance),
        "blocking_clearance_score": float(blocking_clearance_score),
        "route_progress_ratio": float(route_progress_ratio),
        "route_line_lateral_error": float(route_line_lateral_error),
        "repulsion_magnitude": repulsion_magnitude,
        "bypass_magnitude": bypass_magnitude,
        "rejoin_magnitude": float(np.linalg.norm(rejoin_direction)),
    }


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
