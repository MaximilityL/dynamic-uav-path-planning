"""Scenario sampling and obstacle dynamics helpers."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

DEFAULT_SCENARIO_CONFIG: Dict[str, object] = {
    "start_x_progress_range": (0.0833333333, 0.2),
    "goal_x_progress_range": (0.8, 0.9166666667),
    "lateral_margin": 0.7,
    "vertical_margin": 0.2,
    "goal_relative_y_range": None,
    "goal_relative_z_range": None,
    "obstacle_xy_margin": 0.8,
    "obstacle_z_margin": 0.3,
    "min_start_goal_distance": 2.5,
    "obstacle_pair_clearance_margin": 0.15,
    "obstacle_start_goal_clearance_margin": 0.2,
    "obstacle_start_goal_exclusion_radius": None,
    "route_obstacle_count": 0,
    "route_obstacle_progress_range": (0.35, 0.75),
    "route_obstacle_lateral_offset_range": (-0.35, 0.35),
    "route_obstacle_vertical_offset_range": (-0.18, 0.18),
    "route_obstacle_longitudinal_jitter": 0.1,
}


def _scenario_float(config: Dict[str, object], key: str) -> float:
    """Read a float-valued scenario option with a default."""
    return float(config.get(key, DEFAULT_SCENARIO_CONFIG[key]))


def _scenario_int(config: Dict[str, object], key: str) -> int:
    """Read an int-valued scenario option with a default."""
    return int(config.get(key, DEFAULT_SCENARIO_CONFIG[key]))


def _scenario_optional_float(config: Dict[str, object], key: str) -> float | None:
    """Read an optional float-valued scenario option with a default."""
    raw_value = config.get(key, DEFAULT_SCENARIO_CONFIG[key])
    if raw_value is None:
        return None
    return float(raw_value)


def _scenario_range(config: Dict[str, object], key: str) -> tuple[float, float]:
    """Read a 2-tuple scenario range with a default."""
    raw_value = config.get(key, DEFAULT_SCENARIO_CONFIG[key])
    values = tuple(float(item) for item in raw_value)
    if len(values) != 2:
        raise ValueError(f"Scenario range '{key}' must contain exactly two values.")
    return values


def _optional_relative_axis_bounds(
    *,
    config: Dict[str, object],
    key: str,
    start_value: float,
    axis_bounds: np.ndarray,
    margin: float,
) -> tuple[float, float] | None:
    """Resolve an optional relative sampling window around the start value."""
    raw_value = config.get(key)
    if raw_value is None:
        return None
    relative_range = tuple(float(item) for item in raw_value)
    if len(relative_range) != 2:
        raise ValueError(f"Scenario range '{key}' must contain exactly two values.")
    lower = max(float(axis_bounds[0] + margin), float(start_value + relative_range[0]))
    upper = min(float(axis_bounds[1] - margin), float(start_value + relative_range[1]))
    if lower >= upper:
        return None
    return lower, upper


def _progress_to_axis(bounds: np.ndarray, progress_range: tuple[float, float]) -> tuple[float, float]:
    """Convert a normalized axis progress range into world coordinates."""
    span = float(bounds[1] - bounds[0])
    return (
        float(bounds[0] + span * progress_range[0]),
        float(bounds[0] + span * progress_range[1]),
    )


def _sample_uniform_obstacle_candidate(
    *,
    rng: np.random.Generator,
    workspace_bounds: np.ndarray,
    obstacle_xy_margin: float,
    obstacle_z_margin: float,
) -> np.ndarray:
    """Sample one obstacle uniformly from the free workspace volume."""
    x_bounds, y_bounds, z_bounds = workspace_bounds
    return np.asarray(
        [
            rng.uniform(x_bounds[0] + obstacle_xy_margin, x_bounds[1] - obstacle_xy_margin),
            rng.uniform(y_bounds[0] + obstacle_xy_margin, y_bounds[1] - obstacle_xy_margin),
            rng.uniform(z_bounds[0] + obstacle_z_margin, z_bounds[1] - obstacle_z_margin),
        ],
        dtype=np.float32,
    )


def _sample_route_biased_obstacle_candidate(
    *,
    rng: np.random.Generator,
    workspace_bounds: np.ndarray,
    start_position: np.ndarray,
    goal_position: np.ndarray,
    obstacle_xy_margin: float,
    obstacle_z_margin: float,
    scenario_config: Dict[str, object],
) -> np.ndarray:
    """Sample one obstacle near the straight-line route from start to goal."""
    x_bounds, y_bounds, z_bounds = workspace_bounds
    progress = float(np.clip(rng.uniform(*_scenario_range(scenario_config, "route_obstacle_progress_range")), 0.0, 1.0))
    anchor = np.asarray(start_position + progress * (goal_position - start_position), dtype=np.float32)
    longitudinal_jitter = _scenario_float(scenario_config, "route_obstacle_longitudinal_jitter")
    lateral_offset_range = _scenario_range(scenario_config, "route_obstacle_lateral_offset_range")
    vertical_offset_range = _scenario_range(scenario_config, "route_obstacle_vertical_offset_range")

    candidate = anchor.copy()
    candidate[0] += rng.uniform(-longitudinal_jitter, longitudinal_jitter)
    candidate[1] += rng.uniform(*lateral_offset_range)
    candidate[2] += rng.uniform(*vertical_offset_range)
    candidate[0] = np.clip(candidate[0], x_bounds[0] + obstacle_xy_margin, x_bounds[1] - obstacle_xy_margin)
    candidate[1] = np.clip(candidate[1], y_bounds[0] + obstacle_xy_margin, y_bounds[1] - obstacle_xy_margin)
    candidate[2] = np.clip(candidate[2], z_bounds[0] + obstacle_z_margin, z_bounds[1] - obstacle_z_margin)
    return candidate.astype(np.float32)


def _is_valid_obstacle_candidate(
    *,
    candidate: np.ndarray,
    positions: List[np.ndarray],
    start_position: np.ndarray,
    goal_position: np.ndarray,
    obstacle_radius: float,
    pair_clearance_margin: float,
    minimum_clearance: float,
) -> bool:
    """Check whether one sampled obstacle candidate is usable."""
    if np.linalg.norm(candidate - start_position) < minimum_clearance:
        return False
    if np.linalg.norm(candidate - goal_position) < minimum_clearance:
        return False
    if any(np.linalg.norm(candidate - existing) < (2.0 * obstacle_radius + pair_clearance_margin) for existing in positions):
        return False
    return True


def sample_start_position(
    rng: np.random.Generator,
    workspace_bounds: np.ndarray,
    scenario_config: Dict[str, object] | None = None,
) -> np.ndarray:
    """Sample a start position near one side of the airspace."""
    config = dict(DEFAULT_SCENARIO_CONFIG)
    if scenario_config:
        config.update(scenario_config)

    x_bounds, y_bounds, z_bounds = workspace_bounds
    start_x_bounds = _progress_to_axis(x_bounds, _scenario_range(config, "start_x_progress_range"))
    lateral_margin = _scenario_float(config, "lateral_margin")
    vertical_margin = _scenario_float(config, "vertical_margin")
    return np.asarray(
        [
            rng.uniform(start_x_bounds[0], start_x_bounds[1]),
            rng.uniform(y_bounds[0] + lateral_margin, y_bounds[1] - lateral_margin),
            rng.uniform(z_bounds[0] + vertical_margin, z_bounds[1] - vertical_margin),
        ],
        dtype=np.float32,
    )


def sample_goal_position(
    rng: np.random.Generator,
    workspace_bounds: np.ndarray,
    start_position: np.ndarray,
    scenario_config: Dict[str, object] | None = None,
) -> np.ndarray:
    """Sample a goal position separated from the start."""
    config = dict(DEFAULT_SCENARIO_CONFIG)
    if scenario_config:
        config.update(scenario_config)

    x_bounds, y_bounds, z_bounds = workspace_bounds
    goal_x_bounds = _progress_to_axis(x_bounds, _scenario_range(config, "goal_x_progress_range"))
    lateral_margin = _scenario_float(config, "lateral_margin")
    vertical_margin = _scenario_float(config, "vertical_margin")
    min_start_goal_distance = _scenario_float(config, "min_start_goal_distance")
    goal_y_bounds = _optional_relative_axis_bounds(
        config=config,
        key="goal_relative_y_range",
        start_value=float(start_position[1]),
        axis_bounds=y_bounds,
        margin=lateral_margin,
    )
    goal_z_bounds = _optional_relative_axis_bounds(
        config=config,
        key="goal_relative_z_range",
        start_value=float(start_position[2]),
        axis_bounds=z_bounds,
        margin=vertical_margin,
    )
    goal = np.asarray(start_position, dtype=np.float32)
    for _ in range(200):
        goal = np.asarray(
            [
                rng.uniform(goal_x_bounds[0], goal_x_bounds[1]),
                rng.uniform(*(goal_y_bounds or (y_bounds[0] + lateral_margin, y_bounds[1] - lateral_margin))),
                rng.uniform(*(goal_z_bounds or (z_bounds[0] + vertical_margin, z_bounds[1] - vertical_margin))),
            ],
            dtype=np.float32,
        )
        if np.linalg.norm(goal - start_position) > min_start_goal_distance:
            return goal
    return goal


def sample_dynamic_obstacles(
    rng: np.random.Generator,
    workspace_bounds: np.ndarray,
    *,
    num_dynamic_obstacles: int,
    obstacle_radius: float,
    goal_tolerance: float,
    collision_distance: float,
    obstacle_speed_range: tuple[float, float],
    start_position: np.ndarray,
    goal_position: np.ndarray,
    scenario_config: Dict[str, object] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample obstacle positions and velocities away from the start and goal."""
    config = dict(DEFAULT_SCENARIO_CONFIG)
    if scenario_config:
        config.update(scenario_config)

    x_bounds, y_bounds, z_bounds = workspace_bounds
    positions: List[np.ndarray] = []
    velocities: List[np.ndarray] = []
    obstacle_xy_margin = _scenario_float(config, "obstacle_xy_margin")
    obstacle_z_margin = _scenario_float(config, "obstacle_z_margin")
    pair_clearance_margin = _scenario_float(config, "obstacle_pair_clearance_margin")
    obstacle_start_goal_clearance_margin = _scenario_float(config, "obstacle_start_goal_clearance_margin")
    start_goal_exclusion_radius = _scenario_optional_float(config, "obstacle_start_goal_exclusion_radius")
    minimum_clearance = (
        start_goal_exclusion_radius
        if start_goal_exclusion_radius is not None
        else obstacle_radius + goal_tolerance + collision_distance + obstacle_start_goal_clearance_margin
    )

    route_obstacle_target = min(max(_scenario_int(config, "route_obstacle_count"), 0), num_dynamic_obstacles)

    while len(positions) < num_dynamic_obstacles:
        obstacle_index = len(positions)
        prefer_route_bias = obstacle_index < route_obstacle_target
        route_attempts = 0
        while True:
            if prefer_route_bias and route_attempts < 300:
                candidate = _sample_route_biased_obstacle_candidate(
                    rng=rng,
                    workspace_bounds=workspace_bounds,
                    start_position=start_position,
                    goal_position=goal_position,
                    obstacle_xy_margin=obstacle_xy_margin,
                    obstacle_z_margin=obstacle_z_margin,
                    scenario_config=config,
                )
                route_attempts += 1
            else:
                candidate = _sample_uniform_obstacle_candidate(
                    rng=rng,
                    workspace_bounds=workspace_bounds,
                    obstacle_xy_margin=obstacle_xy_margin,
                    obstacle_z_margin=obstacle_z_margin,
                )

            if _is_valid_obstacle_candidate(
                candidate=candidate,
                positions=positions,
                start_position=start_position,
                goal_position=goal_position,
                obstacle_radius=obstacle_radius,
                pair_clearance_margin=pair_clearance_margin,
                minimum_clearance=minimum_clearance,
            ):
                break

        theta = rng.uniform(0.0, 2.0 * np.pi)
        phi = rng.uniform(-0.25, 0.25)
        speed = rng.uniform(obstacle_speed_range[0], obstacle_speed_range[1])
        velocity = np.asarray(
            [
                speed * np.cos(theta),
                speed * np.sin(theta),
                speed * np.sin(phi),
            ],
            dtype=np.float32,
        )
        positions.append(candidate)
        velocities.append(velocity)

    return np.asarray(positions, dtype=np.float32), np.asarray(velocities, dtype=np.float32)


def advance_obstacles(
    obstacle_positions: np.ndarray,
    obstacle_velocities: np.ndarray,
    *,
    workspace_bounds: np.ndarray,
    obstacle_radius: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Move obstacles with simple bounce dynamics inside the workspace."""
    if obstacle_positions.size == 0:
        return obstacle_positions, obstacle_velocities

    next_positions = obstacle_positions + obstacle_velocities * dt
    next_velocities = obstacle_velocities.copy()
    for obstacle_idx in range(next_positions.shape[0]):
        for axis in range(3):
            lower = workspace_bounds[axis, 0] + obstacle_radius
            upper = workspace_bounds[axis, 1] - obstacle_radius
            if next_positions[obstacle_idx, axis] <= lower:
                next_positions[obstacle_idx, axis] = lower
                next_velocities[obstacle_idx, axis] *= -1.0
            elif next_positions[obstacle_idx, axis] >= upper:
                next_positions[obstacle_idx, axis] = upper
                next_velocities[obstacle_idx, axis] *= -1.0

    return next_positions.astype(np.float32), next_velocities.astype(np.float32)
