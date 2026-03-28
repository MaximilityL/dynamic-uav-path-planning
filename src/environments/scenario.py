"""Scenario sampling and obstacle dynamics helpers."""

from __future__ import annotations

from typing import List

import numpy as np


def sample_start_position(rng: np.random.Generator, workspace_bounds: np.ndarray) -> np.ndarray:
    """Sample a start position near one side of the airspace."""
    x_bounds, y_bounds, z_bounds = workspace_bounds
    return np.asarray(
        [
            rng.uniform(x_bounds[0] + 0.5, x_bounds[0] + 1.2),
            rng.uniform(y_bounds[0] + 0.7, y_bounds[1] - 0.7),
            rng.uniform(z_bounds[0] + 0.2, z_bounds[1] - 0.2),
        ],
        dtype=np.float32,
    )


def sample_goal_position(
    rng: np.random.Generator,
    workspace_bounds: np.ndarray,
    start_position: np.ndarray,
) -> np.ndarray:
    """Sample a goal position separated from the start."""
    x_bounds, y_bounds, z_bounds = workspace_bounds
    goal = np.asarray(start_position, dtype=np.float32)
    for _ in range(200):
        goal = np.asarray(
            [
                rng.uniform(x_bounds[1] - 1.2, x_bounds[1] - 0.5),
                rng.uniform(y_bounds[0] + 0.7, y_bounds[1] - 0.7),
                rng.uniform(z_bounds[0] + 0.2, z_bounds[1] - 0.2),
            ],
            dtype=np.float32,
        )
        if np.linalg.norm(goal - start_position) > 2.5:
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
) -> tuple[np.ndarray, np.ndarray]:
    """Sample obstacle positions and velocities away from the start and goal."""
    x_bounds, y_bounds, z_bounds = workspace_bounds
    positions: List[np.ndarray] = []
    velocities: List[np.ndarray] = []
    minimum_clearance = obstacle_radius + goal_tolerance + collision_distance + 0.2

    while len(positions) < num_dynamic_obstacles:
        candidate = np.asarray(
            [
                rng.uniform(x_bounds[0] + 0.8, x_bounds[1] - 0.8),
                rng.uniform(y_bounds[0] + 0.8, y_bounds[1] - 0.8),
                rng.uniform(z_bounds[0] + 0.3, z_bounds[1] - 0.3),
            ],
            dtype=np.float32,
        )
        if np.linalg.norm(candidate - start_position) < minimum_clearance:
            continue
        if np.linalg.norm(candidate - goal_position) < minimum_clearance:
            continue
        if any(np.linalg.norm(candidate - existing) < (2.0 * obstacle_radius + 0.15) for existing in positions):
            continue

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
