"""Graph observation helpers for the dynamic airspace environment."""

from __future__ import annotations

from typing import Dict

import numpy as np


def minimum_obstacle_distance(
    drone_position: np.ndarray,
    obstacle_positions: np.ndarray,
    *,
    default_distance: float,
) -> float:
    """Return the closest obstacle-center distance."""
    if obstacle_positions.size == 0:
        return float(default_distance)
    distances = np.linalg.norm(obstacle_positions - drone_position.reshape(1, 3), axis=1)
    return float(distances.min())


def build_dense_graph_observation(
    *,
    drone_position: np.ndarray,
    drone_velocity: np.ndarray,
    goal_position: np.ndarray,
    obstacle_positions: np.ndarray,
    obstacle_velocities: np.ndarray,
    obstacle_radius: float,
    goal_tolerance: float,
    workspace_bounds: np.ndarray,
    obstacle_speed_range: tuple[float, float],
    connect_radius: float,
    current_step: int,
    max_episode_steps: int,
) -> Dict[str, np.ndarray]:
    """Construct the dense graph observation expected by the policy."""
    max_nodes = 2 + int(obstacle_positions.shape[0])
    node_feature_dim = 10
    edge_feature_dim = 4

    node_features = np.zeros((max_nodes, node_feature_dim), dtype=np.float32)
    node_mask = np.ones((max_nodes,), dtype=np.float32)
    adjacency = np.zeros((max_nodes, max_nodes), dtype=np.float32)
    edge_features = np.zeros((max_nodes, max_nodes, edge_feature_dim), dtype=np.float32)

    workspace_span = np.maximum(workspace_bounds[:, 1] - workspace_bounds[:, 0], 1e-6)
    velocity_scale = max(1.0, float(obstacle_speed_range[1]))
    nodes = [drone_position, goal_position] + [position for position in obstacle_positions]
    velocities = [drone_velocity, np.zeros(3, dtype=np.float32)] + [velocity for velocity in obstacle_velocities]
    radii = [0.0, goal_tolerance] + [obstacle_radius] * int(obstacle_positions.shape[0])
    node_types = [
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    ] + [np.asarray([0.0, 0.0, 1.0], dtype=np.float32) for _ in range(int(obstacle_positions.shape[0]))]

    for index, (position, velocity, radius, node_type) in enumerate(zip(nodes, velocities, radii, node_types)):
        rel_pos = (position - drone_position) / workspace_span
        rel_vel = (velocity - drone_velocity) / velocity_scale
        node_features[index, 0:3] = rel_pos
        node_features[index, 3:6] = rel_vel
        node_features[index, 6] = float(radius / max(workspace_span.max(), 1.0))
        node_features[index, 7:10] = node_type

    for receiver in range(max_nodes):
        for sender in range(max_nodes):
            if receiver == sender:
                continue
            delta = nodes[sender] - nodes[receiver]
            distance = float(np.linalg.norm(delta))
            if receiver == 0 or sender == 0 or distance <= connect_radius:
                adjacency[receiver, sender] = 1.0
                edge_features[receiver, sender, 0:3] = delta / workspace_span
                edge_features[receiver, sender, 3] = distance / max(connect_radius, 1e-6)

    min_distance = minimum_obstacle_distance(
        drone_position,
        obstacle_positions,
        default_distance=connect_radius,
    )
    global_features = np.asarray(
        [
            np.linalg.norm(goal_position - drone_position) / max(float(np.linalg.norm(workspace_span)), 1e-6),
            min_distance / max(connect_radius, 1e-6),
            np.linalg.norm(drone_velocity),
            float(current_step / max(max_episode_steps, 1)),
        ],
        dtype=np.float32,
    )

    return {
        "node_features": node_features,
        "node_mask": node_mask,
        "adjacency": adjacency,
        "edge_features": edge_features,
        "global_features": global_features,
    }
