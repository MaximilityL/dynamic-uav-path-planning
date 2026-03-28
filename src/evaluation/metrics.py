"""Evaluation metrics for dynamic UAV path planning."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _path_efficiency(metric: Dict[str, float]) -> float:
    """Return a simple success-aware path-efficiency score."""
    if float(metric.get("success", 0.0)) <= 0.0:
        return 0.0
    start_to_goal = max(float(metric.get("start_to_goal_distance", 0.0)), 1e-6)
    path_length = max(float(metric.get("path_length", start_to_goal)), start_to_goal)
    return float(start_to_goal / path_length)


def summarize_episodes(episode_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate episode metrics into a compact summary."""
    if not episode_metrics:
        return {
            "num_episodes": 0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "avg_episode_return": 0.0,
            "avg_min_obstacle_distance": 0.0,
            "avg_path_efficiency": 0.0,
            "avg_steps": 0.0,
            "best_episode_return": 0.0,
        }

    returns = np.asarray([float(item.get("episode_return", 0.0)) for item in episode_metrics], dtype=np.float32)
    successes = np.asarray([float(item.get("success", 0.0)) for item in episode_metrics], dtype=np.float32)
    collisions = np.asarray([float(item.get("collision", 0.0)) for item in episode_metrics], dtype=np.float32)
    min_distances = np.asarray(
        [float(item.get("min_obstacle_distance", 0.0)) for item in episode_metrics],
        dtype=np.float32,
    )
    steps = np.asarray([float(item.get("steps", 0.0)) for item in episode_metrics], dtype=np.float32)
    efficiencies = np.asarray([_path_efficiency(item) for item in episode_metrics], dtype=np.float32)

    return {
        "num_episodes": int(len(episode_metrics)),
        "success_rate": float(successes.mean()),
        "collision_rate": float(collisions.mean()),
        "avg_episode_return": float(returns.mean()),
        "avg_min_obstacle_distance": float(min_distances.mean()),
        "avg_path_efficiency": float(efficiencies.mean()),
        "avg_steps": float(steps.mean()),
        "best_episode_return": float(returns.max()),
    }
