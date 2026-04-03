"""Evaluation metrics for dynamic UAV path planning."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np


def _path_efficiency(metric: Dict[str, object]) -> float:
    """Return a simple success-aware path-efficiency score."""
    if float(metric.get("success", 0.0)) <= 0.0:
        return 0.0
    start_to_goal = max(float(metric.get("start_to_goal_distance", 0.0)), 1e-6)
    path_length = max(float(metric.get("path_length", start_to_goal)), start_to_goal)
    return float(start_to_goal / path_length)


def _mean_optional(values: Iterable[Optional[float]]) -> float:
    """Average optional scalar values while ignoring missing entries."""
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return 0.0
    return float(np.mean(np.asarray(filtered, dtype=np.float32)))


def summarize_episodes(episode_metrics: List[Dict[str, object]]) -> Dict[str, float]:
    """Aggregate episode metrics into a compact summary."""
    if not episode_metrics:
        return {
            "num_episodes": 0,
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "avg_episode_return": 0.0,
            "avg_path_length": 0.0,
            "avg_episode_duration": 0.0,
            "avg_time_to_goal": 0.0,
            "avg_min_obstacle_distance": 0.0,
            "avg_min_clearance": 0.0,
            "avg_control_effort": 0.0,
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
    min_clearances = np.asarray([float(item.get("min_clearance", 0.0)) for item in episode_metrics], dtype=np.float32)
    path_lengths = np.asarray([float(item.get("path_length", 0.0)) for item in episode_metrics], dtype=np.float32)
    steps = np.asarray([float(item.get("steps", 0.0)) for item in episode_metrics], dtype=np.float32)
    durations = np.asarray([float(item.get("episode_duration", 0.0)) for item in episode_metrics], dtype=np.float32)
    control_efforts = np.asarray([float(item.get("control_effort", 0.0)) for item in episode_metrics], dtype=np.float32)
    efficiencies = np.asarray([_path_efficiency(item) for item in episode_metrics], dtype=np.float32)
    time_to_goal = _mean_optional(
        item.get("time_to_goal") if item.get("time_to_goal") is not None else None for item in episode_metrics
    )

    return {
        "num_episodes": int(len(episode_metrics)),
        "success_rate": float(successes.mean()),
        "collision_rate": float(collisions.mean()),
        "avg_episode_return": float(returns.mean()),
        "avg_path_length": float(path_lengths.mean()),
        "avg_episode_duration": float(durations.mean()),
        "avg_time_to_goal": time_to_goal,
        "avg_min_obstacle_distance": float(min_distances.mean()),
        "avg_min_clearance": float(min_clearances.mean()),
        "avg_control_effort": float(control_efforts.mean()),
        "avg_path_efficiency": float(efficiencies.mean()),
        "avg_steps": float(steps.mean()),
        "best_episode_return": float(returns.max()),
    }
