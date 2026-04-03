"""Tests for evaluation metric aggregation."""

from __future__ import annotations

import pytest

from src.evaluation.metrics import summarize_episodes


def test_summarize_episodes_aggregates_expected_fields() -> None:
    """Aggregate metrics should be computed from episode summaries."""
    summary = summarize_episodes(
        [
            {
                "episode_return": 10.0,
                "success": 1.0,
                "collision": 0.0,
                "min_obstacle_distance": 0.7,
                "min_clearance": 0.35,
                "path_length": 3.0,
                "start_to_goal_distance": 2.0,
                "steps": 20.0,
                "episode_duration": 2.0,
                "time_to_goal": 1.7,
                "control_effort": 0.3,
            },
            {
                "episode_return": -4.0,
                "success": 0.0,
                "collision": 1.0,
                "min_obstacle_distance": 0.2,
                "min_clearance": -0.15,
                "path_length": 5.0,
                "start_to_goal_distance": 2.5,
                "steps": 30.0,
                "episode_duration": 3.0,
                "time_to_goal": None,
                "control_effort": 0.6,
            },
        ]
    )

    assert summary["num_episodes"] == 2
    assert summary["success_rate"] == 0.5
    assert summary["collision_rate"] == 0.5
    assert summary["best_episode_return"] == 10.0
    assert summary["avg_path_length"] == 4.0
    assert summary["avg_time_to_goal"] == pytest.approx(1.7)
    assert summary["avg_min_clearance"] == pytest.approx(0.1)
    assert summary["avg_control_effort"] == pytest.approx(0.45)
