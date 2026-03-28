"""Reward helpers for dynamic UAV path planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class RewardWeights:
    """Reward coefficients for dynamic path planning."""

    goal: float = 25.0
    progress: float = 6.0
    collision: float = -20.0
    clearance: float = 0.2
    effort: float = -0.01
    time: float = -0.02


def compute_reward(
    *,
    reward_weights: RewardWeights,
    goal_now: bool,
    collision_now: bool,
    progress_delta: float,
    clearance_margin: float,
    action_array: np.ndarray,
) -> tuple[float, Dict[str, float]]:
    """Build reward components and their total reward."""
    reward_components = {
        "goal": reward_weights.goal if goal_now else 0.0,
        "progress": reward_weights.progress * progress_delta,
        "collision": reward_weights.collision if collision_now else 0.0,
        "clearance": reward_weights.clearance * float(np.clip(clearance_margin, -1.0, 1.0)),
        "effort": reward_weights.effort * float(np.linalg.norm(action_array) / max(len(action_array), 1)),
        "time": reward_weights.time,
    }
    return float(sum(reward_components.values())), reward_components
