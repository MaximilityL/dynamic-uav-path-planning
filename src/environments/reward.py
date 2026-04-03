"""Reward helpers for dynamic UAV path planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class RewardWeights:
    """Reward coefficients for dynamic path planning."""

    goal: float = 40.0
    progress: float = 8.0
    collision: float = -40.0
    clearance: float = 0.5
    effort: float = -0.005
    time: float = -0.01


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
    progress_term = reward_weights.progress * float(np.clip(progress_delta, -1.0, 1.0))
    clearance_signal = float(np.tanh(clearance_margin))
    if clearance_signal > 0.0:
        clearance_signal *= 0.02
    clearance_term = reward_weights.clearance * clearance_signal
    effort_scale = float(np.linalg.norm(action_array) / np.sqrt(max(action_array.size, 1)))
    reward_components = {
        "goal": reward_weights.goal if goal_now else 0.0,
        "progress": progress_term,
        "collision": reward_weights.collision if collision_now else 0.0,
        "clearance": clearance_term,
        "effort": reward_weights.effort * effort_scale,
        "time": reward_weights.time,
    }
    return float(sum(reward_components.values())), reward_components
