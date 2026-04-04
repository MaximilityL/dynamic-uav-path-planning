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
    progress_negative_scale: float = 1.0
    frontier_progress: float = 0.0
    commit_bonus_min_clearance: float = 0.0
    goal_proximity_bonus: float = 0.0
    goal_proximity_radius: float = 0.0
    goal_proximity_power: float = 1.0
    progress_milestone_bonus: float = 0.0
    progress_milestone_thresholds: tuple[float, ...] = ()
    progress_milestone_bonus_weights: tuple[float, ...] = ()
    collision: float = -40.0
    clearance: float = 0.5
    clearance_positive_scale: float = 0.02
    danger_clearance_threshold: float = 0.0
    danger_clearance_penalty: float = 0.0
    danger_clearance_power: float = 1.0
    effort: float = -0.005
    time: float = -0.01
    timeout: float = -8.0
    timeout_distance: float = -20.0
    remaining_distance: float = 0.0
    stall: float = 0.0
    stall_window_steps: int = 0
    stall_progress_threshold: float = 0.0
    stall_grace_steps: int = 0
    stall_remaining_ratio_threshold: float = 0.15


def compute_reward(
    *,
    reward_weights: RewardWeights,
    goal_now: bool,
    collision_now: bool,
    progress_delta: float,
    clearance_margin: float,
    action_array: np.ndarray,
    remaining_distance_ratio: float = 0.0,
) -> tuple[float, Dict[str, float]]:
    """Build reward components and their total reward."""
    clipped_progress = float(np.clip(progress_delta, -1.0, 1.0))
    if clipped_progress >= 0.0:
        progress_term = reward_weights.progress * clipped_progress
    else:
        negative_scale = float(max(reward_weights.progress_negative_scale, 0.0))
        progress_term = reward_weights.progress * clipped_progress * negative_scale
    clearance_signal = float(np.tanh(clearance_margin))
    if clearance_signal > 0.0:
        clearance_signal *= float(np.clip(reward_weights.clearance_positive_scale, 0.0, 1.0))
    clearance_term = reward_weights.clearance * clearance_signal
    danger_threshold = float(max(reward_weights.danger_clearance_threshold, 0.0))
    danger_penalty = 0.0
    if danger_threshold > 0.0 and float(reward_weights.danger_clearance_penalty) != 0.0:
        shortfall = max(danger_threshold - float(clearance_margin), 0.0)
        if shortfall > 0.0:
            power = float(max(reward_weights.danger_clearance_power, 1e-6))
            normalized_shortfall = shortfall / max(danger_threshold, 1e-6)
            danger_penalty = -abs(float(reward_weights.danger_clearance_penalty)) * (normalized_shortfall**power)
    effort_scale = float(np.linalg.norm(action_array) / np.sqrt(max(action_array.size, 1)))
    remaining_distance_term = reward_weights.remaining_distance * float(np.clip(remaining_distance_ratio, 0.0, 1.5))
    reward_components = {
        "goal": reward_weights.goal if goal_now else 0.0,
        "progress": progress_term,
        "collision": reward_weights.collision if collision_now else 0.0,
        "clearance": clearance_term,
        "danger_clearance": danger_penalty,
        "effort": reward_weights.effort * effort_scale,
        "time": reward_weights.time,
        "remaining_distance": remaining_distance_term,
    }
    return float(sum(reward_components.values())), reward_components
