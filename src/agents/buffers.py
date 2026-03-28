"""Rollout buffer helpers for PPO training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .tensor_ops import GraphObservation


@dataclass
class EpisodeBuffer:
    """Temporary storage for one episode before GAE is computed."""

    observations: List[GraphObservation] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[float] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def clear(self) -> None:
        """Drop all stored data."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()


@dataclass
class RolloutBuffer:
    """Aggregated rollout data ready for PPO updates."""

    observations: List[GraphObservation] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)

    def clear(self) -> None:
        """Drop all stored data."""
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.returns.clear()
        self.advantages.clear()

    def __len__(self) -> int:
        return len(self.actions)
