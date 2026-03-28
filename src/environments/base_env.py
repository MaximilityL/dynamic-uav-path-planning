"""Base environment interfaces for UAV path-planning tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import gymnasium as gym


class BaseEnvironment(gym.Env, ABC):
    """Minimal abstract base class for the scaffold environments."""

    def __init__(self, num_agents: int, max_episode_steps: int) -> None:
        super().__init__()
        self.num_agents = int(num_agents)
        self.max_episode_steps = int(max_episode_steps)
        self.current_step = 0

    @abstractmethod
    def get_episode_summary(self) -> Dict[str, float]:
        """Return the current episode summary."""

    @abstractmethod
    def export_episode(self) -> Dict[str, object]:
        """Export arrays and metadata for the latest episode."""

