"""Tests for the abstract base environment interface."""

from __future__ import annotations

from src.environments.base_env import BaseEnvironment


class DummyBaseEnvironment(BaseEnvironment):
    """Concrete test double for the abstract base class."""

    def get_episode_summary(self) -> dict[str, float]:
        return {"episode_return": 0.0}

    def export_episode(self) -> dict[str, object]:
        return {"positions": []}


def test_base_environment_stores_core_runtime_state() -> None:
    """The base environment should keep shared counters and constructor args."""
    env = DummyBaseEnvironment(num_agents=2, max_episode_steps=25)
    assert env.num_agents == 2
    assert env.max_episode_steps == 25
    assert env.current_step == 0
