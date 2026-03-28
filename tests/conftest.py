"""Shared pytest fixtures for repository tests."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces


class FakeMultiHoverAviary:
    """Small test double for the PyBullet aviary backend."""

    def __init__(
        self,
        *,
        num_drones: int,
        initial_xyzs: np.ndarray,
        obs,
        act,
        gui: bool,
        record: bool,
        pyb_freq: int,
        ctrl_freq: int,
    ) -> None:
        del num_drones, obs, act, gui, record, pyb_freq, ctrl_freq
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1, 4), dtype=np.float32)
        self.INIT_XYZS = np.asarray(initial_xyzs, dtype=np.float32)
        self.CLIENT = 0
        self._position = self.INIT_XYZS[0].copy()
        self._velocity = np.zeros(3, dtype=np.float32)

    def reset(self, seed: int | None = None) -> None:
        del seed
        self._position = np.asarray(self.INIT_XYZS[0], dtype=np.float32).copy()
        self._velocity = np.zeros(3, dtype=np.float32)

    def step(self, action: np.ndarray) -> None:
        command = np.asarray(action, dtype=np.float32).reshape(1, -1)[0]
        self._velocity = command[:3] * 0.05
        self._position = self._position + self._velocity
        self._position[2] = max(self._position[2], 0.1)

    def _getDroneStateVector(self, drone_idx: int) -> np.ndarray:
        del drone_idx
        state = np.zeros(13, dtype=np.float32)
        state[0:3] = self._position
        state[10:13] = self._velocity
        return state

    def close(self) -> None:
        """Match the real backend close signature."""


@pytest.fixture(autouse=True)
def fake_aviary(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the environment module to use a lightweight backend double."""
    import src.environments.dynamic_airspace_env as env_module

    monkeypatch.setattr(env_module, "MultiHoverAviary", FakeMultiHoverAviary)
