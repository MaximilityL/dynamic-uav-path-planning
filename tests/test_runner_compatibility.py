"""Tests for training runner compatibility exports."""

from __future__ import annotations

from src.training import runner
from src.training.factories import create_agent as factory_create_agent
from src.training.factories import create_environment as factory_create_environment
from src.training.factories import set_global_seeds as factory_set_global_seeds
from src.training.loops import evaluate_agent as loop_evaluate_agent
from src.training.loops import run_episode as loop_run_episode
from src.training.loops import train_agent as loop_train_agent


def test_runner_reexports_match_split_modules() -> None:
    """The compatibility runner should point at the split module implementations."""
    assert runner.create_agent is factory_create_agent
    assert runner.create_environment is factory_create_environment
    assert runner.set_global_seeds is factory_set_global_seeds
    assert runner.run_episode is loop_run_episode
    assert runner.train_agent is loop_train_agent
    assert runner.evaluate_agent is loop_evaluate_agent
