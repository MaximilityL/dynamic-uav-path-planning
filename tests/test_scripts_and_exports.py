"""Tests for script entrypoints and package export surfaces."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from src.agents import GraphPPOAgent
from src.environments import DynamicAirspaceEnv
from src.training import create_agent, create_environment, evaluate_agent, run_episode, set_global_seeds, train_agent
from src.training.factories import build_output_layout


def _import_script_module(module_name: str):
    """Import a script module the way Python sees it from the scripts directory."""
    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"
    scripts_path = str(scripts_dir)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_public_exports_are_stable() -> None:
    """Package-level exports should still expose the current public API."""
    assert GraphPPOAgent.__name__ == "GraphPPOAgent"
    assert DynamicAirspaceEnv.__name__ == "DynamicAirspaceEnv"
    assert callable(create_environment)
    assert callable(create_agent)
    assert callable(run_episode)
    assert callable(train_agent)
    assert callable(evaluate_agent)
    assert callable(set_global_seeds)
    assert callable(build_output_layout)
    assert importlib.import_module("src.core").__name__ == "src.core"
    assert importlib.import_module("src.utils").__name__ == "src.utils"
    assert importlib.import_module("src.evaluation").__name__ == "src.evaluation"
    assert importlib.import_module("src.visualization").__name__ == "src.visualization"


def test_bootstrap_project_adds_repo_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bootstrap should return the project root and keep it on sys.path."""
    common = _import_script_module("_common")
    monkeypatch.setenv("DUPP_SKIP_VENV", "1")
    root = common.bootstrap_project()
    assert Path(root).exists()
    assert str(root) in sys.path


def test_train_script_main_parses_args(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Train CLI should apply overrides and print the returned summary."""
    module = _import_script_module("train")

    class DummyConfig:
        def __init__(self) -> None:
            self.training = type("Training", (), {"num_episodes": 10})()
            self.environment = type("Environment", (), {"seed": 7})()

    config = DummyConfig()

    monkeypatch.setattr(module, "load_config", lambda path: config)
    monkeypatch.setattr(module, "validate_config", lambda cfg: None)
    monkeypatch.setattr(
        module,
        "train_agent",
        lambda **kwargs: {
            "num_episodes": kwargs["num_episodes"],
            "avg_episode_return": 1.0,
            "success_rate": 0.5,
            "collision_rate": 0.25,
            "avg_min_clearance": 0.4,
            "avg_control_effort": 0.2,
            "best_eval_success_rate": 0.6,
            "best_eval_collision_rate": 0.1,
            "best_eval_avg_episode_return": 1.2,
            "best_model_path": "checkpoints/best_model.pth",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", "configs/debug.yaml", "--episodes", "4", "--seed", "99"],
    )

    assert module.main() == 0
    output = capsys.readouterr().out
    assert "training_episodes=4" in output
    assert config.training.num_episodes == 4
    assert config.environment.seed == 99


def test_evaluate_script_main_parses_args(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Evaluate CLI should apply render/output overrides and print summary metrics."""
    module = _import_script_module("evaluate")

    class DummyConfig:
        def __init__(self) -> None:
            self.environment = type("Environment", (), {"seed": 7})()
            self.evaluation = type("Evaluation", (), {"num_episodes": 3, "render": False, "deterministic": True, "output_dir": "results/eval"})()

    config = DummyConfig()

    monkeypatch.setattr(module, "load_config", lambda path: config)
    monkeypatch.setattr(module, "validate_config", lambda cfg: None)
    monkeypatch.setattr(
        module,
        "evaluate_agent",
        lambda **kwargs: {
            "success_rate": 0.5,
            "collision_rate": 0.0,
            "avg_episode_return": 2.0,
            "avg_path_length": 4.5,
            "avg_time_to_goal": 2.5,
            "avg_min_clearance": 1.5,
            "avg_control_effort": 0.7,
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--config",
            "configs/default.yaml",
            "--model",
            "checkpoints/best_model.pth",
            "--episodes",
            "2",
            "--render",
            "--output",
            "tmp/eval",
            "--seed",
            "101",
        ],
    )

    assert module.main() == 0
    output = capsys.readouterr().out
    assert "success_rate=0.500" in output
    assert config.evaluation.num_episodes == 2
    assert config.evaluation.render is True
    assert config.environment.seed == 101
    assert config.evaluation.output_dir.endswith("tmp/eval")


def test_plot_script_main_parses_args(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Plot CLI should forward args to the plotting helper and print the output path."""
    module = _import_script_module("plot_results")
    expected = Path("results/plots/training_overview.png")

    monkeypatch.setattr(module, "plot_training_history", lambda **kwargs: expected)
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_results.py", "--history", "results/train/history.jsonl", "--output-dir", "results/plots"],
    )

    assert module.main() == 0
    output = capsys.readouterr().out
    assert f"plot={expected}" in output


def test_smoke_test_script_main_runs(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Smoke test CLI should return success with a lightweight fake path."""
    module = _import_script_module("smoke_test")

    class DummySpace:
        shape = (4,)

        @staticmethod
        def sample():
            import numpy as np

            return np.zeros((4,), dtype=np.float32)

    class DummyEnv:
        action_space = DummySpace()

        def __init__(self) -> None:
            self.calls = 0

        def reset(self, seed=None):
            import numpy as np

            del seed
            return {
                "node_features": np.zeros((3, 2), dtype=np.float32),
                "adjacency": np.zeros((3, 3), dtype=np.float32),
            }, {}

        def step(self, action):
            import numpy as np

            del action
            self.calls += 1
            return (
                {
                    "node_features": np.zeros((3, 2), dtype=np.float32),
                    "adjacency": np.zeros((3, 3), dtype=np.float32),
                },
                1.0,
                self.calls >= 1,
                False,
                {"distance_to_goal": 1.0, "min_obstacle_distance": 2.0},
            )

        def close(self):
            return None

    class DummyAgent:
        def update(self):
            return {"actor_loss": 0.1, "critic_loss": 0.2, "entropy": 0.3}

    dummy_env = DummyEnv()
    dummy_agent = DummyAgent()

    monkeypatch.setattr(module, "load_config", lambda path: type("Config", (), {
        "environment": type("Environment", (), {"seed": 7, "max_episode_steps": 20, "num_dynamic_obstacles": 3})(),
        "agent": type("Agent", (), {"rollout_episodes": 1, "ppo_epochs": 1, "mini_batch_size": 8})(),
    })())
    monkeypatch.setattr(module, "validate_config", lambda cfg: None)
    monkeypatch.setattr(module, "set_global_seeds", lambda seed: None)
    monkeypatch.setattr(module, "create_environment", lambda *args, **kwargs: dummy_env)
    monkeypatch.setattr(module, "create_agent", lambda *args, **kwargs: dummy_agent)
    monkeypatch.setattr(
        module,
        "run_episode",
        lambda **kwargs: (
            {"episode_return": 1.0, "success": 1.0, "collision": 0.0},
            {},
        ),
    )
    monkeypatch.setattr(sys, "argv", ["smoke_test.py", "--config", "configs/debug.yaml"])

    assert module.main() == 0
    output = capsys.readouterr().out
    assert "smoke_test=passed" in output
