"""Tests for script entrypoints and package export surfaces."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
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
    expected_plot = Path("results/plots/training_overview.png")

    class DummyConfig:
        def __init__(self) -> None:
            self.training = type("Training", (), {"num_episodes": 10})()
            self.environment = type("Environment", (), {"seed": 7})()
            self.visualization = type("Visualization", (), {"plot_dir": "results/plots"})()
            self.training.results_dir = "results"

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
            "avg_path_length": 3.0,
            "avg_steps": 42.0,
            "avg_episode_duration": 1.4,
            "avg_time_to_goal": 1.1,
            "avg_min_clearance": 0.4,
            "avg_control_effort": 0.2,
            "avg_path_efficiency": 0.5,
            "best_eval_success_rate": 0.6,
            "best_eval_collision_rate": 0.1,
            "best_eval_avg_episode_return": 1.2,
            "completed_stage_name": "bridge_crossing_easy",
            "stage_regression_event_count": 1,
            "stage_rollback_count": 1,
            "best_model_path": "checkpoints/best_model.pth",
            "last_model_path": "checkpoints/last_model.pth",
        },
    )
    monkeypatch.setattr(module, "plot_training_history", lambda **kwargs: expected_plot)
    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--config", "configs/debug.yaml", "--episodes", "4", "--seed", "99"],
    )

    assert module.main() == 0
    output = capsys.readouterr().out
    assert f"plot={expected_plot}" in output
    assert "training_episodes=4" in output
    assert "avg_path_length=3.000" in output
    assert "completed_stage=bridge_crossing_easy" in output
    assert "last_model=checkpoints/last_model.pth" in output
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
            "avg_steps": 120.0,
            "avg_episode_duration": 4.0,
            "avg_time_to_goal": 2.5,
            "avg_min_clearance": 1.5,
            "avg_control_effort": 0.7,
            "avg_path_efficiency": 0.5,
            "best_episode_return": 3.0,
            "num_episodes": 2,
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
    assert "avg_steps=120.000" in output
    assert "output_dir=tmp/eval" in output
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


def test_generate_stage_showcase_script_main_parses_args(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Stage showcase CLI should resolve cases and print a compact summary."""
    module = _import_script_module("generate_stage_showcase")

    class DummyConfig:
        def __init__(self) -> None:
            self.environment = type("Environment", (), {"seed": 7})()
            self.evaluation = type("Evaluation", (), {"num_episodes": 5})()
            self.training = type(
                "Training",
                (),
                {
                    "results_dir": str(tmp_path / "results"),
                    "checkpoint_dir": str(tmp_path / "checkpoints"),
                    "eval_episodes": 4,
                    "resume": {},
                    "curriculum": [
                        {"name": "target_bypass_intro_v2"},
                        {"name": "target_default"},
                    ],
                },
            )()

    config = DummyConfig()

    monkeypatch.setattr(module, "load_config", lambda path: config)
    monkeypatch.setattr(module, "validate_config", lambda cfg: None)
    monkeypatch.setattr(module, "_auto_best_reached_stage", lambda cfg: "target_bypass_intro_v2")
    monkeypatch.setattr(
        module,
        "_resolve_model_path",
        lambda **kwargs: (Path(tmp_path / "checkpoints" / "mock_model.pth"), "mock"),
    )
    monkeypatch.setattr(
        module,
        "_generate_case",
        lambda **kwargs: {
            "summary": {"success_rate": 0.5, "collision_rate": 0.0},
            "stage_name": kwargs["stage_name"],
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_stage_showcase.py",
            "--config",
            "configs/default.yaml",
            "--output-dir",
            str(tmp_path / "showcase"),
            "--no-video",
        ],
    )

    assert module.main() == 0
    output = capsys.readouterr().out
    assert f"output_dir={tmp_path / 'showcase'}" in output
    assert "best_reached_stage=target_bypass_intro_v2" in output
    assert "wanted_stage=target_default" in output


def test_generate_stage_showcase_records_frames_into_video_writer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Video replay should write multiple GUI frames into the ffmpeg writer abstraction."""
    module = _import_script_module("generate_stage_showcase")

    class DummyEnv:
        ctrl_freq = 30

        def __init__(self) -> None:
            self.steps = 0

        def reset(self, seed=None):
            del seed
            return {"node_features": np.zeros((1, 1), dtype=np.float32)}, {}

        def capture_gui_frame(self):
            return np.full((4, 6, 3), self.steps, dtype=np.uint8)

        def step(self, action):
            del action
            self.steps += 1
            return {"node_features": np.zeros((1, 1), dtype=np.float32)}, 0.0, self.steps >= 2, False, {}

        def get_episode_summary(self):
            return {"success": 1.0, "distance_to_goal": 0.0, "episode_return": 1.0}

        def close(self):
            return None

    class DummyAgent:
        def load(self, path):
            del path

        def select_action(self, observation, deterministic=True):
            del observation, deterministic
            return np.zeros((4,), dtype=np.float32), {}

    written_frames: list[np.ndarray] = []

    class DummyWriter:
        def __init__(self, *, output_path: Path, width: int, height: int, fps: int) -> None:
            self.output_path = Path(output_path)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_bytes(b"placeholder-video-data" * 80)
            self.width = width
            self.height = height
            self.fps = fps

        def write(self, frame: np.ndarray) -> None:
            written_frames.append(np.asarray(frame))

        def close(self) -> None:
            return None

    monkeypatch.setattr(module, "create_environment", lambda *args, **kwargs: DummyEnv())
    monkeypatch.setattr(module, "create_agent", lambda *args, **kwargs: DummyAgent())
    monkeypatch.setattr(module, "_FFmpegVideoWriter", DummyWriter)

    config = type("Config", (), {"environment": type("EnvCfg", (), {"seed": 7})()})()
    metrics = module._record_episode_video(
        config=config,
        model_path=tmp_path / "model.pth",
        episode_seed=101,
        output_path=tmp_path / "episode.mp4",
    )

    assert metrics["success"] == 1.0
    assert len(written_frames) >= 3


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
