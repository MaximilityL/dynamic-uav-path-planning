"""Tests for utility I/O and plotting modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils.io import append_jsonl, save_json, save_npz
from src.visualization import plot_episode_trajectory_2d, plot_episode_trajectory_3d, save_episode_showcase_plots
from src.visualization.plots import load_jsonl, plot_training_history


def test_save_helpers_and_plot_loader_round_trip(tmp_path: Path) -> None:
    """Artifact helpers should write readable JSON, JSONL, and NPZ payloads."""
    json_path = save_json(
        tmp_path / "payload.json",
        {
            "array": np.asarray([1, 2, 3], dtype=np.int64),
            "flag": np.bool_(True),
        },
    )
    jsonl_path = append_jsonl(tmp_path / "history.jsonl", {"episode": 1, "episode_return": np.float32(1.5)})
    append_jsonl(jsonl_path, {"episode": 2, "episode_return": np.float32(2.5), "success": np.float32(1.0)})
    npz_path = save_npz(tmp_path / "trajectory.npz", {"positions": np.zeros((2, 3), dtype=np.float32)})

    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["array"] == [1, 2, 3]
    assert loaded_json["flag"] is True

    history = load_jsonl(jsonl_path)
    assert len(history) == 2
    assert history[1]["episode"] == 2

    npz = np.load(npz_path)
    assert npz["positions"].shape == (2, 3)


def test_plot_training_history_generates_figure(tmp_path: Path) -> None:
    """The plotting helper should generate an image from JSONL history."""
    history_path = tmp_path / "history.jsonl"
    for episode, reward, success, collision, min_clearance in [
        (1, 1.0, 0.0, 0.0, 0.5),
        (2, 2.0, 1.0, 0.0, 0.7),
        (3, -1.0, 0.0, 1.0, -0.1),
    ]:
        append_jsonl(
            history_path,
            {
                "episode": episode,
                "episode_return": reward,
                "success": success,
                "collision": collision,
                "min_clearance": min_clearance,
            },
        )
    append_jsonl(
        tmp_path / "eval_history.jsonl",
        {
            "train_episode": 3,
            "success_rate": 0.5,
            "collision_rate": 0.25,
            "avg_min_clearance": 0.2,
        },
    )

    output_path = plot_training_history(history_path=history_path, output_dir=tmp_path / "plots")
    assert output_path.exists()
    assert output_path.suffix == ".png"


def test_plot_training_report_marks_stage_regression(tmp_path: Path) -> None:
    """The training report should flag when a stage is solved and then forgotten."""
    history_path = tmp_path / "history.jsonl"
    for episode in range(1, 5):
        append_jsonl(
            history_path,
            {
                "episode": episode,
                "stage_name": "bridge_crossing_commit",
                "episode_return": float(episode),
                "success": 1.0 if episode >= 2 else 0.0,
                "collision": 0.0,
                "min_clearance": 0.4,
            },
        )

    eval_history_path = tmp_path / "eval_history.jsonl"
    for train_episode, success_rate in [(2, 0.9), (3, 0.6), (4, 0.2)]:
        append_jsonl(
            eval_history_path,
            {
                "train_episode": train_episode,
                "stage_name": "bridge_crossing_commit",
                "success_rate": success_rate,
                "collision_rate": 0.0,
                "avg_episode_return": float(train_episode),
                "avg_min_clearance": 0.3,
            },
        )

    plot_training_history(history_path=history_path, output_dir=tmp_path / "plots")
    report = json.loads((tmp_path / "plots" / "training_report.json").read_text(encoding="utf-8"))
    assert report["stage_report"][0]["status"] == "regressed_after_solving"


def test_episode_showcase_plots_generate_2d_and_3d_outputs(tmp_path: Path) -> None:
    """Trajectory showcase helpers should write readable 2D and 3D figures."""
    trajectory = {
        "positions": np.asarray(
            [
                [-2.0, 0.0, 1.0],
                [-1.0, 0.2, 1.0],
                [0.0, 0.4, 1.1],
                [1.0, 0.1, 1.0],
            ],
            dtype=np.float32,
        ),
        "obstacles": np.asarray(
            [
                [[-0.8, -0.4, 1.0], [0.1, 0.7, 1.2]],
                [[-0.5, -0.2, 1.0], [0.2, 0.5, 1.2]],
                [[-0.2, 0.0, 1.0], [0.3, 0.3, 1.2]],
                [[0.1, 0.2, 1.0], [0.4, 0.1, 1.2]],
            ],
            dtype=np.float32,
        ),
        "goal": np.asarray([1.5, 0.0, 1.0], dtype=np.float32),
        "start_position": np.asarray([-2.0, 0.0, 1.0], dtype=np.float32),
        "workspace_bounds": np.asarray([[-3.0, 3.0], [-3.0, 3.0], [0.5, 2.5]], dtype=np.float32),
        "goal_tolerance": np.float32(0.35),
        "summary": {
            "episode_return": 12.5,
            "success": 1.0,
            "collision": 0.0,
            "steps": 4.0,
            "min_clearance": 0.42,
        },
    }

    plot_2d = plot_episode_trajectory_2d(trajectory=trajectory, output_path=tmp_path / "trajectory_2d.png", title="2D")
    plot_3d = plot_episode_trajectory_3d(trajectory=trajectory, output_path=tmp_path / "trajectory_3d.png", title="3D")
    paired = save_episode_showcase_plots(
        trajectory=trajectory,
        output_dir=tmp_path / "showcase",
        stem="episode",
        title="Showcase",
    )

    assert plot_2d.exists()
    assert plot_3d.exists()
    assert paired["plot_2d"].exists()
    assert paired["plot_3d"].exists()
