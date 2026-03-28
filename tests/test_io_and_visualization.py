"""Tests for utility I/O and plotting modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.utils.io import append_jsonl, save_json, save_npz
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
    for episode, reward, success, collision in [
        (1, 1.0, 0.0, 0.0),
        (2, 2.0, 1.0, 0.0),
        (3, -1.0, 0.0, 1.0),
    ]:
        append_jsonl(
            history_path,
            {
                "episode": episode,
                "episode_return": reward,
                "success": success,
                "collision": collision,
            },
        )

    output_path = plot_training_history(history_path=history_path, output_dir=tmp_path / "plots")
    assert output_path.exists()
    assert output_path.suffix == ".png"
