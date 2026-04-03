"""Plotting helpers for saved training artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt


def load_jsonl(path: str | Path) -> List[dict]:
    """Load a JSONL file into a list of dictionaries."""
    payload = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                payload.append(json.loads(stripped))
    return payload


def _rolling_mean(values: Iterable[float], window: int) -> List[float]:
    """Compute a simple causal rolling mean."""
    series = [float(value) for value in values]
    if not series:
        return []
    rolling = []
    for index in range(len(series)):
        start = max(0, index + 1 - window)
        window_values = series[start : index + 1]
        rolling.append(sum(window_values) / len(window_values))
    return rolling


def plot_training_history(history_path: str | Path, output_dir: str | Path) -> Path:
    """Render a compact training overview figure from saved JSONL history."""
    history_path = Path(history_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = load_jsonl(history_path)
    if not history:
        raise ValueError(f"No history records found in {history_path}")

    eval_history_path = history_path.with_name("eval_history.jsonl")
    eval_history = load_jsonl(eval_history_path) if eval_history_path.exists() else []

    episodes = [int(item.get("episode", index + 1)) for index, item in enumerate(history)]
    returns = [float(item.get("episode_return", 0.0)) for item in history]
    successes = [float(item.get("success", 0.0)) for item in history]
    collisions = [float(item.get("collision", 0.0)) for item in history]
    min_clearances = [float(item.get("min_clearance", 0.0)) for item in history]
    rolling_window = min(20, len(returns))

    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex="col")
    ax_return, ax_success, ax_collision, ax_clearance = axes.flatten()

    ax_return.plot(episodes, returns, color="#4c78a8", alpha=0.35, linewidth=1.25, label="episode return")
    ax_return.plot(
        episodes,
        _rolling_mean(returns, rolling_window),
        color="#1f4e79",
        linewidth=2.2,
        label=f"rolling return ({rolling_window})",
    )
    ax_return.set_ylabel("Return")
    ax_return.set_title("Training Return")
    ax_return.legend(loc="best")
    ax_return.grid(alpha=0.3)

    ax_success.plot(
        episodes,
        _rolling_mean(successes, rolling_window),
        color="#2a9d8f",
        linewidth=2.2,
        label=f"rolling train success ({rolling_window})",
    )
    if eval_history:
        ax_success.plot(
            [int(item.get("train_episode", 0)) for item in eval_history],
            [float(item.get("success_rate", 0.0)) for item in eval_history],
            color="#1d3557",
            marker="o",
            linewidth=1.6,
            label="eval success",
        )
    ax_success.set_ylabel("Success Rate")
    ax_success.set_ylim(-0.05, 1.05)
    ax_success.set_title("Success Over Time")
    ax_success.legend(loc="best")
    ax_success.grid(alpha=0.3)

    ax_collision.plot(
        episodes,
        _rolling_mean(collisions, rolling_window),
        color="#e76f51",
        linewidth=2.2,
        label=f"rolling train collision ({rolling_window})",
    )
    if eval_history:
        ax_collision.plot(
            [int(item.get("train_episode", 0)) for item in eval_history],
            [float(item.get("collision_rate", 0.0)) for item in eval_history],
            color="#b22222",
            marker="o",
            linewidth=1.6,
            label="eval collision",
        )
    ax_collision.set_ylabel("Collision Rate")
    ax_collision.set_xlabel("Episode")
    ax_collision.set_ylim(-0.05, 1.05)
    ax_collision.set_title("Collision Over Time")
    ax_collision.legend(loc="best")
    ax_collision.grid(alpha=0.3)

    ax_clearance.plot(
        episodes,
        _rolling_mean(min_clearances, rolling_window),
        color="#6c5ce7",
        linewidth=2.2,
        label="rolling min clearance",
    )
    if eval_history:
        ax_clearance.plot(
            [int(item.get("train_episode", 0)) for item in eval_history],
            [float(item.get("avg_min_clearance", 0.0)) for item in eval_history],
            color="#3a0ca3",
            marker="o",
            linewidth=1.6,
            label="eval avg min clearance",
        )
    ax_clearance.set_ylabel("Clearance")
    ax_clearance.set_xlabel("Episode")
    ax_clearance.set_title("Safety Margin")
    ax_clearance.legend(loc="best")
    ax_clearance.grid(alpha=0.3)

    figure.tight_layout()
    output_path = output_dir / "training_overview.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
