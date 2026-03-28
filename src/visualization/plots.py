"""Plotting helpers for saved training artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

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


def plot_training_history(history_path: str | Path, output_dir: str | Path) -> Path:
    """Render a compact training overview figure from saved JSONL history."""
    history_path = Path(history_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = load_jsonl(history_path)
    if not history:
        raise ValueError(f"No history records found in {history_path}")

    episodes = [int(item.get("episode", index + 1)) for index, item in enumerate(history)]
    returns = [float(item.get("episode_return", 0.0)) for item in history]
    successes = [float(item.get("success", 0.0)) for item in history]
    collisions = [float(item.get("collision", 0.0)) for item in history]

    figure, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(episodes, returns, color="#1f77b4", linewidth=2)
    axes[0].set_ylabel("Return")
    axes[0].set_title("Training Overview")
    axes[0].grid(alpha=0.3)

    axes[1].plot(episodes, successes, color="#2ca02c", linewidth=2, label="success")
    axes[1].plot(episodes, collisions, color="#d62728", linewidth=2, label="collision")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    rolling_window = min(10, len(returns))
    rolling = []
    for index in range(len(returns)):
        start = max(0, index + 1 - rolling_window)
        window = returns[start : index + 1]
        rolling.append(sum(window) / len(window))
    axes[2].plot(episodes, rolling, color="#9467bd", linewidth=2)
    axes[2].set_ylabel("Rolling Return")
    axes[2].set_xlabel("Episode")
    axes[2].grid(alpha=0.3)

    figure.tight_layout()
    output_path = output_dir / "training_overview.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
