"""Plotting helpers for saved training artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

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


def _stage_spans(history: List[dict]) -> List[dict]:
    """Resolve contiguous stage spans from episode history."""
    if not history:
        return []

    spans: List[dict] = []
    current_name = str(history[0].get("stage_name", "main"))
    start_episode = int(history[0].get("episode", 1))
    last_episode = start_episode

    for item in history[1:]:
        stage_name = str(item.get("stage_name", "main"))
        episode = int(item.get("episode", last_episode + 1))
        if stage_name != current_name:
            spans.append(
                {
                    "stage_name": current_name,
                    "start_episode": start_episode,
                    "end_episode": last_episode,
                    "num_episodes": last_episode - start_episode + 1,
                }
            )
            current_name = stage_name
            start_episode = episode
        last_episode = episode

    spans.append(
        {
            "stage_name": current_name,
            "start_episode": start_episode,
            "end_episode": last_episode,
            "num_episodes": last_episode - start_episode + 1,
        }
    )
    return spans


def _stage_palette(stage_spans: List[dict]) -> Dict[str, str]:
    """Assign a stable soft background color to each stage."""
    palette = ["#e8f1ff", "#eef8e8", "#fff2e5", "#f7ebff", "#f2f2f2"]
    mapping: Dict[str, str] = {}
    for index, span in enumerate(stage_spans):
        stage_name = str(span["stage_name"])
        if stage_name not in mapping:
            mapping[stage_name] = palette[index % len(palette)]
    return mapping


def _add_stage_background(ax: plt.Axes, stage_spans: List[dict], palette: Dict[str, str]) -> None:
    """Shade stage ranges in the background of one axis."""
    for span in stage_spans:
        ax.axvspan(
            float(span["start_episode"]),
            float(span["end_episode"]),
            color=palette[str(span["stage_name"])],
            alpha=0.18,
            linewidth=0,
        )


def _latest_value(history: List[dict], key: str, default: float = 0.0) -> float:
    """Fetch the latest numeric value from history."""
    if not history:
        return default
    return float(history[-1].get(key, default))


def _stage_status(records: List[dict]) -> str:
    """Provide a lightweight stage diagnosis from evaluation history."""
    if not records:
        return "not_reached"

    best_success = max(float(item.get("success_rate", 0.0)) for item in records)
    latest = records[-1]
    latest_success = float(latest.get("success_rate", 0.0))
    latest_collision = float(latest.get("collision_rate", 0.0))

    if best_success >= 0.8:
        return "solved_or_near_solved"
    if latest_success > 0.0:
        return "partial_progress"
    if latest_collision <= 0.05:
        return "safe_timeout_pattern"
    return "collision_limited"


def _write_training_report(
    *,
    history_path: Path,
    output_dir: Path,
    history: List[dict],
    eval_history: List[dict],
    stage_spans: List[dict],
    generated_plots: List[str],
) -> Path:
    """Write a compact machine-readable run summary alongside the plots."""
    eval_by_stage: Dict[str, List[dict]] = defaultdict(list)
    for item in eval_history:
        eval_by_stage[str(item.get("stage_name", "main"))].append(item)

    stage_report = []
    for span in stage_spans:
        stage_name = str(span["stage_name"])
        records = eval_by_stage.get(stage_name, [])
        best_success = max((float(item.get("success_rate", 0.0)) for item in records), default=0.0)
        best_collision = min((float(item.get("collision_rate", 1.0)) for item in records), default=1.0)
        best_return = max((float(item.get("avg_episode_return", float("-inf"))) for item in records), default=float("-inf"))
        latest_record = records[-1] if records else {}
        stage_report.append(
            {
                **span,
                "num_evaluations": len(records),
                "best_eval_success_rate": best_success,
                "best_eval_collision_rate": best_collision if records else 0.0,
                "best_eval_avg_return": 0.0 if best_return == float("-inf") else best_return,
                "latest_eval_success_rate": float(latest_record.get("success_rate", 0.0)),
                "latest_eval_collision_rate": float(latest_record.get("collision_rate", 0.0)),
                "latest_eval_avg_return": float(latest_record.get("avg_episode_return", 0.0)),
                "status": _stage_status(records),
            }
        )

    payload = {
        "history_path": str(history_path),
        "num_training_episodes": len(history),
        "num_evaluations": len(eval_history),
        "latest_episode": int(history[-1].get("episode", len(history))) if history else 0,
        "latest_stage_name": str(history[-1].get("stage_name", "main")) if history else "main",
        "latest_rolling_success_rate": _latest_value(history, "rolling_success_rate"),
        "latest_rolling_avg_episode_return": _latest_value(history, "rolling_avg_episode_return"),
        "latest_actor_loss": _latest_value(history, "actor_loss"),
        "latest_critic_loss": _latest_value(history, "critic_loss"),
        "latest_entropy": _latest_value(history, "entropy"),
        "stage_report": stage_report,
        "generated_plots": generated_plots,
    }

    report_path = output_dir / "training_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return report_path


def plot_training_history(history_path: str | Path, output_dir: str | Path) -> Path:
    """Render several training figures and a compact run report."""
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
    path_lengths = [float(item.get("path_length", 0.0)) for item in history]
    steps = [float(item.get("steps", 0.0)) for item in history]
    control_effort = [float(item.get("control_effort", 0.0)) for item in history]
    start_goal_distance = [float(item.get("start_to_goal_distance", 0.0)) for item in history]
    actor_losses = [float(item.get("actor_loss", 0.0)) for item in history]
    critic_losses = [float(item.get("critic_loss", 0.0)) for item in history]
    entropy = [float(item.get("entropy", 0.0)) for item in history]
    rolling_success = [float(item.get("rolling_success_rate", 0.0)) for item in history]
    rolling_return = [float(item.get("rolling_avg_episode_return", 0.0)) for item in history]
    rolling_window = min(20, len(returns))
    stage_spans = _stage_spans(history)
    palette = _stage_palette(stage_spans)
    generated_plots: List[str] = []

    figure, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    ax_return, ax_success, ax_collision, ax_clearance, ax_path, ax_steps = axes.flatten()
    for axis in axes.flatten():
        _add_stage_background(axis, stage_spans, palette)

    ax_return.plot(episodes, returns, color="#4c78a8", alpha=0.35, linewidth=1.25, label="episode return")
    ax_return.plot(episodes, rolling_return or _rolling_mean(returns, rolling_window), color="#1f4e79", linewidth=2.2, label=f"rolling return ({rolling_window})")
    ax_return.set_ylabel("Return")
    ax_return.set_title("Training Return")
    ax_return.legend(loc="best")
    ax_return.grid(alpha=0.3)

    ax_success.plot(episodes, rolling_success or _rolling_mean(successes, rolling_window), color="#2a9d8f", linewidth=2.2, label=f"rolling train success ({rolling_window})")
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

    ax_collision.plot(episodes, _rolling_mean(collisions, rolling_window), color="#e76f51", linewidth=2.2, label=f"rolling train collision ({rolling_window})")
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

    ax_clearance.plot(episodes, _rolling_mean(min_clearances, rolling_window), color="#6c5ce7", linewidth=2.2, label="rolling min clearance")
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
    ax_clearance.set_title("Safety Margin")
    ax_clearance.legend(loc="best")
    ax_clearance.grid(alpha=0.3)

    ax_path.plot(episodes, _rolling_mean(path_lengths, rolling_window), color="#f4a261", linewidth=2.2, label="rolling path length")
    if eval_history:
        ax_path.plot(
            [int(item.get("train_episode", 0)) for item in eval_history],
            [float(item.get("avg_path_length", 0.0)) for item in eval_history],
            color="#bc6c25",
            marker="o",
            linewidth=1.6,
            label="eval avg path length",
        )
    ax_path.set_ylabel("Path Length")
    ax_path.set_xlabel("Episode")
    ax_path.set_title("Path Length")
    ax_path.legend(loc="best")
    ax_path.grid(alpha=0.3)

    ax_steps.plot(episodes, _rolling_mean(steps, rolling_window), color="#577590", linewidth=2.2, label="rolling steps")
    if eval_history:
        ax_steps.plot(
            [int(item.get("train_episode", 0)) for item in eval_history],
            [float(item.get("avg_steps", 0.0)) for item in eval_history],
            color="#264653",
            marker="o",
            linewidth=1.6,
            label="eval avg steps",
        )
    ax_steps.set_ylabel("Steps")
    ax_steps.set_xlabel("Episode")
    ax_steps.set_title("Episode Length")
    ax_steps.legend(loc="best")
    ax_steps.grid(alpha=0.3)

    figure.tight_layout()
    output_path = output_dir / "training_overview.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    generated_plots.append(output_path.name)

    dynamics_figure, dynamics_axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    ax_effort, ax_distance, ax_losses, ax_entropy = dynamics_axes.flatten()
    for axis in dynamics_axes.flatten():
        _add_stage_background(axis, stage_spans, palette)

    ax_effort.plot(episodes, _rolling_mean(control_effort, rolling_window), color="#8ecae6", linewidth=2.2, label="rolling control effort")
    ax_effort.set_ylabel("Effort")
    ax_effort.set_title("Control Effort")
    ax_effort.legend(loc="best")
    ax_effort.grid(alpha=0.3)

    ax_distance.plot(episodes, _rolling_mean(start_goal_distance, rolling_window), color="#219ebc", linewidth=2.2, label="rolling start-goal distance")
    ax_distance.set_ylabel("Distance")
    ax_distance.set_title("Sampled Difficulty")
    ax_distance.legend(loc="best")
    ax_distance.grid(alpha=0.3)

    ax_losses.plot(episodes, actor_losses, color="#ffb703", alpha=0.9, linewidth=1.4, label="actor loss")
    ax_losses.set_ylabel("Actor Loss")
    ax_losses.set_title("Optimization Losses")
    ax_losses.grid(alpha=0.3)
    ax_losses_right = ax_losses.twinx()
    ax_losses_right.plot(episodes, critic_losses, color="#fb8500", alpha=0.7, linewidth=1.2, label="critic loss")
    ax_losses_right.set_ylabel("Critic Loss")
    actor_handles, actor_labels = ax_losses.get_legend_handles_labels()
    critic_handles, critic_labels = ax_losses_right.get_legend_handles_labels()
    ax_losses.legend(actor_handles + critic_handles, actor_labels + critic_labels, loc="best")

    ax_entropy.plot(episodes, entropy, color="#90be6d", linewidth=1.5, label="entropy")
    ax_entropy.set_ylabel("Entropy")
    ax_entropy.set_xlabel("Episode")
    ax_entropy.set_title("Policy Entropy")
    ax_entropy.legend(loc="best")
    ax_entropy.grid(alpha=0.3)

    dynamics_axes[1, 0].set_xlabel("Episode")
    dynamics_figure.tight_layout()
    dynamics_path = output_dir / "training_dynamics.png"
    dynamics_figure.savefig(dynamics_path, dpi=160)
    plt.close(dynamics_figure)
    generated_plots.append(dynamics_path.name)

    if eval_history:
        eval_episodes = [int(item.get("train_episode", 0)) for item in eval_history]
        eval_figure, eval_axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        ax_eval_success, ax_eval_return, ax_eval_clearance, ax_eval_path = eval_axes.flatten()
        for axis in eval_axes.flatten():
            _add_stage_background(axis, stage_spans, palette)

        ax_eval_success.plot(eval_episodes, [float(item.get("success_rate", 0.0)) for item in eval_history], color="#1d3557", marker="o", linewidth=1.8, label="eval success")
        ax_eval_success.plot(eval_episodes, [float(item.get("collision_rate", 0.0)) for item in eval_history], color="#b22222", marker="o", linewidth=1.5, label="eval collision")
        ax_eval_success.set_ylim(-0.05, 1.05)
        ax_eval_success.set_ylabel("Rate")
        ax_eval_success.set_title("Evaluation Success and Collision")
        ax_eval_success.legend(loc="best")
        ax_eval_success.grid(alpha=0.3)

        ax_eval_return.plot(eval_episodes, [float(item.get("avg_episode_return", 0.0)) for item in eval_history], color="#4c78a8", marker="o", linewidth=1.8)
        ax_eval_return.set_ylabel("Return")
        ax_eval_return.set_title("Evaluation Return")
        ax_eval_return.grid(alpha=0.3)

        ax_eval_clearance.plot(eval_episodes, [float(item.get("avg_min_clearance", 0.0)) for item in eval_history], color="#6c5ce7", marker="o", linewidth=1.8)
        ax_eval_clearance.set_ylabel("Clearance")
        ax_eval_clearance.set_xlabel("Train Episode")
        ax_eval_clearance.set_title("Evaluation Clearance")
        ax_eval_clearance.grid(alpha=0.3)

        ax_eval_path.plot(eval_episodes, [float(item.get("avg_path_length", 0.0)) for item in eval_history], color="#f4a261", marker="o", linewidth=1.8, label="eval path length")
        ax_eval_path_right = ax_eval_path.twinx()
        ax_eval_path_right.plot(eval_episodes, [float(item.get("avg_steps", 0.0)) for item in eval_history], color="#577590", marker="o", linewidth=1.4, label="eval steps")
        ax_eval_path.set_ylabel("Path Length")
        ax_eval_path_right.set_ylabel("Steps")
        ax_eval_path.set_xlabel("Train Episode")
        ax_eval_path.set_title("Evaluation Path and Steps")
        left_handles, left_labels = ax_eval_path.get_legend_handles_labels()
        right_handles, right_labels = ax_eval_path_right.get_legend_handles_labels()
        ax_eval_path.legend(left_handles + right_handles, left_labels + right_labels, loc="best")
        ax_eval_path.grid(alpha=0.3)

        eval_figure.tight_layout()
        eval_path = output_dir / "evaluation_overview.png"
        eval_figure.savefig(eval_path, dpi=160)
        plt.close(eval_figure)
        generated_plots.append(eval_path.name)

    timeline_figure, timeline_ax = plt.subplots(figsize=(14, 1.8))
    for index, span in enumerate(stage_spans):
        stage_name = str(span["stage_name"])
        timeline_ax.barh([0], [int(span["num_episodes"])], left=int(span["start_episode"]), height=0.5, color=palette[stage_name], edgecolor="#4a4a4a")
        timeline_ax.text(
            int(span["start_episode"]) + int(span["num_episodes"]) / 2.0,
            0,
            stage_name,
            ha="center",
            va="center",
            fontsize=9,
        )
    timeline_ax.set_yticks([])
    timeline_ax.set_xlabel("Episode")
    timeline_ax.set_title("Curriculum Timeline")
    timeline_ax.set_xlim(min(episodes), max(episodes))
    timeline_ax.grid(axis="x", alpha=0.25)
    timeline_figure.tight_layout()
    timeline_path = output_dir / "stage_timeline.png"
    timeline_figure.savefig(timeline_path, dpi=160)
    plt.close(timeline_figure)
    generated_plots.append(timeline_path.name)

    _write_training_report(
        history_path=history_path,
        output_dir=output_dir,
        history=history,
        eval_history=eval_history,
        stage_spans=stage_spans,
        generated_plots=generated_plots,
    )
    return output_path
