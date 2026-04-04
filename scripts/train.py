#!/usr/bin/env python3
"""Train the dynamic-airspace graph PPO scaffold."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Dict

from _common import bootstrap_project

bootstrap_project()

from src.training.runner import train_agent
from src.utils.config import load_config, validate_config
from src.visualization import plot_training_history


def _format_seconds(value: float) -> str:
    """Format a duration in seconds for compact terminal progress output."""
    seconds = max(int(round(float(value))), 0)
    return str(timedelta(seconds=seconds))


def _format_metric(value: object, digits: int = 2) -> str:
    """Format scalar metrics consistently for terminal output."""
    if value is None:
        return "na"
    return f"{float(value):.{digits}f}"


def _progress_printer(print_every: int):
    """Build a small terminal logger for training progress events."""
    def emit(event: Dict[str, object]) -> None:
        event_type = str(event.get("event", ""))
        if event_type == "start":
            print(
                "training_started "
                f"episodes={event['total_episodes']} "
                f"stage={event['stage_name']} "
                f"curriculum_stages={event['curriculum_stage_count']}"
            )
            return

        if event_type == "resume":
            print(
                "training_resumed "
                f"checkpoint={event['checkpoint_path']} "
                f"stage={event['restored_stage_name']} "
                f"load_optimizer_state={int(float(event['load_optimizer_state']))} "
                f"restore_curriculum_progress={int(float(event['restore_curriculum_progress']))}"
            )
            return

        if event_type == "episode":
            episode = int(event["episode"])
            total_episodes = int(event["total_episodes"])
            if episode % print_every != 0 and episode != total_episodes:
                return
            metrics = dict(event["metrics"])
            rolling_summary = dict(event["rolling_summary"])
            update_metrics = dict(event.get("update_metrics", {}))
            percent = 100.0 * episode / max(total_episodes, 1)
            print(
                "train_progress "
                f"episode={episode}/{total_episodes} "
                f"progress={percent:.1f}% "
                f"stage={event['stage_name']} "
                f"stage_episode={int(event.get('stage_episode', 0))} "
                f"elapsed={_format_seconds(float(event['elapsed_seconds']))} "
                f"eta={_format_seconds(float(event['eta_seconds']))} "
                f"return={_format_metric(metrics.get('episode_return'))} "
                f"success={_format_metric(metrics.get('success'))} "
                f"collision={_format_metric(metrics.get('collision'))} "
                f"steps={_format_metric(metrics.get('steps'), 0)} "
                f"path={_format_metric(metrics.get('path_length'))} "
                f"goal_dist={_format_metric(metrics.get('start_to_goal_distance'))} "
                f"clearance={_format_metric(metrics.get('min_clearance'))} "
                f"effort={_format_metric(metrics.get('control_effort'))} "
                f"rolling_success={_format_metric(rolling_summary.get('success_rate'))} "
                f"rolling_collision={_format_metric(rolling_summary.get('collision_rate'))} "
                f"rolling_return={_format_metric(rolling_summary.get('avg_episode_return'))} "
                f"policy_updated={int(bool(event.get('policy_updated', False)))} "
                f"actor_loss={_format_metric(update_metrics.get('actor_loss'), 4)} "
                f"critic_loss={_format_metric(update_metrics.get('critic_loss'), 4)} "
                f"entropy={_format_metric(update_metrics.get('entropy'), 4)}"
            )
            return

        if event_type == "evaluation":
            evaluation = dict(event["evaluation"])
            print(
                "eval_progress "
                f"episode={event['episode']}/{event['total_episodes']} "
                f"stage={event['stage_name']} "
                f"stage_episode={int(event.get('stage_episode', 0))} "
                f"eval_index={int(evaluation.get('stage_eval_index', 0))} "
                f"elapsed={_format_seconds(float(event['elapsed_seconds']))} "
                f"eta={_format_seconds(float(event['eta_seconds']))} "
                f"success_rate={_format_metric(evaluation.get('success_rate'), 3)} "
                f"collision_rate={_format_metric(evaluation.get('collision_rate'), 3)} "
                f"avg_return={_format_metric(evaluation.get('avg_episode_return'))} "
                f"avg_steps={_format_metric(evaluation.get('avg_steps'), 1)} "
                f"avg_path={_format_metric(evaluation.get('avg_path_length'))} "
                f"avg_time_to_goal={_format_metric(evaluation.get('avg_time_to_goal'))} "
                f"avg_clearance={_format_metric(evaluation.get('avg_min_clearance'))} "
                f"best_stage_success={_format_metric(evaluation.get('stage_best_success_rate'), 3)} "
                f"target_hit={int(float(evaluation.get('stage_target_hit', 0.0)))} "
                f"streak={int(evaluation.get('stage_success_streak', 0))} "
                f"bad_streak={int(evaluation.get('stage_bad_eval_streak', 0))} "
                f"rollback={int(float(evaluation.get('stage_rollback_applied', 0.0)))}"
            )
            return

        if event_type == "stage_regression_detected":
            print(
                "stage_regression "
                f"episode={event['episode']} "
                f"stage={event['stage_name']} "
                f"current_success={float(event['current_success_rate']):.3f} "
                f"best_success={float(event['best_success_rate']):.3f} "
                f"bad_evals={int(event['bad_eval_streak'])}"
            )
            return

        if event_type == "stage_rollback":
            print(
                "stage_rollback "
                f"episode={event['episode']} "
                f"stage={event['stage_name']} "
                f"rollback_count={int(event['rollback_count'])} "
                f"actor_lr={float(event['actor_lr']):.6f} "
                f"critic_lr={float(event['critic_lr']):.6f}"
            )
            return

        if event_type == "stage_transition":
            print(
                "stage_transition "
                f"episode={event['episode']}/{event['total_episodes']} "
                f"from={event['from_stage_name']} "
                f"to={event['to_stage_name']} "
                f"elapsed={_format_seconds(float(event['elapsed_seconds']))}"
            )
            return

        if event_type == "finish":
            summary = dict(event.get("summary", {}))
            print(
                "training_finished "
                f"completed_episodes={event['completed_episodes']}/{event['total_episodes']} "
                f"elapsed={_format_seconds(float(event['elapsed_seconds']))} "
                f"completed_stage={summary.get('completed_stage_name', 'na')} "
                f"best_eval_success={_format_metric(summary.get('best_eval_success_rate'), 3)} "
                f"best_eval_collision={_format_metric(summary.get('best_eval_collision_rate'), 3)} "
                f"rollbacks={int(summary.get('stage_rollback_count', 0))} "
                f"stopped_early={int(float(summary.get('stopped_early', 0.0)))}"
            )

    return emit


def _auto_plot_training_results(config) -> Path:
    """Generate training plots from the just-finished run artifacts."""
    history_path = Path(config.training.results_dir) / "train" / "history.jsonl"
    return plot_training_history(history_path=history_path, output_dir=config.visualization.plot_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the dynamic UAV path-planning scaffold")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from")
    parser.add_argument(
        "--reset-optimizer-on-resume",
        action="store_true",
        help="Resume model weights but start with a fresh optimizer state",
    )
    parser.add_argument(
        "--restore-curriculum-progress",
        action="store_true",
        help="Restore the saved curriculum stage from the checkpoint metadata",
    )
    parser.add_argument("--episodes", type=int, help="Override training episode count")
    parser.add_argument("--seed", type=int, help="Override environment seed")
    parser.add_argument("--print-every", type=int, help="Print training progress every N episodes")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.episodes is not None:
        config.training.num_episodes = int(args.episodes)
    if args.seed is not None:
        config.environment.seed = int(args.seed)
    validate_config(config)
    print_every = max(1, int(args.print_every or getattr(config.training, "print_interval", 10)))
    resume_overrides = {}
    if args.reset_optimizer_on_resume:
        resume_overrides["load_optimizer_state"] = False
    if args.restore_curriculum_progress:
        resume_overrides["restore_curriculum_progress"] = True

    summary = train_agent(
        config=config,
        resume=args.resume,
        num_episodes=args.episodes,
        progress_callback=_progress_printer(print_every),
        resume_overrides=resume_overrides or None,
    )
    try:
        plot_path = _auto_plot_training_results(config)
        print(f"plot={plot_path}")
    except Exception as exc:  # pragma: no cover - defensive CLI fallback
        print(f"plot_generation_failed={exc}")
    print(f"training_episodes={summary['num_episodes']}")
    print(f"avg_episode_return={summary['avg_episode_return']:.3f}")
    print(f"success_rate={summary['success_rate']:.3f}")
    print(f"collision_rate={summary['collision_rate']:.3f}")
    print(f"avg_path_length={summary['avg_path_length']:.3f}")
    print(f"avg_steps={summary['avg_steps']:.3f}")
    print(f"avg_episode_duration={summary['avg_episode_duration']:.3f}")
    print(f"avg_time_to_goal={summary['avg_time_to_goal']:.3f}")
    print(f"avg_min_clearance={summary['avg_min_clearance']:.3f}")
    print(f"avg_control_effort={summary['avg_control_effort']:.3f}")
    print(f"avg_path_efficiency={summary['avg_path_efficiency']:.3f}")
    print(f"best_eval_success_rate={summary['best_eval_success_rate']:.3f}")
    print(f"best_eval_collision_rate={summary['best_eval_collision_rate']:.3f}")
    print(f"best_eval_avg_episode_return={summary['best_eval_avg_episode_return']:.3f}")
    print(f"completed_stage={summary.get('completed_stage_name', 'main')}")
    print(f"stage_regression_event_count={int(summary.get('stage_regression_event_count', 0))}")
    print(f"stage_rollback_count={int(summary.get('stage_rollback_count', 0))}")
    if "resume_checkpoint_path" in summary:
        print(f"resume_checkpoint={summary['resume_checkpoint_path']}")
    print(f"best_model={summary['best_model_path']}")
    print(f"last_model={summary['last_model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
