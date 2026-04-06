#!/usr/bin/env python3
"""Train the dynamic-airspace graph PPO scaffold."""

from __future__ import annotations

import argparse
import sys
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


def _format_name_list(values: object) -> str:
    """Format short config string lists for progress logs."""
    items = [str(item) for item in list(values or [])]
    return ",".join(items) if items else "none"


def _progress_printer(print_every: int):
    """Build a small terminal logger for training progress events."""
    def emit(event: Dict[str, object]) -> None:
        event_type = str(event.get("event", ""))
        if event_type == "start":
            print(
                "training_started "
                f"episodes={event['total_episodes']} "
                f"stage={event['stage_name']} "
                f"curriculum_stages={event['curriculum_stage_count']} "
                f"bc_enabled={int(float(event.get('bc_enabled', 0.0)))} "
                f"bc_stages={_format_name_list(event.get('bc_stages'))} "
                f"bc_teacher_gated={int(float(event.get('bc_teacher_gated_only', 0.0)))} "
                f"bc_gate_signal={event.get('bc_gate_signal', 'teacher_active')} "
                f"bc_demo_pretrain={int(float(event.get('bc_demo_pretrain_enabled', 0.0)))} "
                f"bc_demo_stages={_format_name_list(event.get('bc_demo_pretrain_stages'))} "
                f"stage_entry_reset={int(float(event.get('stage_entry_optimizer_reset_enabled', 0.0)))} "
                f"stage_entry_reset_stages={_format_name_list(event.get('stage_entry_optimizer_reset_stages'))}"
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
            bc_settings = dict(event.get("bc_settings", {}))
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
                f"bc_active={int(bool(bc_settings.get('active', False)))} "
                f"bc_coef={_format_metric(bc_settings.get('current_coef'), 4)} "
                f"bc_gated={int(bool(bc_settings.get('teacher_gated_only', False)))} "
                f"bc_gate_signal={bc_settings.get('gate_signal', 'teacher_active')} "
                f"policy_updated={int(bool(event.get('policy_updated', False)))} "
                f"actor_loss={_format_metric(update_metrics.get('actor_loss'), 4)} "
                f"critic_loss={_format_metric(update_metrics.get('critic_loss'), 4)} "
                f"entropy={_format_metric(update_metrics.get('entropy'), 4)} "
                f"bc_loss={_format_metric(update_metrics.get('bc_loss'), 4)} "
                f"bc_nonzero={int(float(update_metrics.get('bc_nonzero', 0.0)))} "
                f"bc_active_frac={_format_metric(update_metrics.get('bc_active_fraction'), 3)} "
                f"bc_active_samples={_format_metric(update_metrics.get('bc_active_samples'), 1)}"
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
                f"rollback={int(float(evaluation.get('stage_rollback_applied', 0.0)))} "
                f"plateau_recovery={int(float(evaluation.get('stage_plateau_recovery_applied', 0.0)))}"
            )
            return

        if event_type == "pretrain_start":
            print(
                "pretrain_start "
                f"stage={event['stage_name']} "
                f"train_episode_before={int(event.get('train_episode_before', 0))} "
                f"demo_episodes={int(event.get('episodes', 0))} "
                f"epochs={int(event.get('epochs', 0))} "
                f"batch_size={int(event.get('batch_size', 0))} "
                f"teacher_gated={int(float(event.get('teacher_gated_only', 0.0)))} "
                f"success_only={int(float(event.get('successful_episodes_only', 0.0)))} "
                f"gate_signal={event.get('gate_signal', 'teacher_active')} "
                f"eval_after={int(float(event.get('evaluate_after_pretrain', 0.0)))} "
                f"eval_episodes={int(event.get('eval_episodes', 0))} "
                f"post_std={_format_metric(event.get('post_pretrain_action_std'), 3)}"
            )
            return

        if event_type == "pretrain_collect_progress":
            print(
                "pretrain_collect "
                f"stage={event['stage_name']} "
                f"train_episode_before={int(event.get('train_episode_before', 0))} "
                f"demo_episode={int(event.get('demo_episode', 0))}/{int(event.get('demo_episodes', 0))} "
                f"dataset_steps={int(event.get('dataset_steps', 0))} "
                f"kept_steps={int(event.get('kept_steps', 0))} "
                f"source_success={_format_metric(event.get('source_success_rate'), 3)} "
                f"source_collision={_format_metric(event.get('source_collision_rate'), 3)}"
            )
            return

        if event_type == "pretrain_finish":
            summary = dict(event.get("summary", {}))
            print(
                "pretrain_finish "
                f"stage={event['stage_name']} "
                f"train_episode_before={int(event.get('train_episode_before', 0))} "
                f"episodes_kept={_format_metric(summary.get('num_episodes'), 0)} "
                f"source_episodes={_format_metric(summary.get('source_num_episodes'), 0)} "
                f"samples={_format_metric(summary.get('num_samples'), 0)} "
                f"dataset_steps={_format_metric(summary.get('dataset_steps'), 0)} "
                f"active_frac={_format_metric(summary.get('active_fraction'), 3)} "
                f"teacher_success={_format_metric(summary.get('success_rate'), 3)} "
                f"teacher_source_success={_format_metric(summary.get('source_success_rate'), 3)} "
                f"teacher_collision={_format_metric(summary.get('collision_rate'), 3)} "
                f"fallback={int(float(summary.get('success_only_fallback_used', 0.0)))} "
                f"action_std={_format_metric(summary.get('applied_action_std_mean'), 3)} "
                f"bc_pretrain_loss={_format_metric(summary.get('bc_pretrain_loss'), 4)}"
            )
            return

        if event_type == "pretrain_eval":
            evaluation = dict(event.get("evaluation", {}))
            print(
                "pretrain_eval "
                f"stage={event['stage_name']} "
                f"train_episode_before={int(event.get('train_episode_before', 0))} "
                f"stage_episode_before={int(event.get('stage_episode_before', 0))} "
                f"success_rate={_format_metric(evaluation.get('success_rate'), 3)} "
                f"collision_rate={_format_metric(evaluation.get('collision_rate'), 3)} "
                f"avg_return={_format_metric(evaluation.get('avg_episode_return'))} "
                f"avg_steps={_format_metric(evaluation.get('avg_steps'), 1)} "
                f"target_hit={int(float(evaluation.get('stage_target_hit', 0.0)))} "
                f"stage_best_updated={int(float(event.get('stage_best_updated', 0.0)))}"
            )
            return

        if event_type == "stage_entry_optimizer_reset":
            print(
                "stage_entry_optimizer_reset "
                f"stage={event['stage_name']} "
                f"initial_stage={int(float(event.get('is_initial_stage', 0.0)))} "
                f"train_episode_before={int(event.get('train_episode_before', 0))} "
                f"lr_multiplier={_format_metric(event.get('lr_multiplier'), 3)} "
                f"actor_lr={float(event.get('actor_lr', 0.0)):.6f} "
                f"critic_lr={float(event.get('critic_lr', 0.0)):.6f}"
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

        if event_type == "stage_plateau_recovery":
            print(
                "stage_plateau_recovery "
                f"episode={event['episode']} "
                f"stage={event['stage_name']} "
                f"recovery_count={int(event['recovery_count'])} "
                f"reset_stage_episode={int(float(event.get('reset_stage_episode', 0.0)))} "
                f"reran_demo_pretrain={int(float(event.get('reran_demo_pretrain', 0.0)))} "
                f"demo_samples={_format_metric(event.get('demo_samples'), 0)} "
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
                f"plateau_recoveries={int(summary.get('stage_plateau_recovery_count', 0))} "
                f"stopped_early={int(float(summary.get('stopped_early', 0.0)))}"
            )

    return emit


def _auto_plot_training_results(config) -> Path:
    """Generate training plots from the just-finished run artifacts."""
    history_path = Path(config.training.results_dir) / "train" / "history.jsonl"
    return plot_training_history(history_path=history_path, output_dir=config.visualization.plot_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the dynamic UAV path-planning scaffold")
    parser.add_argument("--config", type=str, default="configs/default_curriculum.yml", help="Configuration file")
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
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
        sys.stderr.reconfigure(line_buffering=True, write_through=True)
    except AttributeError:  # pragma: no cover - older Python fallback
        pass
    print_every = max(1, int(args.print_every or getattr(config.training, "print_interval", 10)))
    resume_overrides = {}
    if args.reset_optimizer_on_resume:
        resume_overrides["load_optimizer_state"] = False
    if args.restore_curriculum_progress:
        resume_overrides["restore_curriculum_progress"] = True

    run_name = str(getattr(config, "name", Path(args.config).stem))
    print(
        "training_bootstrap "
        f"config={args.config} "
        f"run_name={run_name} "
        f"episodes={int(config.training.num_episodes)} "
        f"seed={int(config.environment.seed)} "
        f"resume_arg={args.resume or 'none'}"
    )

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
    print(f"stage_entry_optimizer_reset_count={int(summary.get('stage_entry_optimizer_reset_count', 0))}")
    print(f"pretrain_run_count={int(summary.get('pretrain_run_count', 0))}")
    print(f"pretrain_total_samples={summary.get('pretrain_total_samples', 0.0):.1f}")
    print(f"stage_regression_event_count={int(summary.get('stage_regression_event_count', 0))}")
    print(f"stage_rollback_count={int(summary.get('stage_rollback_count', 0))}")
    print(f"stage_plateau_recovery_count={int(summary.get('stage_plateau_recovery_count', 0))}")
    if "resume_checkpoint_path" in summary:
        print(f"resume_checkpoint={summary['resume_checkpoint_path']}")
    print(f"best_model={summary['best_model_path']}")
    print(f"last_model={summary['last_model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
