#!/usr/bin/env python3
"""Train the dynamic-airspace graph PPO scaffold."""

from __future__ import annotations

import argparse
from datetime import timedelta
from typing import Dict

from _common import bootstrap_project

bootstrap_project()

from src.training.runner import train_agent
from src.utils.config import load_config, validate_config


def _format_seconds(value: float) -> str:
    """Format a duration in seconds for compact terminal progress output."""
    seconds = max(int(round(float(value))), 0)
    return str(timedelta(seconds=seconds))


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

        if event_type == "episode":
            episode = int(event["episode"])
            total_episodes = int(event["total_episodes"])
            if episode % print_every != 0 and episode != total_episodes:
                return
            metrics = dict(event["metrics"])
            rolling_summary = dict(event["rolling_summary"])
            percent = 100.0 * episode / max(total_episodes, 1)
            print(
                "train_progress "
                f"episode={episode}/{total_episodes} "
                f"progress={percent:.1f}% "
                f"stage={event['stage_name']} "
                f"elapsed={_format_seconds(float(event['elapsed_seconds']))} "
                f"eta={_format_seconds(float(event['eta_seconds']))} "
                f"return={float(metrics['episode_return']):.2f} "
                f"success={float(metrics['success']):.2f} "
                f"collision={float(metrics['collision']):.2f} "
                f"rolling_success={float(rolling_summary['success_rate']):.2f} "
                f"rolling_return={float(rolling_summary['avg_episode_return']):.2f}"
            )
            return

        if event_type == "evaluation":
            evaluation = dict(event["evaluation"])
            print(
                "eval_progress "
                f"episode={event['episode']}/{event['total_episodes']} "
                f"stage={event['stage_name']} "
                f"elapsed={_format_seconds(float(event['elapsed_seconds']))} "
                f"eta={_format_seconds(float(event['eta_seconds']))} "
                f"success_rate={float(evaluation['success_rate']):.3f} "
                f"collision_rate={float(evaluation['collision_rate']):.3f} "
                f"avg_return={float(evaluation['avg_episode_return']):.2f}"
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
            print(
                "training_finished "
                f"completed_episodes={event['completed_episodes']}/{event['total_episodes']} "
                f"elapsed={_format_seconds(float(event['elapsed_seconds']))}"
            )

    return emit


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the dynamic UAV path-planning scaffold")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from")
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

    summary = train_agent(
        config=config,
        resume=args.resume,
        num_episodes=args.episodes,
        progress_callback=_progress_printer(print_every),
    )
    print(f"training_episodes={summary['num_episodes']}")
    print(f"avg_episode_return={summary['avg_episode_return']:.3f}")
    print(f"success_rate={summary['success_rate']:.3f}")
    print(f"collision_rate={summary['collision_rate']:.3f}")
    print(f"avg_min_clearance={summary['avg_min_clearance']:.3f}")
    print(f"avg_control_effort={summary['avg_control_effort']:.3f}")
    print(f"best_eval_success_rate={summary['best_eval_success_rate']:.3f}")
    print(f"best_eval_collision_rate={summary['best_eval_collision_rate']:.3f}")
    print(f"best_eval_avg_episode_return={summary['best_eval_avg_episode_return']:.3f}")
    print(f"best_model={summary['best_model_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
