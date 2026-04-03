#!/usr/bin/env python3
"""Train the dynamic-airspace graph PPO scaffold."""

from __future__ import annotations

import argparse

from _common import bootstrap_project

bootstrap_project()

from src.training.runner import train_agent
from src.utils.config import load_config, validate_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the dynamic UAV path-planning scaffold")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from")
    parser.add_argument("--episodes", type=int, help="Override training episode count")
    parser.add_argument("--seed", type=int, help="Override environment seed")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.episodes is not None:
        config.training.num_episodes = int(args.episodes)
    if args.seed is not None:
        config.environment.seed = int(args.seed)
    validate_config(config)

    summary = train_agent(config=config, resume=args.resume, num_episodes=args.episodes)
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
