#!/usr/bin/env python3
"""Evaluate a saved graph PPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import bootstrap_project

bootstrap_project()

from src.training.runner import evaluate_agent
from src.utils.config import load_config, validate_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a graph PPO checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--model", type=str, required=True, help="Checkpoint to evaluate")
    parser.add_argument("--episodes", type=int, help="Override evaluation episode count")
    parser.add_argument("--render", action="store_true", help="Run evaluation with the PyBullet GUI")
    parser.add_argument("--output", type=str, help="Override evaluation output directory")
    parser.add_argument("--seed", type=int, help="Override evaluation seed")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.episodes is not None:
        config.evaluation.num_episodes = int(args.episodes)
    if args.render:
        config.evaluation.render = True
    if args.seed is not None:
        config.environment.seed = int(args.seed)
    if args.output:
        config.evaluation.output_dir = str(Path(args.output))
    validate_config(config)

    summary = evaluate_agent(
        config=config,
        model_path=args.model,
        num_episodes=args.episodes,
        render=config.evaluation.render,
        deterministic=config.evaluation.deterministic,
        save_outputs=True,
    )
    print(f"success_rate={summary['success_rate']:.3f}")
    print(f"collision_rate={summary['collision_rate']:.3f}")
    print(f"avg_episode_return={summary['avg_episode_return']:.3f}")
    print(f"avg_path_length={summary['avg_path_length']:.3f}")
    print(f"avg_time_to_goal={summary['avg_time_to_goal']:.3f}")
    print(f"avg_min_clearance={summary['avg_min_clearance']:.3f}")
    print(f"avg_control_effort={summary['avg_control_effort']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
