#!/usr/bin/env python3
"""Evaluate a saved graph PPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from _common import bootstrap_project

bootstrap_project()

from src.training.loops import _stage_config
from src.training.runner import evaluate_agent
from src.utils.config import load_config, validate_config


def _format_metric(value: object, digits: int = 3) -> str:
    """Format scalar evaluation metrics consistently for terminal output."""
    if value is None:
        return "na"
    return f"{float(value):.{digits}f}"


def _resolve_stage(config, stage_name: Optional[str], stage_index: Optional[int]):
    """Resolve an optional curriculum stage selection for evaluation."""
    training = getattr(config, "training", None)
    curriculum = list(getattr(training, "curriculum", []) or [])
    if not curriculum:
        return None

    if stage_name is not None and stage_index is not None:
        raise ValueError("Pass only one of --stage-name or --stage-index.")

    if stage_name is not None:
        for index, stage in enumerate(curriculum):
            if str(stage.get("name", f"stage_{index + 1}")) == stage_name:
                return stage
        available = ", ".join(str(stage.get("name", f"stage_{idx + 1}")) for idx, stage in enumerate(curriculum))
        raise ValueError(f"Unknown stage name '{stage_name}'. Available stages: {available}")

    if stage_index is not None:
        if stage_index < 0 or stage_index >= len(curriculum):
            raise ValueError(f"--stage-index must be between 0 and {len(curriculum) - 1}")
        return curriculum[stage_index]

    # When evaluating a curriculum config, the most useful default is the target/final stage.
    return curriculum[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a graph PPO checkpoint")
    parser.add_argument("--config", type=str, default="configs/default_curriculum.yml", help="Configuration file")
    parser.add_argument("--model", type=str, required=True, help="Checkpoint to evaluate")
    parser.add_argument("--episodes", type=int, help="Override evaluation episode count")
    parser.add_argument("--render", action="store_true", help="Run evaluation with the PyBullet GUI")
    parser.add_argument("--output", type=str, help="Override evaluation output directory")
    parser.add_argument("--seed", type=int, help="Override evaluation seed")
    parser.add_argument("--stage-name", type=str, help="Curriculum stage name to evaluate")
    parser.add_argument("--stage-index", type=int, help="Curriculum stage index to evaluate (0-based)")
    args = parser.parse_args()

    base_config = load_config(args.config)
    stage = _resolve_stage(base_config, args.stage_name, args.stage_index)
    config = _stage_config(base_config, stage) if stage is not None else base_config
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
    if stage is not None:
        print(f"stage={stage.get('name', 'stage')}")
    print(f"model={args.model}")
    print(f"episodes={summary['num_episodes']}")
    print(f"render={int(bool(config.evaluation.render))}")
    print(f"deterministic={int(bool(config.evaluation.deterministic))}")
    print(f"output_dir={config.evaluation.output_dir}")
    print(f"success_rate={summary['success_rate']:.3f}")
    print(f"collision_rate={summary['collision_rate']:.3f}")
    print(f"avg_episode_return={summary['avg_episode_return']:.3f}")
    print(f"avg_path_length={summary['avg_path_length']:.3f}")
    print(f"avg_steps={_format_metric(summary.get('avg_steps'))}")
    print(f"avg_episode_duration={_format_metric(summary.get('avg_episode_duration'))}")
    print(f"avg_time_to_goal={summary['avg_time_to_goal']:.3f}")
    print(f"avg_min_clearance={summary['avg_min_clearance']:.3f}")
    print(f"avg_control_effort={summary['avg_control_effort']:.3f}")
    print(f"avg_path_efficiency={_format_metric(summary.get('avg_path_efficiency'))}")
    print(f"best_episode_return={_format_metric(summary.get('best_episode_return'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
