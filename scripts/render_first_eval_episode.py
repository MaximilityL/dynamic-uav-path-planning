#!/usr/bin/env python3
"""Render the first deterministic evaluation episode for a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import bootstrap_project

bootstrap_project()

from src.training.factories import create_agent, create_environment
from src.training.loops import _stage_config, run_episode
from src.utils.config import load_config, validate_config

EVALUATION_SEED_OFFSET = 100_000


def _resolve_stage(config, stage_name: str | None, stage_index: int | None):
    curriculum = list(config.training.curriculum or [])
    if stage_name is not None and stage_index is not None:
        raise ValueError("Use either --stage-name or --stage-index, not both.")
    if not curriculum or (stage_name is None and stage_index is None):
        return config, "main"
    if stage_name is not None:
        for index, stage in enumerate(curriculum):
            if str(stage.get("name", f"stage_{index + 1}")) == stage_name:
                return _stage_config(config, stage), stage_name
        raise ValueError(f"Unknown stage name '{stage_name}'.")
    if stage_index is None or stage_index < 0 or stage_index >= len(curriculum):
        raise ValueError(f"Stage index must be between 0 and {len(curriculum) - 1}.")
    stage = curriculum[stage_index]
    resolved_name = str(stage.get("name", f"stage_{stage_index + 1}"))
    return _stage_config(config, stage), resolved_name


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the first deterministic evaluation episode for a checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Configuration file")
    parser.add_argument("--model", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--stage-name", type=str, help="Optional curriculum stage name to apply before rendering")
    parser.add_argument("--stage-index", type=int, help="Optional curriculum stage index to apply before rendering")
    parser.add_argument("--seed", type=int, help="Override environment seed before evaluation seeding")
    parser.add_argument("--no-render", action="store_true", help="Run the same episode without opening the GUI")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.environment.seed = int(args.seed)
    config, resolved_stage_name = _resolve_stage(config, args.stage_name, args.stage_index)
    validate_config(config)

    env = create_environment(
        config,
        gui=not args.no_render,
        seed=config.environment.seed + EVALUATION_SEED_OFFSET,
    )
    try:
        agent = create_agent(config, env)
        agent.load(args.model)
        metrics, _ = run_episode(
            env=env,
            agent=agent,
            episode_seed=config.environment.seed + EVALUATION_SEED_OFFSET,
            deterministic=True,
            store_transition=False,
        )
    finally:
        env.close()

    print(f"stage={resolved_stage_name}")
    print(f"model={Path(args.model)}")
    print(f"success={float(metrics['success']):.3f}")
    print(f"collision={float(metrics['collision']):.3f}")
    print(f"episode_return={float(metrics['episode_return']):.3f}")
    print(f"steps={int(metrics['steps'])}")
    print(f"min_clearance={float(metrics['min_clearance']):.3f}")
    print(f"path_length={float(metrics['path_length']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
