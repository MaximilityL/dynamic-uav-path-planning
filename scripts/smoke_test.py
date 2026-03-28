#!/usr/bin/env python3
"""Smoke test for environment creation, rollout collection, and one PPO update."""

from __future__ import annotations

import argparse
import math

import numpy as np
from _common import bootstrap_project

bootstrap_project()

from src.training.runner import create_agent, create_environment, run_episode, set_global_seeds
from src.utils.config import load_config, validate_config


def assert_finite(name: str, value: float) -> None:
    """Fail fast on NaN or inf metrics."""
    if not math.isfinite(float(value)):
        raise RuntimeError(f"{name} is not finite: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a dynamic-airspace smoke test")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    config.environment.max_episode_steps = 20
    config.environment.num_dynamic_obstacles = min(config.environment.num_dynamic_obstacles, 3)
    config.agent.rollout_episodes = 1
    config.agent.ppo_epochs = 1
    config.agent.mini_batch_size = 8
    validate_config(config)
    set_global_seeds(config.environment.seed)

    env = create_environment(config, gui=False, seed=config.environment.seed)
    try:
        obs, info = env.reset(seed=config.environment.seed)
        print(f"node_features_shape={obs['node_features'].shape}")
        print(f"adjacency_shape={obs['adjacency'].shape}")
        print(f"action_shape={env.action_space.shape}")

        for step in range(3):
            random_action = env.action_space.sample().astype(np.float32)
            obs, reward, terminated, truncated, info = env.step(random_action)
            print(
                f"random_step={step + 1} reward={reward:.3f} "
                f"distance_to_goal={info['distance_to_goal']:.3f} "
                f"min_obstacle_distance={info['min_obstacle_distance']:.3f}"
            )
            if terminated or truncated:
                break
    finally:
        env.close()

    env = create_environment(config, gui=False, seed=config.environment.seed + 1)
    agent = create_agent(config, env)
    try:
        episode_metrics, _ = run_episode(
            env=env,
            agent=agent,
            episode_seed=config.environment.seed + 1,
            deterministic=False,
            store_transition=True,
        )
        update_metrics = agent.update()
    finally:
        env.close()

    print(f"episode_return={episode_metrics['episode_return']:.3f}")
    print(f"success={episode_metrics['success']:.3f}")
    print(f"collision={episode_metrics['collision']:.3f}")
    print(f"actor_loss={update_metrics['actor_loss']:.6f}")
    print(f"critic_loss={update_metrics['critic_loss']:.6f}")
    print(f"entropy={update_metrics['entropy']:.6f}")

    assert_finite("episode_return", episode_metrics["episode_return"])
    assert_finite("actor_loss", update_metrics["actor_loss"])
    assert_finite("critic_loss", update_metrics["critic_loss"])
    assert_finite("entropy", update_metrics["entropy"])
    print("smoke_test=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
