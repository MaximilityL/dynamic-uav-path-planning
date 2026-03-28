"""Tests for PPO agent behavior."""

from __future__ import annotations

import math

from src.training.runner import create_agent, create_environment, run_episode
from src.utils.config import load_config


def test_agent_select_action_and_update(tmp_path) -> None:
    """The agent should produce actions and complete a PPO update."""
    config = load_config("configs/default.yaml")
    config.environment.max_episode_steps = 8
    config.environment.num_dynamic_obstacles = 2
    config.agent.rollout_episodes = 1
    config.agent.ppo_epochs = 1
    config.agent.mini_batch_size = 8
    config.training.results_dir = str(tmp_path / "results")
    config.training.log_dir = str(tmp_path / "logs")
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")
    config.visualization.plot_dir = str(tmp_path / "results" / "plots")
    config.visualization.trajectory_dir = str(tmp_path / "results" / "trajectories")
    config.evaluation.output_dir = str(tmp_path / "results" / "eval")

    env = create_environment(config, gui=False, seed=11)
    agent = create_agent(config, env)
    try:
        observation, _ = env.reset(seed=11)
        action, info = agent.select_action(observation)
        assert action.shape == env.action_space.shape
        assert set(info) == {"log_prob", "value"}

        _, _ = run_episode(
            env=env,
            agent=agent,
            episode_seed=12,
            deterministic=False,
            store_transition=True,
        )
        update_metrics = agent.update()
    finally:
        env.close()

    assert math.isfinite(update_metrics["actor_loss"])
    assert math.isfinite(update_metrics["critic_loss"])
    assert math.isfinite(update_metrics["entropy"])
