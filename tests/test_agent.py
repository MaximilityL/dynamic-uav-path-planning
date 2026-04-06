"""Tests for PPO agent behavior."""

from __future__ import annotations

import math
from pathlib import Path

from src.agents import GraphPPOAgent
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


def test_agent_load_can_skip_optimizer_state(tmp_path: Path) -> None:
    """Checkpoint loading should support model-only resume with a fresh optimizer."""
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

    env = create_environment(config, gui=False, seed=21)
    try:
        trained_agent = create_agent(config, env)
        _, _ = run_episode(
            env=env,
            agent=trained_agent,
            episode_seed=22,
            deterministic=False,
            store_transition=True,
        )
        trained_agent.update()
        checkpoint_path = trained_agent.save(tmp_path / "checkpoint.pth")
        assert len(trained_agent.optimizer.state) > 0

        resumed_with_optimizer = create_agent(config, env)
        resumed_with_optimizer.load(checkpoint_path, load_optimizer_state=True)
        assert len(resumed_with_optimizer.optimizer.state) > 0

        resumed_without_optimizer = create_agent(config, env)
        resumed_without_optimizer.load(
            checkpoint_path,
            load_optimizer_state=False,
            reset_optimizer_if_skipped=True,
        )
        assert len(resumed_without_optimizer.optimizer.state) == 0
    finally:
        env.close()


def test_agent_load_can_adapt_legacy_global_feature_expansion(tmp_path: Path) -> None:
    """Loading older checkpoints should tolerate the repo's expanded global feature vector."""
    config = load_config("configs/default.yaml")
    config.environment.max_episode_steps = 8
    config.environment.num_dynamic_obstacles = 2

    env = create_environment(config, gui=False, seed=31)
    try:
        legacy_agent = GraphPPOAgent(
            encoder_type=config.agent.encoder_type,
            max_nodes=env.max_nodes,
            node_feature_dim=env.node_feature_dim,
            edge_feature_dim=env.edge_feature_dim,
            global_feature_dim=4,
            action_dim=env.action_dim,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
            hidden_dim=config.agent.hidden_dim,
            message_passing_steps=config.agent.message_passing_steps,
            lr_actor=config.agent.lr_actor,
            lr_critic=config.agent.lr_critic,
            gamma=config.agent.gamma,
            gae_lambda=config.agent.gae_lambda,
            clip_epsilon=config.agent.clip_epsilon,
            entropy_coef=config.agent.entropy_coef,
            value_coef=config.agent.value_coef,
            ppo_epochs=config.agent.ppo_epochs,
            mini_batch_size=config.agent.mini_batch_size,
            max_grad_norm=config.agent.max_grad_norm,
            action_std_init=config.agent.action_std_init,
            device="cpu",
        )
        checkpoint_path = legacy_agent.save(tmp_path / "legacy_checkpoint.pth")

        current_agent = create_agent(config, env)
        metadata = current_agent.load(checkpoint_path, load_optimizer_state=False, reset_optimizer_if_skipped=True)
    finally:
        env.close()

    loaded_weight = current_agent.model.state_dict()["encoder.output_projection.0.weight"]
    legacy_weight = legacy_agent.model.state_dict()["encoder.output_projection.0.weight"]
    assert metadata == {}
    assert loaded_weight.shape[1] == legacy_weight.shape[1] + 6
    assert loaded_weight[:, : legacy_weight.shape[1]].equal(legacy_weight)
