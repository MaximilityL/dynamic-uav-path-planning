"""Integration-style tests for train/eval artifact generation."""

from __future__ import annotations

from pathlib import Path

from src.training.runner import evaluate_agent, train_agent
from src.utils.config import load_config
from src.visualization import load_jsonl, plot_training_history


def test_train_evaluate_and_plot_pipeline(tmp_path: Path) -> None:
    """Training, evaluation, and plotting should produce the expected artifacts."""
    config = load_config("configs/default.yaml")
    config.environment.max_episode_steps = 8
    config.environment.num_dynamic_obstacles = 2
    config.environment.seed = 3
    config.agent.rollout_episodes = 1
    config.agent.ppo_epochs = 1
    config.agent.mini_batch_size = 8
    config.training.num_episodes = 3
    config.training.save_interval = 1
    config.training.eval_interval = 2
    config.training.eval_episodes = 2
    config.training.results_dir = str(tmp_path / "results")
    config.training.log_dir = str(tmp_path / "logs")
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")
    config.evaluation.output_dir = str(tmp_path / "results" / "eval")
    config.evaluation.num_episodes = 2
    config.visualization.plot_dir = str(tmp_path / "results" / "plots")
    config.visualization.trajectory_dir = str(tmp_path / "results" / "trajectories")

    summary = train_agent(config=config)
    assert summary["num_episodes"] == 3
    assert summary["best_eval_success_rate"] >= 0.0

    best_model_path = Path(summary["best_model_path"])
    assert best_model_path.exists()
    eval_history = load_jsonl(tmp_path / "results" / "train" / "eval_history.jsonl")
    assert len(eval_history) == 2

    evaluation_summary = evaluate_agent(
        config=config,
        model_path=str(best_model_path),
        num_episodes=2,
        render=False,
        deterministic=True,
        save_outputs=True,
    )
    assert evaluation_summary["num_episodes"] == 2
    assert "avg_min_clearance" in evaluation_summary

    plot_path = plot_training_history(
        history_path=tmp_path / "results" / "train" / "history.jsonl",
        output_dir=tmp_path / "results" / "plots",
    )
    assert Path(plot_path).exists()
