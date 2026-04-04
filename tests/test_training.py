"""Integration-style tests for train/eval artifact generation."""

from __future__ import annotations

from pathlib import Path

from src.training import loops
from src.training.runner import evaluate_agent, train_agent
from src.utils.config import load_config
from src.visualization import load_jsonl, plot_training_history


def test_train_evaluate_and_plot_pipeline(tmp_path: Path) -> None:
    """Training, evaluation, and plotting should produce the expected artifacts."""
    config = load_config("configs/default.yaml")
    config.environment.max_episode_steps = 8
    config.environment.num_dynamic_obstacles = 2
    config.environment.seed = 3
    config.environment.teacher_config = {"enabled": True, "reward_scale": 0.0, "action_mix": 0.25}
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


def test_curriculum_training_stops_early_when_final_target_is_reached(tmp_path: Path, monkeypatch) -> None:
    """Curriculum runs should stop once the final stage target has been reached."""
    config = load_config("configs/default.yaml")
    config.environment.max_episode_steps = 6
    config.environment.auto_time_budget_steps_per_meter = 0.0
    config.environment.num_dynamic_obstacles = 2
    config.environment.seed = 11
    config.agent.rollout_episodes = 1
    config.agent.ppo_epochs = 1
    config.agent.mini_batch_size = 8
    config.training.num_episodes = 5
    config.training.save_interval = 1
    config.training.eval_interval = 1
    config.training.eval_episodes = 2
    config.training.results_dir = str(tmp_path / "results")
    config.training.log_dir = str(tmp_path / "logs")
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")
    config.evaluation.output_dir = str(tmp_path / "results" / "eval")
    config.evaluation.num_episodes = 2
    config.visualization.plot_dir = str(tmp_path / "results" / "plots")
    config.visualization.trajectory_dir = str(tmp_path / "results" / "trajectories")
    config.training.curriculum = [
        {
            "name": "fast-stop",
            "min_stage_episodes": 1,
            "required_consecutive_evals": 1,
            "advance_success_rate": 0.9,
            "advance_collision_rate": 0.0,
            "environment": {
                "max_episode_steps": 6,
                "auto_time_budget_steps_per_meter": 0.0,
                "teacher_config": {"enabled": False, "reward_scale": 0.0},
            },
        }
    ]

    def fake_evaluate_current_policy(**kwargs):
        return [], {"success_rate": 1.0, "collision_rate": 0.0, "avg_episode_return": 50.0}, None

    monkeypatch.setattr(loops, "_evaluate_current_policy", fake_evaluate_current_policy)

    summary = train_agent(config=config)
    assert summary["num_episodes"] < 5
    assert summary["stopped_early"] == 1.0
    assert summary["best_eval_success_rate"] == 1.0


def test_checkpoint_score_prefers_success_over_stage_index() -> None:
    """A successful earlier-stage checkpoint should beat a later-stage zero-success checkpoint."""
    successful_early = loops._checkpoint_score(
        {
            "success_rate": 0.5,
            "collision_rate": 0.0,
            "avg_episode_return": 10.0,
        },
        stage_index=0,
    )
    failed_later = loops._checkpoint_score(
        {
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "avg_episode_return": 100.0,
        },
        stage_index=2,
    )

    assert successful_early > failed_later
