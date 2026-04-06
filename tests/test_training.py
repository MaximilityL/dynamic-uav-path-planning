"""Integration-style tests for train/eval artifact generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.environments.teacher import heuristic_teacher_guidance
from src.training import loops
from src.training.runner import create_agent, create_environment, evaluate_agent, train_agent
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


def test_checkpoint_score_prefers_smaller_goal_distance_when_success_is_tied() -> None:
    """When success and collisions tie, the closer-to-goal checkpoint should win."""
    closer_timeout = loops._checkpoint_score(
        {
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "avg_distance_to_goal": 0.8,
            "avg_episode_return": -10.0,
        },
        stage_index=3,
    )
    farther_timeout = loops._checkpoint_score(
        {
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "avg_distance_to_goal": 1.6,
            "avg_episode_return": 20.0,
        },
        stage_index=3,
    )

    assert closer_timeout > farther_timeout


def test_checkpoint_score_prefers_rejoined_route_when_goal_distance_is_tied() -> None:
    """When distance ties, the checkpoint that stays closer to the route line should win."""
    better_rejoin = loops._checkpoint_score(
        {
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "avg_distance_to_goal": 1.2,
            "avg_route_line_lateral_error": 0.3,
            "avg_episode_return": -12.0,
        },
        stage_index=3,
    )
    wider_timeout = loops._checkpoint_score(
        {
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "avg_distance_to_goal": 1.2,
            "avg_route_line_lateral_error": 0.9,
            "avg_episode_return": 10.0,
        },
        stage_index=3,
    )

    assert better_rejoin > wider_timeout


def test_best_episode_score_prefers_success_then_goal_distance() -> None:
    """Showcase trajectory selection should prefer success, then closest approach."""
    far_success = loops._best_episode_score(
        {
            "success": 1.0,
            "collision": 0.0,
            "distance_to_goal": 0.3,
            "episode_return": 5.0,
        }
    )
    close_failure = loops._best_episode_score(
        {
            "success": 0.0,
            "collision": 0.0,
            "distance_to_goal": 0.05,
            "episode_return": 50.0,
        }
    )
    farther_failure = loops._best_episode_score(
        {
            "success": 0.0,
            "collision": 0.0,
            "distance_to_goal": 0.8,
            "episode_return": 100.0,
        }
    )

    assert far_success > close_failure
    assert close_failure > farther_failure


def test_stage_regression_protection_rolls_back_to_stage_best(tmp_path: Path, monkeypatch) -> None:
    """A stage that regresses after being solved should restore its own best checkpoint."""
    config = load_config("configs/default.yaml")
    config.environment.max_episode_steps = 6
    config.environment.auto_time_budget_steps_per_meter = 0.0
    config.environment.num_dynamic_obstacles = 2
    config.environment.seed = 5
    config.agent.rollout_episodes = 1
    config.agent.ppo_epochs = 1
    config.agent.mini_batch_size = 8
    config.training.num_episodes = 3
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
            "name": "bridge_crossing_easy",
            "min_stage_episodes": 99,
            "required_consecutive_evals": 2,
            "advance_success_rate": 1.1,
            "advance_collision_rate": 0.0,
            "environment": {
                "max_episode_steps": 6,
                "auto_time_budget_steps_per_meter": 0.0,
            },
        }
    ]
    config.training.stage_regression_protection = {
        "enabled": True,
        "activate_after_success_rate": 0.7,
        "absolute_drop_threshold": 0.25,
        "relative_drop_fraction": 0.4,
        "consecutive_bad_evals": 2,
        "rollback_on_regression": True,
        "rollback_max_per_stage": 1,
        "lr_multiplier_after_rollback": 0.5,
    }

    eval_summaries = iter(
        [
            {"success_rate": 0.8, "collision_rate": 0.0, "avg_episode_return": 50.0},
            {"success_rate": 0.2, "collision_rate": 0.0, "avg_episode_return": 5.0},
            {"success_rate": 0.1, "collision_rate": 0.0, "avg_episode_return": 0.0},
        ]
    )

    def fake_evaluate_current_policy(**kwargs):
        summary = next(eval_summaries)
        summary = dict(summary)
        summary.setdefault("avg_path_length", 0.0)
        summary.setdefault("avg_episode_duration", 0.0)
        summary.setdefault("avg_time_to_goal", 0.0)
        summary.setdefault("avg_min_obstacle_distance", 0.0)
        summary.setdefault("avg_min_clearance", 0.0)
        summary.setdefault("avg_control_effort", 0.0)
        summary.setdefault("avg_path_efficiency", 0.0)
        summary.setdefault("avg_steps", 0.0)
        summary.setdefault("best_episode_return", summary["avg_episode_return"])
        return [], summary, None

    monkeypatch.setattr(loops, "_evaluate_current_policy", fake_evaluate_current_policy)

    summary = train_agent(config=config)
    assert summary["stage_regression_event_count"] == 1
    assert summary["stage_rollback_count"] == 1

    stage_best_path = tmp_path / "checkpoints" / "stages" / "bridge_crossing_easy" / "best_model.pth"
    assert stage_best_path.exists()

    regression_events = load_jsonl(tmp_path / "results" / "train" / "stage_regression_events.jsonl")
    assert [event["event"] for event in regression_events] == ["stage_regression_detected", "stage_rollback"]

    stage_best_payload = json.loads((tmp_path / "results" / "train" / "stage_best_evaluations.json").read_text())
    assert stage_best_payload["bridge_crossing_easy"]["best_success_rate"] == 0.8


def test_stage_regression_signal_tolerates_float_rounding_at_activation_threshold() -> None:
    """Regression protection should still arm when float rounding lands just under the threshold."""
    signal = loops._stage_regression_signal(
        eval_summary={"success_rate": 0.1},
        stage_state={"best_eval": {"success_rate": 0.699999988079071}},
        protection={
            "enabled": True,
            "activate_after_success_rate": 0.7,
            "absolute_drop_threshold": 0.25,
            "relative_drop_fraction": 0.4,
        },
    )

    assert signal["active"] == 1.0
    assert signal["bad"] == 1.0


def test_summarize_episodes_tracks_distance_to_goal() -> None:
    """Episode summaries should keep final goal distance for checkpoint selection."""
    summary = loops.summarize_episodes(
        [
            {
                "episode_return": 1.0,
                "success": 0.0,
                "collision": 0.0,
                "distance_to_goal": 1.4,
                "route_line_lateral_error": 0.8,
                "min_obstacle_distance": 1.0,
                "min_clearance": 0.5,
                "path_length": 2.0,
                "steps": 10.0,
                "episode_duration": 1.0,
                "control_effort": 0.1,
            },
            {
                "episode_return": 2.0,
                "success": 1.0,
                "collision": 0.0,
                "distance_to_goal": 0.2,
                "route_line_lateral_error": 0.1,
                "min_obstacle_distance": 1.2,
                "min_clearance": 0.7,
                "path_length": 1.6,
                "start_to_goal_distance": 1.6,
                "steps": 8.0,
                "episode_duration": 0.8,
                "control_effort": 0.1,
                "time_to_goal": 0.8,
            },
        ]
    )

    assert summary["avg_distance_to_goal"] == pytest.approx(0.8)
    assert summary["best_distance_to_goal"] == pytest.approx(0.2)
    assert summary["avg_route_line_lateral_error"] == pytest.approx(0.45)
    assert summary["best_route_line_lateral_error"] == pytest.approx(0.1)


def test_heuristic_teacher_guidance_rejoins_route_after_clearance() -> None:
    """Teacher guidance should turn back toward the route line after the blocker has been cleared."""
    guidance = heuristic_teacher_guidance(
        drone_position=np.asarray([2.2, 1.0, 1.0], dtype=np.float32),
        goal_position=np.asarray([4.0, 0.0, 1.0], dtype=np.float32),
        obstacle_positions=np.asarray([[1.0, -0.2, 1.0]], dtype=np.float32),
        action_low=np.asarray([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        route_start_position=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        teacher_config={
            "lateral_avoidance_gain": 1.2,
            "lateral_avoidance_radius": 0.8,
            "forward_lookahead": 1.5,
            "rejoin_gain": 1.8,
            "rejoin_clearance_threshold": 0.8,
            "rejoin_progress_ratio_threshold": 0.5,
            "rejoin_min_lateral_error": 0.1,
        },
    )

    assert guidance["rejoin_active"] is True
    assert guidance["route_line_lateral_error"] == pytest.approx(1.0)
    assert guidance["action"][1] < 0.0


def test_stage_plateau_recovery_triggers_on_zero_success_stagnation(tmp_path: Path, monkeypatch) -> None:
    """Plateau recovery should fire after repeated non-improving zero-success evaluations."""
    config = load_config("configs/default.yaml")
    config.environment.max_episode_steps = 6
    config.environment.auto_time_budget_steps_per_meter = 0.0
    config.environment.num_dynamic_obstacles = 2
    config.environment.seed = 19
    config.agent.rollout_episodes = 1
    config.agent.ppo_epochs = 1
    config.agent.mini_batch_size = 8
    config.training.num_episodes = 3
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
            "name": "target_plateau",
            "min_stage_episodes": 99,
            "required_consecutive_evals": 2,
            "advance_success_rate": 1.1,
            "advance_collision_rate": 0.0,
            "environment": {
                "max_episode_steps": 6,
                "auto_time_budget_steps_per_meter": 0.0,
            },
        }
    ]
    config.training.stage_regression_protection = {
        "enabled": True,
        "activate_after_success_rate": 0.7,
        "absolute_drop_threshold": 0.25,
        "relative_drop_fraction": 0.4,
        "consecutive_bad_evals": 2,
        "rollback_on_regression": False,
        "rollback_max_per_stage": 0,
        "lr_multiplier_after_rollback": 1.0,
        "plateau_recovery_enabled": True,
        "plateau_bad_eval_streak": 2,
        "plateau_max_per_stage": 1,
        "lr_multiplier_after_plateau_recovery": 1.0,
        "reset_stage_episode_on_plateau_recovery": False,
        "rerun_stage_demo_pretrain": False,
    }

    def fake_evaluate_current_policy(**kwargs):
        summary = {
            "success_rate": 0.0,
            "collision_rate": 0.0,
            "avg_episode_return": -10.0,
            "avg_path_length": 0.0,
            "avg_episode_duration": 0.0,
            "avg_time_to_goal": 0.0,
            "avg_min_obstacle_distance": 1.0,
            "avg_min_clearance": 0.5,
            "avg_control_effort": 0.0,
            "avg_path_efficiency": 0.0,
            "avg_steps": 6.0,
            "best_episode_return": -10.0,
        }
        return [], summary, None

    monkeypatch.setattr(loops, "_evaluate_current_policy", fake_evaluate_current_policy)

    summary = train_agent(config=config)
    assert summary["stage_plateau_recovery_count"] == 1

    regression_events = load_jsonl(tmp_path / "results" / "train" / "stage_regression_events.jsonl")
    assert [event["event"] for event in regression_events] == ["stage_plateau_recovery"]


def test_stage_entry_optimizer_reset_uses_base_learning_rates(tmp_path: Path) -> None:
    """Stage-entry resets should scale the agent's base LRs instead of compounding prior resets."""
    config = load_config("configs/default.yaml")
    config.environment.max_episode_steps = 6
    config.environment.auto_time_budget_steps_per_meter = 0.0
    config.environment.num_dynamic_obstacles = 2
    config.training.results_dir = str(tmp_path / "results")
    config.training.log_dir = str(tmp_path / "logs")
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")
    config.evaluation.output_dir = str(tmp_path / "results" / "eval")
    config.visualization.plot_dir = str(tmp_path / "results" / "plots")
    config.visualization.trajectory_dir = str(tmp_path / "results" / "trajectories")
    config.training.stage_entry_optimizer_reset = {
        "enabled": True,
        "stages": ["target_stage"],
        "lr_multiplier": 0.5,
        "reset_on_initial_stage": True,
    }

    env = create_environment(config, gui=False, seed=13)
    agent = create_agent(config, env)
    try:
        agent.scale_learning_rates(0.1)
        event = loops._apply_stage_entry_optimizer_reset(
            config=config,
            agent=agent,
            stage_name="target_stage",
            stage_index=0,
            is_initial_stage=True,
            train_episode_before=0,
            progress_callback=None,
        )
    finally:
        env.close()

    assert event is not None
    assert event["actor_lr"] == pytest.approx(config.agent.lr_actor * 0.5)
    assert event["critic_lr"] == pytest.approx(config.agent.lr_critic * 0.5)


def test_stage_demo_pretrain_seed_salt_offsets_demo_seed_base(tmp_path: Path, monkeypatch) -> None:
    """Stage demo pretraining should expose a deterministic salt for replay refreshes."""
    config = load_config("configs/default.yaml")
    config.environment.max_episode_steps = 6
    config.environment.auto_time_budget_steps_per_meter = 0.0
    config.environment.num_dynamic_obstacles = 2
    config.training.results_dir = str(tmp_path / "results")
    config.training.log_dir = str(tmp_path / "logs")
    config.training.checkpoint_dir = str(tmp_path / "checkpoints")
    config.evaluation.output_dir = str(tmp_path / "results" / "eval")
    config.visualization.plot_dir = str(tmp_path / "results" / "plots")
    config.visualization.trajectory_dir = str(tmp_path / "results" / "trajectories")
    config.training.bc_demo_pretrain = {
        "enabled": True,
        "stages": ["demo_stage"],
        "episodes": 1,
        "epochs": 1,
        "batch_size": 1,
    }

    captured: dict[str, int] = {}

    def fake_collect_teacher_demo_dataset(**kwargs):
        captured["seed_base"] = int(kwargs["seed_base"])
        return [], [], {"num_episodes": 0, "num_samples": 0, "dataset_steps": 0, "active_fraction": 0.0}

    monkeypatch.setattr(loops, "_collect_teacher_demo_dataset", fake_collect_teacher_demo_dataset)

    env = create_environment(config, gui=False, seed=17)
    agent = create_agent(config, env)
    try:
        record = loops._run_stage_demo_pretrain(
            config=config,
            layout=loops.build_output_layout(config),
            env=env,
            agent=agent,
            stage_index=2,
            stage_name="demo_stage",
            stage_config=config,
            train_episode_before=0,
            stage_episode_before=0,
            progress_callback=None,
            seed_salt=123,
        )
    finally:
        env.close()

    assert record is not None
    assert record["seed_salt"] == 123
    assert captured["seed_base"] == config.environment.seed + 200_000 + 20_000 + 123
