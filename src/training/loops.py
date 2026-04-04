"""Training and evaluation loops for the dynamic UAV path-planning scaffold."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..agents import GraphPPOAgent
from ..environments import DynamicAirspaceEnv
from ..environments.teacher import teacher_guided_action
from ..evaluation.metrics import summarize_episodes
from ..utils.config import Config, save_config
from ..utils.io import append_jsonl, save_json, save_npz
from .factories import build_output_layout, create_agent, create_environment, set_global_seeds

EVALUATION_SEED_OFFSET = 100_000


def _unlink_if_exists(*paths: Path) -> None:
    """Remove previous run artifacts that would otherwise be appended to."""
    for path in paths:
        if path.exists():
            path.unlink()


def _reset_training_artifacts(layout: Dict[str, Path]) -> None:
    """Clear the training artifacts owned by this scaffold before a fresh run."""
    _unlink_if_exists(
        layout["train"] / "history.jsonl",
        layout["train"] / "eval_history.jsonl",
        layout["train"] / "summary.json",
        layout["train"] / "best_eval_summary.json",
        layout["train"] / "latest_eval_summary.json",
        layout["train"] / "stage_best_evaluations.json",
        layout["train"] / "stage_regression_events.jsonl",
        layout["checkpoints"] / "best_model.pth",
        layout["checkpoints"] / "last_model.pth",
    )
    for stale_file in layout["train_evaluations"].glob("*"):
        if stale_file.is_file():
            stale_file.unlink()
    stage_checkpoint_dir = layout["checkpoints"] / "stages"
    if stage_checkpoint_dir.exists():
        shutil.rmtree(stage_checkpoint_dir)


def _checkpoint_score(summary: Dict[str, float], stage_index: int = 0) -> tuple[float, float, float, float]:
    """Score evaluation summaries for best-checkpoint selection."""
    return (
        float(summary.get("success_rate", 0.0)),
        float(stage_index),
        -float(summary.get("collision_rate", 0.0)),
        float(summary.get("avg_episode_return", 0.0)),
    )


def _curriculum_stage_name(stage: Dict[str, object], stage_index: int) -> str:
    """Return a stable human-readable stage name."""
    return str(stage.get("name", f"stage_{stage_index + 1}"))


def _stage_directory_name(stage_name: str) -> str:
    """Convert a stage name into a filesystem-safe directory slug."""
    return stage_name.replace("/", "_").replace("\\", "_").replace(" ", "_")


def _apply_component_overrides(component: object, overrides: Dict[str, object]) -> None:
    """Apply stage overrides to one config component."""
    for key, value in overrides.items():
        if not hasattr(component, key):
            raise AttributeError(f"Unsupported curriculum override '{key}' for {type(component).__name__}")
        current_value = getattr(component, key)
        if isinstance(current_value, dict) and isinstance(value, dict):
            merged = dict(current_value)
            merged.update(value)
            setattr(component, key, merged)
        else:
            setattr(component, key, value)


def _validate_curriculum(base_config: Config) -> None:
    """Reject curriculum changes that the fixed-size policy cannot support."""
    if not base_config.training.curriculum:
        return

    base_obstacle_count = int(base_config.environment.num_dynamic_obstacles)
    for stage in base_config.training.curriculum:
        env_overrides = dict(stage.get("environment", {}))
        if "num_dynamic_obstacles" in env_overrides and int(env_overrides["num_dynamic_obstacles"]) != base_obstacle_count:
            raise ValueError(
                "Curriculum stages must keep num_dynamic_obstacles fixed because the current policy uses a fixed graph size."
            )


def _stage_config(base_config: Config, stage: Optional[Dict[str, object]]) -> Config:
    """Build a config copy with one curriculum stage applied."""
    stage_config = deepcopy(base_config)
    if stage is None:
        return stage_config

    env_overrides = dict(stage.get("environment", {}))
    if env_overrides:
        _apply_component_overrides(stage_config.environment, env_overrides)
    return stage_config


def _stage_target_reached(
    *,
    stage: Dict[str, object],
    eval_summary: Dict[str, float],
    stage_episode_count: int,
) -> bool:
    """Check whether the current stage has been solved well enough to advance or stop."""
    min_stage_episodes = int(stage.get("min_stage_episodes", 0))
    if stage_episode_count < min_stage_episodes:
        return False

    success_target = float(stage.get("advance_success_rate", 1.1))
    collision_target = float(stage.get("advance_collision_rate", 1.0))
    return (
        float(eval_summary.get("success_rate", 0.0)) >= success_target
        and float(eval_summary.get("collision_rate", 1.0)) <= collision_target
    )


def _emit_progress(
    progress_callback: Optional[Callable[[Dict[str, object]], None]],
    payload: Dict[str, object],
) -> None:
    """Send a progress event when the caller asked for one."""
    if progress_callback is not None:
        progress_callback(payload)


def _resolve_resume_settings(
    config: Config,
    *,
    resume: Optional[str],
    resume_overrides: Optional[Dict[str, object]],
) -> Dict[str, object]:
    """Merge resume settings from config and CLI-style overrides."""
    settings: Dict[str, object] = {
        "checkpoint_path": None,
        "load_optimizer_state": True,
        "load_scheduler_state": True,
        "restore_curriculum_progress": False,
    }
    settings.update(dict(config.training.resume or {}))
    if resume is not None:
        settings["checkpoint_path"] = resume
    if resume_overrides:
        for key, value in resume_overrides.items():
            if value is not None:
                settings[key] = value
    settings["checkpoint_path"] = settings.get("checkpoint_path")
    settings["load_optimizer_state"] = bool(settings.get("load_optimizer_state", True))
    settings["load_scheduler_state"] = bool(settings.get("load_scheduler_state", True))
    settings["restore_curriculum_progress"] = bool(settings.get("restore_curriculum_progress", False))
    return settings


def _resolve_restored_stage(
    curriculum: List[Dict[str, object]],
    metadata: Dict[str, object],
) -> tuple[int, int] | None:
    """Resolve a saved curriculum stage from checkpoint metadata."""
    if not curriculum:
        return None

    saved_stage_name = metadata.get("stage_name")
    if saved_stage_name is not None:
        for stage_index, stage in enumerate(curriculum):
            if _curriculum_stage_name(stage, stage_index) == str(saved_stage_name):
                saved_stage_episode = metadata.get("stage_episode")
                if saved_stage_episode is None:
                    saved_stage_episode = dict(metadata.get("evaluation", {}) or {}).get("stage_episode", 1)
                return stage_index, max(int(saved_stage_episode), 1)

    saved_stage_index = metadata.get("stage_index")
    if saved_stage_index is None:
        return None

    stage_index = int(saved_stage_index)
    if stage_index < 0 or stage_index >= len(curriculum):
        return None
    saved_stage_episode = metadata.get("stage_episode")
    if saved_stage_episode is None:
        saved_stage_episode = dict(metadata.get("evaluation", {}) or {}).get("stage_episode", 1)
    return stage_index, max(int(saved_stage_episode), 1)


def _regression_protection_settings(config: Config) -> Dict[str, object]:
    """Return normalized stage-regression settings."""
    settings: Dict[str, object] = {
        "enabled": False,
        "activate_after_success_rate": 0.7,
        "absolute_drop_threshold": 0.25,
        "relative_drop_fraction": 0.4,
        "consecutive_bad_evals": 2,
        "rollback_on_regression": True,
        "rollback_max_per_stage": 1,
        "lr_multiplier_after_rollback": 1.0,
    }
    settings.update(dict(config.training.stage_regression_protection or {}))
    settings["enabled"] = bool(settings.get("enabled", False))
    settings["activate_after_success_rate"] = float(settings.get("activate_after_success_rate", 0.7))
    settings["absolute_drop_threshold"] = float(settings.get("absolute_drop_threshold", 0.25))
    settings["relative_drop_fraction"] = float(settings.get("relative_drop_fraction", 0.4))
    settings["consecutive_bad_evals"] = max(int(settings.get("consecutive_bad_evals", 2)), 1)
    settings["rollback_on_regression"] = bool(settings.get("rollback_on_regression", True))
    settings["rollback_max_per_stage"] = max(int(settings.get("rollback_max_per_stage", 1)), 0)
    settings["lr_multiplier_after_rollback"] = float(settings.get("lr_multiplier_after_rollback", 1.0))
    return settings


def _stage_tracking_state(layout: Dict[str, Path], *, stage_index: int, stage_name: str) -> Dict[str, object]:
    """Create per-stage bookkeeping for best checkpoints and rollback state."""
    stage_dir = layout["checkpoints"] / "stages" / _stage_directory_name(stage_name)
    return {
        "stage_index": int(stage_index),
        "stage_name": str(stage_name),
        "best_eval_score": (-float("inf"), -float("inf"), -float("inf"), -float("inf")),
        "best_eval": None,
        "best_train_episode": 0,
        "best_stage_episode": 0,
        "best_eval_index": 0,
        "num_evaluations": 0,
        "bad_eval_streak": 0,
        "rollback_count": 0,
        "best_model_path": stage_dir / "best_model.pth",
        "best_eval_summary_path": stage_dir / "best_eval_summary.json",
    }


def _stage_regression_signal(
    *,
    eval_summary: Dict[str, float],
    stage_state: Dict[str, object],
    protection: Dict[str, object],
) -> Dict[str, float]:
    """Check whether the current stage evaluation is a sharp drop from the stage best."""
    best_eval = stage_state.get("best_eval")
    if best_eval is None or not bool(protection.get("enabled", False)):
        return {"active": 0.0, "bad": 0.0, "best_success_rate": 0.0, "absolute_drop": 0.0, "relative_drop": 0.0}

    best_success = float(dict(best_eval).get("success_rate", 0.0))
    activate_after = float(protection.get("activate_after_success_rate", 0.7))
    # Float32 success rates can land just below the threshold (for example 0.699999988 for 0.7).
    if best_success + 1e-6 < activate_after:
        return {
            "active": 0.0,
            "bad": 0.0,
            "best_success_rate": best_success,
            "absolute_drop": 0.0,
            "relative_drop": 0.0,
        }

    current_success = float(eval_summary.get("success_rate", 0.0))
    absolute_drop = max(best_success - current_success, 0.0)
    relative_drop = absolute_drop / max(best_success, 1e-6)
    bad_eval = (
        absolute_drop >= float(protection.get("absolute_drop_threshold", 0.25))
        and relative_drop >= float(protection.get("relative_drop_fraction", 0.4))
    )
    return {
        "active": 1.0,
        "bad": float(bad_eval),
        "best_success_rate": best_success,
        "absolute_drop": absolute_drop,
        "relative_drop": relative_drop,
    }


def _serialize_stage_best(stage_state: Dict[str, object]) -> Dict[str, object]:
    """Convert internal stage tracking state into a JSON-friendly summary."""
    best_eval = dict(stage_state.get("best_eval") or {})
    return {
        "stage_index": int(stage_state["stage_index"]),
        "stage_name": str(stage_state["stage_name"]),
        "num_evaluations": int(stage_state.get("num_evaluations", 0)),
        "best_success_rate": float(best_eval.get("success_rate", 0.0)),
        "best_collision_rate": float(best_eval.get("collision_rate", 0.0)),
        "best_avg_return": float(best_eval.get("avg_episode_return", 0.0)),
        "best_avg_episode_return": float(best_eval.get("avg_episode_return", 0.0)),
        "best_train_episode": int(stage_state.get("best_train_episode", 0)),
        "best_stage_episode": int(stage_state.get("best_stage_episode", 0)),
        "best_eval_index": int(stage_state.get("best_eval_index", 0)),
        "rollback_count": int(stage_state.get("rollback_count", 0)),
        "best_model_path": str(stage_state["best_model_path"]),
    }


def _evaluate_current_policy(
    *,
    config: Config,
    agent: GraphPPOAgent,
    num_episodes: int,
    render: bool,
    deterministic: bool,
    seed_offset: int = EVALUATION_SEED_OFFSET,
) -> tuple[List[Dict[str, object]], Dict[str, float], Optional[Dict[str, object]]]:
    """Evaluate the in-memory policy on a fixed deterministic seed set."""
    env = create_environment(config, gui=render, seed=config.environment.seed + seed_offset)
    history: List[Dict[str, object]] = []
    best_trajectory = None
    best_return = -float("inf")

    try:
        for episode_idx in range(int(num_episodes)):
            metrics, trajectory = run_episode(
                env=env,
                agent=agent,
                episode_seed=config.environment.seed + seed_offset + episode_idx,
                deterministic=deterministic,
                store_transition=False,
            )
            history.append(metrics)
            if float(metrics.get("episode_return", 0.0)) >= best_return:
                best_return = float(metrics.get("episode_return", 0.0))
                best_trajectory = trajectory
    finally:
        env.close()

    summary = summarize_episodes(history)
    summary["num_episodes"] = int(num_episodes)
    summary["deterministic"] = float(deterministic)
    summary["seed_offset"] = int(seed_offset)
    return history, summary, best_trajectory


def _save_best_trajectory(path_prefix: Path, trajectory: Dict[str, object]) -> None:
    """Persist one evaluation trajectory in the same format as standalone evaluation."""
    save_npz(
        path_prefix.with_suffix(".npz"),
        {
            "positions": trajectory["positions"],
            "obstacles": trajectory["obstacles"],
            "goal": trajectory["goal"],
            "actions": trajectory["actions"],
            "rewards": trajectory["rewards"],
        },
    )
    save_json(path_prefix.with_name(f"{path_prefix.stem}_summary.json"), trajectory["summary"])


def run_episode(
    *,
    env: DynamicAirspaceEnv,
    agent: GraphPPOAgent,
    episode_seed: int,
    deterministic: bool,
    store_transition: bool,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Run one episode and optionally store transitions for PPO updates."""
    observation, _ = env.reset(seed=episode_seed)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, policy_info = agent.select_action(observation, deterministic=deterministic)
        executed_action = action
        if store_transition and not deterministic:
            executed_action = teacher_guided_action(
                policy_action=action,
                teacher_action=env.teacher_action_for_current_state(),
                action_low=env.action_space.low,
                action_high=env.action_space.high,
                teacher_config=env.teacher_config,
            )
            if not np.array_equal(executed_action, action):
                policy_info = agent.evaluate_action(observation, executed_action)

        next_observation, reward, terminated, truncated, _ = env.step(executed_action)
        if store_transition:
            agent.store_transition(
                observation=observation,
                action=executed_action,
                reward=reward,
                done=terminated,
                log_prob=policy_info["log_prob"],
                value=policy_info["value"],
            )
        observation = next_observation

    if store_transition:
        bootstrap_value = 0.0 if terminated else agent.estimate_value(observation)
        agent.finish_rollout(last_value=bootstrap_value)

    return env.get_episode_summary(), env.export_episode()


def train_agent(
    *,
    config: Config,
    resume: Optional[str] = None,
    num_episodes: Optional[int] = None,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    resume_overrides: Optional[Dict[str, object]] = None,
) -> Dict[str, float]:
    """Train the graph PPO baseline for a configurable number of episodes."""
    set_global_seeds(config.environment.seed)
    _validate_curriculum(config)
    layout = build_output_layout(config)
    _reset_training_artifacts(layout)
    episode_budget = int(num_episodes or config.training.num_episodes)

    train_config_path = layout["train"] / "config_used.yaml"
    save_config(config, train_config_path)

    curriculum: List[Dict[str, object]] = list(config.training.curriculum or [])
    current_stage_index = 0
    current_stage = curriculum[current_stage_index] if curriculum else None
    current_stage_name = _curriculum_stage_name(current_stage, current_stage_index) if current_stage else "main"
    current_stage_config = _stage_config(config, current_stage)
    stage_start_episode = 1
    stage_success_streak = 0
    stage_transitions: List[Dict[str, object]] = []
    stage_regression_events: List[Dict[str, object]] = []
    regression_protection = _regression_protection_settings(config)
    stage_states: Dict[str, Dict[str, object]] = {}
    if curriculum:
        for stage_index, stage in enumerate(curriculum):
            stage_name = _curriculum_stage_name(stage, stage_index)
            stage_states[stage_name] = _stage_tracking_state(layout, stage_index=stage_index, stage_name=stage_name)
    else:
        stage_states["main"] = _stage_tracking_state(layout, stage_index=0, stage_name="main")

    env = create_environment(current_stage_config, gui=False, seed=current_stage_config.environment.seed)
    agent = create_agent(config, env)

    best_model_path = layout["checkpoints"] / "best_model.pth"
    last_model_path = layout["checkpoints"] / "last_model.pth"
    resume_settings = _resolve_resume_settings(
        config,
        resume=resume,
        resume_overrides=resume_overrides,
    )
    resume_path = resume_settings.get("checkpoint_path")
    resumed_metadata: Dict[str, object] = {}
    if resume_path:
        resumed_metadata = agent.load(
            str(resume_path),
            load_optimizer_state=bool(resume_settings["load_optimizer_state"]),
            load_scheduler_state=bool(resume_settings["load_scheduler_state"]),
            reset_optimizer_if_skipped=not bool(resume_settings["load_optimizer_state"]),
        )
        agent.clear_rollout_buffers()
        if bool(resume_settings["restore_curriculum_progress"]):
            restored_stage = _resolve_restored_stage(curriculum, resumed_metadata)
            if restored_stage is not None:
                restored_stage_index, restored_stage_episode = restored_stage
                current_stage_index = restored_stage_index
                current_stage = curriculum[current_stage_index]
                current_stage_name = _curriculum_stage_name(current_stage, current_stage_index)
                current_stage_config = _stage_config(config, current_stage)
                stage_start_episode = 2 - restored_stage_episode
                stage_success_streak = 0
                env.close()
                env = create_environment(current_stage_config, gui=False, seed=current_stage_config.environment.seed)

    history: List[Dict[str, object]] = []
    evaluation_history: List[Dict[str, float]] = []
    best_eval_summary: Optional[Dict[str, float]] = None
    latest_eval_summary: Optional[Dict[str, float]] = None
    best_eval_score = (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
    last_update_metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
    episodes_completed = 0
    start_time = perf_counter()

    _emit_progress(
        progress_callback,
        {
            "event": "start",
            "total_episodes": episode_budget,
            "stage_index": current_stage_index,
            "stage_name": current_stage_name,
            "curriculum_stage_count": len(curriculum),
            "config_name": config.name,
        },
    )
    if resume_path:
        _emit_progress(
            progress_callback,
            {
                "event": "resume",
                "checkpoint_path": str(resume_path),
                "load_optimizer_state": float(bool(resume_settings["load_optimizer_state"])),
                "load_scheduler_state": float(bool(resume_settings["load_scheduler_state"])),
                "restore_curriculum_progress": float(bool(resume_settings["restore_curriculum_progress"])),
                "restored_stage_index": current_stage_index,
                "restored_stage_name": current_stage_name,
            },
        )

    try:
        for episode_idx in range(episode_budget):
            episode_number = episode_idx + 1
            stage_episode = episode_number - stage_start_episode + 1
            metrics, _ = run_episode(
                env=env,
                agent=agent,
                episode_seed=current_stage_config.environment.seed + episode_idx,
                deterministic=False,
                store_transition=True,
            )
            history.append(metrics)
            episodes_completed = episode_number

            is_rollout_boundary = (episode_number % config.agent.rollout_episodes) == 0
            is_final_episode = episode_number == episode_budget
            policy_updated = False
            stop_training_early = False
            if (is_rollout_boundary or is_final_episode) and agent.has_pending_rollout():
                last_update_metrics = agent.update()
                policy_updated = True

            recent_window = history[-config.training.moving_average_window :]
            recent_summary = summarize_episodes(recent_window)
            elapsed_seconds = perf_counter() - start_time
            eta_seconds = (elapsed_seconds / max(episode_number, 1)) * max(episode_budget - episode_number, 0)
            append_jsonl(
                layout["train"] / "history.jsonl",
                {
                    "episode": episode_number,
                    **metrics,
                    **last_update_metrics,
                    "stage_index": current_stage_index,
                    "stage_name": current_stage_name,
                    "stage_episode": stage_episode,
                    "policy_updated": float(policy_updated),
                    "rolling_success_rate": recent_summary["success_rate"],
                    "rolling_collision_rate": recent_summary["collision_rate"],
                    "rolling_avg_episode_return": recent_summary["avg_episode_return"],
                },
            )
            _emit_progress(
                progress_callback,
                {
                    "event": "episode",
                    "episode": episode_number,
                    "total_episodes": episode_budget,
                    "stage_index": current_stage_index,
                    "stage_name": current_stage_name,
                    "stage_episode": stage_episode,
                    "elapsed_seconds": elapsed_seconds,
                    "eta_seconds": eta_seconds,
                    "metrics": metrics,
                    "rolling_summary": recent_summary,
                    "policy_updated": policy_updated,
                    "update_metrics": last_update_metrics,
                },
            )

            should_evaluate = (episode_number % config.training.eval_interval) == 0 or is_final_episode
            if should_evaluate:
                eval_history, eval_summary, eval_best_trajectory = _evaluate_current_policy(
                    config=current_stage_config,
                    agent=agent,
                    num_episodes=config.training.eval_episodes,
                    render=False,
                    deterministic=True,
                )
                stage_target_hit = False
                required_consecutive_evals = 1
                if current_stage is not None:
                    stage_target_hit = _stage_target_reached(
                        stage=current_stage,
                        eval_summary=eval_summary,
                        stage_episode_count=stage_episode,
                    )
                    stage_success_streak = stage_success_streak + 1 if stage_target_hit else 0
                    required_consecutive_evals = int(current_stage.get("required_consecutive_evals", 1))
                stage_state = stage_states[current_stage_name]
                stage_state["num_evaluations"] = int(stage_state.get("num_evaluations", 0)) + 1
                stage_eval_index = int(stage_state["num_evaluations"])
                stage_eval_score = _checkpoint_score(eval_summary, current_stage_index)
                stage_best_updated = stage_eval_score > stage_state["best_eval_score"]
                if stage_best_updated:
                    stage_state["best_eval_score"] = stage_eval_score
                    stage_state["best_eval"] = {
                        "stage_index": current_stage_index,
                        "stage_name": current_stage_name,
                        "train_episode": episode_number,
                        "stage_episode": stage_episode,
                        "eval_index": stage_eval_index,
                        **eval_summary,
                    }
                    stage_state["best_train_episode"] = int(episode_number)
                    stage_state["best_stage_episode"] = int(stage_episode)
                    stage_state["best_eval_index"] = int(stage_eval_index)
                    stage_state["bad_eval_streak"] = 0
                    agent.save(
                        stage_state["best_model_path"],
                        metadata={
                            "episode": episode_number,
                            "stage_index": current_stage_index,
                            "stage_name": current_stage_name,
                            "stage_episode": stage_episode,
                            "stage_eval_index": stage_eval_index,
                            "evaluation": {
                                "stage_index": current_stage_index,
                                "stage_name": current_stage_name,
                                "train_episode": episode_number,
                                "stage_episode": stage_episode,
                                "eval_index": stage_eval_index,
                                **eval_summary,
                            },
                            "recent_training_summary": recent_summary,
                            "config_name": config.name,
                        },
                    )
                    save_json(stage_state["best_eval_summary_path"], dict(stage_state["best_eval"]))

                regression_signal = _stage_regression_signal(
                    eval_summary=eval_summary,
                    stage_state=stage_state,
                    protection=regression_protection,
                )
                if stage_best_updated:
                    stage_state["bad_eval_streak"] = 0
                elif bool(regression_signal["bad"]):
                    stage_state["bad_eval_streak"] = int(stage_state.get("bad_eval_streak", 0)) + 1
                else:
                    stage_state["bad_eval_streak"] = 0

                regression_detected = False
                rollback_applied = False
                best_eval_for_stage = dict(stage_state.get("best_eval") or {})
                if (
                    bool(regression_signal["bad"])
                    and int(stage_state["bad_eval_streak"]) >= int(regression_protection["consecutive_bad_evals"])
                ):
                    regression_detected = True
                    detection_event = {
                        "event": "stage_regression_detected",
                        "episode": episode_number,
                        "stage_index": current_stage_index,
                        "stage_name": current_stage_name,
                        "stage_episode": stage_episode,
                        "stage_eval_index": stage_eval_index,
                        "current_success_rate": float(eval_summary.get("success_rate", 0.0)),
                        "best_success_rate": float(best_eval_for_stage.get("success_rate", 0.0)),
                        "absolute_drop": float(regression_signal["absolute_drop"]),
                        "relative_drop": float(regression_signal["relative_drop"]),
                        "bad_eval_streak": int(stage_state["bad_eval_streak"]),
                        "rollback_count": int(stage_state["rollback_count"]),
                    }
                    stage_regression_events.append(detection_event)
                    append_jsonl(layout["train"] / "stage_regression_events.jsonl", detection_event)
                    _emit_progress(progress_callback, detection_event)

                    can_rollback = (
                        bool(regression_protection["rollback_on_regression"])
                        and int(stage_state["rollback_count"]) < int(regression_protection["rollback_max_per_stage"])
                        and Path(stage_state["best_model_path"]).exists()
                    )
                    if can_rollback:
                        agent.clear_rollout_buffers()
                        agent.load(
                            stage_state["best_model_path"],
                            load_optimizer_state=False,
                            load_scheduler_state=False,
                            reset_optimizer_if_skipped=True,
                        )
                        lr_multiplier = float(regression_protection["lr_multiplier_after_rollback"])
                        if lr_multiplier > 0.0 and lr_multiplier != 1.0:
                            learning_rates = agent.scale_learning_rates(lr_multiplier)
                        else:
                            learning_rates = agent.learning_rates()
                        stage_state["rollback_count"] = int(stage_state.get("rollback_count", 0)) + 1
                        stage_state["bad_eval_streak"] = 0
                        stage_success_streak = 0
                        rollback_applied = True
                        rollback_event = {
                            "event": "stage_rollback",
                            "episode": episode_number,
                            "stage_index": current_stage_index,
                            "stage_name": current_stage_name,
                            "stage_episode": stage_episode,
                            "stage_eval_index": stage_eval_index,
                            "rollback_count": int(stage_state["rollback_count"]),
                            "restored_checkpoint": str(stage_state["best_model_path"]),
                            "actor_lr": float(learning_rates["actor"]),
                            "critic_lr": float(learning_rates["critic"]),
                        }
                        stage_regression_events.append(rollback_event)
                        append_jsonl(layout["train"] / "stage_regression_events.jsonl", rollback_event)
                        _emit_progress(progress_callback, rollback_event)

                eval_record = {
                    "train_episode": episode_number,
                    "stage_index": current_stage_index,
                    "stage_name": current_stage_name,
                    "stage_episode": stage_episode,
                    "stage_eval_index": stage_eval_index,
                    "stage_target_hit": float(stage_target_hit),
                    "stage_success_streak": int(stage_success_streak),
                    "stage_best_success_rate": float(best_eval_for_stage.get("success_rate", 0.0)),
                    "stage_best_collision_rate": float(best_eval_for_stage.get("collision_rate", 0.0)),
                    "stage_best_avg_return": float(best_eval_for_stage.get("avg_episode_return", 0.0)),
                    "stage_bad_eval_streak": int(stage_state["bad_eval_streak"]),
                    "stage_regression_active": float(regression_signal["active"]),
                    "stage_regression_bad_eval": float(regression_signal["bad"]),
                    "stage_regression_absolute_drop": float(regression_signal["absolute_drop"]),
                    "stage_regression_relative_drop": float(regression_signal["relative_drop"]),
                    "stage_regression_detected": float(regression_detected),
                    "stage_rollback_applied": float(rollback_applied),
                    **eval_summary,
                }
                evaluation_history.append(eval_record)
                latest_eval_summary = eval_record
                append_jsonl(layout["train"] / "eval_history.jsonl", eval_record)
                save_json(layout["train"] / "latest_eval_summary.json", latest_eval_summary)
                save_json(
                    layout["train"] / "stage_best_evaluations.json",
                    {stage_name: _serialize_stage_best(state) for stage_name, state in stage_states.items()},
                )
                _emit_progress(
                    progress_callback,
                    {
                        "event": "evaluation",
                        "episode": episode_number,
                        "total_episodes": episode_budget,
                        "stage_index": current_stage_index,
                        "stage_name": current_stage_name,
                        "stage_episode": stage_episode,
                        "elapsed_seconds": perf_counter() - start_time,
                        "eta_seconds": (elapsed_seconds / max(episode_number, 1)) * max(episode_budget - episode_number, 0),
                        "evaluation": eval_record,
                    },
                )
                save_json(
                    layout["train_evaluations"] / f"eval_{episode_number:04d}_summary.json",
                    {
                        "summary": eval_record,
                        "episodes": eval_history,
                    },
                )
                if config.evaluation.save_trajectories and eval_best_trajectory is not None:
                    _save_best_trajectory(
                        layout["train_evaluations"] / f"eval_{episode_number:04d}_best_episode",
                        eval_best_trajectory,
                    )

                if _checkpoint_score(eval_summary, current_stage_index) > best_eval_score:
                    best_eval_score = _checkpoint_score(eval_summary, current_stage_index)
                    best_eval_summary = eval_record
                    agent.save(
                        best_model_path,
                        metadata={
                            "episode": episode_number,
                            "stage_index": current_stage_index,
                            "stage_name": current_stage_name,
                            "evaluation": eval_record,
                            "recent_training_summary": recent_summary,
                            "config_name": config.name,
                        },
                    )
                    save_json(layout["train"] / "best_eval_summary.json", eval_record)

                if current_stage is not None and stage_success_streak >= required_consecutive_evals:
                    if current_stage_index + 1 < len(curriculum):
                        next_stage_index = current_stage_index + 1
                        next_stage = curriculum[next_stage_index]
                        transition = {
                            "episode": episode_number,
                            "from_stage_index": current_stage_index,
                            "from_stage_name": current_stage_name,
                            "to_stage_index": next_stage_index,
                            "to_stage_name": _curriculum_stage_name(next_stage, next_stage_index),
                        }
                        stage_transitions.append(transition)
                        _emit_progress(
                            progress_callback,
                            {
                                "event": "stage_transition",
                                "episode": episode_number,
                                "total_episodes": episode_budget,
                                "elapsed_seconds": perf_counter() - start_time,
                                **transition,
                            },
                        )
                        current_stage_index = next_stage_index
                        current_stage = next_stage
                        current_stage_name = _curriculum_stage_name(current_stage, current_stage_index)
                        current_stage_config = _stage_config(config, current_stage)
                        stage_start_episode = episode_number + 1
                        stage_success_streak = 0
                        agent.clear_rollout_buffers()
                        env.close()
                        env = create_environment(
                            current_stage_config,
                            gui=False,
                            seed=current_stage_config.environment.seed,
                        )
                    else:
                        stop_training_early = True

            if (episode_number % config.training.save_interval) == 0 or is_final_episode or stop_training_early:
                agent.save(
                    last_model_path,
                    metadata={
                        "episode": episode_number,
                        "stage_index": current_stage_index,
                        "stage_name": current_stage_name,
                        "stage_episode": stage_episode,
                        "latest_evaluation": latest_eval_summary,
                        "recent_training_summary": recent_summary,
                        "config_name": config.name,
                    },
                )
            if stop_training_early:
                break
    finally:
        env.close()

    if best_eval_summary is None:
        raise RuntimeError("Training finished without any evaluation pass; check eval_interval handling.")

    if latest_eval_summary is not None:
        save_json(layout["train"] / "latest_eval_summary.json", latest_eval_summary)

    summary = summarize_episodes(history)
    summary["num_episodes"] = int(episodes_completed)
    summary["num_evaluations"] = int(len(evaluation_history))
    summary["best_model_path"] = str(best_model_path)
    summary["last_model_path"] = str(last_model_path)
    summary["best_eval_success_rate"] = float(best_eval_summary["success_rate"])
    summary["best_eval_collision_rate"] = float(best_eval_summary["collision_rate"])
    summary["best_eval_avg_episode_return"] = float(best_eval_summary["avg_episode_return"])
    summary["curriculum_stage_count"] = int(len(curriculum))
    summary["completed_stage_index"] = int(current_stage_index)
    summary["completed_stage_name"] = str(current_stage_name)
    summary["stopped_early"] = float(episodes_completed < episode_budget)
    summary["stage_regression_event_count"] = int(
        sum(1 for event in stage_regression_events if str(event.get("event")) == "stage_regression_detected")
    )
    summary["stage_rollback_count"] = int(sum(1 for event in stage_regression_events if str(event.get("event")) == "stage_rollback"))
    if resume_path:
        summary["resume_checkpoint_path"] = str(resume_path)
        summary["resume_loaded_optimizer_state"] = float(bool(resume_settings["load_optimizer_state"]))
        summary["resume_restored_curriculum_progress"] = float(bool(resume_settings["restore_curriculum_progress"]))
    stage_best_evaluations = {
        stage_name: _serialize_stage_best(stage_state) for stage_name, stage_state in stage_states.items()
    }
    total_elapsed_seconds = perf_counter() - start_time

    save_json(layout["train"] / "stage_best_evaluations.json", stage_best_evaluations)

    save_json(
        layout["train"] / "summary.json",
        {
            "summary": summary,
            "best_evaluation": best_eval_summary,
            "latest_evaluation": latest_eval_summary,
            "last_update_metrics": last_update_metrics,
            "stage_best_evaluations": stage_best_evaluations,
            "stage_transitions": stage_transitions,
            "stage_regression_events": stage_regression_events,
        },
    )
    _emit_progress(
        progress_callback,
        {
            "event": "finish",
            "total_episodes": episode_budget,
            "completed_episodes": episodes_completed,
            "elapsed_seconds": total_elapsed_seconds,
            "summary": summary,
        },
    )
    return summary


def evaluate_agent(
    *,
    config: Config,
    model_path: str,
    num_episodes: Optional[int] = None,
    render: bool,
    deterministic: bool,
    save_outputs: bool,
) -> Dict[str, float]:
    """Evaluate a saved graph PPO checkpoint."""
    layout = build_output_layout(config)
    episode_budget = int(num_episodes or config.evaluation.num_episodes)

    agent_env = create_environment(config, gui=False, seed=config.environment.seed)
    agent = create_agent(config, agent_env)
    agent.load(model_path)
    agent_env.close()

    history, summary, best_trajectory = _evaluate_current_policy(
        config=config,
        agent=agent,
        num_episodes=episode_budget,
        render=render,
        deterministic=deterministic,
    )
    summary["model_path"] = str(model_path)
    summary["num_episodes"] = int(episode_budget)

    if save_outputs:
        episodes_path = layout["eval"] / "episodes.jsonl"
        _unlink_if_exists(
            episodes_path,
            layout["eval"] / "summary.json",
            layout["trajectories"] / "eval_best_episode.npz",
            layout["trajectories"] / "eval_best_episode_summary.json",
        )
        for episode_idx, metrics in enumerate(history, start=1):
            append_jsonl(episodes_path, {"episode": episode_idx, **metrics})
        save_json(layout["eval"] / "summary.json", summary)
        if config.evaluation.save_trajectories and best_trajectory is not None:
            _save_best_trajectory(layout["trajectories"] / "eval_best_episode", best_trajectory)

    return summary
