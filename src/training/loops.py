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
        layout["train"] / "pretrain_history.jsonl",
        layout["train"] / "summary.json",
        layout["train"] / "pretrain_summary.json",
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


def _normalize_regression_protection_settings(raw_settings: Dict[str, object]) -> Dict[str, object]:
    """Normalize stage-regression settings after defaults and overrides are merged."""
    settings = dict(raw_settings)
    settings["enabled"] = bool(settings.get("enabled", False))
    settings["activate_after_success_rate"] = float(settings.get("activate_after_success_rate", 0.7))
    settings["absolute_drop_threshold"] = float(settings.get("absolute_drop_threshold", 0.25))
    settings["relative_drop_fraction"] = float(settings.get("relative_drop_fraction", 0.4))
    settings["consecutive_bad_evals"] = max(int(settings.get("consecutive_bad_evals", 2)), 1)
    settings["rollback_on_regression"] = bool(settings.get("rollback_on_regression", True))
    settings["rollback_max_per_stage"] = max(int(settings.get("rollback_max_per_stage", 1)), 0)
    settings["lr_multiplier_after_rollback"] = float(settings.get("lr_multiplier_after_rollback", 1.0))
    settings["plateau_recovery_enabled"] = bool(settings.get("plateau_recovery_enabled", False))
    settings["plateau_bad_eval_streak"] = max(int(settings.get("plateau_bad_eval_streak", 0)), 0)
    if settings["plateau_bad_eval_streak"] <= 0:
        settings["plateau_bad_eval_streak"] = int(settings["consecutive_bad_evals"])
    settings["plateau_max_per_stage"] = max(int(settings.get("plateau_max_per_stage", 0)), 0)
    settings["lr_multiplier_after_plateau_recovery"] = float(
        settings.get("lr_multiplier_after_plateau_recovery", settings["lr_multiplier_after_rollback"])
    )
    settings["rerun_stage_demo_pretrain"] = bool(settings.get("rerun_stage_demo_pretrain", False))
    settings["reset_stage_episode_on_plateau_recovery"] = bool(
        settings.get("reset_stage_episode_on_plateau_recovery", False)
    )
    settings["stage_overrides"] = {
        str(stage_name): dict(stage_override or {})
        for stage_name, stage_override in dict(settings.get("stage_overrides", {}) or {}).items()
    }
    return settings


def _regression_protection_settings(config: Config) -> Dict[str, object]:
    """Return normalized global regression-protection settings."""
    settings: Dict[str, object] = {
        "enabled": False,
        "activate_after_success_rate": 0.7,
        "absolute_drop_threshold": 0.25,
        "relative_drop_fraction": 0.4,
        "consecutive_bad_evals": 2,
        "rollback_on_regression": True,
        "rollback_max_per_stage": 1,
        "lr_multiplier_after_rollback": 1.0,
        "plateau_recovery_enabled": False,
        "plateau_bad_eval_streak": 2,
        "plateau_max_per_stage": 0,
        "lr_multiplier_after_plateau_recovery": 1.0,
        "rerun_stage_demo_pretrain": False,
        "reset_stage_episode_on_plateau_recovery": False,
        "stage_overrides": {},
    }
    settings.update(dict(config.training.stage_regression_protection or {}))
    return _normalize_regression_protection_settings(settings)


def _stage_regression_protection_settings(
    config: Config,
    *,
    stage_name: str,
) -> Dict[str, object]:
    """Resolve stage-local regression-protection settings."""
    base_settings = _regression_protection_settings(config)
    stage_overrides = dict(base_settings["stage_overrides"].get(stage_name, {}))
    resolved_settings = dict(base_settings)
    resolved_settings.update(stage_overrides)
    resolved_settings["stage_name"] = str(stage_name)
    return _normalize_regression_protection_settings(resolved_settings)


def _bc_warm_start_settings(config: Config) -> Dict[str, object]:
    """Return normalized BC warm-start settings."""
    settings: Dict[str, object] = {
        "enabled": False,
        "stages": [],
        "initial_coef": 0.0,
        "final_coef": 0.0,
        "anneal_episodes": 0,
        "teacher_gated_only": False,
        "gate_signal": "teacher_active",
        "normalize_target_action": False,
        "stage_overrides": {},
    }
    settings.update(dict(config.training.bc_warm_start or {}))
    settings["enabled"] = bool(settings.get("enabled", False))
    settings["stages"] = [str(stage_name) for stage_name in list(settings.get("stages", []) or [])]
    settings["initial_coef"] = float(settings.get("initial_coef", 0.0))
    settings["final_coef"] = float(settings.get("final_coef", 0.0))
    settings["anneal_episodes"] = max(int(settings.get("anneal_episodes", 0)), 0)
    settings["teacher_gated_only"] = bool(settings.get("teacher_gated_only", False))
    settings["normalize_target_action"] = bool(settings.get("normalize_target_action", False))
    gate_signal = str(settings.get("gate_signal", "teacher_active")).strip().lower()
    if gate_signal not in {"teacher_active", "bypass_active", "repulsion_active"}:
        gate_signal = "teacher_active"
    settings["gate_signal"] = gate_signal
    settings["stage_overrides"] = {
        str(stage_name): dict(stage_override or {})
        for stage_name, stage_override in dict(settings.get("stage_overrides", {}) or {}).items()
    }
    return settings


def _bc_stage_settings(
    config: Config,
    *,
    stage_name: str,
    stage_episode: int,
) -> Dict[str, object]:
    """Resolve the current stage-local BC settings and annealed coefficient."""
    base_settings = _bc_warm_start_settings(config)
    stage_overrides = dict(base_settings["stage_overrides"].get(stage_name, {}))
    resolved_settings = dict(base_settings)
    resolved_settings.update(stage_overrides)
    resolved_settings["stage_name"] = str(stage_name)
    resolved_settings["stage_episode"] = int(stage_episode)
    resolved_settings["teacher_gated_only"] = bool(resolved_settings.get("teacher_gated_only", False))
    resolved_settings["normalize_target_action"] = bool(resolved_settings.get("normalize_target_action", False))
    gate_signal = str(resolved_settings.get("gate_signal", "teacher_active")).strip().lower()
    if gate_signal not in {"teacher_active", "bypass_active", "repulsion_active"}:
        gate_signal = "teacher_active"
    resolved_settings["gate_signal"] = gate_signal

    stage_filter = list(base_settings.get("stages", []) or [])
    stage_enabled = bool(resolved_settings.get("enabled", base_settings["enabled"])) and (
        not stage_filter or stage_name in stage_filter
    )
    initial_coef = max(float(resolved_settings.get("initial_coef", 0.0)), 0.0)
    final_coef = max(float(resolved_settings.get("final_coef", initial_coef)), 0.0)
    anneal_episodes = max(int(resolved_settings.get("anneal_episodes", 0)), 0)
    if anneal_episodes > 0 and not np.isclose(initial_coef, final_coef):
        anneal_progress = min(max((int(stage_episode) - 1) / anneal_episodes, 0.0), 1.0)
        current_coef = initial_coef + (final_coef - initial_coef) * anneal_progress
    else:
        anneal_progress = 0.0
        current_coef = initial_coef

    if not stage_enabled:
        anneal_progress = 0.0
        current_coef = 0.0

    resolved_settings["enabled"] = stage_enabled
    resolved_settings["active"] = bool(stage_enabled and current_coef > 0.0)
    resolved_settings["initial_coef"] = initial_coef
    resolved_settings["final_coef"] = final_coef
    resolved_settings["anneal_episodes"] = anneal_episodes
    resolved_settings["anneal_progress"] = float(anneal_progress)
    resolved_settings["current_coef"] = float(max(current_coef, 0.0))
    return resolved_settings


def _bc_mask_from_teacher_guidance(
    *,
    teacher_guidance: Optional[Dict[str, object]],
    bc_settings: Optional[Dict[str, object]],
) -> float:
    """Resolve whether BC should apply on the current transition."""
    if not teacher_guidance or not bc_settings or not bool(bc_settings.get("active", False)):
        return 0.0
    if float(bc_settings.get("current_coef", 0.0)) <= 0.0:
        return 0.0
    if not bool(bc_settings.get("teacher_gated_only", False)):
        return 1.0

    gate_signal = str(bc_settings.get("gate_signal", "teacher_active")).strip().lower()
    gate_value = teacher_guidance.get(gate_signal)
    if gate_value is None and gate_signal != "teacher_active":
        gate_value = teacher_guidance.get("teacher_active", False)
    return float(bool(gate_value))


def _bc_demo_pretrain_settings(config: Config) -> Dict[str, object]:
    """Return normalized teacher-demo pretraining settings."""
    settings: Dict[str, object] = {
        "enabled": False,
        "stages": [],
        "episodes": 0,
        "epochs": 1,
        "batch_size": 128,
        "teacher_gated_only": False,
        "gate_signal": "teacher_active",
        "normalize_target_action": False,
        "successful_episodes_only": False,
        "fallback_to_all_episodes": True,
        "evaluate_after_pretrain": False,
        "eval_episodes": 0,
        "post_pretrain_action_std": 0.0,
        "seed_offset": 200_000,
        "stage_overrides": {},
    }
    settings.update(dict(config.training.bc_demo_pretrain or {}))
    settings["enabled"] = bool(settings.get("enabled", False))
    settings["stages"] = [str(stage_name) for stage_name in list(settings.get("stages", []) or [])]
    settings["episodes"] = max(int(settings.get("episodes", 0)), 0)
    settings["epochs"] = max(int(settings.get("epochs", 1)), 1)
    settings["batch_size"] = max(int(settings.get("batch_size", 128)), 1)
    settings["teacher_gated_only"] = bool(settings.get("teacher_gated_only", False))
    settings["normalize_target_action"] = bool(settings.get("normalize_target_action", False))
    settings["successful_episodes_only"] = bool(settings.get("successful_episodes_only", False))
    settings["fallback_to_all_episodes"] = bool(settings.get("fallback_to_all_episodes", True))
    settings["evaluate_after_pretrain"] = bool(settings.get("evaluate_after_pretrain", False))
    settings["eval_episodes"] = max(int(settings.get("eval_episodes", 0)), 0)
    settings["post_pretrain_action_std"] = max(float(settings.get("post_pretrain_action_std", 0.0)), 0.0)
    settings["seed_offset"] = int(settings.get("seed_offset", 200_000))
    gate_signal = str(settings.get("gate_signal", "teacher_active")).strip().lower()
    if gate_signal not in {"teacher_active", "bypass_active", "repulsion_active"}:
        gate_signal = "teacher_active"
    settings["gate_signal"] = gate_signal
    settings["stage_overrides"] = {
        str(stage_name): dict(stage_override or {})
        for stage_name, stage_override in dict(settings.get("stage_overrides", {}) or {}).items()
    }
    return settings


def _stage_bc_demo_pretrain_settings(
    config: Config,
    *,
    stage_name: str,
) -> Dict[str, object]:
    """Resolve stage-local teacher-demo pretraining settings."""
    base_settings = _bc_demo_pretrain_settings(config)
    stage_overrides = dict(base_settings["stage_overrides"].get(stage_name, {}))
    resolved_settings = dict(base_settings)
    resolved_settings.update(stage_overrides)
    resolved_settings["stage_name"] = str(stage_name)
    resolved_settings["teacher_gated_only"] = bool(resolved_settings.get("teacher_gated_only", False))
    resolved_settings["normalize_target_action"] = bool(resolved_settings.get("normalize_target_action", False))
    resolved_settings["successful_episodes_only"] = bool(resolved_settings.get("successful_episodes_only", False))
    resolved_settings["fallback_to_all_episodes"] = bool(resolved_settings.get("fallback_to_all_episodes", True))
    resolved_settings["evaluate_after_pretrain"] = bool(resolved_settings.get("evaluate_after_pretrain", False))
    resolved_settings["eval_episodes"] = max(int(resolved_settings.get("eval_episodes", 0)), 0)
    resolved_settings["post_pretrain_action_std"] = max(float(resolved_settings.get("post_pretrain_action_std", 0.0)), 0.0)
    resolved_settings["episodes"] = max(int(resolved_settings.get("episodes", 0)), 0)
    resolved_settings["epochs"] = max(int(resolved_settings.get("epochs", 1)), 1)
    resolved_settings["batch_size"] = max(int(resolved_settings.get("batch_size", 128)), 1)
    resolved_settings["seed_offset"] = int(resolved_settings.get("seed_offset", 200_000))
    gate_signal = str(resolved_settings.get("gate_signal", "teacher_active")).strip().lower()
    if gate_signal not in {"teacher_active", "bypass_active", "repulsion_active"}:
        gate_signal = "teacher_active"
    resolved_settings["gate_signal"] = gate_signal
    stage_filter = list(base_settings.get("stages", []) or [])
    stage_enabled = bool(resolved_settings.get("enabled", base_settings["enabled"])) and (
        not stage_filter or stage_name in stage_filter
    )
    resolved_settings["enabled"] = stage_enabled
    resolved_settings["active"] = bool(stage_enabled and int(resolved_settings["episodes"]) > 0)
    return resolved_settings


def _stage_entry_optimizer_reset_settings(config: Config) -> Dict[str, object]:
    """Return normalized stage-entry optimizer reset settings."""
    settings: Dict[str, object] = {
        "enabled": False,
        "stages": [],
        "lr_multiplier": 1.0,
        "reset_on_initial_stage": False,
        "stage_overrides": {},
    }
    settings.update(dict(config.training.stage_entry_optimizer_reset or {}))
    settings["enabled"] = bool(settings.get("enabled", False))
    settings["stages"] = [str(stage_name) for stage_name in list(settings.get("stages", []) or [])]
    settings["lr_multiplier"] = float(settings.get("lr_multiplier", 1.0))
    settings["reset_on_initial_stage"] = bool(settings.get("reset_on_initial_stage", False))
    settings["stage_overrides"] = {
        str(stage_name): dict(stage_override or {})
        for stage_name, stage_override in dict(settings.get("stage_overrides", {}) or {}).items()
    }
    return settings


def _resolved_stage_entry_optimizer_reset_settings(
    config: Config,
    *,
    stage_name: str,
    is_initial_stage: bool,
) -> Dict[str, object]:
    """Resolve stage-local optimizer reset behavior for a stage entry."""
    base_settings = _stage_entry_optimizer_reset_settings(config)
    stage_overrides = dict(base_settings["stage_overrides"].get(stage_name, {}))
    resolved_settings = dict(base_settings)
    resolved_settings.update(stage_overrides)
    resolved_settings["stage_name"] = str(stage_name)
    resolved_settings["is_initial_stage"] = bool(is_initial_stage)
    stage_filter = list(base_settings.get("stages", []) or [])
    stage_selected = not stage_filter or stage_name in stage_filter
    allow_initial_stage = bool(resolved_settings.get("reset_on_initial_stage", False)) or not bool(is_initial_stage)
    resolved_settings["active"] = bool(
        resolved_settings.get("enabled", False)
        and stage_selected
        and allow_initial_stage
    )
    resolved_settings["lr_multiplier"] = float(resolved_settings.get("lr_multiplier", 1.0))
    return resolved_settings


def _apply_stage_entry_optimizer_reset(
    *,
    config: Config,
    agent: GraphPPOAgent,
    stage_name: str,
    stage_index: int,
    is_initial_stage: bool,
    train_episode_before: int,
    progress_callback: Optional[Callable[[Dict[str, object]], None]],
) -> Optional[Dict[str, object]]:
    """Reset optimizer state when entering selected stages."""
    reset_settings = _resolved_stage_entry_optimizer_reset_settings(
        config,
        stage_name=stage_name,
        is_initial_stage=is_initial_stage,
    )
    if not bool(reset_settings.get("active", False)):
        return None

    learning_rates = agent.reset_optimizer()
    lr_multiplier = float(reset_settings.get("lr_multiplier", 1.0))
    if lr_multiplier != 1.0:
        learning_rates = agent.scale_learning_rates(lr_multiplier)

    event = {
        "event": "stage_entry_optimizer_reset",
        "stage_index": int(stage_index),
        "stage_name": str(stage_name),
        "is_initial_stage": float(bool(is_initial_stage)),
        "train_episode_before": int(train_episode_before),
        "lr_multiplier": float(lr_multiplier),
        "actor_lr": float(learning_rates["actor"]),
        "critic_lr": float(learning_rates["critic"]),
    }
    _emit_progress(progress_callback, event)
    return event


def _guidance_gate_active(
    *,
    teacher_guidance: Optional[Dict[str, object]],
    gate_signal: str,
) -> bool:
    """Return whether one teacher guidance record activates the configured gate."""
    if not teacher_guidance:
        return False
    normalized_gate_signal = str(gate_signal).strip().lower()
    gate_value = teacher_guidance.get(normalized_gate_signal)
    if gate_value is None and normalized_gate_signal != "teacher_active":
        gate_value = teacher_guidance.get("teacher_active", False)
    return bool(gate_value)


def _collect_teacher_demo_dataset(
    *,
    env: DynamicAirspaceEnv,
    pretrain_settings: Dict[str, object],
    seed_base: int,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    stage_name: str = "main",
    train_episode_before: int = 0,
    stage_episode_before: int = 0,
) -> tuple[List[Dict[str, np.ndarray]], List[np.ndarray], Dict[str, object]]:
    """Collect teacher-policy demonstration data for one stage."""
    episode_records: List[Dict[str, object]] = []
    total_steps = 0
    kept_steps = 0

    teacher_gated_only = bool(pretrain_settings.get("teacher_gated_only", False))
    gate_signal = str(pretrain_settings.get("gate_signal", "teacher_active"))
    successful_episodes_only = bool(pretrain_settings.get("successful_episodes_only", False))
    fallback_to_all_episodes = bool(pretrain_settings.get("fallback_to_all_episodes", True))
    num_episodes = max(int(pretrain_settings.get("episodes", 0)), 0)
    progress_interval = 8 if num_episodes >= 8 else 1
    for demo_index in range(num_episodes):
        observation, _ = env.reset(seed=seed_base + demo_index)
        terminated = False
        truncated = False
        episode_observations: List[Dict[str, np.ndarray]] = []
        episode_target_actions: List[np.ndarray] = []
        episode_kept_steps = 0
        while not (terminated or truncated):
            teacher_guidance = env.teacher_guidance_for_current_state()
            teacher_action = np.asarray(teacher_guidance["action"], dtype=np.float32)
            total_steps += 1
            include_sample = True
            if teacher_gated_only:
                include_sample = _guidance_gate_active(teacher_guidance=teacher_guidance, gate_signal=gate_signal)
            if include_sample:
                episode_observations.append(
                    {key: np.asarray(value, dtype=np.float32).copy() for key, value in observation.items()}
                )
                episode_target_actions.append(teacher_action.copy())
                kept_steps += 1
                episode_kept_steps += 1
            observation, _, terminated, truncated, _ = env.step(teacher_action)
        episode_records.append(
            {
                "summary": env.get_episode_summary(),
                "observations": episode_observations,
                "target_actions": episode_target_actions,
                "kept_steps": int(episode_kept_steps),
            }
        )
        if (
            progress_callback is not None
            and (
                (demo_index + 1) % progress_interval == 0
                or demo_index + 1 == num_episodes
            )
        ):
            source_episode_metrics = [dict(record["summary"]) for record in episode_records]
            source_summary = summarize_episodes(source_episode_metrics) if source_episode_metrics else summarize_episodes([])
            _emit_progress(
                progress_callback,
                {
                    "event": "pretrain_collect_progress",
                    "stage_name": str(stage_name),
                    "train_episode_before": int(train_episode_before),
                    "stage_episode_before": int(stage_episode_before),
                    "demo_episode": int(demo_index + 1),
                    "demo_episodes": int(num_episodes),
                    "dataset_steps": int(total_steps),
                    "kept_steps": int(kept_steps),
                    "source_success_rate": float(source_summary.get("success_rate", 0.0)),
                    "source_collision_rate": float(source_summary.get("collision_rate", 0.0)),
                },
            )

    source_episode_metrics = [dict(record["summary"]) for record in episode_records]
    source_summary = summarize_episodes(source_episode_metrics) if source_episode_metrics else summarize_episodes([])
    selected_episode_records = episode_records
    fallback_used = False
    if successful_episodes_only:
        selected_episode_records = [
            record for record in episode_records if float(dict(record["summary"]).get("success", 0.0)) > 0.5
        ]
        if not selected_episode_records and fallback_to_all_episodes:
            selected_episode_records = episode_records
            fallback_used = True

    observations: List[Dict[str, np.ndarray]] = []
    target_actions: List[np.ndarray] = []
    selected_episode_metrics: List[Dict[str, object]] = []
    selected_steps = 0
    for record in selected_episode_records:
        observations.extend(list(record["observations"]))
        target_actions.extend(list(record["target_actions"]))
        selected_steps += int(record["kept_steps"])
        selected_episode_metrics.append(dict(record["summary"]))

    summary = summarize_episodes(selected_episode_metrics) if selected_episode_metrics else summarize_episodes([])
    summary["num_episodes"] = int(len(selected_episode_records))
    summary["configured_num_episodes"] = int(num_episodes)
    summary["source_num_episodes"] = int(len(episode_records))
    summary["source_success_rate"] = float(source_summary.get("success_rate", 0.0))
    summary["source_collision_rate"] = float(source_summary.get("collision_rate", 0.0))
    summary["source_avg_episode_return"] = float(source_summary.get("avg_episode_return", 0.0))
    summary["num_samples"] = int(len(target_actions))
    summary["dataset_steps"] = int(total_steps)
    summary["active_fraction"] = float(selected_steps / max(total_steps, 1))
    summary["teacher_gated_only"] = float(teacher_gated_only)
    summary["gate_signal"] = gate_signal
    summary["successful_episodes_only"] = float(successful_episodes_only)
    summary["fallback_to_all_episodes"] = float(fallback_to_all_episodes)
    summary["success_only_fallback_used"] = float(fallback_used)
    return observations, target_actions, summary


def _run_stage_demo_pretrain(
    *,
    config: Config,
    layout: Dict[str, Path],
    env: DynamicAirspaceEnv,
    agent: GraphPPOAgent,
    stage_index: int,
    stage_name: str,
    stage_config: Config,
    train_episode_before: int,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    stage_episode_before: int = 0,
) -> Optional[Dict[str, object]]:
    """Run one stage-entry teacher-demo pretraining pass when configured."""
    pretrain_settings = _stage_bc_demo_pretrain_settings(config, stage_name=stage_name)
    if not bool(pretrain_settings.get("active", False)):
        return None

    seed_base = (
        int(stage_config.environment.seed)
        + int(pretrain_settings.get("seed_offset", 200_000))
        + int(stage_index) * 10_000
    )
    _emit_progress(
        progress_callback,
        {
            "event": "pretrain_start",
            "stage_index": stage_index,
            "stage_name": stage_name,
            "train_episode_before": int(train_episode_before),
            "stage_episode_before": int(stage_episode_before),
            "episodes": int(pretrain_settings["episodes"]),
            "epochs": int(pretrain_settings["epochs"]),
            "batch_size": int(pretrain_settings["batch_size"]),
            "teacher_gated_only": float(bool(pretrain_settings["teacher_gated_only"])),
            "successful_episodes_only": float(bool(pretrain_settings["successful_episodes_only"])),
            "evaluate_after_pretrain": float(bool(pretrain_settings["evaluate_after_pretrain"])),
            "eval_episodes": int(pretrain_settings["eval_episodes"]),
            "gate_signal": str(pretrain_settings["gate_signal"]),
            "normalize_target_action": float(bool(pretrain_settings["normalize_target_action"])),
            "post_pretrain_action_std": float(pretrain_settings.get("post_pretrain_action_std", 0.0)),
        },
    )

    observations, target_actions, dataset_summary = _collect_teacher_demo_dataset(
        env=env,
        pretrain_settings=pretrain_settings,
        seed_base=seed_base,
        progress_callback=progress_callback,
        stage_name=stage_name,
        train_episode_before=train_episode_before,
        stage_episode_before=stage_episode_before,
    )
    pretrain_metrics = agent.behavior_clone_pretrain(
        observations=observations,
        target_actions=target_actions,
        epochs=int(pretrain_settings["epochs"]),
        batch_size=int(pretrain_settings["batch_size"]),
        normalize_target_action=bool(pretrain_settings["normalize_target_action"]),
    )
    applied_action_std = None
    if float(pretrain_settings.get("post_pretrain_action_std", 0.0)) > 0.0:
        applied_action_std = agent.set_action_std(float(pretrain_settings["post_pretrain_action_std"]))
    pretrain_record: Dict[str, object] = {
        "stage_index": int(stage_index),
        "stage_name": str(stage_name),
        "train_episode_before": int(train_episode_before),
        "stage_episode_before": int(stage_episode_before),
        "seed_base": int(seed_base),
        "seed_offset": int(pretrain_settings.get("seed_offset", 200_000)),
        "configured_demo_episodes": int(pretrain_settings["episodes"]),
        "configured_epochs": int(pretrain_settings["epochs"]),
        "configured_batch_size": int(pretrain_settings["batch_size"]),
        "teacher_gated_only": float(bool(pretrain_settings["teacher_gated_only"])),
        "successful_episodes_only": float(bool(pretrain_settings["successful_episodes_only"])),
        "evaluate_after_pretrain": float(bool(pretrain_settings["evaluate_after_pretrain"])),
        "eval_episodes": int(pretrain_settings["eval_episodes"]),
        "gate_signal": str(pretrain_settings["gate_signal"]),
        "normalize_target_action": float(bool(pretrain_settings["normalize_target_action"])),
        "post_pretrain_action_std": float(pretrain_settings.get("post_pretrain_action_std", 0.0)),
        "applied_action_std_mean": None if applied_action_std is None else float(np.mean(applied_action_std)),
        **dataset_summary,
        **pretrain_metrics,
    }
    append_jsonl(layout["train"] / "pretrain_history.jsonl", pretrain_record)
    _emit_progress(
        progress_callback,
        {
            "event": "pretrain_finish",
            "stage_index": stage_index,
            "stage_name": stage_name,
            "train_episode_before": int(train_episode_before),
            "summary": pretrain_record,
        },
    )
    return pretrain_record


def _update_stage_best_checkpoint(
    *,
    agent: GraphPPOAgent,
    stage_state: Dict[str, object],
    stage_index: int,
    stage_name: str,
    train_episode: int,
    stage_episode: int,
    stage_eval_index: int,
    eval_summary: Dict[str, float],
    recent_training_summary: Optional[Dict[str, float]],
    config_name: str,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> bool:
    """Update per-stage best checkpoint bookkeeping when one evaluation improves on the prior best."""
    stage_eval_score = _checkpoint_score(eval_summary, stage_index)
    stage_best_updated = stage_eval_score > stage_state["best_eval_score"]
    if not stage_best_updated:
        return False

    stage_state["best_eval_score"] = stage_eval_score
    stage_state["best_eval"] = {
        "stage_index": int(stage_index),
        "stage_name": str(stage_name),
        "train_episode": int(train_episode),
        "stage_episode": int(stage_episode),
        "eval_index": int(stage_eval_index),
        **eval_summary,
    }
    stage_state["best_train_episode"] = int(train_episode)
    stage_state["best_stage_episode"] = int(stage_episode)
    stage_state["best_eval_index"] = int(stage_eval_index)
    stage_state["bad_eval_streak"] = 0
    metadata = {
        "episode": int(train_episode),
        "stage_index": int(stage_index),
        "stage_name": str(stage_name),
        "stage_episode": int(stage_episode),
        "stage_eval_index": int(stage_eval_index),
        "evaluation": dict(stage_state["best_eval"]),
        "recent_training_summary": dict(recent_training_summary or {}),
        "config_name": str(config_name),
    }
    if extra_metadata:
        metadata.update(dict(extra_metadata))
    agent.save(stage_state["best_model_path"], metadata=metadata)
    save_json(stage_state["best_eval_summary_path"], dict(stage_state["best_eval"]))
    return True


def _run_stage_post_pretrain_evaluation(
    *,
    config: Config,
    layout: Dict[str, Path],
    agent: GraphPPOAgent,
    stage: Optional[Dict[str, object]],
    stage_state: Dict[str, object],
    stage_index: int,
    stage_name: str,
    stage_config: Config,
    train_episode_before: int,
    stage_episode_before: int,
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
) -> Optional[Dict[str, object]]:
    """Evaluate the freshly pretrained policy before PPO updates can regress it."""
    pretrain_settings = _stage_bc_demo_pretrain_settings(config, stage_name=stage_name)
    eval_episodes = int(pretrain_settings.get("eval_episodes", 0))
    if not bool(pretrain_settings.get("evaluate_after_pretrain", False)) or eval_episodes <= 0:
        return None

    eval_history, eval_summary, eval_best_trajectory = _evaluate_current_policy(
        config=stage_config,
        agent=agent,
        num_episodes=eval_episodes,
        render=False,
        deterministic=True,
    )
    stage_target_hit = False
    if stage is not None:
        stage_target_hit = _stage_target_reached(
            stage=stage,
            eval_summary=eval_summary,
            stage_episode_count=stage_episode_before,
        )
    stage_state["num_evaluations"] = int(stage_state.get("num_evaluations", 0)) + 1
    stage_eval_index = int(stage_state["num_evaluations"])
    stage_best_updated = _update_stage_best_checkpoint(
        agent=agent,
        stage_state=stage_state,
        stage_index=stage_index,
        stage_name=stage_name,
        train_episode=train_episode_before,
        stage_episode=stage_episode_before,
        stage_eval_index=stage_eval_index,
        eval_summary=eval_summary,
        recent_training_summary=None,
        config_name=config.name,
        extra_metadata={"pretrain_eval": True},
    )
    eval_record = {
        "train_episode": int(train_episode_before),
        "stage_index": int(stage_index),
        "stage_name": str(stage_name),
        "stage_episode": int(stage_episode_before),
        "stage_eval_index": int(stage_eval_index),
        "stage_target_hit": float(stage_target_hit),
        "stage_success_streak": 0,
        "stage_best_success_rate": float(dict(stage_state.get("best_eval") or {}).get("success_rate", 0.0)),
        "stage_best_collision_rate": float(dict(stage_state.get("best_eval") or {}).get("collision_rate", 0.0)),
        "stage_best_avg_return": float(dict(stage_state.get("best_eval") or {}).get("avg_episode_return", 0.0)),
        "stage_bad_eval_streak": int(stage_state.get("bad_eval_streak", 0)),
        "stage_regression_active": 0.0,
        "stage_regression_bad_eval": 0.0,
        "stage_regression_absolute_drop": 0.0,
        "stage_regression_relative_drop": 0.0,
        "stage_regression_detected": 0.0,
        "stage_rollback_applied": 0.0,
        "stage_plateau_recovery_applied": 0.0,
        "pretrain_eval": 1.0,
        **eval_summary,
    }
    append_jsonl(layout["train"] / "eval_history.jsonl", eval_record)
    save_json(layout["train"] / "latest_eval_summary.json", eval_record)
    save_json(
        layout["train_evaluations"] / f"pretrain_eval_{stage_name}_summary.json",
        {
            "summary": eval_record,
            "episodes": eval_history,
        },
    )
    if config.evaluation.save_trajectories and eval_best_trajectory is not None:
        _save_best_trajectory(layout["train_evaluations"] / f"pretrain_eval_{stage_name}_best_episode", eval_best_trajectory)
    _emit_progress(
        progress_callback,
        {
            "event": "pretrain_eval",
            "stage_index": int(stage_index),
            "stage_name": str(stage_name),
            "train_episode_before": int(train_episode_before),
            "stage_episode_before": int(stage_episode_before),
            "evaluation": eval_record,
            "stage_best_updated": float(stage_best_updated),
        },
    )
    return {
        "eval_record": eval_record,
        "eval_summary": eval_summary,
        "best_trajectory": eval_best_trajectory,
        "stage_target_hit": bool(stage_target_hit),
        "stage_best_updated": bool(stage_best_updated),
    }


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
        "plateau_recovery_count": 0,
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
        "plateau_recovery_count": int(stage_state.get("plateau_recovery_count", 0)),
        "best_model_path": str(stage_state["best_model_path"]),
    }


def _apply_stage_plateau_recovery(
    *,
    config: Config,
    layout: Dict[str, Path],
    env: DynamicAirspaceEnv,
    agent: GraphPPOAgent,
    stage_state: Dict[str, object],
    stage_index: int,
    stage_name: str,
    stage_episode: int,
    stage_eval_index: int,
    stage_config: Config,
    train_episode_before: int,
    protection: Dict[str, object],
    progress_callback: Optional[Callable[[Dict[str, object]], None]],
) -> Optional[Dict[str, object]]:
    """Reload the stage best checkpoint and optionally replay demos after a long plateau."""
    plateau_bad_eval_streak = int(protection.get("plateau_bad_eval_streak", protection.get("consecutive_bad_evals", 1)))
    best_model_path = Path(stage_state["best_model_path"])
    can_recover = (
        bool(protection.get("plateau_recovery_enabled", False))
        and int(stage_state.get("bad_eval_streak", 0)) >= plateau_bad_eval_streak
        and int(stage_state.get("plateau_recovery_count", 0)) < int(protection.get("plateau_max_per_stage", 0))
        and best_model_path.exists()
    )
    if not can_recover:
        return None

    agent.clear_rollout_buffers()
    agent.load(
        best_model_path,
        load_optimizer_state=False,
        load_scheduler_state=False,
        reset_optimizer_if_skipped=True,
    )
    lr_multiplier = float(protection.get("lr_multiplier_after_plateau_recovery", 1.0))
    if lr_multiplier > 0.0 and not np.isclose(lr_multiplier, 1.0):
        learning_rates = agent.scale_learning_rates(lr_multiplier)
    else:
        learning_rates = agent.learning_rates()

    pretrain_record = None
    if bool(protection.get("rerun_stage_demo_pretrain", False)):
        pretrain_record = _run_stage_demo_pretrain(
            config=config,
            layout=layout,
            env=env,
            agent=agent,
            stage_index=stage_index,
            stage_name=stage_name,
            stage_config=stage_config,
            train_episode_before=train_episode_before,
            stage_episode_before=stage_episode,
            progress_callback=progress_callback,
        )

    stage_state["plateau_recovery_count"] = int(stage_state.get("plateau_recovery_count", 0)) + 1
    stage_state["bad_eval_streak"] = 0
    reset_stage_episode = bool(protection.get("reset_stage_episode_on_plateau_recovery", False))
    event = {
        "event": "stage_plateau_recovery",
        "episode": int(train_episode_before),
        "stage_index": int(stage_index),
        "stage_name": str(stage_name),
        "stage_episode": int(stage_episode),
        "stage_eval_index": int(stage_eval_index),
        "recovery_count": int(stage_state["plateau_recovery_count"]),
        "bad_eval_streak_trigger": int(plateau_bad_eval_streak),
        "restored_checkpoint": str(best_model_path),
        "actor_lr": float(learning_rates["actor"]),
        "critic_lr": float(learning_rates["critic"]),
        "reran_demo_pretrain": float(pretrain_record is not None),
        "demo_samples": float((pretrain_record or {}).get("num_samples", 0.0)),
        "reset_stage_episode": float(reset_stage_episode),
    }
    return {
        "event": event,
        "pretrain_record": pretrain_record,
        "reset_stage_episode": reset_stage_episode,
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
    bc_settings: Optional[Dict[str, object]] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Run one episode and optionally store transitions for PPO updates."""
    observation, _ = env.reset(seed=episode_seed)
    terminated = False
    truncated = False

    while not (terminated or truncated):
        teacher_guidance = None
        teacher_action = None
        if store_transition and not deterministic:
            teacher_guidance = env.teacher_guidance_for_current_state()
            teacher_action = np.asarray(teacher_guidance["action"], dtype=np.float32)
        action, policy_info = agent.select_action(observation, deterministic=deterministic)
        executed_action = action
        if store_transition and not deterministic:
            executed_action = teacher_guided_action(
                policy_action=action,
                teacher_action=teacher_action,
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
                teacher_action=teacher_action,
                bc_mask=_bc_mask_from_teacher_guidance(teacher_guidance=teacher_guidance, bc_settings=bc_settings),
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
    stage_entry_optimizer_reset_settings = _stage_entry_optimizer_reset_settings(config)
    stage_entry_optimizer_reset_events: List[Dict[str, object]] = []
    bc_warm_start_settings = _bc_warm_start_settings(config)
    bc_demo_pretrain_settings = _bc_demo_pretrain_settings(config)
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
    pretrain_history: List[Dict[str, object]] = []
    best_eval_summary: Optional[Dict[str, float]] = None
    latest_eval_summary: Optional[Dict[str, float]] = None
    best_eval_score = (-float("inf"), -float("inf"), -float("inf"), -float("inf"))
    last_update_metrics = {
        "actor_loss": 0.0,
        "critic_loss": 0.0,
        "entropy": 0.0,
        "bc_loss": 0.0,
        "bc_nonzero": 0.0,
        "bc_coef": 0.0,
        "bc_active_fraction": 0.0,
        "bc_active_samples": 0.0,
        "bc_total_samples": 0.0,
    }
    episodes_completed = 0
    start_time = perf_counter()
    pretrained_stage_names: set[str] = set()

    _emit_progress(
        progress_callback,
        {
            "event": "start",
            "total_episodes": episode_budget,
            "stage_index": current_stage_index,
            "stage_name": current_stage_name,
            "curriculum_stage_count": len(curriculum),
            "config_name": config.name,
            "bc_enabled": float(bool(bc_warm_start_settings["enabled"])),
            "bc_stages": list(bc_warm_start_settings["stages"]),
            "bc_teacher_gated_only": float(bool(bc_warm_start_settings["teacher_gated_only"])),
            "bc_gate_signal": str(bc_warm_start_settings["gate_signal"]),
            "bc_demo_pretrain_enabled": float(bool(bc_demo_pretrain_settings["enabled"])),
            "bc_demo_pretrain_stages": list(bc_demo_pretrain_settings["stages"]),
            "stage_entry_optimizer_reset_enabled": float(bool(stage_entry_optimizer_reset_settings["enabled"])),
            "stage_entry_optimizer_reset_stages": list(stage_entry_optimizer_reset_settings["stages"]),
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
        if stage_start_episode == 1 and current_stage_name not in pretrained_stage_names:
            reset_event = _apply_stage_entry_optimizer_reset(
                config=config,
                agent=agent,
                stage_name=current_stage_name,
                stage_index=current_stage_index,
                is_initial_stage=True,
                train_episode_before=episodes_completed,
                progress_callback=progress_callback,
            )
            if reset_event is not None:
                stage_entry_optimizer_reset_events.append(reset_event)
            pretrain_record = _run_stage_demo_pretrain(
                config=config,
                layout=layout,
                env=env,
                agent=agent,
                stage_index=current_stage_index,
                stage_name=current_stage_name,
                stage_config=current_stage_config,
                train_episode_before=episodes_completed,
                stage_episode_before=0,
                progress_callback=progress_callback,
            )
            if pretrain_record is not None:
                pretrain_history.append(pretrain_record)
                save_json(layout["train"] / "pretrain_summary.json", {"runs": pretrain_history})
            pretrain_eval = _run_stage_post_pretrain_evaluation(
                config=config,
                layout=layout,
                agent=agent,
                stage=current_stage,
                stage_state=stage_states[current_stage_name],
                stage_index=current_stage_index,
                stage_name=current_stage_name,
                stage_config=current_stage_config,
                train_episode_before=episodes_completed,
                stage_episode_before=0,
                progress_callback=progress_callback,
            )
            if pretrain_eval is not None:
                eval_record = dict(pretrain_eval["eval_record"])
                evaluation_history.append(eval_record)
                latest_eval_summary = eval_record
                save_json(
                    layout["train"] / "stage_best_evaluations.json",
                    {stage_name: _serialize_stage_best(state) for stage_name, state in stage_states.items()},
                )
                if _checkpoint_score(dict(pretrain_eval["eval_summary"]), current_stage_index) > best_eval_score:
                    best_eval_score = _checkpoint_score(dict(pretrain_eval["eval_summary"]), current_stage_index)
                    best_eval_summary = eval_record
                    agent.save(
                        best_model_path,
                        metadata={
                            "episode": episodes_completed,
                            "stage_index": current_stage_index,
                            "stage_name": current_stage_name,
                            "evaluation": eval_record,
                            "recent_training_summary": {},
                            "config_name": config.name,
                            "pretrain_eval": True,
                        },
                    )
                    save_json(layout["train"] / "best_eval_summary.json", eval_record)
            pretrained_stage_names.add(current_stage_name)

        for episode_idx in range(episode_budget):
            episode_number = episode_idx + 1
            stage_episode = episode_number - stage_start_episode + 1
            stage_bc_settings = _bc_stage_settings(
                config,
                stage_name=current_stage_name,
                stage_episode=stage_episode,
            )
            metrics, _ = run_episode(
                env=env,
                agent=agent,
                episode_seed=current_stage_config.environment.seed + episode_idx,
                deterministic=False,
                store_transition=True,
                bc_settings=stage_bc_settings,
            )
            history.append(metrics)
            episodes_completed = episode_number

            is_rollout_boundary = (episode_number % config.agent.rollout_episodes) == 0
            is_final_episode = episode_number == episode_budget
            policy_updated = False
            stop_training_early = False
            if (is_rollout_boundary or is_final_episode) and agent.has_pending_rollout():
                last_update_metrics = agent.update(
                    bc_coef=float(stage_bc_settings["current_coef"]),
                    normalize_bc_target_action=bool(stage_bc_settings["normalize_target_action"]),
                )
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
                    "bc_enabled": float(bool(stage_bc_settings["enabled"])),
                    "bc_stage_active": float(bool(stage_bc_settings["active"])),
                    "bc_coef": float(stage_bc_settings["current_coef"]),
                    "bc_anneal_progress": float(stage_bc_settings["anneal_progress"]),
                    "bc_teacher_gated_only": float(bool(stage_bc_settings["teacher_gated_only"])),
                    "bc_gate_signal": str(stage_bc_settings["gate_signal"]),
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
                    "bc_settings": stage_bc_settings,
                },
            )

            should_evaluate = (episode_number % config.training.eval_interval) == 0 or is_final_episode
            if should_evaluate:
                stage_regression_protection = _stage_regression_protection_settings(
                    config,
                    stage_name=current_stage_name,
                )
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
                stage_best_updated = _update_stage_best_checkpoint(
                    agent=agent,
                    stage_state=stage_state,
                    stage_index=current_stage_index,
                    stage_name=current_stage_name,
                    train_episode=episode_number,
                    stage_episode=stage_episode,
                    stage_eval_index=stage_eval_index,
                    eval_summary=eval_summary,
                    recent_training_summary=recent_summary,
                    config_name=config.name,
                )

                regression_signal = _stage_regression_signal(
                    eval_summary=eval_summary,
                    stage_state=stage_state,
                    protection=stage_regression_protection,
                )
                if stage_best_updated:
                    stage_state["bad_eval_streak"] = 0
                elif bool(regression_signal["bad"]):
                    stage_state["bad_eval_streak"] = int(stage_state.get("bad_eval_streak", 0)) + 1
                else:
                    stage_state["bad_eval_streak"] = 0

                regression_detected = False
                rollback_applied = False
                plateau_recovery_applied = False
                best_eval_for_stage = dict(stage_state.get("best_eval") or {})
                if (
                    bool(regression_signal["bad"])
                    and int(stage_state["bad_eval_streak"]) >= int(stage_regression_protection["consecutive_bad_evals"])
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
                        bool(stage_regression_protection["rollback_on_regression"])
                        and int(stage_state["rollback_count"]) < int(stage_regression_protection["rollback_max_per_stage"])
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
                        lr_multiplier = float(stage_regression_protection["lr_multiplier_after_rollback"])
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
                    elif bool(stage_regression_protection.get("plateau_recovery_enabled", False)):
                        plateau_recovery = _apply_stage_plateau_recovery(
                            config=config,
                            layout=layout,
                            env=env,
                            agent=agent,
                            stage_state=stage_state,
                            stage_index=current_stage_index,
                            stage_name=current_stage_name,
                            stage_episode=stage_episode,
                            stage_eval_index=stage_eval_index,
                            stage_config=current_stage_config,
                            train_episode_before=episode_number,
                            protection=stage_regression_protection,
                            progress_callback=progress_callback,
                        )
                        if plateau_recovery is not None:
                            plateau_event = dict(plateau_recovery["event"])
                            stage_regression_events.append(plateau_event)
                            append_jsonl(layout["train"] / "stage_regression_events.jsonl", plateau_event)
                            _emit_progress(progress_callback, plateau_event)
                            pretrain_record = plateau_recovery.get("pretrain_record")
                            if pretrain_record is not None:
                                pretrain_history.append(pretrain_record)
                                save_json(layout["train"] / "pretrain_summary.json", {"runs": pretrain_history})
                            stage_success_streak = 0
                            plateau_recovery_applied = True
                            if bool(plateau_recovery.get("reset_stage_episode", False)):
                                stage_start_episode = episode_number + 1

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
                    "stage_plateau_recovery_applied": float(plateau_recovery_applied),
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
                        reset_event = _apply_stage_entry_optimizer_reset(
                            config=config,
                            agent=agent,
                            stage_name=current_stage_name,
                            stage_index=current_stage_index,
                            is_initial_stage=False,
                            train_episode_before=episode_number,
                            progress_callback=progress_callback,
                        )
                        if reset_event is not None:
                            stage_entry_optimizer_reset_events.append(reset_event)
                        if current_stage_name not in pretrained_stage_names:
                            pretrain_record = _run_stage_demo_pretrain(
                                config=config,
                                layout=layout,
                                env=env,
                                agent=agent,
                                stage_index=current_stage_index,
                                stage_name=current_stage_name,
                                stage_config=current_stage_config,
                                train_episode_before=episode_number,
                                stage_episode_before=0,
                                progress_callback=progress_callback,
                            )
                        if pretrain_record is not None:
                            pretrain_history.append(pretrain_record)
                            save_json(layout["train"] / "pretrain_summary.json", {"runs": pretrain_history})
                        pretrain_eval = _run_stage_post_pretrain_evaluation(
                            config=config,
                            layout=layout,
                            agent=agent,
                            stage=current_stage,
                            stage_state=stage_states[current_stage_name],
                            stage_index=current_stage_index,
                            stage_name=current_stage_name,
                            stage_config=current_stage_config,
                            train_episode_before=episode_number,
                            stage_episode_before=0,
                            progress_callback=progress_callback,
                        )
                        if pretrain_eval is not None:
                            eval_record = dict(pretrain_eval["eval_record"])
                            evaluation_history.append(eval_record)
                            latest_eval_summary = eval_record
                            save_json(
                                layout["train"] / "stage_best_evaluations.json",
                                {stage_name: _serialize_stage_best(state) for stage_name, state in stage_states.items()},
                            )
                            if _checkpoint_score(dict(pretrain_eval["eval_summary"]), current_stage_index) > best_eval_score:
                                best_eval_score = _checkpoint_score(dict(pretrain_eval["eval_summary"]), current_stage_index)
                                best_eval_summary = eval_record
                                agent.save(
                                    best_model_path,
                                    metadata={
                                        "episode": episode_number,
                                        "stage_index": current_stage_index,
                                        "stage_name": current_stage_name,
                                        "evaluation": eval_record,
                                        "recent_training_summary": {},
                                        "config_name": config.name,
                                        "pretrain_eval": True,
                                    },
                                )
                                save_json(layout["train"] / "best_eval_summary.json", eval_record)
                        pretrained_stage_names.add(current_stage_name)
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
    summary["stage_plateau_recovery_count"] = int(
        sum(1 for event in stage_regression_events if str(event.get("event")) == "stage_plateau_recovery")
    )
    summary["stage_entry_optimizer_reset_count"] = int(len(stage_entry_optimizer_reset_events))
    summary["pretrain_run_count"] = int(len(pretrain_history))
    summary["pretrain_total_samples"] = float(sum(float(run.get("bc_pretrain_samples", 0.0)) for run in pretrain_history))
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
            "stage_entry_optimizer_reset_events": stage_entry_optimizer_reset_events,
            "pretrain_runs": pretrain_history,
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
            "bc_warm_start": bc_warm_start_settings,
            "stage_entry_optimizer_reset": stage_entry_optimizer_reset_settings,
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
