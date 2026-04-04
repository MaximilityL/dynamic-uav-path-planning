"""Training and evaluation loops for the dynamic UAV path-planning scaffold."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple

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
    )
    for stale_file in layout["train_evaluations"].glob("*"):
        if stale_file.is_file():
            stale_file.unlink()


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

    env = create_environment(current_stage_config, gui=False, seed=current_stage_config.environment.seed)
    agent = create_agent(config, env)

    best_model_path = layout["checkpoints"] / "best_model.pth"
    last_model_path = layout["checkpoints"] / "last_model.pth"
    if resume:
        agent.load(resume)

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
                eval_record = {
                    "train_episode": episode_number,
                    "stage_index": current_stage_index,
                    "stage_name": current_stage_name,
                    "stage_episode": stage_episode,
                    "stage_target_hit": float(stage_target_hit),
                    "stage_success_streak": int(stage_success_streak),
                    **eval_summary,
                }
                evaluation_history.append(eval_record)
                latest_eval_summary = eval_record
                append_jsonl(layout["train"] / "eval_history.jsonl", eval_record)
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
    total_elapsed_seconds = perf_counter() - start_time

    save_json(
        layout["train"] / "summary.json",
        {
            "summary": summary,
            "best_evaluation": best_eval_summary,
            "latest_evaluation": latest_eval_summary,
            "last_update_metrics": last_update_metrics,
            "stage_transitions": stage_transitions,
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
