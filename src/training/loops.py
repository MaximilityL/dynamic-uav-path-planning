"""Training and evaluation loops for the dynamic UAV path-planning scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..agents import GraphPPOAgent
from ..environments import DynamicAirspaceEnv
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


def _checkpoint_score(summary: Dict[str, float]) -> tuple[float, float, float]:
    """Score evaluation summaries for best-checkpoint selection."""
    return (
        float(summary.get("success_rate", 0.0)),
        -float(summary.get("collision_rate", 0.0)),
        float(summary.get("avg_episode_return", 0.0)),
    )


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
        next_observation, reward, terminated, truncated, _ = env.step(action)
        if store_transition:
            agent.store_transition(
                observation=observation,
                action=action,
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
) -> Dict[str, float]:
    """Train the graph PPO baseline for a configurable number of episodes."""
    set_global_seeds(config.environment.seed)
    layout = build_output_layout(config)
    _reset_training_artifacts(layout)
    episode_budget = int(num_episodes or config.training.num_episodes)

    train_config_path = layout["train"] / "config_used.yaml"
    save_config(config, train_config_path)

    env = create_environment(config, gui=False, seed=config.environment.seed)
    agent = create_agent(config, env)

    best_model_path = layout["checkpoints"] / "best_model.pth"
    last_model_path = layout["checkpoints"] / "last_model.pth"
    if resume:
        agent.load(resume)

    history: List[Dict[str, object]] = []
    evaluation_history: List[Dict[str, float]] = []
    best_eval_summary: Optional[Dict[str, float]] = None
    latest_eval_summary: Optional[Dict[str, float]] = None
    best_eval_score = (-float("inf"), -float("inf"), -float("inf"))
    last_update_metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    try:
        for episode_idx in range(episode_budget):
            episode_number = episode_idx + 1
            metrics, _ = run_episode(
                env=env,
                agent=agent,
                episode_seed=config.environment.seed + episode_idx,
                deterministic=False,
                store_transition=True,
            )
            history.append(metrics)

            is_rollout_boundary = (episode_number % config.agent.rollout_episodes) == 0
            is_final_episode = episode_number == episode_budget
            policy_updated = False
            if (is_rollout_boundary or is_final_episode) and agent.has_pending_rollout():
                last_update_metrics = agent.update()
                policy_updated = True

            recent_window = history[-config.training.moving_average_window :]
            recent_summary = summarize_episodes(recent_window)
            append_jsonl(
                layout["train"] / "history.jsonl",
                {
                    "episode": episode_number,
                    **metrics,
                    **last_update_metrics,
                    "policy_updated": float(policy_updated),
                    "rolling_success_rate": recent_summary["success_rate"],
                    "rolling_collision_rate": recent_summary["collision_rate"],
                    "rolling_avg_episode_return": recent_summary["avg_episode_return"],
                },
            )

            should_evaluate = (episode_number % config.training.eval_interval) == 0 or is_final_episode
            if should_evaluate:
                eval_history, eval_summary, eval_best_trajectory = _evaluate_current_policy(
                    config=config,
                    agent=agent,
                    num_episodes=config.training.eval_episodes,
                    render=False,
                    deterministic=True,
                )
                eval_record = {
                    "train_episode": episode_number,
                    **eval_summary,
                }
                evaluation_history.append(eval_record)
                latest_eval_summary = eval_record
                append_jsonl(layout["train"] / "eval_history.jsonl", eval_record)
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

                if _checkpoint_score(eval_summary) > best_eval_score:
                    best_eval_score = _checkpoint_score(eval_summary)
                    best_eval_summary = eval_record
                    agent.save(
                        best_model_path,
                        metadata={
                            "episode": episode_number,
                            "evaluation": eval_record,
                            "recent_training_summary": recent_summary,
                            "config_name": config.name,
                        },
                    )
                    save_json(layout["train"] / "best_eval_summary.json", eval_record)

            if (episode_number % config.training.save_interval) == 0 or is_final_episode:
                agent.save(
                    last_model_path,
                    metadata={
                        "episode": episode_number,
                        "latest_evaluation": latest_eval_summary,
                        "recent_training_summary": recent_summary,
                        "config_name": config.name,
                    },
                )
    finally:
        env.close()

    if best_eval_summary is None:
        raise RuntimeError("Training finished without any evaluation pass; check eval_interval handling.")

    if latest_eval_summary is not None:
        save_json(layout["train"] / "latest_eval_summary.json", latest_eval_summary)

    summary = summarize_episodes(history)
    summary["num_episodes"] = int(episode_budget)
    summary["num_evaluations"] = int(len(evaluation_history))
    summary["best_model_path"] = str(best_model_path)
    summary["last_model_path"] = str(last_model_path)
    summary["best_eval_success_rate"] = float(best_eval_summary["success_rate"])
    summary["best_eval_collision_rate"] = float(best_eval_summary["collision_rate"])
    summary["best_eval_avg_episode_return"] = float(best_eval_summary["avg_episode_return"])

    save_json(
        layout["train"] / "summary.json",
        {
            "summary": summary,
            "best_evaluation": best_eval_summary,
            "latest_evaluation": latest_eval_summary,
            "last_update_metrics": last_update_metrics,
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
