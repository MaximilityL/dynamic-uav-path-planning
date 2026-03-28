"""Training and evaluation loops for the dynamic UAV path-planning scaffold."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from ..agents import GraphPPOAgent
from ..environments import DynamicAirspaceEnv
from ..evaluation.metrics import summarize_episodes
from ..utils.config import Config, save_config
from ..utils.io import append_jsonl, save_json, save_npz
from .factories import build_output_layout, create_agent, create_environment, set_global_seeds


def run_episode(
    *,
    env: DynamicAirspaceEnv,
    agent: GraphPPOAgent,
    episode_seed: int,
    deterministic: bool,
    store_transition: bool,
) -> Tuple[Dict[str, float], Dict[str, object]]:
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
    episode_budget = int(num_episodes or config.training.num_episodes)

    train_config_path = layout["train"] / "config_used.yaml"
    save_config(config, train_config_path)

    env = create_environment(config, gui=False, seed=config.environment.seed)
    agent = create_agent(config, env)

    best_model_path = layout["checkpoints"] / "best_model.pth"
    last_model_path = layout["checkpoints"] / "last_model.pth"
    if resume:
        agent.load(resume)

    history = []
    best_score = (-float("inf"), -float("inf"))
    last_update_metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    try:
        for episode_idx in range(episode_budget):
            metrics, _ = run_episode(
                env=env,
                agent=agent,
                episode_seed=config.environment.seed + episode_idx,
                deterministic=False,
                store_transition=True,
            )
            history.append(metrics)

            if (episode_idx + 1) % config.agent.rollout_episodes == 0:
                last_update_metrics = agent.update()

            recent_window = history[-config.training.moving_average_window :]
            recent_summary = summarize_episodes(recent_window)
            score = (recent_summary["success_rate"], recent_summary["avg_episode_return"])
            if score >= best_score:
                best_score = score
                agent.save(
                    best_model_path,
                    metadata={
                        "episode": episode_idx + 1,
                        "summary": recent_summary,
                        "config_name": config.name,
                    },
                )

            if (episode_idx + 1) % config.training.save_interval == 0:
                agent.save(
                    last_model_path,
                    metadata={
                        "episode": episode_idx + 1,
                        "summary": recent_summary,
                        "config_name": config.name,
                    },
                )

            append_jsonl(
                layout["train"] / "history.jsonl",
                {
                    "episode": episode_idx + 1,
                    **metrics,
                    **last_update_metrics,
                },
            )
    finally:
        env.close()

    if agent.has_pending_rollout():
        last_update_metrics = agent.update()
        append_jsonl(layout["train"] / "history.jsonl", {"episode": episode_budget, **last_update_metrics})

    agent.save(
        last_model_path,
        metadata={
            "episode": episode_budget,
            "summary": summarize_episodes(history),
            "config_name": config.name,
        },
    )

    summary = summarize_episodes(history)
    summary["num_episodes"] = int(episode_budget)
    summary["best_model_path"] = str(best_model_path)
    save_json(
        layout["train"] / "summary.json",
        {
            "summary": summary,
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

    env = create_environment(config, gui=render, seed=config.environment.seed)
    agent = create_agent(config, env)
    agent.load(model_path)

    history = []
    best_trajectory = None
    best_return = -float("inf")

    try:
        for episode_idx in range(episode_budget):
            metrics, trajectory = run_episode(
                env=env,
                agent=agent,
                episode_seed=config.environment.seed + 10_000 + episode_idx,
                deterministic=deterministic,
                store_transition=False,
            )
            history.append(metrics)
            append_jsonl(
                layout["eval"] / "episodes.jsonl",
                {"episode": episode_idx + 1, **metrics},
            )
            if metrics["episode_return"] >= best_return:
                best_return = metrics["episode_return"]
                best_trajectory = trajectory
    finally:
        env.close()

    summary = summarize_episodes(history)
    summary["model_path"] = str(model_path)
    summary["num_episodes"] = int(episode_budget)

    if save_outputs:
        save_json(layout["eval"] / "summary.json", summary)
        if config.evaluation.save_trajectories and best_trajectory is not None:
            save_npz(
                layout["trajectories"] / "eval_best_episode.npz",
                {
                    "positions": best_trajectory["positions"],
                    "obstacles": best_trajectory["obstacles"],
                    "goal": best_trajectory["goal"],
                    "actions": best_trajectory["actions"],
                    "rewards": best_trajectory["rewards"],
                },
            )
            save_json(layout["trajectories"] / "eval_best_episode_summary.json", best_trajectory["summary"])

    return summary
