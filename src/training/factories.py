"""Factories and runtime helpers for training and evaluation."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from ..agents import GraphPPOAgent
from ..environments import DynamicAirspaceEnv
from ..utils.config import Config, diagnose_torch_device_resolution, resolve_torch_device


def set_global_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_output_layout(config: Config) -> Dict[str, Path]:
    """Create the output directory structure for saved results and artifacts."""
    results_root = Path(config.training.results_dir)
    layout = {
        "results_root": results_root,
        "train": results_root / "train",
        "eval": Path(config.evaluation.output_dir),
        "trajectories": Path(config.visualization.trajectory_dir),
        "plots": Path(config.visualization.plot_dir),
        "logs": Path(config.training.log_dir),
        "checkpoints": Path(config.training.checkpoint_dir),
    }
    for directory in layout.values():
        directory.mkdir(parents=True, exist_ok=True)
    return layout


def create_environment(config: Config, gui: Optional[bool] = None, seed: Optional[int] = None) -> DynamicAirspaceEnv:
    """Instantiate the dynamic airspace environment from config."""
    env_cfg = config.environment
    actual_seed = env_cfg.seed if seed is None else seed
    return DynamicAirspaceEnv(
        num_agents=env_cfg.num_agents,
        num_dynamic_obstacles=env_cfg.num_dynamic_obstacles,
        obstacle_radius=env_cfg.obstacle_radius,
        goal_tolerance=env_cfg.goal_tolerance,
        collision_distance=env_cfg.collision_distance,
        max_episode_steps=env_cfg.max_episode_steps,
        obs=env_cfg.observation_type,
        act=env_cfg.action_type,
        gui=env_cfg.gui if gui is None else gui,
        pyb_freq=env_cfg.pyb_freq,
        ctrl_freq=env_cfg.ctrl_freq,
        connect_radius=env_cfg.connect_radius,
        workspace_bounds=tuple(tuple(bound) for bound in env_cfg.workspace_bounds),
        obstacle_speed_range=tuple(env_cfg.obstacle_speed_range),
        reward_weights=dict(env_cfg.reward_weights),
        seed=actual_seed,
    )


def create_agent(config: Config, env: DynamicAirspaceEnv) -> GraphPPOAgent:
    """Instantiate the graph PPO agent and make action/device mismatches explicit."""
    if config.agent.action_dim != env.action_dim:
        print(
            f"[INFO] Overriding config.agent.action_dim={config.agent.action_dim} "
            f"with environment action_dim={env.action_dim}"
        )
        config.agent.action_dim = env.action_dim

    requested_device = config.agent.device
    resolved_device = resolve_torch_device(requested_device)
    diagnostic_message = diagnose_torch_device_resolution(requested_device)
    if requested_device != resolved_device:
        print(f"[INFO] Requested agent device '{requested_device}' resolved to '{resolved_device}'")
        config.agent.device = resolved_device
        if diagnostic_message:
            print(f"[INFO] {diagnostic_message}")
    elif requested_device == "auto":
        config.agent.device = resolved_device
        print(f"[INFO] Using torch device '{resolved_device}'")
        if diagnostic_message:
            print(f"[INFO] {diagnostic_message}")

    return GraphPPOAgent(
        encoder_type=config.agent.encoder_type,
        max_nodes=env.max_nodes,
        node_feature_dim=env.node_feature_dim,
        edge_feature_dim=env.edge_feature_dim,
        global_feature_dim=env.global_feature_dim,
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
        device=config.agent.device,
    )
