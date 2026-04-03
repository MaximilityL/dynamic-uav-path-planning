"""Configuration helpers for the dynamic UAV path-planning scaffold."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
import yaml


def _as_tuple(value: Any) -> Any:
    """Recursively convert lists into tuples for config stability."""
    if isinstance(value, list):
        return tuple(_as_tuple(item) for item in value)
    if isinstance(value, dict):
        return {key: _as_tuple(item) for key, item in value.items()}
    return value


@dataclass
class EnvironmentConfig:
    """Environment configuration."""

    type: str = "dynamic_airspace"
    backend: str = "gym_pybullet_drones"
    num_agents: int = 1
    num_dynamic_obstacles: int = 5
    obstacle_radius: float = 0.2
    goal_tolerance: float = 0.35
    collision_distance: float = 0.15
    max_episode_steps: int = 800
    gui: bool = False
    observation_type: str = "kin"
    action_type: str = "vel"
    pyb_freq: int = 240
    ctrl_freq: int = 30
    seed: int = 7
    connect_radius: float = 5.0
    workspace_bounds: tuple = ((-3.0, 3.0), (-3.0, 3.0), (0.5, 2.5))
    obstacle_speed_range: tuple = (0.12, 0.35)
    reward_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "goal": 40.0,
            "progress": 8.0,
            "collision": -40.0,
            "clearance": 0.5,
            "effort": -0.005,
            "time": -0.01,
        }
    )


@dataclass
class AgentConfig:
    """Agent configuration."""

    type: str = "graph_ppo"
    encoder_type: str = "gnn"
    node_feature_dim: int = 10
    edge_feature_dim: int = 4
    global_feature_dim: int = 4
    action_dim: int = 4
    hidden_dim: int = 128
    message_passing_steps: int = 3
    lr_actor: float = 3e-4
    lr_critic: float = 6e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    ppo_epochs: int = 4
    mini_batch_size: int = 128
    rollout_episodes: int = 8
    max_grad_norm: float = 0.5
    action_std_init: float = 0.45
    device: str = "auto"


@dataclass
class TrainingConfig:
    """Training configuration."""

    num_episodes: int = 1000
    save_interval: int = 25
    eval_interval: int = 25
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"
    moving_average_window: int = 20
    eval_episodes: int = 10


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""

    num_episodes: int = 20
    render: bool = False
    deterministic: bool = True
    save_trajectories: bool = True
    output_dir: str = "results/eval"


@dataclass
class VisualizationConfig:
    """Visualization configuration."""

    plot_dir: str = "results/plots"
    trajectory_dir: str = "results/trajectories"


@dataclass
class Config:
    """Main configuration bundle."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    name: str = "dynamic_uav_path_planning"
    version: str = "1.0.0"
    description: str = "Main PPO+GNN config for single-UAV dynamic obstacle avoidance"
    tags: list = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Construct the config from a dictionary."""
        payload = dict(config_dict)
        if "environment" in payload:
            env_dict = dict(payload["environment"])
            env_dict["workspace_bounds"] = _as_tuple(env_dict.get("workspace_bounds", ((-3.0, 3.0), (-3.0, 3.0), (0.5, 2.5))))
            env_dict["obstacle_speed_range"] = _as_tuple(env_dict.get("obstacle_speed_range", (0.12, 0.35)))
            payload["environment"] = EnvironmentConfig(**env_dict)
        if "agent" in payload:
            payload["agent"] = AgentConfig(**payload["agent"])
        if "training" in payload:
            payload["training"] = TrainingConfig(**payload["training"])
        if "evaluation" in payload:
            payload["evaluation"] = EvaluationConfig(**payload["evaluation"])
        if "visualization" in payload:
            payload["visualization"] = VisualizationConfig(**payload["visualization"])
        return cls(**payload)


def load_config(config_path: Union[str, Path]) -> Config:
    """Load configuration from YAML or JSON."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
        elif path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")

    return Config.from_dict(data)


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """Save configuration to disk."""
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
        elif path.suffix.lower() == ".json":
            json.dump(config.to_dict(), handle, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {path.suffix}")


def resolve_torch_device(requested: Optional[str]) -> str:
    """Resolve a torch device string in a predictable way."""
    normalized = (requested or "auto").strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return normalized


def diagnose_torch_device_resolution(requested: Optional[str]) -> Optional[str]:
    """Explain why a requested device resolved to a fallback."""
    normalized = (requested or "auto").strip().lower()
    if normalized == "auto" and not torch.cuda.is_available():
        return "CUDA is unavailable, so the scaffold will run on CPU."
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        return f"Requested device '{normalized}' is unavailable, so the scaffold will run on CPU."
    return None


def validate_config(config: Config) -> None:
    """Validate the minimum assumptions of the current scaffold."""
    if config.environment.type != "dynamic_airspace":
        raise ValueError(f"Unsupported environment type: {config.environment.type}")
    if config.environment.num_agents != 1:
        raise ValueError("The bare-minimum scaffold currently supports exactly one controlled UAV.")
    if config.environment.num_dynamic_obstacles < 0:
        raise ValueError("num_dynamic_obstacles must be non-negative.")
    if config.environment.goal_tolerance <= 0.0:
        raise ValueError("goal_tolerance must be positive.")
    if config.environment.collision_distance <= 0.0:
        raise ValueError("collision_distance must be positive.")
    if config.environment.max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be positive.")
    if config.environment.connect_radius <= 0.0:
        raise ValueError("connect_radius must be positive.")
    if len(config.environment.obstacle_speed_range) != 2:
        raise ValueError("obstacle_speed_range must contain exactly two values.")
    if config.environment.obstacle_speed_range[0] < 0.0:
        raise ValueError("obstacle_speed_range values must be non-negative.")
    if config.environment.obstacle_speed_range[1] < config.environment.obstacle_speed_range[0]:
        raise ValueError("obstacle_speed_range max must be >= min.")
    if config.agent.encoder_type not in {"gnn", "mlp"}:
        raise ValueError("encoder_type must be either 'gnn' or 'mlp'.")
    if config.agent.action_dim <= 0:
        raise ValueError("action_dim must be positive.")
    if config.agent.message_passing_steps < 1:
        raise ValueError("message_passing_steps must be >= 1.")
    if config.training.num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")
    if config.training.save_interval <= 0:
        raise ValueError("save_interval must be positive.")
    if config.training.eval_interval <= 0:
        raise ValueError("eval_interval must be positive.")
    if config.training.moving_average_window <= 0:
        raise ValueError("moving_average_window must be positive.")
    if config.training.eval_episodes <= 0:
        raise ValueError("training.eval_episodes must be positive.")
    if config.evaluation.num_episodes <= 0:
        raise ValueError("evaluation.num_episodes must be positive.")
