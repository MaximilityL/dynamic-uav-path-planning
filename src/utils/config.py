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
    auto_time_budget_steps_per_meter: float = 0.0
    auto_time_budget_padding: int = 0
    auto_time_budget_max_steps: int = 0
    scenario_config: Dict[str, Any] = field(default_factory=dict)
    teacher_config: Dict[str, Any] = field(default_factory=dict)
    reward_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "goal": 40.0,
            "progress": 8.0,
            "progress_negative_scale": 1.0,
            "frontier_progress": 0.0,
            "commit_bonus_min_clearance": 0.0,
            "goal_proximity_bonus": 0.0,
            "goal_proximity_radius": 0.0,
            "goal_proximity_power": 1.0,
            "progress_milestone_bonus": 0.0,
            "progress_milestone_thresholds": (),
            "progress_milestone_bonus_weights": (),
            "collision": -40.0,
            "clearance": 0.5,
            "clearance_positive_scale": 0.02,
            "danger_clearance_threshold": 0.0,
            "danger_clearance_penalty": 0.0,
            "danger_clearance_power": 1.0,
            "effort": -0.005,
            "time": -0.01,
            "timeout": -8.0,
            "timeout_distance": -20.0,
            "remaining_distance": 0.0,
            "bypass_clearance_progress": 0.0,
            "route_rejoin_progress": 0.0,
            "stall": 0.0,
            "stall_window_steps": 0,
            "stall_progress_threshold": 0.0,
            "stall_grace_steps": 0,
            "stall_remaining_ratio_threshold": 0.15,
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
    print_interval: int = 10
    resume: Dict[str, Any] = field(default_factory=dict)
    bc_warm_start: Dict[str, Any] = field(default_factory=dict)
    bc_demo_pretrain: Dict[str, Any] = field(default_factory=dict)
    stage_entry_optimizer_reset: Dict[str, Any] = field(default_factory=dict)
    stage_regression_protection: Dict[str, Any] = field(default_factory=dict)
    curriculum: list = field(default_factory=list)


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
    version: str = "1.1.1"
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
            env_dict["scenario_config"] = dict(env_dict.get("scenario_config", {}))
            env_dict["teacher_config"] = dict(env_dict.get("teacher_config", {}))
            payload["environment"] = EnvironmentConfig(**env_dict)
        if "agent" in payload:
            payload["agent"] = AgentConfig(**payload["agent"])
        if "training" in payload:
            training_dict = dict(payload["training"])
            training_dict["resume"] = dict(training_dict.get("resume", {}))
            training_dict["bc_warm_start"] = dict(training_dict.get("bc_warm_start", {}))
            training_dict["bc_demo_pretrain"] = dict(training_dict.get("bc_demo_pretrain", {}))
            training_dict["stage_entry_optimizer_reset"] = dict(training_dict.get("stage_entry_optimizer_reset", {}))
            training_dict["stage_regression_protection"] = dict(training_dict.get("stage_regression_protection", {}))
            training_dict["curriculum"] = list(training_dict.get("curriculum", []))
            payload["training"] = TrainingConfig(**training_dict)
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
    if config.environment.auto_time_budget_steps_per_meter < 0.0:
        raise ValueError("auto_time_budget_steps_per_meter must be non-negative.")
    if config.environment.auto_time_budget_padding < 0:
        raise ValueError("auto_time_budget_padding must be non-negative.")
    if config.environment.auto_time_budget_max_steps < 0:
        raise ValueError("auto_time_budget_max_steps must be non-negative.")
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
    if config.training.print_interval <= 0:
        raise ValueError("training.print_interval must be positive.")
    if not isinstance(config.training.resume, dict):
        raise ValueError("training.resume must be a mapping.")
    if not isinstance(config.training.bc_warm_start, dict):
        raise ValueError("training.bc_warm_start must be a mapping.")
    bc_warm_start = dict(config.training.bc_warm_start)
    if "stages" in bc_warm_start and not isinstance(bc_warm_start["stages"], list):
        raise ValueError("training.bc_warm_start.stages must be a list when provided.")
    if "stage_overrides" in bc_warm_start and not isinstance(bc_warm_start["stage_overrides"], dict):
        raise ValueError("training.bc_warm_start.stage_overrides must be a mapping when provided.")
    if "gate_signal" in bc_warm_start:
        gate_signal = str(bc_warm_start["gate_signal"]).strip().lower()
        if gate_signal not in {"teacher_active", "bypass_active", "repulsion_active"}:
            raise ValueError(
                "training.bc_warm_start.gate_signal must be one of "
                "'teacher_active', 'bypass_active', or 'repulsion_active'."
            )
    for key in ("initial_coef", "final_coef"):
        if key in bc_warm_start and float(bc_warm_start[key]) < 0.0:
            raise ValueError(f"training.bc_warm_start.{key} must be non-negative.")
    if "anneal_episodes" in bc_warm_start and int(bc_warm_start["anneal_episodes"]) < 0:
        raise ValueError("training.bc_warm_start.anneal_episodes must be non-negative.")
    for stage_name, stage_override in dict(bc_warm_start.get("stage_overrides", {}) or {}).items():
        if not isinstance(stage_override, dict):
            raise ValueError(f"training.bc_warm_start.stage_overrides.{stage_name} must be a mapping.")
        if "gate_signal" in stage_override:
            gate_signal = str(stage_override["gate_signal"]).strip().lower()
            if gate_signal not in {"teacher_active", "bypass_active", "repulsion_active"}:
                raise ValueError(
                    f"training.bc_warm_start.stage_overrides.{stage_name}.gate_signal must be one of "
                    "'teacher_active', 'bypass_active', or 'repulsion_active'."
                )
        for key in ("initial_coef", "final_coef"):
            if key in stage_override and float(stage_override[key]) < 0.0:
                raise ValueError(f"training.bc_warm_start.stage_overrides.{stage_name}.{key} must be non-negative.")
        if "anneal_episodes" in stage_override and int(stage_override["anneal_episodes"]) < 0:
            raise ValueError(f"training.bc_warm_start.stage_overrides.{stage_name}.anneal_episodes must be non-negative.")
    if not isinstance(config.training.bc_demo_pretrain, dict):
        raise ValueError("training.bc_demo_pretrain must be a mapping.")
    bc_demo_pretrain = dict(config.training.bc_demo_pretrain)
    if "stages" in bc_demo_pretrain and not isinstance(bc_demo_pretrain["stages"], list):
        raise ValueError("training.bc_demo_pretrain.stages must be a list when provided.")
    if "stage_overrides" in bc_demo_pretrain and not isinstance(bc_demo_pretrain["stage_overrides"], dict):
        raise ValueError("training.bc_demo_pretrain.stage_overrides must be a mapping when provided.")
    if "gate_signal" in bc_demo_pretrain:
        gate_signal = str(bc_demo_pretrain["gate_signal"]).strip().lower()
        if gate_signal not in {"teacher_active", "bypass_active", "repulsion_active"}:
            raise ValueError(
                "training.bc_demo_pretrain.gate_signal must be one of "
                "'teacher_active', 'bypass_active', or 'repulsion_active'."
            )
    for key in ("episodes", "epochs", "batch_size"):
        if key in bc_demo_pretrain and int(bc_demo_pretrain[key]) < 0:
            raise ValueError(f"training.bc_demo_pretrain.{key} must be non-negative.")
    if "eval_episodes" in bc_demo_pretrain and int(bc_demo_pretrain["eval_episodes"]) < 0:
        raise ValueError("training.bc_demo_pretrain.eval_episodes must be non-negative.")
    if "post_pretrain_action_std" in bc_demo_pretrain and float(bc_demo_pretrain["post_pretrain_action_std"]) < 0.0:
        raise ValueError("training.bc_demo_pretrain.post_pretrain_action_std must be non-negative.")
    for stage_name, stage_override in dict(bc_demo_pretrain.get("stage_overrides", {}) or {}).items():
        if not isinstance(stage_override, dict):
            raise ValueError(f"training.bc_demo_pretrain.stage_overrides.{stage_name} must be a mapping.")
        if "gate_signal" in stage_override:
            gate_signal = str(stage_override["gate_signal"]).strip().lower()
            if gate_signal not in {"teacher_active", "bypass_active", "repulsion_active"}:
                raise ValueError(
                    f"training.bc_demo_pretrain.stage_overrides.{stage_name}.gate_signal must be one of "
                    "'teacher_active', 'bypass_active', or 'repulsion_active'."
                )
        for key in ("episodes", "epochs", "batch_size"):
            if key in stage_override and int(stage_override[key]) < 0:
                raise ValueError(
                    f"training.bc_demo_pretrain.stage_overrides.{stage_name}.{key} must be non-negative."
                )
        if "eval_episodes" in stage_override and int(stage_override["eval_episodes"]) < 0:
            raise ValueError(
                f"training.bc_demo_pretrain.stage_overrides.{stage_name}.eval_episodes must be non-negative."
            )
        if "post_pretrain_action_std" in stage_override and float(stage_override["post_pretrain_action_std"]) < 0.0:
            raise ValueError(
                f"training.bc_demo_pretrain.stage_overrides.{stage_name}.post_pretrain_action_std must be non-negative."
            )
    if not isinstance(config.training.stage_entry_optimizer_reset, dict):
        raise ValueError("training.stage_entry_optimizer_reset must be a mapping.")
    stage_entry_optimizer_reset = dict(config.training.stage_entry_optimizer_reset)
    if "stages" in stage_entry_optimizer_reset and not isinstance(stage_entry_optimizer_reset["stages"], list):
        raise ValueError("training.stage_entry_optimizer_reset.stages must be a list when provided.")
    if "stage_overrides" in stage_entry_optimizer_reset and not isinstance(stage_entry_optimizer_reset["stage_overrides"], dict):
        raise ValueError("training.stage_entry_optimizer_reset.stage_overrides must be a mapping when provided.")
    if "lr_multiplier" in stage_entry_optimizer_reset and float(stage_entry_optimizer_reset["lr_multiplier"]) <= 0.0:
        raise ValueError("training.stage_entry_optimizer_reset.lr_multiplier must be positive.")
    for stage_name, stage_override in dict(stage_entry_optimizer_reset.get("stage_overrides", {}) or {}).items():
        if not isinstance(stage_override, dict):
            raise ValueError(f"training.stage_entry_optimizer_reset.stage_overrides.{stage_name} must be a mapping.")
        if "lr_multiplier" in stage_override and float(stage_override["lr_multiplier"]) <= 0.0:
            raise ValueError(
                f"training.stage_entry_optimizer_reset.stage_overrides.{stage_name}.lr_multiplier must be positive."
            )
    if not isinstance(config.training.stage_regression_protection, dict):
        raise ValueError("training.stage_regression_protection must be a mapping.")
    regression_protection = dict(config.training.stage_regression_protection)
    if "stage_overrides" in regression_protection and not isinstance(regression_protection["stage_overrides"], dict):
        raise ValueError("training.stage_regression_protection.stage_overrides must be a mapping when provided.")
    for key in (
        "activate_after_success_rate",
        "absolute_drop_threshold",
        "relative_drop_fraction",
        "lr_multiplier_after_rollback",
        "lr_multiplier_after_plateau_recovery",
    ):
        if key in regression_protection and float(regression_protection[key]) < 0.0:
            raise ValueError(f"training.stage_regression_protection.{key} must be non-negative.")
    for key in ("consecutive_bad_evals", "rollback_max_per_stage", "plateau_bad_eval_streak", "plateau_max_per_stage"):
        if key in regression_protection and int(regression_protection[key]) < 0:
            raise ValueError(f"training.stage_regression_protection.{key} must be non-negative.")
    for stage_name, stage_override in dict(regression_protection.get("stage_overrides", {}) or {}).items():
        if not isinstance(stage_override, dict):
            raise ValueError(f"training.stage_regression_protection.stage_overrides.{stage_name} must be a mapping.")
        for key in (
            "activate_after_success_rate",
            "absolute_drop_threshold",
            "relative_drop_fraction",
            "lr_multiplier_after_rollback",
            "lr_multiplier_after_plateau_recovery",
        ):
            if key in stage_override and float(stage_override[key]) < 0.0:
                raise ValueError(
                    f"training.stage_regression_protection.stage_overrides.{stage_name}.{key} must be non-negative."
                )
        for key in ("consecutive_bad_evals", "rollback_max_per_stage", "plateau_bad_eval_streak", "plateau_max_per_stage"):
            if key in stage_override and int(stage_override[key]) < 0:
                raise ValueError(
                    f"training.stage_regression_protection.stage_overrides.{stage_name}.{key} must be non-negative."
                )
    if not isinstance(config.training.curriculum, list):
        raise ValueError("training.curriculum must be a list.")
    if config.evaluation.num_episodes <= 0:
        raise ValueError("evaluation.num_episodes must be positive.")
