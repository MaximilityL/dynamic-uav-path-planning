"""Dynamic airspace environment with moving obstacles and graph observations."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType
from gymnasium import spaces

from .base_env import BaseEnvironment
from .observation import build_dense_graph_observation, minimum_obstacle_distance
from .reward import RewardWeights, compute_reward
from .scenario import advance_obstacles, sample_dynamic_obstacles, sample_goal_position, sample_start_position
from .teacher import heuristic_teacher_action, teacher_alignment_bonus


def _as_action_type(value: ActionType | str) -> ActionType:
    """Normalize action type configuration values."""
    if isinstance(value, ActionType):
        return value
    normalized = str(value).strip().lower()
    for candidate in ActionType:
        if candidate.value == normalized or candidate.name.lower() == normalized:
            return candidate
    raise ValueError(f"Unsupported action type: {value}")


def _as_observation_type(value: ObservationType | str) -> ObservationType:
    """Normalize observation type configuration values."""
    if isinstance(value, ObservationType):
        return value
    normalized = str(value).strip().lower()
    for candidate in ObservationType:
        if candidate.value == normalized or candidate.name.lower() == normalized:
            return candidate
    raise ValueError(f"Unsupported observation type: {value}")


class DynamicAirspaceEnv(BaseEnvironment):
    """PyBullet-backed single-UAV environment with moving obstacle nodes."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        num_agents: int = 1,
        num_dynamic_obstacles: int = 5,
        obstacle_radius: float = 0.2,
        goal_tolerance: float = 0.3,
        collision_distance: float = 0.15,
        max_episode_steps: int = 300,
        obs: ObservationType | str = ObservationType.KIN,
        act: ActionType | str = ActionType.VEL,
        gui: bool = False,
        pyb_freq: int = 240,
        ctrl_freq: int = 30,
        connect_radius: float = 4.0,
        workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
            (-3.0, 3.0),
            (-3.0, 3.0),
            (0.5, 2.5),
        ),
        obstacle_speed_range: Tuple[float, float] = (0.15, 0.45),
        auto_time_budget_steps_per_meter: float = 0.0,
        auto_time_budget_padding: int = 0,
        auto_time_budget_max_steps: int = 0,
        scenario_config: Optional[Dict[str, object]] = None,
        teacher_config: Optional[Dict[str, object]] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = 0,
    ) -> None:
        super().__init__(num_agents=num_agents, max_episode_steps=max_episode_steps)
        if self.num_agents != 1:
            raise ValueError("The bare-minimum scaffold currently supports exactly one controlled UAV.")

        self.num_dynamic_obstacles = int(num_dynamic_obstacles)
        self.obstacle_radius = float(obstacle_radius)
        self.goal_tolerance = float(goal_tolerance)
        self.collision_distance = float(collision_distance)
        self.obs_type = _as_observation_type(obs)
        self.act_type = _as_action_type(act)
        self.gui = bool(gui)
        self.pyb_freq = int(pyb_freq)
        self.ctrl_freq = int(ctrl_freq)
        self.connect_radius = float(connect_radius)
        self.workspace_bounds = np.asarray(workspace_bounds, dtype=np.float32)
        self.obstacle_speed_range = tuple(float(value) for value in obstacle_speed_range)
        self.configured_max_episode_steps = int(max_episode_steps)
        self.auto_time_budget_steps_per_meter = float(auto_time_budget_steps_per_meter)
        self.auto_time_budget_padding = int(auto_time_budget_padding)
        self.auto_time_budget_max_steps = int(auto_time_budget_max_steps)
        self.scenario_config = dict(scenario_config or {})
        self.teacher_config = dict(teacher_config or {})
        self.reward_weights = RewardWeights(**(reward_weights or {}))
        self.seed_value: Optional[int] = None
        self.rng = np.random.default_rng()

        self.max_nodes = 2 + self.num_dynamic_obstacles
        self.node_feature_dim = 10
        self.edge_feature_dim = 4
        self.global_feature_dim = 4

        self.initial_positions = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
        self.goal_position = np.zeros(3, dtype=np.float32)
        self.obstacle_positions = np.zeros((self.num_dynamic_obstacles, 3), dtype=np.float32)
        self.obstacle_velocities = np.zeros((self.num_dynamic_obstacles, 3), dtype=np.float32)
        self.start_to_goal_distance = 0.0
        self.prev_distance_to_goal = 0.0
        self.episode_return = 0.0
        self.path_length = 0.0
        self.goal_reached = False
        self.collision_occurred = False
        self.min_obstacle_distance = float("inf")
        self.min_clearance = float("inf")
        self.cumulative_control_effort = 0.0
        self.time_to_goal: Optional[float] = None
        self.teacher_bonus_budget_used = 0.0
        self.position_history: List[np.ndarray] = []
        self.obstacle_history: List[np.ndarray] = []
        self.action_history: List[np.ndarray] = []
        self.reward_history: List[float] = []
        self.distance_to_goal_history: List[float] = []
        self.best_progress_ratio = 0.0
        self.best_goal_proximity_score = 0.0
        self.progress_milestones_hit: set[int] = set()
        self._debug_body_ids: List[int] = []

        self._set_seed(seed)
        self._init_pybullet_env()
        self._setup_spaces()

    @property
    def action_dim(self) -> int:
        """Per-agent action dimensionality."""
        return int(self.action_space.shape[0])

    def _set_seed(self, seed: Optional[int]) -> None:
        """Set the environment RNG state."""
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(self.seed_value)

    def _init_pybullet_env(self) -> None:
        """Create the wrapped PyBullet drone environment."""
        self.pybullet_env = MultiHoverAviary(
            num_drones=1,
            initial_xyzs=self.initial_positions.copy(),
            obs=self.obs_type,
            act=self.act_type,
            gui=self.gui,
            record=False,
            pyb_freq=self.pyb_freq,
            ctrl_freq=self.ctrl_freq,
        )

    def _setup_spaces(self) -> None:
        """Define the graph observation and per-agent action spaces."""
        base_low = np.asarray(self.pybullet_env.action_space.low, dtype=np.float32)
        base_high = np.asarray(self.pybullet_env.action_space.high, dtype=np.float32)
        if base_low.ndim == 2:
            base_low = base_low[0]
            base_high = base_high[0]

        self.action_space = spaces.Box(low=base_low, high=base_high, dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_nodes, self.node_feature_dim),
                    dtype=np.float32,
                ),
                "node_mask": spaces.Box(low=0.0, high=1.0, shape=(self.max_nodes,), dtype=np.float32),
                "adjacency": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_nodes, self.max_nodes),
                    dtype=np.float32,
                ),
                "edge_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_nodes, self.max_nodes, self.edge_feature_dim),
                    dtype=np.float32,
                ),
                "global_features": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.global_feature_dim,),
                    dtype=np.float32,
                ),
            }
        )

    def _sync_initial_positions_to_backend(self) -> None:
        """Push the start pose into the wrapped PyBullet environment."""
        self.pybullet_env.INIT_XYZS = self.initial_positions.copy()

    def _physics_client_id(self) -> int:
        """Return the active PyBullet client id."""
        return int(getattr(self.pybullet_env, "CLIENT", 0))

    def _create_debug_sphere(self, radius: float, rgba: list[float], position: np.ndarray) -> int:
        """Create a visual-only debug sphere."""
        client_id = self._physics_client_id()
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=rgba,
            physicsClientId=client_id,
        )
        return int(
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=visual_shape,
                basePosition=position.tolist(),
                physicsClientId=client_id,
            )
        )

    def _refresh_debug_bodies(self, create: bool) -> None:
        """Create or update simple goal and obstacle visuals when GUI mode is enabled."""
        if not self.gui:
            return

        if create:
            self._remove_debug_bodies()
            self._debug_body_ids.append(
                self._create_debug_sphere(self.goal_tolerance, [0.1, 0.8, 0.2, 0.7], self.goal_position)
            )
            for position in self.obstacle_positions:
                self._debug_body_ids.append(
                    self._create_debug_sphere(self.obstacle_radius, [0.9, 0.2, 0.2, 0.7], position)
                )
            return

        for index, body_id in enumerate(self._debug_body_ids):
            position = self.goal_position if index == 0 else self.obstacle_positions[index - 1]
            p.resetBasePositionAndOrientation(
                bodyUniqueId=body_id,
                posObj=position.tolist(),
                ornObj=[0.0, 0.0, 0.0, 1.0],
                physicsClientId=self._physics_client_id(),
            )

    def _remove_debug_bodies(self) -> None:
        """Remove any visual-only helper bodies."""
        for body_id in self._debug_body_ids:
            try:
                p.removeBody(body_id, physicsClientId=self._physics_client_id())
            except Exception:
                continue
        self._debug_body_ids = []

    def _drone_position_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Read the current drone state from the simulator."""
        state = np.asarray(self.pybullet_env._getDroneStateVector(0), dtype=np.float32)
        return state[0:3].copy(), state[10:13].copy()

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Validate and clip a per-agent action."""
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action_array.shape}")
        return np.clip(action_array, self.action_space.low, self.action_space.high).astype(np.float32)

    def _build_observation(
        self,
        drone_position: Optional[np.ndarray] = None,
        drone_velocity: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Construct the dense graph observation expected by the policy."""
        if drone_position is None or drone_velocity is None:
            drone_position, drone_velocity = self._drone_position_velocity()

        return build_dense_graph_observation(
            drone_position=drone_position,
            drone_velocity=drone_velocity,
            goal_position=self.goal_position,
            obstacle_positions=self.obstacle_positions,
            obstacle_velocities=self.obstacle_velocities,
            obstacle_radius=self.obstacle_radius,
            goal_tolerance=self.goal_tolerance,
            collision_distance=self.collision_distance,
            workspace_bounds=self.workspace_bounds,
            obstacle_speed_range=self.obstacle_speed_range,
            connect_radius=self.connect_radius,
            current_step=self.current_step,
            max_episode_steps=self.max_episode_steps,
        )

    def _resolve_episode_step_budget(self, start_to_goal_distance: float) -> int:
        """Expand the episode horizon when a stage requests an automatic time budget."""
        if self.auto_time_budget_steps_per_meter <= 0.0:
            return self.configured_max_episode_steps

        scaled_budget = int(np.ceil(start_to_goal_distance * self.auto_time_budget_steps_per_meter))
        resolved_budget = max(self.configured_max_episode_steps, scaled_budget + self.auto_time_budget_padding)
        if self.auto_time_budget_max_steps > 0:
            resolved_budget = min(resolved_budget, self.auto_time_budget_max_steps)
        return max(resolved_budget, 1)

    def _stall_penalty(self, *, distance_to_goal: float, goal_now: bool) -> float:
        """Penalize late-episode plateaus that fail to make enough net progress."""
        if goal_now:
            return 0.0

        stall_weight = float(self.reward_weights.stall)
        stall_window = max(int(self.reward_weights.stall_window_steps), 0)
        stall_grace_steps = max(int(self.reward_weights.stall_grace_steps), 0)
        stall_progress_threshold = float(self.reward_weights.stall_progress_threshold)
        stall_remaining_ratio_threshold = float(self.reward_weights.stall_remaining_ratio_threshold)
        if stall_weight == 0.0 or stall_window <= 0 or stall_progress_threshold <= 0.0:
            return 0.0
        if self.current_step <= max(stall_window, stall_grace_steps):
            return 0.0
        if len(self.distance_to_goal_history) <= stall_window:
            return 0.0

        remaining_ratio = float(distance_to_goal / max(self.start_to_goal_distance, 1e-6))
        if remaining_ratio < stall_remaining_ratio_threshold:
            return 0.0

        window_start_distance = float(self.distance_to_goal_history[-(stall_window + 1)])
        window_progress = window_start_distance - distance_to_goal
        if window_progress >= stall_progress_threshold:
            return 0.0

        shortfall_ratio = float(
            np.clip((stall_progress_threshold - window_progress) / max(stall_progress_threshold, 1e-6), 0.0, 1.5)
        )
        return stall_weight * shortfall_ratio

    def _commit_bonus_allowed(self, *, clearance_margin: float) -> bool:
        """Allow forward-commit bonuses only when the state is above a minimum safety margin."""
        minimum_clearance = float(getattr(self.reward_weights, "commit_bonus_min_clearance", 0.0))
        return float(clearance_margin) >= minimum_clearance

    def _progress_milestone_bonus(self, *, distance_to_goal: float, clearance_margin: float) -> float:
        """Return one-time bonuses when the episode crosses configured progress thresholds."""
        if not self._commit_bonus_allowed(clearance_margin=clearance_margin):
            return 0.0
        threshold_values = getattr(self.reward_weights, "progress_milestone_thresholds", ())
        if isinstance(threshold_values, (int, float)):
            thresholds = [float(threshold_values)]
        else:
            thresholds = [float(value) for value in threshold_values]
        if not thresholds:
            return 0.0

        progress_ratio = float(
            np.clip(1.0 - distance_to_goal / max(self.start_to_goal_distance, 1e-6), 0.0, 1.2)
        )
        bonus_weights = list(getattr(self.reward_weights, "progress_milestone_bonus_weights", ()) or ())
        default_bonus = float(getattr(self.reward_weights, "progress_milestone_bonus", 0.0))
        total_bonus = 0.0
        for index, threshold in enumerate(sorted(thresholds)):
            threshold = float(np.clip(threshold, 0.0, 1.0))
            if index in self.progress_milestones_hit or progress_ratio < threshold:
                continue
            bonus = float(bonus_weights[index]) if index < len(bonus_weights) else default_bonus
            if bonus == 0.0:
                self.progress_milestones_hit.add(index)
                continue
            total_bonus += bonus
            self.progress_milestones_hit.add(index)
        return total_bonus

    def _frontier_progress_bonus(self, *, distance_to_goal: float, clearance_margin: float) -> float:
        """Reward the episode for extending its furthest committed progress toward the goal."""
        frontier_weight = float(getattr(self.reward_weights, "frontier_progress", 0.0))
        if frontier_weight == 0.0:
            return 0.0
        if not self._commit_bonus_allowed(clearance_margin=clearance_margin):
            return 0.0

        progress_ratio = float(
            np.clip(1.0 - distance_to_goal / max(self.start_to_goal_distance, 1e-6), 0.0, 1.2)
        )
        frontier_delta = max(progress_ratio - float(self.best_progress_ratio), 0.0)
        self.best_progress_ratio = max(float(self.best_progress_ratio), progress_ratio)
        if frontier_delta <= 0.0:
            return 0.0
        return frontier_weight * frontier_delta

    def _goal_proximity_bonus(self, *, distance_to_goal: float, clearance_margin: float) -> float:
        """Reward improved near-goal closeness without paying repeatedly for hovering."""
        proximity_weight = float(getattr(self.reward_weights, "goal_proximity_bonus", 0.0))
        proximity_radius = float(getattr(self.reward_weights, "goal_proximity_radius", 0.0))
        if proximity_weight == 0.0 or proximity_radius <= 0.0:
            return 0.0
        if not self._commit_bonus_allowed(clearance_margin=clearance_margin):
            return 0.0

        proximity_score = 1.0 - float(distance_to_goal) / max(proximity_radius, 1e-6)
        proximity_score = float(np.clip(proximity_score, 0.0, 1.0))
        if proximity_score <= 0.0:
            return 0.0

        proximity_power = float(max(getattr(self.reward_weights, "goal_proximity_power", 1.0), 1e-6))
        shaped_score = proximity_score**proximity_power
        score_delta = max(shaped_score - float(self.best_goal_proximity_score), 0.0)
        self.best_goal_proximity_score = max(float(self.best_goal_proximity_score), shaped_score)
        if score_delta <= 0.0:
            return 0.0
        return proximity_weight * score_delta

    def _teacher_action(self, drone_position: np.ndarray) -> np.ndarray:
        """Return the heuristic teacher action for the current state."""
        return heuristic_teacher_action(
            drone_position=drone_position,
            goal_position=self.goal_position,
            obstacle_positions=self.obstacle_positions,
            action_low=self.action_space.low,
            action_high=self.action_space.high,
            teacher_config=self.teacher_config,
        )

    def teacher_action_for_current_state(self) -> np.ndarray:
        """Return teacher guidance for the current simulator state."""
        drone_position, _ = self._drone_position_velocity()
        return self._teacher_action(drone_position)

    def _minimum_obstacle_distance(self, drone_position: np.ndarray) -> float:
        """Return the closest obstacle-center distance."""
        return minimum_obstacle_distance(
            drone_position,
            self.obstacle_positions,
            default_distance=self.connect_radius,
        )

    def _episode_info(self, reward_components: Optional[Dict[str, float]] = None) -> Dict[str, object]:
        """Build a consistent info dictionary."""
        drone_position, _ = self._drone_position_velocity()
        info = {
            "distance_to_goal": float(np.linalg.norm(self.goal_position - drone_position)),
            "min_obstacle_distance": float(self.min_obstacle_distance),
            "min_clearance": float(self.min_clearance),
            "goal_reached": float(self.goal_reached),
            "collision": float(self.collision_occurred),
            "episode_return": float(self.episode_return),
            "path_length": float(self.path_length),
            "steps": float(self.current_step),
            "control_effort": float(self.cumulative_control_effort / max(self.current_step, 1)),
            "episode_duration": float(self.current_step / max(self.ctrl_freq, 1)),
            "time_to_goal": None if self.time_to_goal is None else float(self.time_to_goal),
        }
        if reward_components is not None:
            info["reward_components"] = reward_components
        return info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Reset the environment and sample a new path-planning scenario."""
        del options
        if seed is not None:
            self._set_seed(seed)

        start_position = sample_start_position(self.rng, self.workspace_bounds, self.scenario_config)
        self.goal_position = sample_goal_position(
            self.rng,
            self.workspace_bounds,
            start_position,
            self.scenario_config,
        )
        self.obstacle_positions, self.obstacle_velocities = sample_dynamic_obstacles(
            self.rng,
            self.workspace_bounds,
            num_dynamic_obstacles=self.num_dynamic_obstacles,
            obstacle_radius=self.obstacle_radius,
            goal_tolerance=self.goal_tolerance,
            collision_distance=self.collision_distance,
            obstacle_speed_range=self.obstacle_speed_range,
            start_position=start_position,
            goal_position=self.goal_position,
            scenario_config=self.scenario_config,
        )
        self.initial_positions = start_position.reshape(1, 3).astype(np.float32)
        self._sync_initial_positions_to_backend()
        self.pybullet_env.reset(seed=self.seed_value)
        self._refresh_debug_bodies(create=True)

        drone_position, _ = self._drone_position_velocity()
        self.current_step = 0
        self.episode_return = 0.0
        self.path_length = 0.0
        self.goal_reached = False
        self.collision_occurred = False
        self.min_obstacle_distance = self._minimum_obstacle_distance(drone_position)
        self.min_clearance = self.min_obstacle_distance - (self.obstacle_radius + self.collision_distance)
        self.cumulative_control_effort = 0.0
        self.time_to_goal = None
        self.teacher_bonus_budget_used = 0.0
        self.start_to_goal_distance = float(np.linalg.norm(self.goal_position - drone_position))
        self.max_episode_steps = self._resolve_episode_step_budget(self.start_to_goal_distance)
        self.prev_distance_to_goal = self.start_to_goal_distance
        self.distance_to_goal_history = [self.start_to_goal_distance]
        self.best_progress_ratio = 0.0
        self.best_goal_proximity_score = 0.0
        self.progress_milestones_hit = set()
        self.position_history = [drone_position.copy()]
        self.obstacle_history = [self.obstacle_positions.copy()]
        self.action_history = []
        self.reward_history = []

        return self._build_observation(drone_position=drone_position, drone_velocity=np.zeros(3, dtype=np.float32)), self._episode_info()

    def step(self, action: np.ndarray) -> tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, float]]:
        """Advance the PyBullet sim and update moving obstacles."""
        action_array = self._normalize_action(action)
        previous_position, _ = self._drone_position_velocity()
        teacher_action = self._teacher_action(previous_position)
        self.pybullet_env.step(action_array.reshape(1, -1))
        self.obstacle_positions, self.obstacle_velocities = advance_obstacles(
            self.obstacle_positions,
            self.obstacle_velocities,
            workspace_bounds=self.workspace_bounds,
            obstacle_radius=self.obstacle_radius,
            dt=1.0 / max(self.ctrl_freq, 1),
        )
        self._refresh_debug_bodies(create=False)

        drone_position, drone_velocity = self._drone_position_velocity()
        self.current_step += 1
        self.path_length += float(np.linalg.norm(drone_position - previous_position))
        self.cumulative_control_effort += float(np.linalg.norm(action_array) / np.sqrt(max(action_array.size, 1)))
        self.position_history.append(drone_position.copy())
        self.obstacle_history.append(self.obstacle_positions.copy())
        self.action_history.append(action_array.copy())

        distance_to_goal = float(np.linalg.norm(self.goal_position - drone_position))
        min_obstacle_distance = self._minimum_obstacle_distance(drone_position)
        safety_distance = self.obstacle_radius + self.collision_distance
        progress_delta = self.prev_distance_to_goal - distance_to_goal
        collision_now = bool(min_obstacle_distance <= safety_distance)
        goal_now = bool(distance_to_goal <= self.goal_tolerance)
        clearance_margin = min_obstacle_distance - safety_distance
        remaining_ratio = float(distance_to_goal / max(self.start_to_goal_distance, 1e-6))
        reward, reward_components = compute_reward(
            reward_weights=self.reward_weights,
            goal_now=goal_now,
            collision_now=collision_now,
            progress_delta=progress_delta,
            clearance_margin=clearance_margin,
            action_array=action_array,
            remaining_distance_ratio=remaining_ratio,
        )
        self.distance_to_goal_history.append(distance_to_goal)
        frontier_term = self._frontier_progress_bonus(distance_to_goal=distance_to_goal, clearance_margin=clearance_margin)
        if frontier_term != 0.0:
            reward += frontier_term
            reward_components["frontier_progress"] = frontier_term
        proximity_term = self._goal_proximity_bonus(distance_to_goal=distance_to_goal, clearance_margin=clearance_margin)
        if proximity_term != 0.0:
            reward += proximity_term
            reward_components["goal_proximity"] = proximity_term
        milestone_term = self._progress_milestone_bonus(distance_to_goal=distance_to_goal, clearance_margin=clearance_margin)
        if milestone_term != 0.0:
            reward += milestone_term
            reward_components["milestone"] = milestone_term
        stall_term = self._stall_penalty(distance_to_goal=distance_to_goal, goal_now=goal_now)
        if stall_term != 0.0:
            reward += stall_term
            reward_components["stall"] = stall_term
        teacher_bonus = teacher_alignment_bonus(
            action_array=action_array,
            teacher_action=teacher_action,
            action_low=self.action_space.low,
            action_high=self.action_space.high,
            teacher_config=self.teacher_config,
        )
        teacher_bonus_budget = float(self.teacher_config.get("episode_bonus_budget", 0.0))
        if teacher_bonus != 0.0 and teacher_bonus_budget > 0.0:
            remaining_budget = max(teacher_bonus_budget - self.teacher_bonus_budget_used, 0.0)
            if remaining_budget <= 0.0:
                teacher_bonus = 0.0
            else:
                teacher_bonus = float(np.clip(teacher_bonus, -remaining_budget, remaining_budget))
                self.teacher_bonus_budget_used += abs(teacher_bonus)
        if teacher_bonus != 0.0:
            reward += teacher_bonus
            reward_components["teacher"] = teacher_bonus

        self.prev_distance_to_goal = distance_to_goal
        self.goal_reached = self.goal_reached or goal_now
        self.collision_occurred = self.collision_occurred or collision_now
        self.min_obstacle_distance = min(self.min_obstacle_distance, min_obstacle_distance)
        self.min_clearance = min(self.min_clearance, clearance_margin)
        self.episode_return += reward
        self.reward_history.append(reward)
        if goal_now and self.time_to_goal is None:
            self.time_to_goal = float(self.current_step / max(self.ctrl_freq, 1))

        terminated = bool(goal_now or collision_now)
        truncated = bool((self.current_step >= self.max_episode_steps) and not terminated)
        if truncated:
            remaining_ratio = float(np.clip(distance_to_goal / max(self.start_to_goal_distance, 1e-6), 0.0, 1.5))
            timeout_term = float(self.reward_weights.timeout)
            timeout_distance_term = float(self.reward_weights.timeout_distance * remaining_ratio)
            reward += timeout_term + timeout_distance_term
            reward_components["timeout"] = timeout_term
            reward_components["timeout_distance"] = timeout_distance_term
            self.episode_return += timeout_term + timeout_distance_term
            self.reward_history[-1] = reward
        observation = self._build_observation(drone_position=drone_position, drone_velocity=drone_velocity)
        return observation, reward, terminated, truncated, self._episode_info(reward_components=reward_components)

    def get_episode_summary(self) -> Dict[str, object]:
        """Return summary metrics for the current episode."""
        return {
            "episode_return": float(self.episode_return),
            "success": float(self.goal_reached),
            "collision": float(self.collision_occurred),
            "min_obstacle_distance": float(self.min_obstacle_distance),
            "min_clearance": float(self.min_clearance),
            "path_length": float(self.path_length),
            "start_to_goal_distance": float(self.start_to_goal_distance),
            "steps": float(self.current_step),
            "control_effort": float(self.cumulative_control_effort / max(self.current_step, 1)),
            "episode_duration": float(self.current_step / max(self.ctrl_freq, 1)),
            "time_to_goal": None if self.time_to_goal is None else float(self.time_to_goal),
        }

    def export_episode(self) -> Dict[str, object]:
        """Export raw episode arrays for later plotting or replay generation."""
        return {
            "positions": np.asarray(self.position_history, dtype=np.float32),
            "obstacles": np.asarray(self.obstacle_history, dtype=np.float32),
            "goal": np.asarray(self.goal_position, dtype=np.float32),
            "actions": np.asarray(self.action_history, dtype=np.float32),
            "rewards": np.asarray(self.reward_history, dtype=np.float32),
            "summary": self.get_episode_summary(),
        }

    def close(self) -> None:
        """Release PyBullet resources."""
        self._remove_debug_bodies()
        self.pybullet_env.close()
