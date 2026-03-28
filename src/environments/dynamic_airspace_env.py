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
        self.position_history: List[np.ndarray] = []
        self.obstacle_history: List[np.ndarray] = []
        self.action_history: List[np.ndarray] = []
        self.reward_history: List[float] = []
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
            workspace_bounds=self.workspace_bounds,
            obstacle_speed_range=self.obstacle_speed_range,
            connect_radius=self.connect_radius,
            current_step=self.current_step,
            max_episode_steps=self.max_episode_steps,
        )

    def _minimum_obstacle_distance(self, drone_position: np.ndarray) -> float:
        """Return the closest obstacle-center distance."""
        return minimum_obstacle_distance(
            drone_position,
            self.obstacle_positions,
            default_distance=self.connect_radius,
        )

    def _episode_info(self, reward_components: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Build a consistent info dictionary."""
        drone_position, _ = self._drone_position_velocity()
        info = {
            "distance_to_goal": float(np.linalg.norm(self.goal_position - drone_position)),
            "min_obstacle_distance": float(self.min_obstacle_distance),
            "goal_reached": float(self.goal_reached),
            "collision": float(self.collision_occurred),
            "episode_return": float(self.episode_return),
            "path_length": float(self.path_length),
            "steps": float(self.current_step),
        }
        if reward_components is not None:
            info["reward_components"] = reward_components
        return info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """Reset the environment and sample a new path-planning scenario."""
        del options
        if seed is not None:
            self._set_seed(seed)

        start_position = sample_start_position(self.rng, self.workspace_bounds)
        self.goal_position = sample_goal_position(self.rng, self.workspace_bounds, start_position)
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
        self.start_to_goal_distance = float(np.linalg.norm(self.goal_position - drone_position))
        self.prev_distance_to_goal = self.start_to_goal_distance
        self.position_history = [drone_position.copy()]
        self.obstacle_history = [self.obstacle_positions.copy()]
        self.action_history = []
        self.reward_history = []

        return self._build_observation(drone_position=drone_position, drone_velocity=np.zeros(3, dtype=np.float32)), self._episode_info()

    def step(self, action: np.ndarray) -> tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, float]]:
        """Advance the PyBullet sim and update moving obstacles."""
        action_array = self._normalize_action(action)
        previous_position, _ = self._drone_position_velocity()
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
        reward, reward_components = compute_reward(
            reward_weights=self.reward_weights,
            goal_now=goal_now,
            collision_now=collision_now,
            progress_delta=progress_delta,
            clearance_margin=clearance_margin,
            action_array=action_array,
        )

        self.prev_distance_to_goal = distance_to_goal
        self.goal_reached = self.goal_reached or goal_now
        self.collision_occurred = self.collision_occurred or collision_now
        self.min_obstacle_distance = min(self.min_obstacle_distance, min_obstacle_distance)
        self.episode_return += reward
        self.reward_history.append(reward)

        terminated = bool(goal_now or collision_now)
        truncated = bool((self.current_step >= self.max_episode_steps) and not terminated)
        observation = self._build_observation(drone_position=drone_position, drone_velocity=drone_velocity)
        return observation, reward, terminated, truncated, self._episode_info(reward_components=reward_components)

    def get_episode_summary(self) -> Dict[str, float]:
        """Return summary metrics for the current episode."""
        return {
            "episode_return": float(self.episode_return),
            "success": float(self.goal_reached),
            "collision": float(self.collision_occurred),
            "min_obstacle_distance": float(self.min_obstacle_distance),
            "path_length": float(self.path_length),
            "start_to_goal_distance": float(self.start_to_goal_distance),
            "steps": float(self.current_step),
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
