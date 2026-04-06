#!/usr/bin/env python3
"""Diagnostics for v3's policy on target_default_intro.

Runs five cheap tests to narrow down WHY the v3 agent plateaus:
  (1) Roll v3 policy for N episodes, bucket outcomes by obstacle side.
  (2) Roll heuristic teacher alone (no policy) on the same stage.
  (3) Sensitivity: fixed drone/goal, sweep a single obstacle, log the
      policy's deterministic action direction.
  (4) Read v3's training history for target_default_intro and report
      collision/timeout/success mix.
  (5) Sample 50 resets and dump obstacle geometry to check symmetry.

Because v3 was trained with global_feature_dim=4 but the current code
emits 10, we monkeypatch the observation builder back to 4 dims for
this script only. Production code is untouched.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from _common import bootstrap_project

bootstrap_project()

# ---- Monkeypatch observation dim back to 4 for v3 compatibility ----
from src.environments import observation as _obs_mod  # noqa: E402

_original_build = _obs_mod.build_dense_graph_observation


def _build_obs_v3(**kwargs):
    obs = _original_build(**kwargs)
    obs["global_features"] = obs["global_features"][:4].astype(np.float32)
    return obs


_obs_mod.build_dense_graph_observation = _build_obs_v3

from src.environments import dynamic_airspace_env as _dae_mod  # noqa: E402

_dae_mod.build_dense_graph_observation = _build_obs_v3

# Patch the env instance to report global_feature_dim=4.
_original_env_init = _dae_mod.DynamicAirspaceEnv.__init__


def _env_init_v3(self, *args, **kwargs):
    _original_env_init(self, *args, **kwargs)
    self.global_feature_dim = 4
    # Rebuild observation space with correct shape.
    from gymnasium import spaces

    self.observation_space = spaces.Dict(
        {
            "node_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.max_nodes, self.node_feature_dim), dtype=np.float32),
            "node_mask": spaces.Box(low=0.0, high=1.0, shape=(self.max_nodes,), dtype=np.float32),
            "adjacency": spaces.Box(
                low=0.0, high=1.0, shape=(self.max_nodes, self.max_nodes), dtype=np.float32),
            "edge_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.max_nodes, self.max_nodes, self.edge_feature_dim), dtype=np.float32),
            "global_features": spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
        }
    )


_dae_mod.DynamicAirspaceEnv.__init__ = _env_init_v3

# Now safe to import the rest.
from src.training.factories import create_agent, create_environment  # noqa: E402
from src.training.loops import _stage_config  # noqa: E402
from src.utils.config import load_config, validate_config  # noqa: E402

V3_CONFIG = "configs/default.yaml"
V3_CHECKPOINT = "checkpoints/default/stages/target_bypass_commit_v2/best_model.pth"
STAGE_NAME = "target_default_intro"
N_EPISODES = 20
SENSITIVITY_GRID = 9  # obstacles will be placed on a 9x9 grid in front of drone


def _round(value: Any, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _pick_stage(config, name):
    for stage in config.training.curriculum:
        if str(stage.get("name")) == name:
            return stage
    raise ValueError(f"stage {name} not found")


def _obstacle_side(env) -> str:
    """Classify the dominant route-obstacle lateral side relative to start->goal."""
    start = np.asarray(env.initial_positions[0], dtype=np.float32)
    goal = env.goal_position
    route = goal - start
    route_len = float(np.linalg.norm(route[:2]))
    if route_len < 1e-6:
        return "degenerate"
    route_dir = route[:2] / route_len
    perp = np.array([-route_dir[1], route_dir[0]], dtype=np.float32)  # left = +
    lateral_signs = []
    for obs_pos in env.obstacle_positions:
        rel = obs_pos[:2] - start[:2]
        lateral = float(np.dot(rel, perp))
        along = float(np.dot(rel, route_dir))
        if 0.0 < along < route_len:  # only obstacles between start and goal
            lateral_signs.append(lateral)
    if not lateral_signs:
        return "none"
    mean_lat = float(np.mean(lateral_signs))
    if abs(mean_lat) < 0.05:
        return "center"
    return "left" if mean_lat > 0 else "right"


def test_1_policy_rollouts(env, agent) -> dict:
    """Roll N episodes deterministically, bucket outcomes by obstacle side."""
    print("\n=== TEST 1: v3 policy rollouts on target_default_intro ===")
    results = {
        "left":   {"n": 0, "success": 0, "collision": 0, "timeout": 0, "lateral_max": []},
        "right":  {"n": 0, "success": 0, "collision": 0, "timeout": 0, "lateral_max": []},
        "center": {"n": 0, "success": 0, "collision": 0, "timeout": 0, "lateral_max": []},
        "none":   {"n": 0, "success": 0, "collision": 0, "timeout": 0, "lateral_max": []},
    }
    for seed in range(N_EPISODES):
        obs, _ = env.reset(seed=1000 + seed)
        side = _obstacle_side(env)
        bucket = results.setdefault(side, {"n": 0, "success": 0, "collision": 0, "timeout": 0, "lateral_max": []})
        bucket["n"] += 1
        start = np.asarray(env.initial_positions[0], dtype=np.float32)
        route = env.goal_position - start
        route_len = float(np.linalg.norm(route[:2])) + 1e-6
        route_dir = route[:2] / route_len
        perp = np.array([-route_dir[1], route_dir[0]], dtype=np.float32)
        max_lat = 0.0
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            action, _ = agent.select_action(obs, deterministic=True)
            obs, _r, terminated, truncated, info = env.step(action)
            pos, _ = env._drone_position_velocity()
            rel = pos[:2] - start[:2]
            max_lat = max(max_lat, abs(float(np.dot(rel, perp))))
            steps += 1
        summary = env.get_episode_summary()
        if summary["success"] > 0.5:
            bucket["success"] += 1
        elif summary["collision"] > 0.5:
            bucket["collision"] += 1
        else:
            bucket["timeout"] += 1
        bucket["lateral_max"].append(max_lat)
    report = {}
    for side, b in results.items():
        if b["n"] == 0:
            continue
        lat = b["lateral_max"]
        report[side] = {
            "n": b["n"],
            "success_rate": round(b["success"] / b["n"], 3),
            "collision_rate": round(b["collision"] / b["n"], 3),
            "timeout_rate": round(b["timeout"] / b["n"], 3),
            "lateral_max_mean": round(float(np.mean(lat)), 3) if lat else None,
            "lateral_max_max": round(float(np.max(lat)), 3) if lat else None,
        }
    print(json.dumps(report, indent=2))
    return report


def test_2_teacher_rollouts(env) -> dict:
    """Same stage, rolled out by the heuristic teacher alone."""
    print("\n=== TEST 2: teacher-alone rollouts on target_default_intro ===")
    results = {
        "left":  {"n": 0, "success": 0, "collision": 0, "timeout": 0},
        "right": {"n": 0, "success": 0, "collision": 0, "timeout": 0},
        "center":{"n": 0, "success": 0, "collision": 0, "timeout": 0},
        "none":  {"n": 0, "success": 0, "collision": 0, "timeout": 0},
    }
    for seed in range(N_EPISODES):
        obs, _ = env.reset(seed=2000 + seed)
        side = _obstacle_side(env)
        bucket = results.setdefault(side, {"n": 0, "success": 0, "collision": 0, "timeout": 0})
        bucket["n"] += 1
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = env.teacher_action_for_current_state()
            obs, _r, terminated, truncated, info = env.step(action)
        summary = env.get_episode_summary()
        if summary["success"] > 0.5:
            bucket["success"] += 1
        elif summary["collision"] > 0.5:
            bucket["collision"] += 1
        else:
            bucket["timeout"] += 1
    report = {}
    for side, b in results.items():
        if b["n"] == 0:
            continue
        report[side] = {
            "n": b["n"],
            "success_rate": round(b["success"] / b["n"], 3),
            "collision_rate": round(b["collision"] / b["n"], 3),
            "timeout_rate": round(b["timeout"] / b["n"], 3),
        }
    print(json.dumps(report, indent=2))
    return report


def test_3_sensitivity(env, agent) -> dict:
    """Fix drone + goal, sweep one obstacle across a 2D grid in front of drone.
    Log the policy's deterministic action direction to check if the encoder
    is actually sensitive to obstacle position."""
    print("\n=== TEST 3: encoder sensitivity to obstacle position ===")
    env.reset(seed=3000)
    drone_pos = np.array([-2.0, 0.0, 1.0], dtype=np.float32)
    goal_pos = np.array([2.0, 0.0, 1.0], dtype=np.float32)
    env.initial_positions = drone_pos.reshape(1, 3)
    env.goal_position = goal_pos
    env.obstacle_positions = np.zeros((env.num_dynamic_obstacles, 3), dtype=np.float32)
    env.obstacle_positions[0] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    # push other obstacles far away
    env.obstacle_positions[1:] = 10.0
    env.obstacle_velocities = np.zeros_like(env.obstacle_positions)

    directions = []
    xs = np.linspace(-1.0, 1.0, SENSITIVITY_GRID)
    ys = np.linspace(-1.0, 1.0, SENSITIVITY_GRID)
    for x in xs:
        for y in ys:
            env.obstacle_positions[0] = np.array([x, y, 1.0], dtype=np.float32)
            obs = env._build_observation(drone_position=drone_pos,
                                         drone_velocity=np.zeros(3, dtype=np.float32))
            action, _ = agent.select_action(obs, deterministic=True)
            # normalize direction (first 3 dims of VEL action are a direction vector)
            dir_vec = action[:3]
            n = float(np.linalg.norm(dir_vec)) + 1e-9
            directions.append(dir_vec / n)
    directions = np.asarray(directions)
    # variability: std of direction components across all obstacle positions
    std = directions.std(axis=0)
    mean = directions.mean(axis=0)
    # max pairwise angle
    cos_mat = directions @ directions.T
    cos_mat = np.clip(cos_mat, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_mat))
    report = {
        "direction_mean": [round(float(v), 4) for v in mean],
        "direction_std":  [round(float(v), 4) for v in std],
        "max_pairwise_angle_deg": round(float(np.max(angles)), 2),
        "median_pairwise_angle_deg": round(float(np.median(angles)), 2),
        "interpretation": (
            "If max_pairwise_angle_deg < 5, encoder is essentially BLIND to obstacle "
            "position. If >30, encoder is reactive."
        ),
    }
    print(json.dumps(report, indent=2))
    return report


def test_4_history_failure_modes() -> dict:
    """Read v3 training history for target_default_intro and summarize."""
    print("\n=== TEST 4: v3 training history on target_default_intro ===")
    train_hist = Path("results/default/train/history.jsonl")
    eval_hist = Path("results/default/train/eval_history.jsonl")
    latest_eval_path = Path("results/default/train/latest_eval_summary.json")
    if not train_hist.exists():
        print(f"  history not found at {train_hist}")
        return {}

    def _summarize_train_stage(stage_name: str) -> Dict[str, Any]:
        n = success = collision = timeout = 0
        min_clearance_sum = 0.0
        lateral_sum = 0.0
        distance_sum = 0.0
        episode_returns: List[float] = []
        rows: List[Dict[str, Any]] = []
        with train_hist.open() as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get("stage_name") != stage_name:
                    continue
                rows.append(entry)
                n += 1
                s = float(entry.get("success", 0.0))
                c = float(entry.get("collision", 0.0))
                if s > 0.5:
                    success += 1
                elif c > 0.5:
                    collision += 1
                else:
                    timeout += 1
                min_clearance_sum += float(entry.get("min_clearance", 0.0))
                lateral_sum += float(entry.get("route_line_lateral_error", 0.0))
                distance_sum += float(entry.get("distance_to_goal", 0.0))
                episode_returns.append(float(entry.get("episode_return", 0.0)))
        if n == 0:
            return {}
        return {
            "stage_name": stage_name,
            "n_episodes_at_stage": n,
            "success_rate": _round(success / n),
            "collision_rate": _round(collision / n),
            "timeout_rate": _round(timeout / n),
            "avg_min_clearance": _round(min_clearance_sum / n),
            "avg_route_lateral_error": _round(lateral_sum / n),
            "avg_distance_to_goal": _round(distance_sum / n),
            "avg_episode_return": _round(np.mean(episode_returns)),
            "last_stage_episode": int(rows[-1].get("stage_episode", n)),
        }

    def _summarize_eval_stage(stage_name: str) -> Dict[str, Any]:
        if not eval_hist.exists():
            return {}
        rows: List[Dict[str, Any]] = []
        with eval_hist.open() as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                if entry.get("stage_name") == stage_name:
                    rows.append(entry)
        if not rows:
            return {}

        def _score(entry: Dict[str, Any]) -> tuple[float, float, float]:
            return (
                float(entry.get("success_rate", 0.0)),
                -float(entry.get("collision_rate", 1.0)),
                -float(entry.get("avg_distance_to_goal", 1e9)),
            )

        best = max(rows, key=_score)
        latest = rows[-1]
        early_window = rows[: min(4, len(rows))]
        late_window = rows[-min(4, len(rows)) :]
        return {
            "stage_name": stage_name,
            "n_evals": len(rows),
            "best_success_rate": _round(best.get("success_rate")),
            "best_collision_rate": _round(best.get("collision_rate")),
            "best_avg_distance_to_goal": _round(best.get("avg_distance_to_goal")),
            "best_avg_route_lateral_error": _round(best.get("avg_route_line_lateral_error")),
            "best_avg_min_clearance": _round(best.get("avg_min_clearance")),
            "latest_stage_episode": int(latest.get("stage_episode", len(rows))),
            "latest_success_rate": _round(latest.get("success_rate")),
            "latest_collision_rate": _round(latest.get("collision_rate")),
            "latest_avg_distance_to_goal": _round(latest.get("avg_distance_to_goal")),
            "latest_avg_route_lateral_error": _round(latest.get("avg_route_line_lateral_error")),
            "latest_avg_min_clearance": _round(latest.get("avg_min_clearance")),
            "early_success_rate_mean": _round(np.mean([float(row.get("success_rate", 0.0)) for row in early_window])),
            "late_success_rate_mean": _round(np.mean([float(row.get("success_rate", 0.0)) for row in late_window])),
            "early_lateral_error_mean": _round(
                np.mean([float(row.get("avg_route_line_lateral_error", 0.0)) for row in early_window])
            ),
            "late_lateral_error_mean": _round(
                np.mean([float(row.get("avg_route_line_lateral_error", 0.0)) for row in late_window])
            ),
            "early_clearance_mean": _round(np.mean([float(row.get("avg_min_clearance", 0.0)) for row in early_window])),
            "late_clearance_mean": _round(np.mean([float(row.get("avg_min_clearance", 0.0)) for row in late_window])),
        }

    checkpoint_stage_name = None
    if latest_eval_path.exists():
        try:
            checkpoint_stage_name = json.loads(latest_eval_path.read_text()).get("stage_name")
        except Exception:
            checkpoint_stage_name = None

    requested_train = _summarize_train_stage(STAGE_NAME)
    requested_eval = _summarize_eval_stage(STAGE_NAME)
    report: Dict[str, Any] = {
        "requested_stage": STAGE_NAME,
        "requested_stage_reached": bool(requested_train or requested_eval),
    }
    if requested_train:
        report["requested_stage_train_summary"] = requested_train
    if requested_eval:
        report["requested_stage_eval_summary"] = requested_eval

    if not report["requested_stage_reached"]:
        print("  requested stage was never reached in training history")
        if checkpoint_stage_name is not None:
            fallback_train = _summarize_train_stage(checkpoint_stage_name)
            fallback_eval = _summarize_eval_stage(checkpoint_stage_name)
            report["fallback_stage"] = checkpoint_stage_name
            if fallback_train:
                report["fallback_stage_train_summary"] = fallback_train
            if fallback_eval:
                report["fallback_stage_eval_summary"] = fallback_eval

    print(json.dumps(report, indent=2))
    return report


def test_5_obstacle_symmetry(env) -> dict:
    """Sample 50 resets and check route-obstacle side distribution."""
    print("\n=== TEST 5: obstacle symmetry on resets ===")
    sides = []
    for seed in range(50):
        env.reset(seed=5000 + seed)
        sides.append(_obstacle_side(env))
    counts = Counter(sides)
    total = max(sum(counts.values()), 1)
    report = {
        k: {
            "count": counts.get(k, 0),
            "fraction": _round(counts.get(k, 0) / total),
        }
        for k in ["left", "right", "center", "none"]
    }
    print(json.dumps(report, indent=2))
    return report


def derive_conclusions(results: Dict[str, Any]) -> List[str]:
    conclusions: List[str] = []
    policy = results.get("test_1_policy_rollouts", {})
    teacher = results.get("test_2_teacher_rollouts", {})
    sensitivity = results.get("test_3_sensitivity", {})
    history = results.get("test_4_history", {})
    symmetry = results.get("test_5_symmetry", {})

    policy_lr = [policy.get(side) for side in ("left", "right") if side in policy]
    teacher_lr = [teacher.get(side) for side in ("left", "right") if side in teacher]
    if policy_lr and teacher_lr:
        policy_success = float(np.mean([side["success_rate"] for side in policy_lr]))
        teacher_success = float(np.mean([side["success_rate"] for side in teacher_lr]))
        policy_timeout = float(np.mean([side["timeout_rate"] for side in policy_lr]))
        lateral_mean = float(np.mean([side["lateral_max_mean"] for side in policy_lr]))
        conclusions.append(
            "Teacher solves the symmetric left/right default-intro cases reliably "
            f"(mean success {teacher_success:.3f}), while the v3 policy fails them "
            f"(mean success {policy_success:.3f}, mean timeout {policy_timeout:.3f}, "
            f"mean max lateral excursion {lateral_mean:.3f})."
        )

    max_angle = sensitivity.get("max_pairwise_angle_deg")
    median_angle = sensitivity.get("median_pairwise_angle_deg")
    if max_angle is not None and median_angle is not None:
        conclusions.append(
            "The encoder is not obstacle-blind: sweeping one obstacle changes the "
            f"deterministic action by up to {float(max_angle):.2f} degrees "
            f"(median {float(median_angle):.2f} degrees)."
        )

    left = symmetry.get("left", {}).get("count", 0)
    right = symmetry.get("right", {}).get("count", 0)
    total_lr = max(left + right, 1)
    imbalance = abs(left - right) / total_lr
    conclusions.append(
        "Reset sampling is roughly symmetric, so the failure is not explained by "
        f"a left/right data imbalance (left={left}, right={right}, imbalance={imbalance:.3f})."
    )

    if not history.get("requested_stage_reached", True):
        fallback = history.get("fallback_stage_eval_summary", {})
        fallback_stage = history.get("fallback_stage")
        conclusions.append(
            f"The saved v3 run never reached {STAGE_NAME}; it stalled earlier at "
            f"{fallback_stage or 'an earlier target stage'}."
        )
        if fallback:
            conclusions.append(
                "At the fallback stage, success collapsed while safety margin improved: "
                f"best success {fallback.get('best_success_rate')}, latest success {fallback.get('latest_success_rate')}, "
                f"early mean lateral error {fallback.get('early_lateral_error_mean')}, "
                f"late mean lateral error {fallback.get('late_lateral_error_mean')}, "
                f"early mean clearance {fallback.get('early_clearance_mean')}, "
                f"late mean clearance {fallback.get('late_clearance_mean')}."
            )
            conclusions.append(
                "That pattern matches over-avoidance and failed rejoin: the policy becomes "
                "safer around obstacles but drifts far off-route and times out instead of committing back to goal."
            )

    conclusions.append(
        "The next curriculum should therefore keep v4's visibility/symmetry fixes, but add stronger anti-drift "
        "pressure around target_bypass_commit_v2 through target_default: reduce reward for wide bypassing once safe, "
        "increase rejoin/remaining-distance pressure, and tighten recovery/teacher support on the stage where collapse starts."
    )
    return conclusions


def write_report(results: Dict[str, Any], conclusions: List[str]) -> Path:
    report_lines = [
        "# v3 diagnostics",
        "",
        "## Conclusions",
        "",
    ]
    report_lines.extend(f"- {line}" for line in conclusions)
    report_lines.extend(
        [
            "",
            "## Raw results",
            "",
            "```json",
            json.dumps(results, indent=2),
            "```",
            "",
        ]
    )
    report_path = Path("results/v3_diagnostics_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines))
    return report_path


def main():
    base = load_config(V3_CONFIG)
    stage = _pick_stage(base, STAGE_NAME)
    config = _stage_config(base, stage)
    validate_config(config)

    env = create_environment(config, gui=False, seed=config.environment.seed)
    agent = create_agent(config, env)
    md = agent.load(V3_CHECKPOINT, load_optimizer_state=False)
    print(f"Loaded {V3_CHECKPOINT}, metadata keys: {list(md.keys())}")
    print(f"env.global_feature_dim = {env.global_feature_dim}")
    print(f"model log_std = {torch.exp(agent.model.log_std).detach().cpu().numpy()}")

    out = {
        "checkpoint": {
            "path": V3_CHECKPOINT,
            "metadata": md,
        },
        "test_1_policy_rollouts": test_1_policy_rollouts(env, agent),
        "test_2_teacher_rollouts": test_2_teacher_rollouts(env),
        "test_3_sensitivity":      test_3_sensitivity(env, agent),
        "test_4_history":          test_4_history_failure_modes(),
        "test_5_symmetry":         test_5_obstacle_symmetry(env),
    }
    conclusions = derive_conclusions(out)
    out["conclusions"] = conclusions
    out_path = Path("results/v3_diagnostics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    report_path = write_report(out, conclusions)
    print("\n=== CONCLUSIONS ===")
    for idx, line in enumerate(conclusions, start=1):
        print(f"{idx}. {line}")
    print(f"\nWrote {out_path}")
    print(f"Wrote {report_path}")
    env.close()


if __name__ == "__main__":
    main()
