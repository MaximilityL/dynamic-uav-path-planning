"""Trajectory plotting helpers for saved evaluation episodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    """Convert one trajectory field into a numeric NumPy array."""
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"Trajectory field '{name}' is empty.")
    return array


def _axis_limits(values: np.ndarray) -> tuple[float, float]:
    """Compute padded axis limits that stay readable for short paths."""
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    padding = max(span * 0.08, 0.18)
    return lower - padding, upper + padding


def _summary_caption(summary: Dict[str, Any]) -> str:
    """Turn core summary metrics into a compact caption."""
    return (
        f"return={float(summary.get('episode_return', 0.0)):.2f} | "
        f"success={int(bool(summary.get('success', 0.0)))} | "
        f"collision={int(bool(summary.get('collision', 0.0)))} | "
        f"steps={int(float(summary.get('steps', 0.0)))} | "
        f"min_clearance={float(summary.get('min_clearance', 0.0)):.2f}"
    )


def _apply_workspace_limits(ax, workspace_bounds: np.ndarray | None, *, x_idx: int, y_idx: int) -> None:
    """Apply stable world limits when the trajectory export includes them."""
    if workspace_bounds is None or workspace_bounds.shape != (3, 2):
        return
    ax.set_xlim(float(workspace_bounds[x_idx, 0]), float(workspace_bounds[x_idx, 1]))
    ax.set_ylim(float(workspace_bounds[y_idx, 0]), float(workspace_bounds[y_idx, 1]))


def _projected_limits(
    *,
    positions: np.ndarray,
    obstacles: np.ndarray,
    start_position: np.ndarray,
    goal: np.ndarray,
    axis_index: int,
) -> tuple[float, float]:
    """Compute padded limits from all visible geometry in one projected axis."""
    values = [positions[:, axis_index], np.asarray([start_position[axis_index], goal[axis_index]], dtype=np.float32)]
    if obstacles.size:
        values.append(obstacles[:, :, axis_index].reshape(-1))
    return _axis_limits(np.concatenate(values))


def _isometric_projection(points: np.ndarray) -> np.ndarray:
    """Project 3D points into a readable 2D pseudo-3D view."""
    yaw = np.deg2rad(-52.0)
    pitch = np.deg2rad(28.0)
    x_values = points[..., 0]
    y_values = points[..., 1]
    z_values = points[..., 2]

    x_rotated = x_values * np.cos(yaw) - y_values * np.sin(yaw)
    y_rotated = x_values * np.sin(yaw) + y_values * np.cos(yaw)
    projected_x = x_rotated
    projected_y = z_values * np.cos(pitch) + y_rotated * np.sin(pitch)
    return np.stack([projected_x, projected_y], axis=-1)


def plot_episode_trajectory_2d(
    *,
    trajectory: Dict[str, Any],
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Render top-down and side-view trajectory plots for one evaluation episode."""
    positions = _as_float_array(trajectory["positions"], name="positions")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Trajectory 'positions' must have shape [T, 3].")

    obstacles = np.asarray(trajectory.get("obstacles", np.zeros((positions.shape[0], 0, 3), dtype=np.float32)), dtype=np.float32)
    if obstacles.ndim == 2:
        obstacles = obstacles.reshape(obstacles.shape[0], 1, obstacles.shape[1])
    if obstacles.ndim != 3 or obstacles.shape[-1] != 3:
        raise ValueError("Trajectory 'obstacles' must have shape [T, N, 3].")

    goal = _as_float_array(trajectory["goal"], name="goal").reshape(3)
    start_position = np.asarray(trajectory.get("start_position", positions[0]), dtype=np.float32).reshape(3)
    workspace_bounds = np.asarray(trajectory.get("workspace_bounds"), dtype=np.float32) if "workspace_bounds" in trajectory else None
    goal_tolerance = float(trajectory.get("goal_tolerance", 0.0))
    summary = dict(trajectory.get("summary", {}) or {})

    figure, axes = plt.subplots(1, 2, figsize=(15, 6))
    view_specs = [
        (0, 1, "Top-Down (x / y)", "x (m)", "y (m)"),
        (0, 2, "Elevation (x / z)", "x (m)", "z (m)"),
    ]
    obstacle_count = int(obstacles.shape[1]) if obstacles.size else 0

    for ax, (x_idx, y_idx, subtitle, x_label, y_label) in zip(axes, view_specs):
        ax.plot(
            positions[:, x_idx],
            positions[:, y_idx],
            color="#1f77b4",
            linewidth=2.4,
            label="UAV trajectory",
        )
        ax.scatter(start_position[x_idx], start_position[y_idx], color="#2a9d8f", s=60, marker="o", label="start", zorder=4)
        ax.scatter(positions[-1, x_idx], positions[-1, y_idx], color="#264653", s=70, marker="X", label="end", zorder=4)
        ax.scatter(goal[x_idx], goal[y_idx], color="#f4a261", s=180, marker="*", edgecolors="#111111", linewidths=0.7, label="target", zorder=5)

        if goal_tolerance > 0.0:
            ax.add_patch(
                Circle(
                    (float(goal[x_idx]), float(goal[y_idx])),
                    radius=goal_tolerance,
                    fill=False,
                    linestyle="--",
                    linewidth=1.2,
                    edgecolor="#f4a261",
                    alpha=0.65,
                )
            )

        for obstacle_idx in range(obstacle_count):
            label = "obstacle track" if obstacle_idx == 0 else None
            ax.plot(
                obstacles[:, obstacle_idx, x_idx],
                obstacles[:, obstacle_idx, y_idx],
                color="#d62828",
                alpha=0.32,
                linewidth=1.3,
                label=label,
            )
            ax.scatter(
                obstacles[0, obstacle_idx, x_idx],
                obstacles[0, obstacle_idx, y_idx],
                color="#f77f00",
                alpha=0.55,
                s=12,
                zorder=3,
            )
            ax.scatter(
                obstacles[-1, obstacle_idx, x_idx],
                obstacles[-1, obstacle_idx, y_idx],
                color="#6a040f",
                alpha=0.7,
                s=12,
                zorder=3,
            )

        ax.set_title(subtitle)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        _apply_workspace_limits(ax, workspace_bounds, x_idx=x_idx, y_idx=y_idx)
        if workspace_bounds is None:
            ax.set_xlim(
                *_projected_limits(
                    positions=positions,
                    obstacles=obstacles,
                    start_position=start_position,
                    goal=goal,
                    axis_index=x_idx,
                )
            )
            ax.set_ylim(
                *_projected_limits(
                    positions=positions,
                    obstacles=obstacles,
                    start_position=start_position,
                    goal=goal,
                    axis_index=y_idx,
                )
            )

    axes[0].legend(loc="best")
    if title:
        figure.suptitle(title, fontsize=14)
    if summary:
        figure.text(0.5, 0.02, _summary_caption(summary), ha="center", fontsize=10)
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.95 if title else 1.0))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_episode_trajectory_3d(
    *,
    trajectory: Dict[str, Any],
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Render a 3D path view for one evaluation episode."""
    positions = _as_float_array(trajectory["positions"], name="positions")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("Trajectory 'positions' must have shape [T, 3].")

    obstacles = np.asarray(trajectory.get("obstacles", np.zeros((positions.shape[0], 0, 3), dtype=np.float32)), dtype=np.float32)
    if obstacles.ndim == 2:
        obstacles = obstacles.reshape(obstacles.shape[0], 1, obstacles.shape[1])
    if obstacles.ndim != 3 or obstacles.shape[-1] != 3:
        raise ValueError("Trajectory 'obstacles' must have shape [T, N, 3].")

    goal = _as_float_array(trajectory["goal"], name="goal").reshape(3)
    start_position = np.asarray(trajectory.get("start_position", positions[0]), dtype=np.float32).reshape(3)
    workspace_bounds = np.asarray(trajectory.get("workspace_bounds"), dtype=np.float32) if "workspace_bounds" in trajectory else None
    summary = dict(trajectory.get("summary", {}) or {})

    use_true_3d = True
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        use_true_3d = False

    figure = plt.figure(figsize=(10, 8))
    obstacle_count = int(obstacles.shape[1]) if obstacles.size else 0
    if use_true_3d:
        axis = figure.add_subplot(111, projection="3d")
        axis.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="#1f77b4", linewidth=2.4, label="UAV trajectory")
        axis.scatter(start_position[0], start_position[1], start_position[2], color="#2a9d8f", s=70, marker="o", label="start")
        axis.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color="#264653", s=80, marker="X", label="end")
        axis.scatter(goal[0], goal[1], goal[2], color="#f4a261", s=200, marker="*", edgecolors="#111111", linewidths=0.7, label="target")

        for obstacle_idx in range(obstacle_count):
            label = "obstacle track" if obstacle_idx == 0 else None
            axis.plot(
                obstacles[:, obstacle_idx, 0],
                obstacles[:, obstacle_idx, 1],
                obstacles[:, obstacle_idx, 2],
                color="#d62828",
                alpha=0.34,
                linewidth=1.25,
                label=label,
            )

        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_zlabel("z (m)")
        axis.view_init(elev=25, azim=-58)

        if workspace_bounds is not None and workspace_bounds.shape == (3, 2):
            axis.set_xlim(float(workspace_bounds[0, 0]), float(workspace_bounds[0, 1]))
            axis.set_ylim(float(workspace_bounds[1, 0]), float(workspace_bounds[1, 1]))
            axis.set_zlim(float(workspace_bounds[2, 0]), float(workspace_bounds[2, 1]))
            spans = workspace_bounds[:, 1] - workspace_bounds[:, 0]
            axis.set_box_aspect(tuple(float(max(span, 0.5)) for span in spans))
        else:
            axis.set_xlim(*_projected_limits(positions=positions, obstacles=obstacles, start_position=start_position, goal=goal, axis_index=0))
            axis.set_ylim(*_projected_limits(positions=positions, obstacles=obstacles, start_position=start_position, goal=goal, axis_index=1))
            axis.set_zlim(*_projected_limits(positions=positions, obstacles=obstacles, start_position=start_position, goal=goal, axis_index=2))

        axis.legend(loc="upper left")
        if title:
            axis.set_title(title)
    else:
        axis = figure.add_subplot(111)
        projected_positions = _isometric_projection(positions)
        projected_start = _isometric_projection(start_position.reshape(1, 3))[0]
        projected_goal = _isometric_projection(goal.reshape(1, 3))[0]
        projected_end = projected_positions[-1]

        axis.plot(projected_positions[:, 0], projected_positions[:, 1], color="#1f77b4", linewidth=2.4, label="UAV trajectory")
        axis.scatter(projected_start[0], projected_start[1], color="#2a9d8f", s=70, marker="o", label="start", zorder=4)
        axis.scatter(projected_end[0], projected_end[1], color="#264653", s=80, marker="X", label="end", zorder=4)
        axis.scatter(projected_goal[0], projected_goal[1], color="#f4a261", s=200, marker="*", edgecolors="#111111", linewidths=0.7, label="target", zorder=5)

        for obstacle_idx in range(obstacle_count):
            label = "obstacle track" if obstacle_idx == 0 else None
            projected_obstacle = _isometric_projection(obstacles[:, obstacle_idx, :])
            axis.plot(
                projected_obstacle[:, 0],
                projected_obstacle[:, 1],
                color="#d62828",
                alpha=0.34,
                linewidth=1.25,
                label=label,
            )

        combined_projection = np.concatenate(
            [
                projected_positions,
                projected_start.reshape(1, 2),
                projected_goal.reshape(1, 2),
                _isometric_projection(obstacles.reshape(-1, 3)) if obstacles.size else np.zeros((0, 2), dtype=np.float32),
            ],
            axis=0,
        )
        axis.set_xlim(*_axis_limits(combined_projection[:, 0]))
        axis.set_ylim(*_axis_limits(combined_projection[:, 1]))
        axis.set_xlabel("projected x")
        axis.set_ylabel("projected z")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="upper left")
        axis.set_title(f"{title} (projected 3D)" if title else "Projected 3D trajectory")

    if summary:
        figure.text(0.5, 0.02, _summary_caption(summary), ha="center", fontsize=10)
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 0.98))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output


def save_episode_showcase_plots(
    *,
    trajectory: Dict[str, Any],
    output_dir: str | Path,
    stem: str,
    title: str | None = None,
) -> Dict[str, Path]:
    """Save a matched pair of 2D and 3D trajectory figures."""
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    return {
        "plot_2d": plot_episode_trajectory_2d(
            trajectory=trajectory,
            output_path=output_root / f"{stem}_2d.png",
            title=title,
        ),
        "plot_3d": plot_episode_trajectory_3d(
            trajectory=trajectory,
            output_path=output_root / f"{stem}_3d.png",
            title=title,
        ),
    }
