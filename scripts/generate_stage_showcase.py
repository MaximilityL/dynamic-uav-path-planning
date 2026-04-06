#!/usr/bin/env python3
"""Generate 2D/3D plots and MP4 replays for reached and target curriculum stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from typing import Dict, Optional, Tuple

import numpy as np

from _common import bootstrap_project

bootstrap_project()

from src.training.factories import create_agent, create_environment
from src.training.loops import (
    _curriculum_stage_name,
    _evaluate_current_policy,
    _save_best_trajectory,
    _stage_config,
    _stage_directory_name,
    _stage_target_reached,
    run_episode,
)
from src.utils.config import load_config, validate_config
from src.utils.io import save_json
from src.visualization import save_episode_showcase_plots


class _FFmpegVideoWriter:
    """Small raw-frame ffmpeg writer for GUI capture."""

    def __init__(self, *, output_path: Path, width: int, height: int, fps: int) -> None:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            raise RuntimeError("ffmpeg is not available on PATH, so showcase MP4 export cannot run.")

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.fps = max(int(fps), 1)
        self._closed = False
        self._process = subprocess.Popen(
            [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-nostats",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{self.width}x{self.height}",
                "-r",
                str(self.fps),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(self.output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        """Append one RGB frame."""
        if self._closed or self._process.stdin is None:
            raise RuntimeError("ffmpeg writer is already closed.")

        frame_array = np.asarray(frame, dtype=np.uint8)
        expected_shape = (self.height, self.width, 3)
        if tuple(frame_array.shape) != expected_shape:
            raise ValueError(f"Expected frame shape {expected_shape}, got {tuple(frame_array.shape)}.")
        self._process.stdin.write(frame_array.tobytes())

    def close(self) -> None:
        """Finalize encoding and surface ffmpeg failures clearly."""
        if self._closed:
            return
        self._closed = True
        stderr_output = b""
        if self._process.stdin is not None:
            self._process.stdin.close()
        stderr_output = self._process.stderr.read() if self._process.stderr is not None else b""
        return_code = self._process.wait()
        if return_code != 0:
            message = stderr_output.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffmpeg failed while writing {self.output_path}.\n{message}")


def _stage_lookup(config) -> Dict[str, Dict[str, object]]:
    """Build a name -> stage lookup for the active curriculum."""
    curriculum = list(getattr(config.training, "curriculum", []) or [])
    return {
        _curriculum_stage_name(stage, index): stage
        for index, stage in enumerate(curriculum)
    }


def _resolve_named_stage(config, stage_name: str) -> Optional[Dict[str, object]]:
    """Resolve one curriculum stage by name."""
    stages = _stage_lookup(config)
    if not stages and stage_name == "main":
        return None
    if stage_name not in stages:
        available = ", ".join(stages.keys())
        raise ValueError(f"Unknown stage '{stage_name}'. Available stages: {available}")
    return stages[stage_name]


def _load_stage_best(results_dir: str | Path) -> Dict[str, Dict[str, object]]:
    """Load saved per-stage best metrics when they exist."""
    path = Path(results_dir) / "train" / "stage_best_evaluations.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {str(key): dict(value or {}) for key, value in dict(payload).items()}


def _auto_best_reached_stage(config) -> str:
    """Pick the furthest stage whose saved best evaluation met its advancement target."""
    curriculum = list(getattr(config.training, "curriculum", []) or [])
    if not curriculum:
        return "main"

    stage_best = _load_stage_best(config.training.results_dir)
    solved_stage_name: Optional[str] = None
    fallback_stage_name: Optional[str] = None
    fallback_score: Tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -float("inf"))

    for index, stage in enumerate(curriculum):
        stage_name = _curriculum_stage_name(stage, index)
        stage_record = dict(stage_best.get(stage_name, {}) or {})
        if not stage_record:
            continue

        eval_summary = {
            "success_rate": float(stage_record.get("best_success_rate", 0.0)),
            "collision_rate": float(stage_record.get("best_collision_rate", 1.0)),
        }
        if _stage_target_reached(stage=stage, eval_summary=eval_summary, stage_episode_count=int(stage.get("min_stage_episodes", 0))):
            solved_stage_name = stage_name

        score = (
            float(index),
            float(stage_record.get("best_success_rate", 0.0)),
            -float(stage_record.get("best_collision_rate", 1.0)),
            float(stage_record.get("best_avg_return", 0.0)),
        )
        if score > fallback_score:
            fallback_score = score
            fallback_stage_name = stage_name

    if solved_stage_name is not None:
        return solved_stage_name
    if fallback_stage_name is not None:
        return fallback_stage_name
    return _curriculum_stage_name(curriculum[0], 0)


def _existing_path(*candidates: Path) -> Optional[Path]:
    """Return the first existing path in order."""
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_model_path(
    *,
    config,
    stage_name: str,
    explicit_model: Optional[str],
    preferred_resume: bool,
    fallback_model: Optional[Path] = None,
) -> Tuple[Path, str]:
    """Resolve the checkpoint to use for one showcase case."""
    if explicit_model is not None:
        path = Path(explicit_model)
        if not path.exists():
            raise FileNotFoundError(f"Explicit model path does not exist: {path}")
        return path, "explicit"

    stage_best = _load_stage_best(config.training.results_dir).get(stage_name, {})
    stage_checkpoint = Path(config.training.checkpoint_dir) / "stages" / _stage_directory_name(stage_name) / "best_model.pth"
    recorded_stage_checkpoint = Path(str(stage_best.get("best_model_path", ""))) if stage_best.get("best_model_path") else None

    resume_path = None
    resume_config = dict(getattr(config.training, "resume", {}) or {})
    if resume_config.get("checkpoint_path"):
        resume_path = Path(str(resume_config["checkpoint_path"]))

    resolved = _existing_path(
        stage_checkpoint,
        *( [recorded_stage_checkpoint] if recorded_stage_checkpoint is not None else [] ),
    )
    if resolved is not None:
        return resolved, "stage_best"

    if preferred_resume and resume_path is not None and resume_path.exists():
        return resume_path, "resume_checkpoint"

    last_checkpoint = Path(config.training.checkpoint_dir) / "last_model.pth"
    best_checkpoint = Path(config.training.checkpoint_dir) / "best_model.pth"
    resolved = _existing_path(last_checkpoint, best_checkpoint)
    if resolved is not None:
        return resolved, "global_checkpoint"

    if fallback_model is not None and fallback_model.exists():
        return fallback_model, "fallback_case_model"

    raise FileNotFoundError(f"Could not resolve a checkpoint for stage '{stage_name}'.")


def _record_episode_video(*, config, model_path: Path, episode_seed: int, output_path: Path) -> Dict[str, object]:
    """Replay one deterministic episode in the GUI and save it as MP4."""
    env = create_environment(config, gui=True, seed=config.environment.seed)
    writer: Optional[_FFmpegVideoWriter] = None
    try:
        agent = create_agent(config, env)
        agent.load(str(model_path))
        observation, _ = env.reset(seed=int(episode_seed))
        initial_frame = env.capture_gui_frame()
        writer = _FFmpegVideoWriter(
            output_path=output_path,
            width=int(initial_frame.shape[1]),
            height=int(initial_frame.shape[0]),
            fps=max(int(getattr(env, "ctrl_freq", 30)), 1),
        )
        writer.write(initial_frame)

        terminated = False
        truncated = False
        last_frame = initial_frame
        while not (terminated or truncated):
            action, _ = agent.select_action(observation, deterministic=True)
            observation, _, terminated, truncated, _ = env.step(action)
            last_frame = env.capture_gui_frame()
            writer.write(last_frame)

        for _ in range(max(int(getattr(env, "ctrl_freq", 30)) // 2, 1)):
            writer.write(last_frame)

        metrics = env.get_episode_summary()
        writer.close()
    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        env.close()
    return dict(metrics)


def _generate_case(
    *,
    base_config,
    stage_name: str,
    case_label: str,
    model_path: Path,
    model_source: str,
    episode_budget: int,
    output_root: Path,
    render_video: bool,
) -> Dict[str, object]:
    """Evaluate one stage, save its best trajectory plots, and optionally render an MP4."""
    stage = _resolve_named_stage(base_config, stage_name)
    config = _stage_config(base_config, stage)
    validate_config(config)

    agent_env = create_environment(config, gui=False, seed=config.environment.seed)
    try:
        agent = create_agent(config, agent_env)
        agent.load(str(model_path))
    finally:
        agent_env.close()

    history, summary, best_trajectory = _evaluate_current_policy(
        config=config,
        agent=agent,
        num_episodes=episode_budget,
        render=False,
        deterministic=True,
    )
    if best_trajectory is None:
        raise RuntimeError(f"No trajectory was produced for stage '{stage_name}'.")

    case_dir = output_root / stage_name
    case_dir.mkdir(parents=True, exist_ok=True)
    _save_best_trajectory(case_dir / "best_episode", best_trajectory)
    plot_paths = save_episode_showcase_plots(
        trajectory=best_trajectory,
        output_dir=case_dir,
        stem="best_episode",
        title=f"{case_label}: {stage_name}",
    )

    video_path = case_dir / "best_episode.mp4"
    video_metrics = None
    video_error = None
    if render_video:
        try:
            video_metrics = _record_episode_video(
                config=config,
                model_path=model_path,
                episode_seed=int(best_trajectory["episode_seed"]),
                output_path=video_path,
            )
            if not video_path.exists():
                video_error = "Replay finished but no MP4 was written."
            elif video_path.stat().st_size <= 1024:
                video_error = f"MP4 was created but appears empty ({video_path.stat().st_size} bytes)."
                video_path.unlink(missing_ok=True)
        except RuntimeError as exc:
            video_error = str(exc)

    payload = {
        "case_label": case_label,
        "stage_name": stage_name,
        "model_path": str(model_path),
        "model_source": model_source,
        "summary": {
            **dict(summary),
            "stage_name": stage_name,
            "case_label": case_label,
        },
        "best_episode_seed": int(best_trajectory["episode_seed"]),
        "plot_2d": str(plot_paths["plot_2d"]),
        "plot_3d": str(plot_paths["plot_3d"]),
        "video_path": str(video_path) if render_video and video_error is None and video_path.exists() else None,
        "video_error": video_error,
        "video_metrics": video_metrics,
        "episodes": history,
    }
    save_json(case_dir / "case_summary.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate stage showcase plots and MP4s")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Configuration file")
    parser.add_argument("--best-stage", type=str, default="auto", help="Best reached stage name or 'auto'")
    parser.add_argument("--wanted-stage", type=str, default="target_default", help="Target stage to showcase")
    parser.add_argument("--best-model", type=str, help="Optional checkpoint override for the reached-stage case")
    parser.add_argument("--wanted-model", type=str, help="Optional checkpoint override for the wanted-stage case")
    parser.add_argument("--episodes", type=int, help="Deterministic evaluation episodes per case")
    parser.add_argument("--output-dir", type=str, help="Output directory for plots, summaries, and videos")
    parser.add_argument("--seed", type=int, help="Override the base environment seed")
    parser.add_argument("--no-video", action="store_true", help="Skip the MP4 replay export")
    args = parser.parse_args()

    base_config = load_config(args.config)
    if args.seed is not None:
        base_config.environment.seed = int(args.seed)
    validate_config(base_config)

    reached_stage_name = _auto_best_reached_stage(base_config) if args.best_stage == "auto" else args.best_stage
    _resolve_named_stage(base_config, reached_stage_name)
    _resolve_named_stage(base_config, args.wanted_stage)

    episode_budget = int(
        args.episodes
        or getattr(base_config.training, "eval_episodes", 0)
        or base_config.evaluation.num_episodes
    )
    output_root = Path(args.output_dir) if args.output_dir else Path(base_config.training.results_dir) / "showcase"
    output_root.mkdir(parents=True, exist_ok=True)

    reached_model_path, reached_model_source = _resolve_model_path(
        config=base_config,
        stage_name=reached_stage_name,
        explicit_model=args.best_model,
        preferred_resume=False,
    )
    wanted_model_path, wanted_model_source = _resolve_model_path(
        config=base_config,
        stage_name=args.wanted_stage,
        explicit_model=args.wanted_model,
        preferred_resume=True,
        fallback_model=reached_model_path,
    )

    reached_case = _generate_case(
        base_config=base_config,
        stage_name=reached_stage_name,
        case_label="Best Reached Stage",
        model_path=reached_model_path,
        model_source=reached_model_source,
        episode_budget=episode_budget,
        output_root=output_root,
        render_video=not args.no_video,
    )
    wanted_case = _generate_case(
        base_config=base_config,
        stage_name=args.wanted_stage,
        case_label="Wanted Stage",
        model_path=wanted_model_path,
        model_source=wanted_model_source,
        episode_budget=episode_budget,
        output_root=output_root,
        render_video=not args.no_video,
    )

    combined_summary = {
        "config": str(Path(args.config)),
        "output_dir": str(output_root),
        "episode_budget": episode_budget,
        "best_stage_requested": args.best_stage,
        "best_stage_resolved": reached_stage_name,
        "wanted_stage": args.wanted_stage,
        "cases": {
            "best_reached": reached_case,
            "wanted": wanted_case,
        },
    }
    save_json(output_root / "showcase_summary.json", combined_summary)

    print(f"output_dir={output_root}")
    print(f"best_reached_stage={reached_stage_name}")
    print(f"best_reached_model={reached_model_path}")
    print(f"best_reached_success_rate={float(reached_case['summary']['success_rate']):.3f}")
    print(f"wanted_stage={args.wanted_stage}")
    print(f"wanted_stage_model={wanted_model_path}")
    print(f"wanted_stage_success_rate={float(wanted_case['summary']['success_rate']):.3f}")
    print(f"wanted_stage_collision_rate={float(wanted_case['summary']['collision_rate']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
