# Dynamic UAV Path Planning

Version: `1.1.0`

This repository is a research codebase for single-UAV path planning in dynamic obstacle fields. It combines a PyBullet-backed UAV, fixed-size dense graph observations, and a readable PPO baseline that is optimized for iteration on curriculum design, reward shaping, teacher guidance, and training infrastructure.

## Current Repo Status

- The canonical full fresh curriculum is `configs/default_curriculum.yml`.
  It is the preserved `v6` baseline, solves the bridge and medium stages, reaches `target_default_intro`, and records a stage-best intro success of `0.5` with `0.0` collision, but it does not advance to `target_default`.
- The canonical target-stage continuation is `configs/default_post_commit.yaml`.
  It is the preserved `v15` post-commit branch, resumes from the default curriculum intro checkpoint, and keeps the aggressive target-stage supervision stack, but `target_default` is still unsolved.
- The helper presets now kept in the repo are `configs/default.yaml`, `configs/debug.yaml`, `configs/easy_train.yaml`, and `configs/hard_obstacles.yaml`.

The repo is therefore strongest today as:

- a working research scaffold
- a reproducible curriculum/evaluation pipeline
- a target-stage experimentation platform

It is not yet a repo with a confirmed stable `target_default` solution.

## What The Repo Includes

- a PyBullet-backed dynamic airspace environment
- fixed-size dense graph observations
- a graph PPO baseline in plain PyTorch
- heuristic teacher reward and teacher-action supervision hooks
- stage-aware behavior-cloning warm start
- success-filtered teacher-demo pretraining
- stage-local best checkpoints, rollback, and plateau recovery
- training, evaluation, plotting, and smoke-test CLIs
- result artifacts, trajectory export, and tests

## Recommended Entry Points

Smoke-test the stack:

```bash
python scripts/smoke_test.py --config configs/debug.yaml
```

Run the canonical full fresh curriculum baseline:

```bash
python scripts/train.py --config configs/default_curriculum.yml
```

Evaluate the best intro checkpoint from that run:

```bash
python scripts/evaluate.py \
  --config configs/default_curriculum.yml \
  --model checkpoints/default_curriculum/stages/target_default_intro/best_model.pth \
  --stage-name target_default_intro
```

Run the canonical target-only continuation:

```bash
python scripts/train.py --config configs/default_post_commit.yaml
```

## Repository Layout

```text
dynamic-uav-path-planning/
├── CHANGELOG.md
├── VERSION
├── README.md
├── configs/
├── docs/
├── scripts/
├── src/
├── tests/
├── checkpoints/
├── logs/
├── results/
├── pyproject.toml
└── requirements.txt
```

Key script entrypoints:

- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/render_first_eval_episode.py`
- `scripts/plot_results.py`
- `scripts/smoke_test.py`

## Current Training Stack

The active stack now supports:

- per-stage curriculum overrides
- deterministic periodic evaluation during training
- model-only resume with optional optimizer reset
- stage-local best checkpoints
- regression detection and rollback
- plateau recovery with optional demo replay
- behavior-cloning warm start from the heuristic teacher
- teacher-demo pretraining with success filtering
- post-pretrain evaluation capture
- optional post-pretrain action-std reduction
- richer terminal logging for training, evaluation, and pretraining

## Current Limits

- single UAV only
- fixed obstacle count within a curriculum
- analytic moving-sphere obstacles, not full multi-agent traffic simulation
- no maps, no-fly zones, or airspace rules
- no confirmed stable `target_default` solution yet

## Docs

The current project snapshot is documented in:

- `docs/current_status_report.md`
- `docs/rl_status_report_2026-04-06.md`
- `docs/architecture.md`
- `docs/experiment_workflow.md`
- `docs/training_upgrade_summary.md`
- `docs/using_repo_for_learning_based_uav_path_planning.md`

## Near-Term Direction

- preserve the graph observation contract
- keep the train/eval workflow reusable
- stabilize target-stage behavior around lateral bypass
- reach a reliable `target_default` continuation from the current intro checkpoint sources
