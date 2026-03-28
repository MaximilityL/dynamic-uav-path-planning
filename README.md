# Dynamic UAV Path Planning

Version: `0.2.0`

This repository is a research scaffold for learning-based UAV path planning in dynamic environments. It currently focuses on a graph-based observation pipeline, a PyBullet-backed single-UAV environment, and a lightweight PPO baseline that is easy to extend.

The intent is still infrastructure-first: make the repo clean, testable, and ready for iteration before claiming that the underlying path-planning problem is solved.

## Focus

- learning-based UAV path planning
- dynamic environments with moving obstacles
- model-free reinforcement learning
- graph neural network-ready policies

## What The Repo Includes

- a PyBullet-backed dynamic airspace environment
- dense graph observations with node, edge, and global features
- a PPO-style graph actor-critic baseline in plain PyTorch
- train, evaluate, smoke-test, and plotting entrypoints
- config presets for default, debug, and ablation workflows
- tests, linting, CI, and lightweight docs for repository maintainability

## Current Scope

This scaffold currently supports:

- one controlled UAV
- one goal node
- configurable moving obstacle nodes
- graph observations with a stable contract
- repeatable train/eval artifact generation

This scaffold does not yet include:

- multi-UAV coordination logic
- urban map assets or airspace rules
- benchmark-complete evaluation suites
- tuned training behavior or benchmark claims
- mature experiment tracking dashboards

## Design Choices

- PyBullet simulation is kept through `gym-pybullet-drones`.
- The graph observation contract is treated as the main stable interface.
- The current baseline is deliberately simple and readable instead of benchmark-optimized.
- Graph message passing is implemented in plain PyTorch so the project can later move to PyG or a different graph stack without redesigning the repo.

## Repository Layout

```text
dynamic-uav-path-planning/
├── CHANGELOG.md
├── configs/
│   ├── baselines/
│   ├── scenarios/
│   ├── debug.yaml
│   └── default.yaml
├── docs/
├── scripts/
│   ├── _common.py
│   ├── evaluate.py
│   ├── plot_results.py
│   ├── smoke_test.py
│   └── train.py
├── src/
│   ├── agents/
│   ├── core/
│   ├── environments/
│   ├── evaluation/
│   ├── training/
│   ├── utils/
│   └── visualization/
├── tests/
├── .github/
├── checkpoints/
├── logs/
├── results/
├── pyproject.toml
├── README.md
├── VERSION
└── requirements.txt
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python scripts/smoke_test.py
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --model checkpoints/best_model.pth
python scripts/plot_results.py
```

`gym-pybullet-drones` is installed from the upstream GitHub repository because it is not resolved from PyPI in this scaffold setup.

## Development Workflow

```bash
ruff check .
pytest
```

Useful config presets:

- `configs/default.yaml`
- `configs/debug.yaml`
- `configs/baselines/mlp.yaml`
- `configs/scenarios/dense_obstacles.yaml`

## Documentation

- `docs/architecture.md`
- `docs/observation_spec.md`
- `docs/metrics.md`
- `docs/experiment_workflow.md`

## Near-Term Direction

- preserve the graph observation contract
- keep train/eval entrypoints stable
- make environment and training internals easier to swap
- add stronger experiment management before adding benchmark claims
