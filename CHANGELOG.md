# Changelog

All notable changes to this project will be documented in this file.

## [1.0.1] - 2026-04-04

### Changed
- Added the working corridor bootstrap and the stage-aware curriculum/evaluation workflow.
- Updated training diagnostics, plotting, rendering, and repository docs to match the current experiments.

## [1.0.0] - 2026-04-03

### Added
- Release-ready training presets for easier and harder obstacle regimes via `configs/easy_train.yaml` and `configs/hard_obstacles.yaml`.
- Periodic evaluation tracking during training, including per-evaluation summaries and saved best-trajectory artifacts.
- Expanded safety and control metrics such as clearance, control effort, episode duration, and time-to-goal reporting across training and evaluation flows.

### Changed
- Promoted the project metadata and config surface to `1.0.0`.
- Reworked the default and debug PPO configuration values around longer training runs, stronger reward shaping, and clearer smoke-test behavior.
- Updated the policy, environment, metrics, and visualization pipeline to emphasize safer action outputs, best-checkpoint selection, and more informative plots.
- Consolidated public repository documentation into `README.md` while keeping local `docs/` content ignored from Git tracking.

## [0.2.0] - 2026-03-28

### Added
- A full bug-oriented test pass across helper modules, package exports, and script entrypoints.
- Initial local Git repository metadata on the `main` branch for version-controlled development.

### Changed
- Bumped project metadata from `0.1.0` to `0.2.0` across the repo.
- Tightened repository validation so the broader module surface is covered by the automated test suite.

## [0.1.0] - 2026-03-28

### Added
- `pyproject.toml` with modern packaging metadata and development extras.
- A first repository hardening pass with docs, tests, CI, config presets, and plotting utilities.
- Architecture, observation, metrics, and experiment workflow documentation under `docs/`.
- Baseline and debug configuration presets under `configs/`.

### Changed
- Split large environment, agent, and training modules into smaller helper modules while preserving the existing public entrypoints.
- Expanded the README to describe the repository layout, development workflow, and supported experiment surfaces.
- Switched the `gym-pybullet-drones` dependency to the upstream GitHub source so the documented install path works.

### Cleaned
- Tightened `.gitignore` for build, cache, coverage, and runtime-generated artifacts.
- Removed tracked runtime artifacts and Python cache files from the working tree.
