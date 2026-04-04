# Changelog

All notable changes to this project are recorded here.

## [1.0.2] - 2026-04-04

- Added model-only resume controls, stage-local best checkpoints, regression rollback, and richer training/evaluation reporting.
- Added route-biased scenario sampling plus stronger reward and teacher shaping for bridge, medium, and target-stage curricula.
- Added the new experiment presets from `curriculum_goal_first_v2` through `post_commit_target_v10` and refreshed the repo docs around the current RL status.

## [1.0.1] - 2026-04-04

- Added the working corridor bootstrap and the first stage-aware curriculum/evaluation workflow.
- Improved diagnostics, plotting, rendering, and repository documentation.

## [1.0.0] - 2026-04-03

- Promoted the project metadata to `1.0.0` and stabilized the public train/eval/plot workflow.
- Added release-ready configs, periodic evaluation summaries, and expanded safety/control metrics.

## [0.2.0] - 2026-03-28

- Expanded test coverage across helpers, exports, and script entrypoints.
- Tightened repository validation and local version-control workflow.

## [0.1.0] - 2026-03-28

- Created the initial scaffold with packaging, configs, docs, tests, and plotting utilities.
- Split the codebase into environment, agent, training, evaluation, and visualization modules.
