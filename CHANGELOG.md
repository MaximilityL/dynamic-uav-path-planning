# Changelog

All notable changes to this project are recorded here.

## [1.1.2] - 2026-04-06

- Reworked the main README around the presentation narrative, added a quick `default.yaml` replication guide, and embedded the most useful default-run plots and showcase figures.
- Updated the default train and evaluate CLI entrypoints to use `configs/default.yaml`, promoted the project metadata to `1.1.2`, and started tracking checkpoints and results directly in git.

## [1.1.1] - 2026-04-06

- Added the stage showcase exporter with 2D and 3D trajectory plots plus GUI video capture for the best reached and wanted stages.
- Switched showcase episode selection to prefer successful goal reaches first, then minimum remaining goal distance, and kept legacy checkpoint loading compatible with the expanded observation features.

## [1.1.0] - 2026-04-06

- Promoted the retained bypass-v3 curriculum to `configs/default.yaml`, moved its kept run artifacts under the `default` output paths, and refreshed the default plots.
- Removed the extra bypass config variants and their stale checkpoints, results, and logs.

## [1.0.6] - 2026-04-06

- Added the new bypass curriculum presets, including `default_curriculum_bypass_v3`, plus rejoin-aware reward and teacher guidance so target-stage policies are pushed to turn back toward the route instead of staying committed to one heading.
- Improved zero-success recovery and checkpoint selection for target stages, and refreshed the project metadata to `1.0.6`.

## [1.0.4] - 2026-04-06

- Renamed the active training presets to `default_curriculum.yml` and `default_post_commit.yaml`, updated the default CLI config paths, and isolated helper presets into dedicated output folders.
- Pruned historical configs and stale experiment artifacts while preserving the `v6` curriculum stage checkpoints under the new default layout.

## [1.0.3] - 2026-04-06

- Added target-stage behavior-cloning warm starts, success-filtered teacher-demo pretraining, post-pretrain evaluation capture, and lower-noise continuation controls for the target curriculum.
- Added the new experiment presets from `curriculum_goal_first_v4` through `post_commit_target_v15`, plus richer startup/pretrain logging and stronger target-stage rollback and plateau-recovery support.

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
