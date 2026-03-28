#!/usr/bin/env python3
"""Generate training plots from saved JSONL history."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import bootstrap_project

bootstrap_project()

from src.visualization import plot_training_history


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot saved training metrics")
    parser.add_argument("--history", type=str, default="results/train/history.jsonl", help="Training history JSONL file")
    parser.add_argument("--output-dir", type=str, default="results/plots", help="Directory for generated plots")
    args = parser.parse_args()

    output_path = plot_training_history(history_path=args.history, output_dir=args.output_dir)
    print(f"plot={Path(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
