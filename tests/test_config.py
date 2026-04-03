"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import (
    Config,
    diagnose_torch_device_resolution,
    load_config,
    resolve_torch_device,
    save_config,
    validate_config,
)


def test_load_partial_config_uses_defaults(tmp_path: Path) -> None:
    """Partial YAML configs should load cleanly on top of dataclass defaults."""
    config_path = tmp_path / "partial.yaml"
    config_path.write_text(
        "\n".join(
            [
                'name: "partial_config"',
                "environment:",
                "  num_dynamic_obstacles: 2",
                "agent:",
                '  encoder_type: "mlp"',
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.name == "partial_config"
    assert config.environment.num_dynamic_obstacles == 2
    assert config.agent.encoder_type == "mlp"
    assert config.environment.max_episode_steps == 800


def test_validate_config_rejects_bad_encoder() -> None:
    """Validation should reject unsupported encoder selections."""
    config = Config()
    config.agent.encoder_type = "transformer"
    with pytest.raises(ValueError, match="encoder_type"):
        validate_config(config)


def test_save_config_round_trip_and_device_helpers(tmp_path: Path) -> None:
    """Config save/load and device helper behavior should be stable."""
    config = Config()
    config.name = "round_trip"
    config_path = tmp_path / "saved.yaml"

    save_config(config, config_path)
    loaded = load_config(config_path)

    assert loaded.name == "round_trip"
    assert resolve_torch_device("cpu") == "cpu"
    assert diagnose_torch_device_resolution("cpu") is None
