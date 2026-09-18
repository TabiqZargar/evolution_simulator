"""Configuration parsing, coercion, and CLI override helpers."""

from __future__ import annotations

import json

import pytest

from evolution_sim.config import (
    SIMULATION_CONFIG_FIELDS,
    SimulationConfig,
    _coerce,
    apply_cli_overrides,
    load_config,
    save_config,
)


def test_defaults_validate() -> None:
    SimulationConfig().validate()


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(mutation_rate=2.0)
    with pytest.raises(ValueError):
        SimulationConfig(world_width=0)
    with pytest.raises(ValueError):
        SimulationConfig(crossover_mode="gauss")


def test_json_roundtrip(tmp_path) -> None:
    cfg = SimulationConfig(seed=8, initial_population=120, resource_density=0.05)
    path = tmp_path / "cfg.json"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded == cfg


def test_load_ignores_unknown_keys(tmp_path) -> None:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"initial_population": 10, "nonsense_key": True}), encoding="utf-8")
    loaded = load_config(path)
    assert loaded.initial_population == 10


def test_apply_cli_overrides_coerces_types() -> None:
    cfg = apply_cli_overrides(
        SimulationConfig(),
        [("initial_population", "250"), ("mutation_rate", "0.2"), ("predators_enabled", "true")],
    )
    assert cfg.initial_population == 250
    assert cfg.mutation_rate == 0.2
    assert cfg.predators_enabled is True


def test_apply_cli_overrides_unknown_field() -> None:
    with pytest.raises(ValueError):
        apply_cli_overrides(SimulationConfig(), [("bogus", "1")])


def test_coerce_handles_optional_ints() -> None:
    assert _coerce("Optional[int]", "5") == 5
    assert _coerce("float | None", "0.5") == 0.5


def test_field_registry_known() -> None:
    assert "seed" in SIMULATION_CONFIG_FIELDS
    assert "initial_population" in SIMULATION_CONFIG_FIELDS


def test_with_updates_frozen() -> None:
    cfg = SimulationConfig(seed=1)
    nested = cfg.with_updates(seed=2)
    assert cfg.seed == 1
    assert nested.seed == 2
