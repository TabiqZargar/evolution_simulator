"""Configuration parsing, coercion, and CLI override helpers."""

from __future__ import annotations

import json

import pytest

from evolution_sim.config import (
    FIELD_GROUPS,
    SIMULATION_CONFIG_FIELDS,
    SimulationConfig,
    _coerce,
    apply_cli_overrides,
    field_group,
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


# --------------------------------------------------------------------------- #
# Field groups
# --------------------------------------------------------------------------- #
def test_field_groups_cover_every_field() -> None:
    covered = {member for members in FIELD_GROUPS.values() for member in members}
    assert covered == set(SIMULATION_CONFIG_FIELDS)


def test_field_groups_have_unique_members() -> None:
    seen: set[str] = set()
    for members in FIELD_GROUPS.values():
        for member in members:
            assert member not in seen, f"field {member} listed in multiple groups"
            seen.add(member)
        assert members


def test_field_group_lookup() -> None:
    assert field_group("seed") == "population"
    assert field_group("event_log_capacity") == "statistics"
    assert field_group("mutation_rate") == "genetics"
    assert field_group("nonsense") == ""


# --------------------------------------------------------------------------- #
# Extended validation
# --------------------------------------------------------------------------- #
def test_reproduction_threshold_must_fit_max_energy() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(reproduction_energy_threshold=200.0, max_energy=120.0)


def test_reproduction_fractions_in_range() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(reproduction_energy_cost_fraction=1.5)
    with pytest.raises(ValueError):
        SimulationConfig(offspring_energy_fraction=-0.1)


def test_lifespan_bounds_consistent() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(lifespan_min_ticks=1000, lifespan_max_ticks=500)


def test_vision_range_ge_vision_base() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(vision_base=8.0, vision_range=3.0)


def test_geometry_and_capacity_positive() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(spatial_cell_size=0)
    with pytest.raises(ValueError):
        SimulationConfig(eating_radius=0)
    with pytest.raises(ValueError):
        SimulationConfig(event_log_capacity=-1)
    with pytest.raises(ValueError):
        SimulationConfig(dead_organism_budget=-5)


def test_event_frequency_range() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(event_frequency=1.5)


def test_initial_population_cannot_exceed_max() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(initial_population=3000, max_population=2000)


def test_predator_population_sanity() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(predators_enabled=True, predator_max_population=5, predator_initial_population=20)


def test_water_penalty_at_least_one() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(terrain_movement_water_penalty=0.5)


def test_event_log_capacity_roundtrips(tmp_path) -> None:
    cfg = SimulationConfig(event_log_capacity=77)
    path = tmp_path / "cfg.json"
    save_config(cfg, path)
    assert load_config(path).event_log_capacity == 77
