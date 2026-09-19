"""End-to-end engine behaviour, determinism, and invariants."""

from __future__ import annotations

import random

import pytest
from conftest import assert_finite

from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.engine import Engine
from evolution_sim.simulation.environment import Biome
from evolution_sim.simulation.genome import Genome


def test_engine_init_populates() -> None:
    cfg = SimulationConfig(world_width=30, world_height=30, initial_population=50)
    eng = Engine(cfg)
    eng.init()
    assert eng.alive_count() == 50
    assert eng.tick == 0
    assert eng.generation == 1


def test_step_advances_state() -> None:
    eng = Engine(SimulationConfig(seed=1, initial_population=50, generation_length=500))
    eng.init()
    before = eng.state_signature()
    eng.step()
    assert eng.tick == 1
    assert eng.state_signature() != before


def test_determinism_ticks() -> None:
    cfg = SimulationConfig(seed=9, initial_population=80, generation_length=400)
    a = Engine(cfg.clone())
    a.init()
    b = Engine(cfg.clone())
    b.init()
    for _ in range(200):
        a.step()
        b.step()
    assert a.state_signature() == b.state_signature()


def test_determinism_generations() -> None:
    cfg = SimulationConfig(seed=9, initial_population=60, generation_length=150)
    a = Engine(cfg)
    a.init()
    a.run_generations(3)
    b = Engine(cfg)
    b.init()
    b.run_generations(3)
    assert a.state_signature() == b.state_signature()
    assert len(a.history) == len(b.history) == 3
    assert a.generation == b.generation == 4


def test_invariants_over_generations() -> None:
    eng = Engine(SimulationConfig(seed=3, initial_population=80, generation_length=150))
    eng.init()
    eng.run_generations(3)
    assert_finite(eng)
    assert len(eng.history) == 3


def test_replenish_restores_population() -> None:
    eng = Engine(SimulationConfig(seed=5, initial_population=20, target_population=60, generation_length=200))
    eng.init()
    eng.run_generations(2)
    last = eng.history.last()
    assert last is not None and last.population >= 60


def test_predators_coexist_with_prey() -> None:
    cfg = SimulationConfig(
        seed=2,
        world_width=40,
        world_height=40,
        initial_population=80,
        target_population=80,
        predators_enabled=True,
        predator_initial_population=10,
        predator_max_population=30,
        generation_length=150,
    )
    eng = Engine(cfg)
    eng.init()
    eng.run_generations(3)
    assert_finite(eng)
    hist = eng.history.entries[-1]
    assert hist.predator_population > 0 and hist.prey_population > 0


def test_species_detection_runs() -> None:
    eng = Engine(SimulationConfig(seed=4, initial_population=60, generation_length=200))
    eng.init()
    eng.run_generations(2)
    assert eng.species_list() is not None
    assert eng.species_list()  # non-empty while alive


def test_organism_ids_monotonic() -> None:
    eng = Engine(SimulationConfig(seed=77, initial_population=20, generation_length=300))
    eng.init()
    ids = [o.organism_id for o in eng.world.organisms]
    assert ids == sorted(ids)
    eng.run_generations(2)
    ids = [o.organism_id for o in eng.world.organisms]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


def test_water_movement_costs_more_than_grass() -> None:
    cfg = SimulationConfig(seed=11, world_width=40, world_height=40, initial_population=0)
    eng = Engine(cfg)
    eng.init()
    terrain = eng.world.terrain

    water_cell = next(
        (cx, cy)
        for cy in range(terrain.height)
        for cx in range(terrain.width)
        if terrain.biome_at(cx, cy) == Biome.WATER
    )
    grass_cell = next(
        (cx, cy)
        for cy in range(terrain.height)
        for cx in range(terrain.width)
        if terrain.biome_at(cx, cy) == Biome.GRASS
    )
    assert water_cell != grass_cell

    genome = Genome.random(random.Random(7), ("speed", "vision"))
    org_water = eng.world.spawn_organism(
        genome, generation=1, energy=50.0, x=water_cell[0] + 0.5, y=water_cell[1] + 0.5
    )
    org_grass = eng.world.spawn_organism(
        genome, generation=1, energy=50.0, x=grass_cell[0] + 0.5, y=grass_cell[1] + 0.5
    )

    eng._apply_movement(org_water, 0.25, 0.0)
    eng._apply_movement(org_grass, 0.25, 0.0)

    cost_water = org_water.energy_spent
    cost_grass = org_grass.energy_spent
    assert cost_water > cost_grass
    # identical step and speed; only the terrain penalty differs
    assert cost_water / cost_grass == pytest.approx(cfg.terrain_movement_water_penalty, rel=1e-6)
    # the penalty affects energy cost only — movement is otherwise unchanged
    assert org_water.x == water_cell[0] + 0.75 and org_water.y == water_cell[1] + 0.5
    assert org_grass.x == grass_cell[0] + 0.75 and org_grass.y == grass_cell[1] + 0.5
    assert org_water.energy < org_grass.energy
