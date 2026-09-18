"""Spatial index and world construction tests."""

from __future__ import annotations

import random
from dataclasses import dataclass

from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.genome import Genome
from evolution_sim.simulation.world import SpatialGrid, World


@dataclass
class _Point:
    x: float
    y: float
    label: str


def test_grid_insert_query() -> None:
    grid = SpatialGrid(2.0)
    grid.add(_Point(0.5, 0.5, "a"))
    grid.add(_Point(4.0, 4.0, "b"))
    grid.add(_Point(20.0, 20.0, "c"))
    near = grid.query(0.5, 0.5, 1.5)
    assert any(item.label == "a" for item in near)
    assert all(not item.label == "c" for item in near)


def test_grid_nearest_filters() -> None:
    grid = SpatialGrid(2.0)
    grid.add(_Point(1.0, 1.0, "a"))
    grid.add(_Point(10.0, 10.0, "b"))
    best = grid.nearest(0.0, 0.0, 20.0, predicate=lambda item: item.label == "b")
    assert best is not None and best.label == "b"


def test_world_spawns_and_indexes() -> None:
    cfg = SimulationConfig(world_width=50, world_height=50, seed=3)
    world = World(cfg, random.Random(3))
    world.generate_food()
    assert len(world.patches) > 0
    assert world.total_food() > 0

    genome = Genome.random(random.Random(4), ("speed", "vision"))
    org = world.spawn_organism(genome, generation=1, energy=50.0)
    world.rebuild_indices()
    found = world.org_grid.query(org.x, org.y, 1.0)
    assert org in found

    # food grid was indexed once at generate_food
    near_food = world.food_grid.query(org.x, org.y, 100.0)
    assert near_food

    # dead organisms remain in the list until a render/query needs them
    org.alive = False
    world.rebuild_indices()
    assert org in world.organisms
