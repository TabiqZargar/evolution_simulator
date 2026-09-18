"""Reproduction mechanics: eligibility, mate choice, offspring creation."""

from __future__ import annotations

import random

from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.genome import Genome
from evolution_sim.simulation.organism import Organism
from evolution_sim.simulation.reproduction import can_reproduce, create_offspring, find_mate


def _mk(oid: int, energy: float, age: int = 50, predator: bool = False) -> Organism:
    traits = ("attack",) if predator else ()
    genome = Genome.random(random.Random(oid), ("speed", "vision", "metabolism") + traits)
    return Organism(organism_id=oid, genome=genome, x=float(oid), y=0.0, energy=energy, age=age)


def test_can_reproduce_thresholds() -> None:
    cfg = SimulationConfig(reproduction_energy_threshold=55.0, min_age_to_reproduce=40)
    hungry = _mk(1, energy=20, age=50)
    young = _mk(2, energy=90, age=10)
    cooldown = _mk(3, energy=90, age=50)
    cooldown.reproduction_cooldown = 5
    ready = _mk(4, energy=90, age=50)
    assert not can_reproduce(hungry, cfg)
    assert not can_reproduce(young, cfg)
    assert not can_reproduce(cooldown, cfg)
    assert can_reproduce(ready, cfg)


def test_find_mate_same_kind_only() -> None:
    cfg = SimulationConfig(mating_range=10.0)
    me = _mk(1, energy=80.0, predator=True)
    same = _mk(2, energy=80.0, predator=True)
    different = _mk(3, energy=80.0, predator=False)

    mate = find_mate(me, [same, different], cfg, random.Random(1))
    assert mate is same


def test_create_offspring_cost_and_energy_floor() -> None:
    cfg = SimulationConfig(
        reproduction_energy_cost_fraction=0.25,
        offspring_energy_fraction=0.5,
        offspring_energy_floor=40.0,
    )
    mother = _mk(1, energy=80.0)
    father = _mk(2, energy=80.0)
    genome, energy, total_cost = create_offspring(mother, father, cfg, random.Random(2))
    assert genome.is_valid()
    assert energy >= cfg.offspring_energy_floor - 1e-9
    assert total_cost == 2 * 80.0 * 0.25
    assert mother.energy == 80.0 - 20.0
    assert mother.reproduction_cooldown == cfg.reproduction_cooldown
    assert mother.children == 1 and father.children == 1
