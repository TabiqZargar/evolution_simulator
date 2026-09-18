"""Organism phenotype caching and life ledger."""

from __future__ import annotations

import random

from evolution_sim.simulation.genome import Genome
from evolution_sim.simulation.organism import Organism


def _org() -> Organism:
    genome = Genome.random(random.Random(5), ("speed", "size", "attack"))
    return Organism(organism_id=1, genome=genome, x=0.0, y=0.0)


def test_is_predator_from_genome() -> None:
    prey = Organism(organism_id=1, genome=Genome({"speed": 0.5}), x=0.0, y=0.0)
    predator = Organism(
        organism_id=2, genome=Genome({"speed": 0.5, "attack": 0.8}), x=0.0, y=0.0
    )
    assert not prey.is_predator
    assert predator.is_predator
    assert predator.attack == 0.8


def test_unknown_trait_default() -> None:
    org = _org()
    assert org.trait("nonexistent", 0.25) == 0.25


def test_death_records() -> None:
    org = _org()
    org.record_death("starvation", final_fitness=0.5)
    assert not org.alive
    assert org.death_cause == "starvation"
    assert org.final_fitness == 0.5


def test_distance() -> None:
    a = Organism(organism_id=1, genome=Genome({"speed": 0.5}), x=0.0, y=0.0)
    b = Organism(organism_id=2, genome=Genome({"speed": 0.5}), x=3.0, y=4.0)
    assert a.distance_to(b) == 5.0
