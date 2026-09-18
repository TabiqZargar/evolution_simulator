"""Tests for the genome model."""

from __future__ import annotations

import random

from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.genome import (
    PREDATOR_TRAITS,
    PREY_TRAITS,
    TRAIT_SPECS,
    Genome,
    trait_bounds,
)


def test_prey_has_no_attack() -> None:
    assert "attack" not in PREY_TRAITS
    assert "attack" in PREDATOR_TRAITS
    assert set(PREY_TRAITS) < set(PREDATOR_TRAITS)


def test_random_genome_is_valid_and_bounded() -> None:
    rng = random.Random(1)
    for _ in range(300):
        g = Genome.random(rng, PREY_TRAITS)
        assert g.is_valid()
        for key, (lo, hi) in TRAIT_SPECS.items():
            if key in PREY_TRAITS:
                assert lo <= g.gene(key) <= hi


def test_random_respects_center() -> None:
    rng = random.Random(2)
    center = {"speed": 0.9, "vision": 0.1}
    values = [Genome.random(rng, PREY_TRAITS, center=center).gene("speed") for _ in range(200)]
    # the center contributes ~50% of the mix, so the mean lands near 0.7
    assert 0.6 < sum(values) / len(values) < 0.8


def test_gene_default_for_missing() -> None:
    g = Genome({"speed": 0.5})
    assert g.gene("attack", 0.25) == 0.25


def test_genome_is_immutable() -> None:
    g = Genome({"speed": 0.5})
    with_test = g.with_gene("speed", 0.9)
    assert g.gene("speed") == 0.5
    assert with_test.gene("speed") == 0.9
    try:
        g.traits["speed"] = 0.9  # type: ignore[index]
        raise AssertionError("mutation succeeded")
    except TypeError:
        pass


def test_distance_matches_manhattan() -> None:
    a = Genome({"speed": 0.0, "vision": 0.0, "size": 0.5, "metabolism": 0.5, "efficiency": 0.5,
                "fertility": 0.5, "lifespan": 0.5, "aggression": 0.5, "temperature_tolerance": 0.5,
                "mutation_rate": 0.05})
    b = a.with_gene("speed", 1.0).with_gene("vision", 1.0)
    # distance is measured over the diversity trait subset (6 genes)
    assert abs(a.distance_to(b) - 2.0 / 6) < 1e-9


def test_config_mutation_rate_bounds() -> None:
    cfg = SimulationConfig(mutation_rate=0.5, gene_mutation_rate_max=0.5)
    rng = random.Random(3)
    g = Genome.random(rng, PREY_TRAITS, center={"mutation_rate": 0.5})
    assert g.gene("mutation_rate") <= cfg.gene_mutation_rate_max


def test_trait_bounds_known() -> None:
    assert trait_bounds("speed") == (0.0, 1.0)
    assert trait_bounds("mutation_rate") == (0.0, 0.5)
