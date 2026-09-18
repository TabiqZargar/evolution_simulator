"""Mutation and crossover must stay in bounds and be heritable."""

from __future__ import annotations

import random

from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.genetics import crossover, mutate
from evolution_sim.simulation.genome import PREY_TRAITS, Genome


def _rng() -> random.Random:
    return random.Random(11)


def test_mutate_stays_in_bounds() -> None:
    cfg = SimulationConfig(mutation_rate=1.0, mutation_strength=1.0)
    rng = _rng()
    original = Genome.random(rng, PREY_TRAITS)
    from evolution_sim.simulation.genome import trait_bounds

    for _ in range(200):
        mutated = mutate(original, cfg, rng)
        assert mutated.is_valid()
        for key in PREY_TRAITS:
            lo, hi = trait_bounds(key)
            assert lo <= mutated.gene(key) <= hi


def test_zero_rate_means_no_change() -> None:
    # mutation rate is heritable: anchor the gene at 0 and the config at 0
    cfg = SimulationConfig(mutation_rate=0.0)
    rng = _rng()
    original = Genome.random(rng, PREY_TRAITS).with_gene("mutation_rate", 0.0)
    mutated = mutate(original, cfg, rng)
    for key in PREY_TRAITS:
        assert mutated.gene(key) == original.gene(key)


def test_config_rate_is_a_floor_contribution() -> None:
    # a zero-rate genome still mutates when the config baseline is high
    cfg = SimulationConfig(mutation_rate=1.0)
    rng = _rng()
    original = Genome.random(rng, PREY_TRAITS).with_gene("mutation_rate", 0.0)
    changed = sum(
        mutate(original, cfg, rng).gene(k) != original.gene(k) for k in PREY_TRAITS
    )
    assert changed > 0


def test_crossover_mixes_genes() -> None:
    cfg = SimulationConfig(crossover_mode="uniform")
    rng = _rng()
    g1 = Genome.random(rng, PREY_TRAITS, center={"speed": 0.95})
    g2 = Genome.random(rng, PREY_TRAITS, center={"speed": 0.05})
    child = crossover(g1, g2, cfg, rng)
    assert child.gene("speed") in (g1.gene("speed"), g2.gene("speed"))
    assert child.is_valid()


def test_blend_crossover_between_parents() -> None:
    cfg = SimulationConfig(crossover_mode="blend", crossover_blend_alpha=0.0)
    rng = _rng()
    g1 = Genome.random(rng, PREY_TRAITS, center={"speed": 0.9})
    g2 = Genome.random(rng, PREY_TRAITS, center={"speed": 0.1})
    child = crossover(g1, g2, cfg, rng)
    low = min(g1.gene("speed"), g2.gene("speed"))
    high = max(g1.gene("speed"), g2.gene("speed"))
    assert low - 1e-9 <= child.gene("speed") <= high + 1e-9
