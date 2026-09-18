"""Fitness model.

Fitness is a *reporting* metric derived from what actually happened to an
organism, not a score the engine optimises against. Selection happens purely
through survival (starvation, thermal stress, old age, predation) and
non-random reproduction (only well-fed, mature organisms reproduce). Fitness
then *summarises* reproductive success so we can watch evolution happen:

    fitness = 0.40 * fecundity     (actual offspring produced vs. an expectation)
            + 0.30 * lineage       (offspring that still wander the world)
            + 0.20 * longevity     (age achieved vs. genome-implied lifespan)
            + 0.10 * efficiency    (food harvested per unit of energy spent)

Every component is a normalised, bounded quantity in [0, 1], so fitness is a
weighted average in [0, 1]. Raw per-organism records (children, age, energy
ledger, fights) are kept for transparency.
"""

from __future__ import annotations

from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.organism import Organism

_W_FECUNDITY = 0.40
_W_LINEAGE = 0.30
_W_LONGEVITY = 0.20
_W_EFFICIENCY = 0.10

# Units for the normalising expectations.
_EXPECTED_CHILDREN = 4.0
_EFFICIENCY_SCALE = 0.10  # energy_harvested / (energy_spent + eps) == 0.1 scores 1.0


def expected_lifespan(organism: Organism, config: SimulationConfig) -> int:
    trait = organism.lifespan_trait
    return config.lifespan_min_ticks + int(
        round((config.lifespan_max_ticks - config.lifespan_min_ticks) * trait)
    )


def fecundity_score(organism: Organism) -> float:
    return min(1.0, organism.children / _EXPECTED_CHILDREN)


def lineage_score(organism: Organism, living_children: int) -> float:
    return min(1.0, living_children / _EXPECTED_CHILDREN)


def longevity_score(organism: Organism, config: SimulationConfig) -> float:
    expected = expected_lifespan(organism, config)
    if expected <= 0:
        return 0.0
    return min(1.0, organism.age / expected)


def efficiency_score(organism: Organism) -> float:
    spent = organism.energy_spent + 1e-9
    raw = organism.energy_harvested / spent
    return min(1.0, raw / _EFFICIENCY_SCALE)


def compute_fitness(
    organism: Organism,
    config: SimulationConfig,
    living_children_count: int = 0,
) -> float:
    """Weighted fitness in [0, 1] for a single organism."""
    fit = (
        _W_FECUNDITY * fecundity_score(organism)
        + _W_LINEAGE * lineage_score(organism, living_children_count)
        + _W_LONGEVITY * longevity_score(organism, config)
        + _W_EFFICIENCY * efficiency_score(organism)
    )
    return min(max(fit, 0.0), 1.0)


def population_fitness(organisms: list[Organism], config: SimulationConfig) -> float:
    """Average fitness across ``organisms`` (dead or alive both accepted)."""
    if not organisms:
        return 0.0
    total = 0.0
    for org in organisms:
        total += compute_fitness(org, config)
    return total / len(organisms)
