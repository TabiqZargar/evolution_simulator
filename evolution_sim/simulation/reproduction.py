"""Sexual reproduction: mate finding, energy costs and offspring creation."""

from __future__ import annotations

from random import Random

from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.genetics import crossover, mutate_with_count
from evolution_sim.simulation.genome import Genome
from evolution_sim.simulation.organism import Organism


def can_reproduce(organism: Organism, config: SimulationConfig) -> bool:
    return (
        organism.alive
        and organism.age >= config.min_age_to_reproduce
        and organism.reproduction_cooldown <= 0
        and organism.energy >= config.reproduction_energy_threshold
    )


def find_mate(
    organism: Organism,
    candidates: list[Organism],
    config: SimulationConfig,
    rng: Random,
) -> Organism | None:
    """Pick an eligible mate among ``candidates`` (same kind), biasing towards
    nearer partners and adding a little randomness to avoid pathological
    nearest-neighbour lock-in. Returns None if nobody is eligible."""
    eligible = [
        c
        for c in candidates
        if c.alive
        and c is not organism
        and c.is_predator == organism.is_predator
        and can_reproduce(c, config)
        and c.age >= config.min_age_to_reproduce
    ]
    if not eligible:
        return None
    eligible.sort(key=lambda c: (organism.distance_to(c), c.organism_id))
    top = eligible[:3]
    return rng.choice(top)


def create_offspring(
    mother: Organism,
    father: Organism,
    config: SimulationConfig,
    rng: Random,
) -> tuple[Genome, float, float]:
    """Return ``(genome, energy, total_cost)`` for a new offspring.

    Energy is a fraction of the parents' *residual* energy after both have
    paid the reproduction cost, with a floor so newborns are never born
    starving. The genome is a crossover of the parents, then mutated with a
    rate blended from both parents' mutation_rate genes and the configured
    baseline.
    """
    offspring = create_offspring_with_lifecycle(mother, father, config, rng)
    return offspring[0], offspring[1], offspring[2]


def create_offspring_with_lifecycle(
    mother: Organism,
    father: Organism,
    config: SimulationConfig,
    rng: Random,
) -> tuple[Genome, float, float, int]:
    """Like :func:`create_offspring`, also returning the mutation count.

    The extra return value is observational metadata for the simulation
    ledger; the random number stream consumed is identical, so callers do not
    need to worry about determinism differences.
    """
    genome = crossover(mother.genome, father.genome, config, rng)
    genome, mutations = mutate_with_count(genome, config, rng)

    cost_mother = mother.energy * config.reproduction_energy_cost_fraction
    cost_father = father.energy * config.reproduction_energy_cost_fraction
    mother.energy -= cost_mother
    father.energy -= cost_father
    mother.children += 1
    father.children += 1
    mother.reproduction_cooldown = config.reproduction_cooldown
    father.reproduction_cooldown = config.reproduction_cooldown

    residual = (mother.energy + father.energy) / 2.0
    energy = max(
        residual * config.offspring_energy_fraction,
        config.offspring_energy_floor,
    )
    return genome, energy, cost_mother + cost_father, mutations


def offspring_position(
    mother: Organism, father: Organism, rng: Random, config: SimulationConfig
) -> tuple[float, float]:
    """Spawn the child near the mother (slightly jittered)."""
    spread = config.mating_range * 0.4
    dx = (rng.random() * 2.0 - 1.0) * spread
    dy = (rng.random() * 2.0 - 1.0) * spread
    return mother.x + dx, mother.y + dy
