"""Genetic operators: crossover and mutation.

Reproduction inherits a child genome from two parents:

    parent A genome + parent B genome
        |   crossover ("uniform" or "blend")
        v
    child genome (pre-mutation)
        |   mutation (continuous variation + occasional point mutation)
        v
    child genome (post-mutation)

Mutations are deliberately *not* biased towards improvement. They perturb
genes probabilistically; whether the change is beneficial, neutral or harmful
depends entirely on the environment.
"""

from __future__ import annotations

from random import Random

from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.genome import Genome, clamp_trait, trait_bounds


def crossover(parent_a: Genome, parent_b: Genome, config: SimulationConfig, rng: Random) -> Genome:
    """Produce a child genome from two parents.

    * ``uniform``: each gene is copied from a randomly chosen parent.
    * ``blend``:   each gene is an interpolation of the parents' values, using
      ``crossover_blend_alpha`` as the mix ratio.
    """
    keys: set[str] = set(parent_a.traits) | set(parent_b.traits)
    if not keys:
        return parent_a

    if config.crossover_mode == "uniform":
        genes: dict[str, float] = {}
        for key in keys:
            pick = parent_a if rng.random() < 0.5 else parent_b
            genes[key] = pick.gene(key)
        return Genome(genes)

    # blend mode
    alpha = config.crossover_blend_alpha
    genes = {
        key: alpha * parent_a.gene(key) + (1.0 - alpha) * parent_b.gene(key)
        for key in keys
    }
    return Genome(genes)


def mutate(genome: Genome, config: SimulationConfig, rng: Random) -> Genome:
    """Mutate a genome in place conceptually (returns a new genome).

    Every gene independently mutates with probability ``base_mutation_rate``
    (drawn from the genome's own ``mutation_rate`` gene). A mutated gene
    receives a continuous random variation of magnitude up to
    ``mutation_strength``; with ``point_mutation_chance`` probability the
    gene instead snaps to a brand new random value (a point mutation).
    """
    genes = dict(genome.traits)
    base_rate = genome.gene("mutation_rate", config.mutation_rate) * 2.0 + config.mutation_rate * 0.5
    rate = min(max(base_rate, 0.0), 1.0)

    for key in genes:
        if rng.random() >= rate:
            continue
        lo, hi = trait_bounds(key)
        span = hi - lo
        if rng.random() < config.point_mutation_chance:
            genes[key] = lo + rng.random() * span
        else:
            delta = (rng.random() * 2.0 - 1.0) * config.mutation_strength * span
            genes[key] = clamp_trait(key, genes[key] + delta)
    genes["mutation_rate"] = clamp_trait(
        "mutation_rate",
        min(
            max(genes.get("mutation_rate", config.mutation_rate), config.gene_mutation_rate_min),
            config.gene_mutation_rate_max,
        ),
    )
    return Genome(genes)
