"""Population statistics, genetic diversity and species detection.

The statistics layer is *pure*: it consumes a list of organisms (and a few
scalars) and returns numbers/records. It never mutates the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.genome import DIVERSITY_TRAITS, trait_bounds
from evolution_sim.simulation.organism import Organism


# --------------------------------------------------------------------------- #
# Trait statistics
# --------------------------------------------------------------------------- #
def trait_mean(organisms: Iterable[Organism], name: str) -> float:
    values = [o.genome.gene(name, 0.5) for o in organisms]
    if not values:
        return 0.0
    return sum(values) / len(values)


def trait_std(organisms: Iterable[Organism], name: str) -> float:
    values = [o.genome.gene(name, 0.5) for o in organisms]
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return float(variance ** 0.5)


def genetic_diversity(organisms: Iterable[Organism]) -> float:
    """Mean per-gene normalised spread.

    For each gene we compute the normalised mean absolute deviation from the
    population mean (0 when everyone is identical, ~0.25 for a uniform
    distribution). Diversity is the average over genes, 0 → no diversity,
    1 → maximum possible.
    """
    orgs = list(organisms)
    if len(orgs) < 2:
        return 0.0
    total = 0.0
    for name in DIVERSITY_TRAITS:
        lo, hi = trait_bounds(name)
        span = max(hi - lo, 1e-9)
        values = [o.genome.gene(name, 0.5) for o in orgs]
        mean = sum(values) / len(values)
        mad = sum(abs(v - mean) for v in values) / len(values)
        total += min(mad / (span / 2.0), 1.0)
    return total / len(DIVERSITY_TRAITS)


def trait_distribution(organisms: Iterable[Organism]) -> dict[str, float]:
    """One histogram (10 buckets, 0..1) per diversity gene, as a flat dict."""
    orgs = list(organisms)
    out: dict[str, float] = {}
    for name in DIVERSITY_TRAITS:
        values = [o.genome.gene(name, 0.5) for o in orgs]
        if not values:
            out[name] = 0.0
            continue
        buckets = 10
        counts = [0] * buckets
        for v in values:
            idx = min(int(v * buckets), buckets - 1)
            counts[idx] += 1
        for i, c in enumerate(counts):
            out[f"{name}|{i}"] = c / len(orgs)
    return out


# --------------------------------------------------------------------------- #
# Species detection
# --------------------------------------------------------------------------- #
@dataclass
class Species:
    id: int
    centroid: tuple[float, ...]
    labels: tuple[str, ...]
    members: list[Organism] = field(default_factory=list)

    @property
    def population(self) -> int:
        return len(self.members)

    def average_trait(self, name: str) -> float:
        if not self.members:
            return 0.0
        return sum(o.genome.gene(name, 0.5) for o in self.members) / len(self.members)

    def center_position(self) -> tuple[float, float] | None:
        if not self.members:
            return None
        return (
            sum(o.x for o in self.members) / len(self.members),
            sum(o.y for o in self.members) / len(self.members),
        )


def _vector(organism: Organism, trait_keys: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(organism.trait(k) for k in trait_keys)


def detect_species(
    organisms: Iterable[Organism],
    config: SimulationConfig,
    trait_keys: Iterable[str] = DIVERSITY_TRAITS,
) -> list[Species]:
    """Deterministic greedy clustering by genetic similarity.

    Organisms are walked in id order and each joins the nearest existing
    species whose centroid is within ``1 - similarity_threshold`` distance,
    or else founds a new species. The trait vectors are cached per organism
    once and all arithmetic is plain floats, keeping this affordable even at
    the end of every generation.
    """
    keys = tuple(trait_keys)
    threshold = 1.0 - config.species_similarity_threshold
    species_list: list[Species] = []
    centroids: list[list[float]] = []
    for org in sorted(organisms, key=lambda o: o.organism_id):
        vec = _vector(org, keys)
        best_species: Optional[Species] = None
        best_dist = 1.0
        for idx, centroid in enumerate(centroids):
            d = _vec_dist(vec, centroid)
            if d < best_dist:
                best_dist = d
                best_species = species_list[idx]

        if best_species is not None and best_dist <= threshold:
            sp = best_species
            sp.members.append(org)
            n = len(sp.members)
            centroid = centroids[sp.id - 1]
            for i in range(len(centroid)):
                centroid[i] = (centroid[i] * (n - 1) + vec[i]) / n
            org.species_id = sp.id
        else:
            new_id = len(species_list) + 1
            species_list.append(Species(id=new_id, centroid=vec, labels=keys, members=[org]))
            centroids.append(list(vec))
            org.species_id = new_id
    return species_list


def _vec_dist(a: tuple[float, ...], b: list[float]) -> float:
    total = 0.0
    for i in range(len(a)):
        total += abs(a[i] - b[i])
    return total / len(a)


# --------------------------------------------------------------------------- #
# Per-generation statistics
# --------------------------------------------------------------------------- #
@dataclass
class GenerationStats:
    generation: int
    tick: int
    population: int
    births: int = 0
    deaths: int = 0
    avg_fitness: float = 0.0
    max_fitness: float = 0.0
    avg_speed: float = 0.0
    avg_vision: float = 0.0
    avg_size: float = 0.0
    avg_metabolism: float = 0.0
    avg_lifespan: float = 0.0
    avg_efficiency: float = 0.0
    avg_aggression: float = 0.0
    diversity: float = 0.0
    species_count: int = 0
    resource_total: float = 0.0
    temperature: float = 0.5
    event_label: Optional[str] = None
    extinct: bool = False
    predator_population: int = 0
    prey_population: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerationStats":
        return cls(**{f: data[f] for f in cls.__dataclass_fields__ if f in data})


def snapshot_stats(
    config: SimulationConfig,
    generation: int,
    tick: int,
    organisms: list[Organism],
    births: int,
    deaths: int,
    resource_total: float,
    temperature: float,
    event_label: Optional[str],
    species_list: Optional[list[Species]] = None,
    exact_fitness: bool = True,
) -> GenerationStats:
    alive = [o for o in organisms if o.alive]
    stats = GenerationStats(
        generation=generation,
        tick=tick,
        population=len(alive),
        births=births,
        deaths=deaths,
        resource_total=resource_total,
        temperature=temperature,
        event_label=event_label,
        predator_population=sum(1 for o in alive if o.is_predator),
        prey_population=sum(1 for o in alive if not o.is_predator),
        extinct=not alive,
    )
    for name in ("speed", "vision", "size", "metabolism", "lifespan", "efficiency", "aggression"):
        setattr(stats, f"avg_{name}", trait_mean(alive, name))
    stats.diversity = genetic_diversity(alive)
    if exact_fitness:
        fits = [_safe_fitness(o, config) for o in alive]
        stats.avg_fitness = sum(fits) / len(fits) if fits else 0.0
        stats.max_fitness = max(fits) if fits else 0.0
    if species_list is not None:
        stats.species_count = len(species_list)
    return stats


def _safe_fitness(organism: Organism, config: SimulationConfig) -> float:
    from evolution_sim.simulation.fitness import compute_fitness

    return compute_fitness(organism, config)
