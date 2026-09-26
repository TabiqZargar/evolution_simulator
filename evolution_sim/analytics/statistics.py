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

# Canonical death causes recorded in :class:`GenerationStats`.
DEATH_STARVATION = "starvation"
DEATH_STRESS = "stress"
DEATH_OLD_AGE = "old age"
DEATH_HUNTED = "hunted"
DEATH_CAUSES: tuple[str, ...] = (
    DEATH_STARVATION,
    DEATH_STRESS,
    DEATH_OLD_AGE,
    DEATH_HUNTED,
)


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
class GenerationLedger:
    """Lifecycle tallies accumulated between generation boundaries.

    The engine fills this in during the ticks of one generation and passes it
    to :func:`snapshot_stats`; it is reset when the generation advances.
    """

    mutations: int = 0
    reproductions: int = 0
    resources_consumed: float = 0.0
    ages_at_death_sum: int = 0
    deaths_at_death_count: int = 0
    deaths_by_cause: dict[str, int] = field(default_factory=dict)

    def record_death(self, age: int, cause: str) -> None:
        self.ages_at_death_sum += age
        self.deaths_at_death_count += 1
        self.deaths_by_cause[cause] = self.deaths_by_cause.get(cause, 0) + 1

    @property
    def avg_lifespan(self) -> float:
        if self.deaths_at_death_count == 0:
            return 0.0
        return self.ages_at_death_sum / self.deaths_at_death_count


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
    mutations: int = 0
    reproductions: int = 0
    resources_consumed: float = 0.0
    avg_age: float = 0.0
    avg_energy: float = 0.0
    environmental_deaths: int = 0
    deaths_by_cause: dict[str, int] = field(default_factory=dict)
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
    ledger: Optional[GenerationLedger] = None,
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
    for name in ("speed", "vision", "size", "metabolism", "efficiency", "aggression"):
        setattr(stats, f"avg_{name}", trait_mean(alive, name))
    stats.diversity = genetic_diversity(alive)
    if exact_fitness:
        fits = [_safe_fitness(o, config) for o in alive]
        stats.avg_fitness = sum(fits) / len(fits) if fits else 0.0
        stats.max_fitness = max(fits) if fits else 0.0
    if species_list is not None:
        stats.species_count = len(species_list)
    if alive:
        stats.avg_age = sum(o.age for o in alive) / len(alive)
        stats.avg_energy = sum(o.energy for o in alive) / len(alive)
    if ledger is not None:
        stats.mutations = ledger.mutations
        stats.reproductions = ledger.reproductions
        stats.resources_consumed = ledger.resources_consumed
        stats.avg_lifespan = ledger.avg_lifespan
        stats.deaths_by_cause = dict(ledger.deaths_by_cause)
        stats.environmental_deaths = stats.deaths_by_cause.get(DEATH_STRESS, 0)
    return stats


def _safe_fitness(organism: Organism, config: SimulationConfig) -> float:
    from evolution_sim.simulation.fitness import compute_fitness

    return compute_fitness(organism, config)


# --------------------------------------------------------------------------- #
# Run-level summary
# --------------------------------------------------------------------------- #
@dataclass
class RunSummary:
    """Aggregate metrics over the whole history of a run."""

    generations: int
    last_generation: int
    final_population: int
    total_births: int = 0
    total_deaths: int = 0
    total_mutations: int = 0
    total_reproductions: int = 0
    total_resources_consumed: float = 0.0
    environmental_deaths: int = 0
    max_population: int = 0
    min_population: int = 0
    avg_population: float = 0.0
    avg_lifespan: float = 0.0
    extinct: bool = False


def summarize(entries: Iterable[GenerationStats]) -> RunSummary:
    """Aggregate generation records into a single run-level summary.

    ``avg_lifespan`` is weighted by the number of deaths of each generation,
    so it reflects the population-wide mean achieved lifespan.
    """
    stats_list = list(entries)
    if not stats_list:
        return RunSummary(generations=0, last_generation=0, final_population=0)
    populations = [e.population for e in stats_list]
    total_births = sum(e.births for e in stats_list)
    total_deaths = sum(e.deaths for e in stats_list)
    total_lifespan = sum(e.avg_lifespan * e.deaths for e in stats_list)
    return RunSummary(
        generations=len(stats_list),
        last_generation=max(e.generation for e in stats_list),
        final_population=stats_list[-1].population,
        total_births=total_births,
        total_deaths=total_deaths,
        total_mutations=sum(e.mutations for e in stats_list),
        total_reproductions=sum(e.reproductions for e in stats_list),
        total_resources_consumed=sum(e.resources_consumed for e in stats_list),
        environmental_deaths=sum(e.environmental_deaths for e in stats_list),
        max_population=max(populations),
        min_population=min(populations),
        avg_population=sum(populations) / len(populations),
        avg_lifespan=total_lifespan / total_deaths if total_deaths else 0.0,
        extinct=stats_list[-1].extinct,
    )
