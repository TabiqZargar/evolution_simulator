"""The organism and its life ledger."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from evolution_sim.simulation.genome import Genome


@dataclass
class Organism:
    """A living (or recently dead) organism in the simulation.

    This class is *data* plus phenotype accessors. All behaviour lives in the
    engine so the model stays easy to extend.

    Gene-derived phenotype values are cached on first access: a genome is
    immutable for the life of an organism, so traits never need recomputing
    on every engine query (crucial for the hot loop).
    """

    organism_id: int
    genome: Genome
    x: float
    y: float
    generation: int = 1
    energy: float = 60.0
    health: float = 1.0
    age: int = 0
    alive: bool = True
    reproduction_cooldown: int = 0
    hunt_cooldown: int = 0  # ticks the predator must wait between hunts
    parent_a: Optional[int] = None
    parent_b: Optional[int] = None
    children: int = 0
    # energy ledger (used by the fitness model)
    energy_harvested: float = 0.0
    energy_spent: float = 0.0
    fights_won: int = 0
    fights_lost: int = 0
    # optional species label assigned by the analytics layer
    species_id: int = 0
    death_cause: Optional[str] = None
    # set every tick so the renderer can interpolate motion
    last_x: float = 0.0
    last_y: float = 0.0
    _final_fitness: Optional[float] = field(default=None, repr=False)
    _cache: dict[str, float] = field(default_factory=dict, repr=False, init=False)

    def __post_init__(self) -> None:
        # Fast-path caches for the engine hot loop.
        traits = self.genome.traits
        self._cache["_predator"] = 1.0 if "attack" in traits else 0.0
        for key, value in traits.items():
            self._cache.setdefault(key, float(value))
        self.last_x = self.x
        self.last_y = self.y

    def trait(self, name: str, default: float = 0.5) -> float:
        value = self._cache.get(name)
        if value is not None:
            return value
        resolved = self.genome.gene(name, default)
        self._cache[name] = resolved
        return resolved

    # -- phenotype helpers ----------------------------------------------------
    @property
    def is_predator(self) -> bool:
        return self._cache["_predator"] != 0.0

    @property
    def speed(self) -> float:
        return self.trait("speed")

    @property
    def vision(self) -> float:
        return self.trait("vision")

    @property
    def size(self) -> float:
        return self.trait("size")

    @property
    def metabolism(self) -> float:
        return self.trait("metabolism")

    @property
    def efficiency(self) -> float:
        return self.trait("efficiency")

    @property
    def fertility(self) -> float:
        return self.trait("fertility")

    @property
    def lifespan_trait(self) -> float:
        return self.trait("lifespan")

    @property
    def aggression(self) -> float:
        return self.trait("aggression")

    @property
    def temperature_tolerance(self) -> float:
        return self.trait("temperature_tolerance")

    @property
    def attack(self) -> float:
        return self.trait("attack")

    # -- state -----------------------------------------------------------------
    def distance_to(self, other: "Organism") -> float:
        return float(((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5)

    @property
    def starving(self) -> bool:
        return self.energy <= 0.0

    def record_death(self, cause: str, final_fitness: Optional[float] = None) -> None:
        self.alive = False
        self.death_cause = cause
        if final_fitness is not None:
            self._final_fitness = final_fitness

    @property
    def final_fitness(self) -> Optional[float]:
        return self._final_fitness
