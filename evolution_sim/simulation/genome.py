"""Genetic representation.

An organism's *genome* is a dictionary of named traits with values on
[0, 1] (``mutation_rate`` is clamped to a sub-range). Genes influence the
phenotype (speed, vision radius, size, metabolism, ...) but never hardcode
trait behaviour into the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from random import Random
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, Optional

# Lower and upper bounds, per trait.
TRAIT_SPECS: dict[str, tuple[float, float]] = {
    "speed": (0.0, 1.0),
    "vision": (0.0, 1.0),
    "size": (0.25, 1.0),  # organisms below this size are considered degenerate
    "metabolism": (0.0, 1.0),
    "efficiency": (0.0, 1.0),
    "fertility": (0.0, 1.0),
    "lifespan": (0.0, 1.0),
    "aggression": (0.0, 1.0),
    "temperature_tolerance": (0.0, 1.0),
    "mutation_rate": (0.0, 0.5),
    "attack": (0.0, 1.0),  # used by predators
}

# Predators add an attack gene on top of the shared trait set.
PREDATOR_TRAITS: tuple[str, ...] = tuple(TRAIT_SPECS.keys())

# Herbivore / prey trait keys (everything shared except "attack").
PREY_TRAITS: tuple[str, ...] = tuple(k for k in TRAIT_SPECS if k != "attack")

# Which genes participate in similarity / diversity / species computations.
DIVERSITY_TRAITS: tuple[str, ...] = (
    "speed",
    "vision",
    "size",
    "metabolism",
    "efficiency",
    "temperature_tolerance",
)


def trait_bounds(name: str) -> tuple[float, float]:
    lo, hi = TRAIT_SPECS[name]
    return lo, hi


def clamp_trait(name: str, value: float) -> float:
    lo, hi = trait_bounds(name)
    return min(max(value, lo), hi)


@dataclass(frozen=True)
class Genome:
    """Genome of one organism. Immutable: evolution never mutates a parent
    genome in place; offspring are built with :func:`evolution_sim.simulation.genetics`."""

    traits: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validated = {k: clamp_trait(k, float(v)) for k, v in self.traits.items()}
        # Dataclasses are frozen; assign via object.__setattr__. We wrap the
        # dict in a mapping proxy so the genome is *truly* immutable — nothing
        # can mutate a parent's genes in place.
        object.__setattr__(self, "traits", MappingProxyType(validated))
        if "mutation_rate" in self.traits:
            rate = min(
                max(self.traits["mutation_rate"], 0.0),
                TRAIT_SPECS["mutation_rate"][1],
            )
            object.__setattr__(self, "traits", MappingProxyType({**validated, "mutation_rate": rate}))

    def gene(self, name: str, default: float = 0.5) -> float:
        return float(self.traits.get(name, default))

    def has_attack(self) -> bool:
        return "attack" in self.traits

    def keys(self) -> Iterable[str]:
        return self.traits.keys()

    def __getitem__(self, name: str) -> float:
        return self.gene(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self.traits)

    def __len__(self) -> int:
        return len(self.traits)

    def with_gene(self, name: str, value: float) -> "Genome":
        return replace(self, traits={**dict(self.traits), name: clamp_trait(name, value)})

    @classmethod
    def random(
        cls,
        rng: "Random",
        trait_keys: Iterable[str] = PREY_TRAITS,
        *,
        center: Optional[Mapping[str, float]] = None,
        spread: float = 1.0,
    ) -> "Genome":
        """Generate a random genome within bounds. ``center`` biases values
        towards some midpoint (used to seed predators with useful traits)."""
        out: dict[str, float] = {}
        for key in trait_keys:
            lo, hi = trait_bounds(key)
            lo -= (hi - lo) / 2 * (spread - 1.0)  # widen beyond bounds, clamp below
            hi += (hi - lo) / 2 * (spread - 1.0)
            raw = lo + rng.random() * (hi - lo)
            c = center.get(key) if center else None
            if c is not None:
                raw = raw * 0.5 + c * 0.5
            out[key] = raw
        return cls(out)

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> "Genome":
        return cls(dict(data))

    def as_dict(self) -> dict[str, float]:
        return dict(self.traits)

    def validation_report(self) -> list[str]:
        problems: list[str] = []
        for key, value in self.traits.items():
            lo, hi = trait_bounds(key)
            if not (lo <= value <= hi):
                problems.append(f"{key}={value:.3f} out of bounds [{lo}, {hi}]")
        return problems

    def is_valid(self) -> bool:
        return not self.validation_report()

    def distance_to(self, other: "Genome", trait_keys: Iterable[str] = DIVERSITY_TRAITS) -> float:
        """Mean absolute gene distance over the given traits, in [0, 1]."""
        keys = tuple(trait_keys)
        if not keys:
            return 0.0
        total = 0.0
        for key in keys:
            lo, hi = trait_bounds(key)
            total += abs(self.gene(key) - other.gene(key, 0.5)) / max(hi - lo, 1e-9)
        return total / len(keys)

    def similarity_to(self, other: "Genome", trait_keys: Iterable[str] = DIVERSITY_TRAITS) -> float:
        return max(0.0, 1.0 - self.distance_to(other, trait_keys))

    def summary(self) -> str:
        parts = ", ".join(f"{k}={v:.2f}" for k, v in self.traits.items())
        return f"Genome({parts})"

    def __str__(self) -> str:
        return self.summary()
