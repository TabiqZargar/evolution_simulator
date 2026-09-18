"""Food resources: patches that regenerate over time and are consumed by
organisms. Total food is finite per tick, so competition for resources is a
real selection pressure."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable

from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.environment import Terrain


@dataclass
class FoodPatch:
    patch_id: int
    x: float
    y: float
    quantity: float
    max_quantity: float
    regen_rate: float  # per tick, before event multipliers
    nutrition: float  # energy delivered per unit of food
    fertility: float = 1.0  # cached terrain fertility at the patch location

    def consume(self, amount: float) -> float:
        """Remove up to ``amount`` food, returning how much was actually taken."""
        taken = min(amount, self.quantity)
        self.quantity -= taken
        return taken

    def regrow(self, amount: float) -> None:
        self.quantity = min(self.max_quantity, self.quantity + amount)

    def is_depleted(self) -> bool:
        return self.quantity <= 0.0


def spawn_food_patches(terrain: Terrain, rng: Random, config: SimulationConfig) -> list[FoodPatch]:
    """Place foods stochastically over fertile terrain (deterministic given rng).

    Each cell is a candidate food location; fertile cells are more likely to
    yield a patch, and water never does. Quantity and regen vary per patch.
    """
    patches: list[FoodPatch] = []
    pid = 1
    for cy in range(terrain.height):
        for cx in range(terrain.width):
            fertility = terrain.fertility_at(cx, cy)
            if fertility <= 0.05:
                continue
            if rng.random() > config.resource_density * (0.5 + 0.5 * fertility):
                continue
            qty = config.resource_quantity * (0.6 + 0.8 * rng.random())
            patches.append(
                FoodPatch(
                    patch_id=pid,
                    x=cx + 0.5,
                    y=cy + 0.5,
                    quantity=qty * config.resource_initial_fill,
                    max_quantity=qty,
                    regen_rate=config.resource_regen_rate,
                    nutrition=config.resource_nutrition,
                    fertility=fertility,
                )
            )
            pid += 1
    return patches


def regrow_patches(patches: Iterable[FoodPatch], terrain: Terrain, multiplier: float) -> None:
    for patch in patches:
        if patch.quantity >= patch.max_quantity:
            continue
        patch.regrow(patch.regen_rate * patch.fertility * multiplier)


def total_food(patches: Iterable[FoodPatch]) -> float:
    return sum(p.quantity for p in patches)
