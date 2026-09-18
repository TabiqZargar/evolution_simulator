"""The world: terrain, food patches and a grid-based spatial index.

The spatial index lets behaviour queries inspect *nearby* neighbours instead
of scanning the whole population, keeping the hot loop O(n + k) rather than
O(n^2).
"""

from __future__ import annotations

from random import Random
from typing import Callable, Generic, Iterable, Iterator, Optional, Protocol, TypeVar

from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.environment import Terrain
from evolution_sim.simulation.genome import Genome
from evolution_sim.simulation.organism import Organism
from evolution_sim.simulation.resources import FoodPatch, regrow_patches, spawn_food_patches


class Positioned(Protocol):
    x: float
    y: float


T = TypeVar("T", bound=Positioned)

PADDING = 0.5  # keep organisms spawned near edges on the map


class SpatialGrid(Generic[T]):
    """Uniform grid of cells; each cell holds a list of positioned items."""

    def __init__(self, cell_size: float) -> None:
        self.cell_size = cell_size
        self.cells: dict[int, list[T]] = {}

    def _key(self, cx: int, cy: int) -> int:
        return (cx << 16) | (cy & 0xFFFF)

    def _cell_of(self, x: float, y: float) -> int:
        return (int(x // self.cell_size) << 16) | (int(y // self.cell_size) & 0xFFFF)

    def clear(self) -> None:
        self.cells.clear()

    def add(self, item: T) -> None:
        key = self._cell_of(item.x, item.y)
        bucket = self.cells.get(key)
        if bucket is None:
            self.cells[key] = [item]
        else:
            bucket.append(item)

    def rebuild(self, items: Iterable[T]) -> None:
        self.clear()
        cell_size = self.cell_size
        cells = self.cells
        for item in items:
            key = (int(item.x // cell_size) << 16) | (int(item.y // cell_size) & 0xFFFF)
            bucket = cells.get(key)
            if bucket is None:
                cells[key] = [item]
            else:
                bucket.append(item)

    def query(self, x: float, y: float, radius: float) -> list[T]:
        """Return all items within ``radius`` of (x, y)."""
        cells = self.cells
        cell_size = self.cell_size
        radius2 = radius * radius
        cx0 = int((x - radius) // cell_size)
        cx1 = int((x + radius) // cell_size)
        cy0 = int((y - radius) // cell_size)
        cy1 = int((y + radius) // cell_size)
        out: list[T] = []
        for cy in range(cy0, cy1 + 1):
            cysh = cy & 0xFFFF
            for cx in range(cx0, cx1 + 1):
                bucket = cells.get((cx << 16) | cysh)
                if bucket is None:
                    continue
                for item in bucket:
                    ddx = item.x - x
                    ddy = item.y - y
                    if ddx * ddx + ddy * ddy <= radius2:
                        out.append(item)
        return out

    def max_in(
        self, x: float, y: float, radius: float, key: Callable[[T], float]
    ) -> T | None:
        """Return the item maximising ``key`` within radius, without a list."""
        cells = self.cells
        cell_size = self.cell_size
        radius2 = radius * radius
        cx0 = int((x - radius) // cell_size)
        cx1 = int((x + radius) // cell_size)
        cy0 = int((y - radius) // cell_size)
        cy1 = int((y + radius) // cell_size)
        best: T | None = None
        best_score: float = float("-inf")
        for cy in range(cy0, cy1 + 1):
            cysh = cy & 0xFFFF
            for cx in range(cx0, cx1 + 1):
                bucket = cells.get((cx << 16) | cysh)
                if not bucket:
                    continue
                for item in bucket:
                    ddx = item.x - x
                    ddy = item.y - y
                    if ddx * ddx + ddy * ddy <= radius2:
                        score = key(item)
                        if score > best_score:
                            best_score = score
                            best = item
        return best

    def nearest(
        self,
        x: float,
        y: float,
        radius: float,
        predicate: Optional[Callable[[T], bool]] = None,
    ) -> T | None:
        """Return the nearest item matching ``radius``/``predicate`` without a list."""
        cells = self.cells
        cell_size = self.cell_size
        radius2 = radius * radius
        cx0 = int((x - radius) // cell_size)
        cx1 = int((x + radius) // cell_size)
        cy0 = int((y - radius) // cell_size)
        cy1 = int((y + radius) // cell_size)
        best: T | None = None
        best_dist2 = float("inf")
        for cy in range(cy0, cy1 + 1):
            cysh = cy & 0xFFFF
            for cx in range(cx0, cx1 + 1):
                bucket = cells.get((cx << 16) | cysh)
                if not bucket:
                    continue
                for item in bucket:
                    if predicate is not None and not predicate(item):
                        continue
                    ddx = item.x - x
                    ddy = item.y - y
                    d2 = ddx * ddx + ddy * ddy
                    if d2 <= radius2 and d2 < best_dist2:
                        best_dist2 = d2
                        best = item
        return best

    def items(self) -> Iterator[T]:
        for bucket in self.cells.values():
            yield from bucket


class World:
    """Holds terrain, food, organisms and the spatial indices."""

    def __init__(self, config: SimulationConfig, rng: Random) -> None:
        self.config = config
        self.rng = rng
        self.width = config.world_width
        self.height = config.world_height
        self.terrain = Terrain(self.width, self.height, config.seed)
        self.patches: list[FoodPatch] = []
        self.organisms: list[Organism] = []
        self.org_grid: SpatialGrid[Organism] = SpatialGrid(config.spatial_cell_size)
        self.food_grid: SpatialGrid[FoodPatch] = SpatialGrid(max(config.spatial_cell_size, 2.0))
        self.patch_by_id: dict[int, FoodPatch] = {}
        self._next_patch_id = 1
        self._next_organism_id = 1

    # -- setup ----------------------------------------------------------------
    def generate_food(self) -> None:
        self.patches = spawn_food_patches(self.terrain, self.rng, self.config)
        self.patch_by_id = {p.patch_id: p for p in self.patches}
        self._next_patch_id = (max(p.patch_id for p in self.patches) + 1) if self.patches else 1
        # Patches never move; only their quantity changes, so the grid is
        # indexed once here and never rebuilt.
        self.food_grid.rebuild(self.patches)

    def spawn_organism(
        self,
        genome: Genome,
        generation: int,
        energy: float,
        parent_a: int | None = None,
        parent_b: int | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> Organism:
        if x is None:
            x = PADDING + self.rng.random() * (self.width - 2 * PADDING)
        if y is None:
            y = PADDING + self.rng.random() * (self.height - 2 * PADDING)
        org = Organism(
            organism_id=self._next_organism_id,
            genome=genome,
            x=min(max(x, 0.0), self.width),
            y=min(max(y, 0.0), self.height),
            generation=generation,
            energy=energy,
            parent_a=parent_a,
            parent_b=parent_b,
        )
        org.last_x = org.x
        org.last_y = org.y
        self._next_organism_id += 1
        self.organisms.append(org)
        return org

    def rebuild_indices(self) -> None:
        self.org_grid.rebuild(self.organisms)

    def prune_dead(self, max_dead: int) -> None:
        """Drop the oldest dead bodies so long runs stay fast.

        Always preserves every living organism; when more than ``max_dead``
        corpses have accumulated, only the most recent ``max_dead`` are kept
        (the organism list is append-ordered, so "most recent" is the tail).
        Deterministic — no RNG involved — so state signatures and
        save/load determinism are unaffected.
        """
        if max_dead < 0:
            return
        alive: list[Organism] = []
        dead: list[Organism] = []
        for org in self.organisms:
            if org.alive:
                alive.append(org)
            else:
                dead.append(org)
        if len(dead) <= max_dead:
            if len(alive) != len(self.organisms):
                self.organisms = alive + dead
                self.rebuild_indices()
            return
        self.organisms = alive + dead[-max_dead:]
        self.rebuild_indices()

    def food_near(self, x: float, y: float, radius: float) -> list[FoodPatch]:
        return self.food_grid.query(x, y, radius)

    def regrow_food(self, multiplier: float) -> None:
        regrow_patches(self.patches, self.terrain, multiplier)

    def total_food(self) -> float:
        return sum(p.quantity for p in self.patches)

    def nearest_organism(
        self,
        x: float,
        y: float,
        radius: float,
        predicate: Optional[Callable[[Organism], bool]] = None,
    ) -> Organism | None:
        return self.org_grid.nearest(x, y, radius, predicate)
