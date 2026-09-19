"""Procedural terrain, climate and per-tile fertility.

Terrain is generated with a tiny deterministic value-noise field (integer
lattice + bilinear interpolation + a couple of octaves). The same seed always
produces the same world. Climate is a slow seasonal temperature oscillation
which environmental events (see :mod:`events`) can push around.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import Optional

from evolution_sim.simulation.events import EventManager

BARREN_FERTILITY = 0.10
GRASS_FERTILITY = 0.75
FOREST_FERTILITY = 1.0
WATER_FERTILITY = 0.0


class Biome(Enum):
    WATER = "water"
    BARREN = "barren"
    GRASS = "grass"
    FOREST = "forest"


@dataclass(frozen=True)
class TerrainInfo:
    biome: Biome
    fertility: float  # 0..1 multiplier applied to food regeneration
    movement_penalty: float  # >1 increases movement energy cost


BIOME_INFO: dict[Biome, TerrainInfo] = {
    Biome.WATER: TerrainInfo(Biome.WATER, WATER_FERTILITY, 1.0),
    Biome.BARREN: TerrainInfo(Biome.BARREN, BARREN_FERTILITY, 1.0),
    Biome.GRASS: TerrainInfo(Biome.GRASS, GRASS_FERTILITY, 1.0),
    Biome.FOREST: TerrainInfo(Biome.FOREST, FOREST_FERTILITY, 1.0),
}
BIOME_INFO_FINAL: dict[Biome, TerrainInfo] = BIOME_INFO


class ValueNoise:
    """Deterministic multi-octave value noise on a small integer lattice."""

    def __init__(self, seed: int, lattice: int = 12) -> None:
        self.lattice = lattice
        self.cells: dict[tuple[int, int], float] = {}
        self._rng = Random(seed)

    def _cell(self, cx: int, cy: int) -> float:
        key = (cx, cy)
        value = self.cells.get(key)
        if value is None:
            value = self._rng.random()
            self.cells[key] = value
        return value

    @staticmethod
    def _smooth(t: float) -> float:
        return t * t * (3.0 - 2.0 * t)

    def sample(self, x: float, y: float) -> float:
        lx = x / self.lattice
        ly = y / self.lattice
        x0 = math.floor(lx)
        y0 = math.floor(ly)
        tx = ValueNoise._smooth(lx - x0)
        ty = ValueNoise._smooth(ly - y0)
        v00 = self._cell(x0, y0)
        v10 = self._cell(x0 + 1, y0)
        v01 = self._cell(x0, y0 + 1)
        v11 = self._cell(x0 + 1, y0 + 1)
        top = v00 + (v10 - v00) * tx
        bottom = v01 + (v11 - v01) * tx
        return top + (bottom - top) * ty

    def fractal(self, x: float, y: float, octaves: int = 3, persistence: float = 0.5) -> float:
        total = 0.0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        for _ in range(octaves):
            total += self.sample(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2.0
        return total / max_value


class Terrain:
    """2D biome grid + fertility map, generated deterministically from a seed."""

    def __init__(
        self,
        width: int,
        height: int,
        seed: int,
        water_penalty: float = 1.0,
    ) -> None:
        self.width = width
        self.height = height
        self.water_penalty = water_penalty
        self.noise_elev = ValueNoise(seed * 2 + 1, lattice=10)
        self.noise_moist = ValueNoise(seed * 2 + 2, lattice=7)
        self.biomes: list[list[Biome]] = []
        self.fertility: list[list[float]] = []
        self._generate()

    def _classify(self, elevation: float, moisture: float) -> Biome:
        if elevation < 0.32:
            return Biome.WATER
        if elevation > 0.78:
            return Biome.BARREN
        if elevation > 0.68 and moisture < 0.45:
            return Biome.BARREN
        if moisture > 0.72:
            return Biome.FOREST
        return Biome.GRASS

    def _generate(self) -> None:
        for cy in range(self.height):
            row_biomes: list[Biome] = []
            row_fert: list[float] = []
            for cx in range(self.width):
                fx = cx / max(self.width, 1)
                fy = cy / max(self.height, 1)
                elevation = self.noise_elev.fractal(cx, cy) * 0.9 + 0.1 * (0.5 + 0.5 * fx * fy)
                moisture = self.noise_moist.fractal(cx, cy)
                biome = self._classify(elevation, moisture)
                row_biomes.append(biome)
                info = BIOME_INFO_FINAL[biome]
                variance = 0.85 + 0.3 * self.noise_moist.sample(cx * 3.0, cy * 3.0)
                row_fert.append(info.fertility * variance / 1.15)
            self.biomes.append(row_biomes)
            self.fertility.append(row_fert)

    def biome_at(self, x: int, y: int) -> Biome:
        x = min(max(x, 0), self.width - 1)
        y = min(max(y, 0), self.height - 1)
        return self.biomes[y][x]

    def fertility_at(self, x: float, y: float) -> float:
        cx = min(max(int(x), 0), self.width - 1)
        cy = min(max(int(y), 0), self.height - 1)
        return self.fertility[cy][cx]

    def movement_penalty_at(self, x: float, y: float) -> float:
        biome = self.biome_at(int(x), int(y))
        if biome == Biome.WATER:
            return self.water_penalty
        return BIOME_INFO_FINAL[biome].movement_penalty

    def water_at(self, x: float, y: float) -> bool:
        return self.biome_at(int(x), int(y)) == Biome.WATER


@dataclass
class Climate:
    """Slowly oscillating temperature plus whatever events are currently active."""

    tick: int = 0
    season_offset: float = 0.0
    base_temperature: float = 0.5
    seasonal_amplitude: float = 0.12
    seasonal_period: int = 2400
    events_enabled: bool = True
    event_manager: Optional[EventManager] = None

    @property
    def temperature(self) -> float:
        """Combine the seasonal cycle with the strongest active temperature event."""
        season = math.sin(2.0 * math.pi * (self.tick + self.season_offset) / max(self.seasonal_period, 1))
        temp = self.base_temperature + self.seasonal_amplitude * season
        if self.event_manager is not None:
            temp += self.event_manager.temperature_delta()
        return min(max(temp, 0.0), 1.0)

    def temperature_label(self) -> str:
        t = self.temperature
        if t <= 0.2:
            return "frozen"
        if t >= 0.8:
            return "scorching"
        if t <= 0.38:
            return "cold"
        if t >= 0.62:
            return "warm"
        return "mild"

    def update(self, dt: int = 1) -> None:
        self.tick += dt


