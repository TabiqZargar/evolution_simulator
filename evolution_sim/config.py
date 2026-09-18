"""Central simulation configuration.

All tunable parameters live in :class:`SimulationConfig` as a single frozen
dataclass. Configuration can come from Python, a JSON file, or CLI arguments
(CLI arguments override config-file values, config-file values override the
built-in defaults).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable, TypeVar

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Main configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SimulationConfig:
    """Everything the simulation engine needs to run reproducibly."""

    # --- World / population -------------------------------------------------
    world_width: int = 100
    world_height: int = 100
    initial_population: int = 500
    target_population: int = 500
    max_population: int = 2000
    seed: int = 42

    # --- Generation model ---------------------------------------------------
    generation_length: int = 400  # ticks per generation
    replenish_to_target: bool = True  # recover the population up to target after a generation

    # --- Genetics -----------------------------------------------------------
    mutation_rate: float = 0.05  # per-gene probability of a mutating
    mutation_strength: float = 0.10  # max continuous mutation magnitude (fraction of range)
    point_mutation_chance: float = 0.02  # chance a mutated gene snaps to a brand new value
    crossover_mode: str = "uniform"  # "uniform" (pick a parent per gene) or "blend"
    crossover_blend_alpha: float = 0.5
    gene_mutation_rate_min: float = 0.0
    gene_mutation_rate_max: float = 0.5

    # --- Organism / energy --------------------------------------------------
    starting_energy: float = 80.0
    max_energy: float = 120.0
    initial_health: float = 1.0
    movement_energy_cost: float = 0.045
    ingestion_rate: float = 3.0  # food units consumed per tick while feeding
    rest_energy_threshold: float = 15.0  # below this energy an organism rests
    eating_radius: float = 1.2

    # --- Reproduction -------------------------------------------------------
    reproduction_energy_threshold: float = 55.0
    reproduction_energy_cost_fraction: float = 0.25  # of current energy, per parent
    offspring_energy_fraction: float = 0.5  # of the mid-parent residual energy
    offspring_energy_floor: float = 40.0  # newborns never start below this energy
    reproduction_cooldown: int = 40  # ticks between births
    mating_range: float = 3.0
    min_age_to_reproduce: int = 40

    # --- Aging --------------------------------------------------------------
    lifespan_min_ticks: int = 240
    lifespan_max_ticks: int = 1400  # lifespan trait [0,1] maps linearly into this span

    # --- Vision / movement --------------------------------------------------
    vision_base: float = 3.0  # world units for vision trait == 0
    vision_range: float = 10.0  # world units for vision trait == 1
    wander_angle_sigma: float = 0.9  # radians of random heading jitter while wandering
    flee_vision_factor: float = 1.4  # how far predators are noticed beyond food vision

    # --- Temperature / environment ------------------------------------------
    base_temperature: float = 0.5  # seasonal mean on [0, 1]
    seasonal_amplitude: float = 0.12
    seasonal_period: int = 2400  # ticks of a full seasonal cycle
    temperature_stress_energy: float = 0.06  # energy penalty per unit of thermal stress
    temperature_stress_health: float = 0.008  # health penalty per unit of excess stress
    terrain_movement_water_penalty: float = 1.8

    # --- Resources ----------------------------------------------------------
    resource_density: float = 0.10  # fraction of fertile cells hosting a food patch
    resource_quantity: float = 40.0
    resource_regen_rate: float = 0.10
    resource_nutrition: float = 0.8
    resource_initial_fill: float = 0.7

    # --- Environmental events ----------------------------------------------
    events_enabled: bool = True
    event_frequency: float = 0.003  # probability per tick of scheduling a new event
    event_min_duration: int = 80
    event_max_duration: int = 320

    # --- Predators (optional) ------------------------------------------------
    predators_enabled: bool = False
    predator_initial_population: int = 20
    predator_max_population: int = 120
    predator_attack_range: float = 1.4
    predator_hunt_reward: float = 45.0
    predator_hunt_cooldown: int = 18
    predator_hunt_energy_fraction: float = 0.65  # only hunt below this energy fraction
    predator_replenish_fraction: float = 0.12  # predators replenished as ~12% of prey
    predator_metabolism_multiplier: float = 1.15
    predator_initial_speed_floor: float = 0.45
    prey_attack_penalty: float = 0.35  # prey aggression is not an attack advantage

    # --- Species detection --------------------------------------------------
    species_enabled: bool = True
    species_similarity_threshold: float = 0.86  # 1 - mean gene distance groups a species

    # --- Ancestry -----------------------------------------------------------
    ancestry_depth: int = 8  # how many ancestor levels the UI walks
    pedigree_record_limit: int = 40000  # FIFO cap on remembered ancestry records

    # --- Diversity ----------------------------------------------------------
    diversity_warning_threshold: float = 0.06

    # --- Performance --------------------------------------------------------
    spatial_cell_size: float = 8.0
    dead_organism_budget: int = 4000  # keep at most this many dead bodies around

    # --- Statistics reporting ----------------------------------------------
    report_every: int = 25  # headless CLI progress every N generations

    # ------------------------------------------------------------------ extras
    def validate(self) -> None:
        """Raise ValueError when the configuration is self-inconsistent."""
        if self.world_width <= 0 or self.world_height <= 0:
            raise ValueError("world dimensions must be positive")
        if self.initial_population < 0 or self.target_population < 0:
            raise ValueError("population sizes cannot be negative")
        if self.max_population < self.target_population:
            raise ValueError("max_population must be >= target_population")
        if self.generation_length <= 0:
            raise ValueError("generation_length must be positive")
        if not (0.0 <= self.mutation_rate <= 1.0):
            raise ValueError("mutation_rate must be in [0, 1]")
        if not (0.0 <= self.mutation_strength <= 1.0):
            raise ValueError("mutation_strength must be in [0, 1]")
        if self.crossover_mode not in ("uniform", "blend"):
            raise ValueError("crossover_mode must be 'uniform' or 'blend'")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed cannot be negative")
        if not (0.0 < self.resource_density <= 1.0):
            raise ValueError("resource_density must be in (0, 1]")
        if self.species_similarity_threshold <= 0.0 or self.species_similarity_threshold > 1.0:
            raise ValueError("species_similarity_threshold must be in (0, 1]")
        if not (0.0 <= self.base_temperature <= 1.0):
            raise ValueError("base_temperature must be in [0, 1]")

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------ helpers
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SimulationConfig":
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in known:
                kwargs[key] = value
        return cls(**kwargs)

    def with_updates(self, **updates: Any) -> "SimulationConfig":
        """Return a copy of the config with the given fields overridden."""
        return self.from_dict({**self.to_dict(), **updates})

    def clone(self) -> "SimulationConfig":
        """Deep-ish copy in case a consumer mutates nested structures."""
        return self.from_dict(self.to_dict())


SIMULATION_CONFIG_FIELDS: dict[str, Any] = {
    f.name: f.type for f in fields(SimulationConfig)
}


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #
def _as_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: getattr(obj, f.name) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _as_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_plain(v) for v in obj]
    return obj


def load_config(path: str | Path) -> SimulationConfig:
    """Load a ``SimulationConfig`` from a JSON file (unknown keys are ignored)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config file {path} must contain a JSON object")
    return SimulationConfig.from_dict(raw)


def save_config(config: SimulationConfig, path: str | Path) -> None:
    """Persist a ``SimulationConfig`` to a JSON file."""
    Path(path).write_text(
        json.dumps(_as_plain(config.to_dict()), indent=2), encoding="utf-8"
    )


def apply_cli_overrides(config: SimulationConfig, overrides: Iterable[tuple[str, Any]]) -> SimulationConfig:
    """Apply ``(field, value)`` pairs, validating that each field exists and has the right type."""
    updates: dict[str, Any] = {}
    field_map = {f.name: f for f in fields(SimulationConfig)}
    for key, value in overrides:
        field = field_map.get(key)
        if field is None:
            raise ValueError(f"unknown configuration field: {key!r}")
        if value is None:
            continue
        updates[key] = _coerce(str(field.type), value)
    return config.with_updates(**updates)


def _coerce(annotation: str, value: Any) -> Any:
    """Coerce a raw CLI value to the dataclass field's annotated type."""
    ann = annotation.strip()
    if "Optional" in ann or "|" in ann:
        ann = ann.replace("Optional[", "").split("]")[0]
        ann = ann.split("|")[0].strip()
    if isinstance(value, str):
        text = value.strip()
        if ann == "bool":
            return text.lower() in ("1", "true", "yes", "on")
        if ann == "int":
            return int(float(text))
        if ann == "float":
            return float(text)
        return value
    if isinstance(value, float) and ann in ("int", "Optional[int]"):
        return int(value)
    return value
