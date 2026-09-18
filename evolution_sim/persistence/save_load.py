"""Save / load a simulation state.

The whole engine state is captured in a single JSON file:

* configuration
* the RNG state (so a resumed run stays byte-for-byte deterministic)
* the world (terrain seed, food patches, every organism, next ids)
* the event manager, history and pedigree log

The format is versioned; unknown keys in the file are ignored on load so
older saved files can stay readable.
"""

from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Any

from evolution_sim.analytics.history import History
from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.engine import Engine, EngineState
from evolution_sim.simulation.events import EventManager
from evolution_sim.simulation.genome import Genome
from evolution_sim.simulation.organism import Organism
from evolution_sim.simulation.resources import FoodPatch
from evolution_sim.simulation.world import World

FORMAT_VERSION = 1


# --------------------------------------------------------------------------- #
# RNG state (random.Random.getstate() is a tuple of mixed scalars)
# --------------------------------------------------------------------------- #
def encode_rng(rng: random.Random) -> dict[str, Any]:
    version, state, gauss_next = rng.getstate()
    return {
        "version": version,
        "state": list(state),
        "gauss_next": gauss_next,
    }


def decode_rng(data: dict[str, Any]) -> random.Random:
    rng = random.Random(0)
    rng.setstate((data["version"], tuple(data["state"]), data["gauss_next"]))
    return rng


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def _patch_to_dict(patch: FoodPatch) -> dict[str, Any]:
    return {
        "patch_id": patch.patch_id,
        "x": patch.x,
        "y": patch.y,
        "quantity": patch.quantity,
        "max_quantity": patch.max_quantity,
        "regen_rate": patch.regen_rate,
        "nutrition": patch.nutrition,
        "fertility": patch.fertility,
    }


def _patch_from_dict(data: dict[str, Any]) -> FoodPatch:
    return FoodPatch(
        patch_id=int(data["patch_id"]),
        x=float(data["x"]),
        y=float(data["y"]),
        quantity=float(data["quantity"]),
        max_quantity=float(data["max_quantity"]),
        regen_rate=float(data.get("regen_rate", 0.1)),
        nutrition=float(data.get("nutrition", 0.8)),
        fertility=float(data.get("fertility", 1.0)),
    )


def _organism_to_dict(org: Organism) -> dict[str, Any]:
    return {
        "organism_id": org.organism_id,
        "genome": dict(org.genome.traits),
        "x": org.x,
        "y": org.y,
        "generation": org.generation,
        "energy": org.energy,
        "health": org.health,
        "age": org.age,
        "alive": org.alive,
        "reproduction_cooldown": org.reproduction_cooldown,
        "hunt_cooldown": org.hunt_cooldown,
        "parent_a": org.parent_a,
        "parent_b": org.parent_b,
        "children": org.children,
        "energy_harvested": org.energy_harvested,
        "energy_spent": org.energy_spent,
        "fights_won": org.fights_won,
        "fights_lost": org.fights_lost,
        "species_id": org.species_id,
        "death_cause": org.death_cause,
        "last_x": org.last_x,
        "last_y": org.last_y,
        "final_fitness": org.final_fitness,
    }


def _organism_from_dict(data: dict[str, Any]) -> Organism:
    org = Organism(
        organism_id=int(data["organism_id"]),
        genome=Genome(data["genome"]),
        x=float(data["x"]),
        y=float(data["y"]),
        generation=int(data.get("generation", 1)),
        energy=float(data.get("energy", 0.0)),
        health=float(data.get("health", 1.0)),
        age=int(data.get("age", 0)),
        alive=bool(data.get("alive", True)),
        reproduction_cooldown=int(data.get("reproduction_cooldown", 0)),
        hunt_cooldown=int(data.get("hunt_cooldown", 0)),
        parent_a=data.get("parent_a"),
        parent_b=data.get("parent_b"),
        children=int(data.get("children", 0)),
        energy_harvested=float(data.get("energy_harvested", 0.0)),
        energy_spent=float(data.get("energy_spent", 0.0)),
        fights_won=int(data.get("fights_won", 0)),
        fights_lost=int(data.get("fights_lost", 0)),
        species_id=int(data.get("species_id", 0)),
        death_cause=data.get("death_cause"),
    )
    org.alive = bool(data.get("alive", True))
    org.last_x = float(data.get("last_x", org.x))
    org.last_y = float(data.get("last_y", org.y))
    org._final_fitness = data.get("final_fitness")
    return org


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
def save_state(engine: Engine, path: str | Path) -> None:
    """Write ``engine``'s full state to ``path`` as JSON."""
    state = engine.snapshot()
    payload = _state_to_dict(state)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _state_to_dict(state: EngineState) -> dict[str, Any]:
    world = state.world
    return {
        "format_version": FORMAT_VERSION,
        "config": state.config.to_dict(),
        "rng_state": encode_rng(state.rng),
        "generation": state.generation,
        "tick": state.tick,
        "founder_count": state.founder_count,
        "next_organism_id": world._next_organism_id,
        "next_patch_id": world._next_patch_id,
        "event_manager": state.event_manager.to_dict(),
        "patches": [_patch_to_dict(p) for p in world.patches],
        "organisms": [_organism_to_dict(o) for o in world.organisms],
        "history": state.history.to_dict(),
        "pedigree": list(state.pedigree),
    }


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load_state(path: str | Path) -> Engine:
    """Rebuild an :class:`Engine` from a save file.

    Terrain is regenerated deterministically from the saved config's seed;
    food patches and organisms are restored verbatim, so evolution continues
    exactly where it stopped.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"save file {path} must contain a JSON object")

    config = SimulationConfig.from_dict(raw.get("config", {}))
    rng = decode_rng(raw["rng_state"])
    world = _build_world(config, rng, raw)

    event_manager = EventManager.from_dict(raw.get("event_manager", {}), rng)
    history = History.from_list(raw.get("history", []))
    pedigree = deque(raw.get("pedigree", []))

    engine = Engine(
        config=config,
        rng=rng,
        world=world,
        generation=int(raw.get("generation", 1)),
        tick=int(raw.get("tick", 0)),
        event_manager=event_manager,
        history=history,
        pedigree=pedigree,
        founder_count=int(raw.get("founder_count", 0)),
    )
    return engine


def _build_world(config: SimulationConfig, rng: random.Random, raw: dict[str, Any]) -> World:
    world = World(config, rng)
    world.patches = [_patch_from_dict(p) for p in raw.get("patches", [])]
    world.patch_by_id = {p.patch_id: p for p in world.patches}
    world._next_patch_id = int(raw.get("next_patch_id", 1))
    for o in raw.get("organisms", []):
        world.organisms.append(_organism_from_dict(o))
    world._next_organism_id = int(raw.get("next_organism_id", 1))
    world.food_grid.rebuild(world.patches)
    world.org_grid.rebuild(world.organisms)
    return world
