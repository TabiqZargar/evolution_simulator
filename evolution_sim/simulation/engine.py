"""The simulation engine.

The engine is fully independent of Pygame. It:

* builds a deterministic world (terrain, food, climate, events)
* simulates organism behaviour tick by tick (only near neighbours are queried
  through the spatial index)
* resolves feeding, predation and reproduction
* advances generations and records statistics

Everything random flows through one ``random.Random`` instance, so a given
seed + configuration reproduces the same run.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

from evolution_sim.analytics.history import History
from evolution_sim.analytics.statistics import (
    GenerationStats,
    Species,
    detect_species,
    snapshot_stats,
)
from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.events import EventManager
from evolution_sim.simulation.fitness import compute_fitness, expected_lifespan
from evolution_sim.simulation.genome import PREDATOR_TRAITS, PREY_TRAITS, Genome
from evolution_sim.simulation.organism import Organism
from evolution_sim.simulation.reproduction import (
    can_reproduce,
    create_offspring,
    find_mate,
    offspring_position,
)
from evolution_sim.simulation.resources import FoodPatch
from evolution_sim.simulation.world import World

logger = logging.getLogger(__name__)


@dataclass
class EngineState:
    """Plain-data snapshot used by the persistence layer."""

    config: SimulationConfig
    rng: random.Random
    world: World
    generation: int
    tick: int
    event_manager: EventManager
    history: History
    pedigree: deque[dict[str, Any]]
    founder_count: int


class Engine:
    """Owns the world and advances it."""

    def __init__(
        self,
        config: SimulationConfig,
        *,
        rng: Optional[random.Random] = None,
        world: Optional[World] = None,
        generation: int = 1,
        tick: int = 0,
        event_manager: Optional[EventManager] = None,
        history: Optional[History] = None,
        pedigree: Optional[deque[dict[str, Any]]] = None,
        founder_count: int = 0,
    ) -> None:
        self.config = config
        self.rng = rng if rng is not None else random.Random(config.seed)
        self.world = world if world is not None else World(self.config, self.rng)
        self.generation = generation
        self.tick = tick
        self.history = history if history is not None else History()
        self.event_manager = event_manager or EventManager(
            enabled=config.events_enabled,
            frequency=config.event_frequency,
            min_duration=config.event_min_duration,
            max_duration=config.event_max_duration,
            rng=self.rng,
        )
        self.pedigree: deque[dict[str, Any]] = pedigree if pedigree is not None else deque()
        self.founder_count = founder_count
        self.births = 0
        self.deaths = 0
        self._species_list: Optional[list[Species]] = None

    # ------------------------------------------------------------------ setup
    def populate(self) -> None:
        """Create the founder population (and predators when enabled)."""
        for _ in range(self.config.initial_population):
            genome = Genome.random(self.rng, PREY_TRAITS)
            self.founder_count += 1
            self.world.spawn_organism(genome, generation=1, energy=self.config.starting_energy)
        if self.config.predators_enabled:
            for _ in range(self.config.predator_initial_population):
                center = {
                    "attack": 0.65,
                    "speed": 0.55,
                    "vision": 0.55,
                    "size": 0.7,
                }
                genome = Genome.random(self.rng, PREDATOR_TRAITS, center=center)
                self.founder_count += 1
                self.world.spawn_organism(genome, generation=1, energy=self.config.starting_energy)

    def init(self) -> None:
        """Generate food + organisms + indices. Call before stepping."""
        self.world.generate_food()
        self.populate()
        self.world.rebuild_indices()

    # ------------------------------------------------------------------ access
    def alive(self) -> list[Organism]:
        return [o for o in self.world.organisms if o.alive]

    def alive_count(self) -> int:
        return sum(1 for o in self.world.organisms if o.alive)

    def alive_predators(self) -> list[Organism]:
        return [o for o in self.world.organisms if o.alive and o.is_predator]

    def alive_prey(self) -> list[Organism]:
        return [o for o in self.world.organisms if o.alive and not o.is_predator]

    def organism_by_id(self, oid: int) -> Optional[Organism]:
        for o in self.world.organisms:
            if o.organism_id == oid:
                return o
        return None

    def current_temperature(self) -> float:
        return self._temperature_now()

    def species_list(self) -> Optional[list[Species]]:
        return self._species_list

    # ------------------------------------------------------------------ core
    def step(self) -> None:
        """Advance the simulation by exactly one tick."""
        self.tick += 1
        self._update_environment()

        if not self.alive():
            self._record_generation_if_due()
            return

        temperature = self._temperature_now()
        self.world.rebuild_indices()
        for organism in self.world.organisms:
            if organism.alive:
                self._step_organism(organism, temperature)

        self.world.rebuild_indices()
        self._feed_all()
        if self.config.predators_enabled:
            self._resolve_predation()
        self._try_reproduce()
        self._apply_deaths()
        self._record_generation_if_due()

    def _update_environment(self) -> None:
        self.event_manager.update(self.tick)
        food_mult = self.event_manager.food_multiplier()
        self.world.regrow_food(food_mult)

    def _temperature_now(self) -> float:
        season = math.sin(
            2.0 * math.pi * (self.tick + 40) / max(self.config.seasonal_period, 1)
        )
        temp = self.config.base_temperature + self.config.seasonal_amplitude * season
        temp += self.event_manager.temperature_delta()
        return min(max(temp, 0.0), 1.0)

    # ------------------------------------------------------------- thermal
    def _thermal_stress(self, organism: Organism, temperature: float) -> float:
        tolerance = organism.temperature_tolerance
        halfwidth = 0.12 + tolerance * 0.38  # 0.12 (fragile) .. 0.5 (hardy)
        optimum = 0.5 + (tolerance - 0.5) * 0.5  # prefers 0.25 (cold) .. 0.75 (warm)
        deviation = abs(temperature - optimum)
        if deviation <= halfwidth:
            return 0.0
        return min(2.0, (deviation - halfwidth) / 0.5)

    # ------------------------------------------------------------- organism step
    def _step_organism(self, organism: Organism, temperature: float) -> None:
        organism.age += 1
        if organism.reproduction_cooldown > 0:
            organism.reproduction_cooldown -= 1

        lifespan = self._max_age(organism)
        progress = organism.age / max(lifespan, 1)
        if progress > 0.8:
            organism.health -= 0.012 * (progress - 0.8) * 10.0
        if organism.age >= lifespan:
            organism.health -= 0.05

        stress = self._thermal_stress(organism, temperature)
        if stress > 0.0:
            organism.energy = max(0.0, organism.energy - stress * self.config.temperature_stress_energy)
            if stress > 0.5:
                organism.health -= (stress - 0.5) * self.config.temperature_stress_health

        base = 0.02 + 0.03 * organism.metabolism
        multiplier = 1.0 + stress * 0.5
        if organism._cache.get("_predator", 0.0) != 0.0:
            multiplier *= self.config.predator_metabolism_multiplier
        if temperature < 0.3 or temperature > 0.7:
            multiplier *= 1.15
        metabolism = base * multiplier
        organism.energy_spent += metabolism
        organism.energy = max(0.0, organism.energy - metabolism)

        if organism.energy < self.config.rest_energy_threshold:
            return  # rest: no movement when low on energy

        vision = self._vision_radius(organism)
        if organism.is_predator:
            self._behave_predator(organism, vision)
        else:
            self._behave_prey(organism, vision)

    # ------------------------------------------------------------- perception
    def _vision_radius(self, organism: Organism) -> float:
        trait = organism.vision
        return self.config.vision_base + (self.config.vision_range - self.config.vision_base) * trait

    def _step_distance(self, organism: Organism) -> float:
        return 0.15 + organism.speed * 0.55

    # ------------------------------------------------------------- behaviour
    def _behave_prey(self, organism: Organism, vision: float) -> None:
        if self.config.predators_enabled:
            # limit the "panic" radius so distant predators don't cause an
            # expensive whole-arena scan every tick.
            flee_vision = min(vision * self.config.flee_vision_factor, 9.0)
            danger = self.world.org_grid.nearest(
                organism.x,
                organism.y,
                flee_vision,
                predicate=lambda o: o is not organism and o.alive and o.is_predator,
            )
            if danger is not None:
                if organism.speed > danger.speed + 0.04:
                    self._move_away(organism, danger.x, danger.y)
                else:
                    organism.fights_lost += 1  # stand ground; may wound the hunter later
                return

        # Graze in place when standing on a food patch: feed without moving.
        # Full organisms stop grazing and go explore for fresh patches.
        if organism.energy < self.config.max_energy and self.world.food_grid.query(
            organism.x, organism.y, self.config.eating_radius
        ):
            return

        food: list[FoodPatch] = self.world.food_grid.query(organism.x, organism.y, vision)
        if food:
            best = max(food, key=lambda f: (f.quantity, -f.patch_id))
            self._move_towards(organism, best.x, best.y)
            return

        self._wander(organism)

    def _behave_predator(self, organism: Organism, vision: float) -> None:
        prey = self.world.org_grid.nearest(
            organism.x,
            organism.y,
            vision,
            predicate=lambda o: o is not organism and o.alive and not o.is_predator,
        )
        if prey is not None:
            self._move_towards(organism, prey.x, prey.y)
            return
        self._wander(organism)

    # ------------------------------------------------------------- movement
    def _apply_movement(self, organism: Organism, dx: float, dy: float, *, wander: bool = False) -> None:
        organism.last_x, organism.last_y = organism.x, organism.y
        new_x = organism.x + dx
        new_y = organism.y + dy

        penalty = self.world.terrain.movement_penalty_at(new_x, new_y) if not wander else 1.0
        cost = self.config.movement_energy_cost * (0.5 + organism.speed) * penalty
        if organism.energy - cost <= 0.0:
            organism.energy = 0.0
            organism.energy_spent += cost
            return
        organism.energy -= cost
        organism.energy_spent += cost

        if new_x <= 0.0 or new_x >= self.world.width or new_y <= 0.0 or new_y >= self.world.height:
            organism.x = min(max(organism.x, 0.0), self.world.width)
            organism.y = min(max(organism.y, 0.0), self.world.height)
            return
        organism.x = new_x
        organism.y = new_y

    def _move_towards(self, organism: Organism, target_x: float, target_y: float) -> None:
        step = self._step_distance(organism)
        dx = target_x - organism.x
        dy = target_y - organism.y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= 1e-6:
            return
        self._apply_movement(organism, dx / dist * step, dy / dist * step)

    def _move_away(self, organism: Organism, from_x: float, from_y: float) -> None:
        dx = organism.x - from_x
        dy = organism.y - from_y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist <= 1e-6:
            return
        step = self._step_distance(organism) * 1.2
        self._apply_movement(organism, dx / dist * step, dy / dist * step)

    def _max_age(self, organism: Organism) -> int:
        cached = organism._cache.get("max_age")
        if cached is None:
            lifespan = expected_lifespan(organism, self.config)
            organism._cache["max_age"] = lifespan
            return lifespan
        return int(cached)

    def _wander(self, organism: Organism) -> None:
        angle = (self.rng.random() * 2.0 - 1.0) * self.config.wander_angle_sigma
        speed_trait = organism.speed
        magnitude = self._step_distance(organism) * (0.4 + 0.6 * speed_trait)
        heading = self._heading_of(organism)
        self._apply_movement(
            organism,
            magnitude * math.cos(heading + angle),
            magnitude * math.sin(heading + angle),
            wander=True,
        )

    def _heading_of(self, organism: Organism) -> float:
        if organism.x != organism.last_x or organism.y != organism.last_y:
            return math.atan2(organism.y - organism.last_y, organism.x - organism.last_x)
        return self.rng.random() * 2.0 * math.pi

    # ------------------------------------------------------------- feeding
    def _feed_all(self) -> None:
        config = self.config
        eating_radius = config.eating_radius
        max_energy = config.max_energy
        for organism in self.world.organisms:
            if not organism.alive:
                continue
            capacity = max_energy - organism.energy
            if capacity <= 0.0:
                continue
            near: list[FoodPatch] = self.world.food_grid.query(organism.x, organism.y, eating_radius)
            if not near:
                continue
            patch = max(near, key=lambda p: (p.quantity, -p.patch_id))
            if patch.quantity <= 0.0:
                continue
            take = min(config.ingestion_rate, patch.quantity, capacity)
            taken = patch.consume(max(take, 0.0))
            efficiency = 0.35 + organism.efficiency * 0.65
            gain = taken * patch.nutrition * efficiency
            organism.energy = min(max_energy, organism.energy + gain)
            organism.energy_harvested += gain

    # ------------------------------------------------------------- predation
    def _resolve_predation(self) -> None:
        config = self.config
        for predator in self.world.organisms:
            if not predator.alive or not predator.is_predator:
                continue
            if predator.energy > config.max_energy * config.predator_hunt_energy_fraction:
                continue  # not hungry: don't bother hunting
            vision = self._vision_radius(predator)
            target = self.world.org_grid.nearest(
                predator.x,
                predator.y,
                vision,
                predicate=lambda o: o.alive and not o.is_predator,
            )
            if target is None:
                predator.health -= 0.002  # hungry predators waste away
                continue
            within_range = predator.distance_to(target) <= config.predator_attack_range + target.size * 0.4
            if within_range:
                self._attack(predator, target)
            else:
                self._move_towards(predator, target.x, target.y)

    def _attack(self, predator: Organism, prey: Organism) -> None:
        if not prey.alive:
            return
        config = self.config
        p_success = 0.22 + 0.15 * predator.attack
        p_success += 0.16 * (predator.speed - prey.speed)
        p_success += 0.08 * predator.size - 0.08 * prey.size - 0.10 * prey.aggression
        p_success = min(max(p_success, 0.04), 0.9)

        if predator.hunt_cooldown > 0:
            predator.hunt_cooldown -= 1
            self._move_towards(predator, prey.x, prey.y)
            return

        if self.rng.random() < p_success:
            efficacy = 0.3 + 0.7 * (0.4 + predator.efficiency * 0.6)
            gain = config.predator_hunt_reward * efficacy
            predator.energy = min(config.max_energy, predator.energy + gain)
            predator.energy_harvested += gain
            predator.fights_won += 1
            self._kill(prey, cause="hunted")
        else:
            counter = prey.aggression * prey.size * 0.45
            if self.rng.random() < counter:
                predator.health -= 0.06 + 0.10 * prey.size
                predator.energy = max(0.0, predator.energy - 4.0)
                prey.fights_won += 1
            else:
                predator.energy = max(0.0, predator.energy - 2.0)
        predator.hunt_cooldown = config.predator_hunt_cooldown
        predator.energy_spent += 2.0

    # ------------------------------------------------------------- reproduction
    def _try_reproduce(self) -> None:
        config = self.config
        pop = self.alive_count()
        if pop >= config.max_population:
            return
        for organism in self.world.organisms:
            if not organism.alive or not can_reproduce(organism, config):
                continue
            if pop >= config.max_population:
                break
            if organism.is_predator:
                pred_count = self._count_kind(True)
                if pred_count >= config.predator_max_population:
                    continue
                if self.alive_prey_count() == 0:
                    continue  # no prey to support more predators
            mates: list[Organism] = [
                o
                for o in self.world.org_grid.query(organism.x, organism.y, config.mating_range)
                if o.alive and o is not organism and o.is_predator == organism.is_predator
            ]
            mate = find_mate(organism, mates, config, self.rng)
            if mate is None:
                continue
            self._birth(organism, mate)
            pop += 1

    def _birth(self, mother: Organism, father: Organism) -> None:
        config = self.config
        genome, energy, _cost = create_offspring(mother, father, config, self.rng)
        child_gen = max(mother.generation, father.generation) + 1
        x, y = offspring_position(mother, father, self.rng, config)
        child = self.world.spawn_organism(
            genome,
            generation=child_gen,
            energy=energy,
            parent_a=mother.organism_id,
            parent_b=father.organism_id,
            x=x,
            y=y,
        )
        self._remember_pedigree(child)
        self.births += 1

    def _remember_pedigree(self, organism: Organism) -> None:
        self.pedigree.append({"id": organism.organism_id, "a": organism.parent_a, "b": organism.parent_b})
        if len(self.pedigree) > self.config.pedigree_record_limit:
            self.pedigree.popleft()

    # ------------------------------------------------------------- deaths
    def _apply_deaths(self) -> None:
        for organism in self.world.organisms:
            if not organism.alive:
                continue
            cause: Optional[str] = None
            if organism.energy <= 0.0:
                cause = "starvation"
            elif organism.health <= 0.0:
                cause = "stress"
            elif organism.age >= self._max_age(organism) and organism.health <= 0.65:
                cause = "old age"
            if cause:
                self._kill(organism, cause)

    def _kill(self, organism: Organism, cause: str) -> None:
        if not organism.alive:
            return
        living_children = self._living_children().get(organism.organism_id, 0)
        fitness = compute_fitness(organism, self.config, living_children)
        organism.record_death(cause, final_fitness=fitness)
        self.deaths += 1

    def _living_children(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for o in self.world.organisms:
            if not o.alive:
                continue
            if o.parent_a is not None:
                counts[o.parent_a] = counts.get(o.parent_a, 0) + 1
            if o.parent_b is not None:
                counts[o.parent_b] = counts.get(o.parent_b, 0) + 1
        return counts

    # ------------------------------------------------------------- generations
    def _record_generation_if_due(self) -> None:
        if self.tick % self.config.generation_length == 0:
            self.advance_generation()

    def advance_generation(self) -> None:
        """End the current generation: replenish, prune, detect species, record stats."""
        alive = self.alive()
        if self.config.replenish_to_target and alive:
            self._replenish(alive)
        self.world.prune_dead(self.config.dead_organism_budget)
        alive = self.alive()
        self._species_list = detect_species(alive, self.config) if (self.config.species_enabled and alive) else None

        event = self.event_manager.strongest_event()
        stats = snapshot_stats(
            self.config,
            self.generation,
            self.tick,
            self.world.organisms,
            self.births,
            self.deaths,
            self.world.total_food(),
            self._temperature_now(),
            event.label() if event else None,
            self._species_list,
        )
        self.history.push(stats)
        self.births = 0
        self.deaths = 0
        self.generation += 1
        if stats.extinct:
            logger.warning("World went extinct at generation %d.", stats.generation)

    def _replenish(self, alive: list[Organism]) -> None:
        """Recovery mechanism: if a kind's population fell below its target,
        let the most fit survivors of that kind reproduce until the target is
        restored. This biases *whose* genes get restored, but never edits
        genomes directly. Predators are capped at a fraction of the prey
        population, so a crash in prey always drags predators down with it."""
        config = self.config
        prey_alive = [o for o in alive if not o.is_predator]
        pred_alive = [o for o in alive if o.is_predator]
        self._replenish_kind(prey_alive, config.target_population)
        if config.predators_enabled and pred_alive:
            pred_target = min(
                config.predator_max_population,
                round(self.alive_prey_count() * config.predator_replenish_fraction),
            )
            pred_target = max(pred_target, len(pred_alive))
            if pred_target < config.predator_initial_population:
                pred_target = config.predator_initial_population
            self._replenish_kind(pred_alive, pred_target)
        self.world.rebuild_indices()

    def alive_prey_count(self) -> int:
        return sum(1 for o in self.world.organisms if o.alive and not o.is_predator)

    def _replenish_kind(self, candidates: list[Organism], target: int) -> None:
        """Birth from the fittest same-kind survivors until ``target`` is met."""
        if not candidates or self._count_kind(candidates[0].is_predator) >= target:
            return
        config = self.config
        living = self._living_children()
        rank = sorted(
            candidates,
            key=lambda o: compute_fitness(o, config, living.get(o.organism_id, 0)),
            reverse=True,
        )
        top = rank[: max(4, len(rank) // 2)]
        safety = 0
        is_predator = candidates[0].is_predator
        while (
            self._count_kind(is_predator) < target
            and safety < target * 3 + config.target_population
        ):
            safety += 1
            mother = self.rng.choice(top)
            father = self.rng.choice(top)
            if mother is father or mother.is_predator != is_predator:
                continue
            self._birth(mother, father)

    def _count_kind(self, is_predator: bool) -> int:
        return sum(1 for o in self.world.organisms if o.alive and o.is_predator == is_predator)

    # ------------------------------------------------------------- batch runs
    def run_ticks(self, ticks: int) -> None:
        for _ in range(ticks):
            self.step()

    def run_generations(self, generations: int) -> None:
        """Run whole generations; ``report_every`` controls the printed cadence."""
        target = self.generation + generations
        last_reported = self.generation - 1
        while self.generation < target:
            remaining = self.config.generation_length - (self.tick % self.config.generation_length)
            self.run_ticks(remaining)
            last = self.history.last()
            if last is not None and last.generation >= last_reported + self.config.report_every:
                self._report(last)
                last_reported = last.generation
            if self._is_extinct():
                break

    def _report(self, stats: GenerationStats) -> None:
        print(
            f"[gen {stats.generation:>5}] pop={stats.population:>4} "
            f"avg_fit={stats.avg_fitness:5.3f} max_fit={stats.max_fitness:5.3f} "
            f"diversity={stats.diversity:5.3f} food={stats.resource_total:7.1f} "
            f"temp={stats.temperature:4.2f} event={stats.event_label or '-':<24}"
        )

    def _is_extinct(self) -> bool:
        last = self.history.last()
        return bool(last and last.extinct)

    # ------------------------------------------------------------- persistence
    def snapshot(self) -> EngineState:
        return EngineState(
            config=self.config,
            rng=self.rng,
            world=self.world,
            generation=self.generation,
            tick=self.tick,
            event_manager=self.event_manager,
            history=self.history,
            pedigree=self.pedigree,
            founder_count=self.founder_count,
        )

    def state_signature(self) -> str:
        """Deterministic fingerprint of the current state, for tests/persistence."""
        alive = sorted((o for o in self.world.organisms if o.alive), key=lambda o: o.organism_id)
        parts = [f"gen={self.generation} tick={self.tick} pop={len(alive)}"]
        for o in alive:
            genes = "|".join(f"{k}:{o.genome.gene(k):.6f}" for k in sorted(o.genome.traits))
            parts.append(f"{o.organism_id}:{o.x:.4f},{o.y:.4f}:{o.energy:.2f}:{o.health:.3f}:{genes}")
        return hashlib.sha1(";".join(parts).encode("utf-8")).hexdigest()
