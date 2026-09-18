"""Environmental events (drought, cold snaps, heat waves, booms, collapses).

Events are scheduled probabilistically by :class:`EventManager` and have a
duration and severity. They act as *selection pressure*: they change food
regeneration and/or temperature, and organisms either cope or suffer. Events
never edit genomes directly — adaptation emerges from differential survival.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from random import Random
from typing import Any, Optional


class EventKind(Enum):
    DROUGHT = "drought"
    COLD_SNAP = "cold_snap"
    HEAT_WAVE = "heat_wave"
    RESOURCE_BOOM = "resource_boom"
    RESOURCE_COLLAPSE = "resource_collapse"


@dataclass
class EnvironmentEvent:
    kind: EventKind
    magnitude: float  # in (0, 1]
    remaining_ticks: int

    @property
    def food_multiplier(self) -> float:
        """Multiplier applied to food regeneration while this event is live."""
        if self.kind == EventKind.DROUGHT or self.kind == EventKind.RESOURCE_COLLAPSE:
            return max(0.05, 1.0 - self.magnitude)
        if self.kind == EventKind.RESOURCE_BOOM:
            return 1.0 + self.magnitude
        return 1.0

    @property
    def temperature_delta(self) -> float:
        if self.kind == EventKind.COLD_SNAP:
            return -self.magnitude * 0.5
        if self.kind == EventKind.HEAT_WAVE:
            return self.magnitude * 0.5
        return 0.0

    def is_temperature_event(self) -> bool:
        return self.kind in (EventKind.COLD_SNAP, EventKind.HEAT_WAVE)

    def label(self) -> str:
        return f"{self.kind.value} -{self.remaining_ticks}"


class EventManager:
    """Schedules and tracks active events. Deterministic given the RNG."""

    def __init__(
        self,
        enabled: bool,
        frequency: float,
        min_duration: int,
        max_duration: int,
        rng: Optional[Random] = None,
    ) -> None:
        self.enabled = enabled
        self.frequency = frequency
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.rng = rng if rng is not None else Random(0)
        self.active: list[EnvironmentEvent] = []
        self.counts: dict[str, int] = {
            kind.value: 0 for kind in EventKind
        }
        self.last_triggered_tick: int = 0

    def _roll_kind(self) -> EventKind:
        roll = self.rng.random()
        # Lean towards drought / heat waves as the "default" stressors.
        cumulative: list[tuple[float, EventKind]] = [
            (0.28, EventKind.DROUGHT),
            (0.46, EventKind.COLD_SNAP),
            (0.64, EventKind.HEAT_WAVE),
            (0.82, EventKind.RESOURCE_BOOM),
            (1.00, EventKind.RESOURCE_COLLAPSE),
        ]
        for threshold, kind in cumulative:
            if roll <= threshold:
                return kind
        return EventKind.DROUGHT

    def update(self, tick: int) -> None:
        """Tick the event system: decay active events, maybe schedule a new one."""
        self.last_triggered_tick = max(self.last_triggered_tick, tick)
        if self.enabled and self.active:
            for event in self.active:
                event.remaining_ticks -= 1
            self.active = [e for e in self.active if e.remaining_ticks > 0]

        if self.enabled and len(self.active) < 2:
            if self.rng.random() < self.frequency:
                magnitude = 0.25 + self.rng.random() * 0.65
                duration = self.min_duration + self.rng.randint(0, self.max_duration - self.min_duration)
                kind = self._roll_kind()
                self.active.append(EnvironmentEvent(kind, magnitude, duration))
                self.counts[kind.value] += 1

    def food_multiplier(self) -> float:
        if not self.active:
            return 1.0
        product = 1.0
        for event in self.active:
            product *= event.food_multiplier
        return product

    def temperature_delta(self) -> float:
        if not self.active:
            return 0.0
        return sum(e.temperature_delta for e in self.active)

    def strongest_event(self) -> Optional[EnvironmentEvent]:
        if not self.active:
            return None
        return max(self.active, key=lambda e: e.magnitude)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "frequency": self.frequency,
            "min_duration": self.min_duration,
            "max_duration": self.max_duration,
            "active": [
                {"kind": e.kind.value, "magnitude": e.magnitude, "remaining_ticks": e.remaining_ticks}
                for e in self.active
            ],
            "counts": dict(self.counts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], rng: Random) -> "EventManager":
        manager = cls(
            enabled=data.get("enabled", True),
            frequency=data.get("frequency", 0.003),
            min_duration=data.get("min_duration", 80),
            max_duration=data.get("max_duration", 320),
            rng=rng,
        )
        manager.active = [
            EnvironmentEvent(EventKind(e["kind"]), e["magnitude"], e["remaining_ticks"])
            for e in data.get("active", [])
        ]
        for k, v in data.get("counts", {}).items():
            manager.counts[k] = v
        return manager
