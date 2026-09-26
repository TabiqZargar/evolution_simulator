"""Lightweight, deterministic event log for observing a running simulation.

The engine records high-level lifecycle events (births, deaths, mutations,
generation boundaries and feeding) into a capped in-memory queue. The log is
*observational only*: it never feeds back into the simulation, is not part of
``EngineState``/persistence, and consumes randomness only through values the
engine already computes, so a seeded run produces an identical event stream.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator

# Event kinds. ``EV_MUTATION`` carries the gene count in ``value``; ``EV_FEED``
# carries the tick's total food consumed on ``value``.
EV_BIRTH = "birth"
EV_DEATH = "death"
EV_MUTATION = "mutation"
EV_GENERATION = "generation"
EV_FEED = "feed"
EVENT_KINDS: tuple[str, ...] = (
    EV_BIRTH,
    EV_DEATH,
    EV_MUTATION,
    EV_GENERATION,
    EV_FEED,
)


@dataclass(frozen=True)
class SimEvent:
    kind: str
    tick: int
    generation: int
    detail: str = ""
    value: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "tick": self.tick,
            "generation": self.generation,
            "detail": self.detail,
            "value": self.value,
        }


class EventLog:
    """A bounded, append-only buffer of :class:`SimEvent` records.

    ``capacity`` limits the number of retained events (ring behaviour: oldest
    are dropped first). A capacity of 0 disables recording entirely.
    """

    __slots__ = ("capacity", "_events")

    def __init__(self, capacity: int = 5000) -> None:
        self.capacity = max(capacity, 0)
        self._events: deque[SimEvent] = deque(maxlen=self.capacity)

    def record(
        self,
        kind: str,
        tick: int,
        generation: int,
        detail: str = "",
        value: float = 0.0,
    ) -> None:
        if self.capacity <= 0:
            return
        self._events.append(SimEvent(kind, tick, generation, detail, value))

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self) -> Iterator[SimEvent]:
        return iter(self._events)

    def __getitem__(self, index: int) -> SimEvent:
        return self._events[index]

    def last(self) -> SimEvent | None:
        return self._events[-1] if self._events else None

    def to_list(self) -> list[SimEvent]:
        """A shallow snapshot of the retained events (oldest first)."""
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()
