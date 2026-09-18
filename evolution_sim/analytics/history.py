"""Historical record of generation statistics for charts and experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evolution_sim.analytics.statistics import GenerationStats


@dataclass
class History:
    entries: list[GenerationStats] = field(default_factory=list)

    def push(self, stats: GenerationStats) -> None:
        self.entries.append(stats)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> GenerationStats:
        return self.entries[index]

    def series(self, name: str) -> list[float]:
        return [float(getattr(e, name)) for e in self.entries]

    def generations(self) -> list[int]:
        return [e.generation for e in self.entries]

    def last(self) -> GenerationStats | None:
        return self.entries[-1] if self.entries else None

    def summary(self, tail: int = 5) -> list[GenerationStats]:
        return self.entries[-tail:]

    def to_dict(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.entries]

    @classmethod
    def from_list(cls, data: list[dict[str, Any]]) -> "History":
        history = cls()
        for entry in data:
            history.entries.append(GenerationStats.from_dict(entry))
        return history
