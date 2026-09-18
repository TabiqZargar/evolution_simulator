"""Parameter-sweep experiments for headless evolution runs.

An experiment varies one configuration field across several values and runs
each combination over several independent seeds. Results are returned as plain
rows (dicts) and can be written to CSV/JSON.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.engine import Engine

_STAT_FIELDS = (
    "population",
    "avg_fitness",
    "max_fitness",
    "diversity",
    "avg_speed",
    "avg_vision",
    "avg_size",
    "avg_metabolism",
    "avg_aggression",
    "species_count",
    "resource_total",
    "prey_population",
    "predator_population",
    "extinct",
)


def run_sweep(
    config: SimulationConfig,
    field: str,
    values: list[Any],
    seeds: list[int],
    generations: int,
    on_row: Optional[Callable[[dict[str, Any]], None]] = None,
) -> list[dict[str, Any]]:
    """Run ``field`` over ``values`` and each value over ``seeds``.

    ``field`` must be a valid ``SimulationConfig`` field; each run is executed
    fresh with ``seed``, ``field``, and ``value`` applied to a copy of
    ``config``. Returns one row dict per (seed, value) run.
    """
    if not hasattr(config, field):
        raise ValueError(f"unknown configuration field: {field!r}")

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for value in values:
            cfg = config.with_updates(**{field: value, "seed": int(seed)})
            engine = Engine(cfg)
            engine.init()
            started = time.perf_counter()
            engine.run_generations(generations)
            wall = time.perf_counter() - started

            row: dict[str, Any] = {
                "field": field,
                "value": _plain(value),
                "seed": int(seed),
                "generations": generations,
                "wall_seconds": round(wall, 2),
                "ticks": engine.tick,
                "final_generation": engine.generation,  # 1 + run length if it survived
            }
            last = engine.history.last()
            if last is not None:
                for f in _STAT_FIELDS:
                    row[f] = _plain(getattr(last, f))
            else:
                for f in _STAT_FIELDS:
                    row[f] = _plain(0)
            rows.append(row)
            if on_row is not None:
                on_row(row)
    return rows


def _plain(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str)) or value is None:
        return value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return str(value)


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(rows: list[dict[str, Any]], path: str | Path) -> None:
    Path(path).write_text(json.dumps(rows, indent=2), encoding="utf-8")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average the numeric stats of each value across its seeds."""
    summary: dict[Any, dict[str, float]] = {}
    for row in rows:
        key = row["value"]
        bucket = summary.setdefault(key, {"_count": 0})
        bucket["_count"] += 1
        for f in _STAT_FIELDS:
            if isinstance(row.get(f), (int, float)):
                bucket[f] = bucket.get(f, 0.0) + row[f]
    out: list[dict[str, Any]] = []
    for key, bucket in summary.items():
        n = bucket.pop("_count")
        means = {f: round(bucket[f] / n, 4) for f in bucket}
        out.append({"value": key, "runs": n, **means})
    out.sort(key=lambda r: _sortable(r["value"]))
    return out


def _sortable(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def print_table(rows: list[dict[str, Any]]) -> None:
    summary = summarize(rows)
    if not summary:
        return
    header = "value | runs | population | avg_fitness | max_fitness | diversity | avg_speed | extinct"
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['value']!s:>7} | {row['runs']:>4} | {row['population']:>10.1f} "
            f"| {row['avg_fitness']:>11.3f} | {row['max_fitness']:>11.3f} "
            f"| {row['diversity']:>9.3f} | {row['avg_speed']:>9.3f} | {row['extinct']:>7.0f}"
        )
