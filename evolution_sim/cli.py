"""Command-line interface.

Run modes:

* default (no ``--headless``): open the Pygame visualisation
* ``--headless``: run generations silently, printing/progressing to files
* ``--experiment``: parameter sweep over seeds (headless)

Any ``SimulationConfig`` field can be overridden as ``--field-name value``,
e.g. ``--population 300 --mutation-rate 0.1``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Optional

from evolution_sim import __version__
from evolution_sim.analytics import experiments
from evolution_sim.config import (
    SIMULATION_CONFIG_FIELDS,
    SimulationConfig,
    apply_cli_overrides,
    load_config,
)
from evolution_sim.persistence import load_state, save_state
from evolution_sim.simulation.engine import Engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evolution_sim",
        description="Pure-Python evolution simulator (genetics, ecology, visualisation).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    mode = parser.add_argument_group("mode")
    mode.add_argument("--headless", action="store_true", help="run without the Pygame window")
    mode.add_argument(
        "--experiment",
        metavar="FIELD",
        help="sweep this configuration field across --experiment-values",
    )

    run = parser.add_argument_group("run controls")
    run.add_argument("--generations", type=int, default=50, help="generations to run (default 50)")
    run.add_argument("--seed", type=int, default=None, help="random seed (default 42)")
    run.add_argument("--config", metavar="PATH", default=None, help="JSON config file to load")
    run.add_argument("--report-every", type=int, default=None, help="progress line every N generations")
    run.add_argument("--load", metavar="PATH", default=None, help="resume from a saved state")
    run.add_argument("--save", metavar="PATH", default=None, help="save state to PATH when done")
    run.add_argument("--save-every", type=int, default=None, help="auto-save every N generations")
    run.add_argument("--json-stats", metavar="PATH", default=None, help="write per-generation stats as JSON")
    run.add_argument("--csv-stats", metavar="PATH", default=None, help="write per-generation stats as CSV")

    exp = parser.add_argument_group("experiment options")
    exp.add_argument(
        "--experiment-values",
        default="",
        help="comma-separated values to try for --experiment (e.g. 0.02,0.05,0.10)",
    )
    exp.add_argument(
        "--experiment-seeds",
        default="",
        help="comma-separated seeds (default 1,2,3)",
    )
    exp.add_argument(
        "--experiment-output",
        metavar="PATH",
        default=None,
        help="write experiment rows to CSV (defaults to --csv-stats)",
    )

    return parser


ParserWithArgs = tuple[argparse.ArgumentParser, Any, dict[str, Any]]


def _collect_overrides(argv: list[str]) -> ParserWithArgs:
    parser = build_parser()
    known, leftover = parser.parse_known_args(argv)

    overrides: dict[str, Any] = {}
    if leftover:
        overrides = _parse_leftover_pairs(leftover)
    return parser, known, overrides


def _parse_leftover_pairs(leftover: list[str]) -> dict[str, Any]:
    """Turn ``['--mutation-rate', '0.1', '--population', '300']`` into a dict."""
    if not leftover:
        return {}
    if odd := len(leftover) % 2:
        raise SystemExit(f"unrecognised argument(s) or unpaired --field value: {leftover[-odd:]!r}")
    out: dict[str, Any] = {}
    for i in range(0, len(leftover), 2):
        flag = leftover[i]
        value = leftover[i + 1]
        if not flag.startswith("--"):
            raise SystemExit(f"unrecognised argument {flag!r}")
        field = flag[2:].replace("-", "_")
        if len(value) > 0 and value[0] == "-" and value[1:].isdigit() is False and value.lstrip("-").isdigit() is False:
            raise SystemExit(f"expected a value after {flag!r}, got {value!r}")
        out[field] = value
    return out


def _build_config(parser: argparse.ArgumentParser, args: Any, overrides: dict[str, Any]) -> SimulationConfig:
    config = load_config(args.config) if args.config else SimulationConfig()
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.report_every is not None:
        overrides["report_every"] = args.report_every
    try:
        return apply_cli_overrides(config, overrides.items())
    except ValueError as error:
        parser.error(str(error))


def _coerce_values(raw: str) -> list[Any]:
    values = [v.strip() for v in raw.split(",") if v.strip() != ""]
    if not values:
        raise SystemExit("--experiment-values must be a comma-separated list")
    coerced: list[Any] = []
    for v in values:
        try:
            coerced.append(int(v))
        except ValueError:
            try:
                coerced.append(float(v))
            except ValueError:
                coercion = v.lower() in ("true", "false")
                coerced.append(coercion if v.lower() in ("true", "false") else v)
    return coerced


def _coerce_seeds(raw: str) -> list[int]:
    values = [v.strip() for v in raw.split(",") if v.strip() != ""]
    if not values:
        return [1, 2, 3]
    return [int(v) for v in values]


def _run_headless(
    args: Any,
    config: SimulationConfig,
    overrides: dict[str, Any],
) -> Engine:
    if args.load:
        engine = load_state(args.load)
        if overrides:
            engine.config = apply_cli_overrides(engine.config, overrides.items())
        print(f"resumed from {args.load} at tick {engine.tick}", file=sys.stderr)
    else:
        engine = Engine(config)
        engine.init()

    started = time.perf_counter()
    generations = args.generations
    if args.save_every is not None and args.save_every > 0:
        os.makedirs("saves", exist_ok=True)
        while not engine._is_extinct() and generations > 0:
            step = min(generations, max(args.save_every, 1))
            engine.run_generations(step)
            generations -= step
            save_state(engine, f"saves/autosave_gen{engine.generation}.json")
    else:
        engine.run_generations(generations)
    elapsed = time.perf_counter() - started

    last = engine.history.last()
    if last is not None:
        print(
            f"[final gen {last.generation}] population={last.population} "
            f"avg_fitness={last.avg_fitness:.3f} max_fitness={last.max_fitness:.3f} "
            f"diversity={last.diversity:.3f} avg_speed={last.avg_speed:.3f} "
            f"prey={last.prey_population} predators={last.predator_population} "
            f"extinct={last.extinct}"
        )
    print(f"{elapsed:.1f}s wall time for {args.generations} generations", file=sys.stderr)

    if args.json_stats:
        _write_json_stats(engine, args.json_stats)
    if args.csv_stats:
        _write_csv_stats(engine, args.csv_stats)
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        save_state(engine, args.save)
        print(f"saved state to {args.save}", file=sys.stderr)
    return engine


def _write_json_stats(engine: Engine, path: str) -> None:

    payload = {"config": engine.config.to_dict(), "generations": engine.history.to_dict()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _write_csv_stats(engine: Engine, path: str) -> None:
    import csv

    if not engine.history.entries:
        return
    columns = list(engine.history.entries[0].to_dict().keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for entry in engine.history.entries:
            writer.writerow(entry.to_dict())


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser, args, overrides = _collect_overrides(argv)
    config = _build_config(parser, args, overrides)

    if args.experiment:
        if args.experiment not in SIMULATION_CONFIG_FIELDS:
            parser.error(f"unknown field for --experiment: {args.experiment!r}")
        values = _coerce_values(args.experiment_values)
        seeds = _coerce_seeds(args.experiment_seeds)
        rows = experiments.run_sweep(
            config,
            args.experiment,
            values,
            seeds,
            args.generations,
        )
        experiments.print_table(rows)
        out = args.experiment_output or args.csv_stats
        if out:
            experiments.write_csv(rows, out)
            print(f"wrote {len(rows)} experiment rows to {out}", file=sys.stderr)
        return 0

    if not args.headless:
        from evolution_sim.visualization import app

        return app.run(config, args, overrides, parser)

    engine = _run_headless(args, config, overrides)
    _ = engine
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
