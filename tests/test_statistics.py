"""Statistics: seeded reproducibility, generation tracking, lifecycle tallies."""

from __future__ import annotations

import pytest

from evolution_sim.analytics.statistics import (
    DEATH_CAUSES,
    DEATH_STRESS,
    GenerationLedger,
    summarize,
)
from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.engine import Engine


def _run(cfg: SimulationConfig, generations: int) -> Engine:
    eng = Engine(cfg)
    eng.init()
    eng.run_generations(generations)
    return eng


# --------------------------------------------------------------------------- #
# Seeded reproducibility
# --------------------------------------------------------------------------- #
def test_seeded_run_reproduces_identical_history_and_state() -> None:
    cfg = SimulationConfig(
        seed=11,
        world_width=40,
        world_height=40,
        initial_population=120,
        target_population=120,
        generation_length=100,
        event_frequency=0.0,
    )
    a = _run(cfg, 4)
    b = _run(cfg, 4)
    assert a.state_signature() == b.state_signature()
    assert a.generation == b.generation and a.tick == b.tick
    assert a.history.to_dict() == b.history.to_dict()
    assert [e.to_dict() for e in a.events] == [e.to_dict() for e in b.events]


def test_different_seed_produces_different_run() -> None:
    base = SimulationConfig(
        seed=11,
        world_width=40,
        world_height=40,
        initial_population=120,
        generation_length=100,
    )
    a = _run(base, 3)
    b = _run(base.with_updates(seed=12), 3)
    assert a.state_signature() != b.state_signature()


# --------------------------------------------------------------------------- #
# Generation tracking
# --------------------------------------------------------------------------- #
def test_generation_records_advance_in_order(small_config: SimulationConfig) -> None:
    eng = _run(small_config, 3)
    assert len(eng.history) == 3
    assert eng.generation == 4
    gens = eng.history.generations()
    assert gens == [1, 2, 3]
    assert eng.history.run_summary().last_generation == 3
    for entry in eng.history:
        assert entry.tick == entry.generation * small_config.generation_length
        assert entry.population >= 0


# --------------------------------------------------------------------------- #
# Births / deaths / reproduction tally
# --------------------------------------------------------------------------- #
def test_births_and_deaths_are_consistent(small_config: SimulationConfig) -> None:
    eng = _run(small_config, 4)
    entries = eng.history.entries
    for entry in entries:
        assert entry.births >= 0
        assert entry.deaths >= 0
        assert entry.reproductions == entry.births
    summary = eng.history.run_summary()
    assert summary.total_births == sum(e.births for e in entries)
    assert summary.total_deaths == sum(e.deaths for e in entries)
    assert summary.total_reproductions == summary.total_births
    assert summary.total_births > 0  # replenishment keeps the population alive


def test_resources_consumed_is_recorded(small_config: SimulationConfig) -> None:
    eng = _run(small_config, 3)
    total = eng.history.run_summary().total_resources_consumed
    assert total >= 0.0
    assert any(e.resources_consumed > 0.0 for e in eng.history)


# --------------------------------------------------------------------------- #
# Mutation tracking
# --------------------------------------------------------------------------- #
def test_mutation_tally_matches_event_stream(small_config: SimulationConfig) -> None:
    eng = _run(small_config, 4)
    from evolution_sim.analytics.observability import EV_MUTATION

    total = eng.history.run_summary().total_mutations
    assert all(e.mutations >= 0 for e in eng.history)
    assert total == sum(e.mutations for e in eng.history)
    assert total == sum(e.value for e in eng.events if e.kind == EV_MUTATION)
    assert total > 0  # random founder mutation_rate genes guarantee mutations


def test_low_config_rate_records_fewer_mutations(small_config: SimulationConfig) -> None:
    baseline = _run(small_config, 4).history.run_summary().total_mutations
    reduced = _run(small_config.with_updates(mutation_rate=0.0), 4).history.run_summary().total_mutations
    assert 0 <= reduced < baseline


# --------------------------------------------------------------------------- #
# Lifespan and death causes
# --------------------------------------------------------------------------- #
def test_death_ledger_records_age_and_cause() -> None:
    ledger = GenerationLedger()
    ledger.record_death(10, "starvation")
    ledger.record_death(20, "starvation")
    ledger.record_death(30, "stress")
    assert ledger.deaths_at_death_count == 3
    assert ledger.avg_lifespan == pytest.approx(20.0)
    assert ledger.deaths_by_cause == {"starvation": 2, "stress": 1}


def test_deaths_by_cause_in_history(predator_config: SimulationConfig) -> None:
    eng = _run(predator_config, 3)
    for entry in eng.history:
        assert set(entry.deaths_by_cause) <= set(DEATH_CAUSES)
        assert entry.environmental_deaths == entry.deaths_by_cause.get(DEATH_STRESS, 0)
        assert entry.avg_lifespan >= 0.0
    summary = eng.history.run_summary()
    assert summary.environmental_deaths >= 0
    assert summary.total_deaths == sum(e.deaths for e in eng.history)


# --------------------------------------------------------------------------- #
# Alive-population descriptors
# --------------------------------------------------------------------------- #
def test_avg_age_and_energy_are_finite(predator_config: SimulationConfig) -> None:
    eng = _run(predator_config, 3)
    for entry in eng.history:
        assert 0.0 <= entry.avg_age <= predator_config.lifespan_max_ticks
        assert 0.0 <= entry.avg_energy <= predator_config.max_energy


# --------------------------------------------------------------------------- #
# Run-level summary
# --------------------------------------------------------------------------- #
def test_run_summary_population_bounds(small_config: SimulationConfig) -> None:
    eng = _run(small_config, 4)
    summary = eng.history.run_summary()
    populations = [e.population for e in eng.history]
    assert summary.max_population == max(populations)
    assert summary.min_population == min(populations)
    assert summary.min_population <= summary.avg_population <= summary.max_population
    assert summary.final_population == populations[-1]
    assert summary.generations == len(eng.history)
    assert summary.extinct == (populations[-1] == 0)


def test_summarize_empty_history() -> None:
    summary = summarize([])
    assert summary.generations == 0
    assert summary.final_population == 0
    assert summary.max_population == 0 and summary.min_population == 0
