"""Event log: recording, capacity, determinism and persistence isolation."""

from __future__ import annotations

from evolution_sim.analytics.observability import (
    EV_BIRTH,
    EV_DEATH,
    EV_FEED,
    EV_GENERATION,
    EV_MUTATION,
    EventLog,
)
from evolution_sim.config import SimulationConfig
from evolution_sim.persistence import load_state, save_state
from evolution_sim.simulation.engine import Engine


def _run(cfg: SimulationConfig, generations: int) -> Engine:
    eng = Engine(cfg)
    eng.init()
    eng.run_generations(generations)
    return eng


def test_event_log_is_capped_and_ring() -> None:
    log = EventLog(capacity=10)
    for n in range(25):
        log.record(EV_BIRTH, tick=n, generation=1, detail=str(n))
    assert len(log) == 10
    oldest = log[0]
    newest = log.last()
    assert oldest.detail == "15"
    assert newest.detail == "24"
    assert log.to_list()[0].tick == 15


def test_event_log_capacity_zero_disables() -> None:
    log = EventLog(capacity=0)
    log.record(EV_DEATH, tick=1, generation=1, detail="starvation")
    assert len(log) == 0


def test_event_log_clears() -> None:
    log = EventLog(capacity=5)
    log.record(EV_FEED, tick=1, generation=1, value=1.5)
    assert len(log) == 1
    log.clear()
    assert len(log) == 0


def test_engine_records_lifecycle_events(small_config: SimulationConfig) -> None:
    eng = _run(small_config, 3)
    kinds = {e.kind for e in eng.events}
    assert EV_GENERATION in kinds
    assert EV_BIRTH in kinds
    assert EV_FEED in kinds
    # death is not guaranteed every generation; it occurred in these runs
    assert EV_DEATH in kinds


def test_birth_and_mutation_events_carry_payload(small_config: SimulationConfig) -> None:
    eng = _run(small_config, 3)
    births = [e for e in eng.events if e.kind == EV_BIRTH]
    mutations = [e for e in eng.events if e.kind == EV_MUTATION]
    assert births
    assert all(e.detail.isdigit() for e in births)
    assert mutations
    assert all(e.value >= 1 for e in mutations)


def test_death_events_carry_cause_detail(predator_config: SimulationConfig) -> None:
    eng = _run(predator_config, 3)
    deaths = [e for e in eng.events if e.kind == EV_DEATH]
    assert deaths
    assert {e.detail for e in deaths} <= {"starvation", "stress", "old age", "hunted"}


def test_generation_events_recorded_at_boundaries(predator_config: SimulationConfig) -> None:
    eng = _run(predator_config, 3)
    gens = [e for e in eng.events if e.kind == EV_GENERATION]
    assert len(gens) == 3
    assert [e.generation for e in gens] == [1, 2, 3]
    assert all(e.value == float(eng.history[i].population) for i, e in enumerate(gens))


def test_seeded_events_are_identical() -> None:
    cfg = SimulationConfig(
        seed=9,
        world_width=40,
        world_height=40,
        initial_population=100,
        target_population=100,
        generation_length=100,
        event_frequency=0.0,
    )
    a = _run(cfg, 3)
    b = _run(cfg, 3)
    assert [e.to_dict() for e in a.events] == [e.to_dict() for e in b.events]


def test_events_are_not_persisted(tmp_path) -> None:
    eng = _run(SimulationConfig(seed=21, initial_population=50, generation_length=100), 2)
    assert len(eng.events) > 0
    save_state(eng, tmp_path / "state.json")
    restored = load_state(tmp_path / "state.json")
    assert len(restored.events) == 0
    assert restored.events.capacity == eng.events.capacity


def test_event_log_capacity_config_maps_to_log(small_config: SimulationConfig) -> None:
    cfg = small_config.with_updates(event_log_capacity=123)
    eng = Engine(cfg)
    eng.init()
    assert eng.events.capacity == 123

    disabled = small_config.with_updates(event_log_capacity=0)
    eng = Engine(disabled)
    eng.init()
    eng.run_generations(2)
    assert len(eng.events) == 0
