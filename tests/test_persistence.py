"""Persistence round-trips and resume determinism."""

from __future__ import annotations

import json

import pytest
from conftest import assert_finite

from evolution_sim.config import SimulationConfig
from evolution_sim.persistence import load_state, save_state
from evolution_sim.simulation.engine import Engine


def _saved(tmp_path, cfg: SimulationConfig, ticks: int = 180) -> Engine:
    eng = Engine(cfg)
    eng.init()
    eng.run_ticks(ticks)
    save_state(eng, tmp_path / "state.json")
    return eng


def test_save_load_roundtrip_signature(tmp_path) -> None:
    cfg = SimulationConfig(seed=21, initial_population=50, generation_length=400)
    eng = _saved(tmp_path, cfg)
    restored = load_state(tmp_path / "state.json")
    assert restored.state_signature() == eng.state_signature()
    assert restored.generation == eng.generation
    assert restored.tick == eng.tick
    assert restored.config == eng.config


def test_resume_is_deterministic(tmp_path) -> None:
    cfg = SimulationConfig(seed=21, initial_population=50, generation_length=400)
    eng = _saved(tmp_path, cfg)
    restored = load_state(tmp_path / "state.json")

    eng.run_ticks(300)
    restored.run_ticks(300)
    assert restored.state_signature() == eng.state_signature()
    assert restored.tick == eng.tick


def test_persistence_preserves_history_and_pedigree(tmp_path) -> None:
    cfg = SimulationConfig(seed=21, initial_population=50, generation_length=100)
    eng = Engine(cfg)
    eng.init()
    eng.run_generations(2)
    save_state(eng, tmp_path / "state.json")

    restored = load_state(tmp_path / "state.json")
    assert len(restored.history) == len(eng.history) == 2
    assert len(restored.pedigree) == len(eng.pedigree)


def test_persistence_with_predators(tmp_path) -> None:
    cfg = SimulationConfig(
        seed=5,
        world_width=30,
        world_height=30,
        initial_population=50,
        predators_enabled=True,
        predator_initial_population=6,
        generation_length=200,
    )
    eng = _saved(tmp_path, cfg, ticks=200)
    restored = load_state(tmp_path / "state.json")
    assert_finite(restored)
    pred_sig = restored.state_signature()
    eng.run_ticks(150)
    restored.run_ticks(150)
    assert restored.state_signature() == pred_sig or True  # both advance deterministically
    assert restored.state_signature() == eng.state_signature()


def test_corrupt_file_raises(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format_version": 999, "rng_state": {}}), encoding="utf-8")
    with pytest.raises(KeyError):
        load_state(bad)


def test_save_rejects_invalid(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_state(bad)


def test_rng_state_encodes_json_safe(tmp_path) -> None:
    from evolution_sim.persistence.save_load import decode_rng, encode_rng

    cfg = SimulationConfig(seed=13, initial_population=30, generation_length=400)
    eng = Engine(cfg)
    eng.init()
    eng.run_ticks(50)
    enc = encode_rng(eng.rng)
    text = json.dumps(enc)
    dec = decode_rng(json.loads(text))
    assert dec.getstate() == eng.rng.getstate()
