"""Shared fixtures: tiny worlds and fast engines for unit tests."""

from __future__ import annotations

import math

import pytest

from evolution_sim.config import SimulationConfig
from evolution_sim.simulation.engine import Engine


@pytest.fixture
def config() -> SimulationConfig:
    return SimulationConfig()


@pytest.fixture
def small_config() -> SimulationConfig:
    """A fast configuration for integration tests (predators off)."""
    return SimulationConfig(
        seed=7,
        world_width=40,
        world_height=40,
        initial_population=120,
        target_population=120,
        generation_length=120,
        event_frequency=0.0,
        report_every=100,
    )


@pytest.fixture
def predator_config() -> SimulationConfig:
    return SimulationConfig(
        seed=7,
        world_width=40,
        world_height=40,
        initial_population=120,
        target_population=120,
        predator_initial_population=12,
        generation_length=120,
        event_frequency=0.0,
        predators_enabled=True,
        report_every=100,
    )


@pytest.fixture
def engine(small_config: SimulationConfig) -> Engine:
    eng = Engine(small_config)
    eng.init()
    return eng


def assert_finite(engine: Engine) -> None:
    cfg = engine.config
    for org in engine.world.organisms:
        assert math.isfinite(org.x)
        assert math.isfinite(org.y)
        assert math.isfinite(org.energy)
        assert math.isfinite(org.health)
        assert org.genome.is_valid()
        assert 0.0 <= org.energy <= cfg.max_energy + 1e-6
        assert 0.0 <= org.x <= cfg.world_width
        assert 0.0 <= org.y <= cfg.world_height
