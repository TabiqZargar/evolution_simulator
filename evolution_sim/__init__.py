"""Evolution Simulator.

A pure-Python evolutionary simulation: organisms forage for food, compete,
reproduce sexually, inherit and mutate genomes, and adapt to a changing
environment over many generations. The simulation engine is independent of
the Pygame visualization layer.
"""

__version__ = "0.1.0"

from evolution_sim.config import SimulationConfig, load_config, save_config

__all__ = ["SimulationConfig", "load_config", "save_config", "__version__"]
