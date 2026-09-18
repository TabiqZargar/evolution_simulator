"""Simulation engine package (independent of any visualization).

Submodules are imported explicitly by consumers; this package does no eager
imports so ``evolution_sim.persistence`` and the analytics layer can always
import ``evolution_sim.simulation.genome`` etc. without import cycles.
"""
