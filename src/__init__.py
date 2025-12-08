"""
OffarmLab - Offshore Farm Laboratory
====================================

A Python library for dynamic simulation of offshore aquaculture systems.

Modules:
    - config: Configuration loading and variable substitution
    - waves: Wave spectrum and wave field generation
    - forces: External force calculations (mooring, drag, wave)
    - hydro: Hydrodynamic coefficients and element systems
    - cables: Static cable mechanics
    - solvers: Numerical integration methods
    - simulation: Main simulation runner
"""

from offarmlab.config import ConfigLoader, VariableSystem
from offarmlab.simulation import Simulation

__version__ = "0.1.0"
__all__ = ["ConfigLoader", "VariableSystem", "Simulation", "__version__"]

