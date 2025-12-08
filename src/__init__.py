"""
Offarm-py - Offshore Farm Laboratory (Python)
==============================================

A Python library for dynamic simulation of offshore aquaculture systems.

Modules:
    - params: Configuration dataclasses (Env, OffSys, LineTypes, etc.)
    - Forces: External force calculations (mooring, drag, wave)
    - Cable: Cable mechanics and quasi-static mooring
    - Simu: Main simulation runner
    - integrator: Numerical integration methods

Quick Start:
    >>> from offarm import run_a_simulation
    >>> results = run_a_simulation("path/to/INPUT_manifest.json")

CLI Usage:
    # Run simulation from manifest file
    $ offarm run path/to/INPUT_manifest.json
    
    # Or with Python module
    $ python -m offarm run path/to/INPUT_manifest.json
"""

__version__ = "0.1.0"

# Lazy imports to avoid circular dependencies and improve startup time
def __getattr__(name):
    """Lazy load modules and classes on first access."""
    
    # Configuration classes
    if name in ("Env", "Current", "Wave", "Wind", "OffSys", "FloatBody", 
                "ElSys", "LineType", "LineTypes", "SimuConfig", "VarSys",
                "InputManifest", "load_json", "load_env_config", "load_line_types",
                "load_simu_config", "load_manifest", "load_var_sys"):
        from . import params
        return getattr(params, name)
    
    # Force classes
    if name in ("DragForceNB_OT", "CurrentForceNB", "QSmoorForce"):
        from . import Forces
        return getattr(Forces, name)
    
    # Simulation
    if name in ("run_a_simulation", "SimulationResults", "Configs", 
                "build_state_space", "load_diy_configs"):
        from . import Simu
        return getattr(Simu, name)
    
    # Cable mechanics
    if name in ("Cable", "FQSmoor"):
        from . import Cable
        return getattr(Cable, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Define what gets exported with "from offarm import *"
__all__ = [
    # Version
    "__version__",
    
    # Main simulation function
    "run_a_simulation",
    "SimulationResults",
    "Configs",
    
    # Configuration classes
    "Env",
    "Current",
    "Wave",
    "OffSys",
    "FloatBody",
    "ElSys",
    "LineType",
    "LineTypes",
    "SimuConfig",
    "VarSys",
    "InputManifest",
    
    # Force calculators
    "DragForceNB_OT",
    "CurrentForceNB",
    "QSmoorForce",
    
    # Cable mechanics
    "Cable",
    "FQSmoor",
    
    # Loaders
    "load_json",
    "load_env_config",
    "load_line_types",
    "load_simu_config",
    "load_manifest",
    "load_var_sys",
]
