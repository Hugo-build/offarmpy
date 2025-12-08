"""
Example script: Creating and using VariableSet for offshore simulations.

This script demonstrates how to use the Variables module to:
1. Create Variable objects with different distributions
2. Build a VariableSet for batch simulations
3. Define injection targets for config files
4. Sample values and inject into configurations
5. Convert to varSys.json format for backward compatibility

Usage:
    python create_varSet.py
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np

# Add project root to path for utils imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

from utils.Variables import Variable, VariableSet, inject_single_config


# =============================================================================
# Method 1: Create VariableSet programmatically
# =============================================================================

def create_varset_programmatic() -> VariableSet:
    """
    Create a VariableSet by defining variables programmatically.
    
    This approach gives full control over variable definitions and
    allows specifying injection targets for automatic config updates.
    """
    print("=" * 60)
    print("Method 1: Creating VariableSet programmatically")
    print("=" * 60)
    
    # Define variables for offshore simulation
    variables = []
    
    # Variable 1: Significant wave height (Hs)
    var_hs = Variable(
        name="Hs",
        kind="uniform",
        description="Significant wave height",
        params={"low": 0.5, "high": 7.0},
        unit="m",
        default=3.0,
    )
    # Add injection target: where in config this variable should be injected
    var_hs.add_target(doc="env", path="wave.Hs")
    variables.append(var_hs)
    
    # Variable 2: Peak wave period (Tp)
    var_tp = Variable(
        name="Tp",
        kind="uniform",
        description="Peak wave period",
        params={"low": 7.0, "high": 20.0},
        unit="s",
        default=11.0,
    )
    var_tp.add_target(doc="env", path="wave.Tp")
    variables.append(var_tp)
    
    # Variable 3: Current velocity (UC)
    var_uc = Variable(
        name="UC",
        kind="uniform",
        description="Current velocity",
        params={"low": 0.1, "high": 1.0},
        unit="m/s",
        default=0.5,
    )
    # Current velocity is in a list: current.vel[0]
    var_uc.add_target(doc="env", path="current.vel[0]")
    variables.append(var_uc)
    
    # Variable 4: Wave propagation direction
    var_dir_wave = Variable(
        name="dirWave",
        kind="uniform",
        description="Wave propagation direction",
        params={"low": 0.0, "high": 359.0},
        unit="deg",
        default=0.0,
    )
    var_dir_wave.add_target(doc="env", path="wave.propDir")
    variables.append(var_dir_wave)
    
    # Variable 5: Current direction
    var_dir_cur = Variable(
        name="dirCur",
        kind="uniform",
        description="Current direction",
        params={"low": 0.0, "high": 359.0},
        default=0.0,
    )
    var_dir_cur.add_target(doc="env", path="current.propDir")
    variables.append(var_dir_cur)
    
    # Create VariableSet
    var_set = VariableSet(variables=variables)
    
    # Print summary
    print(f"\nCreated VariableSet with {len(var_set.variables)} variables:")
    for var in var_set.variables:
        bounds = [var.params.get("low", 0), var.params.get("high", 1)]
        targets = [t["path"] for t in var.targets]
        print(f"  - {var.name}: [{bounds[0]}, {bounds[1]}] -> {targets}")
    
    return var_set


# =============================================================================
# Method 2: Load from existing varSys.json format
# =============================================================================

def load_from_varsys_json(filepath: str | Path) -> VariableSet:
    """
    Load VariableSet from existing varSys.json format.
    
    This maintains backward compatibility with the MATLAB-style
    configuration files while using the pySMC Variable system.
    
    Args:
        filepath: Path to varSys.json file
        
    Returns:
        VariableSet with variables loaded from JSON
    """
    print("\n" + "=" * 60)
    print("Method 2: Loading from varSys.json")
    print("=" * 60)
    
    with open(filepath, "r") as f:
        data = json.load(f)
    
    # Extract data from varSys.json format
    var_placeholders = data["vars"]           # ["var1", "var2", ...]
    var_names = data["varNames"]              # ["HS", "TP", ...]
    var_units = data.get("varUnits", [])      # ["m", "s", ...]
    var_descriptions = data.get("varDescription", [])
    bounds = data["bounds"]                   # [[0.5, 7], [7, 20], ...]
    
    # Mapping from placeholder to config path (if defined)
    # This maps var1, var2, etc. to their locations in env_withVar.json
    placeholder_to_path = {
        "var1": "wave.Hs",
        "var2": "wave.Tp",
        "var3": "current.vel[0]",
        "var4": "wave.propDir",
        "var5": "current.propDir",
    }
    
    variables = []
    for i, (placeholder, name, bound) in enumerate(zip(var_placeholders, var_names, bounds)):
        var = Variable(
            name=name,  # Use descriptive name (HS, TP, etc.)
            kind="uniform",
            params={"low": bound[0], "high": bound[1]},
        )
        
        # Add target if mapping exists
        if placeholder in placeholder_to_path:
            var.add_target(doc="env", path=placeholder_to_path[placeholder])
        
        # Store additional metadata
        if i < len(var_units):
            var.unit = var_units[i]
        if i < len(var_descriptions):
            var.description = var_descriptions[i]
        
        variables.append(var)
    
    var_set = VariableSet(variables=variables)
    
    print(f"\nLoaded from: {filepath}")
    print(f"Number of variables: {len(var_set.variables)}")
    for var in var_set.variables:
        unit = getattr(var, 'unit', '')
        desc = getattr(var, 'description', '')
        print(f"  - {var.name} [{unit}]: {desc}")
    
    return var_set


# =============================================================================
# Method 3: Export VariableSet to varSys.json format
# =============================================================================

def export_to_varsys_json(var_set: VariableSet, filepath: str | Path) -> None:
    """
    Export VariableSet to varSys.json format for backward compatibility.
    
    Args:
        var_set: VariableSet to export
        filepath: Output path for JSON file
    """
    print("\n" + "=" * 60)
    print("Method 3: Exporting to varSys.json format")
    print("=" * 60)
    
    data = {
        "vars": [f"var{i+1}" for i in range(len(var_set.variables))],
        "varNames": [v.name for v in var_set.variables],
        "varUnits": [getattr(v, 'unit', '') for v in var_set.variables],
        "varDescription": [getattr(v, 'description', '') for v in var_set.variables],
        "bounds": [[v.params.get("low", 0), v.params.get("high", 1)] 
                   for v in var_set.variables],
        "path_in_config": [v.targets[0]["path"] if v.targets else "" 
                          for v in var_set.variables],
        "exprs": [],
        "method": "random",
    }
    
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported to: {filepath}")


# =============================================================================
# Demonstration: Sampling and Config Injection
# =============================================================================

def demonstrate_sampling_and_injection(var_set: VariableSet) -> None:
    """
    Demonstrate how to sample variables and inject into configs.
    """
    print("\n" + "=" * 60)
    print("Demonstration: Sampling and Config Injection")
    print("=" * 60)
    
    # Create a sample environment config (simplified)
    env_config = {
        "wave": {
            "specType": "Jonswap",
            "Hs": None,  # Will be injected
            "Tp": None,  # Will be injected
            "gamma": 3.3,
            "propDir": None,  # Will be injected
        },
        "current": {
            "vel": [None, 0],  # vel[0] will be injected
            "zlevel": [0.0, -30.0],
            "propDir": None,  # Will be injected
        },
    }
    
    # Create RNG for reproducibility
    rng = np.random.default_rng(seed=42)
    
    # Sample 5 configurations
    n_samples = 5
    configs = {"env": env_config}
    
    sampled_configs, samples = var_set.sample_configs(configs, n_samples, rng)
    
    print(f"\nGenerated {n_samples} samples:")
    print(f"  Sample matrix shape: {samples.shape}")
    print(f"\nVariable bounds:")
    lows, highs = var_set.bounds()
    for i, var in enumerate(var_set.variables):
        print(f"  {var.name}: [{lows[i]:.2f}, {highs[i]:.2f}]")
    
    print(f"\nSample values:")
    header = "  " + " | ".join([f"{v.name:>8}" for v in var_set.variables])
    print(header)
    print("  " + "-" * len(header))
    for i, sample in enumerate(samples):
        row = " | ".join([f"{val:8.3f}" for val in sample])
        print(f"  {row}")
    
    # Show one injected config
    print(f"\nExample injected config (sample 0):")
    injected_env = sampled_configs[0]["env"]
    print(f"  wave.Hs = {injected_env['wave']['Hs']:.3f}")
    print(f"  wave.Tp = {injected_env['wave']['Tp']:.3f}")
    print(f"  wave.propDir = {injected_env['wave']['propDir']:.3f}")
    print(f"  current.vel[0] = {injected_env['current']['vel'][0]:.3f}")
    print(f"  current.propDir = {injected_env['current']['propDir']:.3f}")


# =============================================================================
# SALib Integration
# =============================================================================

def demonstrate_salib_integration(var_set: VariableSet) -> None:
    """
    Demonstrate conversion to SALib format for sensitivity analysis.
    """
    print("\n" + "=" * 60)
    print("SALib Integration")
    print("=" * 60)
    
    problem = var_set.to_SAlib()
    
    print("\nSALib problem definition:")
    print(f"  num_vars: {problem['num_vars']}")
    print(f"  names: {problem['names']}")
    print(f"  bounds: {problem['bounds']}")
    
    print("\nThis can be used with SALib sampling methods:")
    print("  from SALib.sample import saltelli")
    print("  samples = saltelli.sample(problem, N=1024)")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Method 1: Create programmatically
    var_set_1 = create_varset_programmatic()
    
    # Method 2: Load from existing varSys.json
    varsys_path = script_dir / "varSys.json"
    if varsys_path.exists():
        var_set_2 = load_from_varsys_json(varsys_path)
    
    # Method 3: Export back to JSON format
    export_path = script_dir / "varSys_exported.json"
    export_to_varsys_json(var_set_1, export_path)
    
    # Demonstrate sampling and injection
    demonstrate_sampling_and_injection(var_set_1)
    
    # Show SALib integration
    demonstrate_salib_integration(var_set_1)
    
    print("\n" + "=" * 60)
    print("Script completed successfully!")
    print("=" * 60)
