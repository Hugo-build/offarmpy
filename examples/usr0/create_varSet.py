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
from pprint import pprint
import numpy as np


# Add project root to path for pySMC imports
script_dir = Path(__file__).parent
project_root = script_dir.parent  # MC_fishCageArray (contains pySMC)
sys.path.insert(0, str(project_root))

from pySMC import Variable, VariableSet, inject_single_config


# =============================================================================
# Method 1: Create VariableSet programmatically
# =============================================================================


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
var_hs.add_target(path="env.wave.Hs", doc="configs")
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
var_tp.add_target(path="env.wave.Tp", doc="configs")
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
var_uc.add_target(path="env.current.vel[0]", doc="configs")
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
var_dir_wave.add_target(path="env.wave.propDir", doc="configs")
variables.append(var_dir_wave)

# Variable 5: Current direction
var_dir_cur = Variable(
    name="dirCur",
    kind="uniform",
    description="Current direction",
    params={"low": 0.0, "high": 359.0},
    default=0.0,
)
var_dir_cur.add_target(path="env.current.propDir", doc="configs")
variables.append(var_dir_cur)

# Create VariableSet
var_set = VariableSet(variables=variables)

# Print summary
print(f"\nCreated VariableSet with {len(var_set.variables)} variables:")

pprint(var_set.to_dict())
json.dump(var_set.to_dict(), open("varSet_envOnly.json", "w"), indent=4)



# =============================================================================
# Simple manifest loader - returns raw dicts for injection
# =============================================================================

def load_configs_from_manifest(manifest_path: Path | str) -> dict:
    """
    Load all config files specified in manifest as raw dicts.
    
    Files are loaded from the same directory as the manifest.
    Returns dict with keys matching manifest file keys:
        configs["env"], configs["lineType"], configs["lineSys"], etc.
    """
    manifest_path = Path(manifest_path)
    base_dir = manifest_path.parent  # Files are in same dir as manifest
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    configs = {}
    
    for key, filename in manifest.get("files", {}).items():
        file_path = base_dir / filename
        if file_path.exists():
            with open(file_path, "r") as f:
                configs[key] = json.load(f)
            print(f"  Loaded {key}: {filename}")
        else:
            configs[key] = None
            print(f"  Missing {key}: {filename}")
    
    return configs


# =============================================================================
# TEST: Load configs from manifest and inject values
# =============================================================================

print("\n" + "=" * 60)
print("Loading configs from manifest")
print("=" * 60)

manifest_path = script_dir / "INPUT_manifest.json"
configs = load_configs_from_manifest(manifest_path)

# Show what's loaded
print(f"\nLoaded configs: {list(configs.keys())}")



# Variable names in order: [Hs, Tp, UC, dirWave, dirCur]
var_names = ["Hs", "Tp", "UC", "dirWave", "dirCur"]

# Define test cases: each row is [Hs, Tp, UC, dirWave, dirCur]
caseList = [
    [5.0, 11.0, 0.5, 0.0, 0.0],      # Base case
    [3.0, 9.0, 0.3, 45.0, 45.0],     # Moderate conditions
    [7.0, 15.0, 0.8, 90.0, 180.0],   # Severe conditions
]

print("\n" + "=" * 60)
print("Injecting cases into env config")
print("=" * 60)

for i, case in enumerate(caseList):
    print("\n" + "-" * 60)
    print(f"Injecting case {i+1} of {len(caseList)}: {case}\n")
    
    # Build values dict from case list
    values = dict(zip(var_names, case))
    
    # Inject into env config (creates a new copy)
    injected_configs = inject_single_config(configs, var_set.variables, values)


print("\n✓ All cases injected successfully!")







