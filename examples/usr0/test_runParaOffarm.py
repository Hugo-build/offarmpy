"""
_____________________________________________________________________________

Test script demonstrating parallel simulation execution with 
config injection.

This script shows how to use ParallelRunner to safely run multiple 
simulations in parallel by pre-injecting variable values into configs 
before distributing to worker processes. This avoids file I/O race 
conditions.

Key concept:
    - Configs are loaded ONCE in the main process
    - Values are injected into COPIES of configs before 
    parallel execution
    - Workers receive Python dicts (not file paths) - no file conflicts!

_____________________________________________________________________________

Welcome to OffarmPy!
                               
   ▄▄▄▄    ▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄     ▄▄     ▄▄▄▄▄▄    ▄▄▄  ▄▄▄ 
  ██▀▀██   ██▀▀▀▀▀▀  ██▀▀▀▀▀▀    ████    ██▀▀▀▀██  ███  ███ 
 ██    ██  ██        ██          ████    ██    ██  ████████ 
 ██    ██  ███████   ███████    ██  ██   ███████   ██ ██ ██ 
 ██    ██  ██        ██         ██████   ██  ▀██▄  ██ ▀▀ ██ 
  ██▄▄██   ██        ██        ▄██  ██▄  ██    ██  ██    ██ 
   ▀▀▀▀    ▀▀        ▀▀        ▀▀    ▀▀  ▀▀    ▀▀▀ ▀▀    ▀▀ 
                                                            
_____________________________________________________________________________
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add parent paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

from pySMC.io.runner import ParallelRunner, RunnerConfig, get_n_workers
from pySMC.io.manifest import load_configs_from_manifest
from pySMC import VariableSet

from offarmpy.src.Simu import run_a_simulation


# =============================================================================
# Define simulation wrapper function for ParallelRunner
# =============================================================================

def offarm_simu_func(configs: dict, case_id: int) -> dict:
    """
    Simulation wrapper for ParallelRunner.
    
    This function receives pre-injected configs as a dict and runs the simulation.
    The configs dict contains all config data with variables already injected,
    so no file I/O race conditions can occur.
    
    Args:
        configs: Dict with keys "env", "lineSys", "lineType", "simu"
                 All values are already injected for this specific case.
        case_id: Case identifier
        
    Returns:
        Dict with simulation results
    """
    # Get workspace_path from configs metadata (injected by ParallelRunner)
    workspace_path = configs.get("_workspace_path", None)
    
    # Run simulation using the dict-based manifest
    result = run_a_simulation(
        manifest=configs,  # Pass dict instead of path - no file I/O!
        case_id=case_id,
        info_string=f"Parallel case {case_id}",
        results_dir="results",
        save_results=True,  # Don't save individual results in parallel
        show_progress=False,  # Disable progress bar in workers
        plot_results=False,
        workspace_path=workspace_path,
    )
    
    # Return results as dict for ParallelRunner
    return result.to_dict()



# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    
    # -------------------------------------------------------------------------
    # Setup paths
    # -------------------------------------------------------------------------
    script_dir = Path(os.getcwd())
    usr_path = script_dir / "usr0"
    manifest_path = usr_path / "INPUT_manifest.json"
    varSet_path = usr_path / "varSet_envOnly.json"
    samples_path = usr_path / "usr0_random_samples.csv"
    
    print("=" * 70)
    print(" " * 15 + "PARALLEL SIMULATION TEST")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Check available workers
    # -------------------------------------------------------------------------
    n_workers = get_n_workers(use_fraction=0.5)
    print(f"\nNumber of workers available: {n_workers}")
    
    # -------------------------------------------------------------------------
    # Load VariableSet
    # -------------------------------------------------------------------------
    var_set = VariableSet.from_json(str(varSet_path))
    print(f"\nLoaded VariableSet with {len(var_set.variables)} variables:")
    for v in var_set.variables:
        print(f"  - {v.name}: {v.kind} [{v.params.get('low', '?')}, {v.params.get('high', '?')}]")
    
    # -------------------------------------------------------------------------
    # Load sample cases (just a few for testing)
    # -------------------------------------------------------------------------
    # Load from CSV
    samples_data = np.loadtxt(samples_path, delimiter=',', skiprows=1)
    
    # Use only first few samples for testing
    n_test_cases = 7  # Small number for testing
    test_cases = samples_data[:n_test_cases]
    
    var_names = [v.name for v in var_set.variables]
    print(f"\nTest cases ({n_test_cases} samples):")
    for i, case in enumerate(test_cases):
        print(f"  Case {i}: {dict(zip(var_names, case))}")
    
    # -------------------------------------------------------------------------
    # Option 0: Single simulation test (for verification)
    # -------------------------------------------------------------------------
    # print("\n" + "+" * 70)
    # print(" " * 15 + "SINGLE SIMULATION TEST")
    # print("+" * 70)
    
    # # Load configs once
    # configs = load_configs_from_manifest(manifest_path)
    
    # # Inject workspace path for DIY configs
    # configs["_workspace_path"] = usr_path
    
    # # Manually inject first case values
    # from pySMC.io.manifest import inject_from_varset
    # values = dict(zip(var_names, test_cases[0]))
    # injected_configs = inject_from_varset(configs, var_set, values)
    # injected_configs["_workspace_path"] = usr_path
    
    # print(f"\nInjected values: {values}")
    
    # # Run single simulation with injected dict
    # single_result = run_a_simulation(
    #     manifest=injected_configs,
    #     case_id=0,
    #     info_string="Single test with injected configs",
    #     results_dir="results",
    #     save_results=True,
    #     show_progress=True,
    #     plot_results=False,
    #     workspace_path=usr_path,
    # )
    
    # print(f"\nSingle simulation result:")
    # print(f"  Success: {single_result.success}")
    # print(f"  Max displacement: {single_result.max_displacement:.4f} m")
    
    # -------------------------------------------------------------------------
    # Option 1: Parallel simulation with ParallelRunner
    # -------------------------------------------------------------------------
    print("\n" + "+" * 70)
    print(" " * 15 + "PARALLEL SIMULATION TEST")
    print("+" * 70)
    
    # Configure runner
    runner_config = RunnerConfig(
        n_workers=min(n_workers, n_test_cases),  # Don't use more workers than cases
        verbose=True,
        save_results=True,
        output_dir=usr_path / "results",
        output_prefix="parallel_test",
    )
    
    # Create ParallelRunner
    # Note: workspace_path is automatically injected from manifest_path.parent
    runner = ParallelRunner(
        manifest_path=manifest_path,
        var_set=var_set,
        simu_func=offarm_simu_func,
        config=runner_config,
    )
    
    # Run parallel simulations
    print(f"\nRunning {n_test_cases} cases with {runner_config.n_workers} workers...")
    results = runner.run(
        cases=test_cases,
        var_names=var_names,
    )
    
    # ----------------
    # Display results
    # ----------------
    print("\n" + "+" * 70)
    print(" " * 15 + "RESULTS SUMMARY")
    print("+" * 70)
    
    n_success = sum(1 for r in results if r.get("success", False))
    print(f"\nTotal cases: {len(results)}")
    print(f"Successful:  {n_success}")
    print(f"Failed:      {len(results) - n_success}")
    
    print("\nIndividual results:")
    for i, r in enumerate(results):
        if r.get("success", False):
            data = r.get("data", r)
            print(f"  Case {i}: max_disp={data.get('max_displacement', 'N/A'):.4f}m, "
                  f"time={data.get('elapsed_time', 0):.2f}s")
        else:
            print(f"  Case {i}: FAILED - {r.get('error', 'Unknown error')}")
