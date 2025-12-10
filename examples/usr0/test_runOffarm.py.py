import sys
import os
from pprint import pprint
from pathlib import Path


sys.path.append(str(Path(__file__).parent))         # Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))  # Add the grandparent directory to the path

from offarmpy.src.Simu import run_a_simulation


if __name__ == "__main__":
    """
    Simple demonstration of run_a_simulation function.
    
    This mirrors MATLAB's simOffarm.m workflow in a single function call.
    """
    
    
    # Path to manifest file

    script_dir = Path(os.getcwd())
    usr_path = script_dir/ "usr0"
    manifest_path = usr_path / "INPUT_manifest.json"
    
    # Run simulation with a single function call
    results = run_a_simulation(
        manifest=manifest_path,  # Can be path OR dict with pre-injected configs
        info_string="Test case: 6-cage system with mooring and drag loads (wave and ocean current)",
        results_dir="results",
        save_results=True,
        show_progress=True,
        plot_results=True
    )
    
    # Access results
    print("\n" + "=" * 70)
    print(" " * 15 + "RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Success: {results.success}")
    print(f"  Status: {results.status} (1=success, -1=failed, 0=not run)")
    print(f"  Elapsed time: {results.elapsed_time:.2f}s")
    print(f"  RHS evaluations: {results.n_rhs_calls}")
    print(f"  Time steps: {results.n_time_steps}")
    print(f"  Max displacement: {results.max_displacement:.4f} m")

    print(f"\n  Saved files:")
    print(f"    Displacement: {results.displacement_file}")
    print(f"    Velocity: {results.velocity_file}")
    print(f"    Summary: {results.summary_file}")