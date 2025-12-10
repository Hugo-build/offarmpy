"""Main simulation runner for OffarmLab."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

# Import configuration classes from params module
# Support both package import and direct script execution
try:
    from .params import (
        Env, OffSys, ElSys, SimuConfig, LineTypes, FloatBody,
        load_json, load_env_config, load_line_types, load_simu_config, 
        load_manifest, load_var_sys, load_offsys,
    )
    from .Forces import (
        DragForceNB_OT, CurrentForceNB, QSmoorForce,
    )
except ImportError:
    from params import (
        Env, OffSys, ElSys, SimuConfig, LineTypes, FloatBody,
        load_json, load_env_config, load_line_types, load_simu_config, 
        load_manifest, load_var_sys, load_offsys,
    )
    from Forces import (
        DragForceNB_OT, CurrentForceNB, QSmoorForce,
    )


#=============================================================================
#                           DIY Config Function
#=============================================================================

def load_diy_configs(
    workspace_path: Path,
    configs: "Configs",
    offsys: OffSys = None,
    line_types: LineTypes = None,
    env: Env = None,
) -> tuple[OffSys, LineTypes, Env]:
    """
    Load and run user-defined configuration function if it exists.
    
    Mirrors the MATLAB pattern in simOffarm.m that checks for and runs
    myFuncDIYconfigs.m. This allows users to customize configurations
    at runtime (e.g., setting anchor positions, line types, etc.)
    
    The DIY config file should be named 'diy_configs.py' and placed in
    the workspace directory. It should define a function:
    
        def diy_configs(configs, offsys, line_types, env):
            # Modify configurations as needed
            return offsys, line_types, env
    
    Args:
        workspace_path: Path to workspace directory containing diy_configs.py
        configs: Loaded Configs object (for reference)
        offsys: Offshore system to modify
        line_types: Line types to modify  
        env: Environment to modify
        
    Returns:
        Tuple of (offsys, line_types, env) - modified or original
    """
    diy_file = workspace_path / "diy_configs.py"
    
    # Use passed values or get from configs
    offsys = offsys or configs.offsys
    line_types = line_types or configs.line_types
    env = env or configs.env
    
    if not diy_file.exists():
        print("The function for <DIY configs within SIMU> does not exist.")
        return offsys, line_types, env
    
    try:
        print("The function for <DIY configs within SIMU> exists.")
        print(f"  Loading: {diy_file}")
        
        # Dynamically load the module
        spec = importlib.util.spec_from_file_location("diy_configs", diy_file)
        diy_module = importlib.util.module_from_spec(spec)
        
        # Add workspace to path temporarily so imports work
        sys.path.insert(0, str(workspace_path))
        try:
            spec.loader.exec_module(diy_module)
        finally:
            sys.path.pop(0)
        
        # Check if the diy_configs function exists
        if hasattr(diy_module, 'diy_configs'):
            offsys, line_types, env = diy_module.diy_configs(
                configs, offsys, line_types, env
            )
            print("The function for <DIY configs within SIMU> gets finished.")
        else:
            print("  WARNING: diy_configs.py exists but has no 'diy_configs' function")
            
    except Exception as e:
        print(f"  ERROR loading diy_configs.py: {e}")
        import traceback
        traceback.print_exc()
    
    return offsys, line_types, env


@dataclass
class Configs:
    """Container for all loaded configurations."""
    env: Env = None
    offsys: OffSys = None
    line_types: LineTypes = None
    simu_config: SimuConfig = None
    var_sys: Any = None


def load_configs_from_manifest(
    manifest_path: Path | str,
    base_path: Path | str = None,
) -> Configs:
    """
    Load all configurations from an INPUT_manifest.json file.
    
    The manifest specifies paths to individual config files:
        {
            "workspace_path": "usr0",
            "files": {
                "env": "env_withoutVar.json",
                "lineType": "lineTypes.json",
                "lineSys": "sys6cage.json",
                "simu": "simu_config.json",
                "varSys": "varSys.json"
            }
        }
    
    Args:
        manifest_path: Path to the INPUT_manifest.json file
        base_path: Optional base path for resolving relative paths.
                   If None, uses the manifest file's parent directory.
    
    Returns:
        Configs dataclass with all loaded configurations
    
    Example:
        >>> configs = load_configs_from_manifest("usr0/INPUT_manifest.json")
        >>> print(configs.env.wave.Hs)
    """
    manifest_path = Path(manifest_path)
    
    # Determine base path for resolving file references
    if base_path is None:
        base_path = manifest_path.parent
    else:
        base_path = Path(base_path)
    
    # Load the manifest
    manifest = load_manifest(manifest_path)
    
    # Resolve workspace path relative to base_path
    workspace = base_path / manifest.workspace_path
    
    # Initialize config holders
    env = None
    offsys = None
    line_types = None
    simu_config = None
    var_sys = None
    
    # Load each config file if specified in manifest
    files = manifest.files
    
    # Environment config
    if "env" in files:
        env_path = workspace / files["env"]
        if env_path.exists():
            env = load_env_config(env_path)
            print(f"  Loaded env: {env_path.name}")
    
    # Offshore system (lineSys)
    if "lineSys" in files:
        sys_path = workspace / files["lineSys"]
        if sys_path.exists():
            offsys = load_offsys(sys_path)
            print(f"  Loaded offsys: {sys_path.name}")
    
    # Line types
    if "lineType" in files:
        lt_path = workspace / files["lineType"]
        if lt_path.exists():
            line_types = load_line_types(lt_path)
            print(f"  Loaded line_types: {lt_path.name}")
    
    # Simulation config
    if "simu" in files:
        simu_path = workspace / files["simu"]
        if simu_path.exists():
            simu_config = load_simu_config(simu_path)
            print(f"  Loaded simu_config: {simu_path.name}")
    
    # Variable system
    if "varSys" in files:
        var_path = workspace / files["varSys"]
        if var_path.exists():
            var_sys = load_var_sys(var_path)
            print(f"  Loaded var_sys: {var_path.name}")
    
    return Configs(
        env=env,
        offsys=offsys,
        line_types=line_types,
        simu_config=simu_config,
        var_sys=var_sys,
    )


#=============================================================================
#                State Space System for explicit integration
#=============================================================================

def build_state_space(offsys: OffSys) -> tuple[NDArray, NDArray]:
    """
    Build state-space matrices from offshore system.
    
    Constructs the combined mass-stiffness-damping system as:
        M*x'' + C*x' + K*x = F
        
    Converted to state-space form:
        [x']    [  0       I     ] [x]  +  [ 0 ]
        [x''] = [-M^-1*K  -M^-1*C] [x'] + [M^-1] * F
        
    Args:
        offsys: Offshore system configuration
        
    Returns:
        Tuple of (A, B) state-space matrices
    """
    nbod = offsys.nbod
    ndof = offsys.nDoF
    
    # Initialize global matrices
    M = np.zeros((ndof, ndof))
    C = np.zeros((ndof, ndof))
    K = np.zeros((ndof, ndof))
    
    # Assemble from each floating body
    for ibod, body in enumerate(offsys.floatBodies):
        cal_dof = offsys.calDoF[ibod]
        dof_start = cal_dof[0] - 1  # Convert to 0-based
        dof_end = cal_dof[1]
        dof_range = slice(dof_start, dof_end)
        
        # Mass matrix: M + added mass
        M_body = np.array(body.MM) + np.array(body.MMaInf)
        M[dof_range, dof_range] = M_body
        
        # Damping matrix
        C[dof_range, dof_range] = np.array(body.BBlin)
        
        # Stiffness matrix
        K[dof_range, dof_range] = np.array(body.CClin)
    
    # Build state-space form
    # State vector: z = [x; x'] (positions and velocities)
    # dz/dt = A*z + B*u
    
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        print("Warning: Mass matrix is singular, using pseudo-inverse")
        M_inv = np.linalg.pinv(M)
    
    # State matrix A (2n x 2n)
    A = np.zeros((2 * ndof, 2 * ndof))
    A[:ndof, ndof:] = np.eye(ndof)  # Upper right: I
    A[ndof:, :ndof] = -M_inv @ K     # Lower left: -M^-1 * K
    A[ndof:, ndof:] = -M_inv @ C     # Lower right: -M^-1 * C
    
    # Input matrix B (2n x n)
    B = np.zeros((2 * ndof, ndof))
    B[ndof:, :] = M_inv  # Lower half: M^-1

    
    
    return A, B


#=============================================================================
#                           Simulation Results
#=============================================================================

def get_logo() -> str:
    return """
_____________________________________________________________

Welcome to OffarmPy!

                                                            
   ▄▄▄▄    ▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄     ▄▄     ▄▄▄▄▄▄    ▄▄▄  ▄▄▄ 
  ██▀▀██   ██▀▀▀▀▀▀  ██▀▀▀▀▀▀    ████    ██▀▀▀▀██  ███  ███ 
 ██    ██  ██        ██          ████    ██    ██  ████████ 
 ██    ██  ███████   ███████    ██  ██   ███████   ██ ██ ██ 
 ██    ██  ██        ██         ██████   ██  ▀██▄  ██ ▀▀ ██ 
  ██▄▄██   ██        ██        ▄██  ██▄  ██    ██  ██    ██ 
   ▀▀▀▀    ▀▀        ▀▀        ▀▀    ▀▀  ▀▀    ▀▀▀ ▀▀    ▀▀ 
                                                            
_____________________________________________________________                                              
"""

@dataclass
class SimulationResults:
    """
    Container for simulation results.
    
    Mirrors MATLAB Results struct from simOffarm.m for consistency.
    """
    # Time and state data
    time: NDArray[np.floating] = field(default_factory=lambda: np.array([]))
    displacement: NDArray[np.floating] = field(default_factory=lambda: np.array([]))
    velocity: NDArray[np.floating] = field(default_factory=lambda: np.array([]))
    
    # Case information
    case_id: int = 0
    case_info: dict = field(default_factory=dict)
    info_string: str = ""
    
    # Status tracking (like MATLAB caseStatus)
    success: bool = True
    status: int = 0  # 0=not run, 1=success, -1=failed (NaN detected)
    message: str = ""
    
    # File paths for saved data
    displacement_file: str = ""
    velocity_file: str = ""
    summary_file: str = ""
    
    # Timing
    timestamp: str = ""
    elapsed_time: float = 0.0
    n_rhs_calls: int = 0
    
    # Summary statistics
    max_displacement: float = 0.0
    n_time_steps: int = 0
    
    def to_dict(self) -> dict:
        """Convert results to dictionary for JSON serialization."""
        return {
            "case_id": self.case_id,
            "info_string": self.info_string,
            "success": self.success,
            "status": self.status,
            "message": self.message,
            "displacement_file": self.displacement_file,
            "velocity_file": self.velocity_file,
            "timestamp": self.timestamp,
            "elapsed_time": self.elapsed_time,
            "n_rhs_calls": self.n_rhs_calls,
            "max_displacement": self.max_displacement,
            "n_time_steps": self.n_time_steps,
            "case_info": self.case_info,
        }

#=============================================================================
#               Single Case Simulation (Manual RHS, Progress Tracking)
#=============================================================================

def run_a_simulation(
    manifest: str | Path | dict,
    case_id: int = 1,
    info_string: str = "",
    results_dir: str = "results",
    save_results: bool = True,
    show_progress: bool = True,
    rtol: float = 1e-5, # relative tolerance for ODE solver
    atol: float = 1e-8, # absolute tolerance for ODE solver
    plot_results: bool = False,
    custom_force_func: Callable = None,
    env_file: str | Path = None,  # Optional: override env file from manifest
    workspace_path: str | Path = None,  # Required when manifest is dict for saving results
) -> SimulationResults:
    """
    Run a single simulation case with progress tracking and results saving.
    
    This function mirrors MATLAB's simOffarm.m workflow:
    1. Load configurations from manifest (direct loading like test code)
    2. Run DIY configs if available
    3. Build state-space system and force calculators
    4. Run static equilibrium (if enabled)
    5. Run dynamic simulation with progress bar
    6. Save results to binary and JSON files
    
    Args:
        manifest: Either:
            - Path to INPUT_manifest.json (str or Path), OR
            - Dict of pre-loaded/injected configs with keys: "env", "lineSys", 
              "lineType", "simu". This is useful for parallel execution where
              configs are pre-injected to avoid file I/O race conditions.
        case_id: Case identifier
        info_string: Description of the case
        results_dir: Directory for saving results
        save_results: Whether to save results to files
        show_progress: Show progress bar during integration
        rtol: Relative tolerance for ODE solver
        atol: Absolute tolerance for ODE solver
        plot_results: Generate and save displacement plots
        custom_force_func: Optional custom force function f(t, x, configs)
        env_file: Optional path to override env file (only used when manifest is path)
        workspace_path: Required when manifest is dict, for DIY configs and saving results
        
    Returns:
        SimulationResults object with all simulation data
    """
    
    print(get_logo())

    import time
    from scipy.integrate import solve_ivp
    from scipy.optimize import root
    
    # Optional tqdm import
    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False
    
    # Determine if manifest is a dict (pre-injected configs) or a path
    manifest_is_dict = isinstance(manifest, dict)
    
    if manifest_is_dict:
        # When manifest is a dict, workspace_path must be provided for saving results
        if workspace_path is not None:
            workspace_path = Path(workspace_path)
            results_path = workspace_path / results_dir
        else:
            # No workspace - results will be saved to current directory
            results_path = Path(results_dir)
    else:
        # Standard path-based loading
        manifest_path = Path(manifest)
        workspace_path = manifest_path.parent
        results_path = workspace_path / results_dir
    
    # Initialize results
    results = SimulationResults(
        case_id=case_id,
        info_string=info_string,
        timestamp=datetime.now().strftime("%d-%b-%Y_%H-%M-%S"),
    )
    
    print("\n")
    print("=" * 70)
    print(" " * 15 + f"SIMULATION CASE {case_id}")
    print("=" * 70)
    if info_string:
        print(f"  {info_string}")
    print("\n")
    
    # ================================================================
    # STEP 1: Load configurations from manifest (direct loading or dict)
    # ================================================================
    print("-" * 70)
    
    if manifest_is_dict:
        # ---- Dict mode: configs already pre-loaded/injected ----
        print(" " * 20 + "< loading configs from dict: >\n")
        print(f"  Config keys: {list(manifest.keys())}")
        
        # Parse configs directly from dict using from_dict methods
        # The dict has keys: "env", "lineSys", "lineType", "simu"
        
        # ___Line types___
        if "lineType" in manifest and manifest["lineType"] is not None:
            line_types = LineTypes.from_dict(manifest["lineType"])
        else:
            raise ValueError("manifest dict must contain 'lineType' key")
        print(f"  ✓ Parsed lineTypes: {len(line_types)} types")
        
        # ___Offshore system___
        if "lineSys" in manifest and manifest["lineSys"] is not None:
            offsys = OffSys.from_dict(manifest["lineSys"])
        else:
            raise ValueError("manifest dict must contain 'lineSys' key")
        print(f"  ✓ Parsed lineSys: {offsys.nbod} bodies, {offsys.nDoF} DoF")
        
        # ___Simulation config___
        if "simu" in manifest and manifest["simu"] is not None:
            simu_config = SimuConfig.from_dict(manifest["simu"])
        else:
            raise ValueError("manifest dict must contain 'simu' key")
        print(f"  ✓ Parsed simu: static={simu_config.static_enabled}, dynamic={simu_config.dynamic_enabled}")
        
        # ___Environment___
        if "env" in manifest and manifest["env"] is not None:
            env = Env.from_dict(manifest["env"])
        else:
            raise ValueError("manifest dict must contain 'env' key")
        print(f"  ✓ Parsed env: wave Hs={env.wave.Hs}m, Tp={env.wave.Tp}s")
        print(f"               current vel={env.current.vel}, dir={env.current.propDir}°")
        
    else:
        # ---- Path mode: load from files (existing behavior) ----
        print(" " * 20 + "< loading configs from manifest file: >\n")
        print(f"  Manifest path: {manifest_path}")
        
        # ___Load manifest___
        manifest_data = load_manifest(manifest_path)
        base = workspace_path  # Use the actual workspace path
        files = manifest_data.files
        
        print(f"  Workspace: {base}")
        print(f"  Files: {list(files.keys())}")
        
        # ___Load line types___
        line_types = load_line_types(base / files["lineType"])
        print(f"  ✓ Loaded lineTypes: {len(line_types)} types")
        
        # ___Load offshore system___
        offsys_data = load_json(base / files["lineSys"])
        offsys = OffSys.from_dict(offsys_data)
        print(f"  ✓ Loaded lineSys: {offsys.nbod} bodies, {offsys.nDoF} DoF")
        
        # ___Load simulation config___
        simu_config = load_simu_config(base / files["simu"])
        print(f"  ✓ Loaded simu: static={simu_config.static_enabled}, dynamic={simu_config.dynamic_enabled}")
        
        # ___Load environment (use override if provided, otherwise from manifest)___
        if env_file is not None:
            env_path = base / env_file
        elif "env" in files:
            env_path = base / files["env"]
        else:
            # Default to env_withoutVar.json if available
            env_path = base / "env_withoutVar.json"
        
        env = load_env_config(env_path)
        print(f"  ✓ Loaded env: wave Hs={env.wave.Hs}m, Tp={env.wave.Tp}s")
        print(f"               current vel={env.current.vel}, dir={env.current.propDir}°")
    
    # Create a Configs object for DIY configs
    configs = Configs(
        env=env,
        offsys=offsys,
        line_types=line_types,
        simu_config=simu_config,
        var_sys=None,
    )
    
    # ================================================================
    # STEP 2: Run DIY configs if available
    # ================================================================
    print("\n" + "-" * 70)
    print(" " * 20 + "< checking for DIY configs: >\n")
    
    if workspace_path is not None:
        offsys, line_types, env = load_diy_configs(
            workspace_path=workspace_path,
            configs=configs,
            offsys=offsys,
            line_types=line_types,
            env=env,
        )
    else:
        print("  Skipping DIY configs (no workspace_path provided)")
    
    # ================================================================
    # STEP 3: Build state-space system and force calculators
    # ================================================================
    print("\n" + "-" * 70)
    print(" " * 20 + "< building force calculators: >\n")
    
    ndof = offsys.nDoF
    
    # Build element system
    elsys = ElSys.from_floatBodies(offsys.floatBodies)
    print(f"  ElSys: {elsys.nbod} bodies, {elsys.nNodes4nbod} nodes")
    
    # Build state-space matrices
    A, B = build_state_space(offsys)
    print(f"  State-space: A={A.shape}, B={B.shape}, ndof={ndof}")
    
    # Create mooring force calculator
    float_bodies_data = [
        {"name": fb.name, "fairleadIndex": fb.fairleadIndex,
         "AlineSlave": fb.AlineSlave, "SlineSlave": fb.SlineSlave}
        for fb in offsys.floatBodies
    ]
    anchor_line_type = getattr(offsys, 'anchorLineType', None) or [1] * offsys.nAnchorLine
    shared_line_type = getattr(offsys, 'sharedLineType', None) or [2] * (offsys.nSharedLine or 0)
    
    offsys_dict = {
        "nbod": offsys.nbod, "nDoF": offsys.nDoF, "calDoF": offsys.calDoF,
        "fairleadPos_init": offsys.fairleadPos_init.tolist() if hasattr(offsys.fairleadPos_init, 'tolist') else offsys.fairleadPos_init,
        "anchorPos_init": offsys.anchorPos_init.tolist() if hasattr(offsys.anchorPos_init, 'tolist') else offsys.anchorPos_init,
        "anchorLinePair": offsys.anchorLinePair,
        "sharedLinePair": offsys.sharedLinePair or [],
        "anchorLineType": anchor_line_type,
        "sharedLineType": shared_line_type,
    }
    
    mooring_force = QSmoorForce.from_config(
        line_sys=offsys_dict, float_bodies=float_bodies_data, line_types_config=line_types,
    )
    print(f"  ✓ Created QSmoorForce")
    
    # Create current force calculator
    current_force = CurrentForceNB(elsys, env.current)
    print(f"  ✓ Created CurrentForceNB")
    
    # Create drag force calculator (current + wave)
    drag_force = DragForceNB_OT(Elsys=elsys, current=env.current, wave=env.wave, threshold=0.5)
    print(f"  ✓ Created DragForceNB_OT (current + wave)")
    
    # Initial condition
    x0 = np.zeros(2 * ndof)
    x_static = np.zeros(ndof)
    
    # ================================================================
    # STEP 4: Static simulation (if enabled)
    # ================================================================
    if simu_config.static_enabled:
        print("\n" + "-" * 70)
        print(" " * 10 + "< STATIC SIMULATION >\n")
        
        # Compute constant current force at origin
        x0_full = np.zeros(2 * ndof)
        F_current_const = current_force(0.0, x0_full).copy()
        
        # Zero out rotational components
        nbod = ndof // 6
        for ibod in range(nbod):
            F_current_const[6*ibod + 3 : 6*ibod + 6] = 0.0
        
        print(f"  F_current_const (translational):")
        for ibod in range(min(nbod, 3)):
            print(f"    Body {ibod}: F=[{F_current_const[6*ibod]:.2e}, {F_current_const[6*ibod+1]:.2e}, {F_current_const[6*ibod+2]:.2e}] N")
        
        # Step 1: Mooring-only equilibrium
        print(f"\n  Solving for mooring-only equilibrium...")
        print(" " * 50 + ">>> PROGRAM ON <<<")
        
        def static_force_moor_only(x_pos):
            x_full = np.zeros(2 * ndof)
            x_full[:ndof] = x_pos
            return mooring_force(x_full)
        
        from scipy.optimize import fsolve
        x_static_moor = fsolve(static_force_moor_only, np.zeros(ndof))
        
        print(" " * 50 + "<<< PROGRAM OFF >>>")
        print(f"\n  Mooring-only equilibrium:")
        for ibod in range(min(nbod, 6)):
            print(f"    Body {ibod}: x=[{x_static_moor[6*ibod]:.4f}, {x_static_moor[6*ibod+1]:.4f}, {x_static_moor[6*ibod+2]:.4f}] m")
        
        # Step 2: Mooring + current equilibrium
        print(f"\n  Solving for mooring + current equilibrium...")
        print("-" * 50 + "< PROGRAM ON >")
        
        def static_force_moor_current(x_pos):
            x_full = np.zeros(2 * ndof)
            x_full[:ndof] = x_pos
            F_moor = mooring_force(x_full)
            return F_moor + F_current_const
        
        result_static = root(static_force_moor_current, x_static_moor, method='lm',
                            options={'maxiter': 5000, 'xtol': 1e-10, 'ftol': 1e-10})
        x_static = result_static.x
        
        print("-" * 50 + "< PROGRAM OFF >")
        print(f"\n  Solver success: {result_static.success}")
        
        print(f"\n  Static equilibrium (mooring + current):")
        for ibod in range(min(nbod, 6)):
            print(f"    Body {ibod}: x=[{x_static[6*ibod]:.4f}, {x_static[6*ibod+1]:.4f}, {x_static[6*ibod+2]:.4f}] m")
        
        # Store static result as initial condition
        x0[:ndof] = x_static
    
    # ================================================================
    # STEP 5: Dynamic simulation (if enabled)
    # ================================================================
    if simu_config.dynamic_enabled:
        print("\n" + "-" * 70)
        print(" " * 10 + "< DYNAMIC SIMULATION >\n")
        
        # Get time settings
        dyn_cfg = simu_config.dynamic_simu
        t_start = dyn_cfg.tStart
        t_end = dyn_cfg.tEnd
        dt = dyn_cfg.dt
        
        print(f"  Time settings: t=[{t_start}, {t_end}]s, dt={dt}s")
        print(f"  Starting from static equilibrium")
        
        n_steps = int((t_end - t_start) / dt) + 1
        t_eval = np.arange(t_start, t_end + dt, dt)
        
        print(f"  Running simulation: {n_steps} output points")
        
        # Progress tracking
        class SimulationTracker:
            """Track simulation time, RHS calls, and wall-clock time for progress display."""
            def __init__(self, t_start, t_end, update_interval=0.3):
                self.t_start = t_start
                self.t_end = t_end
                self.t_current = t_start
                self.call_count = 0
                self.duration = t_end - t_start
                self.update_interval = update_interval
                self.last_update_time = time.perf_counter()
                self.start_wall_time = time.perf_counter()
                
            def update(self, t):
                self.t_current = t
                self.call_count += 1
                
            def progress(self):
                return (self.t_current - self.t_start) / self.duration * 100
            
            def elapsed_wall_time(self):
                """Return elapsed wall-clock time in seconds."""
                return time.perf_counter() - self.start_wall_time
            
            def estimated_remaining(self):
                """Estimate remaining wall-clock time based on current progress."""
                progress = self.progress()
                if progress <= 0:
                    return float('inf')
                elapsed = self.elapsed_wall_time()
                total_estimated = elapsed / (progress / 100.0)
                return total_estimated - elapsed
            
            def format_time(self, seconds):
                """Format seconds into human-readable string (HH:MM:SS or MM:SS)."""
                if seconds == float('inf') or seconds < 0:
                    return "--:--"
                hours, remainder = divmod(int(seconds), 3600)
                minutes, secs = divmod(remainder, 60)
                if hours > 0:
                    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
                else:
                    return f"{minutes:02d}:{secs:02d}"
            
            def should_update_progress(self):
                now = time.perf_counter()
                if now - self.last_update_time >= self.update_interval:
                    self.last_update_time = now
                    return True
                return False
        
        tracker = SimulationTracker(t_start, t_end, update_interval=0.3)
        
        # Define RHS function with progress tracking
        def dxdt_with_progress(t, x):
            """
            State-space dynamics: dx/dt = A*x + B*u(t,x)
            where u = F_moor + F_drag + F_custom
            """
            tracker.update(t)
            
            # Update progress bar (throttled)
            if show_progress and HAS_TQDM and tracker.should_update_progress() and hasattr(dxdt_with_progress, 'pbar'):
                pbar = dxdt_with_progress.pbar
                progress_pct = int(tracker.progress())
                elapsed = tracker.elapsed_wall_time()
                remaining = tracker.estimated_remaining()
                
                # Use update() with delta instead of setting n directly
                delta = progress_pct - pbar.n
                if delta > 0:
                    pbar.update(delta)
                pbar.set_postfix_str(
                    f"t={t:.1f}s | {tracker.call_count} calls | "
                    f"elapsed: {tracker.format_time(elapsed)} | "
                    f"ETA: {tracker.format_time(remaining)}"
                )
            
            # Compute forces
            u = mooring_force(x) + drag_force.calculate_with_cache(t, x)
            
            # Add custom force if provided
            if custom_force_func is not None:
                u += custom_force_func(t, x, configs)
            
            # State-space: dx/dt = A*x + B*u
            return A @ x + B @ u
        
        # Run integration
        print(f"\n  Integrating with RK45...")
        print(" " * 50 + ">>> PROGRAM ON <<<")
        
        t0_sim = time.perf_counter()
        
        if show_progress and HAS_TQDM:
            pbar = tqdm(
                total=100, 
                desc="Simulating", 
                unit="%",
                bar_format='{desc}: |{bar}| {percentage:.1f}% [{postfix}]',
                dynamic_ncols=True,  # Adapt to terminal width
                position=0,          # Force single line
                leave=True,          # Keep bar after completion
                mininterval=0.3,     # Minimum update interval (seconds)
            )
            pbar.set_postfix_str(f"t={t_start:.1f}s | 0 calls | elapsed: 00:00 | ETA: --:--")
            dxdt_with_progress.pbar = pbar
        
        try:
            ivp_result = solve_ivp(
                dxdt_with_progress,
                (t_start, t_end),
                x0,
                method='RK45',
                t_eval=t_eval,
                rtol=rtol, # relative tolerance for ODE solver
                atol=atol, # absolute tolerance for ODE solver
            )
        finally:
            if show_progress and HAS_TQDM:
                elapsed = tracker.elapsed_wall_time()
                # Complete the progress bar properly
                remaining = 100 - pbar.n
                if remaining > 0:
                    pbar.update(remaining)
                pbar.set_postfix_str(
                    f"t={t_end:.1f}s | {tracker.call_count} calls | "
                    f"elapsed: {tracker.format_time(elapsed)} | Done!"
                )
                pbar.close()
        
        elapsed = time.perf_counter() - t0_sim
        
        print("-" * 50 + "< PROGRAM OFF >")
        print(f"\n  ✓ Integration complete in {elapsed:.2f}s")
        print(f"    Total RHS calls: {tracker.call_count}")
        print(f"    Solver success: {ivp_result.success}")
        
        # Extract results
        t_out = ivp_result.t
        x_out = ivp_result.y.T
        
        results.time = t_out
        results.displacement = x_out[:, :ndof]
        results.velocity = x_out[:, ndof:]
        results.elapsed_time = elapsed
        results.n_rhs_calls = tracker.call_count
        results.n_time_steps = len(t_out)
        
        # =========================================================================
        # STEP 6: Check for NaN, Inf, and divergence (unreasonably large values)
        # =========================================================================
       
        MAX_REASONABLE_DISP = 1e4  # 10 km - anything larger is clearly divergence
        
        max_disp = float(np.max(np.abs(results.displacement)))
        results.max_displacement = max_disp
        
        if np.any(np.isnan(x_out)):
            results.success = False
            results.status = -1
            results.message = "Solution contains NaN values"
            print(f"  ⚠ WARNING: {results.message}")
        elif np.any(np.isinf(x_out)):
            results.success = False
            results.status = -1
            results.message = "Solution contains Inf values"
            print(f"  ⚠ WARNING: {results.message}")
        elif max_disp > MAX_REASONABLE_DISP:
            results.success = False
            results.status = -1
            results.message = f"Solution diverged (max_disp={max_disp:.2e}m exceeds {MAX_REASONABLE_DISP:.0e}m)"
            print(f"  ⚠ WARNING: {results.message}")
        else:
            results.success = True
            results.status = 1
            print(f"  ✓ Solution is stable")
            print(f"  Max displacement: {results.max_displacement:.4f} m")
        
        # Print final positions
        print(f"\n  Final positions:")
        nbod = ndof // 6
        for ibod in range(min(nbod, 6)):
            print(f"    Body {ibod}: x=[{x_out[-1, 6*ibod]:.4f}, {x_out[-1, 6*ibod+1]:.4f}, {x_out[-1, 6*ibod+2]:.4f}] m")
        
        # ================================================================
        # STEP 7: Save results (like MATLAB simOffarm.m)
        # ================================================================
        if save_results:
            print("\n" + "-" * 70)
            print(" " * 20 + "< SAVING RESULTS >\n")
            
            # Create directories
            results_path.mkdir(parents=True, exist_ok=True)
            bin_path = results_path / "bin"
            bin_path.mkdir(exist_ok=True)
            
            timestamp = results.timestamp
            
            # Helper to get relative path (or absolute if no workspace_path)
            def get_rel_path(file_path):
                if workspace_path is not None:
                    return str(file_path.relative_to(workspace_path))
                return str(file_path)
            
            # Save displacement data as binary (like MATLAB writeBinary)
            disp_data = np.column_stack([t_out, results.displacement])
            disp_file = bin_path / f"disp_{timestamp}.npy"
            np.save(disp_file, disp_data)
            results.displacement_file = get_rel_path(disp_file)
            print(f"  Saved displacement data: {results.displacement_file}")
            
            # Save velocity data
            vel_data = np.column_stack([t_out, results.velocity])
            vel_file = bin_path / f"vel_{timestamp}.npy"
            np.save(vel_file, vel_data)
            results.velocity_file = get_rel_path(vel_file)
            print(f"  Saved velocity data: {results.velocity_file}")
            
            # Save summary as JSON (like MATLAB jsonMake)
            summary_file = results_path / f"results_{timestamp}_{case_id}.json"
            with open(summary_file, "w") as f:
                json.dump(results.to_dict(), f, indent=2)
            results.summary_file = get_rel_path(summary_file)
            print(f"  Saved summary: {results.summary_file}")
        
        # =========================================================================
        # STEP 8: Plot results (optional)
        # =========================================================================
        if plot_results:
            try:
                import matplotlib.pyplot as plt
                
                print("\n" + "-" * 70)
                print(" " * 20 + "< PLOTTING RESULTS >\n")
                
                nbod = ndof // 6
                ncols = min(3, nbod)
                nrows = int(np.ceil(nbod / ncols))
                
                fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                                        sharex=True, squeeze=False)
                fig.suptitle('Dynamic Response - Surge Displacement', fontsize=14, fontweight='bold')
                
                for ibod in range(nbod):
                    row, col = divmod(ibod, ncols)
                    ax = axes[row, col]
                    dof_idx = 6 * ibod
                    ax.plot(t_out, results.displacement[:, dof_idx], 'b-', lw=1.2)
                    ax.set_ylabel('Surge [m]')
                    ax.set_title(f'Body {ibod}')
                    ax.grid(True, alpha=0.3)
                
                # Hide unused subplots
                for idx in range(nbod, nrows * ncols):
                    row, col = divmod(idx, ncols)
                    axes[row, col].set_visible(False)
                
                # Set x-label on bottom row
                for col in range(ncols):
                    if nrows > 0 and axes[-1, col].get_visible():
                        axes[-1, col].set_xlabel('Time [s]')
                
                plt.tight_layout()
                
                # Save figure
                if save_results:
                    fig_file = results_path / f"dynamic_response_{timestamp}.png"
                    plt.savefig(fig_file, dpi=150)
                    fig_rel = fig_file.relative_to(workspace_path) if workspace_path else fig_file
                    print(f"  Saved figure: {fig_rel}")
                
                plt.show()
            except ImportError:
                print("  matplotlib not available, skipping plots")
    
    print("\n" + "=" * 70)
    print(" " * 15 + "SIMULATION COMPLETE")
    print("=" * 70)
    
    return results


def write_binary(data: NDArray, filepath: str | Path) -> Path:
    """
    Save data to binary file (numpy format).
    
    Mirrors MATLAB's writeBinary function.
    
    Args:
        data: NumPy array to save
        filepath: Output file path (without extension)
        
    Returns:
        Path to saved file
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix('.npy')
    np.save(filepath, data)
    return filepath


def read_binary(filepath: str | Path) -> NDArray:
    """
    Load data from binary file (numpy format).
    
    Mirrors MATLAB's readBinFile function.
    
    Args:
        filepath: Path to binary file
        
    Returns:
        NumPy array
    """
    return np.load(filepath)


#=============================================================================
#                                Example Usage
#=============================================================================

if __name__ == "__main__":
    """
    Simple demonstration of run_a_simulation function.
    
    This mirrors MATLAB's simOffarm.m workflow in a single function call.
    """
    from pathlib import Path
    
    
    # Path to manifest file
    script_dir = Path(__file__).parent
    usr_path = script_dir.parent/ "examples/usr0"
    manifest_path = usr_path / "INPUT_manifest.json"
    
    # Run simulation with a single function call
    # Use env_withoutVar.json for testing (no variable placeholders)
    results = run_a_simulation(
        manifest=manifest_path,  # Can be path OR dict with pre-injected configs
        case_id=1,
        info_string="Test case: 6-cage system with mooring and wave loads",
        results_dir="results",
        save_results=True,
        show_progress=True,
        plot_results=True,
        env_file="env_withoutVar.json",  # Use non-variable env for direct testing
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

