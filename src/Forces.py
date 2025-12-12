
#=============================================================================
#                               Import necessary modules
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable
import json
import warnings
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve
from scipy.interpolate import interp1d

# Optional numba import for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Import dataclasses from params module
# Support both package import and direct script execution
try:
    from .params import (
        Current, Wave, Env, FloatBody, OffSys, ElSys,
        LineType, LineTypes, SimuConfig, VarSys,
        load_json, load_env_config, load_line_types, load_simu_config,
    )
    from .Cable import FQSmoor
except ImportError:
    from params import (
        Current, Wave, Env, FloatBody, OffSys, ElSys,
        LineType, LineTypes, SimuConfig, VarSys,
        load_json, load_env_config, load_line_types, load_simu_config,
    )
    from Cable import FQSmoor




#=============================================================================
#                               Timeit decorator
def timer(fn):
    import time
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        end = time.perf_counter()
        print(f"[TIME] {fn.__name__}: {end - start:.6e} s")
        return result
    return wrapper

def debug_call(fn):
    def wrapper(*args, **kwargs):
        print(f"[DEBUG] Calling {fn.__name__} with args={args[1:]}, kwargs={kwargs}")
        result = fn(*args, **kwargs)
        print(f"[DEBUG] Result: {result}")
        return result
    return wrapper

def check_shape(expected_shape):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            if hasattr(result, 'shape') and result.shape != expected_shape:
                raise ValueError(f"Expected shape {expected_shape}, got {result.shape}")
            return result
        return wrapper
    return decorator

def disabled_if(condition):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            if condition():
                return 0  # or np.zeros(...)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


    

#=============================================================================
#                               Helper Functions
def net_hydro_var(
    cos_theta: NDArray[np.floating], 
    Sn: float | NDArray[np.floating] = 0.162, 
    model: int = 2
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Calculate varying drag and lift coefficients for net panels.
    
    Mirrors MATLAB netHydroVar.m function.

    Args:
        cos_theta: Cosine of angle between flow and normal
        Sn: Net solidity ratio (default 0.162)
        model: Hydrodynamic model selection:
               1 = Kristiansen & Faltinsen model
               2 = Løland's screen model

    Returns:
        Tuple of (Cd, Cl) arrays
    """
    abs_cos = np.abs(cos_theta)
    sin2_theta = 1 - cos_theta**2  # sin²θ = 1 - cos²θ

    if model == 1:
        # Kristiansen & Faltinsen model (MATLAB case 1)
        Cd = 0.04 + (-0.04 + Sn - 1.24*Sn**2 + 13.7*Sn**3) * abs_cos
        Cl = (0.57*Sn - 3.54*Sn**2 + 10.1*Sn**3) * sin2_theta
    elif model == 2:
        # Løland's screen model (MATLAB case 2)
        Cd = 0.04 + (-0.04 + 0.33*Sn + 6.54*Sn**2 - 4.88*Sn**3) * abs_cos
        Cl = (-0.05*Sn + 2.3*Sn**2 - 1.76*Sn**3) * sin2_theta
    else:
        # Default: constant Cd
        Cd = Sn * np.ones_like(cos_theta)
        Cl = np.zeros_like(cos_theta)

    return Cd, Cl

def currentProfile(
    z: NDArray[np.floating] | float,
    vel: NDArray[np.floating] | list[float],
    zlevel: NDArray[np.floating] | list[float],
) -> NDArray[np.floating]:
    """
    Calculate current velocity profile based on depth using linear interpolation.
    
    Translates the MATLAB currentProfile function. Supports both single-level
    and multi-level current profiles.
    
    Args:
        z: Depth values to evaluate (negative below surface). 
           Can be scalar or array.
        vel: Current velocities at different levels [m/s].
             Shape (n_levels,) or (n_levels, 1).
        zlevel: Depth levels corresponding to velocities [m].
                Should be in descending order (surface to bottom).
                Shape (n_levels,) or (n_levels, 1).
    
    Returns:
        Current velocity at each depth point. Same shape as input z.
    
    Examples:
        >>> # Single level (simple linear decay)
        >>> z = np.array([0, -5, -10, -20])
        >>> vel = np.array([1.0, 0.5])
        >>> zlevel = np.array([0, -20])
        >>> currentProfile(z, vel, zlevel)
        array([1.0, 0.875, 0.75, 0.5])
        
        >>> # Multi-level profile
        >>> vel = np.array([1.2, 0.8, 0.3])
        >>> zlevel = np.array([0, -10, -30])
        >>> currentProfile(z, vel, zlevel)
    """
    # Handle scalar input
    scalar_input = np.isscalar(z)
    z = np.atleast_1d(np.asarray(z, dtype=float))
    
    # Ensure vel and zlevel are 1D arrays
    vel = np.atleast_1d(np.asarray(vel, dtype=float)).flatten()
    zlevel = np.atleast_1d(np.asarray(zlevel, dtype=float)).flatten()
    
    # Initialize output (zero by default - important for z > 0)
    u = np.zeros_like(z)
    
    # For nodes above water (z > 0), current velocity is zero
    # Only compute current for underwater nodes
    underwater_mask = z <= 0
    if not np.any(underwater_mask):
        # All nodes are above water
        return u[0] if scalar_input else u
    
    # Validate inputs
    if len(vel) != len(zlevel):
        raise ValueError(f"vel and zlevel must have same length, got {len(vel)} and {len(zlevel)}")
    
    n_levels = len(zlevel)
    
    if n_levels == 1:
        # Constant velocity for underwater nodes only
        u[underwater_mask] = vel[0]
    elif n_levels == 2:
        # Simple two-level linear interpolation (common case, optimized)
        # Only process underwater nodes (z <= 0)
        
        # Between surface and zlevel[0]: use surface velocity
        mask_above = underwater_mask & (z > zlevel[0])
        u[mask_above] = vel[0]
        
        # Below bottom level: use bottom velocity  
        mask_below = underwater_mask & (z <= zlevel[1])
        u[mask_below] = vel[1]
        
        # Linear interpolation in between
        mask_mid = underwater_mask & (z <= zlevel[0]) & (z > zlevel[1])
        if np.any(mask_mid):
            slope = (vel[0] - vel[1]) / (zlevel[0] - zlevel[1])
            u[mask_mid] = slope * (z[mask_mid] - zlevel[0]) + vel[0]
    else:
        # Multi-level profile: piecewise linear interpolation
        # Between surface and top zlevel
        mask_above = underwater_mask & (z > zlevel[0])
        u[mask_above] = vel[0]
        
        # Below bottom level
        mask_below = underwater_mask & (z <= zlevel[-1])
        u[mask_below] = vel[-1]
        
        # Interpolate between each pair of levels
        for i in range(n_levels - 1):
            mask = underwater_mask & (z <= zlevel[i]) & (z > zlevel[i + 1])
            if np.any(mask):
                slope = (vel[i] - vel[i + 1]) / (zlevel[i] - zlevel[i + 1])
                u[mask] = slope * (z[mask] - zlevel[i]) + vel[i]
    
    # Nodes above water (z > 0) remain zero (already initialized)
    return u[0] if scalar_input else u

def get_tension(line_type: LineType, x_distance: float, 
                extrapolate: bool = True, warn_extrap: bool = True) -> tuple[float, float]:
    """
    Get tension at fairlead for given horizontal distance.

    Mirrors the MATLAB getTension function, with optional extrapolation support.

    Args:
        line_type: Line type properties (from params.LineType)
        x_distance: Horizontal distance from fairlead to anchor
        extrapolate: If True, linearly extrapolate beyond lookup table range.
                     If False, clamp to boundary values (default: True)
        warn_extrap: If True, emit warning when extrapolating (default: True)

    Returns:
        Tuple of (horizontal_tension, vertical_tension)
    """
    # Get lookup tables from line_type (already numpy arrays from params)
    XX2anch = np.asarray(line_type.XX2anch).flatten()
    SS = np.asarray(line_type.SS).flatten()
    HH = np.asarray(line_type.HH).flatten()
    
    # Get unique sorted values for interpolation
    unique_xx = np.unique(XX2anch)
    unique_ss = np.unique(SS)
    unique_hh = np.unique(HH)

    if len(unique_xx) < 2:
        return 0.0, 0.0

    # Check if extrapolation is needed and warn
    x_min, x_max = unique_xx.min(), unique_xx.max()
    if warn_extrap and (x_distance < x_min or x_distance > x_max):
        warnings.warn(
            f"get_tension: x_distance={x_distance:.2f}m is outside lookup table range "
            f"[{x_min:.2f}, {x_max:.2f}]m. {'Extrapolating' if extrapolate else 'Clamping'}.",
            RuntimeWarning, stacklevel=2
        )

    if extrapolate:
        # Use interp1d with linear extrapolation
        interp_ss = interp1d(unique_xx, unique_ss, kind='linear', 
                             fill_value='extrapolate', bounds_error=False)
        interp_hh = interp1d(unique_xx, unique_hh, kind='linear',
                             fill_value='extrapolate', bounds_error=False)
        s_interp = float(interp_ss(x_distance))
        H0 = float(interp_hh(x_distance))
    else:
        # Use np.interp (clamps to boundary values)
        s_interp = float(np.interp(x_distance, unique_xx, unique_ss))
        H0 = float(np.interp(x_distance, unique_xx, unique_hh))

    # Update segment length at touchdown
    s = np.asarray(line_type.s).copy()
    if line_type.touchDownSeg > 0 and line_type.touchDownSeg <= len(s):
        s[line_type.touchDownSeg - 1] = s_interp

    # Calculate vertical tension (sum of weight of suspended segments)
    w = np.asarray(line_type.w)
    V0 = float(np.sum(w[: line_type.touchDownSeg] * s[: line_type.touchDownSeg]))

    return H0, V0


def get_tension_2ends(line_type: LineType, X2F_local: float,
                      extrapolate: bool = True, warn_extrap: bool = True) -> tuple[NDArray, NDArray]:
    """
    Get tensions at both ends of a shared line (fairlead-to-fairlead).

    Mirrors the MATLAB getTension2ends function for lines connecting two
    floating bodies (not anchored), with optional extrapolation support.

    Args:
        line_type: Line type properties (must have XXF2F, and 2-row HH/VV for asymmetric cases)
        X2F_local: Local horizontal span between fairleads [m]
        extrapolate: If True, linearly extrapolate beyond lookup table range.
                     If False, clamp to boundary values (default: True)
        warn_extrap: If True, emit warning when extrapolating (default: True)

    Returns:
        Tuple of (H0, V0) where:
            H0: array [H01, H02] horizontal tensions at end 1 and end 2 [N]
            V0: array [V01, V02] vertical tensions at end 1 and end 2 [N]
    """
    # Get lookup tables
    HH = np.atleast_2d(line_type.HH)
    VV = np.atleast_2d(line_type.VV)
    
    # Use XXF2F for fairlead-to-fairlead distance table
    if line_type.XXF2F is not None and len(line_type.XXF2F) > 0:
        XXF2F = np.asarray(line_type.XXF2F).flatten()
    else:
        # Fallback to XX2anch if XXF2F not available
        XXF2F = np.asarray(line_type.XX2anch).flatten()
    
    # Find valid (non-zero) indices for end 1
    kN0_1 = np.where(HH[0, :] != 0)[0]
    if len(kN0_1) == 0:
        return np.array([0.0, 0.0]), np.array([0.0, 0.0])
    
    # Get unique sorted x-values for interpolation range check
    xx_end1 = np.unique(XXF2F[kN0_1])
    x_min, x_max = xx_end1.min(), xx_end1.max()
    
    # Check if extrapolation is needed and warn
    if warn_extrap and (X2F_local < x_min or X2F_local > x_max):
        warnings.warn(
            f"get_tension_2ends: X2F_local={X2F_local:.2f}m is outside lookup table range "
            f"[{x_min:.2f}, {x_max:.2f}]m. {'Extrapolating' if extrapolate else 'Clamping'}.",
            RuntimeWarning, stacklevel=2
        )
    
    # Helper function for interpolation/extrapolation
    def _interp_extrap(x_val: float, x_arr: NDArray, y_arr: NDArray) -> float:
        """Interpolate or extrapolate based on settings."""
        if extrapolate:
            if len(x_arr) < 2:
                return float(y_arr[0]) if len(y_arr) > 0 else 0.0
            interp_func = interp1d(x_arr, y_arr, kind='linear',
                                   fill_value='extrapolate', bounds_error=False)
            return float(interp_func(x_val))
        else:
            return float(np.interp(x_val, x_arr, y_arr))
    
    # Interpolate horizontal tension at end 1
    H01 = _interp_extrap(X2F_local, xx_end1, np.unique(HH[0, kN0_1]))
    
    if line_type.Z2F == 0:
        # Symmetric case: both fairleads at same depth
        # H02 = H01, V01 = V02 = half of total line weight
        H02 = H01
        w = np.atleast_1d(line_type.w).flatten()
        s = np.atleast_1d(line_type.s).flatten()
        V01 = float(np.sum(w * s) / 2.0)
        V02 = V01
    else:
        # Asymmetric case: interpolate separately for each end
        V01 = _interp_extrap(X2F_local, xx_end1, np.unique(VV[0, kN0_1]))
        
        # Find valid indices for end 2
        kN0_2 = np.where(HH[1, :] != 0)[0] if HH.shape[0] > 1 else kN0_1
        xx_end2 = np.unique(XXF2F[kN0_2])
        
        H02 = _interp_extrap(X2F_local, xx_end2,
                             np.unique(HH[1, kN0_2] if HH.shape[0] > 1 else HH[0, kN0_2]))
        V02 = _interp_extrap(X2F_local, xx_end2,
                             np.unique(VV[1, kN0_2] if VV.shape[0] > 1 else VV[0, kN0_2]))
    
    H0 = np.array([H01, H02])
    V0 = np.array([V01, V02])
    
    return H0, V0


    
#=============================================================================
#                                 Forces Library - Base Class
class BaseForce(ABC):
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
    

#=============================================================================
#                                 Simple Forces

@dataclass
class HarmonicForce(BaseForce):
    amplitude: float
    omega: float
    phase: float = 0

    def __call__(self, t, x):
        return self.amplitude * np.sin(self.omega * t + self.phase)



class pulloutForce(BaseForce):
    def __init__(self, Fcons:np.ndarray, Tramp:float):
        self.Fcons = Fcons
        self.Tramp = Tramp
    
    def __call__(self, t, x):
        if t < self.Tramp:
            return self.Fcons/self.Tramp * t
        elif t >= self.Tramp and t <= self.Tcons:
            return self.Fcons
       

class DecayForce(BaseForce):
    def __init__(self, Fcons:np.ndarray, Tcons:float, Tramp:float):
        self.Fcons = Fcons
        self.Tcons = Tcons
        self.Tramp = Tramp
    
    def __call__(self, t, x):
        if t < self.Tramp:
            return self.Fcons/self.Tramp * t
        elif t >= self.Tramp and t <= self.Tcons:
            return self.Fcons
        else:  # t > Tcons
            return 0 * self.Fcons


class convForceOT(BaseForce):
    def __init__(self):
        pass

    def __call__(self, t, x):
        pass


#=============================================================================
#                                 Current Force

class CurrentForce(BaseForce):
    def __init__(self, floatBody, current):
        self.floatBody = floatBody
        self.current = current
        self.rho = 1025
        self.g = 9.81

    def current_profile(self, z, vel, zlevel):
        """
        Calculate current velocity profile based on depth.
        
        Delegates to standalone currentProfile function for consistency.
        See currentProfile() for full documentation.
        """
        return currentProfile(z, vel, zlevel)

    def __call__(self, t, x):
        nbod = len(self.floatBody)
        ndof = len(x) // 2
        F = np.zeros(ndof)

        print(F)

        # Current direction vector
        current_dir = np.array([
            np.cos(np.deg2rad(self.current.propDir)),
            np.sin(np.deg2rad(self.current.propDir)),
            0.0
        ])

        for ibod, body in enumerate(self.floatBody):
            # Update node positions with current displacement
            # Assuming 6 DOF per body: [x, y, z, rx, ry, rz]
            displacement = x[6*ibod:6*ibod+3]
            updated_positions = body.attachNodePos_init.copy();
            updated_positions[:3,:] += displacement.reshape(3,1)

            # Calculate current velocity profile at node depths
            Uc = (self.current.wakeRatio * 
                  self.current_profile(updated_positions[2, :], 
                                     self.current.vel, 
                                     self.current.zlevel))
            #print(Uc)
            # Calculate relative velocity
            current_velocity_3d = current_dir.reshape(3, 1) * Uc.reshape(1, -1)
            body_velocity = x[ndof + 6*ibod:ndof + 6*ibod + 3].reshape(3, 1)
            Urel = current_velocity_3d - body_velocity
            #print(Urel)
            F_thisBody = np.zeros(3)

            if hasattr(body, "type2node_L") and "cylinder" in body.type2node_L:
                cyl_start = body.type2node_L["cylinder"]
                cyl_end   = body.type2node_R["cylinder"]
                cyl_nodes = np.arange(cyl_start, cyl_end + 1)

                # calculate normal velocity (perpendicular to cylinder axis)
                Urel_cyl = Urel[:, cyl_nodes]
                node_vec_cyl = body.attachNodeVec[:, cyl_nodes]

                # Remove component parallel to cylinder axis
                parallel_component = np.sum(Urel_cyl * node_vec_cyl, axis=0)
                Un = Urel_cyl - parallel_component * node_vec_cyl
                print(Un)
                # Calculate magnitude
                Un_mod = np.sqrt(np.sum(Un**2, axis=0))

                # Calculate drag force: F = 0.5 * rho * Cd * A * |U|² * U_unit
                drag_coeff = (0.5 * self.rho * 
                            body.attachNodeCd[cyl_nodes] * 
                            body.attachNodeArea[cyl_nodes])
                
                F_cyl = Un * (Un_mod * drag_coeff).reshape(1, -1)
                F_thisBody += np.sum(F_cyl, axis=1)
            
            # Handle net elements
            if hasattr(body, 'type2node_L') and 'net' in body.type2node_L:
                net_start = body.type2node_L['net']
                net_end = body.type2node_R['net']
                net_nodes = slice(net_start, net_end + 1)
                
                # For nets, use full relative velocity (no normal component calculation)
                Ud = Urel[:, net_nodes]
                Ud_mod = np.sqrt(np.sum(Ud**2, axis=0))
                
                # Calculate drag force
                drag_coeff = (0.5 * self.rho * 
                            body.attachNodeCd[net_nodes] * 
                            body.attachNodeArea[net_nodes])
                
                F_net = Ud * (Ud_mod * drag_coeff).reshape(1, -1)
                F_thisBody += np.sum(F_net, axis=1)
            
            # Assign forces to global force vector
            F[6*ibod:6*ibod+3] = F_thisBody  
        return F


#=============================================================================
#                          Current Force (N-Body, no wave)

@dataclass
class CurrentForceNB(BaseForce):
    """
    N-Body current force calculation (no wave contribution).
    
    Follows the same efficient structure as DragForceNB but only includes
    current-induced forces, with no wave kinematics.
    """
    Elsys: ElSys
    current: Current
    rho: float = 1025.0
    g: float = 9.81
    
    def current_profile(self, z, vel, zlevel):
        """
        Calculate current velocity profile based on depth.
        
        Delegates to standalone currentProfile function for consistency.
        See currentProfile() for full documentation.
        """
        return currentProfile(z, vel, zlevel)

    def netHydroVar(self, cosTheta, Sn, model=2):
        """
        Calculate varying drag and lift coefficients for net panels.
        
        Delegates to standalone net_hydro_var function for consistency.
        See net_hydro_var() for full documentation.

        cosTheta: array of cosine of angle between flow and normal
        Sn: solidity ratio
        model: screen model
        
        Returns:
            Cd: drag coefficient
            Cl: lift coefficient
        """
        return net_hydro_var(cosTheta, Sn, model)
    
    def compute_relative_velocity(self, x):
        """
        Compute relative velocity between current and structure at all nodes.
        
        Note: No time parameter needed since current is steady-state.
        
        Args:
            x: State vector [positions; velocities]
            
        Returns:
            Urel: Relative velocity at each node (3, nNodes4nbod)
            Ufluid: Fluid velocity at each node (3, nNodes4nbod)
            attachNodePos: Updated node positions (3, nNodes4nbod)
        """
        ndof = len(x) // 2

        # Update all attachNodes' global position in the system
        attachNodePos = self.Elsys.attachNodePos_globInit.copy()
        for ibod in range(self.Elsys.nbod):
            idx_start = self.Elsys.Els2bod[ibod, 0]
            idx_end = self.Elsys.Els2bod[ibod, 1]
            dof_indices = self.Elsys.bod2DoF_Tran[ibod, :]
            attachNodePos[:3, idx_start:idx_end+1] += x[dof_indices].reshape(3, -1)

        # Calculate current velocities based on depth
        Uc = self.current_profile(attachNodePos[2, :], self.current.vel, self.current.zlevel)

        # Current direction vector (no wave contribution)
        current_dir = np.array([
            [np.cos(np.deg2rad(self.current.propDir))],
            [np.sin(np.deg2rad(self.current.propDir))],
            [0.0]
        ])

        # Fluid velocity is purely from current (no wave)
        Ufluid = current_dir @ Uc.reshape(1, -1)

        # Calculate relative velocities
        Urel = Ufluid.copy()
        for ibod in range(self.Elsys.nbod):
            idx_start = self.Elsys.Els2bod[ibod, 0]
            idx_end = self.Elsys.Els2bod[ibod, 1]
            dof_indices = self.Elsys.bod2DoF_Tran[ibod, :]
            Urel[:, idx_start:idx_end+1] = (Ufluid[:, idx_start:idx_end+1] * 
                                          self.current.wakeRatio - 
                                          x[ndof + dof_indices].reshape(3, -1))

        return Urel, Ufluid, attachNodePos

    def compute_cylinder_velocity(self, Urel):
        """
        Compute normal velocity for cylindrical elements.
        
        For cylinders, only the velocity component normal to the cylinder axis
        contributes to drag: Un = Urel - (Urel · nodeVec) * nodeVec
        
        Args:
            Urel: Relative velocity at all nodes (3, nNodes4nbod)
            
        Returns:
            Un: Normal velocity at cylinder nodes (3, nCylNodes)
            Un_mod: Magnitude of normal velocity (nCylNodes,)
        """
        idx_cyl = self.Elsys.Index_cylType - 1  # Convert to 0-based
        
        Urel_cyl = Urel[:, idx_cyl]
        nodeVec_cyl = self.Elsys.attachNodeVec[:, idx_cyl]
        dot_product = np.sum(Urel_cyl * nodeVec_cyl, axis=0, keepdims=True)
        Un = Urel_cyl - dot_product * nodeVec_cyl
        
        Un_mod = np.sqrt(np.sum(Un**2, axis=0))
        
        return Un, Un_mod

    def compute_net_velocity(self, Urel):
        """
        Compute velocity and drag coefficient for net elements.
        
        For nets, the full velocity is used, and Cd varies with angle of attack.
        
        Args:
            Urel: Relative velocity at all nodes (3, nNodes4nbod)
            
        Returns:
            Ud: Velocity at net nodes (3, nNetNodes)
            Ud_mod: Magnitude of velocity (nNetNodes,)
            Cd: Drag coefficient at each net node (nNetNodes,)
            Cl: Lift coefficient at each net node (nNetNodes,)
        """
        idx_net = self.Elsys.Index_netType - 1  # Convert to 0-based
        
        Ud = Urel[:, idx_net]
        Ud_mod = np.sqrt(np.sum(Ud**2, axis=0))
        
        # Unit direction vector: ed = Ud / |Ud|
        ed = np.divide(Ud, Ud_mod[np.newaxis, :], 
                      out=np.zeros_like(Ud), 
                      where=Ud_mod[np.newaxis, :] != 0)

        # cosTheta = dot(ed, nodeVec) for angle of attack
        cosTheta = np.sum(ed * self.Elsys.attachNodeVec[:, idx_net], axis=0)
        
        # Use specific Sn if available, otherwise default
        if self.Elsys.attachNodeSn is not None:
            Sn = self.Elsys.attachNodeSn[idx_net]
        else:
            Sn = 0.162

        Cd, Cl = self.netHydroVar(cosTheta, Sn, 1)
        
        return Ud, Ud_mod, Cd, Cl

    def assemble_velocities(self, Un, Un_mod, Ud, Ud_mod):
        """
        Assemble cylinder and net velocities into full arrays.
        
        Args:
            Un: Normal velocity at cylinder nodes (3, nCylNodes)
            Un_mod: Magnitude at cylinder nodes (nCylNodes,)
            Ud: Velocity at net nodes (3, nNetNodes)
            Ud_mod: Magnitude at net nodes (nNetNodes,)
            
        Returns:
            Ucal: Assembled velocity array (3, nNodes4nbod)
            Ucal_mod: Assembled magnitude array (nNodes4nbod,)
        """
        idx_cyl = self.Elsys.Index_cylType - 1
        idx_net = self.Elsys.Index_netType - 1
        
        Ucal = np.zeros((3, self.Elsys.nNodes4nbod))
        Ucal_mod = np.zeros(self.Elsys.nNodes4nbod)

        Ucal[:, idx_cyl] = Un
        Ucal_mod[idx_cyl] = Un_mod
        Ucal[:, idx_net] = Ud
        Ucal_mod[idx_net] = Ud_mod
        
        return Ucal, Ucal_mod

    def compute_morison_forces(self, Ucal, Ucal_mod, Cd_net=None):
        """
        Compute Morison drag forces and moments from assembled velocities.
        
        F = 0.5 * rho * Cd * A * |U| * U
        M = cross(r_local, F)
        
        Args:
            Ucal: Velocity at all nodes (3, nNodes4nbod)
            Ucal_mod: Velocity magnitude at all nodes (nNodes4nbod,)
            Cd_net: Optional drag coefficients for net elements (nNetNodes,)
                   If provided, updates attachNodeCd for net elements.
            
        Returns:
            F_allEls: Force on each element node (3, nNodes4nbod)
            M_allEls: Moment on each element node (3, nNodes4nbod)
        """
        # Update Cd for net elements if provided
        if Cd_net is not None:
            idx_net = self.Elsys.Index_netType - 1
            self.Elsys.attachNodeCd[idx_net] = Cd_net

        UUmod = Ucal_mod * Ucal
        halvRhoCddA = 0.5 * self.rho * self.Elsys.attachNodeCd * self.Elsys.attachNodeArea
        F_allEls = UUmod * halvRhoCddA
        M_allEls = np.cross(self.Elsys.attachNodePos_loc, UUmod, axis=0) * halvRhoCddA

        return F_allEls, M_allEls

    def compute_element_forces(self, x):
        """
        Compute forces and moments for all elements (current only).
        
        Note: No time parameter needed since current is steady-state.
        
        Args:
            x: State vector [positions; velocities]
            
        Returns:
            F_allEls: Force on each element node (3, nNodes4nbod)
            M_allEls: Moment on each element node (3, nNodes4nbod)
        """
        # Step 1: Compute relative velocity (no time dependency)
        Urel, Ufluid, attachNodePos = self.compute_relative_velocity(x)
        
        # Step 2: Compute velocity components for each element type
        Un, Un_mod = self.compute_cylinder_velocity(Urel)
        Ud, Ud_mod, Cd, Cl = self.compute_net_velocity(Urel)
        
        # Step 3: Assemble into full arrays
        Ucal, Ucal_mod = self.assemble_velocities(Un, Un_mod, Ud, Ud_mod)
        
        # Step 4: Compute Morison forces
        F_allEls, M_allEls = self.compute_morison_forces(Ucal, Ucal_mod, Cd_net=Cd)

        return F_allEls, M_allEls

    def __call__(self, t, x):
        """
        Calculate total forces and moments for each body (current only).
        
        Note: Time parameter is accepted for interface compatibility but not used,
        since current is steady-state.
        
        Args:
            t: Time [s] (unused, for interface compatibility)
            x: State vector [positions; velocities]
            
        Returns:
            F: Force vector (ndof,) with forces and moments for each body
        """
        ndof = len(x) // 2
        F = np.zeros(ndof)

        # Compute element-level forces and moments
        F_allEls, M_allEls = self.compute_element_forces(x)

        # Sum forces for each body
        for ibod in range(self.Elsys.nbod):
            idx_start = self.Elsys.Els2bod[ibod, 0]
            idx_end = self.Elsys.Els2bod[ibod, 1]
            
            F_indices = self.Elsys.DoF_Tran[3*ibod:3*(ibod+1)]
            M_indices = self.Elsys.DoF_Rot[3*ibod:3*(ibod+1)]
          
            F[F_indices] = np.sum(F_allEls[:, idx_start:idx_end+1], axis=1)
            F[M_indices] = np.sum(M_allEls[:, idx_start:idx_end+1], axis=1)

        return F



#=============================================================================
#                                 Drag Force (N-Body)

@dataclass
class DragForceNB(BaseForce):
    """
    N-Body drag force calculation for multiple ocean structures.
    """
    Elsys: ElSys
    current: Current
    wave: Wave
    rho: float = 1025.0
    g: float = 9.81
    
    def current_profile(self, z, vel, zlevel):
        """
        Calculate current velocity profile based on depth.
        
        Delegates to standalone currentProfile function for consistency.
        See currentProfile() for full documentation.
        """
        return currentProfile(z, vel, zlevel)

    def netHydroVar(self, cosTheta, Sn, model=1):
        """
        Calculate varying drag and lift coefficients for net panels.
        
        Delegates to standalone net_hydro_var function for consistency.
        See net_hydro_var() for full documentation.
        """
        return net_hydro_var(cosTheta, Sn, model)
    
    def compute_relative_velocity(self, t, x):
        """
        Compute relative velocity between fluid and structure at all nodes.
        
        Args:
            t: Time [s]
            x: State vector [positions; velocities]
            
        Returns:
            Urel: Relative velocity at each node (3, nNodes4nbod)
            Ufluid: Fluid velocity at each node (3, nNodes4nbod)
            attachNodePos: Updated node positions (3, nNodes4nbod)
        """
        ndof = len(x) // 2

        # Update all attachNodes' global position in the system
        attachNodePos = self.Elsys.attachNodePos_globInit.copy()
        for ibod in range(self.Elsys.nbod):
            idx_start = self.Elsys.Els2bod[ibod, 0]
            idx_end = self.Elsys.Els2bod[ibod, 1]
            dof_indices = self.Elsys.bod2DoF_Tran[ibod, :]
            attachNodePos[:3, idx_start:idx_end+1] += x[dof_indices].reshape(3, -1)

        # Calculate current velocities
        Uc = self.current_profile(attachNodePos[2, :], self.current.vel, self.current.zlevel)
 
        # Calculate wave-induced velocities
        # omegaZa shape: (1, nWaveComponents) for matrix multiplication later
        omegaZa = (self.wave.omegaCal * self.wave.ZaCal).T
        
        # kX*x + kY*y for each wave component and each node
        kXx_plus_kYy = (self.wave.kXCal @ attachNodePos[0:1, :] + 
                        self.wave.kYCal @ attachNodePos[1:2, :])
        
        # Phase argument: -omega*t + (kX*x + kY*y) + phase
        omegat_Kxy_phase = (-self.wave.omegaCal * t + kXx_plus_kYy + self.wave.phaseCal)
        
        # Depth attenuation factor: eKz = exp(k * z)
        # For z > 0 (above water), cap z at 0 to avoid exponential blowup
        # This uses surface wave kinematics for any nodes above water
        z_capped = np.minimum(attachNodePos[2:3, :], 0.0)
        eKz = np.exp(self.wave.k.reshape(-1, 1) * z_capped)
        
        Uwave_XY = omegaZa @ (np.sin(omegat_Kxy_phase) * eKz)
        Uwave_Z = omegaZa @ (np.cos(omegat_Kxy_phase) * eKz)

        # Calculate fluid velocities
        current_dir = np.array([
            [np.cos(np.deg2rad(self.current.propDir))],
            [np.sin(np.deg2rad(self.current.propDir))],
            [0]
        ])
        wave_dir = np.array([
            [np.cos(np.deg2rad(self.wave.propDir))],
            [np.sin(np.deg2rad(self.wave.propDir))],
            [0]
        ])

        Ufluid = (current_dir @ Uc.reshape(1, -1) + 
                 np.vstack([wave_dir[:2] @ Uwave_XY, Uwave_Z]))

        # Calculate relative velocities
        Urel = Ufluid.copy()
        for ibod in range(self.Elsys.nbod):
            idx_start = self.Elsys.Els2bod[ibod, 0]
            idx_end = self.Elsys.Els2bod[ibod, 1]
            dof_indices = self.Elsys.bod2DoF_Tran[ibod, :]
            Urel[:, idx_start:idx_end+1] = (Ufluid[:, idx_start:idx_end+1] * 
                                          self.current.wakeRatio - 
                                          x[ndof + dof_indices].reshape(3, -1))

        return Urel, Ufluid, attachNodePos

    def compute_cylinder_velocity(self, Urel):
        """
        Compute normal velocity for cylindrical elements.
        
        For cylinders, only the velocity component normal to the cylinder axis
        contributes to drag: Un = Urel - (Urel · nodeVec) * nodeVec
        
        Args:
            Urel: Relative velocity at all nodes (3, nNodes4nbod)
            
        Returns:
            Un: Normal velocity at cylinder nodes (3, nCylNodes)
            Un_mod: Magnitude of normal velocity (nCylNodes,)
        """
        idx_cyl = self.Elsys.Index_cylType - 1  # Convert to 0-based
        
        Urel_cyl = Urel[:, idx_cyl]
        nodeVec_cyl = self.Elsys.attachNodeVec[:, idx_cyl]
        dot_product = np.sum(Urel_cyl * nodeVec_cyl, axis=0, keepdims=True)
        Un = Urel_cyl - dot_product * nodeVec_cyl
        
        Un_mod = np.sqrt(np.sum(Un**2, axis=0))
        
        return Un, Un_mod

    def compute_net_velocity(self, Urel):
        """
        Compute velocity and drag coefficient for net elements.
        
        For nets, the full velocity is used, and Cd varies with angle of attack.
        
        Args:
            Urel: Relative velocity at all nodes (3, nNodes4nbod)
            
        Returns:
            Ud: Velocity at net nodes (3, nNetNodes)
            Ud_mod: Magnitude of velocity (nNetNodes,)
            Cd: Drag coefficient at each net node (nNetNodes,)
            Cl: Lift coefficient at each net node (nNetNodes,)
        """
        idx_net = self.Elsys.Index_netType - 1  # Convert to 0-based
        
        Ud = Urel[:, idx_net]
        Ud_mod = np.sqrt(np.sum(Ud**2, axis=0))
        
        # Unit direction vector: ed = Ud / |Ud|
        ed = np.divide(Ud, Ud_mod[np.newaxis, :], 
                      out=np.zeros_like(Ud), 
                      where=Ud_mod[np.newaxis, :] != 0)

        # cosTheta = dot(ed, nodeVec) for angle of attack
        cosTheta = np.sum(ed * self.Elsys.attachNodeVec[:, idx_net], axis=0)
        
        # Use specific Sn if available, otherwise default
        if self.Elsys.attachNodeSn is not None:
            Sn = self.Elsys.attachNodeSn[idx_net]
            #print(Sn)
        else:
            Sn = 0.162

        Cd, Cl = self.netHydroVar(cosTheta, Sn, 1)
        
        return Ud, Ud_mod, Cd, Cl

    def assemble_velocities(self, Un, Un_mod, Ud, Ud_mod):
        """
        Assemble cylinder and net velocities into full arrays.
        
        Args:
            Un: Normal velocity at cylinder nodes (3, nCylNodes)
            Un_mod: Magnitude at cylinder nodes (nCylNodes,)
            Ud: Velocity at net nodes (3, nNetNodes)
            Ud_mod: Magnitude at net nodes (nNetNodes,)
            
        Returns:
            Ucal: Assembled velocity array (3, nNodes4nbod)
            Ucal_mod: Assembled magnitude array (nNodes4nbod,)
        """
        idx_cyl = self.Elsys.Index_cylType - 1
        idx_net = self.Elsys.Index_netType - 1
        
        Ucal = np.zeros((3, self.Elsys.nNodes4nbod))
        Ucal_mod = np.zeros(self.Elsys.nNodes4nbod)

        Ucal[:, idx_cyl] = Un
        Ucal_mod[idx_cyl] = Un_mod
        Ucal[:, idx_net] = Ud
        Ucal_mod[idx_net] = Ud_mod
        
        return Ucal, Ucal_mod

    def compute_morison_forces(self, Ucal, Ucal_mod, Cd_net=None):
        """
        Compute Morison drag forces and moments from assembled velocities.
        
        F = 0.5 * rho * Cd * A * |U| * U
        M = cross(r_local, F)
        
        Args:
            Ucal: Velocity at all nodes (3, nNodes4nbod)
            Ucal_mod: Velocity magnitude at all nodes (nNodes4nbod,)
            Cd_net: Optional drag coefficients for net elements (nNetNodes,)
                   If provided, updates attachNodeCd for net elements.
            
        Returns:
            F_allEls: Force on each element node (3, nNodes4nbod)
            M_allEls: Moment on each element node (3, nNodes4nbod)
        """
        # Update Cd for net elements if provided
        if Cd_net is not None:
            idx_net = self.Elsys.Index_netType - 1
            self.Elsys.attachNodeCd[idx_net] = Cd_net

        UUmod = Ucal_mod * Ucal
        halvRhoCddA = 0.5 * self.rho * self.Elsys.attachNodeCd * self.Elsys.attachNodeArea
        F_allEls = UUmod * halvRhoCddA
        M_allEls = np.cross(self.Elsys.attachNodePos_loc, UUmod, axis=0) * halvRhoCddA

        return F_allEls, M_allEls

    def compute_element_forces(self, t, x):
        """
        Compute forces and moments for all elements.
        
        This is a convenience method that calls the component functions in sequence.
        For debugging, call the component functions individually.
        
        Args:
            t: Time [s]
            x: State vector [positions; velocities]
            
        Returns:
            F_allEls: Force on each element node (3, nNodes4nbod)
            M_allEls: Moment on each element node (3, nNodes4nbod)
        """
        # Step 1: Compute relative velocity
        Urel, Ufluid, attachNodePos = self.compute_relative_velocity(t, x)
        
        # Step 2: Compute velocity components for each element type
        Un, Un_mod = self.compute_cylinder_velocity(Urel)
        Ud, Ud_mod, Cd, Cl = self.compute_net_velocity(Urel)
        
        # Step 3: Assemble into full arrays
        Ucal, Ucal_mod = self.assemble_velocities(Un, Un_mod, Ud, Ud_mod)
        
        # Step 4: Compute Morison forces
        F_allEls, M_allEls = self.compute_morison_forces(Ucal, Ucal_mod, Cd_net=Cd)

        return F_allEls, M_allEls

    def __call__(self, t, x):
        """
        Calculate total forces and moments for each body.
        
        Args:
            t: Time [s]
            x: State vector [positions; velocities]
            
        Returns:
            F: Force vector (ndof,) with forces and moments for each body
        """
        ndof = len(x) // 2
        F = np.zeros(ndof)

        # Compute element-level forces and moments
        F_allEls, M_allEls = self.compute_element_forces(t, x)

        # Sum forces for each body
        for ibod in range(self.Elsys.nbod):
            idx_start = self.Elsys.Els2bod[ibod, 0]
            idx_end = self.Elsys.Els2bod[ibod, 1]
            
            F_indices = self.Elsys.DoF_Tran[3*ibod:3*(ibod+1)]
            M_indices = self.Elsys.DoF_Rot[3*ibod:3*(ibod+1)]
          
            F[F_indices] = np.sum(F_allEls[:, idx_start:idx_end+1], axis=1)
            
            F[M_indices] = np.sum(M_allEls[:, idx_start:idx_end+1], axis=1)

        return F


class DragForceNB_OT(DragForceNB):
    """
    Optimized drag force calculator with caching.
    
    Inherits from DragForceNB and adds time-based caching to reduce 
    computation when called frequently (e.g., in ODE solvers).
    
    Mirrors the MATLAB FdragNB_v2OT function.
    """
    
    def __init__(self, Elsys, current, wave, rho=1025, g=9.81, threshold=1.0):
        """
        Initialize with caching parameters.
        
        Args:
            Elsys: Element system
            current: Current profile
            wave: Wave parameters
            rho: Water density [kg/m^3]
            g: Gravity [m/s^2]
            threshold: Time threshold for cache invalidation [s]
        """
        super().__init__(Elsys, current, wave, rho, g)
        self._f_hist = np.array([])
        self._t_hist = 0.0
        self._threshold = threshold

    def __call__(self, t: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Calculate drag forces with caching optimization."""
        return self.calculate_with_cache(t, x)

    def calculate_with_cache(self, t: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Calculate drag force with caching for optimization.

        Args:
            t: Time [s]
            x: State vector

        Returns:
            Force vector
        """
        ndof = len(x) // 2
        beta = 1.0

        if t == 0 or len(self._f_hist) == 0:
            self._f_hist = np.zeros(ndof)
            self._t_hist = 0.0

        if t - self._t_hist < self._threshold:
            return self._f_hist.copy()
        else:
            # Call parent's __call__ method for actual calculation
            f1 = super().__call__(t, x)
            F = (1 - beta) * self._f_hist + beta * f1
            self._t_hist = t
            self._f_hist = F.copy()
            return F

    def get_force_function(self) -> Callable[[float, NDArray], NDArray]:
        """Get a force function for ODE solver."""
        def force_func(t: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
            return self.calculate_with_cache(t, x)
        return force_func



#______________ Numba-accelerated helper functions _______________
if HAS_NUMBA:
    from numba import njit, prange
    
    @njit(cache=True, fastmath=True)
    def _wave_kinematics_numba(t, node_x, node_y, node_z, 
                                omega, kX, kY, k, phase, omegaZa):
        """
        Numba-accelerated wave kinematics calculation.
        
        Args:
            t: Time [s]
            node_x, node_y, node_z: Node positions (nNodes,)
            omega, kX, kY, k, phase, omegaZa: Wave parameters (nWaves,)
            
        Returns:
            Uwave_XY, Uwave_Z: Wave velocities (nNodes,)
        """
        nNodes = node_x.shape[0]
        nWaves = omega.shape[0]
        
        Uwave_XY = np.zeros(nNodes)
        Uwave_Z = np.zeros(nNodes)
        
        for iNode in range(nNodes):
            x = node_x[iNode]
            y = node_y[iNode]
            z = min(node_z[iNode], 0.0)  # Cap z at surface
            
            sum_xy = 0.0
            sum_z = 0.0
            for iWave in range(nWaves):
                arg = -omega[iWave] * t + kX[iWave] * x + kY[iWave] * y + phase[iWave]
                eKz = np.exp(k[iWave] * z)
                oZa = omegaZa[iWave]
                sum_xy += oZa * np.sin(arg) * eKz
                sum_z += oZa * np.cos(arg) * eKz
            
            Uwave_XY[iNode] = sum_xy
            Uwave_Z[iNode] = sum_z
        
        return Uwave_XY, Uwave_Z
    
    @njit(cache=True, fastmath=True)
    def _current_profile_numba(z, vel, zlevel):
        """
        Numba-accelerated current profile interpolation.
        
        Args:
            z: Depth values (nNodes,)
            vel: Current velocities at levels (nLevels,)
            zlevel: Depth levels (nLevels,)
            
        Returns:
            u: Current velocity at each depth (nNodes,)
        """
        nNodes = z.shape[0]
        nLevels = vel.shape[0]
        u = np.zeros(nNodes)
        
        for i in range(nNodes):
            zi = z[i]
            if zi > 0:
                u[i] = 0.0
            elif nLevels == 1:
                u[i] = vel[0]
            elif nLevels == 2:
                if zi > zlevel[0]:
                    u[i] = vel[0]
                elif zi <= zlevel[1]:
                    u[i] = vel[1]
                else:
                    slope = (vel[0] - vel[1]) / (zlevel[0] - zlevel[1])
                    u[i] = vel[0] + slope * (zi - zlevel[0])
            else:
                # Multi-level interpolation
                if zi >= zlevel[0]:
                    u[i] = vel[0]
                elif zi <= zlevel[-1]:
                    u[i] = vel[-1]
                else:
                    for j in range(nLevels - 1):
                        if zlevel[j] >= zi > zlevel[j + 1]:
                            slope = (vel[j + 1] - vel[j]) / (zlevel[j + 1] - zlevel[j])
                            u[i] = vel[j] + slope * (zi - zlevel[j])
                            break
        return u

else:
    # Fallback pure NumPy versions (same logic, no JIT)
    def _wave_kinematics_numba(t, node_x, node_y, node_z, 
                                omega, kX, kY, k, phase, omegaZa):
        """Pure NumPy fallback for wave kinematics."""
        z_cap = np.minimum(node_z, 0.0)
        kXx_plus_kYy = np.outer(kX, node_x) + np.outer(kY, node_y)
        arg = -np.outer(omega, np.ones_like(node_x)) * t + kXx_plus_kYy + phase[:, np.newaxis]
        eKz = np.exp(np.outer(k, z_cap))
        
        Uwave_XY = (omegaZa[:, np.newaxis] * np.sin(arg) * eKz).sum(axis=0)
        Uwave_Z = (omegaZa[:, np.newaxis] * np.cos(arg) * eKz).sum(axis=0)
        return Uwave_XY, Uwave_Z
    
    def _current_profile_numba(z, vel, zlevel):
        """Pure NumPy fallback for current profile."""
        return currentProfile(z, vel, zlevel)


@dataclass
class DragForceNB_Fast(BaseForce):
    """
    Optimized N-Body drag force with cached constants and optional Numba acceleration.
    
    This is a drop-in replacement for DragForceNB with better performance:
    - Pre-computes constant values at initialization
    - Uses Numba JIT for wave kinematics (if available)
    - Minimizes memory allocations in hot path
    
    Usage:
        # Same interface as DragForceNB
        drag = DragForceNB_Fast(Elsys, current, wave)
        F = drag(t, x)
    """
    Elsys: ElSys
    current: Current
    wave: Wave
    rho: float = 1025.0
    g: float = 9.81
    
    # Cached arrays (initialized in __post_init__)
    _current_dir: NDArray = field(init=False, repr=False)
    _wave_dir_xy: NDArray = field(init=False, repr=False)
    _omegaZa: NDArray = field(init=False, repr=False)
    _omega_flat: NDArray = field(init=False, repr=False)
    _kX_flat: NDArray = field(init=False, repr=False)
    _kY_flat: NDArray = field(init=False, repr=False)
    _k_flat: NDArray = field(init=False, repr=False)
    _phase_flat: NDArray = field(init=False, repr=False)
    _vel_flat: NDArray = field(init=False, repr=False)
    _zlevel_flat: NDArray = field(init=False, repr=False)
    _idx_cyl: NDArray = field(init=False, repr=False)
    _idx_net: NDArray = field(init=False, repr=False)
    _halvRhoCddA: NDArray = field(init=False, repr=False)
    
    def __post_init__(self):
        """Pre-compute constant values for faster evaluation."""
        # Direction vectors (shape: (3, 1) for broadcasting)
        self._current_dir = np.array([
            [np.cos(np.deg2rad(self.current.propDir))],
            [np.sin(np.deg2rad(self.current.propDir))],
            [0.0]
        ])
        self._wave_dir_xy = np.array([
            [np.cos(np.deg2rad(self.wave.propDir))],
            [np.sin(np.deg2rad(self.wave.propDir))]
        ])
        
        # Wave parameters as contiguous 1D arrays (for Numba)
        self._omegaZa = np.ascontiguousarray((self.wave.omegaCal * self.wave.ZaCal).flatten())
        self._omega_flat = np.ascontiguousarray(self.wave.omegaCal.flatten())
        self._kX_flat = np.ascontiguousarray(self.wave.kXCal.flatten())
        self._kY_flat = np.ascontiguousarray(self.wave.kYCal.flatten())
        self._k_flat = np.ascontiguousarray(self.wave.k.flatten())
        self._phase_flat = np.ascontiguousarray(self.wave.phaseCal.flatten())
        
        # Current profile parameters
        self._vel_flat = np.ascontiguousarray(np.atleast_1d(self.current.vel).flatten())
        self._zlevel_flat = np.ascontiguousarray(np.atleast_1d(self.current.zlevel).flatten())
        
        # Element indices (0-based)
        self._idx_cyl = self.Elsys.Index_cylType - 1
        self._idx_net = self.Elsys.Index_netType - 1
        
        # Pre-compute Morison constant (will be updated for nets if needed)
        self._halvRhoCddA = 0.5 * self.rho * self.Elsys.attachNodeCd * self.Elsys.attachNodeArea
        
        # Pre-allocate working arrays
        self._Ucal = np.zeros((3, self.Elsys.nNodes4nbod))
        self._Ucal_mod = np.zeros(self.Elsys.nNodes4nbod)
    
    def compute_relative_velocity(self, t, x):
        """Compute relative velocity with cached constants."""
        ndof = len(x) // 2
        
        # Update node positions
        attachNodePos = self.Elsys.attachNodePos_globInit.copy()
        for ibod in range(self.Elsys.nbod):
            idx_start = self.Elsys.Els2bod[ibod, 0]
            idx_end = self.Elsys.Els2bod[ibod, 1]
            dof_indices = self.Elsys.bod2DoF_Tran[ibod, :]
            attachNodePos[:3, idx_start:idx_end+1] += x[dof_indices].reshape(3, -1)
        
        # Current velocity (using Numba if available)
        Uc = _current_profile_numba(
            attachNodePos[2, :], 
            self._vel_flat, 
            self._zlevel_flat
        )
        
        # Wave kinematics (using Numba if available)
        Uwave_XY, Uwave_Z = _wave_kinematics_numba(
            t,
            np.ascontiguousarray(attachNodePos[0, :]),
            np.ascontiguousarray(attachNodePos[1, :]),
            np.ascontiguousarray(attachNodePos[2, :]),
            self._omega_flat,
            self._kX_flat,
            self._kY_flat,
            self._k_flat,
            self._phase_flat,
            self._omegaZa
        )
        
        # Assemble fluid velocity (using cached directions)
        Ufluid = (self._current_dir @ Uc.reshape(1, -1) + 
                  np.vstack([self._wave_dir_xy * Uwave_XY, Uwave_Z]))
        
        # Relative velocity
        Urel = Ufluid.copy()
        for ibod in range(self.Elsys.nbod):
            idx_start = self.Elsys.Els2bod[ibod, 0]
            idx_end = self.Elsys.Els2bod[ibod, 1]
            dof_indices = self.Elsys.bod2DoF_Tran[ibod, :]
            Urel[:, idx_start:idx_end+1] = (
                Ufluid[:, idx_start:idx_end+1] * self.current.wakeRatio - 
                x[ndof + dof_indices].reshape(3, -1)
            )
        
        return Urel, Ufluid, attachNodePos
    
    def compute_cylinder_velocity(self, Urel):
        """Compute normal velocity for cylindrical elements."""
        Urel_cyl = Urel[:, self._idx_cyl]
        nodeVec_cyl = self.Elsys.attachNodeVec[:, self._idx_cyl]
        dot_product = np.einsum('ij,ij->j', Urel_cyl, nodeVec_cyl)
        Un = Urel_cyl - dot_product * nodeVec_cyl
        Un_mod = np.sqrt(np.einsum('ij,ij->j', Un, Un))
        return Un, Un_mod
    
    def compute_net_velocity(self, Urel):
        """Compute velocity and drag coefficient for net elements."""
        Ud = Urel[:, self._idx_net]
        Ud_mod = np.sqrt(np.einsum('ij,ij->j', Ud, Ud))
        
        # Unit direction
        with np.errstate(divide='ignore', invalid='ignore'):
            ed = np.where(Ud_mod > 0, Ud / Ud_mod, 0.0)
        
        # Angle of attack
        cosTheta = np.einsum('ij,ij->j', ed, self.Elsys.attachNodeVec[:, self._idx_net])
        Cd, Cl = net_hydro_var(cosTheta, 0.162, 1)
        
        return Ud, Ud_mod, Cd, Cl
    
    def compute_element_forces(self, t, x):
        """Compute forces and moments for all elements."""
        Urel, _, _ = self.compute_relative_velocity(t, x)
        
        # Cylinder velocity
        Un, Un_mod = self.compute_cylinder_velocity(Urel)
        
        # Net velocity
        Ud, Ud_mod, Cd, Cl = self.compute_net_velocity(Urel)
        
        # Assemble (reuse pre-allocated arrays)
        self._Ucal[:, self._idx_cyl] = Un
        self._Ucal_mod[self._idx_cyl] = Un_mod
        self._Ucal[:, self._idx_net] = Ud
        self._Ucal_mod[self._idx_net] = Ud_mod
        
        # Update Cd for nets
        self._halvRhoCddA[self._idx_net] = (
            0.5 * self.rho * Cd * self.Elsys.attachNodeArea[self._idx_net]
        )
        
        # Morison force
        UUmod = self._Ucal_mod * self._Ucal
        F_allEls = UUmod * self._halvRhoCddA
        M_allEls = np.cross(self.Elsys.attachNodePos_loc, UUmod, axis=0) * self._halvRhoCddA
        
        return F_allEls, M_allEls
    
    def __call__(self, t, x):
        """Calculate total forces and moments for each body."""
        ndof = len(x) // 2
        F = np.zeros(ndof)
        
        F_allEls, M_allEls = self.compute_element_forces(t, x)
        
        # Sum forces per body
        for ibod in range(self.Elsys.nbod):
            idx_start = self.Elsys.Els2bod[ibod, 0]
            idx_end = self.Elsys.Els2bod[ibod, 1]
            
            F_indices = self.Elsys.DoF_Tran[3*ibod:3*(ibod+1)]
            M_indices = self.Elsys.DoF_Rot[3*ibod:3*(ibod+1)]
            
            F[F_indices] = F_allEls[:, idx_start:idx_end+1].sum(axis=1)
            F[M_indices] = M_allEls[:, idx_start:idx_end+1].sum(axis=1)
        
        return F


#=============================================================================
#                                 Mooring Force

# Note: LineType is now imported from params module


@dataclass
class QSmoorForce(BaseForce):
    """
    Quasi-static mooring force calculator.

    A dataclass-based implementation that mirrors the MATLAB FQSmoor function.
    Provides `from_config` factory method for easy initialization from JSON configs.
    
    For the simpler wrapper around Cable.FQSmoor, use QSmoorForce instead.
    """

    line_sys: dict = field(default_factory=dict)
    float_bodies: list = field(default_factory=list)
    line_types: list[LineType] = field(default_factory=list)

    @classmethod
    def from_config(
        cls, line_sys: dict, float_bodies: list[dict], line_types_config: dict | LineTypes
    ) -> "QSmoorForce":
        """
        Create MooringForce from configuration.

        Args:
            line_sys: Line system configuration
            float_bodies: List of floating body configurations
            line_types_config: Line types configuration dict or LineTypes object

        Returns:
            MooringForce instance
        """
        # Parse line types - support both dict and LineTypes object
        if isinstance(line_types_config, LineTypes):
            line_types = line_types_config.lineType
        else:
            line_types_data = line_types_config.get("lineType", [])
            if isinstance(line_types_data, dict):
                line_types_data = [line_types_data]
            line_types = [LineType.from_dict(lt) for lt in line_types_data]

        return cls(
            line_sys=line_sys,
            float_bodies=float_bodies,
            line_types=line_types,
        )

    def __call__(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Calculate mooring forces for given state.

        Args:
            x: State vector (displacements and velocities)

        Returns:
            Force vector
        """
        return self.calculate(x)

    def calculate(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Calculate mooring forces.

        Mirrors MATLAB FQSmoor function.

        Args:
            x: State vector [displacements; velocities]

        Returns:
            Force vector (same dimension as half of x)
        """
        ndof = len(x) // 2
        F = np.zeros(ndof)

        if not self.line_sys or not self.line_types:
            return F

        nbod = self.line_sys.get("nbod", 1)
        cal_dof = self.line_sys.get("calDoF", [[1, 6]])
        if isinstance(cal_dof, list) and len(cal_dof) > 0:
            if isinstance(cal_dof[0], (int, float)):
                cal_dof = [cal_dof]
        cal_dof = np.array(cal_dof, dtype=np.int32)

        # Get positions
        fairlead_pos_init = np.array(
            self.line_sys.get("fairleadPos_init", [[0], [0], [0]]), dtype=np.float64
        )
        fairlead_pos = fairlead_pos_init.copy()
        # Support both "anchorPos" and "anchorPos_init" field names
        anchor_pos = np.array(
            self.line_sys.get("anchorPos", 
                self.line_sys.get("anchorPos_init", [[0], [0], [0]])), 
            dtype=np.float64
        )

        # Update fairlead positions based on body motion
        for ibod in range(nbod):
            if ibod < len(self.float_bodies):
                fb = self.float_bodies[ibod]
                fairlead_indices = fb.get("fairleadIndex", [])
                if len(fairlead_indices) > 0:
                    dof_start = int(cal_dof[ibod, 0]) - 1  # Convert to 0-based
                    for idx in fairlead_indices:
                        if idx - 1 < fairlead_pos.shape[1]:
                            fairlead_pos[0:2, idx - 1] += x[dof_start : dof_start + 2]

        # Anchor line force calculation
        anchor_line_pair = self.line_sys.get("anchorLinePair", [])
        anchor_line_type = self.line_sys.get("anchorLineType", [])

        n_anchor_lines = len(anchor_line_pair)
        Aline_FH = np.zeros(n_anchor_lines)
        Aline_FV = np.zeros(n_anchor_lines)
        Aline_proj2xy = np.zeros((2, n_anchor_lines))

        for iline in range(n_anchor_lines):
            pair = anchor_line_pair[iline]
            anchor_idx = pair[0] - 1  # Convert to 0-based
            fairlead_idx = pair[1] - 1

            # Vector from fairlead to anchor
            xF2Anch = anchor_pos[0:2, anchor_idx] - fairlead_pos[0:2, fairlead_idx]
            distance = np.linalg.norm(xF2Anch)

            if distance > 0 and len(self.line_types) > 0:
                line_type_idx = (
                    anchor_line_type[iline] - 1 if iline < len(anchor_line_type) else 0
                )
                if line_type_idx < len(self.line_types):
                    Aline_FH[iline], Aline_FV[iline] = get_tension(
                        self.line_types[line_type_idx], distance
                    )
                    Aline_proj2xy[:, iline] = xF2Anch / distance

        # Shared line force calculation
        shared_line_pair = self.line_sys.get("sharedLinePair", [])
        shared_line_type = self.line_sys.get("sharedLineType", [])

        n_shared_lines = len(shared_line_pair)
        Sline_FH = np.zeros((2, n_shared_lines))
        Sline_FV = np.zeros((2, n_shared_lines))
        Sline_proj2xy = np.zeros((2, 2 * n_shared_lines))

        for iline in range(n_shared_lines):
            pair = shared_line_pair[iline]
            f1_idx = pair[0] - 1
            f2_idx = pair[1] - 1

            # Vector from fairlead 1 to fairlead 2
            xF2F = fairlead_pos[0:2, f2_idx] - fairlead_pos[0:2, f1_idx]
            distance = np.linalg.norm(xF2F)

            if distance > 0 and len(self.line_types) > 0:
                line_type_idx = (
                    shared_line_type[iline] - 1 if iline < len(shared_line_type) else 0
                )
                if line_type_idx < len(self.line_types):
                    FH, FV = get_tension_2ends(self.line_types[line_type_idx], distance)
                    Sline_FH[:, iline] = FH
                    Sline_FV[:, iline] = FV
                    Sline_proj2xy[:, 2 * iline] = xF2F / distance
                    Sline_proj2xy[:, 2 * iline + 1] = -xF2F / distance

        # Assemble forces to bodies
        for ibod in range(nbod):
            if ibod < len(self.float_bodies):
                fb = self.float_bodies[ibod]
                dof_start = int(cal_dof[ibod, 0]) - 1

                # Anchor line contributions
                aline_slave = fb.get("AlineSlave", [])
                for line_idx in aline_slave:
                    if line_idx - 1 < n_anchor_lines:
                        F[dof_start : dof_start + 2] += (
                            Aline_FH[line_idx - 1] * Aline_proj2xy[:, line_idx - 1]
                        )

                # Shared line contributions
                # SlineSlave structure: [[line_indices], [end_indicators]]
                # - SlineSlave[0]: line indices (1-based)
                # - SlineSlave[1]: end indicators (0 or 1)
                # MATLAB: Sline_FH(1, line_indices) * Sline_proj2xy(:, 2*line_idx - end_indicator)
                sline_slave = fb.get("SlineSlave", [])
                if len(sline_slave) >= 2 and len(sline_slave[0]) > 0:
                    line_indices = sline_slave[0]  # List of line indices (1-based)
                    end_indicators = sline_slave[1]  # List of end indicators (0 or 1)
                    
                    for j in range(len(line_indices)):
                        line_idx = line_indices[j] - 1  # Convert to 0-based
                        end_indicator = end_indicators[j] if j < len(end_indicators) else 0
                        
                        if 0 <= line_idx < n_shared_lines:
                            # MATLAB: proj_idx = 2*line_idx - end_indicator (1-based)
                            # Python: proj_idx = 2*line_idx + (1 - end_indicator) (0-based)
                            proj_idx = 2 * line_idx + (1 - end_indicator)
                            # Use Sline_FH[0, line_idx] - always first row like MATLAB
                            F[dof_start : dof_start + 2] += (
                                Sline_FH[0, line_idx] * Sline_proj2xy[:, proj_idx]
                            )

        return F

    def get_force_function(self) -> Callable[[float, NDArray], NDArray]:
        """
        Get a time-dependent force function for ODE solver.

        Returns:
            Function f(t, x) -> F
        """

        def force_func(t: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
            return self.calculate(x)

        return force_func


#=============================================================================
#                                 PotentialWave Force

@dataclass
class Wave2Force:
    """
    Transfer function from wave to force.
    
    Contains amplitude and phase transfer functions that convert wave
    elevation to forces at each DOF.
    """
    Xamp: NDArray[np.floating] = field(default_factory=lambda: np.array([]))  # Amplitude transfer (ndof, n_freq)
    phase: NDArray[np.floating] = field(default_factory=lambda: np.array([]))  # Phase transfer [rad] (ndof, n_freq)


@dataclass
class PotWaveForce(BaseForce):
    """
    Irregular wave excitation force calculator using potential flow theory.

    Mirrors the MATLAB FirregWaveNB function. Computes wave forces based on:
    - Wave spectrum (from Wave class in params.py)
    - Wave-to-force transfer functions (RAOs from hydrodynamic analysis)
    - Body positions (accounts for spatial phase differences)
    
    The force at each DOF is computed as:
        F = sum(Xamp * Za * sin(omega*t - kX*x - kY*y + phase_wave + phase_tf))
    
    where Xamp and phase_tf are the transfer function amplitude and phase.
    """

    wave: Any = field(default=None)  # Wave object from params.py
    wave_to_force: Wave2Force = field(default_factory=Wave2Force)
    line_sys: dict = field(default_factory=dict)

    @classmethod
    def from_config(
        cls, wave: Any, float_bodies: list[dict], line_sys: dict
    ) -> "PotWaveForce":
        """
        Create PotWaveForce with interpolated transfer functions.

        Mirrors MATLAB interpWaveTrans1st function. Interpolates the
        wave-to-force transfer functions to match the wave spectrum frequencies.

        Args:
            wave: Wave object from params.py (with omegaCal, ZaCal, etc.)
            float_bodies: List of floating body configurations with waveTrans1st
            line_sys: Line system configuration (nbod, nDoF, calDoF, bodPos_init)

        Returns:
            PotWaveForce instance with interpolated transfer functions
        """
        ndof = int(line_sys.get("nDoF", 6))
        nbod = int(line_sys.get("nbod", 1))
        
        # Get wave frequencies for interpolation
        omega_cal = np.asarray(wave.omegaCal).flatten()
        n_freq = len(omega_cal)

        # Initialize transfer functions
        Xamp = np.zeros((ndof, n_freq))
        phase = np.zeros((ndof, n_freq))

        prop_dir = wave.propDir

        for ibod in range(nbod):
            if ibod >= len(float_bodies):
                continue

            fb = float_bodies[ibod]
            wave_trans = fb.get("waveTrans1st", {})

            if not wave_trans:
                continue

            # Get transfer function data from floatBody
            dir_vec = np.array(wave_trans.get("dirVec", [0]), dtype=np.float64).flatten()
            omega_vec = np.array(wave_trans.get("omegaVec", []), dtype=np.float64).flatten()
            Xamp_data = np.array(wave_trans.get("Xamp", []), dtype=np.float64)
            phase_data = np.array(wave_trans.get("phase", []), dtype=np.float64)

            if len(omega_vec) == 0 or Xamp_data.size == 0:
                continue

            # Find closest direction index
            idir = int(np.argmin(np.abs(dir_vec - prop_dir)))

            # Interpolate to wave frequencies for each motion DOF
            for imot in range(6):  # 6 DOF per body
                dof_idx = ibod * 6 + imot
                if dof_idx >= ndof:
                    continue

                # Select the right slice based on data dimensions
                # Xamp_data can be: (n_dir, n_freq, 6), (n_freq, 6), or (n_freq,)
                if Xamp_data.ndim == 3:
                    Xamp_slice = Xamp_data[idir, :, imot]
                    phase_slice = phase_data[idir, :, imot]
                elif Xamp_data.ndim == 2:
                    Xamp_slice = Xamp_data[:, imot]
                    phase_slice = phase_data[:, imot]
                elif Xamp_data.ndim == 1 and imot == 0:
                    Xamp_slice = Xamp_data
                    phase_slice = phase_data
                else:
                    continue

                # Interpolate to wave spectrum frequencies
                Xamp[dof_idx, :] = np.interp(omega_cal, omega_vec, Xamp_slice)
                phase[dof_idx, :] = np.interp(omega_cal, omega_vec, phase_slice)

        wave_to_force = Wave2Force(Xamp=Xamp, phase=phase)

        print(f"PotWaveForce: Interpolated transfer functions for {nbod} bodies, {n_freq} frequencies")

        return cls(wave=wave, wave_to_force=wave_to_force, line_sys=line_sys)

    def __call__(self, t: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Calculate wave forces at time t."""
        return self.calculate(t, x)

    def calculate(self, t: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Calculate wave excitation forces.

        Mirrors MATLAB FirregWaveNB function:
            kXx_plus_kYy = XYpos_temp * [wave.kX(:), wave.kY(:)]';
            F = sum(wave2F.Xamp .* (wave.ZaCal' .* sin(wave.omegaCal'*t - kXx_plus_kYy + 
                    wave.phaseCal' + wave2F.phase)), 2);

        Args:
            t: Time [s]
            x: State vector [displacements; velocities]

        Returns:
            Force vector (ndof,)
        """
        ndof = len(x) // 2
        nbod = int(self.line_sys.get("nbod", 1))
        cal_dof = self.line_sys.get("calDoF", [[1, 6]])
        if isinstance(cal_dof, list) and len(cal_dof) > 0:
            if isinstance(cal_dof[0], (int, float)):
                cal_dof = [cal_dof]
        cal_dof = np.array(cal_dof, dtype=np.int32)

        # Get initial body positions
        bod_pos_init = np.array(
            self.line_sys.get("bodPos_init", np.zeros((3, nbod))), dtype=np.float64
        )

        # Build XY position arrays for each DOF
        # Each DOF inherits the XY position of its parent body
        XY_pos = np.zeros((ndof, 2))

        for ibod in range(nbod):
            dof_start = int(cal_dof[ibod, 0]) - 1  # Convert to 0-based
            dof_end = int(cal_dof[ibod, 1])
            
            # Initial body position + current displacement
            XY_pos[dof_start:dof_end, 0] = bod_pos_init[0, ibod] + x[dof_start]
            XY_pos[dof_start:dof_end, 1] = bod_pos_init[1, ibod] + x[dof_start + 1]

        # Get wave parameters (from Wave class in params.py)
        kX = np.asarray(self.wave.kXCal).flatten()  # (n_freq,)
        kY = np.asarray(self.wave.kYCal).flatten()  # (n_freq,)
        omega = np.asarray(self.wave.omegaCal).flatten()  # (n_freq,)
        Za = np.asarray(self.wave.ZaCal).flatten()  # (n_freq,)
        phase_wave = np.asarray(self.wave.phaseCal).flatten()  # (n_freq,)

        # Phase calculation: kX*x + kY*y for each DOF and frequency
        # kXY shape: (2, n_freq), XY_pos shape: (ndof, 2)
        # Result: (ndof, n_freq)
        kXx_plus_kYy = XY_pos @ np.vstack([kX, kY])

        # Force calculation using superposition
        # omega*t: scalar * (n_freq,) -> (n_freq,)
        # Broadcast: omega*t (n_freq,) - kXx_plus_kYy (ndof, n_freq) + phase_wave (n_freq,)
        w2f = self.wave_to_force
        
        # F = sum over frequencies: Xamp * Za * sin(omega*t - kXx - kYy + phase_wave + phase_tf)
        F = np.sum(
            w2f.Xamp * Za * np.sin(omega * t - kXx_plus_kYy + phase_wave + w2f.phase),
            axis=1,
        )

        return F

    def get_force_function(self) -> Callable[[float, NDArray], NDArray]:
        """Get a force function for ODE solver."""
        def force_func(t: float, x: NDArray[np.floating]) -> NDArray[np.floating]:
            return self.calculate(t, x)
        return force_func


#=============================================================================
#                                 Exports
__all__ = [
    # Base
    "BaseForce",
    # Helper functions
    "net_hydro_var",
    "currentProfile",
    "get_tension",
    "get_tension_2ends",
    # Simple forces
    "HarmonicForce",
    "irregWaveForceNB",
    "pulloutForce",
    "DecayForce",
    "convForceOT",
    # Current forces
    "CurrentForce",
    # Drag forces
    "DragForceNB",
    "DragForceNB_OT",
    # Mooring forces
    "QSmoorForce",       
    # Wave forces
    "Wave2Force",
    "PotWaveForce",
    # Decorators
    "timer",
    "debug_call",
    "check_shape",
    "disabled_if",
]


#=============================================================================
#                                 testing
# ------------------------
# Testing forces with usr0 configs
if __name__ == "__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    import params
    from pprint import pprint
    
    # Get the usr0 directory path
    script_dir = Path(__file__).parent
    usr0_path = script_dir.parent.parent / "usr0"
    
    print("="*70)
    print(" "*15 + "TESTING FORCE FUNCTIONS WITH usr0 JSONs")
    print("="*70)
    print(f"usr0 path: {usr0_path}")
    print()

    # ----------------------------------------------------------------
    #                   < Testing HarmonicForce >
    print("-"*70)
    print(" "*20 + "< testing HarmonicForce: >")
    hf = HarmonicForce(amplitude=1.0, omega=0.1*2*np.pi, phase=0.0)
    pprint(hf.__annotations__)
    pprint(hf)

    @timer
    def run_force():
        result = hf(t=0.25, x=[0])
        print(f"  HarmonicForce(t=0.25) = {result:.6f}")
        print(f"  Expected: sin(2.0 * 0.25) = {np.sin(0.5):.6f}")
        return result
    run_force()

    t = np.linspace(0, 100, 1000)
    x = np.sin(2.0 * t)
    f = np.zeros(len(t))
    for i in range(len(t)):
        f[i] = hf(t[i], [x[i]])
   
    plt.figure()
    plt.title("Test of 'HarmonicForce' function")
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.plot(t, f)
    #plt.show()







    # ----------------------------------------------------------------
    #         < testing LineTypes loading and get_tension >
    print("-"*70)
    print(" "*20 + "< testing LineTypes loading and get_tension: >")
    line_types = params.load_line_types(usr0_path / "lineTypes_23-Feb-2025_12-25-34.json")
    print(f"  Loaded {len(line_types)} line types")
    
    if len(line_types) > 0:

        lt0 = line_types[0]
        test_distance = 1000.0  # test horizontal distance
        H, V = get_tension(lt0, test_distance)
        print(f"  get_tension(distance={test_distance}m):")
        print(f"    Horizontal tension: {H:.2f} N")
        print(f"    Vertical tension: {V:.2f} N")


        lt1 = line_types[1]
        print(f"  Max XXF2F: {np.max(lt1.XXF2F):.2f} m")
        test_distance = 1000.0  # test horizontal distance
        H2, V2 = get_tension_2ends(lt1, test_distance)
        print(f"  get_tension(distance={test_distance}m):")
        print("\nAt end 1:")
        print(f"    Horizontal tension: {H2[0]:.2f} N")
        print(f"    Vertical tension: {V2[0]:.2f} N")
        print("\nAt end 2:")
        print(f"    Horizontal tension: {H2[1]:.2f} N")
        print(f"    Vertical tension: {V2[1]:.2f} N")

        # ---------------------------------------------------------------------------
        #    < Plot interpolation test over range of distances >
        #    Including extrapolation beyond lookup table boundaries
        print("\n" + "-"*50)
        print("  Testing interpolation/extrapolation over range of distances...")
        
        # Extrapolation margin: extend test range by this percentage beyond lookup table
        extrap_margin = 0.15  # 15% beyond each boundary
        
        # --- Test get_tension (anchor line - lt0) ---
        XX2anch_lt0 = np.asarray(lt0.XX2anch).flatten()
        x_min_lt0, x_max_lt0 = np.min(XX2anch_lt0), np.max(XX2anch_lt0)
        x_range_lt0 = x_max_lt0 - x_min_lt0
        # Extend test range beyond lookup table for extrapolation testing
        x_test_min_lt0 = x_min_lt0 - extrap_margin * x_range_lt0
        x_test_max_lt0 = x_max_lt0 + extrap_margin * x_range_lt0
        test_distances_lt0 = np.linspace(x_test_min_lt0, x_test_max_lt0, 250)
        
        print(f"    lt0 lookup range: [{x_min_lt0:.1f}, {x_max_lt0:.1f}] m")
        print(f"    lt0 test range:   [{x_test_min_lt0:.1f}, {x_test_max_lt0:.1f}] m")
        
        # Demo: show one extrapolation warning example
        print("\n  Demo: triggering extrapolation warning for get_tension:")
        _ = get_tension(lt0, x_test_min_lt0, extrapolate=True, warn_extrap=True)
        
        # Loop over distances (suppress warnings to avoid flooding console)
        H_interp_lt0 = np.zeros_like(test_distances_lt0)
        V_interp_lt0 = np.zeros_like(test_distances_lt0)
        for i, d in enumerate(test_distances_lt0):
            H_interp_lt0[i], V_interp_lt0[i] = get_tension(lt0, d, warn_extrap=False)
        
        # --- Test get_tension_2ends (fairlead-to-fairlead line - lt1) ---
        XXF2F_lt1 = np.asarray(lt1.XXF2F).flatten()
        x_min_lt1, x_max_lt1 = np.min(XXF2F_lt1), np.max(XXF2F_lt1)
        x_range_lt1 = x_max_lt1 - x_min_lt1
        # Extend test range beyond lookup table for extrapolation testing
        x_test_min_lt1 = x_min_lt1 - extrap_margin * x_range_lt1
        x_test_max_lt1 = x_max_lt1 + extrap_margin * x_range_lt1
        test_distances_lt1 = np.linspace(x_test_min_lt1, x_test_max_lt1, 250)
        
        print(f"\n    lt1 lookup range: [{x_min_lt1:.1f}, {x_max_lt1:.1f}] m")
        print(f"    lt1 test range:   [{x_test_min_lt1:.1f}, {x_test_max_lt1:.1f}] m")
        
        # Demo: show one extrapolation warning example
        print("\n  Demo: triggering extrapolation warning for get_tension_2ends:")
        _ = get_tension_2ends(lt1, x_test_min_lt1, extrapolate=True, warn_extrap=True)
        
        # Loop over distances (suppress warnings to avoid flooding console)
        H_interp_lt1 = np.zeros((len(test_distances_lt1), 2))
        V_interp_lt1 = np.zeros((len(test_distances_lt1), 2))
        for i, d in enumerate(test_distances_lt1):
            H_arr, V_arr = get_tension_2ends(lt1, d, warn_extrap=False)
            H_interp_lt1[i, :] = H_arr
            V_interp_lt1[i, :] = V_arr

        # --- Plotting ---
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle("Tension Interpolation & Extrapolation Test", fontsize=14, fontweight='bold')
        
        # Style for extrapolation boundary lines
        boundary_style = dict(color='gray', linestyle='--', lw=1.5, alpha=0.8)
        
        # Plot lt0: get_tension (anchor line)
        ax1 = axes[0, 0]
        ax1.plot(test_distances_lt0, H_interp_lt0 / 1e6, 'b-', lw=2, label='H interp/extrap')
        # Overlay original lookup table points
        HH_lt0 = np.asarray(lt0.HH).flatten()
        ax1.scatter(XX2anch_lt0, HH_lt0 / 1e6, c='r', s=25, marker='o', 
                    label='H lookup', zorder=5, alpha=0.8)
        # Add extrapolation boundaries
        ax1.axvline(x_min_lt0, **boundary_style, label='interp bounds')
        ax1.axvline(x_max_lt0, **boundary_style)
        # Shade extrapolation regions
        ax1.axvspan(x_test_min_lt0, x_min_lt0, alpha=0.1, color='red', label='extrap zone')
        ax1.axvspan(x_max_lt0, x_test_max_lt0, alpha=0.1, color='red')
        ax1.set_xlabel("Horizontal Distance [m]")
        ax1.set_ylabel("Horizontal Tension [MN]")
        ax1.set_title("lt0 (Anchor Line) - Horizontal Tension")
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        ax2.plot(test_distances_lt0, V_interp_lt0 / 1e6, 'b-', lw=2, label='V interp/extrap')
        VV_lt0 = np.asarray(lt0.VV).flatten()
        ax2.scatter(XX2anch_lt0, VV_lt0 / 1e6, c='r', s=25, marker='o',
                    label='V lookup', zorder=5, alpha=0.8)
        ax2.axvline(x_min_lt0, **boundary_style, label='interp bounds')
        ax2.axvline(x_max_lt0, **boundary_style)
        ax2.axvspan(x_test_min_lt0, x_min_lt0, alpha=0.1, color='red', label='extrap zone')
        ax2.axvspan(x_max_lt0, x_test_max_lt0, alpha=0.1, color='red')
        ax2.set_xlabel("Horizontal Distance [m]")
        ax2.set_ylabel("Vertical Tension [MN]")
        ax2.set_title("lt0 (Anchor Line) - Vertical Tension")
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Plot lt1: get_tension_2ends (fairlead-to-fairlead line)
        ax3 = axes[1, 0]
        ax3.plot(test_distances_lt1, H_interp_lt1[:, 0] / 1e6, 'b-', lw=2, label='H1 interp/extrap')
        ax3.plot(test_distances_lt1, H_interp_lt1[:, 1] / 1e6, 'g--', lw=2, label='H2 interp/extrap')
        # Overlay original lookup table points
        HH_lt1 = np.atleast_2d(lt1.HH)
        if HH_lt1.shape[0] >= 2:
            ax3.scatter(XXF2F_lt1, HH_lt1[0, :] / 1e6, c='r', s=25, marker='o',
                        label='H1 lookup', zorder=5, alpha=0.8)
            ax3.scatter(XXF2F_lt1, HH_lt1[1, :] / 1e6, c='orange', s=25, marker='s',
                        label='H2 lookup', zorder=5, alpha=0.8)
        ax3.axvline(x_min_lt1, **boundary_style, label='interp bounds')
        ax3.axvline(x_max_lt1, **boundary_style)
        ax3.axvspan(x_test_min_lt1, x_min_lt1, alpha=0.1, color='red', label='extrap zone')
        ax3.axvspan(x_max_lt1, x_test_max_lt1, alpha=0.1, color='red')
        ax3.set_xlabel("Fairlead-to-Fairlead Distance [m]")
        ax3.set_ylabel("Horizontal Tension [MN]")
        ax3.set_title("lt1 (F2F Line) - Horizontal Tension (both ends)")
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        ax4.plot(test_distances_lt1, V_interp_lt1[:, 0] / 1e6, 'b-', lw=2, label='V1 interp/extrap')
        ax4.plot(test_distances_lt1, V_interp_lt1[:, 1] / 1e6, 'g--', lw=2, label='V2 interp/extrap')
        VV_lt1 = np.atleast_2d(lt1.VV)
        if VV_lt1.shape[0] >= 2:
            ax4.scatter(XXF2F_lt1, VV_lt1[0, :] / 1e6, c='r', s=25, marker='o',
                        label='V1 lookup', zorder=5, alpha=0.8)
            ax4.scatter(XXF2F_lt1, VV_lt1[1, :] / 1e6, c='orange', s=25, marker='s',
                        label='V2 lookup', zorder=5, alpha=0.8)
        ax4.axvline(x_min_lt1, **boundary_style, label='interp bounds')
        ax4.axvline(x_max_lt1, **boundary_style)
        ax4.axvspan(x_test_min_lt1, x_min_lt1, alpha=0.1, color='red', label='extrap zone')
        ax4.axvspan(x_max_lt1, x_test_max_lt1, alpha=0.1, color='red')
        ax4.set_xlabel("Fairlead-to-Fairlead Distance [m]")
        ax4.set_ylabel("Vertical Tension [MN]")
        ax4.set_title("lt1 (F2F Line) - Vertical Tension (both ends)")
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(usr0_path / "tension_interpolation_test.png", dpi=150)
        print(f"  Saved plot to: {usr0_path / 'tension_interpolation_test.png'}")
        plt.show()
    
    



    # ----------------------------------------------------------------
    #                    < testing CurrentProfile >
    print("-"*70)
    print(" "*20 + "< testing CurrentForce with usr0 config: >")
    
    # Load environment config
    env = params.load_env_config(usr0_path / "env_withoutVar.json")
    
    print(f"  Current velocity: {env.current.vel}")
    print(f"  Current zlevel: {env.current.zlevel}")
    print(f"  Current wake ratio: {env.current.wakeRatio}")
    print(f"  Current propagation dir: {env.current.propDir} deg")
    
    # Load floating bodies from offshore system
    offsys_data = params.load_json(usr0_path / "sys6cage_DgEls_wT1st.json")
    pprint(offsys_data.keys())



    float_bodies_data = offsys_data.get("floatBody", [])
    float_bodies = [FloatBody.from_dict(fb_data) for fb_data in float_bodies_data]
    
    print(f"  Loaded {len(float_bodies)} floating bodies")
    
    # Create current force calculator
    Fcurrent = CurrentForce(floatBody=float_bodies, current=env.current)
    offsys = params.OffSys.from_dict(offsys_data)
    pprint(Fcurrent.current_profile(z=offsys.bodPos_init[2,:],
                                       vel=env.current.vel, 
                                       zlevel=env.current.zlevel))
    

    #----------------------------------------------------------------
    #         < testing CurrentForceNB (current only, no wave) >
    print("-"*70)
    print(" "*20 + "< testing CurrentForceNB (current only): >")
    
    # Load floating bodies from offshore system
    env = params.load_env_config(usr0_path / "env_withoutVar.json")
    offsys = params.OffSys.from_dict(offsys_data)
    elsys  = params.ElSys.from_floatBodies(offsys.floatBodies)
    
    # Create CurrentForceNB (no wave dependency)
    Fcurrent_nb = CurrentForceNB(elsys, env.current)
    
    # Create test state vector
    ndof = len(offsys.floatBodies) * 6
    x0 = np.zeros(ndof * 2)  # [displacements; velocities]
    
    print(f"\n  Created CurrentForceNB with {elsys.nbod} bodies, {elsys.nNodes4nbod} nodes")
    print(f"  Current: vel={env.current.vel}, dir={env.current.propDir}°")
    
    # Test current profile
    print("\n___ Checking current profile: ___")
    z_nodes = elsys.attachNodePos_globInit[2, :]
    Uc = Fcurrent_nb.current_profile(z_nodes, env.current.vel, env.current.zlevel)
    print(f"  z range: [{z_nodes.min():.2f}, {z_nodes.max():.2f}] m")
    print(f"  Uc range: [{Uc.min():.4f}, {Uc.max():.4f}] m/s")
    
    # Test force calculation
    print("\n___ Checking CurrentForceNB output: ___")
    F_current = Fcurrent_nb(t=0.0, x=x0)
    print(f"  Shape: {F_current.shape}")
    print(f"  F (first 6 DOF): {F_current[:6]}")
    print(f"  Max force: {np.max(np.abs(F_current)):.2f} N")
    
    # Compare with DragForceNB (with wave) at t=0
    print("\n___ Comparing with DragForceNB at t=0: ___")
    Fdrag_nb = DragForceNB(elsys, env.current, env.wave)
    F_drag = Fdrag_nb(t=0.0, x=x0)
    
    diff = F_drag - F_current
    print(f"  DragForceNB F[:]: {F_drag[:]}")
    print(f"  CurrentForceNB F[:]: {F_current[:]}")
    print(f"  Difference (wave contribution) F[:]: {diff[:]}")
    print(f"  Max diff: {np.max(np.abs(diff)):.2f} N")
    







    #----------------------------------------------------------------
    #             < testing DragForce with zero wave >
    print("-"*70)
    print(" "*20 + "< testing DragForce with zero wave: >")
    
    # Load floating bodies from offshore system
    env = params.load_env_config(usr0_path / "env_withoutVar_0Wave.json")
    offsys = params.OffSys.from_dict(offsys_data)
    elsys  = params.ElSys.from_floatBodies(offsys.floatBodies)
    Fdrag = DragForceNB(elsys, env.current, env.wave)

    # Create test state vector (6 DOF per body: x,y,z,rx,ry,rz)
    ndof = len(offsys.floatBodies) * 6
    x0 = np.zeros(ndof * 2)  # [displacements; velocities]
    print("\n___ Checking current profile: ___")
    pprint(Fdrag.current_profile(z= elsys.attachNodePos_globInit[2,:].copy(),
                                    vel=env.current.vel, 
                                    zlevel=env.current.zlevel))

    print("\n___ Checking wave shape: ___")
    print("Shape ofZaCal: ", env.wave.ZaCal.shape)
    print("Shape of omegaCal: ", env.wave.omegaCal.shape)
    print("Shape of phaseCal: ", env.wave.phaseCal.shape)
    print("Shape of kYCal: ", env.wave.kYCal.shape)
    print("Shape of kXCal: ", env.wave.kXCal.shape)

    print("\n___ Checking Fdrag shape: ___")
    print("Shape of Fdrag: ", Fdrag(t=0.25, x=x0).shape)
    print("Fdrag: ", Fdrag(t=0.25, x=x0))


    #----------------------------------------------------------------
    #             < testing DragForceNB with wave >
    print("-"*70)
    print(" "*20 + "< testing DragForceNB with wave: >")
    
    # Load floating bodies from offshore system
    env = params.load_env_config(usr0_path / "env_withoutVar.json")
    offsys = params.OffSys.from_dict(offsys_data)
    elsys  = params.ElSys.from_floatBodies(offsys.floatBodies)
    Fdrag = DragForceNB(elsys, env.current, env.wave)

    # Create test state vector (6 DOF per body: x,y,z,rx,ry,rz)
    ndof = len(offsys.floatBodies) * 6
    x0 = np.zeros(ndof * 2)  # [displacements; velocities]
    print("\n___ Checking current profile: ___\n")
    pprint(Fdrag.current_profile(z= elsys.attachNodePos_globInit[2,:].copy(),
                                    vel=env.current.vel, 
                                    zlevel=env.current.zlevel))

    print("\n___ Checking wave shape: ___\n")
    print("Shape of ZaCal: ", env.wave.ZaCal.shape)
    print("Shape of omegaCal: ", env.wave.omegaCal.shape)
    print("Shape of phaseCal: ", env.wave.phaseCal.shape)
    print("Shape of kYCal: ", env.wave.kYCal.shape)
    print("Shape of kXCal: ", env.wave.kXCal.shape)

     
    # Step-by-step velocity and force computation for debugging
    t_test = 0.25
    
    print("\n___ Step 1: Compute relative velocity (Urel) ___")
    Urel, Ufluid, attachNodePos = Fdrag.compute_relative_velocity(t=t_test, x=x0)
    print(f"Shape of Ufluid: {Ufluid.shape}")
    print(f"Ufluid[:, :5]:\n{Ufluid[:, :5]}")
    print(f"Shape of Urel: {Urel.shape}")
    print(f"Urel[:, :5]:\n{Urel[:, :5]}")
    
    print("\n___ Step 2a: Compute cylinder velocity (Un, Un_mod) ___")
    Un, Un_mod = Fdrag.compute_cylinder_velocity(Urel)
    print(f"Shape of Un: {Un.shape}")
    print(f"Un[:, :5]:\n{Un[:, :5]}")
    print(f"Shape of Un_mod: {Un_mod.shape}")
    print(f"Un_mod[:5]: {Un_mod[:5]}")
    
    print("\n___ Step 2b: Compute net velocity (Ud, Ud_mod, Cd) ___")
    Ud, Ud_mod, Cd, Cl = Fdrag.compute_net_velocity(Urel)
    print(f"Shape of Ud: {Ud.shape}")
    print(f"Ud[:, :5]:\n{Ud[:, :5]}")
    print(f"Shape of Ud_mod: {Ud_mod.shape}")
    print(f"Ud_mod[:5]: {Ud_mod[:5]}")
    print(f"Shape of Cd: {Cd.shape}")
    print(f"Cd[:5]: {Cd[:5]}")
    
    print("\n___ Step 3: Assemble velocities (Ucal, Ucal_mod) ___")
    Ucal, Ucal_mod = Fdrag.assemble_velocities(Un, Un_mod, Ud, Ud_mod)
    print(f"Shape of Ucal: {Ucal.shape}")
    print(f"Ucal[:, :5]:\n{Ucal[:, :5]}")
    print(f"Shape of Ucal_mod: {Ucal_mod.shape}")
    print(f"Ucal_mod[:5]: {Ucal_mod[:5]}")
    
    print("\n___ Step 4: Compute Morison forces ___")
    F_allEls, M_allEls = Fdrag.compute_morison_forces(Ucal, Ucal_mod, Cd_net=Cd)
    print(f"Shape of F_allEls: {F_allEls.shape}")
    print(f"F_allEls[:, :5]:\n{F_allEls[:, :5]}")
    print(f"Shape of M_allEls: {M_allEls.shape}")
    print(f"M_allEls[:, :5]:\n{M_allEls[:, :5]}")

    # Save all intermediate values to CSV for MATLAB comparison
    import pandas as pd
    
    # Save velocities
    df_Ufluid = pd.DataFrame(Ufluid.T, columns=['Ux', 'Uy', 'Uz'])
    df_Urel = pd.DataFrame(Urel.T, columns=['Ux', 'Uy', 'Uz'])
    df_Ucal = pd.DataFrame(Ucal.T, columns=['Ux', 'Uy', 'Uz'])
    df_Ucal['Ucal_mod'] = Ucal_mod
    
    df_Ufluid.to_csv(usr0_path / "Ufluid.csv", index=False)
    df_Urel.to_csv(usr0_path / "Urel.csv", index=False)
    df_Ucal.to_csv(usr0_path / "Ucal.csv", index=False)
    
    # Save forces
    df_F = pd.DataFrame(F_allEls.T, columns=['Fx', 'Fy', 'Fz'])
    df_M = pd.DataFrame(M_allEls.T, columns=['Mx', 'My', 'Mz'])
    df_F.to_csv(usr0_path / "F_allEls.csv", index=False)
    df_M.to_csv(usr0_path / "M_allEls.csv", index=False)
    
    print(f"\n___ Saved CSV files to {usr0_path} ___\n")
    print("  - Ufluid.csv, Urel.csv, Ucal.csv")
    print("  - F_allEls.csv, M_allEls.csv")
    
    # Debug: Element system info
    print("\n___ Element system info ___")
    print(f"idx_cyl (0-based): first 5 = {(elsys.Index_cylType - 1)[:5]}")
    print(f"idx_net (0-based): first 5 = {(elsys.Index_netType - 1)[:5]}")
    print(f"attachNodeVec shape: {elsys.attachNodeVec.shape}")
    print(f"attachNodeVec[:, :3]:\n{elsys.attachNodeVec[:, :3]}")
    print(f"attachNodePos_loc shape: {elsys.attachNodePos_loc.shape}")
    print(f"attachNodeCd (first 10): {elsys.attachNodeCd[:10]}")
    print(f"attachNodeArea (first 10): {elsys.attachNodeArea[:10]}")

    print("\n___ Checking Fdrag shape: ___\n")
    print("Shape of Fdrag: ", Fdrag(t=0.25, x=x0).shape)
    print("Fdrag: ", Fdrag(t=0.25, x=x0))





    # ----------------------------------------------------------------
    #           < Test Quasi-static mooring force >
    print("-"*70)
    print(" "*20 + "< testing MooringForce with usr0 config: >")
    
    # Create MooringForce from configs
    mooring_force = QSmoorForce.from_config(
        line_sys=offsys_data,
        float_bodies=float_bodies_data,
        line_types_config=line_types,
    )
    print(f"  MooringForce created with {len(mooring_force.line_types)} line types")
    
    # Get system dimensions
    nbod = offsys_data.get("nbod", 6)
    ndof = nbod * 6  # 6 DoF per body
    
    # Test with zero displacement
    print("\n___ Test 1: Zero displacement ___\n")
    x0_moor = np.zeros(ndof * 2)  # [displacements; velocities]
    F_moor_0 = mooring_force(x0_moor)
    print(f"Shape of F_moor: {F_moor_0.shape}")
    print(f"F_moor (zero displacement):")
    for ibod in range(min(nbod, 6)):  # Show first 3 bodies
        print(f"  Body {ibod}: F=[{F_moor_0[6*ibod]:.1f}, {F_moor_0[6*ibod+1]:.1f}, {F_moor_0[6*ibod+2]:.1f}] N, "
              f"M=[{F_moor_0[6*ibod+3]:.1f}, {F_moor_0[6*ibod+4]:.1f}, {F_moor_0[6*ibod+5]:.1f}] Nm")
    
    # Test with small horizontal displacement (1m in x for all bodies)
    print("\n___ Test 2: Small x-displacement (1m) ___\n")
    x1_moor = np.zeros(ndof * 2)
    for ibod in range(nbod):
        x1_moor[6*ibod] = 1.0  # 1m displacement in x
    F_moor_1 = mooring_force(x1_moor)
    print(f"F_moor (1m x-displacement):")
    for ibod in range(min(nbod, 6)):
        print(f"  Body {ibod}: F=[{F_moor_1[6*ibod]:.1f}, {F_moor_1[6*ibod+1]:.1f}, {F_moor_1[6*ibod+2]:.1f}] N")
    
    # Test with larger displacement (10m in x)
    print("\n___ Test 3: Larger x-displacement (10m) ___\n")
    x2_moor = np.zeros(ndof * 2)
    for ibod in range(nbod):
        x2_moor[6*ibod] = 10.0  # 10m displacement in x
    F_moor_2 = mooring_force(x2_moor)
    print(f"F_moor (10m x-displacement):")
    for ibod in range(min(nbod, 6)):
        print(f"  Body {ibod}: F=[{F_moor_2[6*ibod]:.1f}, {F_moor_2[6*ibod+1]:.1f}, {F_moor_2[6*ibod+2]:.1f}] N")
    
    # Check restoring force direction (should be opposite to displacement)
    print("\n___ Force direction check ___\n")
    delta_F = F_moor_1 - F_moor_0
    print(f"Change in Fx for body 0: {delta_F[0]:.2f} N (should be negative for +x displacement)")
    
    # Solve for equilibrium (find x where F ≈ 0)
    print("\n___ Test 4: Equilibrium solver ___\n")
    
    def force_residual(x_disp):
        """Residual function for equilibrium: returns displacement DOFs only."""
        x_full = np.zeros(ndof * 2)
        x_full[:ndof] = x_disp
        F = mooring_force(x_full)
        return F  # We want F = 0 at equilibrium
    
    # Initial guess (small perturbation from zero)
    x0_guess = np.zeros(ndof)
    x0_guess[::6] = 0.1  # Small x-displacement initial guess for each body
    
    # Solve for equilibrium
    x_eq, info, ier, msg = fsolve(force_residual, x0_guess, full_output=True)
    
    if ier == 1:
        print(f"Equilibrium found! (converged)")
        F_eq = force_residual(x_eq)
        print(f"Max residual force: {np.max(np.abs(F_eq)):.2e} N")
        print(f"Equilibrium displacements (first 3 bodies):")
        for ibod in range(min(nbod, 6)):
            print(f"  Body {ibod}: x=[{x_eq[6*ibod]:.4f}, {x_eq[6*ibod+1]:.4f}, {x_eq[6*ibod+2]:.4f}] m, "
                  f"rot=[{np.rad2deg(x_eq[6*ibod+3]):.4f}, {np.rad2deg(x_eq[6*ibod+4]):.4f}, {np.rad2deg(x_eq[6*ibod+5]):.4f}] deg")
    else:
        print(f"Equilibrium solver did not converge: {msg}")
        print(f"Final residual norm: {np.linalg.norm(force_residual(x_eq)):.2e} N")
    
    # Print mooring system info
    print("\n___ Mooring system info ___\n")
    print(f"Number of bodies: {nbod}")
    print(f"Number of anchor lines: {len(offsys_data.get('anchorLinePair', []))}")
    print(f"Number of shared lines: {len(offsys_data.get('sharedLinePair', []))}")
    print(f"Number of line types: {len(mooring_force.line_types)}")






    # ----------------------------------------------------------------
    #              < Test Potential wave load >
    print("-"*70)
    print(" "*20 + "< testing PotWaveForce: >")
    
    # Check if waveTrans1st exists in float bodies
    has_wave_trans = any(fb.get("waveTrans1st") for fb in float_bodies_data)
    
    if has_wave_trans:
        # Create PotWaveForce from configs
        pot_wave_force = PotWaveForce.from_config(
            wave=env.wave,
            float_bodies=float_bodies_data,
            line_sys=offsys_data,
        )
        
        # Test at different times
        print("\n___ Wave force time series test ___\n")
        t_vals = [0.0, 0.25, 0.5, 1.0, 2.0]
        x0_wave = np.zeros(ndof * 2)
        
        for t_test in t_vals:
            F_wave = pot_wave_force(t_test, x0_wave)
            print(f"t={t_test:.2f}s: F_wave[0:3] = [{F_wave[0]:.1f}, {F_wave[1]:.1f}, {F_wave[2]:.1f}] N")
        
        # Check force variation over time
        print("\n___ Wave force oscillation check ___\n")
        F0 = pot_wave_force(0.0, x0_wave)
        F1 = pot_wave_force(env.wave.Tp / 4, x0_wave)  # Quarter period
        F2 = pot_wave_force(env.wave.Tp / 2, x0_wave)  # Half period
        
        print(f"At t=0: F[0] = {F0[0]:.2f} N")
        print(f"At t=Tp/4: F[0] = {F1[0]:.2f} N")
        print(f"At t=Tp/2: F[0] = {F2[0]:.2f} N")
        print(f"Wave period Tp = {env.wave.Tp:.2f} s")
        
    else:
        print("  No waveTrans1st data in float bodies, skipping PotWaveForce test")
        print("  To test PotWaveForce, add 'waveTrans1st' field to float body configs")

    print("="*70)
    print(" "*20 + "ALL FORCE TESTS COMPLETED")
    print("="*70)




    #----------------------------------------------------------------
    #         < testing DragForceNB_Fast time history with plots >
    print("-"*70)
    print(" "*20 + "< Comparing DragForceNB vs DragForceNB_Fast: >")
    
    # Create both force calculators
    Fdrag_orig = DragForceNB(elsys, env.current, env.wave)
    Fdrag_fast = DragForceNB_Fast(elsys, env.current, env.wave)
    
    print(f"\n  Numba available: {HAS_NUMBA}")
    
    # Get number of bodies
    nbod = len(offsys.floatBodies)
    
    # Time parameters
    T_end = 100.0  # Total simulation time [s]
    dt = 0.1       # Time step [s]
    t_history = np.arange(0, T_end, dt)
    n_steps = len(t_history)
    
    # Initialize arrays to store force history
    F_history_orig = np.zeros((n_steps, ndof))
    F_history_fast = np.zeros((n_steps, ndof))
    
    #--- Accuracy check: compare single call ---
    print("\n___ Accuracy comparison (single call at t=0.25s) ___")
    F_orig_single = Fdrag_orig(0.25, x0)
    F_fast_single = Fdrag_fast(0.25, x0)
    
    max_abs_diff = np.max(np.abs(F_orig_single - F_fast_single))
    max_rel_diff = np.max(np.abs(F_orig_single - F_fast_single) / (np.abs(F_orig_single) + 1e-10))
    
    print(f"  Max absolute difference: {max_abs_diff:.2e}")
    print(f"  Max relative difference: {max_rel_diff:.2e}")
    print(f"  Results match: {np.allclose(F_orig_single, F_fast_single, rtol=1e-10, atol=1e-12)}")
    
    #--- Benchmark: time history computation ---
    print(f"\n___ Benchmark: {n_steps} time steps (T_end={T_end}s, dt={dt}s) ___")
    
    # Warm-up call for Numba JIT compilation
    _ = Fdrag_fast(0.0, x0)
    
    @timer
    def compute_drag_history_orig():
        for i, t in enumerate(t_history):
            print(f"Computing Fdrag_orig at step: {i} of {n_steps}")
            F_history_orig[i, :] = Fdrag_orig(t, x0)
        return F_history_orig
    
    @timer
    def compute_drag_history_fast():
        for i, t in enumerate(t_history):
            print(f"Computing Fdrag_fast at step: {i} of {n_steps}")
            F_history_fast[i, :] = Fdrag_fast(t, x0)
        return F_history_fast
    
    import time
    
    print("\n  Running DragForceNB (original)...")
    t0 = time.perf_counter()
    compute_drag_history_orig()
    t_orig = time.perf_counter() - t0
    
    print("  Running DragForceNB_Fast...")
    t0 = time.perf_counter()
    compute_drag_history_fast()
    t_fast = time.perf_counter() - t0
    
    speedup = t_orig / t_fast
    print(f"\n  === SPEEDUP: {speedup:.2f}x ===")
    print(f"  Original: {t_orig:.3f}s, Fast: {t_fast:.3f}s")
    
    # Verify time history matches
    max_hist_diff = np.max(np.abs(F_history_orig - F_history_fast))
    print(f"\n  Time history max difference: {max_hist_diff:.2e}")
    print(f"  Time history matches: {np.allclose(F_history_orig, F_history_fast, rtol=1e-10, atol=1e-12)}")
    
    # Use fast version for plots
    F_history = F_history_fast
    
    # Plot force time history for first body (6 DOF)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('DragForceNB_Fast Time History - Body 0', fontsize=14)
    
    dof_labels = ['Fx [N]', 'Fy [N]', 'Fz [N]', 'Mx [Nm]', 'My [Nm]', 'Mz [Nm]']
    
    for i, (ax, label) in enumerate(zip(axes.flatten(), dof_labels)):
        ax.plot(t_history, F_history[:, i], 'b-', linewidth=0.8)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(label)
        ax.set_title(f'DOF {i}: {label}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(usr0_path / "DragForce_TimeHistory_Body0.png", dpi=150)
    print(f"  Saved: {usr0_path / 'DragForce_TimeHistory_Body0.png'}")
    
    # Plot total force magnitude for all bodies
    fig2, axes2 = plt.subplots(2, 3, figsize=(14, 8))
    fig2.suptitle('DragForceNB_Fast Time History - All Bodies (Translational Forces)', fontsize=14)
    
    for ibod in range(min(nbod, 6)):
        ax = axes2.flatten()[ibod]
        Fx = F_history[:, 6*ibod]
        Fy = F_history[:, 6*ibod + 1]
        Fz = F_history[:, 6*ibod + 2]
        F_mag = np.sqrt(Fx**2 + Fy**2 + Fz**2)
        
        ax.plot(t_history, Fx, 'r-', linewidth=0.8, label='Fx')
        ax.plot(t_history, Fy, 'g-', linewidth=0.8, label='Fy')
        ax.plot(t_history, Fz, 'b-', linewidth=0.8, label='Fz')
        ax.plot(t_history, F_mag, 'k--', linewidth=1.0, label='|F|')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Force [N]')
        ax.set_title(f'Body {ibod}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(usr0_path / "DragForce_TimeHistory_AllBodies.png", dpi=150)
    print(f"  Saved: {usr0_path / 'DragForce_TimeHistory_AllBodies.png'}")
    
    # Statistics summary
    print("\n___ Force statistics (Body 0) ___\n")
    for i, label in enumerate(dof_labels):
        f_dof = F_history[:, i]
        print(f"  {label}: mean={np.mean(f_dof):.2f}, std={np.std(f_dof):.2f}, "
              f"min={np.min(f_dof):.2f}, max={np.max(f_dof):.2f}")
    
    # Spectral analysis (FFT) of surge force
    print("\n___ Spectral analysis of Fx (Body 0) ___\n")
    from scipy.fft import fft, fftfreq
    
    Fx_body0 = F_history[:, 0]
    Fx_detrend = Fx_body0 - np.mean(Fx_body0)  # Remove mean
    
    N = len(Fx_detrend)
    Fx_fft = fft(Fx_detrend)
    freq = fftfreq(N, dt)
    
    # Only positive frequencies
    pos_mask = freq > 0
    freq_pos = freq[pos_mask]
    Fx_amp = 2.0 / N * np.abs(Fx_fft[pos_mask])
    
    # Find dominant frequency
    peak_idx = np.argmax(Fx_amp)
    peak_freq = freq_pos[peak_idx]
    peak_period = 1.0 / peak_freq if peak_freq > 0 else np.inf
    
    print(f"  Dominant frequency: {peak_freq:.4f} Hz")
    print(f"  Dominant period: {peak_period:.2f} s")
    print(f"  Wave peak period (Tp): {env.wave.Tp:.2f} s")
    
    # Plot spectrum
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.semilogy(freq_pos, Fx_amp, 'b-', linewidth=0.8)
    ax3.axvline(1/env.wave.Tp, color='r', linestyle='--', label=f'Wave freq (1/Tp={1/env.wave.Tp:.4f} Hz)')
    ax3.set_xlabel('Frequency [Hz]')
    ax3.set_ylabel('Amplitude [N]')
    ax3.set_title('FFT of Surge Drag Force (Fx, Body 0)')
    ax3.set_xlim([0, 0.5])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(usr0_path / "DragForce_FFT_Body0.png", dpi=150)
    print(f"  Saved: {usr0_path / 'DragForce_FFT_Body0.png'}")
    
    plt.show()
