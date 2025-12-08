"""Time integration solvers."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

# Optional tqdm import for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


class TextProgressBar:
    """Simple text-based progress bar (fallback when tqdm not available)."""
    
    def __init__(self, total: int, desc: str = "", width: int = 40):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self._last_percent = -1
        
    def update(self, n: int = 1):
        self.current += n
        percent = int(100 * self.current / self.total)
        if percent != self._last_percent:
            self._last_percent = percent
            filled = int(self.width * self.current / self.total)
            bar = '█' * filled + '░' * (self.width - filled)
            sys.stdout.write(f'\r  {self.desc}: |{bar}| {percent}% ({self.current}/{self.total})')
            sys.stdout.flush()
    
    def close(self):
        sys.stdout.write('\n')
        sys.stdout.flush()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


@dataclass
class RK45Solver:
    """
    Runge-Kutta 4-5 solver wrapper.

    Wraps scipy.integrate.solve_ivp for ODE integration.
    """

    t_start: float = 0.0
    t_end: float = 100.0
    dt: float = 0.1
    rtol: float = 1e-6
    atol: float = 1e-9
    method: str = "RK45"

    def solve(
        self,
        f: Callable[[float, NDArray], NDArray],
        x0: NDArray[np.floating],
        t_eval: NDArray[np.floating] | None = None,
    ) -> tuple[NDArray, NDArray]:
        """
        Solve the ODE system.

        Args:
            f: Function f(t, x) returning dx/dt
            x0: Initial state
            t_eval: Times at which to store solution

        Returns:
            Tuple of (t, x) arrays
        """
        if t_eval is None:
            t_eval = np.arange(self.t_start, self.t_end + self.dt, self.dt)

        result = solve_ivp(
            f,
            [self.t_start, self.t_end],
            x0,
            method=self.method,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
        )

        return result.t, result.y.T


def solve_dynamics(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    force_func: Callable[[float, NDArray], NDArray],
    x0: NDArray[np.floating],
    t_span: tuple[float, float],
    dt: float = 0.1,
    method: str = "RK45",
    rtol: float = 1e-4,
) -> tuple[NDArray, NDArray]:
    """
    Solve state-space dynamics with external forces.

    Solves: dx/dt = A*x + B*u(t,x)

    Mirrors MATLAB ode45 integration.

    Args:
        A: State matrix
        B: Input matrix
        force_func: Function u(t, x) returning force vector
        x0: Initial state
        t_span: (t_start, t_end)
        dt: Time step for output
        method: Integration method
        rtol: Relative tolerance

    Returns:
        Tuple of (t, x) arrays
    """
    t_start, t_end = t_span
    t_eval = np.arange(t_start, t_end + dt, dt)

    def dxdt(t: float, x: NDArray) -> NDArray:
        u = force_func(t, x)
        return A @ x + B @ u

    result = solve_ivp(
        dxdt,
        t_span,
        x0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
    )

    if not result.success:
        print(f"Warning: Integration may not have converged: {result.message}")

    return result.t, result.y.T


def solve_dynamics_with_progress(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    force_func: Callable[[float, NDArray], NDArray],
    x0: NDArray[np.floating],
    t_span: tuple[float, float],
    dt: float = 0.1,
    method: str = "RK45",
    rtol: float = 1e-4,
    atol: float = 1e-9,
    desc: str = "Simulating",
    chunk_size: int = 50,
) -> tuple[NDArray, NDArray]:
    """
    Solve state-space dynamics with progress bar (chunked integration).

    Uses chunked integration for better performance while showing progress.
    Much faster than step-by-step while still providing progress updates.

    Args:
        A: State matrix
        B: Input matrix
        force_func: Function u(t, x) returning force vector
        x0: Initial state
        t_span: (t_start, t_end)
        dt: Time step for output
        method: Integration method
        rtol: Relative tolerance
        atol: Absolute tolerance
        desc: Progress bar description
        chunk_size: Number of output steps per integration chunk (default 50)

    Returns:
        Tuple of (t, x) arrays
    """
    t_start, t_end = t_span
    t_eval = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t_eval)
    
    # Pre-allocate results
    x_out = np.zeros((n_steps, len(x0)))
    x_out[0] = x0
    
    def dxdt(t: float, x: NDArray) -> NDArray:
        u = force_func(t, x)
        return A @ x + B @ u
    
    # Determine number of chunks
    n_chunks = (n_steps - 1 + chunk_size - 1) // chunk_size
    
    # Choose progress bar
    if HAS_TQDM:
        pbar = tqdm(total=n_chunks, desc=desc, unit="chunk")
    else:
        pbar = TextProgressBar(total=n_chunks, desc=desc)
    
    # Chunked integration (much faster than step-by-step)
    try:
        chunk_start = 0
        while chunk_start < n_steps - 1:
            chunk_end = min(chunk_start + chunk_size, n_steps - 1)
            
            # Time points for this chunk
            t_chunk = t_eval[chunk_start:chunk_end + 1]
            
            result = solve_ivp(
                dxdt,
                [t_chunk[0], t_chunk[-1]],
                x_out[chunk_start],
                method=method,
                t_eval=t_chunk[1:],  # Exclude first point (already have it)
                rtol=rtol,
                atol=atol,
            )
            
            if result.success:
                x_out[chunk_start + 1:chunk_end + 1] = result.y.T
            else:
                print(f"\nWarning at t={t_chunk[0]:.2f}: {result.message}")
                if result.y.size > 0:
                    x_out[chunk_start + 1:chunk_end + 1] = result.y.T
            
            chunk_start = chunk_end
            pbar.update(1)
    finally:
        pbar.close()
    
    return t_eval, x_out


def solve_dynamics_fast(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    force_func: Callable[[float, NDArray], NDArray],
    x0: NDArray[np.floating],
    t_span: tuple[float, float],
    dt: float = 0.1,
    method: str = "RK45",
    rtol: float = 1e-4,
    atol: float = 1e-9,
    max_step: float | None = None,
    verbose: bool = True,
    show_progress: bool = True,
    progress_interval: float = 1.0,
) -> tuple[NDArray, NDArray]:
    """
    Fastest integration - single solve_ivp call with optional progress tracking.

    This is the recommended method for production simulations.
    Progress is tracked via force function calls (minimal overhead).

    Args:
        A: State matrix
        B: Input matrix
        force_func: Function u(t, x) returning force vector
        x0: Initial state
        t_span: (t_start, t_end)
        dt: Time step for output
        method: Integration method ('RK45', 'RK23', 'DOP853', 'Radau', 'BDF')
        rtol: Relative tolerance
        atol: Absolute tolerance
        max_step: Maximum step size (None = no limit)
        verbose: Print timing info
        show_progress: Show progress bar during integration
        progress_interval: Minimum seconds between progress updates (default 1.0)

    Returns:
        Tuple of (t, x) arrays
    """
    import time
    
    t_start, t_end = t_span
    t_eval = np.arange(t_start, t_end + dt, dt)
    duration = t_end - t_start
    
    # Progress tracking state
    progress_state = {
        'last_update': time.perf_counter(),
        'last_t': t_start,
        'nfev': 0,
        'pbar': None,
    }
    
    # Create progress bar
    if show_progress:
        if HAS_TQDM:
            progress_state['pbar'] = tqdm(
                total=100, 
                desc="Integrating", 
                unit="%",
                bar_format='{desc}: |{bar}| {percentage:.0f}% [t={postfix[0]:.1f}s, {postfix[1]} evals]',
                postfix=[t_start, 0]
            )
        else:
            progress_state['pbar'] = TextProgressBar(total=100, desc="Integrating")
    
    def dxdt_with_progress(t: float, x: NDArray) -> NDArray:
        """Wrapper that tracks progress via force function calls."""
        progress_state['nfev'] += 1
        
        # Update progress bar (throttled to avoid overhead)
        if show_progress:
            now = time.perf_counter()
            if now - progress_state['last_update'] >= progress_interval:
                progress = int(100 * (t - t_start) / duration)
                delta = progress - (progress_state['pbar'].current if hasattr(progress_state['pbar'], 'current') else 0)
                
                if HAS_TQDM:
                    progress_state['pbar'].n = progress
                    progress_state['pbar'].postfix = [t, progress_state['nfev']]
                    progress_state['pbar'].refresh()
                elif delta > 0:
                    progress_state['pbar'].update(delta)
                
                progress_state['last_update'] = now
                progress_state['last_t'] = t
        
        u = force_func(t, x)
        return A @ x + B @ u
    
    if verbose and not show_progress:
        print(f"  Integrating {t_start:.1f}s -> {t_end:.1f}s (dt={dt}s, {len(t_eval)} points)...")
    
    t0 = time.perf_counter()
    
    try:
        result = solve_ivp(
            dxdt_with_progress,
            t_span,
            x0,
            method=method,
            t_eval=t_eval,
            rtol=rtol,
            atol=atol,
            max_step=max_step if max_step else np.inf,
            dense_output=False,
        )
    finally:
        # Close progress bar
        if show_progress and progress_state['pbar']:
            if HAS_TQDM:
                progress_state['pbar'].n = 100
                progress_state['pbar'].postfix = [t_end, progress_state['nfev']]
                progress_state['pbar'].refresh()
            progress_state['pbar'].close()
    
    elapsed = time.perf_counter() - t0
    
    if verbose:
        status = "✓" if result.success else "✗"
        print(f"  {status} Completed in {elapsed:.2f}s ({progress_state['nfev']} function evals)")
    
    if not result.success:
        print(f"  Warning: {result.message}")
    
    return result.t, result.y.T


def solve_dynamics_stepwise(
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    force_func: Callable[[float, NDArray], NDArray],
    x0: NDArray[np.floating],
    t_span: tuple[float, float],
    dt: float = 0.1,
    method: str = "RK45",
    rtol: float = 1e-4,
    desc: str = "Simulating",
) -> tuple[NDArray, NDArray]:
    """
    Step-by-step integration with progress bar (SLOW - for debugging only).

    This method calls solve_ivp for each time step, which is inefficient.
    Use solve_dynamics_fast() or solve_dynamics_with_progress() instead.

    Only use this when you need:
    - Fine-grained progress updates
    - Ability to inspect/modify state at each step
    - Debugging integration issues

    Args:
        A: State matrix
        B: Input matrix
        force_func: Function u(t, x) returning force vector
        x0: Initial state
        t_span: (t_start, t_end)
        dt: Time step for output
        method: Integration method
        rtol: Relative tolerance
        desc: Progress bar description

    Returns:
        Tuple of (t, x) arrays
    """
    t_start, t_end = t_span
    t_eval = np.arange(t_start, t_end + dt, dt)
    n_steps = len(t_eval)
    
    # Pre-allocate results
    x_out = np.zeros((n_steps, len(x0)))
    x_out[0] = x0
    
    def dxdt(t: float, x: NDArray) -> NDArray:
        u = force_func(t, x)
        return A @ x + B @ u
    
    # Choose progress bar
    if HAS_TQDM:
        pbar = tqdm(total=n_steps - 1, desc=desc, unit="step")
    else:
        pbar = TextProgressBar(total=n_steps - 1, desc=desc)
    
    # Step-by-step integration (slow but detailed)
    try:
        for i in range(n_steps - 1):
            result = solve_ivp(
                dxdt,
                [t_eval[i], t_eval[i + 1]],
                x_out[i],
                method=method,
                t_eval=[t_eval[i + 1]],
                rtol=rtol,
            )
            
            if result.success:
                x_out[i + 1] = result.y[:, -1]
            else:
                print(f"\nWarning at t={t_eval[i]:.2f}: {result.message}")
                try:
                    y_arr = np.asarray(result.y)
                    if y_arr.size > 0:
                        x_out[i + 1] = y_arr[:, -1]
                    else:
                        x_out[i + 1] = x_out[i]
                except:
                    x_out[i + 1] = x_out[i]
            
            pbar.update(1)
    finally:
        pbar.close()
    
    return t_eval, x_out


def solve_static(
    force_func: Callable[[NDArray], NDArray],
    x0: NDArray[np.floating],
    tol: float = 1e-4,
    max_iter: int = 1000,
    method: str = "hybr",
    verbose: bool = True,
) -> NDArray[np.floating]:
    """
    Solve for static equilibrium.

    Finds x such that force_func(x) = 0 using root-finding algorithms.

    Args:
        force_func: Function F(x) returning force vector
        x0: Initial guess
        tol: Convergence tolerance
        max_iter: Maximum iterations
        method: Solver method - 'hybr' (Powell hybrid), 'lm' (Levenberg-Marquardt),
                'broyden1', 'krylov', or 'fsolve' (legacy)
        verbose: Print convergence info

    Returns:
        Equilibrium state
    """
    from scipy.optimize import root, fsolve
    
    if method == "fsolve":
        # Legacy fsolve approach
        result = fsolve(force_func, x0, full_output=True, maxfev=max_iter*len(x0))
        x_eq = result[0]
        info = result[1]
        residual = np.linalg.norm(info["fvec"])
    else:
        # Use scipy.optimize.root with better methods
        result = root(
            force_func, 
            x0, 
            method=method,
            tol=tol,
            options={'maxiter': max_iter}
        )
        x_eq = result.x
        residual = np.linalg.norm(result.fun)
        
        if verbose and not result.success:
            print(f"  Root finder message: {result.message}")
    
    if verbose and residual > tol:
        print(f"Warning: Static solution may not have converged. Residual: {residual:.6e}")

    return x_eq


def solve_static_iterative(
    force_func: Callable[[NDArray], NDArray],
    x0: NDArray[np.floating],
    tol: float = 1e-2,
    max_iter: int = 100,
    relaxation: float = 0.5,
    verbose: bool = True,
) -> NDArray[np.floating]:
    """
    Solve for static equilibrium using iterative relaxation.
    
    This is more robust for problems with large force imbalances.
    Uses: x_new = x_old + relaxation * K_inv @ F(x_old)
    where K_inv is an approximate stiffness inverse.

    Args:
        force_func: Function F(x) returning force vector
        x0: Initial guess
        tol: Convergence tolerance (relative force change)
        max_iter: Maximum iterations
        relaxation: Relaxation factor (0 < relaxation <= 1)
        verbose: Print iteration info

    Returns:
        Equilibrium state
    """
    x = x0.copy()
    ndof = len(x)
    
    # Estimate stiffness from initial force gradient
    F0 = force_func(x)
    F0_norm = np.linalg.norm(F0)
    
    if verbose:
        print(f"  Initial force norm: {F0_norm:.2e}")
    
    # Simple scaling: assume ~1e6 N/m stiffness per DOF
    k_est = 1e6  # N/m approximate stiffness
    
    for i in range(max_iter):
        F = force_func(x)
        F_norm = np.linalg.norm(F)
        
        # Update: move in direction of force (F > 0 means push in +x direction)
        dx = relaxation * F / k_est
        
        # Limit step size to avoid overshoot
        dx_max = 1.0  # max 1m per iteration
        dx_norm = np.linalg.norm(dx)
        if dx_norm > dx_max:
            dx = dx * (dx_max / dx_norm)
        
        x = x + dx
        
        # Check convergence
        rel_change = np.linalg.norm(dx) / (np.linalg.norm(x) + 1e-10)
        
        if verbose and (i % 10 == 0 or i < 5):
            print(f"    Iter {i:3d}: |F|={F_norm:.2e}, |dx|={np.linalg.norm(dx):.4f}, rel={rel_change:.2e}")
        
        if F_norm < tol or rel_change < 1e-8:
            if verbose:
                print(f"  Converged at iteration {i}, |F|={F_norm:.2e}")
            break
    else:
        if verbose:
            print(f"  Max iterations reached. |F|={F_norm:.2e}")
    
    return x

