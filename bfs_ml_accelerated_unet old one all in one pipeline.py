"""
BFS ML-Accelerated CFD Simulation (Progressive U-Net Version)

This script runs Backward-Facing Step (BFS) simulations with ML acceleration using
a cascaded U-Net architecture:
1. Run coarse mesh simulation (10x10)
2. Use pretrained progressive U-Net stages to super-resolve to fine mesh (400x400)
   - 10→20→40→80→200→400 (5 stages)
3. Run fine mesh simulation with ML initialization
4. Compare with normal (non-accelerated) simulation

Uses the same ML models trained on lid-driven cavity to test generalization.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import h5py
import os
import sys
from typing import Dict, Optional, Tuple, List
from numba import njit, prange
from datetime import datetime
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import interpolate


# ==============================================================================
# Helper Functions
# ==============================================================================

def bicubic_interpolate_batch(x, target_size):
    """
    Bicubic interpolation for batched multi-channel images.
    
    Args:
        x: (N, H, W, C) tensor or numpy array
        target_size: (target_H, target_W) tuple
    Returns:
        Interpolated tensor/array of shape (N, target_H, target_W, C)
    """
    # Convert to tensor if numpy array
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    else:
        x_tensor = x
    
    # Perform bicubic interpolation
    result = tf.image.resize(x_tensor, target_size, method='bicubic')
    
    # Convert back to numpy if input was numpy
    if is_numpy:
        return result.numpy()
    return result


def create_timestamped_output_dir(base_dir: str = "outputs") -> str:
    """
    Create a timestamped output directory in format: outputs/dd-mm-yyyy-h-m-s
    
    Args:
        base_dir: Base directory name (default: "outputs")
    
    Returns:
        Path to the created timestamped directory
    """
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_run_summary(filepath: str, info: Dict[str, Dict[str, str]]):
    """
    Save simulation configuration and results summary to a text file.
    
    Args:
        filepath: Path to save the summary file
        info: Dictionary of sections, where each section is a dictionary of key-value pairs
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{'='*70}\n")
            f.write(f"BFS SIMULATION RUN SUMMARY\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")
            
            for section, content in info.items():
                f.write(f"{section.upper()}\n")
                f.write(f"{'-'*len(section)}\n")
                for key, value in content.items():
                    f.write(f"{key:<30}: {value}\n")
                f.write("\n")
                
        print(f"Run summary saved to: {filepath}")
    except Exception as e:
        print(f"Failed to save run summary: {e}")


def load_solver_from_hdf5(filepath: str, Re: float, nx: int, ny: int,
                          dt: float, scheme: str, lx: float, ly: float,
                          bc: 'BoundaryConditions',
                          step_height: float = 0.94, h: float = 1.0, Ub: float = 1.0,
                          relaxation_factors: Dict[str, float] = None) -> 'CFDSolver':
    """
    Load a previously saved CFD solver state from HDF5 file.
    
    Args:
        filepath: Path to the HDF5 file
        Re, nx, ny, dt, scheme, lx, ly: Solver parameters
        bc: Boundary conditions
        step_height, h, Ub: BFS geometry parameters
        relaxation_factors: Under-relaxation factors
    
    Returns:
        CFDSolver object with loaded state
    """
    print(f"\nLoading solver state from: {filepath}")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")
    
    # Create a new solver with the same parameters
    mesh = MeshParameters(nx=nx, ny=ny, lx=lx, ly=ly)
    fluid = FluidProperties(Re=Re, rho=1.0)
    
    solver_settings = SolverSettings(
        dt=dt,
        scheme=scheme,
        max_iterations=1,  # Not used when loading
        convergence_criteria={'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6},
        relaxation_factors=relaxation_factors
    )
    
    solver = CFDSolver(mesh, fluid, solver_settings, bc,
                      step_height=step_height, h=h, Ub=Ub)
    
    # Load data from HDF5
    with h5py.File(filepath, 'r') as f:
        group_name = f"Re{Re}_mesh{nx}x{ny}"
        
        if group_name not in f:
            available_groups = list(f.keys())
            raise ValueError(f"Group '{group_name}' not found in HDF5. Available: {available_groups}")
        
        grp = f[group_name]
        
        # Load u, v, p fields
        u_flat = grp['u'][:]
        v_flat = grp['v'][:]
        p_flat = grp['p'][:]
        
        # Reshape to grid (note: saved as flattened transpose)
        u_2d = u_flat.reshape((ny, nx)).T
        v_2d = v_flat.reshape((ny, nx)).T
        p_2d = p_flat.reshape((ny, nx)).T
        
        # Load into solver (with ghost cells)
        solver.Var[0, 1:-1, 1:-1] = u_2d
        solver.Var[1, 1:-1, 1:-1] = v_2d
        solver.Var[2, 1:-1, 1:-1] = p_2d
        
        # Copy to VarOld
        solver.VarOld = solver.Var.copy()
        
        print(f"✓ Loaded solver state from {filepath}")
        print(f"  Resolution: {grp.attrs['nx']}x{grp.attrs['ny']}")
        print(f"  Reynolds: {grp.attrs['reynolds_number']}")
    
    return solver



def standardize_with_stats(arr, mean, std):
    """Standardize array with given mean and std"""
    std = 1e-8 if std == 0 else std
    return (arr - mean) / std


def inverse_standardize(arr, mean, std):
    """Inverse standardization"""
    return arr * std + mean


def normalize_with_stats_3channel(arr_3ch: np.ndarray, stats: Dict) -> np.ndarray:
    """
    Apply 3-channel normalization (u, v, p as channels).
    
    Args:
        arr_3ch: (H, W, 3) array with channels [u, v, p]
        stats: Dict with keys 'u', 'v', 'p', each containing (mean, std)
    
    Returns:
        Normalized (H, W, 3) array
    """
    normalized = np.zeros_like(arr_3ch)
    for ch_idx, ch_name in enumerate(['u', 'v', 'p']):
        mean, std = stats[ch_name]
        normalized[..., ch_idx] = (arr_3ch[..., ch_idx] - mean) / std
    return normalized


def denormalize_with_stats_3channel(arr_3ch: np.ndarray, stats: Dict) -> np.ndarray:
    """
    Inverse 3-channel normalization.
    
    Args:
        arr_3ch: (H, W, 3) normalized array
        stats: Dict with keys 'u', 'v', 'p', each containing (mean, std)
    
    Returns:
        Denormalized (H, W, 3) array
    """
    denormalized = np.zeros_like(arr_3ch)
    for ch_idx, ch_name in enumerate(['u', 'v', 'p']):
        mean, std = stats[ch_name]
        denormalized[..., ch_idx] = arr_3ch[..., ch_idx] * std + mean
    return denormalized


def normalize_per_sample(arr: np.ndarray):
    """
    Per-sample Z-score normalization of 3-channel flow fields.

    Args:
        arr: (N, H, W, 3) array with channels [u, v, p]
    Returns:
        (normalized, stats) where stats has shape (N, 3, 2) — per-channel [mean, std]
    """
    arr = arr.astype(np.float32)
    N = arr.shape[0]
    stats = np.zeros((N, 3, 2), dtype=np.float32)
    normalized = np.copy(arr)
    for i in range(N):
        for c in range(3):
            mean = arr[i, :, :, c].mean()
            std = float(arr[i, :, :, c].std())
            if std < 1e-8:
                std = 1e-8
            stats[i, c, 0] = mean
            stats[i, c, 1] = std
            normalized[i, :, :, c] = (arr[i, :, :, c] - mean) / std
    return normalized, stats


def denormalize_per_sample(arr: np.ndarray, stats: np.ndarray) -> np.ndarray:
    """
    Inverse per-sample normalization.

    Args:
        arr: (N, H, W, 3) normalized array
        stats: (N, 3, 2) array from normalize_per_sample
    Returns:
        Denormalized (N, H, W, 3) array
    """
    result = np.copy(arr).astype(np.float32)
    N = arr.shape[0]
    for i in range(N):
        for c in range(3):
            mean = stats[i, c, 0]
            std = stats[i, c, 1]
            result[i, :, :, c] = arr[i, :, :, c] * std + mean
    return result


def make_coord_channels_batch(dim: int, lx_arr, ly_arr) -> np.ndarray:
    """
    Generate normalized coordinate channels for a batch of samples.

    Encoding: x̄ = x / max(lx, ly),  ȳ = y / max(lx, ly)  (aspect-ratio preserving)

    Args:
        dim: Grid dimension (square grid dim×dim)
        lx_arr: List/array of lx values, one per sample
        ly_arr: List/array of ly values, one per sample
    Returns:
        (N, dim, dim, 2) array of coordinate channels [x̄, ȳ]
    """
    N = len(lx_arr)
    coords = np.zeros((N, dim, dim, 2), dtype=np.float32)
    for i in range(N):
        lx = lx_arr[i]
        ly = ly_arr[i]
        L = max(lx, ly)
        x = np.linspace(0, lx, dim) / L
        y = np.linspace(0, ly, dim) / L
        xx, yy = np.meshgrid(x, y)  # both (dim, dim)
        coords[i, :, :, 0] = xx
        coords[i, :, :, 1] = yy
    return coords


def reshape_rectangular_to_square(fields: Dict[str, np.ndarray], 
                                  nx_rect: int, ny_rect: int,
                                  lx: float, ly: float) -> Dict[str, np.ndarray]:
    """
    Resample rectangular grid data to square grid for ML model input.
    
    This is needed when testing a model trained on square geometry (e.g., lid-driven cavity)
    on rectangular geometry (e.g., BFS). The model expects square aspect ratio data.
    
    Args:
        fields: Dictionary with 'u', 'v', 'p' fields of shape (ny_rect, nx_rect)
        nx_rect, ny_rect: Rectangular grid dimensions
        lx, ly: Physical domain dimensions
        
    Returns:
        Dictionary with resampled fields in square coordinate system
    """
    print(f"  Resampling rectangular ({nx_rect}×{ny_rect}) → square ({nx_rect}×{nx_rect})...")
    print(f"  Physical domain: {lx}×{ly} (aspect ratio: {lx/ly:.2f}:1)")
    
    # Create coordinate systems
    x_rect = np.linspace(0, lx, nx_rect)
    y_rect = np.linspace(0, ly, ny_rect)
    
    # Square coordinate system (use max dimension)
    L_square = max(lx, ly)
    x_square = np.linspace(0, L_square, nx_rect)
    y_square = np.linspace(0, L_square, nx_rect)
    
    fields_square = {}
    for component in ['u', 'v', 'p']:
        field_rect = fields[component]  # Shape: (ny_rect, nx_rect)
        
        # Create interpolator
        interpolator = interpolate.RectBivariateSpline(y_rect, x_rect, field_rect, kx=3, ky=3)
        
        # Resample to square grid
        field_square = interpolator(y_square, x_square)
        
        fields_square[component] = field_square
        print(f"    {component.upper()}: {field_rect.shape} → {field_square.shape}")
    
    return fields_square


def reshape_square_to_rectangular(fields: Dict[str, np.ndarray],
                                  nx_rect: int, ny_rect: int,
                                  lx: float, ly: float) -> Dict[str, np.ndarray]:
    """
    Resample square grid data back to rectangular grid after ML prediction.
    
    Args:
        fields: Dictionary with 'u', 'v', 'p' fields of shape (nx_square, nx_square)
        nx_rect, ny_rect: Target rectangular grid dimensions
        lx, ly: Physical domain dimensions
        
    Returns:
        Dictionary with fields resampled to rectangular coordinate system
    """
    # Assume input is square
    nx_square = fields['u'].shape[0]
    
    print(f"  Resampling square ({nx_square}×{nx_square}) → rectangular ({nx_rect}×{ny_rect})...")
    print(f"  Physical domain: {lx}×{ly} (aspect ratio: {lx/ly:.2f}:1)")
    
    # Create coordinate systems
    L_square = max(lx, ly)
    x_square = np.linspace(0, L_square, nx_square)
    y_square = np.linspace(0, L_square, nx_square)
    
    x_rect = np.linspace(0, lx, nx_rect)
    y_rect = np.linspace(0, ly, ny_rect)
    
    fields_rect = {}
    for component in ['u', 'v', 'p']:
        field_square = fields[component]  # Shape: (nx_square, nx_square)
        
        # Create interpolator
        interpolator = interpolate.RectBivariateSpline(y_square, x_square, field_square, kx=3, ky=3)
        
        # Resample to rectangular grid
        field_rect = interpolator(y_rect, x_rect)
        
        fields_rect[component] = field_rect
        print(f"    {component.upper()}: {field_square.shape} → {field_rect.shape}")
    
    return fields_rect


# ==============================================================================
# Import all CFD-related code from the AE version (unchanged)
# ==============================================================================

@dataclass
class BoundaryCondition:
    """Class to define boundary conditions"""
    type: str  # 'dirichlet' or 'neumann'
    value: float = 0.0

class BoundaryConditions:
    """Container for all boundary conditions"""
    def __init__(self):
        self.u_boundaries = {
            'left': BoundaryCondition('dirichlet', 1.0),
            'right': BoundaryCondition('dirichlet', 0.0),
            'top': BoundaryCondition('dirichlet', 0.0),
            'bottom': BoundaryCondition('dirichlet', 0.0)
        }
        self.v_boundaries = {
            'left': BoundaryCondition('dirichlet', 0.0),
            'right': BoundaryCondition('dirichlet', 0.0),
            'top': BoundaryCondition('dirichlet', 0.0),
            'bottom': BoundaryCondition('dirichlet', 0.0)
        }
        self.p_boundaries = {
            'left': BoundaryCondition('neumann', 0.0),
            'right': BoundaryCondition('neumann', 0.0),
            'top': BoundaryCondition('neumann', 0.0),
            'bottom': BoundaryCondition('neumann', 0.0)
        }

class MeshParameters:
    """Class to handle mesh parameters"""
    def __init__(self, nx: int = 100, ny: int = 100, lx: float = 10.0, ly: float = 3.0):
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.dx = lx / nx
        self.dy = ly / ny
        self.volp = self.dx * self.dy

class FluidProperties:
    """Class to handle fluid properties"""
    def __init__(self, Re: float = 100.0, rho: float = 1.0):
        self.Re = Re
        self.rho = rho
        self.nu = 1.0 / Re  # kinematic viscosity

class SolverSettings:
    """Class to handle solver settings"""
    def __init__(self, dt: float = 0.001, max_iterations: int = 100000,
                 convergence_criteria: Dict[str, float] = None,
                 scheme: str = 'UPWIND',
                 relaxation_factors: Dict[str, float] = None):
        self.dt = dt
        self.max_iterations = max_iterations
        self.scheme = scheme  # 'QUICK' or 'UPWIND'
        
        if convergence_criteria is None:
            self.convergence_criteria = {
                'u': 1e-6,
                'v': 1e-6,
                'p': 1e-6,
                'continuity': 1e-6
            }
        else:
            self.convergence_criteria = convergence_criteria

        # Under-relaxation factors (important for BFS stability)
        if relaxation_factors is None:
            self.relaxation_factors = {
                'u': 0.5,
                'v': 0.5,
                'p': 0.2
            }
        else:
            self.relaxation_factors = relaxation_factors


# ==============================================================================
# Numba-compiled functions for performance
# ==============================================================================

@njit
def copy_new_to_old(Var, VarOld, nVar, Nx, Ny):
    for k in range(nVar):
        for i in range(Nx + 2):
            for j in range(Ny + 2):
                VarOld[k, i, j] = Var[k, i, j]

@njit
def apply_bc_configured(Var, k, Nx, Ny, bc_types, bc_values):
    """Apply boundary conditions based on configuration
    bc_types: array of ints [left, right, top, bottom] where 0=dirichlet, 1=neumann
    bc_values: array of floats [left, right, top, bottom] with boundary values
    """
    # Left and Right boundaries
    for j in range(1, Ny + 1):
        if bc_types[0] == 0:  # Dirichlet
            Var[k, 0, j] = 2 * bc_values[0] - Var[k, 1, j]
        else:  # Neumann
            Var[k, 0, j] = Var[k, 1, j]
        
        if bc_types[1] == 0:  # Dirichlet
            Var[k, Nx + 1, j] = 2 * bc_values[1] - Var[k, Nx, j]
        else:  # Neumann
            Var[k, Nx + 1, j] = Var[k, Nx, j]
    
    # Top and Bottom boundaries
    for i in range(1, Nx + 1):
        if bc_types[2] == 0:  # Dirichlet
            Var[k, i, Ny + 1] = 2 * bc_values[2] - Var[k, i, Ny]
        else:  # Neumann
            Var[k, i, Ny + 1] = Var[k, i, Ny]
        
        if bc_types[3] == 0:  # Dirichlet
            Var[k, i, 0] = 2 * bc_values[3] - Var[k, i, 1]
        else:  # Neumann
            Var[k, i, 0] = Var[k, i, 1]

@njit
def linear_interpolation(Var, Ff, Nx, Ny, dx, dy):
    for i in range(1, Nx + 1):
        for j in range(1, Ny + 1):
            Ff[0, i, j] = (Var[0, i, j] + Var[0, i + 1, j]) * dy * 0.5  # East Face
            Ff[1, i, j] = (Var[1, i, j] + Var[1, i, j + 1]) * dx * 0.5  # North Face
            Ff[2, i, j] = -(Var[0, i, j] + Var[0, i - 1, j]) * dy * 0.5  # West Face
            Ff[3, i, j] = -(Var[1, i, j] + Var[1, i, j - 1]) * dx * 0.5  # South Face

@njit
def simple_upwind(Var, Ff, k, i, j, volp):
    ue, uw, un, us = 0.0, 0.0, 0.0, 0.0
    sum_flux = 0.0
    
    if Ff[0, i, j] >= 0:
        ue = Var[k, i, j]
        sum_flux += Ff[0, i, j]
    else:
        ue = Var[k, i + 1, j]
    
    if Ff[2, i, j] >= 0:
        uw = Var[k, i, j]
        sum_flux += Ff[2, i, j]
    else:
        uw = Var[k, i - 1, j]
    
    if Ff[1, i, j] >= 0:
        un = Var[k, i, j]
        sum_flux += Ff[1, i, j]
    else:
        un = Var[k, i, j + 1]
    
    if Ff[3, i, j] >= 0:
        us = Var[k, i, j]
        sum_flux += Ff[3, i, j]
    else:
        us = Var[k, i, j - 1]
    
    Fc = ue * Ff[0, i, j] + uw * Ff[2, i, j] + un * Ff[1, i, j] + us * Ff[3, i, j]
    ap_c = sum_flux * volp
    
    return Fc, ap_c

@njit
def quick_scheme(Var, Ff, k, i, j, volp):
    ue, uw, un, us = 0.0, 0.0, 0.0, 0.0
    sum_flux = 0.0
    
    # East face
    if Ff[0, i, j] >= 0:
        ue = 0.75 * Var[k, i, j] + 0.375 * Var[k, i + 1, j] - 0.125 * Var[k, i - 1, j]
        sum_flux += 0.75 * Ff[0, i, j]
    else:
        ue = 0.75 * Var[k, i + 1, j] + 0.375 * Var[k, i, j] - 0.125 * Var[k, i + 2, j]
        sum_flux += 0.375 * Ff[0, i, j]
    
    # West face
    if Ff[2, i, j] >= 0:
        uw = 0.75 * Var[k, i, j] + 0.375 * Var[k, i - 1, j] - 0.125 * Var[k, i + 1, j]
        sum_flux += 0.75 * Ff[2, i, j]
    else:
        uw = 0.75 * Var[k, i - 1, j] + 0.375 * Var[k, i, j] - 0.125 * Var[k, i - 2, j]
        sum_flux += 0.375 * Ff[2, i, j]
    
    # North face
    if Ff[1, i, j] >= 0:
        un = 0.75 * Var[k, i, j] + 0.375 * Var[k, i, j + 1] - 0.125 * Var[k, i, j - 1]
        sum_flux += 0.75 * Ff[1, i, j]
    else:
        un = 0.75 * Var[k, i, j + 1] + 0.375 * Var[k, i, j] - 0.125 * Var[k, i, j + 2]
        sum_flux += 0.375 * Ff[1, i, j]
    
    # South face
    if Ff[3, i, j] >= 0:
        us = 0.75 * Var[k, i, j] + 0.375 * Var[k, i, j - 1] - 0.125 * Var[k, i, j + 1]
        sum_flux += 0.75 * Ff[3, i, j]
    else:
        us = 0.75 * Var[k, i, j - 1] + 0.375 * Var[k, i, j] - 0.125 * Var[k, i, j - 2]
        sum_flux += 0.375 * Ff[3, i, j]
    
    Fc = ue * Ff[0, i, j] + uw * Ff[2, i, j] + un * Ff[1, i, j] + us * Ff[3, i, j]
    ap_c = sum_flux * volp
    
    return Fc, ap_c

@njit
def diffusive_flux(Var, k, i, j, dx, dy, volp):
    Fd = volp * ((Var[k, i + 1, j] - 2.0 * Var[k, i, j] + Var[k, i - 1, j]) / (dx * dx) +
                 (Var[k, i, j + 1] - 2.0 * Var[k, i, j] + Var[k, i, j - 1]) / (dy * dy))
    ap_d = -volp * (2.0 / (dx * dx) + 2.0 / (dy * dy))
    return Fd, ap_d

@njit
def update_flux(Var, Ff, dt, rho, Nx, Ny, dx, dy):
    for i in range(1, Nx + 1):
        for j in range(1, Ny + 1):
            Ff[0, i, j] += -dt / rho * (Var[2, i + 1, j] - Var[2, i, j]) * dy / dx
            Ff[1, i, j] += -dt / rho * (Var[2, i, j + 1] - Var[2, i, j]) * dx / dy
            Ff[2, i, j] += -dt / rho * (Var[2, i - 1, j] - Var[2, i, j]) * dy / dx
            Ff[3, i, j] += -dt / rho * (Var[2, i, j - 1] - Var[2, i, j]) * dx / dy

@njit
def under_relax_field(Var, VarOld, k, Nx, Ny, alpha):
    for i in range(1, Nx + 1):
        for j in range(1, Ny + 1):
            Var[k, i, j] = VarOld[k, i, j] + alpha * (Var[k, i, j] - VarOld[k, i, j])

@njit(parallel=True)
def solve_momentum_quick(Var, VarOld, Ff, k, Nx, Ny, dx, dy, dt, nu, volp):
    tolerance = 1e-6
    max_iter = 1000
    
    for iter in range(max_iter):
        rms = 0.0
        for i in prange(1, Nx + 1):
            for j in range(1, Ny + 1):
                Fc, ap_c = quick_scheme(Var, Ff, k, i, j, volp)
                Fd, ap_d = diffusive_flux(Var, k, i, j, dx, dy, volp)
                
                R = -(volp / dt * (Var[k, i, j] - VarOld[k, i, j]) + Fc + (-nu) * Fd)
                ap = volp / dt + ap_c + (-nu) * ap_d
                
                Var[k, i, j] = Var[k, i, j] + R / ap
                rms += R * R
        
        rms = np.sqrt(rms / (Nx * Ny))
        if rms < tolerance:
            break

@njit(parallel=True)
def solve_momentum_upwind(Var, VarOld, Ff, k, Nx, Ny, dx, dy, dt, nu, volp):
    tolerance = 1e-6
    max_iter = 1000
    
    for iter in range(max_iter):
        rms = 0.0
        for i in prange(1, Nx + 1):
            for j in range(1, Ny + 1):
                Fc, ap_c = simple_upwind(Var, Ff, k, i, j, volp)
                Fd, ap_d = diffusive_flux(Var, k, i, j, dx, dy, volp)
                
                R = -(volp / dt * (Var[k, i, j] - VarOld[k, i, j]) + Fc + (-nu) * Fd)
                ap = volp / dt + ap_c + (-nu) * ap_d
                
                Var[k, i, j] = Var[k, i, j] + R / ap
                rms += R * R
        
        rms = np.sqrt(rms / (Nx * Ny))
        if rms < tolerance:
            break

@njit(parallel=True)
def solve_pressure(Var, Ff, Nx, Ny, dx, dy, dt, rho, volp):
    tolerance = 1e-6
    max_iter = 1000
    k = 2  # Pressure
    
    for iter in range(max_iter):
        rms = 0.0
        for i in prange(1, Nx + 1):
            for j in range(1, Ny + 1):
                Fd, ap_d = diffusive_flux(Var, k, i, j, dx, dy, volp)
                
                LHS = Fd
                RHS = rho / dt * (Ff[0, i, j] + Ff[1, i, j] + Ff[2, i, j] + Ff[3, i, j])
                R = RHS - LHS
                ap = ap_d
                
                Var[k, i, j] = Var[k, i, j] + R / ap
                rms += R * R
        
        rms = np.sqrt(rms / (Nx * Ny))
        if rms < tolerance:
            break

@njit(parallel=True)
def correct_velocity(Var, VarOld, dt, rho, Nx, Ny, dx, dy):
    res_u = 0.0
    res_v = 0.0
    res_p = 0.0
    for i in prange(1, Nx + 1):
        for j in range(1, Ny + 1):
            # U velocity correction
            Var[0, i, j] = Var[0, i, j] - dt / rho * (Var[2, i + 1, j] - Var[2, i - 1, j]) / (2 * dx)
            # V velocity correction
            Var[1, i, j] = Var[1, i, j] - dt / rho * (Var[2, i, j + 1] - Var[2, i, j - 1]) / (2 * dy)

            # Accumulate residuals locally
            du = Var[0, i, j] - VarOld[0, i, j]
            dv = Var[1, i, j] - VarOld[1, i, j]
            dp = Var[2, i, j] - VarOld[2, i, j]
            res_u += du * du
            res_v += dv * dv
            res_p += dp * dp
    return res_u, res_v, res_p


# ==============================================================================
# CFDSolver Class with BFS Support (unchanged from AE version)
# ==============================================================================

class CFDSolver:
    """Main CFD Solver class with BFS-specific features"""
    def __init__(self, mesh: MeshParameters, fluid: FluidProperties,
                 solver_settings: SolverSettings, bc: BoundaryConditions,
                 step_height: float = 1.0, h: float = 2.0, Ub: float = 1.0):
        self.mesh = mesh
        self.fluid = fluid
        self.settings = solver_settings
        self.bc = bc
        
        # BFS-specific parameters
        self.case_type = 'BFS'
        self.step_height = step_height
        self.h = h
        self.Ub = Ub
        
        # Solution variables
        self.nVar = 3
        self.Var = np.zeros((self.nVar, mesh.nx + 2, mesh.ny + 2))
        self.VarOld = np.zeros((self.nVar, mesh.nx + 2, mesh.ny + 2))
        self.residual = np.zeros(self.nVar)
        self.Ff = np.zeros((4, mesh.nx + 2, mesh.ny + 2))  # Face fluxes
        self.residual_history = {'u': [], 'v': [], 'p': []}
        
        # Initialize fields
        self._initialize_fields()
    
    def _get_bc_arrays(self, k: int):
        """Convert boundary condition dictionaries to arrays for Numba functions"""
        if k == 0:  # U velocity
            bc_dict = self.bc.u_boundaries
        elif k == 1:  # V velocity
            bc_dict = self.bc.v_boundaries
        else:  # Pressure
            bc_dict = self.bc.p_boundaries
        
        # Convert to arrays: [left, right, top, bottom]
        bc_types = np.array([
            0 if bc_dict['left'].type == 'dirichlet' else 1,
            0 if bc_dict['right'].type == 'dirichlet' else 1,
            0 if bc_dict['top'].type == 'dirichlet' else 1,
            0 if bc_dict['bottom'].type == 'dirichlet' else 1
        ], dtype=np.int32)
        
        bc_values = np.array([
            bc_dict['left'].value,
            bc_dict['right'].value,
            bc_dict['top'].value,
            bc_dict['bottom'].value
        ], dtype=np.float64)
        
        return bc_types, bc_values
    
    def _apply_bfs_inlet(self, k: int):
        """Apply BFS inlet/wall mixture on the left boundary.
        - For y < step_height: enforce wall (Dirichlet 0) via ghost cell reflection
        - For y >= step_height: apply parabolic U inlet, V = 0
        Only active when self.case_type == 'BFS'.
        """
        if self.case_type != 'BFS':
            return
        if k not in (0, 1):
            return
        ny = self.mesh.ny
        dy = self.mesh.dy
        step_h = self.step_height
        h = self.h
        Ub = self.Ub

        for j in range(1, ny + 1):
            y = (j - 0.5) * dy
            if y < step_h:
                # Inlet blocked by the step: no-slip wall at x=0
                # Enforce Dirichlet(0) using ghost reflection
                self.Var[k, 0, j] = -self.Var[k, 1, j]
            else:
                # Open inlet part
                if k == 1:
                    # V = 0 across inlet
                    self.Var[1, 0, j] = -self.Var[1, 1, j]
                else:
                    # Parabolic U profile over height h above the step
                    yprime = y - step_h
                    # Clamp within [0, h]
                    if yprime < 0.0:
                        yprime = 0.0
                    if yprime > h:
                        yprime = h
                    u_in = 6.0 * Ub * (yprime / h) * (1.0 - (yprime / h))
                    self.Var[0, 0, j] = 2.0 * u_in - self.Var[0, 1, j]
                    # Also ensure V ghost enforces v=0 consistently
                    self.Var[1, 0, j] = -self.Var[1, 1, j]
    
    def _apply_bc_wrapper(self, k: int):
        """Wrapper to apply boundary conditions based on settings"""
        bc_types, bc_values = self._get_bc_arrays(k)
        apply_bc_configured(self.Var, k, self.mesh.nx, self.mesh.ny, bc_types, bc_values)
        # Override left boundary for BFS inlet/wall mix
        self._apply_bfs_inlet(k)
    
    def _initialize_fields(self):
        """Initialize all fields to zero"""
        self.Var.fill(0.0)
        self.VarOld.fill(0.0)
        self.Ff.fill(0.0)
        
        # Apply boundary conditions
        for k in range(self.nVar):
            self._apply_bc_wrapper(k)
        
        copy_new_to_old(self.Var, self.VarOld, self.nVar, self.mesh.nx, self.mesh.ny)
        linear_interpolation(self.Var, self.Ff, self.mesh.nx, self.mesh.ny, 
                           self.mesh.dx, self.mesh.dy)
    
    def solve(self, output_base_name: str = "output", verbose: bool = True,
              callback=None, callback_interval: int = 1000):
        """Main solver loop"""
        count = 0
        converged = False
        start_time = time.time()
        
        if verbose:
            print(f"Starting BFS simulation with Re={self.fluid.Re}, mesh={self.mesh.nx}x{self.mesh.ny}")
            print(f"Time step: {self.settings.dt}, Scheme: {self.settings.scheme}")
            print(f"Step height: {self.step_height}, Channel height: {self.h}")
            print("\nIteration\tU-RMS\t\tV-RMS\t\tP-RMS")
            print("-" * 60)
        
        while not converged and count < self.settings.max_iterations:
            count += 1
            self._implicit_solve()
            
            # Execute callback if provided
            if callback is not None and count % callback_interval == 0:
                callback(self, count)
            
            if verbose and count % 100 == 0:
                print(f"{count}", end="")
            
            converged, rms_residuals = self._convergence_check(verbose and count % 100 == 0)
            if count % 100 == 0:
                self.residual_history['u'].append(rms_residuals[0])
                self.residual_history['v'].append(rms_residuals[1])
                self.residual_history['p'].append(rms_residuals[2])
        
        # Final callback execution if not already done
        if callback is not None and (count % callback_interval != 0):
             callback(self, count)

        end_time = time.time()
        
        if verbose:
            print(f"\n\nBFS simulation completed in {end_time - start_time:.2f} seconds")
            print(f"Total iterations: {count}")
        
        # Save results
        self._save_results(output_base_name)
        
        return count, end_time - start_time
    
    def _implicit_solve(self):
        """Implicit solver step using SIMPLE algorithm with under-relaxation"""
        self.residual.fill(0.0)

        # Fetch under-relaxation factors
        alpha_u = self.settings.relaxation_factors.get('u', 0.5)
        alpha_v = self.settings.relaxation_factors.get('v', 0.5)
        alpha_p = self.settings.relaxation_factors.get('p', 0.2)
        
        # Solve momentum equations (U and V)
        for k in range(2):
            if self.settings.scheme == 'QUICK':
                solve_momentum_quick(self.Var, self.VarOld, self.Ff, k, self.mesh.nx, 
                                   self.mesh.ny, self.mesh.dx, self.mesh.dy, 
                                   self.settings.dt, self.fluid.nu, self.mesh.volp)
            else:  # UPWIND
                solve_momentum_upwind(self.Var, self.VarOld, self.Ff, k, self.mesh.nx, 
                                    self.mesh.ny, self.mesh.dx, self.mesh.dy, 
                                    self.settings.dt, self.fluid.nu, self.mesh.volp)
            
            # Under-relax U and V
            if k == 0:
                under_relax_field(self.Var, self.VarOld, 0, self.mesh.nx, self.mesh.ny, alpha_u)
            else:
                under_relax_field(self.Var, self.VarOld, 1, self.mesh.nx, self.mesh.ny, alpha_v)
            
            self._apply_bc_wrapper(k)
        
        linear_interpolation(self.Var, self.Ff, self.mesh.nx, self.mesh.ny, 
                           self.mesh.dx, self.mesh.dy)
        
        # Solve pressure equation
        solve_pressure(self.Var, self.Ff, self.mesh.nx, self.mesh.ny, 
                      self.mesh.dx, self.mesh.dy, self.settings.dt, 
                      self.fluid.rho, self.mesh.volp)
        
        # Under-relax pressure before correction
        under_relax_field(self.Var, self.VarOld, 2, self.mesh.nx, self.mesh.ny, alpha_p)
        self._apply_bc_wrapper(2)
        
        # Correct velocities and compute residuals
        res_u, res_v, res_p = correct_velocity(self.Var, self.VarOld, self.settings.dt, self.fluid.rho, 
                                               self.mesh.nx, self.mesh.ny, self.mesh.dx, self.mesh.dy)
        self.residual[0] = res_u
        self.residual[1] = res_v
        self.residual[2] = res_p
        
        self._apply_bc_wrapper(0)
        self._apply_bc_wrapper(1)
        
        update_flux(self.Var, self.Ff, self.settings.dt, self.fluid.rho, 
                   self.mesh.nx, self.mesh.ny, self.mesh.dx, self.mesh.dy)
    
    def _convergence_check(self, print_residuals: bool = False) -> Tuple[bool, np.ndarray]:
        """Check convergence based on residuals"""
        rms = np.zeros(self.nVar)
        for k in range(self.nVar):
            rms[k] = np.sqrt(self.residual[k] / (self.mesh.nx * self.mesh.ny))
            rms[k] = rms[k] / self.settings.dt
            if print_residuals:
                print(f"\t{rms[k]:.6e}", end="")
        
        if print_residuals:
            print()
        
        # Check for NaN or Inf in residuals
        if np.isnan(rms).any() or np.isinf(rms).any():
            print(f"\n❌ ERROR: NaN or Inf detected in residuals!")
            print(f"   U-residual: {rms[0]:.6e}, V-residual: {rms[1]:.6e}, P-residual: {rms[2]:.6e}")
            print(f"   This indicates solver instability or bad initial conditions.")
            raise ValueError("Solver failed: NaN/Inf in residuals")
        
        # Check convergence criteria
        converged = True
        if rms[0] > self.settings.convergence_criteria['u']:
            converged = False
        if rms[1] > self.settings.convergence_criteria['v']:
            converged = False
        if rms[2] > self.settings.convergence_criteria['p']:
            converged = False
        
        if not converged:
            copy_new_to_old(self.Var, self.VarOld, self.nVar, self.mesh.nx, self.mesh.ny)
        
        return converged, rms
    
    def _save_results(self, output_base_name: str):
        """Save all results"""
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_base_name)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save to HDF5
        group_name = f"Re{self.fluid.Re}_mesh{self.mesh.nx}x{self.mesh.ny}"
        self._save_results_hdf5(f"{output_base_name}.h5", group_name)
        
        # Generate plots
        self._generate_plots(output_base_name)
    
    def _save_results_hdf5(self, filename: str, group_name: str):
        """Save results to an HDF5 file."""
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with h5py.File(filename, 'a') as f:
            if group_name in f:
                del f[group_name]
            
            grp = f.create_group(group_name)
            
            grp.attrs["case_name"] = "backward facing step"
            grp.attrs["reynolds_number"] = self.fluid.Re
            grp.attrs["nx"] = self.mesh.nx
            grp.attrs["ny"] = self.mesh.ny
            grp.attrs["lx"] = self.mesh.lx
            grp.attrs["ly"] = self.mesh.ly
            grp.attrs["step_height"] = self.step_height
            grp.attrs["total_points"] = self.mesh.nx * self.mesh.ny
            
            x = np.linspace(0, self.mesh.lx, self.mesh.nx)
            y = np.linspace(0, self.mesh.ly, self.mesh.ny)
            
            X, Y = np.meshgrid(x, y)
            
            grp.create_dataset("x", data=X.flatten())
            grp.create_dataset("y", data=Y.flatten())
            grp.create_dataset("u", data=self.Var[0, 1:-1, 1:-1].T.flatten())
            grp.create_dataset("v", data=self.Var[1, 1:-1, 1:-1].T.flatten())
            grp.create_dataset("p", data=self.Var[2, 1:-1, 1:-1].T.flatten())
    
    def _generate_plots(self, output_base_name: str):
        """Generate visualization plots"""
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_base_name)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Centerline plots
        self._plot_centerlines(f"{output_base_name}_centerlines.png")
        
        # Contour plots
        self._plot_contours(f"{output_base_name}_contours.png")

        # Convergence plot
        self._plot_convergence(f"{output_base_name}_convergence.png")
    
    def _plot_centerlines(self, filename: str):
        """Plot centerline velocity profiles"""
        # Extract centerline data (at y = step_height + h/2)
        y_centerline_idx = int((self.step_height + self.h/2) / self.mesh.dy)
        y_centerline_idx = min(max(y_centerline_idx, 0), self.mesh.ny - 1)
        
        u_centerline = self.Var[0, 1:-1, y_centerline_idx + 1]
        v_centerline = self.Var[1, 1:-1, y_centerline_idx + 1]
        x = np.linspace(0, self.mesh.lx, self.mesh.nx)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(x, u_centerline, 'b-', linewidth=2)
        ax1.set_xlabel('X Position', fontsize=12)
        ax1.set_ylabel('U Velocity', fontsize=12)
        ax1.set_title(f'U velocity along horizontal centerline\n(Re={self.fluid.Re}, {self.mesh.nx}x{self.mesh.ny})', fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(x, v_centerline, 'r-', linewidth=2)
        ax2.set_xlabel('X Position', fontsize=12)
        ax2.set_ylabel('V Velocity', fontsize=12)
        ax2.set_title(f'V velocity along horizontal centerline\n(Re={self.fluid.Re}, {self.mesh.nx}x{self.mesh.ny})', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_contours(self, filename: str):
        """Plot contour plots of all variables"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 8))
        
        # Create meshgrid for plotting
        x = np.linspace(0, self.mesh.lx, self.mesh.nx)
        y = np.linspace(0, self.mesh.ly, self.mesh.ny)
        X, Y = np.meshgrid(x, y)
        
        # U velocity contour
        im1 = axes[0, 0].contourf(X, Y, self.Var[0, 1:-1, 1:-1].T, levels=20, cmap='RdBu')
        axes[0, 0].set_title('U Velocity')
        axes[0, 0].set_xlabel('X')
        axes[0, 0].set_ylabel('Y')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # V velocity contour
        im2 = axes[0, 1].contourf(X, Y, self.Var[1, 1:-1, 1:-1].T, levels=20, cmap='RdBu')
        axes[0, 1].set_title('V Velocity')
        axes[0, 1].set_xlabel('X')
        axes[0, 1].set_ylabel('Y')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Pressure contour
        im3 = axes[1, 0].contourf(X, Y, self.Var[2, 1:-1, 1:-1].T, levels=20, cmap='viridis')
        axes[1, 0].set_title('Pressure')
        axes[1, 0].set_xlabel('X')
        axes[1, 0].set_ylabel('Y')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Velocity magnitude and streamlines
        u_mag = np.sqrt(self.Var[0, 1:-1, 1:-1]**2 + self.Var[1, 1:-1, 1:-1]**2)
        im4 = axes[1, 1].contourf(X, Y, u_mag.T, levels=20, cmap='plasma')
        axes[1, 1].set_title('Velocity Magnitude with Streamlines')
        axes[1, 1].set_xlabel('X')
        axes[1, 1].set_ylabel('Y')
        axes[1, 1].set_aspect('equal')
        plt.colorbar(im4, ax=axes[1, 1])
        
        # Add streamlines
        axes[1, 1].streamplot(X, Y, self.Var[0, 1:-1, 1:-1].T, 
                             self.Var[1, 1:-1, 1:-1].T, 
                             color='white', linewidth=0.35, density=1.5)
        
        plt.suptitle(f'Backward-Facing Step Flow (Re={self.fluid.Re})', fontsize=16)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
    
    def _plot_convergence(self, filename: str):
        """Plot convergence history"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(100, 100 * (len(self.residual_history['u']) + 1), 100)
        
        ax.plot(iterations, self.residual_history['u'], 'b-o', label='U-velocity')
        ax.plot(iterations, self.residual_history['v'], 'r-s', label='V-velocity')
        ax.plot(iterations, self.residual_history['p'], 'g-^', label='Pressure')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMS Residual')
        ax.set_yscale('log')
        ax.set_title(f'Convergence History - BFS (Re={self.fluid.Re})')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()


# ==============================================================================
# ML-Accelerated CFD Workflow Functions - Coarse Simulation (unchanged)
# ==============================================================================

def run_coarse_simulation(Re: float, lr_dim: int = 10, 
                         dt: float = 0.002, scheme: str = 'UPWIND',
                         convergence_criteria: Dict[str, float] = None,
                         max_iterations: int = 100000,
                         output_dir: str = None,
                         bc: Optional[BoundaryConditions] = None,
                         step_height: float = 1.0,
                         h: float = 2.0,
                         Ub: float = 1.0,
                         lx: float = 10.0,
                         ly: float = 3.0,
                         relaxation_factors: Dict[str, float] = None) -> Dict[str, np.ndarray]:
    """
    Step 1: Run a coarse (10x10) BFS CFD simulation
    
    Args:
        Re: Reynolds number
        lr_dim: Low resolution dimension (default: 10)
        dt: Time step
        scheme: Numerical scheme
        convergence_criteria: Convergence criteria dict
        max_iterations: Maximum iterations
        output_dir: Directory to save outputs. If None, creates timestamped directory.
        bc: BoundaryConditions object. If None, uses default BFS BCs.
        step_height: BFS step height
        h: BFS channel height above step
        Ub: BFS bulk velocity
        lx: Domain length in x
        ly: Domain length in y
        relaxation_factors: Under-relaxation factors
    
    Returns:
        Tuple of (coarse_fields, iterations, time_elapsed)
        - coarse_fields: Dictionary with 'u', 'v', 'p' fields of shape (lr_dim, lr_dim)
        - iterations: Number of iterations completed
        - time_elapsed: Time taken in seconds
    """
    print(f"\n{'='*70}")
    print(f"STEP 1: Running Coarse BFS Simulation (Re={Re}, mesh={lr_dim}x{lr_dim})")
    print(f"{'='*70}")
    
    # Create mesh for coarse simulation
    mesh = MeshParameters(nx=lr_dim, ny=lr_dim, lx=lx, ly=ly)
    fluid = FluidProperties(Re=Re, rho=1.0)
    
    if convergence_criteria is None:
        convergence_criteria = {'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6}
    
    solver_settings = SolverSettings(dt=dt, scheme=scheme, 
                                   max_iterations=max_iterations,
                                   convergence_criteria=convergence_criteria,
                                   relaxation_factors=relaxation_factors)
    
    # Use provided boundary conditions or create default BFS BCs
    if bc is None:
        bc = BoundaryConditions()
        # Default BFS boundary conditions
        bc.u_boundaries['right'] = BoundaryCondition('neumann', 0.0)
        bc.v_boundaries['right'] = BoundaryCondition('neumann', 0.0)
        bc.p_boundaries['right'] = BoundaryCondition('dirichlet', 0.0)
    
    # Create solver and run
    solver = CFDSolver(mesh, fluid, solver_settings, bc, 
                      step_height=step_height, h=h, Ub=Ub)
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = create_timestamped_output_dir()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir, f"bfs_coarse_Re{Re}_{lr_dim}x{lr_dim}_{max_iterations}_coarse_iterations")
    
    print(f"Saving coarse simulation output to: {output_dir}")
    
    iterations, time_elapsed = solver.solve(output_name, verbose=True)
    
    print(f"Coarse BFS simulation completed in {iterations} iterations ({time_elapsed:.2f} seconds)")
    
    # Extract the solution fields (internal cells only, no ghost cells)
    coarse_fields = {
        'u': solver.Var[0, 1:-1, 1:-1].T.copy(),  # Shape: (lr_dim, lr_dim)
        'v': solver.Var[1, 1:-1, 1:-1].T.copy(),  # Shape: (lr_dim, lr_dim)
        'p': solver.Var[2, 1:-1, 1:-1].T.copy(),  # Shape: (lr_dim, lr_dim)
    }
    
    return coarse_fields, iterations, time_elapsed


# ==============================================================================
# ML Super-Resolution - MODIFIED FOR CASCADED U-NET
# ==============================================================================

def ml_super_resolution_unet(coarse_fields: Dict[str, np.ndarray], 
                             lr_dim: int, hr_dim: int,
                             stage_dims: List[int],
                             model_name_pattern: str,
                             lx: float = 1.0, ly: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Step 2: Use cascaded U-Net models to super-resolve coarse simulation to fine resolution.

    Uses per-sample Z-score normalization and 5-channel input [u, v, p, x̄, ȳ] where
    coordinate channels encode the physical geometry (aspect-ratio preserving).

    Cascaded inference: lr_dim → stage_dims[-1] (e.g. 10→20→40→80→200→400)

    Args:
        coarse_fields: Dictionary with 'u', 'v', 'p' arrays of shape (lr_dim, lr_dim)
        lr_dim: Low resolution dimension (e.g., 10)
        hr_dim: High resolution dimension (e.g., 400)
        stage_dims: List of intermediate dimensions (e.g., [20, 40, 80, 200, 400])
        model_name_pattern: Pattern for stage model filenames (e.g., "unet_stage_{from_dim}to{to_dim}_NAME.h5")
        lx, ly: Physical domain dimensions (used for coordinate channel encoding)

    Returns:
        Dictionary with 'u', 'v', 'p' fields of shape (hr_dim, hr_dim)
    """
    print(f"\n{'='*70}")
    print(f"STEP 2: ML Super-Resolution - Progressive U-Net ({lr_dim}x{lr_dim} -> {hr_dim}x{hr_dim})")
    print(f"  Cascaded stages: {lr_dim} → {' → '.join(map(str, stage_dims))}")
    print(f"  Per-sample normalization: ENABLED")
    print(f"  Coordinate channels: ENABLED  (lx={lx}, ly={ly})")
    print(f"{'='*70}")

    # Stack fields into (1, lr_dim, lr_dim, 3) — batch of one sample
    x_lr_3ch = np.stack([
        coarse_fields['u'],
        coarse_fields['v'],
        coarse_fields['p']
    ], axis=-1).astype(np.float32)  # (lr_dim, lr_dim, 3)
    x_lr_batch = x_lr_3ch[np.newaxis]  # (1, lr_dim, lr_dim, 3)

    print(f"\nPreparing 5-channel input...")
    print(f"  Flow field shape: {x_lr_batch.shape}")
    print(f"  Value ranges: U=[{x_lr_3ch[...,0].min():.4f}, {x_lr_3ch[...,0].max():.4f}], "
          f"V=[{x_lr_3ch[...,1].min():.4f}, {x_lr_3ch[...,1].max():.4f}], "
          f"P=[{x_lr_3ch[...,2].min():.4f}, {x_lr_3ch[...,2].max():.4f}]")

    # Per-sample Z-score normalization
    x_lr_norm, sample_stats = normalize_per_sample(x_lr_batch)  # (1, lr_dim, lr_dim, 3), (1, 3, 2)
    print(f"  Per-sample stats computed:")
    ch_names = ['U', 'V', 'P']
    for c, name in enumerate(ch_names):
        print(f"    {name}: mean={sample_stats[0,c,0]:.6f}, std={sample_stats[0,c,1]:.6f}")

    # Generate coordinate channels at lr_dim resolution and concatenate → 5-channel input
    coords = make_coord_channels_batch(lr_dim, [lx], [ly])  # (1, lr_dim, lr_dim, 2)
    x_current = np.concatenate([x_lr_norm, coords], axis=-1)  # (1, lr_dim, lr_dim, 5)
    print(f"  5-channel input shape: {x_current.shape}")

    # Load and run cascaded U-Net stages
    print(f"\nLoading and running {len(stage_dims)} U-Net stages...")

    prev_dim = lr_dim
    for stage_idx, target_dim in enumerate(stage_dims):
        stage_name = f"{prev_dim}to{target_dim}"
        model_file = model_name_pattern.format(from_dim=prev_dim, to_dim=target_dim)

        print(f"\n  Stage {stage_idx+1}/{len(stage_dims)}: {stage_name}")
        print(f"    Loading model: {model_file}")

        if not os.path.exists(model_file):
            print(f"    ❌ ERROR: Model file not found!")
            raise FileNotFoundError(f"U-Net stage model not found: {model_file}")

        try:
            unet_model = tf.keras.models.load_model(model_file, compile=False)
            print(f"    ✓ U-Net model loaded successfully")
        except Exception as e:
            print(f"    ❌ ERROR loading model: {e}")
            raise

        # Apply InterpolateRefineModel logic manually on 5-channel input:
        # Step 1: Bicubic interpolation (5-ch) from prev_dim to target_dim
        print(f"    Interpolating: {x_current.shape} → (1, {target_dim}, {target_dim}, 5)")
        x_interp = bicubic_interpolate_batch(x_current, (target_dim, target_dim))
        print(f"    ✓ Interpolated shape: {x_interp.shape}")

        # Step 2: U-Net predicts residual from 5-ch interpolated input → 3-ch output
        print(f"    Predicting residual correction...")
        residual = unet_model.predict(x_interp, verbose=0)  # (1, target_dim, target_dim, 3)

        # Step 3: Scale residual and add to interpolated flow channels
        residual = residual * 0.1
        x_flow = x_interp[..., :3] + residual  # (1, target_dim, target_dim, 3)

        print(f"    ✓ Flow output shape: {x_flow.shape}")
        print(f"    Value ranges: U=[{x_flow[0,...,0].min():.4f}, {x_flow[0,...,0].max():.4f}], "
              f"V=[{x_flow[0,...,1].min():.4f}, {x_flow[0,...,1].max():.4f}], "
              f"P=[{x_flow[0,...,2].min():.4f}, {x_flow[0,...,2].max():.4f}]")

        # Check for NaN/Inf
        if np.isnan(x_flow).any() or np.isinf(x_flow).any():
            nan_count = np.isnan(x_flow).sum()
            inf_count = np.isinf(x_flow).sum()
            print(f"    ⚠️  WARNING: Stage output contains {nan_count} NaN and {inf_count} Inf values!")
            print(f"    Replacing with zeros to prevent propagation...")
            x_flow = np.nan_to_num(x_flow, nan=0.0, posinf=0.0, neginf=0.0)

        # Reattach coord channels at target resolution for the next stage input
        if stage_idx < len(stage_dims) - 1:
            coords_new = make_coord_channels_batch(target_dim, [lx], [ly])  # (1, target_dim, target_dim, 2)
            x_current = np.concatenate([x_flow, coords_new], axis=-1)  # 5-ch for next stage
        else:
            x_current = x_flow  # final 3-ch output — no coord channels needed

        prev_dim = target_dim

    # Denormalize using per-sample stats computed from the original coarse input
    print(f"\nDenormalizing output using per-sample stats...")
    hr_3ch_real = denormalize_per_sample(x_current, sample_stats)  # (1, hr_dim, hr_dim, 3)

    hr_fields = {
        'u': hr_3ch_real[0, ..., 0],
        'v': hr_3ch_real[0, ..., 1],
        'p': hr_3ch_real[0, ..., 2]
    }

    print(f"  Final output ranges:")
    print(f"    U: [{hr_fields['u'].min():.6f}, {hr_fields['u'].max():.6f}]")
    print(f"    V: [{hr_fields['v'].min():.6f}, {hr_fields['v'].max():.6f}]")
    print(f"    P: [{hr_fields['p'].min():.6f}, {hr_fields['p'].max():.6f}]")

    print(f"\n  ✓ Progressive U-Net super-resolution complete")
    return hr_fields


# Copy remaining functions from AE version (unchanged)
# These are: run_fine_simulation_with_ml_init, run_normal_simulation,
# generate_coarse_mesh_solution, run_ml_accelerated_fine_simulation,
# extract_centerlines, format_bc_summary, plot_centerline_comparison

def run_fine_simulation_with_ml_init(Re: float, nx: int, ny: int,
                                     ml_initial_fields: Dict[str, np.ndarray],
                                     dt: float = 0.002, scheme: str = 'UPWIND',
                                     convergence_criteria: Dict[str, float] = None,
                                     max_iterations: int = 100000,
                                     output_name: str = "bfs_accelerated",
                                     bc: Optional[BoundaryConditions] = None,
                                     step_height: float = 1.0,
                                     h: float = 2.0,
                                     Ub: float = 1.0,
                                     lx: float = 10.0,
                                     ly: float = 3.0,
                                     relaxation_factors: Dict[str, float] = None,
                                     callback=None,
                                     callback_interval: int = 1000) -> tuple:
    """
    Step 3: Run fine-resolution BFS simulation with ML-predicted initialization
    
    Args:
        Re: Reynolds number
        nx, ny: Fine mesh dimensions
        ml_initial_fields: Dictionary with 'u', 'v', 'p' fields of shape (ny, nx)
        dt: Time step
        scheme: Numerical scheme
        convergence_criteria: Convergence criteria
        output_name: Base name for output files
        bc: BoundaryConditions object
        step_height: BFS step height
        h: BFS channel height above step
        Ub: BFS bulk velocity
        lx: Domain length in x
        ly: Domain length in y
        relaxation_factors: Under-relaxation factors
        callback: Optional callback function(solver, iteration)
        callback_interval: Interval for callback execution
    
    Returns:
        (solver, iterations, time_elapsed)
    """
    print(f"\n{'='*70}")
    print(f"STEP 3: Running Fine BFS Simulation with ML Initialization")
    print(f"        (Re={Re}, mesh={nx}x{ny})")
    print(f"{'='*70}")
    
    # Create mesh and settings for fine simulation
    mesh = MeshParameters(nx=nx, ny=ny, lx=lx, ly=ly)
    fluid = FluidProperties(Re=Re, rho=1.0)
    
    if convergence_criteria is None:
        convergence_criteria = {'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6}
    
    solver_settings = SolverSettings(dt=dt, scheme=scheme, 
                                   max_iterations=max_iterations,
                                   convergence_criteria=convergence_criteria,
                                   relaxation_factors=relaxation_factors)
    
    # Use provided boundary conditions or create default BFS BCs
    if bc is None:
        bc = BoundaryConditions()
        bc.u_boundaries['right'] = BoundaryCondition('neumann', 0.0)
        bc.v_boundaries['right'] = BoundaryCondition('neumann', 0.0)
        bc.p_boundaries['right'] = BoundaryCondition('dirichlet', 0.0)
    
    # Create solver
    solver = CFDSolver(mesh, fluid, solver_settings, bc,
                      step_height=step_height, h=h, Ub=Ub)
    
    # Initialize with ML predictions (inject into internal cells)
    print("Injecting ML predictions into solver fields...")
    print(f"  - U field: shape {ml_initial_fields['u'].shape} -> Var[0, 1:-1, 1:-1]")
    print(f"  - V field: shape {ml_initial_fields['v'].shape} -> Var[1, 1:-1, 1:-1]")
    print(f"  - P field: shape {ml_initial_fields['p'].shape} -> Var[2, 1:-1, 1:-1]")
    
    # The ML output is in shape (ny, nx), solver internal grid is Var[k, 1:-1, 1:-1] with shape (nx, ny)
    # So we need to transpose
    solver.Var[0, 1:-1, 1:-1] = ml_initial_fields['u'].T
    solver.Var[1, 1:-1, 1:-1] = ml_initial_fields['v'].T
    solver.Var[2, 1:-1, 1:-1] = ml_initial_fields['p'].T
    
    # Apply boundary conditions to ghost cells
    print("Applying boundary conditions to ghost cells...")
    for k in range(solver.nVar):
        solver._apply_bc_wrapper(k)
    
    # Update VarOld and flux fields
    copy_new_to_old(solver.Var, solver.VarOld, solver.nVar, solver.mesh.nx, solver.mesh.ny)
    linear_interpolation(solver.Var, solver.Ff, solver.mesh.nx, solver.mesh.ny, 
                        solver.mesh.dx, solver.mesh.dy)
    
    print("  ✓ ML-based initialization complete")
    
    # Add "_accelerated" suffix to output name
    if not output_name.endswith("_accelerated"):
        output_name = f"{output_name}_accelerated"
    
    # Run simulation
    iterations, time_elapsed = solver.solve(output_name, verbose=True,
                                          callback=callback, 
                                          callback_interval=callback_interval)
    
    return solver, iterations, time_elapsed


def run_normal_simulation(Re: float, nx: int, ny: int,
                         dt: float = 0.002, scheme: str = 'UPWIND',
                         convergence_criteria: Dict[str, float] = None,
                         max_iterations: int = 100000,
                         output_name: str = "bfs_normal",
                         bc: Optional[BoundaryConditions] = None,
                         step_height: float = 1.0,
                         h: float = 2.0,
                         Ub: float = 1.0,
                         lx: float = 10.0,
                         ly: float = 3.0,
                         relaxation_factors: Dict[str, float] = None,
                         check_interval: int = 1000) -> tuple:
    """
    Run a normal BFS CFD simulation without ML acceleration
    
    Args:
        Re: Reynolds number
        nx, ny: Mesh dimensions
        dt: Time step
        scheme: Numerical scheme
        convergence_criteria: Convergence criteria
        max_iterations: Maximum number of iterations
        output_name: Base name for output files
        bc: BoundaryConditions object
        step_height: BFS step height
        h: BFS channel height above step
        Ub: BFS bulk velocity
        lx: Domain length in x
        ly: Domain length in y
        relaxation_factors: Under-relaxation factors
        check_interval: Interval for saving intermediate plots
    
    Returns:
        (solver, iterations, time_elapsed)
    """
    print(f"\n{'='*70}")
    print(f"RUNNING NORMAL (NON-ACCELERATED) BFS SIMULATION")
    print(f"Re={Re}, mesh={nx}x{ny}")
    print(f"{'='*70}")
    
    # Create mesh and settings
    mesh = MeshParameters(nx=nx, ny=ny, lx=lx, ly=ly)
    fluid = FluidProperties(Re=Re, rho=1.0)
    
    if convergence_criteria is None:
        convergence_criteria = {'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6}
    
    solver_settings = SolverSettings(dt=dt, scheme=scheme, 
                                   max_iterations=max_iterations,
                                   convergence_criteria=convergence_criteria,
                                   relaxation_factors=relaxation_factors)
    
    # Use provided boundary conditions or create default BFS BCs
    if bc is None:
        bc = BoundaryConditions()
        bc.u_boundaries['right'] = BoundaryCondition('neumann', 0.0)
        bc.v_boundaries['right'] = BoundaryCondition('neumann', 0.0)
        bc.p_boundaries['right'] = BoundaryCondition('dirichlet', 0.0)
    
    # Create solver and run
    solver = CFDSolver(mesh, fluid, solver_settings, bc,
                      step_height=step_height, h=h, Ub=Ub)
    
    # Add "_normal" suffix to output name
    if not output_name.endswith("_normal"):
        output_name = f"{output_name}_normal"
    
    # Callback for normal simulation monitoring
    def normal_monitor_callback(solver, iteration):
        print(f"\n[Monitor] Normal Simulation Iteration {iteration}: Saving contours...")
        
        # Create Checkpoint Directory
        checkpoint_dir = os.path.join(os.path.dirname(output_name), "checkpoints_normal")
        os.makedirs(checkpoint_dir, exist_ok=True)
             
        # Plot Contours
        contour_filename = os.path.join(checkpoint_dir, f"iter_{iteration}_normal_contours.png")
        try:
            solver._plot_contours(contour_filename)
        except Exception as e:
            print(f"  ⚠️ Warning: Could not plot contours: {e}")

    iterations, time_elapsed = solver.solve(output_name, verbose=True,
                                          callback=normal_monitor_callback,
                                          callback_interval=check_interval)
    
    print(f"Normal BFS simulation completed in {iterations} iterations ({time_elapsed:.2f} seconds)")
    
    return solver, iterations, time_elapsed


def generate_coarse_mesh_solution(
    Re: float,
    lr_dim: int = 10,
    dt: float = 0.002,
    scheme: str = 'UPWIND',
    convergence_criteria: Dict[str, float] = None,
    max_iterations_coarse: int = 100000,
    output_dir: str = None,
    bc: Optional[BoundaryConditions] = None,
    step_height: float = 1.0,
    h: float = 2.0,
    Ub: float = 1.0,
    lx: float = 10.0,
    ly: float = 3.0,
    relaxation_factors: Dict[str, float] = None
) -> tuple:
    """
    Generate coarse mesh BFS solution
    
    Args:
        Re: Reynolds number
        lr_dim: Low resolution dimension for coarse simulation (default: 10)
        dt: Time step
        scheme: Numerical scheme ('QUICK' or 'UPWIND')
        convergence_criteria: Convergence criteria dict
        max_iterations_coarse: Maximum iterations for coarse mesh simulation
        output_dir: Directory for outputs. If None, creates timestamped directory
        bc: BoundaryConditions object
        step_height: BFS step height
        h: BFS channel height above step
        Ub: BFS bulk velocity
        lx: Domain length in x
        ly: Domain length in y
        relaxation_factors: Under-relaxation factors
    
    Returns:
        Tuple of (coarse_fields, iterations_coarse, time_coarse, output_dir)
    """
    
    print(f"\n{'#'*70}")
    print(f"# GENERATING COARSE MESH BFS SOLUTION")
    print(f"# Re={Re}, Coarse Resolution={lr_dim}x{lr_dim}")
    print(f"{'#'*70}\n")
    
    # Create timestamped output directory if not provided
    if output_dir is None:
        output_dir = create_timestamped_output_dir()
        print(f"Created timestamped output directory: {output_dir}")
    
    # Run coarse simulation
    coarse_fields, iterations_coarse, time_coarse = run_coarse_simulation(
        Re=Re, 
        lr_dim=lr_dim, 
        dt=dt, 
        scheme=scheme,
        convergence_criteria=convergence_criteria,
        max_iterations=max_iterations_coarse,
        output_dir=output_dir,
        bc=bc,
        step_height=step_height,
        h=h,
        Ub=Ub,
        lx=lx,
        ly=ly,
        relaxation_factors=relaxation_factors
    )
    
    print(f"\n{'#'*70}")
    print(f"# COARSE MESH BFS SOLUTION COMPLETE")
    print(f"{'#'*70}\n")
    
    return coarse_fields, iterations_coarse, time_coarse, output_dir


def run_ml_accelerated_fine_simulation_unet(
    coarse_fields: Dict[str, np.ndarray],
    Re: float,
    nx: int, 
    ny: int,
    lr_dim: int = 10,
    dt: float = 0.002,
    scheme: str = 'UPWIND',
    convergence_criteria: Dict[str, float] = None,
    max_iterations_fine: int = 100000,
    output_name: str = None,
    stage_dims: List[int] = None,
    model_name_pattern: str = None,
    bc: Optional[BoundaryConditions] = None,
    step_height: float = 1.0,
    h: float = 2.0,
    Ub: float = 1.0,
    lx: float = 10.0,
    ly: float = 3.0,
    relaxation_factors: Dict[str, float] = None,
    normal_solver = None,
    check_interval: int = 1000
) -> tuple:
    """
    Run ML-accelerated fine BFS simulation using coarse mesh solution with cascaded U-Net
    
    Args:
        coarse_fields: Dictionary with 'u', 'v', 'p' fields from coarse simulation
        Re: Reynolds number
        nx, ny: Target fine mesh dimensions
        lr_dim: Low resolution dimension used for coarse simulation (default: 10)
        dt: Time step
        scheme: Numerical scheme
        convergence_criteria: Convergence criteria dict
        max_iterations_fine: Maximum iterations for fine mesh simulation
        output_name: Base name for output files
        stage_dims: List of intermediate dimensions (e.g., [20, 40, 80, 200, 400])
        model_name_pattern: Pattern for stage model filenames
        bc: BoundaryConditions object
        step_height: BFS step height
        h: BFS channel height above step
        Ub: BFS bulk velocity
        lx: Domain length in x
        ly: Domain length in y
        relaxation_factors: Under-relaxation factors
        normal_solver: Optional converged normal solver for reference comparison
        check_interval: Interval for checkpointing and comparison
    
    Returns:
        (solver, iterations, time_elapsed)
    """
    
    print(f"\n{'#'*70}")
    print(f"# ML-ACCELERATED FINE BFS SIMULATION (Progressive U-Net)")
    print(f"# Re={Re}, Target Resolution={nx}x{ny}")
    print(f"# Using coarse solution from {lr_dim}x{lr_dim}")
    print(f"{'#'*70}\n")
    
    # Set default parameters if not provided
    if stage_dims is None:
        stage_dims = [20, 40, 80, 200, 400]
    if model_name_pattern is None:
        # FIXED: Match the actual filename pattern from notebook
        model_name_pattern = "unet_stage_{from_dim}to{to_dim}_progressive_residual_unet_(20-40-80-200-400)_trained_single_and_double_LDC_and_one_bfs.h5"
    if output_name is None:
        output_name = f"bfs_Re{Re}_{nx}x{ny}"
    
    # STEP 1: ML super-resolution (U-Net)
    hr_fields = ml_super_resolution_unet(
        coarse_fields=coarse_fields,
        lr_dim=lr_dim,
        hr_dim=nx,
        stage_dims=stage_dims,
        model_name_pattern=model_name_pattern,
        lx=lx,
        ly=ly
    )

    # Plot LR input / HR predicted (+ HR ground truth when available)
    sr_plot_dir = os.path.join(os.path.dirname(output_name), 'sr_comparison')
    print(f"\nGenerating super-resolution comparison plots → {sr_plot_dir}")
    plot_ml_sr_comparison(
        coarse_fields=coarse_fields,
        hr_fields=hr_fields,
        lr_dim=lr_dim,
        hr_dim=nx,
        lx=lx,
        ly=ly,
        save_dir=sr_plot_dir,
        normal_solver=normal_solver  # None → 2-panel plot; solver → 3-panel with ground truth
    )

    # Monitor Callback for ML Simulation (comparing to Normal Solver)
    metrics_history = {'iter': [], 'u_diff_rms': [], 'v_diff_rms': [], 'u_diff_max': [], 'v_diff_max': []}
    
    def monitor_callback(solver, iteration):
        # Only run if normal_solver is available
        if normal_solver is None:
             return

        print(f"\n[Monitor] Iteration {iteration}: Comparing with Normal simulation...")
        
        # 1. Extract centerlines
        current_centerlines = extract_centerlines(solver, nx, ny)
        normal_centerlines = extract_centerlines(normal_solver, nx, ny)
             
        # Calculate differences
        u_diff = np.abs(current_centerlines['u_vertical']['values'] - normal_centerlines['u_vertical']['values'])
        v_diff = np.abs(current_centerlines['v_horizontal']['values'] - normal_centerlines['v_horizontal']['values'])
             
        u_rms = np.sqrt(np.mean(u_diff**2))
        v_rms = np.sqrt(np.mean(v_diff**2))
        u_max = np.max(u_diff)
        v_max = np.max(v_diff)
             
        # Store metrics
        metrics_history['iter'].append(iteration)
        metrics_history['u_diff_rms'].append(u_rms)
        metrics_history['v_diff_rms'].append(v_rms)
        metrics_history['u_diff_max'].append(u_max)
        metrics_history['v_diff_max'].append(v_max)
             
        print(f"  Diff RMS: U={u_rms:.6e}, V={v_rms:.6e}")
             
        # 3. Create Checkpoint Plots
        checkpoint_dir = os.path.join(os.path.dirname(output_name), "checkpoints_comparison")
        os.makedirs(checkpoint_dir, exist_ok=True)
             
        plot_filename = os.path.join(checkpoint_dir, f"iter_{iteration}_ML_accelerated_comparison.png")
             
        plot_centerline_comparison(
            current_centerlines, 
            normal_centerlines, 
            Re=Re,
            save_path=plot_filename,
            bc=bc,
            show=False  # Do not block execution
        )
        
        # 3b. Plot Contours
        contour_filename = os.path.join(checkpoint_dir, f"iter_{iteration}_ML_accelerated_contours.png")
        # Access the private method _plot_contours from the solver instance
        try:
            solver._plot_contours(contour_filename)
        except Exception as e:
            print(f"  ⚠️ Warning: Could not plot contours: {e}")
             
        # 4. Save fields
        field_filename = os.path.join(checkpoint_dir, f"iter_{iteration}_ML_accelerated_fields.npz")
        np.savez(field_filename, 
             u=solver.Var[0, 1:-1, 1:-1].T, 
             v=solver.Var[1, 1:-1, 1:-1].T,
             p=solver.Var[2, 1:-1, 1:-1].T,
             iteration=iteration)
    
    # STEP 2: Run fine simulation with ML initialization
    solver, iterations, time_elapsed = run_fine_simulation_with_ml_init(
        Re=Re,
        nx=nx,
        ny=ny,
        ml_initial_fields=hr_fields,
        dt=dt,
        scheme=scheme,
        convergence_criteria=convergence_criteria,
        max_iterations=max_iterations_fine,
        output_name=output_name,
        bc=bc,
        step_height=step_height,
        h=h,
        Ub=Ub,
        lx=lx,
        ly=ly,
        relaxation_factors=relaxation_factors,
        callback=monitor_callback if normal_solver else None,
        callback_interval=check_interval
    )
    
    # Plot error evolution
    if len(metrics_history['iter']) > 0:
        plt.figure(figsize=(10, 6))
        plt.semilogy(metrics_history['iter'], metrics_history['u_diff_rms'], 'b-o', label='U RMS Diff')
        plt.semilogy(metrics_history['iter'], metrics_history['v_diff_rms'], 'r-s', label='V RMS Diff')
        plt.xlabel('Iteration')
        plt.ylabel('RMS Difference vs Normal Solution')
        plt.title('Convergence of ML-Accelerated Solution towards Normal Solution')
        plt.grid(True, which="both", ls="-")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(output_name), "error_evolution.png"))
        plt.close()
    
    print(f"\n{'#'*70}")
    print(f"# ML-ACCELERATED FINE BFS SIMULATION COMPLETE")
    print(f"# Converged in {iterations} iterations ({time_elapsed:.2f} seconds)")
    print(f"# Output saved with '_accelerated' suffix")
    print(f"{'#'*70}\n")
    
    return solver, iterations, time_elapsed


# ==============================================================================
# Centerline Extraction and Plotting (unchanged)
# ==============================================================================

def format_bc_summary(bc: Optional[BoundaryConditions]) -> str:
    """
    Format boundary conditions into a detailed summary string
    
    Args:
        bc: BoundaryConditions object or None
    
    Returns:
        Formatted BC summary string
    """
    if bc is None:
        return "BC: Default (not specified)"
    
    def format_boundary_dict(boundary_dict, var_name):
        """Format a single variable's boundary conditions"""
        sides = ['left', 'right', 'top', 'bottom']
        side_abbrev = {'left': 'L', 'right': 'R', 'top': 'T', 'bottom': 'B'}
        
        values = []
        types = []
        for side in sides:
            bc_obj = boundary_dict.get(side)
            if bc_obj is None:
                values.append('?')
                types.append('?')
            else:
                bc_type = 'D' if bc_obj.type.lower() == 'dirichlet' else 'N'
                types.append(bc_type)
                values.append(f"{bc_obj.value:.2f}")
        
        # Check if all values and types are the same
        if len(set(values)) == 1 and len(set(types)) == 1:
            if types[0] == 'D':
                return f"{var_name}(all:{values[0]})"
            else:
                return f"{var_name}(all Neumann)"
        
        # Otherwise, show each side
        parts = [f"{side_abbrev[side]}:{val}" for side, val in zip(sides, values)]
        return f"{var_name}({', '.join(parts)})"
    
    # Format each variable
    u_bc = format_boundary_dict(bc.u_boundaries, 'U')
    v_bc = format_boundary_dict(bc.v_boundaries, 'V')
    p_bc = format_boundary_dict(bc.p_boundaries, 'P')
    
    return f"BC: {u_bc} {v_bc} {p_bc}"


def extract_centerlines(solver, nx: int, ny: int) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract centerline velocities from solver
    
    Args:
        solver: CFDSolver instance
        nx, ny: Mesh dimensions
    
    Returns:
        Dictionary with centerline data:
        {
            'u_vertical': {'y': array, 'values': array},   # U along vertical centerline (x=Lx/2)
            'v_horizontal': {'x': array, 'values': array}  # V along horizontal centerline (y=Ly/2)
        }
    """
    # Get mesh coordinates
    x = np.linspace(0, solver.mesh.lx, nx)
    y = np.linspace(0, solver.mesh.ly, ny)
    
    # Extract fields (internal cells only, no ghost cells)
    u_field = solver.Var[0, 1:-1, 1:-1].T.copy()  # Shape: (ny, nx)
    v_field = solver.Var[1, 1:-1, 1:-1].T.copy()  # Shape: (ny, nx)
    
    # U velocity along vertical centerline (x = Lx/2, varying y)
    centerline_x_idx = nx // 2
    u_vertical = u_field[:, centerline_x_idx]
    
    # V velocity along horizontal centerline (y = Ly/2, varying x)
    centerline_y_idx = ny // 2
    v_horizontal = v_field[centerline_y_idx, :]
    
    return {
        'u_vertical': {'y': y, 'values': u_vertical},
        'v_horizontal': {'x': x, 'values': v_horizontal}
    }


def compute_centerline_metrics(ml_centerlines: Dict, normal_centerlines: Dict) -> Dict[str, float]:
    """Compute quantitative difference metrics between centerlines
    
    Args:
        ml_centerlines: Centerline data from ML-accelerated simulation
        normal_centerlines: Centerline data from normal simulation
    
    Returns:
        Dictionary with metrics: 'u_l2', 'u_max', 'v_l2', 'v_max'
    """
    u_ml = ml_centerlines['u_vertical']['values']
    u_normal = normal_centerlines['u_vertical']['values']
    v_ml = ml_centerlines['v_horizontal']['values']
    v_normal = normal_centerlines['v_horizontal']['values']
    
    u_diff = u_ml - u_normal
    v_diff = v_ml - v_normal
    
    metrics = {
        'u_l2': float(np.sqrt(np.mean(u_diff**2))),
        'u_max': float(np.max(np.abs(u_diff))),
        'u_mean': float(np.mean(np.abs(u_diff))),
        'v_l2': float(np.sqrt(np.mean(v_diff**2))),
        'v_max': float(np.max(np.abs(v_diff))),
        'v_mean': float(np.mean(np.abs(v_diff)))
    }
    
    return metrics


def plot_ml_sr_comparison(coarse_fields: Dict[str, np.ndarray],
                          hr_fields: Dict[str, np.ndarray],
                          lr_dim: int, hr_dim: int,
                          lx: float, ly: float,
                          save_dir: str,
                          normal_solver=None) -> None:
    """
    Plot LR input / HR predicted (and optionally HR ground truth) for each of u, v, p.

    Produces 3 PNG files (one per variable) saved to save_dir.
    When normal_solver is provided: 3 subplots (LR input | HR ground | HR predicted).
    When normal_solver is None:     2 subplots (LR input | HR predicted).
    Contourf style matches _plot_contours.

    Args:
        coarse_fields: {'u','v','p'} arrays of shape (lr_dim, lr_dim)
        hr_fields:     {'u','v','p'} ML super-resolution output, shape (hr_dim, hr_dim)
        lr_dim, hr_dim: Grid dimensions
        lx, ly: Physical domain lengths
        save_dir: Directory to save the plots
        normal_solver: Optional converged CFDSolver at fine resolution (HR ground truth)
    """
    os.makedirs(save_dir, exist_ok=True)

    var_info = [
        ('u', 'U Velocity',  'RdBu'),
        ('v', 'V Velocity',  'RdBu'),
        ('p', 'Pressure',    'viridis'),
    ]

    have_ground = normal_solver is not None

    # Build HR ground-truth arrays from the solver (if available)
    ground = {}
    if have_ground:
        ground = {
            'u': normal_solver.Var[0, 1:-1, 1:-1].T.copy(),
            'v': normal_solver.Var[1, 1:-1, 1:-1].T.copy(),
            'p': normal_solver.Var[2, 1:-1, 1:-1].T.copy(),
        }

    print(f"\n{'='*70}")
    print(f"ML SUPER-RESOLUTION FIELD COMPARISON")
    print(f"{'='*70}")
    for key in ['u', 'v', 'p']:
        lr = coarse_fields[key]
        hr = hr_fields[key]
        print(f"  {key.upper()}:")
        print(f"    LR  input : [{lr.min():.6f}, {lr.max():.6f}]  (mean {lr.mean():.6f})")
        if have_ground:
            gt = ground[key]
            print(f"    HR  ground: [{gt.min():.6f}, {gt.max():.6f}]  (mean {gt.mean():.6f})")
        print(f"    HR  pred  : [{hr.min():.6f}, {hr.max():.6f}]  (mean {hr.mean():.6f})")
    print(f"{'='*70}")

    # Coordinate grids
    x_lr = np.linspace(0, lx, lr_dim)
    y_lr = np.linspace(0, ly, lr_dim)
    X_lr, Y_lr = np.meshgrid(x_lr, y_lr)

    x_hr = np.linspace(0, lx, hr_dim)
    y_hr = np.linspace(0, ly, hr_dim)
    X_hr, Y_hr = np.meshgrid(x_hr, y_hr)

    n_cols = 3 if have_ground else 2

    for key, label, cmap in var_info:
        lr_data = coarse_fields[key]   # (lr_dim, lr_dim)
        pr_data = hr_fields[key]       # (hr_dim, hr_dim)

        vmin_lr = lr_data.min()
        vmax_lr = lr_data.max()

        if have_ground:
            gt_data = ground[key]
            vmin_hr = min(gt_data.min(), pr_data.min())
            vmax_hr = max(gt_data.max(), pr_data.max())
        else:
            vmin_hr = pr_data.min()
            vmax_hr = pr_data.max()

        ar = lx / ly  # physical aspect ratio for figure sizing
        col_w = max(4.0, min(8.0, 6.0 * ar))
        row_h = max(1.5, min(6.0, col_w / ar))
        fig, axes = plt.subplots(1, n_cols, figsize=(col_w * n_cols + 1.5, row_h + 1.2))

        # --- LR input ---
        im0 = axes[0].contourf(X_lr, Y_lr, lr_data, levels=20, cmap=cmap,
                               vmin=vmin_lr, vmax=vmax_lr)
        axes[0].set_title(f'LR Input ({lr_dim}×{lr_dim})', fontsize=11)
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0])

        if have_ground:
            # --- HR ground truth ---
            im1 = axes[1].contourf(X_hr, Y_hr, gt_data, levels=20, cmap=cmap,
                                   vmin=vmin_hr, vmax=vmax_hr)
            axes[1].set_title(f'HR Ground Truth ({hr_dim}×{hr_dim})', fontsize=11)
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            axes[1].set_aspect('equal')
            plt.colorbar(im1, ax=axes[1])
            pred_ax = axes[2]
            title_suffix = 'LR Input / HR Ground / HR Predicted'
        else:
            pred_ax = axes[1]
            title_suffix = 'LR Input / HR Predicted'

        # --- HR predicted ---
        im_pred = pred_ax.contourf(X_hr, Y_hr, pr_data, levels=20, cmap=cmap,
                                   vmin=vmin_hr, vmax=vmax_hr)
        pred_ax.set_title(f'HR Predicted ({hr_dim}×{hr_dim})', fontsize=11)
        pred_ax.set_xlabel('X')
        pred_ax.set_ylabel('Y')
        pred_ax.set_aspect('equal')
        plt.colorbar(im_pred, ax=pred_ax)

        fig.suptitle(f'{label} — {title_suffix}  (Lx={lx}, Ly={ly})',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out_path = os.path.join(save_dir, f'sr_comparison_{key}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {out_path}")


def plot_centerline_comparison(ml_centerlines: Dict, normal_centerlines: Dict, 
                               Re: float, save_path: str = None, 
                               bc: Optional[BoundaryConditions] = None,
                               iteration: int = None, metrics: Dict = None,
                               show: bool = True):
    """
    Plot centerline comparison between ML-accelerated and normal BFS simulations
    
    Args:
        ml_centerlines: Centerline data from ML-accelerated simulation
        normal_centerlines: Centerline data from normal simulation
        Re: Reynolds number
        save_path: Optional path to save the figure
        bc: BoundaryConditions object (optional, for display in plot)
        iteration: Iteration number to display in title
        metrics: Optional dictionary of computed metrics to display
        show: If True, display the plot (default: True)
    """
    import matplotlib.pyplot as plt
    
    # Compute metrics if not provided
    if metrics is None:
        metrics = compute_centerline_metrics(ml_centerlines, normal_centerlines)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot U velocity along vertical centerline
    ax1 = axes[0]
    ax1.plot(ml_centerlines['u_vertical']['values'], 
             ml_centerlines['u_vertical']['y'],
             'b-o', linewidth=2, markersize=4, label='ML-Accelerated (U-Net)', alpha=0.7)
    ax1.plot(normal_centerlines['u_vertical']['values'], 
             normal_centerlines['u_vertical']['y'],
             'r--s', linewidth=2, markersize=4, label='Normal', alpha=0.7)
    ax1.set_xlabel('U Velocity', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    title_str = 'U Velocity along Vertical Centerline (x=Lx/2)'
    if iteration is not None:
        title_str += f' @ Iter {iteration}'
    ax1.set_title(title_str, fontsize=11)
    # Add metrics text box
    metrics_text = f"L2: {metrics['u_l2']:.2e}\nMax: {metrics['u_max']:.2e}\nMean: {metrics['u_mean']:.2e}"
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
             fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot V velocity along horizontal centerline
    ax2 = axes[1]
    ax2.plot(normal_centerlines['v_horizontal']['x'],
             normal_centerlines['v_horizontal']['values'],
             'r--s', linewidth=2, markersize=4, label='Normal', alpha=0.7)
    ax2.plot(ml_centerlines['v_horizontal']['x'],
             ml_centerlines['v_horizontal']['values'],
             'b-o', linewidth=2, markersize=4, label='ML-Accelerated (U-Net)', alpha=0.7)
    ax2.set_xlabel('X Position', fontsize=12)
    ax2.set_ylabel('V Velocity', fontsize=12)
    title_str = 'V Velocity along Horizontal Centerline (y=Ly/2)'
    if iteration is not None:
        title_str += f' @ Iter {iteration}'
    ax2.set_title(title_str, fontsize=11)
    # Add metrics text box
    metrics_text = f"L2: {metrics['v_l2']:.2e}\nMax: {metrics['v_max']:.2e}\nMean: {metrics['v_mean']:.2e}"
    ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, 
             fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add BC summary as subtitle if provided
    if bc is not None:
        bc_summary = format_bc_summary(bc)
        fig.suptitle(f'BFS Centerline Velocity Comparison - Progressive U-Net (Re={Re})\n{bc_summary}', 
                    fontsize=14, fontweight='bold', y=0.98)
    else:
        fig.suptitle(f'BFS Centerline Velocity Comparison - Progressive U-Net (Re={Re})', 
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    # Calculate and print differences
    print("\n" + "="*70)
    print("CENTERLINE COMPARISON STATISTICS")
    print("="*70)
    
    u_diff = np.abs(ml_centerlines['u_vertical']['values'] - normal_centerlines['u_vertical']['values'])
    v_diff = np.abs(ml_centerlines['v_horizontal']['values'] - normal_centerlines['v_horizontal']['values'])
    
    print(f"U Velocity (vertical centerline):")
    print(f"  Max absolute difference: {np.max(u_diff):.6e}")
    print(f"  Mean absolute difference: {np.mean(u_diff):.6e}")
    print(f"  RMS difference: {np.sqrt(np.mean(u_diff**2)):.6e}")
    
    print(f"\nV Velocity (horizontal centerline):")
    print(f"  Max absolute difference: {np.max(v_diff):.6e}")
    print(f"  Mean absolute difference: {np.mean(v_diff):.6e}")
    print(f"  RMS difference: {np.sqrt(np.mean(v_diff**2)):.6e}")
    print("="*70)


def save_checkpoint(iteration: int, solver, checkpoint_dir: str, prefix: str):
    """Save solver checkpoint at specified iteration
    
    Args:
        iteration: Current iteration number
        solver: CFDSolver instance
        checkpoint_dir: Directory to save checkpoints
        prefix: Prefix for checkpoint files (e.g., 'normal' or 'ml_accelerated')
    """
    checkpoint_file = os.path.join(checkpoint_dir, f"{prefix}_checkpoint_iter{iteration}.npz")
    
    # Extract fields (internal cells only)
    u_field = solver.Var[0, 1:-1, 1:-1].T.copy()
    v_field = solver.Var[1, 1:-1, 1:-1].T.copy()
    p_field = solver.Var[2, 1:-1, 1:-1].T.copy()
    
    np.savez_compressed(checkpoint_file,
                       iteration=iteration,
                       u=u_field,
                       v=v_field,
                       p=p_field,
                       nx=solver.mesh.nx,
                       ny=solver.mesh.ny,
                       lx=solver.mesh.lx,
                       ly=solver.mesh.ly,
                       Re=solver.fluid.Re)
    
    print(f"  ✓ Checkpoint saved: {checkpoint_file}")


def load_checkpoint(checkpoint_file: str) -> Dict:
    """Load solver checkpoint from file
    
    Args:
        checkpoint_file: Path to checkpoint file
    
    Returns:
        Dictionary with checkpoint data
    """
    data = np.load(checkpoint_file)
    return {
        'iteration': int(data['iteration']),
        'u': data['u'],
        'v': data['v'],
        'p': data['p'],
        'nx': int(data['nx']),
        'ny': int(data['ny']),
        'lx': float(data['lx']),
        'ly': float(data['ly']),
        'Re': float(data['Re'])
    }


def plot_error_evolution(metrics_history: List[Dict], save_path: str, Re: float):
    """Plot how errors evolve over iterations
    
    Args:
        metrics_history: List of (iteration, metrics) tuples
        save_path: Path to save the figure
        Re: Reynolds number
    """
    import matplotlib.pyplot as plt
    
    iterations = [m['iteration'] for m in metrics_history]
    u_l2 = [m['u_l2'] for m in metrics_history]
    u_max = [m['u_max'] for m in metrics_history]
    v_l2 = [m['v_l2'] for m in metrics_history]
    v_max = [m['v_max'] for m in metrics_history]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # U L2 error
    axes[0, 0].plot(iterations, u_l2, 'b-o', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Iteration', fontsize=12)
    axes[0, 0].set_ylabel('L2 Error', fontsize=12)
    axes[0, 0].set_title('U Velocity L2 Error', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # U Max error
    axes[0, 1].plot(iterations, u_max, 'b-s', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Iteration', fontsize=12)
    axes[0, 1].set_ylabel('Max Error', fontsize=12)
    axes[0, 1].set_title('U Velocity Max Error', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # V L2 error
    axes[1, 0].plot(iterations, v_l2, 'r-o', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('Iteration', fontsize=12)
    axes[1, 0].set_ylabel('L2 Error', fontsize=12)
    axes[1, 0].set_title('V Velocity L2 Error', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    # V Max error
    axes[1, 1].plot(iterations, v_max, 'r-s', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('Iteration', fontsize=12)
    axes[1, 1].set_ylabel('Max Error', fontsize=12)
    axes[1, 1].set_title('V Velocity Max Error', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')
    
    plt.suptitle(f'Error Evolution: ML-Accelerated vs Normal (Re={Re})', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Error evolution plot saved: {save_path}")


def run_comparison_with_checkpoints(
    Re: float,
    nx: int, ny: int,
    lr_dim: int = 10,
    checkpoint_interval: int = 1000,
    dt: float = 0.002,
    scheme: str = 'UPWIND',
    convergence_criteria: Dict[str, float] = None,
    max_iterations: int = 100000,
    output_dir: str = None,
    stage_dims: List[int] = None,
    model_name_pattern: str = None,
    bc: Optional[BoundaryConditions] = None,
    step_height: float = 1.0,
    h: float = 2.0,
    Ub: float = 1.0,
    lx: float = 10.0,
    ly: float = 3.0,
    relaxation_factors: Dict[str, float] = None
):
    """Run complete comparison workflow with iterative checkpointing
    
    This function:
    1. Runs normal simulation with checkpoints every N iterations
    2. Runs coarse simulation
    3. Runs ML-accelerated simulation with checkpoints
    4. Compares centerlines at each checkpoint
    5. Generates error evolution plots
    
    Args:
        Re: Reynolds number
        nx, ny: Fine mesh dimensions
        lr_dim: Coarse mesh dimension
        checkpoint_interval: Save checkpoints every N iterations
        ... (other standard parameters)
    
    Returns:
        Dictionary with output paths and metrics history
    """
    print(f"\n{'#'*80}")
    print(f"# BFS SIMULATION WITH ITERATIVE CHECKPOINT COMPARISON")
    print(f"# Re={Re}, Fine mesh={nx}x{ny}, Coarse mesh={lr_dim}x{lr_dim}")
    print(f"# Checkpoint interval: {checkpoint_interval} iterations")
    print(f"{'#'*80}\n")
    
    # Create output directory
    if output_dir is None:
        output_dir = create_timestamped_output_dir("outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    plots_dir = os.path.join(output_dir, "checkpoint_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set defaults
    if convergence_criteria is None:
        convergence_criteria = {'u': 1e-6, 'v': 1e-6, 'p': 1e-6}
    if stage_dims is None:
        stage_dims = [20, 40, 80, 200, 400]
    if model_name_pattern is None:
        # FIXED: Match the actual filename pattern from notebook
        model_name_pattern = "unet_stage_{from_dim}to{to_dim}_progressive_residual_unet_(20-40-80-200-400)_trained_single_and_double_LDC_and_one_bfs.h5"
    if bc is None:
        bc = BoundaryConditions()
        bc.set_bfs_boundaries()
    if relaxation_factors is None:
        relaxation_factors = {'u': 0.7, 'v': 0.7, 'p': 0.3}
    
    # ====================
    # STEP 1: Normal Simulation with Checkpoints
    # ====================
    print(f"\n{'='*80}")
    print(f"STEP 1: Running Normal (Reference) Simulation with Checkpoints")
    print(f"{'='*80}")
    
    mesh = MeshParameters(nx=nx, ny=ny, lx=lx, ly=ly)
    fluid = FluidProperties(Re=Re, rho=1.0)
    solver_settings = SolverSettings(dt=dt, scheme=scheme,
                                    max_iterations=max_iterations,
                                    convergence_criteria=convergence_criteria,
                                    relaxation_factors=relaxation_factors)
    
    normal_solver = CFDSolver(mesh, fluid, solver_settings, bc,
                             step_height=step_height, h=h, Ub=Ub)
    
    # Checkpoint callback for normal simulation
    def normal_checkpoint_callback(iteration, solver):
        save_checkpoint(iteration, solver, checkpoint_dir, "normal")
    
    output_name = os.path.join(output_dir, f"bfs_normal_Re{Re}_{nx}x{ny}")
    normal_iter, normal_time = normal_solver.solve(
        output_name, verbose=True,
        checkpoint_interval=checkpoint_interval,
        checkpoint_callback=normal_checkpoint_callback
    )
    
    print(f"\n✓ Normal simulation complete: {normal_iter} iterations, {normal_time:.2f}s")
    
    # ====================
    # STEP 2: Coarse Simulation
    # ====================
    print(f"\n{'='*80}")
    print(f"STEP 2: Running Coarse Simulation")
    print(f"{'='*80}")
    
    coarse_fields, coarse_iter, coarse_time = run_coarse_simulation(
        Re=Re, lr_dim=lr_dim, dt=dt, scheme=scheme,
        convergence_criteria=convergence_criteria,
        max_iterations=max_iterations,
        output_dir=output_dir,
        bc=bc,
        step_height=step_height, h=h, Ub=Ub,
        lx=lx, ly=ly,
        relaxation_factors=relaxation_factors
    )
    
    print(f"\n✓ Coarse simulation complete: {coarse_iter} iterations, {coarse_time:.2f}s")
    
    # ====================
    # STEP 3: ML Super-Resolution
    # ====================
    print(f"\n{'='*80}")
    print(f"STEP 3: ML Super-Resolution")
    print(f"{'='*80}")
    
    hr_fields = ml_super_resolution_unet(
        coarse_fields=coarse_fields,
        lr_dim=lr_dim, hr_dim=nx,
        stage_dims=stage_dims,
        model_name_pattern=model_name_pattern,
        lx=lx, ly=ly
    )
    
    # ====================
    # STEP 4: ML-Accelerated Simulation with Checkpoints
    # ====================
    print(f"\n{'='*80}")
    print(f"STEP 4: Running ML-Accelerated Simulation with Checkpoints")
    print(f"{'='*80}")
    
    ml_solver = CFDSolver(mesh, fluid, solver_settings, bc,
                         step_height=step_height, h=h, Ub=Ub)
    
    # Initialize with ML predictions
    print("Injecting ML predictions into solver fields...")
    ml_solver.Var[0, 1:-1, 1:-1] = hr_fields['u'].T
    ml_solver.Var[1, 1:-1, 1:-1] = hr_fields['v'].T
    ml_solver.Var[2, 1:-1, 1:-1] = hr_fields['p'].T
    
    for k in range(ml_solver.nVar):
        ml_solver._apply_bc_wrapper(k)
    
    copy_new_to_old(ml_solver.Var, ml_solver.VarOld, ml_solver.nVar,
                   ml_solver.mesh.nx, ml_solver.mesh.ny)
    linear_interpolation(ml_solver.Var, ml_solver.Ff,
                        ml_solver.mesh.nx, ml_solver.mesh.ny,
                        ml_solver.mesh.dx, ml_solver.mesh.dy)
    
    print("  ✓ ML-based initialization complete")
    
    # Checkpoint callback for ML simulation
    def ml_checkpoint_callback(iteration, solver):
        save_checkpoint(iteration, solver, checkpoint_dir, "ml_accelerated")
    
    output_name_ml = os.path.join(output_dir, f"bfs_ml_accelerated_Re{Re}_{nx}x{ny}")
    ml_iter, ml_time = ml_solver.solve(
        output_name_ml, verbose=True,
        checkpoint_interval=checkpoint_interval,
        checkpoint_callback=ml_checkpoint_callback
    )
    
    print(f"\n✓ ML-accelerated simulation complete: {ml_iter} iterations, {ml_time:.2f}s")
    
    # ====================
    # STEP 5: Generate Comparison Plots at Each Checkpoint
    # ====================
    print(f"\n{'='*80}")
    print(f"STEP 5: Generating Checkpoint Comparison Plots")
    print(f"{'='*80}")
    
    # Find all checkpoint iterations
    normal_checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("normal_checkpoint")])
    ml_checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("ml_accelerated_checkpoint")])
    
    # Extract iteration numbers
    def get_iteration(filename):
        return int(filename.split("iter")[1].split(".")[0])
    
    normal_iters = [get_iteration(f) for f in normal_checkpoints]
    ml_iters = [get_iteration(f) for f in ml_checkpoints]
    
    # Find common checkpoints
    common_iters = sorted(set(normal_iters) & set(ml_iters))
    
    print(f"\nFound {len(common_iters)} common checkpoint iterations: {common_iters}")
    
    metrics_history = []
    
    for iteration in common_iters:
        print(f"\nProcessing checkpoint at iteration {iteration}...")
        
        # Load checkpoints
        normal_ckpt = load_checkpoint(os.path.join(checkpoint_dir, f"normal_checkpoint_iter{iteration}.npz"))
        ml_ckpt = load_checkpoint(os.path.join(checkpoint_dir, f"ml_accelerated_checkpoint_iter{iteration}.npz"))
        
        # Create temporary solvers for centerline extraction
        temp_normal_solver = CFDSolver(mesh, fluid, solver_settings, bc,
                                      step_height=step_height, h=h, Ub=Ub)
        temp_ml_solver = CFDSolver(mesh, fluid, solver_settings, bc,
                                  step_height=step_height, h=h, Ub=Ub)
        
        # Load fields
        temp_normal_solver.Var[0, 1:-1, 1:-1] = normal_ckpt['u'].T
        temp_normal_solver.Var[1, 1:-1, 1:-1] = normal_ckpt['v'].T
        temp_normal_solver.Var[2, 1:-1, 1:-1] = normal_ckpt['p'].T
        
        temp_ml_solver.Var[0, 1:-1, 1:-1] = ml_ckpt['u'].T
        temp_ml_solver.Var[1, 1:-1, 1:-1] = ml_ckpt['v'].T
        temp_ml_solver.Var[2, 1:-1, 1:-1] = ml_ckpt['p'].T
        
        # Extract centerlines
        normal_centerlines = extract_centerlines(temp_normal_solver, nx, ny)
        ml_centerlines = extract_centerlines(temp_ml_solver, nx, ny)
        
        # Compute metrics
        metrics = compute_centerline_metrics(ml_centerlines, normal_centerlines)
        metrics['iteration'] = iteration
        metrics_history.append(metrics)
        
        print(f"  Metrics - U: L2={metrics['u_l2']:.2e}, Max={metrics['u_max']:.2e}")
        print(f"           V: L2={metrics['v_l2']:.2e}, Max={metrics['v_max']:.2e}")
        
        # Generate comparison plot
        plot_path = os.path.join(plots_dir, f"centerline_comparison_iter{iteration:06d}.png")
        plot_centerline_comparison(ml_centerlines, normal_centerlines, Re,
                                  save_path=plot_path, bc=bc,
                                  iteration=iteration, metrics=metrics,
                                  show=False)
        print(f"  ✓ Plot saved: {plot_path}")
    
    # ====================
    # STEP 6: Save Metrics and Generate Summary Plot
    # ====================
    print(f"\n{'='*80}")
    print(f"STEP 6: Generating Summary Plots and Saving Metrics")
    print(f"{'='*80}")
    
    # Save metrics to CSV
    metrics_file = os.path.join(output_dir, "metrics_history.csv")
    with open(metrics_file, 'w') as f:
        f.write("iteration,u_l2,u_max,u_mean,v_l2,v_max,v_mean\n")
        for m in metrics_history:
            f.write(f"{m['iteration']},{m['u_l2']},{m['u_max']},{m['u_mean']},"
                   f"{m['v_l2']},{m['v_max']},{m['v_mean']}\n")
    print(f"\n✓ Metrics saved to: {metrics_file}")
    
    # Generate error evolution plot
    evolution_plot = os.path.join(output_dir, "error_evolution.png")
    plot_error_evolution(metrics_history, evolution_plot, Re)
    
    # ====================
    # Summary
    # ====================
    total_ml_time = coarse_time + ml_time
    
    print(f"\n{'#'*80}")
    print(f"# COMPARISON WORKFLOW COMPLETE")
    print(f"#")
    print(f"# Normal simulation:     {normal_iter} iterations ({normal_time:.2f}s)")
    print(f"# Coarse simulation:     {coarse_iter} iterations ({coarse_time:.2f}s)")
    print(f"# ML-accelerated (fine): {ml_iter} iterations ({ml_time:.2f}s)")
    print(f"# Total ML time:         {total_ml_time:.2f}s (coarse + fine)")
    print(f"# Speedup:               {normal_time/total_ml_time:.2f}x")
    print(f"# Iteration savings:     {normal_iter - ml_iter} iterations (fine mesh only, {100*(1-ml_iter/normal_iter):.1f}%)")
    print(f"#")
    print(f"# Outputs:")
    print(f"#   - Checkpoints:         {checkpoint_dir}")
    print(f"#   - Comparison plots:    {plots_dir}")
    print(f"#   - Metrics history:     {metrics_file}")
    print(f"#   - Error evolution:     {evolution_plot}")
    print(f"{'#'*80}\n")
    
    return {
        'output_dir': output_dir,
        'checkpoint_dir': checkpoint_dir,
        'plots_dir': plots_dir,
        'metrics_history': metrics_history,
        'normal_iterations': normal_iter,
        'ml_iterations': ml_iter,
        'coarse_iterations': coarse_iter,
        'speedup': normal_time / total_ml_time
    }


# ==============================================================================
# Main Execution Block - MODIFIED FOR U-NET
# ==============================================================================

if __name__ == "__main__":
    """
    Example: Run ML-accelerated BFS simulation with Progressive U-Net
    
    Required files (ensure these are in the same directory or specify full paths):
    - norm_stats_10to400_progressive_residual_unet_(20-40-80-200-400)_trained_with_single-double_LDCs_with_one_bfs.txt
    - unet_stage_10to20_progressive_residual_unet_(20-40-80-200-400)_trained_with_single-double_LDCs_with_one_bfs.h5
    - unet_stage_20to40_progressive_residual_unet_(20-40-80-200-400)_trained_with_single-double_LDCs_with_one_bfs.h5
    - unet_stage_40to80_progressive_residual_unet_(20-40-80-200-400)_trained_with_single-double_LDCs_with_one_bfs.h5
    - unet_stage_80to200_progressive_residual_unet_(20-40-80-200-400)_trained_with_single-double_LDCs_with_one_bfs.h5
    - unet_stage_200to400_progressive_residual_unet_(20-40-80-200-400)_trained_with_single-double_LDCs_with_one_bfs.h5
    """
    
    # =========================================================================
    # CONFIGURATION - Customize these parameters
    # =========================================================================
    
    # Reynolds number
    Re = 100
    
    # Fine mesh dimensions
    nx = 400
    ny = 400
    
    # Coarse mesh dimension
    lr_dim = 10
    
    # Progressive U-Net stage dimensions
    stage_dims = [20, 40, 80, 200, 400]
    
    # BFS geometry parameters
    lx = 20.0          # Domain length in x
    ly = 1.94           # Domain length in y
    step_height = 0.94  # Step height
    h = 1.0            # Channel height above step
    Ub = 1.0           # Bulk velocity
    
    # Time step (reduced for stability at fine mesh)
    dt = 1e-3  # Time step (reduced for stability at fine mesh)
    
    # Numerical scheme ('QUICK' or 'UPWIND')
    # UPWIND is more stable for BFS with flow separation
    scheme = 'UPWIND'
    
    # Under-relaxation factors (important for BFS stability)
    # More conservative values to prevent NaN/Inf
    relaxation_factors = {
        'u': 0.20,
        'v': 0.20,
        'p': 0.01
    }
    
    # Maximum iterations for different simulations
    max_iterations_coarse = 200000   # Max iterations for coarse mesh (10x10)
    max_iterations_fine_ml = 35000jhfhadsjkfdskjhdsfdskjsds
    1

      # Max iterations for fine mesh with ML initialization
    max_iterations_normal = 125000   # Max iterations for normal simulation
    
    # Iteration interval for monitoring and saving plots
    monitoring_interval = 100

    # Model name pattern (used to format filenames for each stage)
    # FIXED: Match the actual suffix from the notebook
    model_suffix = "progressive_residual_unet_(20-40-80-200-400)_trained with bfs Re 300"
    model_name_pattern = f"unet_stage_{{from_dim}}to{{to_dim}}_{model_suffix}.h5"
    
    # =========================================================================
    # COARSE SIMULATION OPTIONS (TIME SAVING)
    # =========================================================================

    # Options:
    #   'run'  : Run coarse simulation
    #   'load' : Load from a previously saved coarse simulation HDF5 file

    coarse_simulation_mode = 'load'  # Change to 'run' or 'load' as needed

    # If mode='load', specify the path to the previously saved coarse HDF5 file:
    # Example: "outputs/17-02-2026-01-39-30/bfs_coarse_Re400_10x10_200000_coarse_iterations_coarse.h5"
    previous_coarse_hdf5 = "C:\\Users\\NAVANEETH\\Downloads\\ddd\\ddd\\Re 300\\bfs_Re100_mesh10x10.h5"

    # =========================================================================
    # NORMAL SIMULATION OPTIONS (TIME SAVING)
    # =========================================================================
    
    # The normal simulation is only used for comparison with ML-accelerated results.
    # It's NOT required to run the ML-accelerated simulation.
    # Options:
    #   'run'  : Run normal simulation (takes longest time ~hours)
    #   'load' : Load from a previously saved normal simulation HDF5 file
    #   'skip' : Skip normal simulation entirely (fastest, no comparison)
    
    normal_simulation_mode = 'load'  # Change to 'run' or 'load' or 'skip' as needed
    
    # If mode='load', specify the path to the previously saved HDF5 file:
    # Example: "outputs/17-02-2026-01-39-30/bfs_Re400_400x400_125000_NORMAL_normal.h5"
    previous_normal_hdf5 = "C:\\Users\\NAVANEETH\\Downloads\\ddd\\ddd\\Re 300\\bfs_Re100_mesh400x400.h5"
    
    # =========================================================================


    print(f"\n{'='*70}")
    print(f"CONFIGURATION SUMMARY - PROGRESSIVE U-NET")
    print(f"{'='*70}")
    print(f"Reynolds Number: {Re}")
    print(f"Fine Mesh: {nx}×{ny}")
    print(f"Coarse Mesh: {lr_dim}×{lr_dim}")
    print(f"Progressive Stages: {lr_dim} → {' → '.join(map(str, stage_dims))}")
    print(f"BFS Geometry: Lx={lx}, Ly={ly} (aspect ratio: {lx/ly:.2f}:1)")
    print(f"Per-Sample Normalization: ENABLED")
    print(f"Coordinate Channels: ENABLED (5-channel input)")
    print(f"Coarse Simulation Mode: {coarse_simulation_mode.upper()}")
    if coarse_simulation_mode == 'load':
        print(f"  Loading from: {previous_coarse_hdf5}")
    print(f"Normal Simulation Mode: {normal_simulation_mode.upper()}")
    if normal_simulation_mode == 'load':
        print(f"  Loading from: {previous_normal_hdf5}")
    elif normal_simulation_mode == 'skip':
        print(f"  ⚠️  No comparison plot will be generated")
    print(f"{'='*70}\n")
    
    # =========================================================================
    # PRE-FLIGHT CHECK: ALL REQUIRED FILES
    # =========================================================================

    print("\n" + "="*70)
    print("PRE-FLIGHT CHECK")
    print("="*70)

    _missing = []

    # --- ML stage model files ---
    print("\n  [ML Models]")
    _prev_dim = lr_dim
    for _target_dim in stage_dims:
        _model_file = model_name_pattern.format(from_dim=_prev_dim, to_dim=_target_dim)
        if os.path.exists(_model_file):
            print(f"  ✓ Stage {_prev_dim}→{_target_dim}: {_model_file}")
        else:
            print(f"  ✗ Stage {_prev_dim}→{_target_dim}: {_model_file}  ← NOT FOUND")
            _missing.append(_model_file)
        _prev_dim = _target_dim

    # --- Coarse simulation HDF5 (only when load mode) ---
    if coarse_simulation_mode == 'load':
        print("\n  [Coarse Simulation - Load File]")
        if os.path.exists(previous_coarse_hdf5):
            print(f"  ✓ Coarse HDF5: {previous_coarse_hdf5}")
        else:
            print(f"  ✗ Coarse HDF5: {previous_coarse_hdf5}  ← NOT FOUND")
            _missing.append(previous_coarse_hdf5)

    # --- Normal simulation HDF5 (only when load mode) ---
    if normal_simulation_mode == 'load':
        print("\n  [Normal Simulation - Load File]")
        if os.path.exists(previous_normal_hdf5):
            print(f"  ✓ Normal HDF5: {previous_normal_hdf5}")
        else:
            print(f"  ✗ Normal HDF5: {previous_normal_hdf5}  ← NOT FOUND")
            _missing.append(previous_normal_hdf5)

    # --- Result ---
    print()
    if _missing:
        print("❌ PRE-FLIGHT FAILED — the following required files are missing:")
        for _f in _missing:
            print(f"     • {_f}")
        print("\nAborting. Fix the paths or missing files before re-running.")
        print("="*70)
        sys.exit(1)
    else:
        print("✓ All required files found — proceeding with simulation.")
        print("="*70)

    
    # =========================================================================
    # BOUNDARY CONDITIONS - BFS setup
    # =========================================================================
    
    bc = BoundaryConditions()
    
    # Left boundary: Mixed (handled by _apply_bfs_inlet method)
    # Below step_height: wall (u=v=0)
    # Above step_height: parabolic inlet profile
    bc.u_boundaries['left'] = BoundaryCondition('dirichlet', 0.0)  # Placeholder
    bc.v_boundaries['left'] = BoundaryCondition('dirichlet', 0.0)
    
    # Right boundary: Pressure outlet
    bc.u_boundaries['right'] = BoundaryCondition('neumann', 0.0)
    bc.v_boundaries['right'] = BoundaryCondition('neumann', 0.0)
    bc.p_boundaries['right'] = BoundaryCondition('dirichlet', 0.0)
    
    # Top and bottom: No-slip walls
    bc.u_boundaries['top'] = BoundaryCondition('dirichlet', 0.0)
    bc.u_boundaries['bottom'] = BoundaryCondition('dirichlet', 0.0)
    bc.v_boundaries['top'] = BoundaryCondition('dirichlet', 0.0)
    bc.v_boundaries['bottom'] = BoundaryCondition('dirichlet', 0.0)
    
    # Pressure: Neumann on walls and inlet
    bc.p_boundaries['left'] = BoundaryCondition('neumann', 0.0)
    bc.p_boundaries['top'] = BoundaryCondition('neumann', 0.0)
    bc.p_boundaries['bottom'] = BoundaryCondition('neumann', 0.0)
    
    # Create a single timestamped output directory for this run
    output_dir = create_timestamped_output_dir()
    print(f"All outputs will be saved to: {output_dir}")

    # =========================================================================
    # PART 1: NORMAL SIMULATION (BASELINE) - OPTIONAL
    # =========================================================================
    
    solver_normal = None
    iterations_normal = None
    elapsed_time_normal = None
    
    if normal_simulation_mode == 'run':
        print("\n" + "#"*70)
        print("# PART 1: NORMAL BFS SIMULATION (BASELINE)")
        print("#"*70)
        
        solver_normal, iterations_normal, elapsed_time_normal = run_normal_simulation(
            Re=Re,
            nx=nx,
            ny=ny,
            dt=dt,
            scheme=scheme,
            convergence_criteria={'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6},
            max_iterations=max_iterations_normal,
            output_name=os.path.join(output_dir, 
                                    f"bfs_Re{Re}_{nx}x{ny}_{max_iterations_normal}_NORMAL"),
            bc=bc,
            step_height=step_height,
            h=h,
            Ub=Ub,
            lx=lx,
            ly=ly,
            relaxation_factors=relaxation_factors,
            check_interval=monitoring_interval
        )
    
    elif normal_simulation_mode == 'load':
        print("\n" + "#"*70)
        print("# PART 1: LOADING PREVIOUS NORMAL SIMULATION")
        print("#"*70)
        
        try:
            solver_normal = load_solver_from_hdf5(
                filepath=previous_normal_hdf5,
                Re=Re, nx=nx, ny=ny, dt=dt, scheme=scheme,
                lx=lx, ly=ly, bc=bc,
                step_height=step_height, h=h, Ub=Ub,
                relaxation_factors=relaxation_factors
            )
            iterations_normal = max_iterations_normal  # Assumed from filename
            elapsed_time_normal = None  # Not available from saved file
            print("✓ Normal simulation loaded successfully")
            print("⚠️  Note: Time information not available from saved file - speedup cannot be calculated")
        except Exception as e:
            print(f"❌ Failed to load normal simulation: {e}")
            print("   Continuing without normal simulation (no comparison will be generated)")
            normal_simulation_mode = 'skip'
            solver_normal = None
    
    else:  # skip
        print("\n" + "#"*70)
        print("# PART 1: SKIPPING NORMAL SIMULATION")
        print("#"*70)
        print("Normal simulation skipped - ML-accelerated only mode")
        print("No comparison plot will be generated.")
    
    # =========================================================================
    # PART 2A: GENERATE COARSE MESH SOLUTION
    # =========================================================================
    
    print("\n" + "#"*70)
    print("# PART 2A: COARSE MESH BFS SOLUTION")
    print("#"*70)

    iterations_coarse = None
    elapsed_time_coarse = None

    if coarse_simulation_mode == 'run':
        coarse_fields, iterations_coarse, elapsed_time_coarse, output_dir = generate_coarse_mesh_solution(
            Re=Re,
            lr_dim=lr_dim,
            dt=dt,
            scheme=scheme,
            convergence_criteria={'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6},
            max_iterations_coarse=max_iterations_coarse,
            output_dir=output_dir,
            bc=bc,
            step_height=step_height,
            h=h,
            Ub=Ub,
            lx=lx,
            ly=ly,
            relaxation_factors=relaxation_factors
        )
    else:  # load
        print("# LOADING PREVIOUS COARSE BFS SIMULATION")
        try:
            coarse_solver = load_solver_from_hdf5(
                filepath=previous_coarse_hdf5,
                Re=Re, nx=lr_dim, ny=lr_dim, dt=dt, scheme=scheme,
                lx=lx, ly=ly, bc=bc,
                step_height=step_height, h=h, Ub=Ub,
                relaxation_factors=relaxation_factors
            )
            coarse_fields = {
                'u': coarse_solver.Var[0, 1:-1, 1:-1].T.copy(),
                'v': coarse_solver.Var[1, 1:-1, 1:-1].T.copy(),
                'p': coarse_solver.Var[2, 1:-1, 1:-1].T.copy(),
            }
            iterations_coarse = max_iterations_coarse  # assumed from config
            elapsed_time_coarse = None
            print("✓ Coarse simulation loaded successfully")
            print("⚠️  Note: Coarse time not available from saved file")
        except Exception as e:
            print(f"❌ Failed to load coarse simulation: {e}")
            print("   Falling back to running coarse simulation...")
            coarse_fields, iterations_coarse, elapsed_time_coarse, output_dir = generate_coarse_mesh_solution(
                Re=Re,
                lr_dim=lr_dim,
                dt=dt,
                scheme=scheme,
                convergence_criteria={'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6},
                max_iterations_coarse=max_iterations_coarse,
                output_dir=output_dir,
                bc=bc,
                step_height=step_height,
                h=h,
                Ub=Ub,
                lx=lx,
                ly=ly,
                relaxation_factors=relaxation_factors
            )
    
    # =========================================================================
    # PART 2B: RUN ML-ACCELERATED FINE SIMULATION (PROGRESSIVE U-NET)
    # =========================================================================
    
    print("\n" + "#"*70)
    print("# PART 2B: ML-ACCELERATED FINE BFS SIMULATION (PROGRESSIVE U-NET)")
    print("#"*70)
    
    solver_ml, iterations_ml, elapsed_time_ml = run_ml_accelerated_fine_simulation_unet(
        coarse_fields=coarse_fields,
        Re=Re,
        nx=nx,
        ny=ny,
        lr_dim=lr_dim,
        dt=dt,
        scheme=scheme,
        convergence_criteria={'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6},
        # Use max_iterations_fine_ml for ML-accelerated fine simulation
        max_iterations_fine=max_iterations_fine_ml,
        output_name=os.path.join(output_dir, 
                                f"bfs_Re{Re}_{nx}x{ny}_{max_iterations_coarse}_coarse_{max_iterations_fine_ml}_fine_UNET"),
        stage_dims=stage_dims,
        model_name_pattern=model_name_pattern,
        bc=bc,
        step_height=step_height,
        h=h,
        Ub=Ub,
        lx=lx,
        ly=ly,
        relaxation_factors=relaxation_factors,
        normal_solver=solver_normal,  # Pass normal solver for monitoring
        check_interval=monitoring_interval            # Check every monitoring_interval iterations
    )
    
    # =========================================================================
    # PART 3: EXTRACTING CENTERLINES (Final Comparison) - CONDITIONAL
    # =========================================================================
    
    if normal_simulation_mode != 'skip' and solver_normal is not None:
        print("\n" + "#"*70)
        print("# PART 3: EXTRACTING CENTERLINES")
        print("#"*70)
        print("Extracting centerline data from ML-accelerated simulation...")
        ml_centerlines = extract_centerlines(solver_ml, nx, ny)
        print("Extracting centerline data from normal simulation...")
        normal_centerlines = extract_centerlines(solver_normal, nx, ny)
        
        # =========================================================================
        # PART 4: PLOTTING COMPARISON
        # =========================================================================
        
        print("\n" + "#"*70)
        print("# PART 4: PLOTTING COMPARISON")
        print("#"*70)
        
        plot_centerline_comparison(
            ml_centerlines, 
            normal_centerlines, 
            Re=Re,
            save_path=os.path.join(output_dir, 
                                  f"bfs_centerline_comparison_UNET_Re{Re}_{nx}x{ny}_coarse{max_iterations_coarse}_ML{max_iterations_fine_ml}_NORMAL{max_iterations_normal}.png"),
            bc=bc,
            show=False
        )
    else:
        print("\n" + "#"*70)
        print("# PART 3 & 4: COMPARISON SKIPPED")
        print("#"*70)
        print("Normal simulation not available - skipping comparison plots")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "="*70)
    print("FINAL SUMMARY - BFS ML-ACCELERATED SIMULATION (PROGRESSIVE U-NET)")
    print("="*70)
    print(f"Reynolds Number: {Re}")
    print(f"Mesh: {nx}x{ny}")
    print(f"BFS Geometry: Lx={lx}, Ly={ly}, Step Height={step_height}, h={h}")
    
    # Calculate total ML-accelerated time (coarse + fine)
    if elapsed_time_coarse is not None:
        total_ml_time = elapsed_time_coarse + elapsed_time_ml
    else:
        total_ml_time = elapsed_time_ml  # coarse time not available (loaded)

    print(f"\nML-Accelerated Simulation (Progressive U-Net):")
    print(f"  Coarse mesh iterations ({lr_dim}x{lr_dim}): {iterations_coarse}")
    if elapsed_time_coarse is not None:
        print(f"  Coarse mesh time: {elapsed_time_coarse:.2f} seconds")
    else:
        print(f"  Coarse mesh time: Not available (loaded from file)")
    print(f"  Fine mesh iterations ({nx}x{ny}): {iterations_ml}")
    print(f"  Fine mesh time: {elapsed_time_ml:.2f} seconds")
    print(f"  Total ML time (coarse + fine): {total_ml_time:.2f} seconds")
    
    if normal_simulation_mode != 'skip' and solver_normal is not None:
        print(f"\nNormal Simulation ({normal_simulation_mode}):")
        print(f"  Iterations: {iterations_normal}")
        if elapsed_time_normal is not None and elapsed_time_normal > 0:
            print(f"  Time: {elapsed_time_normal:.2f} seconds")
            print(f"\nSpeedup Factor: {elapsed_time_normal/total_ml_time:.2f}x")
            print(f"  (Normal time / Total ML time: {elapsed_time_normal:.2f}s / {total_ml_time:.2f}s)")
        else:
            print(f"  Time: Not available (loaded from file)")
            print(f"  (Speedup calculation requires running normal simulation, not loading)")
        print(f"Iteration Reduction (fine mesh only): {iterations_normal - iterations_ml} iterations saved")
    else:
        print(f"\nNormal Simulation: SKIPPED (no comparison)")
    
    print(f"\nAll outputs saved to: {output_dir}")
    print("="*70)
    print("\n✓ BFS ML-Accelerated Simulation Complete (Progressive U-Net)!")
    print("  Testing generalization of lid-driven cavity trained U-Net models on BFS geometry")
    print("="*70)
    
    # Save summary to file
    summary_info = {
        "Configuration": {
            "Reynolds Number": str(Re),
            "Resolution (Fine)": f"{nx}x{ny}",
            "Resolution (Coarse)": f"{lr_dim}x{lr_dim}",
            "Domain Size": f"{lx} x {ly}",
            "Step Height": str(step_height),
            "Channel Height (h)": str(h),
            "Diff. Scheme": scheme,
            "Time Step": str(dt),
            "Normal Simulation Mode": normal_simulation_mode
        },
        "ML Acceleration Settings (Progressive U-Net)": {
            "Model Pattern": model_name_pattern,
            "Stages": f"{lr_dim} -> {' -> '.join(map(str, stage_dims))}",
            "Normalization": "Per-sample Z-score",
            "Input Channels": "5 (u, v, p, x̄, ȳ)",
        },
        "Results": {
            "Coarse Iterations": str(iterations_coarse),
            "Coarse Time (s)": f"{elapsed_time_coarse:.2f}" if elapsed_time_coarse is not None else "Not available (loaded)",
            "ML+Fine Iterations": str(iterations_ml),
            "ML+Fine Time (s)": f"{elapsed_time_ml:.2f}",
            "Total ML Time (s)": f"{total_ml_time:.2f}",
            "Output Directory": output_dir
        }
    }
    
    # Add normal simulation results if available
    if normal_simulation_mode != 'skip' and iterations_normal is not None:
        summary_info["Results"]["Normal Iterations"] = str(iterations_normal)
        summary_info["Results"]["Normal Mode"] = normal_simulation_mode
        if elapsed_time_normal is not None and elapsed_time_normal > 0:
            summary_info["Results"]["Normal Time (s)"] = f"{elapsed_time_normal:.2f}"
            summary_info["Results"]["Speedup Factor"] = f"{elapsed_time_normal/total_ml_time:.2f}x"
        else:
            summary_info["Results"]["Normal Time (s)"] = "Not available (loaded from file)"
            summary_info["Results"]["Speedup Factor"] = "Cannot calculate (time not available)"
        summary_info["Results"]["Iterations Saved (Fine only)"] = f"{iterations_normal - iterations_ml}"
    
    save_run_summary(os.path.join(output_dir, "run_summary.txt"), summary_info)
