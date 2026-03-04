
"""
BFS ML-Accelerated CFD Simulation

This script runs Backward-Facing Step (BFS) simulations with ML acceleration:
1. Run coarse mesh simulation (10x10)
2. Use pretrained autoencoder to super-resolve to fine mesh (400x400)
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
        with open(filepath, 'w') as f:
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



def standardize_with_stats(arr, mean, std):
    """Standardize array with given mean and std"""
    std = 1e-8 if std == 0 else std
    return (arr - mean) / std


def inverse_standardize(arr, mean, std):
    """Inverse standardization"""
    return arr * std + mean


# ==============================================================================
# CoordConv Helpers
# ==============================================================================

def make_coord_channels(grid_dim: int, Lx: float, Ly: float) -> np.ndarray:
    """
    Build two coordinate grids (x and y) using relative coordinates:
        x_channel = x / Lx  ∈ [0, 1]
        y_channel = y / Ly  ∈ [0, 1]

    Both channels always span [0, 1] regardless of domain size — the model
    sees identical coordinate ranges for LDC (1×1) and BFS (20×1.94).
    No reference length or stats file needed.

    Args:
        grid_dim : number of grid points (H = W)
        Lx, Ly   : physical domain dimensions

    Returns:
        (grid_dim, grid_dim, 2) float32 array  [x_rel_coord, y_rel_coord]
    """
    xs = np.linspace(0.0, 1.0, grid_dim, dtype=np.float32)  # x / Lx
    ys = np.linspace(0.0, 1.0, grid_dim, dtype=np.float32)  # y / Ly
    xx, yy = np.meshgrid(xs, ys)           # (H, W) each
    return np.stack([xx, yy], axis=-1)     # (H, W, 2)


def append_coords_batch(field_batch: np.ndarray,
                        domain_sizes: np.ndarray,
                        grid_dim: int) -> np.ndarray:
    """
    Appends relative coordinate channels (x/Lx, y/Ly) to a field batch.

    Supports any number of input channels C:
        Input : (N, H, W, C)
        Output: (N, H, W, C+2)  — last two channels are x/Lx and y/Ly

    Args:
        field_batch  : (N, H, W, C)  normalised field values (any C ≥ 1)
        domain_sizes : (N, 2)        [[Lx, Ly], ...] per sample
        grid_dim     : int           H = W = grid_dim

    Returns:
        (N, H, W, C+2)
    """
    N = field_batch.shape[0]
    C = field_batch.shape[-1]
    result = np.empty((N, grid_dim, grid_dim, C + 2), dtype=np.float32)
    result[..., :C] = field_batch
    for i in range(N):
        Lx_i = float(domain_sizes[i, 0])
        Ly_i = float(domain_sizes[i, 1])
        result[i, ..., C:] = make_coord_channels(grid_dim, Lx_i, Ly_i)
    return result


def plot_ml_superres_comparison(lr_fields: Dict[str, np.ndarray],
                                hr_fields: Dict[str, np.ndarray],
                                lx: float, ly: float,
                                lr_dim: int, hr_dim: int,
                                save_path: str = None,
                                hr_true_fields: Dict[str, np.ndarray] = None):
    """
    Plot CoordConv super-resolution result.

    Produces three separate PNG files — one per field (U, V, P).
    If *hr_true_fields* is provided (ground truth from a normal simulation h5),
    each figure has 3 panels: LR Input | HR Ground Truth | HR Prediction.
    Otherwise, 2 panels: LR Input | HR Prediction.

    File names are derived from *save_path* by stripping the extension and
    appending ``_U.png``, ``_V.png``, ``_P.png``.

    Args:
        lr_fields      : dict with 'u','v','p' arrays of shape (lr_dim, lr_dim)
        hr_fields      : dict with 'u','v','p' arrays of shape (hr_dim, hr_dim)  — ML prediction
        lx, ly         : Physical domain dimensions
        lr_dim, hr_dim : grid dimensions
        save_path      : Base path; extension stripped, per-field suffix appended.
        hr_true_fields : Optional dict with 'u','v','p' ground truth HR arrays (hr_dim, hr_dim)
    """
    field_specs = [
        ('u', 'U Velocity', 'RdBu'),
        ('v', 'V Velocity', 'RdBu'),
        ('p', 'Pressure',   'viridis'),
    ]

    # Derive a base path without extension so we can append _U / _V / _P
    if save_path:
        base_path = os.path.splitext(save_path)[0]
        os.makedirs(os.path.dirname(base_path) or '.', exist_ok=True)
    else:
        base_path = None

    # Build meshgrids for LR and HR grids (physical coordinates, matching _plot_contours)
    x_lr = np.linspace(0, lx, lr_dim)
    y_lr = np.linspace(0, ly, lr_dim)
    X_lr, Y_lr = np.meshgrid(x_lr, y_lr)

    x_hr = np.linspace(0, lx, hr_dim)
    y_hr = np.linspace(0, ly, hr_dim)
    X_hr, Y_hr = np.meshgrid(x_hr, y_hr)

    has_gt = hr_true_fields is not None
    n_cols  = 3 if has_gt else 2
    fig_w   = 15 if not has_gt else 22   # wider for 3 panels

    for comp, comp_label, cmap in field_specs:
        fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, 8))

        panels = [(lr_fields, X_lr, Y_lr, f'LR Input  ({lr_dim}×{lr_dim})')]
        if has_gt:
            panels.append((hr_true_fields, X_hr, Y_hr, f'HR Ground Truth  ({hr_dim}×{hr_dim})'))
        panels.append((hr_fields, X_hr, Y_hr, f'HR Prediction  ({hr_dim}×{hr_dim})'))

        for ax, (fields, X, Y, col_title) in zip(axes, panels):
            fld = fields[comp]   # (ny, nx) — already transposed (Var.T), matches meshgrid layout
            im  = ax.contourf(X, Y, fld, levels=20, cmap=cmap)
            ax.set_title(col_title)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_aspect('equal')
            plt.colorbar(im, ax=ax)

        comp_upper = comp.upper()
        plt.suptitle(f'CoordConv Super-Resolution — {comp_label}', fontsize=16)
        plt.tight_layout()

        if base_path:
            out_path = f"{base_path}_{comp_upper}.png"
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Super-resolution plot saved: {out_path}")

        plt.close(fig)


# ==============================================================================
# Classes for CFD Simulation
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
# CFDSolver Class with BFS Support
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
              callback = None, callback_interval: int = 1000):
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
            
            # Run callback if provided (e.g., for ML vs Normal monitoring)
            if callback is not None and count % callback_interval == 0:
                callback(self, count)
            
            if verbose and count % 100 == 0:
                print(f"{count}", end="")
            
            converged, rms_residuals = self._convergence_check(verbose and count % 100 == 0)
            if count % 100 == 0:
                self.residual_history['u'].append(rms_residuals[0])
                self.residual_history['v'].append(rms_residuals[1])
                self.residual_history['p'].append(rms_residuals[2])
        
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
                             color='white', linewidth=0.5, density=1.5)
        
        plt.suptitle(f'Backward-Facing Step Flow (Re={self.fluid.Re})', fontsize=16)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
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
# ML Helper Classes
# ==============================================================================

class SuperResolutionAE(Model):
    """
    Minimal wrapper for loading and inference of the multi-field CoordConv model.
      Input : (N, LR, LR, 5)  — [U_norm | V_norm | P_norm | x/Lx | y/Ly]
      Output: (N, HR, HR, 3)  — [U_pred_norm | V_pred_norm | P_pred_norm]
    """
    def __init__(self, encoder_lr, decoder_hr, **kwargs):
        super().__init__(**kwargs)
        self.encoder_lr = encoder_lr
        self.decoder_hr = decoder_hr

    def call(self, inputs, training=False):
        z = self.encoder_lr(inputs, training=training)
        recon_hr = self.decoder_hr(z, training=training)
        return recon_hr


# ==============================================================================
# ML-Accelerated CFD Workflow Functions
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
        Dictionary with 'u', 'v', 'p' fields of shape (lr_dim, lr_dim)
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
    
    return coarse_fields, time_elapsed


def ml_super_resolution(coarse_fields: Dict[str, np.ndarray],
                        lr_dim: int, hr_dim: int,
                        stats_file: str = None, encoder_file: str = None, decoder_file: str = None,
                        lx: float = 1.0, ly: float = 1.0,
                        plot_save_path: str = None,
                        normal_sim_h5: str = None, re: float = None) -> Dict[str, np.ndarray]:
    """
    Step 2: CoordConv super-resolution with multi-field joint prediction.

    All three fields (U, V, P) are super-resolved in a single forward pass:
        1. Stack LR fields  : (LR, LR, 3)  [U | V | P]
        2. Normalize        : per-component LR mean/std  →  (LR, LR, 3)
        3. Append coords    : (LR, LR, 3)  →  (LR, LR, 5)  [U|V|P|x/Lx|y/Ly]
        4. Predict          : encoder → latent → decoder  →  (HR, HR, 3)
        5. Denormalize      : per-component  →  {u, v, p} physical fields

    Joint prediction lets the model exploit U–V–P coupling (pressure Poisson
    equation, continuity) for more accurate pressure super-resolution.

    Args:
        coarse_fields: Dict with 'u', 'v', 'p' arrays of shape (lr_dim, lr_dim)
        lr_dim, hr_dim: grid dimensions
        stats_file: Ignored (API compatibility)
        encoder_file, decoder_file: .h5 model paths
        lx, ly: physical domain dimensions
        plot_save_path: optional path to save LR vs HR comparison plot

    Returns:
        Dict with 'u', 'v', 'p' arrays of shape (hr_dim, hr_dim)
    """
    print(f"\n{'='*70}")
    print(f"STEP 2: Multi-field CoordConv Super-Resolution ({lr_dim}x{lr_dim} -> {hr_dim}x{hr_dim})")
    print(f"  Physical domain : Lx={lx}, Ly={ly}  (aspect {lx/ly:.2f}:1)")
    print(f"  Input channels  : U + V + P + x/Lx + y/Ly  (5 channels)")
    print(f"  Output channels : U + V + P  (3 channels, single forward pass)")
    print(f"  Normalisation   : per-sample per-component LR mean/std")
    print(f"{'='*70}")

    # Load models
    print(f"\nLoading encoder from '{encoder_file}'...")
    print(f"Loading decoder from '{decoder_file}'...")
    try:
        encoder_lr = tf.keras.models.load_model(encoder_file, compile=False)
        decoder_hr = tf.keras.models.load_model(decoder_file, compile=False)
        inference_model = SuperResolutionAE(encoder_lr, decoder_hr)
        print(f"  ✓ Models loaded successfully")
    except (IOError, OSError) as e:
        print(f"❌ FATAL: Error loading models: {e}")
        raise

    # Stack U, V, P into a single (1, LR, LR, 3) tensor
    x_lr_uvp = np.stack([
        coarse_fields['u'].astype(np.float32),
        coarse_fields['v'].astype(np.float32),
        coarse_fields['p'].astype(np.float32),
    ], axis=-1)[np.newaxis]   # (1, LR, LR, 3)

    # Per-component normalization: mean/std over spatial dims, one per channel
    mean_i = np.mean(x_lr_uvp, axis=(1, 2), keepdims=True)   # (1, 1, 1, 3)
    std_i  = np.std( x_lr_uvp, axis=(1, 2), keepdims=True)   # (1, 1, 1, 3)
    std_i  = np.where(std_i < 1e-8, 1e-8, std_i)

    x_lr_norm = (x_lr_uvp - mean_i) / std_i   # (1, LR, LR, 3)

    # Bilinear upsample LR norm to HR — serves as the residual baseline
    x_lr_up = tf.image.resize(x_lr_norm, [hr_dim, hr_dim], method='bilinear').numpy()  # (1, HR, HR, 3)

    # Append coord channels: (1, LR, LR, 3) → (1, LR, LR, 5)
    domain_sizes_batch = np.array([[lx, ly]], dtype=np.float32)
    x_lr_with_coords   = append_coords_batch(x_lr_norm, domain_sizes_batch, lr_dim)

    # Single forward pass → (1, HR, HR, 3) residual prediction
    print(f"\nRunning single forward pass ({lr_dim}x{lr_dim}x5 → {hr_dim}x{hr_dim}x3)...")
    pred_residual = inference_model.predict(x_lr_with_coords, verbose=0)[0]   # (HR, HR, 3)

    # Reconstruct HR norm = bilinear baseline + predicted residual, then denormalize
    pred_hr_norm = x_lr_up[0] + pred_residual             # (HR, HR, 3)
    pred_real    = pred_hr_norm * std_i[0, 0] + mean_i[0, 0]   # (HR, HR, 3)

    hr_fields = {}
    print(f"\n{'─'*72}")
    print(f"  {'Field':<6}  {'Input range':>20}  {'Norm range':>16}  {'Output range':>20}")
    print(f"{'─'*72}")
    for c_idx, comp in enumerate(['u', 'v', 'p']):
        raw   = x_lr_uvp[0, :, :, c_idx]
        norm  = x_lr_norm[0, :, :, c_idx]
        field = pred_real[..., c_idx]

        # Guard against NaN/Inf
        if np.isnan(field).any() or np.isinf(field).any():
            nan_c = np.isnan(field).sum()
            inf_c = np.isinf(field).sum()
            print(f"  ⚠️  {comp.upper()}: {nan_c} NaN, {inf_c} Inf — replacing with 0")
            field = np.nan_to_num(field, nan=0.0, posinf=0.0, neginf=0.0)

        hr_fields[comp] = field
        print(f"  {comp.upper():<6}  "
              f"[{raw.min():>9.4f}, {raw.max():>9.4f}]  "
              f"[{norm.min():>6.2f}, {norm.max():>6.2f}]  "
              f"[{field.min():>9.4f}, {field.max():>9.4f}]")
    print(f"{'─'*72}")
    
    # -------------------------------------------------------------------------
    # LR vs HR DIAGNOSTIC PLOT
    # -------------------------------------------------------------------------
    if plot_save_path is not None:
        print(f"\n  Generating super-resolution comparison plot...")

        # Optionally load ground truth HR from a normal simulation h5 file
        hr_true_fields = None
        if normal_sim_h5 is not None and re is not None and os.path.isfile(normal_sim_h5):
            try:
                with h5py.File(normal_sim_h5, 'r') as _f:
                    # Key may use int Re (100) or float Re (100.0) — try both
                    _key_int   = f"Re{int(re)}_mesh{hr_dim}x{hr_dim}"
                    _key_float = f"Re{float(re)}_mesh{hr_dim}x{hr_dim}"
                    _key = _key_int if _key_int in _f else (_key_float if _key_float in _f else None)
                    if _key is None:
                        # Last resort: pick first key whose dimensions match
                        _suffix = f"_mesh{hr_dim}x{hr_dim}"
                        _key = next((k for k in _f.keys() if k.endswith(_suffix)), None)
                    if _key:
                        hr_true_fields = {
                            c: _f[_key][c][()].astype(np.float32).reshape(hr_dim, hr_dim)
                            for c in ['u', 'v', 'p']
                        }
                        print(f"  ✓ Ground truth HR loaded from '{normal_sim_h5}'  (key: {_key})")
                    else:
                        print(f"  ⚠️  No matching key for Re={re} {hr_dim}x{hr_dim} in '{normal_sim_h5}' — plotting without GT")
            except Exception as _e:
                print(f"  ⚠️  Could not load ground truth from h5: {_e} — plotting without GT")

        plot_ml_superres_comparison(
            lr_fields=coarse_fields,
            hr_fields=hr_fields,
            lx=lx, ly=ly,
            lr_dim=lr_dim,
            hr_dim=hr_dim,
            save_path=plot_save_path,
            hr_true_fields=hr_true_fields
        )

    print(f"\n  ✓ Super-resolution complete")
    return hr_fields


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
                                     callback = None,
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
        callback: Function to run at intervals
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
        Tuple of (coarse_fields, output_dir)
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
    coarse_fields, elapsed_time_coarse = run_coarse_simulation(
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
    
    return coarse_fields, output_dir, elapsed_time_coarse


def run_ml_accelerated_fine_simulation(
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
    stats_file: str = None,
    encoder_file: str = None,
    decoder_file: str = None,
    bc: Optional[BoundaryConditions] = None,
    step_height: float = 1.0,
    h: float = 2.0,
    Ub: float = 1.0,
    lx: float = 10.0,
    ly: float = 3.0,
    relaxation_factors: Dict[str, float] = None,
    normal_solver = None,
    check_interval: int = 1000,
    normal_sim_h5: str = None
) -> tuple:
    """
    Run ML-accelerated fine BFS simulation using coarse mesh solution
    """

    print(f"\n{'#'*70}")
    print(f"# ML-ACCELERATED FINE BFS SIMULATION")
    print(f"# Re={Re}, Target Resolution={nx}x{ny}")
    print(f"# Using coarse solution from {lr_dim}x{lr_dim}")
    print(f"# CoordConv geometry-aware inference (Lx={lx}, Ly={ly})")
    print(f"# Normalisation: per-sample LR mean/std")
    print(f"{'#'*70}\n")
    
    # Set default file paths if not provided
    if stats_file is None:
        stats_file = f"standardization_stats_{lr_dim}to{nx}_swish_trained_upto_700_multiBC.txt"
    if encoder_file is None:
        encoder_file = f"coord_encoder{lr_dim}_to_{nx}_swish_trained_upto_700_multiBC.h5"
    if decoder_file is None:
        decoder_file = f"coord_decoder{nx}_from_{lr_dim}_swish_trained_upto_700_multiBC.h5"
    if output_name is None:
        output_name = f"bfs_Re{Re}_{nx}x{ny}"
    
    # Verify files exist (Moved to main block)
    # print("Checking required ML model files...")
    # for fname, desc in [(stats_file, "Stats file"), 
    #                     (encoder_file, "Encoder model"), 
    #                     (decoder_file, "Decoder model")]:
    #     if os.path.exists(fname):
    #         print(f"  ✓ {desc}: {fname}")
    #     else:
    #         print(f"  ✗ {desc}: {fname} NOT FOUND")
            # We don't raise error here to allow dry runs if files missing, but it will fail later
    
    # STEP 1: ML super-resolution
    _pipeline_plot_path = os.path.join(
        os.path.dirname(output_name) if os.path.dirname(output_name) else '.',
        "ml_pipeline_stages.png"
    )
    hr_fields = ml_super_resolution(
        coarse_fields=coarse_fields,
        lr_dim=lr_dim,
        hr_dim=nx,
        stats_file=stats_file,
        encoder_file=encoder_file,
        decoder_file=decoder_file,
        lx=lx,
        ly=ly,
        plot_save_path=_pipeline_plot_path,
        normal_sim_h5=normal_sim_h5,
        re=Re
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
# Centerline Extraction and Plotting
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


def plot_centerline_comparison(ml_centerlines: Dict, normal_centerlines: Dict, 
                               Re: float, save_path: str = None, 
                               bc: Optional[BoundaryConditions] = None,
                               show: bool = True,
                               iteration: int = None, metrics: Dict = None):
    """
    Plot centerline comparison between ML-accelerated and normal BFS simulations
    
    Args:
        ml_centerlines: Centerline data from ML-accelerated simulation
        normal_centerlines: Centerline data from normal simulation
        Re: Reynolds number
        save_path: Optional path to save the figure
        bc: BoundaryConditions object (optional, for display in plot)
        show: Whether to display the plot window
        iteration: Iteration number to display in title
        metrics: Optional dictionary of computed metrics to display
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
             'b-o', linewidth=2, markersize=4, label='ML-Accelerated', alpha=0.7)
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
             'b-o', linewidth=2, markersize=4, label='ML-Accelerated', alpha=0.7)
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
        fig.suptitle(f'BFS Centerline Velocity Comparison (Re={Re})\n{bc_summary}', 
                    fontsize=14, fontweight='bold', y=0.98)
    else:
        fig.suptitle(f'BFS Centerline Velocity Comparison (Re={Re})', 
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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
    stats_file: str = None,
    encoder_file: str = None,
    decoder_file: str = None,
    bc: Optional[BoundaryConditions] = None,
    step_height: float = 1.0,
    h: float = 2.0,
    Ub: float = 1.0,
    lx: float = 10.0,
    ly: float = 3.0,
    relaxation_factors: Dict[str, float] = None,
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
    
    if bc is None:
        bc = BoundaryConditions()
        # Setup default BCs
        bc.u_boundaries['left'] = BoundaryCondition('dirichlet', 0.0)
        bc.v_boundaries['left'] = BoundaryCondition('dirichlet', 0.0)
        bc.u_boundaries['right'] = BoundaryCondition('neumann', 0.0)
        bc.v_boundaries['right'] = BoundaryCondition('neumann', 0.0)
        bc.p_boundaries['right'] = BoundaryCondition('dirichlet', 0.0)
        bc.u_boundaries['top'] = BoundaryCondition('dirichlet', 0.0)
        bc.u_boundaries['bottom'] = BoundaryCondition('dirichlet', 0.0)
        bc.v_boundaries['top'] = BoundaryCondition('dirichlet', 0.0)
        bc.v_boundaries['bottom'] = BoundaryCondition('dirichlet', 0.0)
        bc.p_boundaries['left'] = BoundaryCondition('neumann', 0.0)
        bc.p_boundaries['top'] = BoundaryCondition('neumann', 0.0)
        bc.p_boundaries['bottom'] = BoundaryCondition('neumann', 0.0)

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
    def normal_checkpoint_callback(solver, iteration): 
        save_checkpoint(iteration, solver, checkpoint_dir, "normal")
    
    output_name = os.path.join(output_dir, f"bfs_normal_Re{Re}_{nx}x{ny}")
    
    normal_iter, normal_time = normal_solver.solve(
        output_name, verbose=True,
        callback=normal_checkpoint_callback,
        callback_interval=checkpoint_interval
    )
    
    print(f"\n✓ Normal simulation complete: {normal_iter} iterations, {normal_time:.2f}s")
    
    # ====================
    # STEP 2: Coarse Simulation
    # ====================
    print(f"\n{'='*80}")
    print(f"STEP 2: Running Coarse Simulation")
    print(f"{'='*80}")
    
    coarse_fields, coarse_time = run_coarse_simulation(
        Re=Re, lr_dim=lr_dim, dt=dt, scheme=scheme,
        convergence_criteria=convergence_criteria,
        max_iterations=max_iterations,
        output_dir=output_dir,
        bc=bc,
        step_height=step_height, h=h, Ub=Ub,
        lx=lx, ly=ly,
        relaxation_factors=relaxation_factors
    )
    
    # ====================
    # STEP 3: ML Super-Resolution
    # ====================
    print(f"\n{'='*80}")
    print(f"STEP 3: ML Super-Resolution")
    print(f"{'='*80}")
    
    hr_fields = ml_super_resolution(
        coarse_fields=coarse_fields,
        lr_dim=lr_dim, hr_dim=nx,
        stats_file=stats_file,
        encoder_file=encoder_file,
        decoder_file=decoder_file,
        lx=lx, ly=ly,
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
    def ml_checkpoint_callback(solver, iteration):
        save_checkpoint(iteration, solver, checkpoint_dir, "ml_accelerated")
    
    output_name_ml = os.path.join(output_dir, f"bfs_ml_accelerated_Re{Re}_{nx}x{ny}")
    ml_iter, ml_time = ml_solver.solve(
        output_name_ml, verbose=True,
        callback=ml_checkpoint_callback,
        callback_interval=checkpoint_interval
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
                                  iteration=iteration, metrics=metrics, show=False)
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
    print(f"\n{'#'*80}")
    print(f"# COMPARISON WORKFLOW COMPLETE")
    print(f"#")
    print(f"# Normal simulation: {normal_iter} iterations ({normal_time:.2f}s)")
    print(f"# ML-accelerated:    {ml_iter} iterations ({coarse_time + ml_time:.2f}s = {coarse_time:.2f}s coarse + {ml_time:.2f}s fine)")
    print(f"# Speedup:           {normal_time/(coarse_time + ml_time):.2f}x")
    print(f"# Iteration savings: {normal_iter - ml_iter} iterations ({100*(1-ml_iter/normal_iter):.1f}%)")
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
        'speedup': normal_time / (coarse_time + ml_time)
    }


# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    """
    Example: Run ML-accelerated BFS simulation for configurable Re and mesh
    
    Required files (ensure these are in the same directory):
    - standardization_stats_10to400_swish_trained_upto_700_multiBC.txt
    - coord_encoder10_to_400_swish_trained_upto_700_multiBC.h5
    - coord_decoder400_from_10_swish_trained_upto_700_multiBC.h5
    """
    
    # =========================================================================
    # CONFIGURATION - Customize these parameters
    # =========================================================================
    
    # Reynolds number
    Re = 400
    
    # Fine mesh dimensions
    nx = 400
    ny = 400
    
    # Coarse mesh dimension
    lr_dim = 10
    
    # BFS geometry parameters
    lx = 20.0          # Domain length in x
    ly = 1.94          # Domain length in y
    step_height = 0.94 # Step height
    h = 1.0            # Channel height above step
    Ub = 1.0           # Bulk velocity
    
    # Time step
    dt = 1e-3
    
    # Numerical scheme ('QUICK' or 'UPWIND')
    scheme = 'UPWIND'
    
    # Under-relaxation factors (important for BFS stability)
    relaxation_factors = {
        'u': 0.3,
        'v': 0.2,
        'p': 0.01
    }
    
    # Maximum iterations for different simulations
    max_iterations_coarse = 130000   # Max iterations for coarse mesh (10x10)
    max_iterations_fine_ml = 200     # Max iterations for fine mesh with ML initialization
    max_iterations_normal = 300   # Max iterations for normal simulation
    
    # Iteration interval for monitoring and saving plots
    monitoring_interval = 100

    # Model file suffix
    other_details = "ae_10_to_400_trained_LDCs_and_BFS_u,v,p,x,y channels_together_without stats"
              
    # =========================================================================
    # ASPECT RATIO CORRECTION FLAG


    # =========================================================================
    # =========================================================================
    # ADDITIONAL ML IMPROVEMENTS
    # =========================================================================
    
    # Per-sample normalisation is used at inference: each field is normalised
    # by its own LR mean/std — no global stats blending needed.

    # =========================================================================
    # LOAD MODE (Optional): Skip coarse + fine simulation, load existing H5
    # =========================================================================
    # Set to the path of an existing ML-accelerated fine simulation H5 file
    # to load it directly instead of re-running the simulation.
    # Leave as None to run normally (coarse + ML-accelerated fine).
    #
    # Example:
    #   load_ml_accelerated_h5 = "outputs/01-12-2025-04-08-57 (BFS)/bfs_Re400_400x400_100000_coarse_5000_fine_ML_accelerated.h5"
    load_ml_accelerated_h5 = None

    # =========================================================================
    # LOAD MODE (Optional): Load existing COARSE simulation H5 (skip coarse run,
    # still runs ML SR + fine simulation using the loaded coarse fields)
    # =========================================================================
    # Set to the path of an existing coarse simulation H5 file.
    # The coarse fields will be loaded and passed directly to the ML model.
    # Leave as None to run the coarse simulation fresh.
    # NOTE: Ignored if load_ml_accelerated_h5 is set (that skips everything).
    #
    # Example:
    load_coarse_h5 = r"C:\Users\amirm\Downloads\required files\bfs_coarse_Re400_10x10_130000_coarse_iterations.h5"
    #load_coarse_h5 = None

    # =========================================================================
    # LOAD MODE (Optional): Load existing NORMAL (reference) simulation H5
    # =========================================================================
    # Set to the path of an existing normal fine simulation H5 file to skip
    # running it again.  Leave as None to run the normal simulation fresh.
    #
    # Example:
    load_normal_simulation_h5 = r"C:\Users\amirm\Downloads\required files\bfs_Re400_400x400_125000_NORMAL_normal.h5"
    #load_normal_simulation_h5 = None

    # =========================================================================


    print(f"\n{'='*70}")
    print(f"CONFIGURATION SUMMARY")
    print(f"{'='*70}")
    print(f"Reynolds Number: {Re}")
    print(f"Fine Mesh: {nx}×{ny}")
    print(f"Coarse Mesh: {lr_dim}×{lr_dim}")
    print(f"BFS Geometry: Lx={lx}, Ly={ly} (aspect ratio: {lx/ly:.2f}:1)")
    print(f"CoordConv: geometry-aware coord channels + per-sample LR normalisation")
    print(f"Normal sim: {'LOAD from ' + load_normal_simulation_h5 if load_normal_simulation_h5 else 'RUN fresh'}")
    print(f"ML fine sim: {'LOAD from ' + load_ml_accelerated_h5 if load_ml_accelerated_h5 else 'RUN (coarse → ML SR → fine)'}")
    print(f"{'='*70}\n")
    
    # =========================================================================
    # PRE-FLIGHT CHECK: ML MODELS  (skipped in full-load mode)
    # =========================================================================

    if load_ml_accelerated_h5 is None:
        print(f"Checking required ML model files...")
        _encoder_file = f"coord_encoder{lr_dim}_to_{nx}_{other_details}.h5"
        _decoder_file = f"coord_decoder{nx}_from_{lr_dim}_{other_details}.h5"

        for fname, desc in [(_encoder_file, "Encoder model"),
                            (_decoder_file, "Decoder model")]:
            if os.path.exists(fname):
                print(f"  ✓ {desc}: {fname}")
            else:
                print(f"  ✗ {desc}: {fname} NOT FOUND")
                print("  ❌ FATAL: Required ML model file missing. Aborting simulation.")
                sys.exit(1)
    else:
        print(f"  ℹ️  Pre-flight ML model check skipped (fine sim will be loaded from file).")

    
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
    
    # =========================================================================
    # PART 1: NORMAL SIMULATION (BASELINE) — or load from H5
    # =========================================================================

    print("\n" + "#"*70)
    print("# PART 1: NORMAL BFS SIMULATION (BASELINE)")
    print("#"*70)

    # Create a single timestamped output directory for this run
    output_dir = create_timestamped_output_dir()
    print(f"All outputs will be saved to: {output_dir}")

    if load_normal_simulation_h5 is not None:
        # ------------------------------------------------------------------
        # LOAD the normal simulation from an existing H5 file
        # ------------------------------------------------------------------
        print(f"  Loading normal simulation from: {load_normal_simulation_h5}")
        if not os.path.exists(load_normal_simulation_h5):
            print(f"  ❌ FATAL: File not found: {load_normal_simulation_h5}")
            sys.exit(1)

        _mesh_n  = MeshParameters(nx=nx, ny=ny, lx=lx, ly=ly)
        _fluid_n = FluidProperties(Re=Re, rho=1.0)
        _sett_n  = SolverSettings(dt=dt, scheme=scheme,
                                  max_iterations=max_iterations_normal,
                                  convergence_criteria={'u': 1e-6, 'v': 1e-6, 'p': 1e-6},
                                  relaxation_factors=relaxation_factors)
        solver_normal = CFDSolver(_mesh_n, _fluid_n, _sett_n, bc,
                                  step_height=step_height, h=h, Ub=Ub)

        _grp_name_n = f"Re{Re}_mesh{nx}x{ny}"
        with h5py.File(load_normal_simulation_h5, 'r') as _fn:
            if _grp_name_n not in _fn:
                _avail_n = list(_fn.keys())
                print(f"  ⚠️  Group '{_grp_name_n}' not found. Available: {_avail_n}")
                if not _avail_n:
                    print("  ❌ FATAL: H5 file contains no groups."); sys.exit(1)
                _grp_name_n = _avail_n[0]
                print(f"  Using group: '{_grp_name_n}'")
            _gn = _fn[_grp_name_n]
            solver_normal.Var[0, 1:-1, 1:-1] = _gn['u'][:].reshape(ny, nx).T
            solver_normal.Var[1, 1:-1, 1:-1] = _gn['v'][:].reshape(ny, nx).T
            solver_normal.Var[2, 1:-1, 1:-1] = _gn['p'][:].reshape(ny, nx).T
        print(f"  ✓ Loaded normal u/v/p fields from '{_grp_name_n}'")

        iterations_normal   = 0
        elapsed_time_normal = 0.0
        _normal_sim_loaded  = True

    else:
        # ------------------------------------------------------------------
        # RUN the normal simulation from scratch
        # ------------------------------------------------------------------
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
        _normal_sim_loaded = False
    
    if load_ml_accelerated_h5 is not None:
        # =====================================================================
        # LOAD MODE: Load ML-accelerated fine simulation directly from H5 file
        # (skips coarse mesh + ML-accelerated fine simulation)
        # =====================================================================
        print("\n" + "#"*70)
        print("# LOAD MODE: Loading ML-accelerated simulation from H5 file")
        print("#"*70)
        print(f"  File: {load_ml_accelerated_h5}")

        if not os.path.exists(load_ml_accelerated_h5):
            print(f"  ❌ FATAL: File not found: {load_ml_accelerated_h5}")
            sys.exit(1)

        # Build a solver object to hold the loaded fields (needed for
        # centerline extraction and plotting).
        _mesh_ml  = MeshParameters(nx=nx, ny=ny, lx=lx, ly=ly)
        _fluid_ml = FluidProperties(Re=Re, rho=1.0)
        _sett_ml  = SolverSettings(dt=dt, scheme=scheme,
                                   max_iterations=max_iterations_normal,
                                   convergence_criteria={'u': 1e-6, 'v': 1e-6, 'p': 1e-6},
                                   relaxation_factors=relaxation_factors)
        solver_ml = CFDSolver(_mesh_ml, _fluid_ml, _sett_ml, bc,
                              step_height=step_height, h=h, Ub=Ub)

        # Locate the correct group inside the H5 file.
        group_name = f"Re{Re}_mesh{nx}x{ny}"
        with h5py.File(load_ml_accelerated_h5, 'r') as _f:
            if group_name not in _f:
                _available = list(_f.keys())
                print(f"  ⚠️  Group '{group_name}' not found.")
                print(f"     Available groups: {_available}")
                if not _available:
                    print("  ❌ FATAL: H5 file contains no groups.")
                    sys.exit(1)
                group_name = _available[0]
                print(f"  Using group: '{group_name}'")
            _grp   = _f[group_name]
            _u_flat = _grp['u'][:]
            _v_flat = _grp['v'][:]
            _p_flat = _grp['p'][:]

        # Fields were saved as solver.Var[k, 1:-1, 1:-1].T.flatten()
        # (shape ny×nx, row-major) — reshape and transpose back.
        solver_ml.Var[0, 1:-1, 1:-1] = _u_flat.reshape(ny, nx).T
        solver_ml.Var[1, 1:-1, 1:-1] = _v_flat.reshape(ny, nx).T
        solver_ml.Var[2, 1:-1, 1:-1] = _p_flat.reshape(ny, nx).T
        print(f"  ✓ Loaded u/v/p fields from group '{group_name}'")

        # Placeholders — no simulation was run, so no timing/iteration data.
        elapsed_time_coarse = 0.0
        elapsed_time_ml     = 0.0
        iterations_ml       = 0
        _fine_sim_loaded    = True
        _coarse_loaded      = False  # moot when fine is loaded, but keeps the variable defined

    else:
        # =====================================================================
        # PART 2A: COARSE MESH SOLUTION — load or run
        # =====================================================================

        if load_coarse_h5 is not None:
            # ------------------------------------------------------------------
            # LOAD the coarse fields from an existing H5 file
            # ------------------------------------------------------------------
            print("\n" + "#"*70)
            print("# PART 2A: LOADING COARSE SOLUTION FROM H5 FILE")
            print("#"*70)
            print(f"  File: {load_coarse_h5}")

            if not os.path.exists(load_coarse_h5):
                print(f"  \u274c FATAL: File not found: {load_coarse_h5}")
                sys.exit(1)

            _grp_name_c = f"Re{Re}_mesh{lr_dim}x{lr_dim}"
            with h5py.File(load_coarse_h5, 'r') as _fc:
                if _grp_name_c not in _fc:
                    _avail_c = list(_fc.keys())
                    print(f"  \u26a0\ufe0f  Group '{_grp_name_c}' not found. Available: {_avail_c}")
                    if not _avail_c:
                        print("  \u274c FATAL: H5 file contains no groups."); sys.exit(1)
                    _grp_name_c = _avail_c[0]
                    print(f"  Using group: '{_grp_name_c}'")
                _gc = _fc[_grp_name_c]
                # Fields saved as solver.Var[k, 1:-1, 1:-1].T.flatten() -> shape (lr_dim*lr_dim,)
                coarse_fields = {
                    'u': _gc['u'][:].reshape(lr_dim, lr_dim),
                    'v': _gc['v'][:].reshape(lr_dim, lr_dim),
                    'p': _gc['p'][:].reshape(lr_dim, lr_dim),
                }
            print(f"  \u2713 Loaded coarse u/v/p fields ({lr_dim}\u00d7{lr_dim}) from '{_grp_name_c}'")
            elapsed_time_coarse = 0.0
            _coarse_loaded = True

        else:
            # ------------------------------------------------------------------
            # RUN the coarse simulation from scratch
            # ------------------------------------------------------------------
            print("\n" + "#"*70)
            print("# PART 2A: GENERATE COARSE MESH BFS SOLUTION")
            print("#"*70)

            coarse_fields, _, elapsed_time_coarse = generate_coarse_mesh_solution(
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
            _coarse_loaded = False

        # =====================================================================
        # PART 2B: RUN ML-ACCELERATED FINE SIMULATION
        # =====================================================================

        print("\n" + "#"*70)
        print("# PART 2B: ML-ACCELERATED FINE BFS SIMULATION")
        print("#"*70)

        solver_ml, iterations_ml, elapsed_time_ml = run_ml_accelerated_fine_simulation(
            coarse_fields=coarse_fields,
            Re=Re,
            nx=nx,
            ny=ny,
            lr_dim=lr_dim,
            dt=dt,
            scheme=scheme,
            convergence_criteria={'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6},
            max_iterations_fine=max_iterations_fine_ml,
            output_name=os.path.join(output_dir,
                                     f"bfs_Re{Re}_{nx}x{ny}_{max_iterations_coarse}_coarse_ML_accelerated"),
            stats_file=f"standardization_stats_{lr_dim}to{nx}_{other_details}.txt",
            encoder_file=f"coord_encoder{lr_dim}_to_{nx}_{other_details}.h5",
            decoder_file=f"coord_decoder{nx}_from_{lr_dim}_{other_details}.h5",
            bc=bc,
            step_height=step_height,
            h=h,
            Ub=Ub,
            lx=lx,
            ly=ly,
            relaxation_factors=relaxation_factors,
            normal_solver=solver_normal,
            check_interval=monitoring_interval,
            normal_sim_h5=load_normal_simulation_h5
        )
        _fine_sim_loaded = False

    # =========================================================================
    # PART 3: EXTRACTING CENTERLINES
    # =========================================================================
    
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
                              f"bfs_centerline_comparison_Re{Re}_{nx}x{ny}_coarse{max_iterations_coarse}_ML{max_iterations_fine_ml}_NORMAL{max_iterations_normal}.png"),
        bc=bc
    )
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "="*70)
    print("FINAL SUMMARY - BFS ML-ACCELERATED SIMULATION")
    print("="*70)
    print(f"Reynolds Number: {Re}")
    print(f"Mesh: {nx}x{ny}")
    print(f"BFS Geometry: Lx={lx}, Ly={ly}, Step Height={step_height}, h={h}")
    print(f"CoordConv: geometry-aware inference — no pixel reshaping needed")

    print(f"\nNormal Simulation (reference):")
    if _normal_sim_loaded:
        print(f"  Loaded from: {load_normal_simulation_h5}")
    else:
        print(f"  Iterations: {iterations_normal}")
        print(f"  Time: {elapsed_time_normal:.2f} seconds")

    print(f"\nML-Accelerated Simulation:")
    if _fine_sim_loaded:
        print(f"  Fine mesh ({nx}x{ny}): loaded from file (no simulation run)")
        print(f"  Source: {load_ml_accelerated_h5}")
    else:
        if _coarse_loaded:
            print(f"  Coarse mesh ({lr_dim}x{lr_dim}): loaded from {load_coarse_h5}")
        else:
            print(f"  Coarse mesh iterations ({lr_dim}x{lr_dim}): {max_iterations_coarse}")
            print(f"  Coarse sim time: {elapsed_time_coarse:.2f} seconds")
        print(f"  Fine mesh iterations ({nx}x{ny}): {iterations_ml}")
        print(f"  Fine sim time: {elapsed_time_ml:.2f} seconds")
        print(f"  Total time: {elapsed_time_coarse + elapsed_time_ml:.2f} seconds")

    if _fine_sim_loaded or _normal_sim_loaded:
        _reason = "fine sim" if _fine_sim_loaded else "normal sim"
        print(f"\nSpeedup Factor: N/A ({_reason} was loaded from file)")
        print(f"Iteration Reduction: N/A ({_reason} was loaded from file)")
    else:
        _total_ml_time = elapsed_time_coarse + elapsed_time_ml
        if _total_ml_time > 0:
            print(f"\nSpeedup Factor: {elapsed_time_normal / _total_ml_time:.2f}x")
        else:
            print(f"\nSpeedup Factor: N/A (zero ML time recorded)")
        print(f"Iteration Reduction (fine mesh): {iterations_normal - iterations_ml} iterations saved")

    print(f"\nAll outputs saved to: {output_dir}")
    print("="*70)
    print("\n✓ BFS ML-Accelerated Simulation Complete!")
    print("="*70)

    # Save summary to file
    _can_speedup = (not _fine_sim_loaded) and (not _coarse_loaded) and (not _normal_sim_loaded) and (elapsed_time_coarse + elapsed_time_ml) > 0
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
            "Inference Method": "CoordConv (geometry-aware, no pixel reshaping)"
        },
        "ML Acceleration Settings": {
            "Stats File": f"standardization_stats_{lr_dim}to{nx}_{other_details}.txt",
            "Encoder File": f"coord_encoder{lr_dim}_to_{nx}_{other_details}.h5",
            "Decoder File": f"coord_decoder{nx}_from_{lr_dim}_{other_details}.h5",
            "Normalisation": "per-sample LR mean/std",
        },
        "Results": {
            "Normal Iterations": "N/A (loaded)" if _normal_sim_loaded else str(iterations_normal),
            "Normal Time (s)": "N/A (loaded)" if _normal_sim_loaded else f"{elapsed_time_normal:.2f}",
            "Normal Loaded From": load_normal_simulation_h5 if _normal_sim_loaded else "N/A",
            "Coarse Iterations": "N/A (loaded)" if (_fine_sim_loaded or _coarse_loaded) else str(max_iterations_coarse),
            "ML+Fine Iterations": "N/A (loaded)" if _fine_sim_loaded else str(iterations_ml),
            "Coarse Time (s)": "N/A (loaded)" if (_fine_sim_loaded or _coarse_loaded) else f"{elapsed_time_coarse:.2f}",
            "ML+Fine Time (s)": "N/A (loaded)" if _fine_sim_loaded else f"{elapsed_time_ml:.2f}",
            "Total ML Time (s)": "N/A (loaded)" if _fine_sim_loaded else f"{elapsed_time_coarse + elapsed_time_ml:.2f}",
            "Speedup Factor": (f"{elapsed_time_normal / (elapsed_time_coarse + elapsed_time_ml):.2f}x"
                               if _can_speedup else "N/A (one or more runs were loaded)"),
            "Iterations Saved": "N/A (loaded)" if (_fine_sim_loaded or _normal_sim_loaded) else f"{iterations_normal - iterations_ml}",
            "Coarse Loaded From": load_coarse_h5 if _coarse_loaded else "N/A",
            "Fine Loaded From": load_ml_accelerated_h5 if _fine_sim_loaded else "N/A",
            "Output Directory": output_dir
        }
    }

    save_run_summary(os.path.join(output_dir, "run_summary.txt"), summary_info)


