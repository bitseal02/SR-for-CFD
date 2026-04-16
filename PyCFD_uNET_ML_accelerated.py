

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
    """Save simulation configuration and results summary to a text file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("PYCFD ML-ACCELERATED RUN SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            for section, section_data in info.items():
                f.write(f"[{section}]\n")
                for key, value in section_data.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

        print(f"Run summary saved to: {filepath}")
    except Exception as e:
        print(f"Failed to save run summary: {e}")


# ==============================================================================
# Classes and Functions from PyCFD (7).py
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
            'left': BoundaryCondition('dirichlet', 0.0),
            'right': BoundaryCondition('dirichlet', 0.0),
            'top': BoundaryCondition('dirichlet', 1.0),  # Moving lid
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
    def __init__(self, nx: int = 100, ny: int = 100, lx: float = 1.0, ly: float = 1.0):
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
        # For lid-driven cavity with characteristic length L=1 and velocity U=1
        self.nu = 1.0 / Re  # kinematic viscosity

class SolverSettings:
    """Class to handle solver settings"""
    def __init__(self, dt: float = 0.001, max_iterations: int = 100000,
                 convergence_criteria: Dict[str, float] = None,
                 scheme: str = 'QUICK'):
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


# Numba-compiled functions

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
def correct_velocity(Var, VarOld, dt, rho, Nx, Ny, dx, dy, residual):
    for i in prange(1, Nx + 1):
        for j in range(1, Ny + 1):
            # U velocity correction
            Var[0, i, j] = Var[0, i, j] - dt / rho * (Var[2, i + 1, j] - Var[2, i - 1, j]) / (2 * dx)
            # V velocity correction
            Var[1, i, j] = Var[1, i, j] - dt / rho * (Var[2, i, j + 1] - Var[2, i, j - 1]) / (2 * dy)
            
            # Calculate residuals
            residual[0] += (Var[0, i, j] - VarOld[0, i, j]) ** 2
            residual[1] += (Var[1, i, j] - VarOld[1, i, j]) ** 2
            residual[2] += (Var[2, i, j] - VarOld[2, i, j]) ** 2


class CFDSolver:
    """Main CFD Solver class"""
    def __init__(self, mesh: MeshParameters, fluid: FluidProperties,
                 solver_settings: SolverSettings, bc: BoundaryConditions):
        self.mesh = mesh
        self.fluid = fluid
        self.settings = solver_settings
        self.bc = bc
        
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
    
    def _apply_bc_wrapper(self, k: int):
        """Wrapper to apply boundary conditions based on settings"""
        bc_types, bc_values = self._get_bc_arrays(k)
        apply_bc_configured(self.Var, k, self.mesh.nx, self.mesh.ny, bc_types, bc_values)
    
    def solve(self, output_base_name: str = "output", verbose: bool = True,
              callback=None, callback_interval: int = 1000):
        """Main solver loop with optional interval callback."""
        count = 0
        converged = False
        start_time = time.time()

        if callback_interval <= 0:
            callback_interval = 1000
        
        if verbose:
            print(f"Starting simulation with Re={self.fluid.Re}, mesh={self.mesh.nx}x{self.mesh.ny}")
            print(f"Time step: {self.settings.dt}, Scheme: {self.settings.scheme}")
            print("\nIteration\tU-RMS\t\tV-RMS\t\tP-RMS")
            print("-" * 60)
        
        while not converged and count < self.settings.max_iterations:
            count += 1
            self._implicit_solve()
            
            if verbose and count % 100 == 0:
                print(f"{count}", end="")
            
            converged, rms_residuals = self._convergence_check(verbose and count % 100 == 0)
            if count % 100 == 0:
                self.residual_history['u'].append(rms_residuals[0])
                self.residual_history['v'].append(rms_residuals[1])
                self.residual_history['p'].append(rms_residuals[2])

            if callback is not None and (count % callback_interval == 0):
                callback(self, count)

        if callback is not None and (count % callback_interval != 0):
            callback(self, count)
        
        end_time = time.time()
        
        if verbose:
            print(f"\n\nSimulation completed in {end_time - start_time:.2f} seconds")
            print(f"Total iterations: {count}")
        
        # Save results
        self._save_results(output_base_name)
        
        return count, end_time - start_time
    
    def _implicit_solve(self):
        """Implicit solver step using SIMPLE algorithm"""
        self.residual.fill(0.0)
        
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
            
            self._apply_bc_wrapper(k)
        
        linear_interpolation(self.Var, self.Ff, self.mesh.nx, self.mesh.ny, 
                           self.mesh.dx, self.mesh.dy)
        
        # Solve pressure equation
        solve_pressure(self.Var, self.Ff, self.mesh.nx, self.mesh.ny, 
                      self.mesh.dx, self.mesh.dy, self.settings.dt, 
                      self.fluid.rho, self.mesh.volp)
        self._apply_bc_wrapper(2)
        
        # Correct velocities
        correct_velocity(self.Var, self.VarOld, self.settings.dt, self.fluid.rho, 
                        self.mesh.nx, self.mesh.ny, self.mesh.dx, self.mesh.dy, 
                        self.residual)
        
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
        
        # Check for NaN or Inf in residuals (indicates solver failure)
        if np.isnan(rms).any() or np.isinf(rms).any():
            print(f"\n❌ ERROR: NaN or Inf detected in residuals!")
            print(f"   U-residual: {rms[0]:.6e}, V-residual: {rms[1]:.6e}, P-residual: {rms[2]:.6e}")
            print(f"   This indicates solver instability or bad initial conditions.")
            print(f"   Check ML predictions and boundary conditions.")
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
            
            grp.attrs["case_name"] = "lid driven cavity"
            grp.attrs["reynolds_number"] = self.fluid.Re
            grp.attrs["nx"] = self.mesh.nx
            grp.attrs["ny"] = self.mesh.ny
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
        u_center = self.Var[0, self.mesh.nx//2, 1:-1]
        v_center = self.Var[1, 1:-1, self.mesh.ny//2]
        y = np.linspace(0, self.mesh.ly, self.mesh.ny)
        x = np.linspace(0, self.mesh.lx, self.mesh.nx)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(u_center, y, 'b-', linewidth=2)
        ax1.set_xlabel('U velocity')
        ax1.set_ylabel('Y')
        ax1.set_title(f'U velocity along vertical centerline (Re={self.fluid.Re})')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(x, v_center, 'r-', linewidth=2)
        ax2.set_xlabel('X')
        ax2.set_ylabel('V velocity')
        ax2.set_title(f'V velocity along horizontal centerline (Re={self.fluid.Re})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
    
    def _plot_contours(self, filename: str):
        """Plot contour plots of all variables"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
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
        
        plt.suptitle(f'Lid-Driven Cavity Flow (Re={self.fluid.Re})', fontsize=16)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
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
        ax.set_title(f'Convergence History (Re={self.fluid.Re})')
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()


# ==============================================================================
# ML Helper Functions (from cfdtemp)
# ==============================================================================

def load_solver_from_hdf5(filepath: str, Re: float, nx: int, ny: int,
                          dt: float, scheme: str,
                          bc: BoundaryConditions,
                          lx: float = 1.0, ly: float = 1.0,
                          convergence_criteria: Dict[str, float] = None) -> CFDSolver:
    """Load a previously saved CFD solver state from an HDF5 file."""
    print(f"\nLoading solver state from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"HDF5 file not found: {filepath}")

    if convergence_criteria is None:
        convergence_criteria = {'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6}

    mesh = MeshParameters(nx=nx, ny=ny, lx=lx, ly=ly)
    fluid = FluidProperties(Re=Re, rho=1.0)
    solver_settings = SolverSettings(
        dt=dt,
        scheme=scheme,
        max_iterations=1,
        convergence_criteria=convergence_criteria
    )
    solver = CFDSolver(mesh, fluid, solver_settings, bc)

    with h5py.File(filepath, 'r') as f:
        group_name = f"Re{Re}_mesh{nx}x{ny}"

        if group_name not in f:
            raise KeyError(f"Group '{group_name}' not found in {filepath}")

        grp = f[group_name]
        u_flat = grp['u'][:]
        v_flat = grp['v'][:]
        p_flat = grp['p'][:]

        u_2d = u_flat.reshape((ny, nx)).T
        v_2d = v_flat.reshape((ny, nx)).T
        p_2d = p_flat.reshape((ny, nx)).T

        solver.Var[0, 1:-1, 1:-1] = u_2d
        solver.Var[1, 1:-1, 1:-1] = v_2d
        solver.Var[2, 1:-1, 1:-1] = p_2d

        solver.VarOld = solver.Var.copy()

    for k in range(solver.nVar):
        solver._apply_bc_wrapper(k)

    linear_interpolation(solver.Var, solver.Ff, solver.mesh.nx, solver.mesh.ny,
                        solver.mesh.dx, solver.mesh.dy)

    print(f"Loaded solver state: Re={Re}, mesh={nx}x{ny}")
    return solver

def standardize_with_stats(arr, mean, std):
    """Standardize array with given mean and std"""
    std = 1e-8 if std == 0 else std
    return (arr - mean) / std


def inverse_standardize(arr, mean, std):
    """Inverse standardization"""
    return arr * std + mean


def normalize_per_sample(arr: np.ndarray):
    """Per-sample Z-score normalization for 3-channel flow fields."""
    arr = arr.astype(np.float32)
    n_samples = arr.shape[0]
    stats = np.zeros((n_samples, 3, 2), dtype=np.float32)
    normalized = np.copy(arr)

    for i in range(n_samples):
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
    """Inverse of per-sample normalization."""
    result = np.copy(arr).astype(np.float32)
    n_samples = arr.shape[0]

    for i in range(n_samples):
        for c in range(3):
            mean = stats[i, c, 0]
            std = stats[i, c, 1]
            result[i, :, :, c] = arr[i, :, :, c] * std + mean

    return result


def make_coord_channels_batch(dim: int, lx_arr, ly_arr) -> np.ndarray:
    """Generate normalized coordinate channels [x_bar, y_bar] for each sample."""
    n_samples = len(lx_arr)
    coords = np.zeros((n_samples, dim, dim, 2), dtype=np.float32)
    for i in range(n_samples):
        lx = lx_arr[i]
        ly = ly_arr[i]
        length_scale = max(lx, ly)
        x = np.linspace(0, lx, dim) / length_scale
        y = np.linspace(0, ly, dim) / length_scale
        xx, yy = np.meshgrid(x, y)
        coords[i, :, :, 0] = xx
        coords[i, :, :, 1] = yy
    return coords


def bicubic_interpolate_batch(x, target_size):
    """
    Bicubic interpolation for batched multi-channel images.
    
    Args:
        x: (N, H, W, C) tensor
        target_size: (target_H, target_W) tuple
    Returns:
        Interpolated tensor of shape (N, target_H, target_W, C)
    """
    return tf.image.resize(x, target_size, method='bicubic')


class InterpolateRefineModel(Model):
    """
    Complete stage model: Interpolate -> Predict Residual -> Add.
    Output = Bicubic(input) + U-Net(Bicubic(input)) * 0.1
    """
    def __init__(self, unet_model, target_size, **kwargs):
        super().__init__(**kwargs)
        self.unet = unet_model
        self.target_size = target_size
    
    def call(self, inputs, training=False):
        # Step 1: Bicubic interpolation to target size
        interpolated = bicubic_interpolate_batch(inputs, self.target_size)
        
        # Step 2: Predict residual correction
        residual = self.unet(interpolated, training=training)
        residual = residual * 0.1  # Scale down residual to stabilize
        
        # Step 3: Add residual to interpolated (residual learning)
        refined = interpolated + residual
        
        return refined


# ==============================================================================
# ML-Accelerated CFD Workflow
# ==============================================================================

def run_coarse_simulation(Re: float, lr_dim: int = 10,
                         dt: float = 0.001, scheme: str = 'QUICK',
                         convergence_criteria: Dict[str, float] = None,
                         max_iterations: int = 100000,
                         output_dir: str = None,
                         bc: Optional[BoundaryConditions] = None) -> tuple:
    """
    Step 1: Run a coarse (10x10) CFD simulation
    
    Args:
        Re: Reynolds number
        lr_dim: Low resolution dimension (default: 10)
        dt: Time step
        scheme: Numerical scheme
        convergence_criteria: Convergence criteria dict
        max_iterations: Maximum iterations
        output_dir: Directory to save outputs. If None, creates timestamped directory.
        bc: BoundaryConditions object. If None, uses default lid-driven cavity BCs.
    
    Returns:
        Tuple of (coarse_fields, iterations, time_elapsed)
    """
    print(f"\n{'='*70}")
    print(f"STEP 1: Running Coarse Simulation (Re={Re}, mesh={lr_dim}x{lr_dim})")
    print(f"{'='*70}")
    
    # Create mesh for coarse simulation
    mesh = MeshParameters(nx=lr_dim, ny=lr_dim, lx=1.0, ly=1.0)
    fluid = FluidProperties(Re=Re, rho=1.0)
    
    if convergence_criteria is None:
        convergence_criteria = {'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6}
    
    solver_settings = SolverSettings(dt=dt, scheme=scheme, 
                                   max_iterations=max_iterations,
                                   convergence_criteria=convergence_criteria)
    
    # Use provided boundary conditions or create default lid-driven cavity BCs
    if bc is None:
        bc = BoundaryConditions()
    
    # Create solver and run
    solver = CFDSolver(mesh, fluid, solver_settings, bc)
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = create_timestamped_output_dir()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir, f"coarse_Re{Re}_{lr_dim}x{lr_dim}_{max_iterations}_coarse_iterations")
    
    print(f"Saving coarse simulation output to: {output_dir}")
    
    iterations, time_elapsed = solver.solve(output_name, verbose=True)
    
    print(f"Coarse simulation completed in {iterations} iterations ({time_elapsed:.2f} seconds)")
    
    # Extract the solution fields (internal cells only, no ghost cells)
    coarse_fields = {
        'u': solver.Var[0, 1:-1, 1:-1].T.copy(),  # Shape: (lr_dim, lr_dim)
        'v': solver.Var[1, 1:-1, 1:-1].T.copy(),  # Shape: (lr_dim, lr_dim)
        'p': solver.Var[2, 1:-1, 1:-1].T.copy(),  # Shape: (lr_dim, lr_dim)
    }
    
    return coarse_fields, iterations, time_elapsed


def ml_super_resolution(coarse_fields: Dict[str, np.ndarray],
                        lr_dim: int, hr_dim: int,
                        stage_dims: List[int],
                        model_name_pattern: str,
                        lx: float = 1.0, ly: float = 1.0) -> Dict[str, np.ndarray]:
    """
    Step 2: Use BFS-compatible cascaded U-Net models to super-resolve coarse simulation.
    
    Args:
        coarse_fields: Dictionary with 'u', 'v', 'p' arrays of shape (lr_dim, lr_dim)
        lr_dim: Low resolution dimension (e.g., 10)
        hr_dim: High resolution dimension (e.g., 400)
        stage_dims: Intermediate stage dimensions (e.g., [20, 40, 80, 200, 400])
        model_name_pattern: Pattern like "unet_stage_{from_dim}to{to_dim}_NAME.h5"
        lx, ly: Physical domain lengths for coordinate channels
    
    Returns:
        Dictionary with 'u', 'v', 'p' fields of shape (hr_dim, hr_dim)
    """
    print(f"\n{'='*70}")
    print(f"STEP 2: ML Super-Resolution (Progressive U-Net {lr_dim}x{lr_dim} -> {hr_dim}x{hr_dim})")
    print(f"  Cascaded stages: {lr_dim} -> {' -> '.join(map(str, stage_dims))}")
    print(f"  Per-sample normalization: ENABLED")
    print(f"  Coordinate channels: ENABLED (lx={lx}, ly={ly})")
    print(f"{'='*70}")

    x_lr_3ch = np.stack([
        coarse_fields['u'],
        coarse_fields['v'],
        coarse_fields['p']
    ], axis=-1).astype(np.float32)
    x_lr_batch = x_lr_3ch[np.newaxis]

    x_lr_norm, sample_stats = normalize_per_sample(x_lr_batch)
    coords = make_coord_channels_batch(lr_dim, [lx], [ly])
    x_current = np.concatenate([x_lr_norm, coords], axis=-1)

    prev_dim = lr_dim
    for stage_idx, target_dim in enumerate(stage_dims):
        stage_name = f"{prev_dim}to{target_dim}"
        model_file = model_name_pattern.format(from_dim=prev_dim, to_dim=target_dim)

        print(f"\n  Stage {stage_idx + 1}/{len(stage_dims)}: {stage_name}")
        print(f"    Loading model: {model_file}")

        if not os.path.exists(model_file):
            raise FileNotFoundError(f"U-Net stage model not found: {model_file}")

        unet_model = tf.keras.models.load_model(model_file, compile=False)

        x_interp = bicubic_interpolate_batch(x_current, (target_dim, target_dim))
        residual = unet_model.predict(x_interp, verbose=0)
        residual = residual * 0.1
        x_flow = x_interp[..., :3] + residual

        if np.isnan(x_flow).any() or np.isinf(x_flow).any():
            x_flow = np.nan_to_num(x_flow, nan=0.0, posinf=0.0, neginf=0.0)

        if stage_idx < len(stage_dims) - 1:
            coords_new = make_coord_channels_batch(target_dim, [lx], [ly])
            x_current = np.concatenate([x_flow, coords_new], axis=-1)
        else:
            x_current = x_flow

        prev_dim = target_dim

    hr_3ch_real = denormalize_per_sample(x_current, sample_stats)
    hr_fields = {
        'u': hr_3ch_real[0, ..., 0],
        'v': hr_3ch_real[0, ..., 1],
        'p': hr_3ch_real[0, ..., 2]
    }

    print("\n  Final output ranges:")
    print(f"    U: [{hr_fields['u'].min():.6f}, {hr_fields['u'].max():.6f}]")
    print(f"    V: [{hr_fields['v'].min():.6f}, {hr_fields['v'].max():.6f}]")
    print(f"    P: [{hr_fields['p'].min():.6f}, {hr_fields['p'].max():.6f}]")
    print("\n  Progressive super-resolution complete")
    return hr_fields


def run_fine_simulation_with_ml_init(Re: float, nx: int, ny: int,
                                     ml_initial_fields: Dict[str, np.ndarray],
                                     dt: float = 0.001, scheme: str = 'QUICK',
                                     convergence_criteria: Dict[str, float] = None,
                                     max_iterations: int = 100000,
                                     output_name: str = "cavity_accelerated",
                                     bc: Optional[BoundaryConditions] = None,
                                     callback=None,
                                     callback_interval: int = 1000) -> tuple:
    """
    Step 3: Run fine-resolution simulation with ML-predicted initialization
    
    Args:
        Re: Reynolds number
        nx, ny: Fine mesh dimensions
        ml_initial_fields: Dictionary with 'u', 'v', 'p' fields of shape (ny, nx)
        dt: Time step
        scheme: Numerical scheme
        convergence_criteria: Convergence criteria
        output_name: Base name for output files (will have "_accelerated" added)
        bc: BoundaryConditions object. If None, uses default lid-driven cavity BCs.
    
    Returns:
        (solver, iterations, time_elapsed)
    """
    print(f"\n{'='*70}")
    print(f"STEP 3: Running Fine Simulation with ML Initialization")
    print(f"        (Re={Re}, mesh={nx}x{ny})")
    print(f"{'='*70}")
    
    # Create mesh and settings for fine simulation
    mesh = MeshParameters(nx=nx, ny=ny, lx=1.0, ly=1.0)
    fluid = FluidProperties(Re=Re, rho=1.0)
    
    if convergence_criteria is None:
        convergence_criteria = {'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6}
    
    solver_settings = SolverSettings(dt=dt, scheme=scheme, 
                                   max_iterations=max_iterations,
                                   convergence_criteria=convergence_criteria)
    
    # Use provided boundary conditions or create default lid-driven cavity BCs
    if bc is None:
        bc = BoundaryConditions()
    
    # Create solver
    solver = CFDSolver(mesh, fluid, solver_settings, bc)
    
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
    iterations, time_elapsed = solver.solve(
        output_name,
        verbose=True,
        callback=callback,
        callback_interval=callback_interval
    )
    
    return solver, iterations, time_elapsed


# ==============================================================================
# Main ML-Accelerated Workflow
# ==============================================================================

def generate_coarse_mesh_solution(
    Re: float,
    lr_dim: int = 10,
    dt: float = 0.001,
    scheme: str = 'QUICK',
    convergence_criteria: Dict[str, float] = None,
    max_iterations_coarse: int = 100000,
    output_dir: str = None,
    bc: Optional[BoundaryConditions] = None
) -> tuple:
    """
    Generate coarse mesh solution
    
    Args:
        Re: Reynolds number
        lr_dim: Low resolution dimension for coarse simulation (default: 10)
        dt: Time step
        scheme: Numerical scheme ('QUICK' or 'UPWIND')
        convergence_criteria: Convergence criteria dict
        max_iterations_coarse: Maximum iterations for coarse mesh simulation
        output_dir: Directory for outputs. If None, creates timestamped directory in outputs/
        bc: BoundaryConditions object. If None, uses default lid-driven cavity BCs.
    
    Returns:
        Tuple of (coarse_fields, iterations_coarse, time_coarse, output_dir)
    """
    
    print(f"\n{'#'*70}")
    print(f"# GENERATING COARSE MESH SOLUTION")
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
        bc=bc
    )
    
    print(f"\n{'#'*70}")
    print(f"# COARSE MESH SOLUTION COMPLETE")
    print(f"{'#'*70}\n")
    
    return coarse_fields, iterations_coarse, time_coarse, output_dir


def run_ml_accelerated_fine_simulation(
    coarse_fields: Dict[str, np.ndarray],
    Re: float,
    nx: int, 
    ny: int,
    lr_dim: int = 10,
    dt: float = 0.001,
    scheme: str = 'QUICK',
    convergence_criteria: Dict[str, float] = None,
    max_iterations_fine: int = 100000,
    output_name: str = None,
    stage_dims: List[int] = None,
    model_name_pattern: str = None,
    bc: Optional[BoundaryConditions] = None,
    lx: float = 1.0,
    ly: float = 1.0,
    normal_solver=None,
    check_interval: int = 1000
) -> tuple:
    """
    Run ML-accelerated fine simulation using progressive cascaded U-Net with coarse mesh solution
    
    Args:
        coarse_fields: Dictionary with 'u', 'v', 'p' fields from coarse simulation
        Re: Reynolds number
        nx, ny: Target fine mesh dimensions
        lr_dim: Low resolution dimension that was used for coarse simulation (default: 10)
        dt: Time step
        scheme: Numerical scheme ('QUICK' or 'UPWIND')
        convergence_criteria: Convergence criteria dict
        max_iterations_fine: Maximum iterations for fine mesh simulation
        output_name: Base name for output files
        stage_dims: Intermediate dimensions for cascaded stages
        model_name_pattern: Pattern for stage model files with from_dim/to_dim keys
        bc: BoundaryConditions object. If None, uses default lid-driven cavity BCs.
        lx, ly: Domain lengths for coordinate channels
        normal_solver: Optional normal solver for SR comparison plot
    
    Returns:
        (solver, iterations, time_elapsed)
    """
    
    print(f"\n{'#'*70}")
    print(f"# ML-ACCELERATED FINE SIMULATION")
    print(f"# Re={Re}, Target Resolution={nx}x{ny}")
    print(f"# Using coarse solution from {lr_dim}x{lr_dim}")
    print(f"{'#'*70}\n")
    
    if stage_dims is None:
        stage_dims = [20, 40, 80, 200, 400]
    if model_name_pattern is None:
        model_name_pattern = "unet_stage_{from_dim}to{to_dim}_progressive_residual_unet.h5"
    if output_name is None:
        output_name = f"cavity_Re{Re}_{nx}x{ny}"

    print("Checking required ML model files...")

    prev_dim = lr_dim
    for target_dim in stage_dims:
        stage_file = model_name_pattern.format(from_dim=prev_dim, to_dim=target_dim)
        if os.path.exists(stage_file):
            print(f"  ✓ Stage {prev_dim}->{target_dim}: {stage_file}")
        else:
            print(f"  ✗ Stage {prev_dim}->{target_dim}: {stage_file} NOT FOUND")
            raise FileNotFoundError(f"Stage model file not found: {stage_file}")
        prev_dim = target_dim

    hr_fields = ml_super_resolution(
        coarse_fields=coarse_fields,
        lr_dim=lr_dim,
        hr_dim=nx,
        stage_dims=stage_dims,
        model_name_pattern=model_name_pattern,
        lx=lx,
        ly=ly
    )

    sr_plot_dir = os.path.join(os.path.dirname(output_name), 'sr_comparison')
    plot_ml_sr_comparison(
        coarse_fields=coarse_fields,
        hr_fields=hr_fields,
        lr_dim=lr_dim,
        hr_dim=nx,
        lx=lx,
        ly=ly,
        save_dir=sr_plot_dir,
        normal_solver=normal_solver
    )
    
    checkpoint_dir = os.path.join(os.path.dirname(output_name), 'checkpoints')
    checkpoint_plot_dir = os.path.join(os.path.dirname(output_name), 'checkpoint_plots')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(checkpoint_plot_dir, exist_ok=True)

    metrics_history = []
    monitor_callback = None
    if normal_solver is not None:
        def monitor_callback(solver, iteration):
            save_checkpoint(iteration, solver, checkpoint_dir, 'ml_accelerated')
            ml_center = extract_centerlines(solver, nx, ny)
            normal_center = extract_centerlines(normal_solver, nx, ny)
            metrics = compute_centerline_metrics(ml_center, normal_center)
            metrics_history.append({'iteration': iteration, **metrics})

            checkpoint_plot = os.path.join(
                checkpoint_plot_dir,
                f"centerline_comparison_iter{iteration}.png"
            )
            plot_centerline_comparison(
                ml_center,
                normal_center,
                Re=Re,
                save_path=checkpoint_plot,
                bc=bc
            )

            print(
                f"[Monitor @ {iteration}] "
                f"U_L2={metrics['u_l2']:.3e}, V_L2={metrics['v_l2']:.3e}, "
                f"U_MAX={metrics['u_max']:.3e}, V_MAX={metrics['v_max']:.3e}"
            )

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
        callback=monitor_callback,
        callback_interval=check_interval
    )
    
    print(f"\n{'#'*70}")
    print(f"# ML-ACCELERATED FINE SIMULATION COMPLETE")
    print(f"# Converged in {iterations} iterations ({time_elapsed:.2f} seconds)")
    print(f"# Output saved with '_accelerated' suffix")
    print(f"{'#'*70}\n")

    if metrics_history:
        metrics_file = os.path.join(os.path.dirname(output_name), 'metrics_history.csv')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            f.write('iteration,u_l2,u_max,u_mean,v_l2,v_max,v_mean\n')
            for row in metrics_history:
                f.write(
                    f"{row['iteration']},{row['u_l2']},{row['u_max']},{row['u_mean']},"
                    f"{row['v_l2']},{row['v_max']},{row['v_mean']}\n"
                )

        evolution_plot = os.path.join(os.path.dirname(output_name), 'error_evolution.png')
        plot_error_evolution(metrics_history, evolution_plot, Re)

        print("Checkpoint comparison artifacts generated:")
        print(f"  Checkpoints: {checkpoint_dir}")
        print(f"  Plots:       {checkpoint_plot_dir}")
        print(f"  Metrics CSV: {metrics_file}")
        print(f"  Evolution:   {evolution_plot}")
    else:
        print("No checkpoint comparison artifacts were generated (normal solver not provided).")
    
    return solver, iterations, time_elapsed


# ==============================================================================
# Normal (Non-accelerated) Simulation
# ==============================================================================

def run_normal_simulation(Re: float, nx: int, ny: int,
                         dt: float = 0.001, scheme: str = 'QUICK',
                         convergence_criteria: Dict[str, float] = None,
                         max_iterations: int = 100000,
                         output_name: str = "cavity_normal",
                         bc: Optional[BoundaryConditions] = None,
                         check_interval: int = 1000) -> tuple:
    """
    Run a normal CFD simulation without ML acceleration
    
    Args:
        Re: Reynolds number
        nx, ny: Mesh dimensions
        dt: Time step
        scheme: Numerical scheme
        convergence_criteria: Convergence criteria
        max_iterations: Maximum number of iterations
        output_name: Base name for output files
        bc: BoundaryConditions object. If None, uses default lid-driven cavity BCs.
    
    Returns:
        (solver, iterations, time_elapsed)
    """
    print(f"\n{'='*70}")
    print(f"RUNNING NORMAL (NON-ACCELERATED) SIMULATION")
    print(f"Re={Re}, mesh={nx}x{ny}")
    print(f"{'='*70}")
    
    # Create mesh and settings
    mesh = MeshParameters(nx=nx, ny=ny, lx=1.0, ly=1.0)
    fluid = FluidProperties(Re=Re, rho=1.0)
    
    if convergence_criteria is None:
        convergence_criteria = {'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6}
    
    solver_settings = SolverSettings(dt=dt, scheme=scheme, 
                                   max_iterations=max_iterations,
                                   convergence_criteria=convergence_criteria)
    
    # Use provided boundary conditions or create default lid-driven cavity BCs
    if bc is None:
        bc = BoundaryConditions()
    
    # Create solver and run
    solver = CFDSolver(mesh, fluid, solver_settings, bc)
    
    # Add "_normal" suffix to output name
    if not output_name.endswith("_normal"):
        output_name = f"{output_name}_normal"
    
    iterations, time_elapsed = solver.solve(
        output_name,
        verbose=True,
        callback_interval=check_interval
    )
    
    print(f"Normal simulation completed in {iterations} iterations ({time_elapsed:.2f} seconds)")
    
    return solver, iterations, time_elapsed


# ==============================================================================
# Centerline Extraction and Plotting
# ==============================================================================

def plot_ml_sr_comparison(coarse_fields: Dict[str, np.ndarray],
                          hr_fields: Dict[str, np.ndarray],
                          lr_dim: int, hr_dim: int,
                          lx: float, ly: float,
                          save_dir: str,
                          normal_solver=None) -> None:
    """Plot LR input, HR predicted, and optional HR reference for u/v/p."""
    os.makedirs(save_dir, exist_ok=True)

    var_info = [
        ('u', 'U Velocity', 'RdBu'),
        ('v', 'V Velocity', 'RdBu'),
        ('p', 'Pressure', 'viridis'),
    ]
    have_ground = normal_solver is not None
    ground = {}
    if have_ground:
        ground = {
            'u': normal_solver.Var[0, 1:-1, 1:-1].T.copy(),
            'v': normal_solver.Var[1, 1:-1, 1:-1].T.copy(),
            'p': normal_solver.Var[2, 1:-1, 1:-1].T.copy(),
        }

    x_lr = np.linspace(0, lx, lr_dim)
    y_lr = np.linspace(0, ly, lr_dim)
    x_hr = np.linspace(0, lx, hr_dim)
    y_hr = np.linspace(0, ly, hr_dim)
    xlr, ylr = np.meshgrid(x_lr, y_lr)
    xhr, yhr = np.meshgrid(x_hr, y_hr)

    n_cols = 3 if have_ground else 2

    for key, label, cmap in var_info:
        fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
        if n_cols == 2:
            axes = [axes[0], axes[1]]

        vmin = min(coarse_fields[key].min(), hr_fields[key].min())
        vmax = max(coarse_fields[key].max(), hr_fields[key].max())
        if have_ground:
            vmin = min(vmin, ground[key].min())
            vmax = max(vmax, ground[key].max())

        c0 = axes[0].contourf(xlr, ylr, coarse_fields[key], levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title(f"LR Input ({lr_dim}x{lr_dim})")
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        axes[0].set_aspect('equal')
        plt.colorbar(c0, ax=axes[0])

        if have_ground:
            c1 = axes[1].contourf(xhr, yhr, ground[key], levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[1].set_title(f"HR Reference ({hr_dim}x{hr_dim})")
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            axes[1].set_aspect('equal')
            plt.colorbar(c1, ax=axes[1])

            c2 = axes[2].contourf(xhr, yhr, hr_fields[key], levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[2].set_title(f"HR Predicted ({hr_dim}x{hr_dim})")
            axes[2].set_xlabel('X')
            axes[2].set_ylabel('Y')
            axes[2].set_aspect('equal')
            plt.colorbar(c2, ax=axes[2])
        else:
            c1 = axes[1].contourf(xhr, yhr, hr_fields[key], levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
            axes[1].set_title(f"HR Predicted ({hr_dim}x{hr_dim})")
            axes[1].set_xlabel('X')
            axes[1].set_ylabel('Y')
            axes[1].set_aspect('equal')
            plt.colorbar(c1, ax=axes[1])

        fig.suptitle(f"{label} Super-Resolution Comparison", fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sr_compare_{key}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved SR comparison plot: {save_path}")

def format_bc_summary(bc: Optional[BoundaryConditions]) -> str:
    """
    Format boundary conditions into a detailed summary string
    
    Args:
        bc: BoundaryConditions object or None
    
    Returns:
        Formatted BC summary string
        Example: "BC: U(L:0.00, R:0.00, T:1.00, B:0.00) V(L:0.00, R:0.00, T:0.00, B:0.00) P(all Neumann)"
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
            'u_vertical': {'y': array, 'values': array},   # U along vertical centerline (x=0.5)
            'v_horizontal': {'x': array, 'values': array}  # V along horizontal centerline (y=0.5)
        }
    """
    # Get mesh coordinates
    x = np.linspace(0, 1.0, nx)
    y = np.linspace(0, 1.0, ny)
    
    # Extract fields (internal cells only, no ghost cells)
    u_field = solver.Var[0, 1:-1, 1:-1].T.copy()  # Shape: (ny, nx)
    v_field = solver.Var[1, 1:-1, 1:-1].T.copy()  # Shape: (ny, nx)
    
    # U velocity along vertical centerline (x = 0.5, varying y)
    centerline_x_idx = nx // 2
    u_vertical = u_field[:, centerline_x_idx]
    
    # V velocity along horizontal centerline (y = 0.5, varying x)
    centerline_y_idx = ny // 2
    v_horizontal = v_field[centerline_y_idx, :]
    
    return {
        'u_vertical': {'y': y, 'values': u_vertical},
        'v_horizontal': {'x': x, 'values': v_horizontal}
    }


def compute_centerline_metrics(ml_centerlines: Dict, normal_centerlines: Dict) -> Dict[str, float]:
    """Compute quantitative differences between ML and reference centerlines."""
    u_ml = ml_centerlines['u_vertical']['values']
    u_ref = normal_centerlines['u_vertical']['values']
    v_ml = ml_centerlines['v_horizontal']['values']
    v_ref = normal_centerlines['v_horizontal']['values']

    u_diff = u_ml - u_ref
    v_diff = v_ml - v_ref

    return {
        'u_l2': float(np.sqrt(np.mean(u_diff ** 2))),
        'u_max': float(np.max(np.abs(u_diff))),
        'u_mean': float(np.mean(np.abs(u_diff))),
        'v_l2': float(np.sqrt(np.mean(v_diff ** 2))),
        'v_max': float(np.max(np.abs(v_diff))),
        'v_mean': float(np.mean(np.abs(v_diff))),
    }


def save_checkpoint(iteration: int, solver, checkpoint_dir: str, prefix: str):
    """Save solver fields at an iteration checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"{prefix}_checkpoint_iter{iteration}.npz")

    u_field = solver.Var[0, 1:-1, 1:-1].T.copy()
    v_field = solver.Var[1, 1:-1, 1:-1].T.copy()
    p_field = solver.Var[2, 1:-1, 1:-1].T.copy()

    np.savez_compressed(
        checkpoint_file,
        iteration=iteration,
        u=u_field,
        v=v_field,
        p=p_field,
        nx=solver.mesh.nx,
        ny=solver.mesh.ny,
        lx=solver.mesh.lx,
        ly=solver.mesh.ly,
        Re=solver.fluid.Re,
    )
    print(f"Saved checkpoint: {checkpoint_file}")


def plot_error_evolution(metrics_history: List[Dict], save_path: str, Re: float):
    """Plot checkpoint-wise centerline error evolution."""
    if not metrics_history:
        return

    iterations = [m['iteration'] for m in metrics_history]
    u_l2 = [m['u_l2'] for m in metrics_history]
    u_max = [m['u_max'] for m in metrics_history]
    v_l2 = [m['v_l2'] for m in metrics_history]
    v_max = [m['v_max'] for m in metrics_history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(iterations, u_l2, 'b-o', linewidth=2, markersize=5)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('L2 Error')
    axes[0, 0].set_title('U Velocity L2 Error')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(iterations, u_max, 'b-s', linewidth=2, markersize=5)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Max Error')
    axes[0, 1].set_title('U Velocity Max Error')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(iterations, v_l2, 'r-o', linewidth=2, markersize=5)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('L2 Error')
    axes[1, 0].set_title('V Velocity L2 Error')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(iterations, v_max, 'r-s', linewidth=2, markersize=5)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('Max Error')
    axes[1, 1].set_title('V Velocity Max Error')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(f'Checkpoint Error Evolution (Re={Re})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved error evolution plot: {save_path}")


def plot_centerline_comparison(ml_centerlines: Dict, normal_centerlines: Dict, 
                               Re: float, save_path: str = None, bc: Optional[BoundaryConditions] = None):
    """
    Plot centerline comparison between ML-accelerated and normal simulations
    
    Args:
        ml_centerlines: Centerline data from ML-accelerated simulation
        normal_centerlines: Centerline data from normal simulation
        Re: Reynolds number
        save_path: Optional path to save the figure
        bc: BoundaryConditions object (optional, for display in plot)
    """
    import matplotlib.pyplot as plt
    
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
    ax1.set_title('U Velocity along Vertical Centerline (x=0.5)', fontsize=11)
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
    ax2.set_title('V Velocity along Horizontal Centerline (y=0.5)', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Add BC summary as subtitle if provided
    if bc is not None:
        bc_summary = format_bc_summary(bc)
        fig.suptitle(f'Centerline Velocity Comparison (Re={Re})\n{bc_summary}', 
                    fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    

    
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


# ==============================================================================
# Example Usage
# ==============================================================================
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    Re = 900
    nx = 400
    ny = 400
    lr_dim = 10
    stage_dims = [20, 40, 80, 200, 400]
    lx = 1.0
    ly = 1.0

    dt = 1e-3
    scheme = 'QUICK'
    convergence_criteria = {'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6}

    max_iterations_coarse = 150000
    max_iterations_fine_ml = 30000
    max_iterations_normal = 100000
    monitoring_interval = 500

    model_suffix = "progressive_residual_unet_(20-40-80-200-400)_trained along with bfs 100,300"

    model_name_pattern = f"unet_stage_{{from_dim}}to{{to_dim}}_{model_suffix}.h5"

    coarse_simulation_mode = 'skip'   # 'run' or 'load'
    normal_simulation_mode = 'skip'  # 'run', 'load', or 'skip'
    previous_coarse_hdf5 = r"C:\Users\NAVANEETH\Downloads\vvvvvv\outputs\13-04-2026-17-21-27\coarse_Re900_10x10_150000_coarse_iterations.h5"
    previous_normal_hdf5 = r"C:\Users\NAVANEETH\Downloads\vvvvvv\outputs\13-04-2026-17-21-27\cavity_Re900_400x400_100000_NORMAL_normal.h5"

    # Default cavity BCs
    bc = BoundaryConditions()
    bc.u_boundaries['left'] = BoundaryCondition('dirichlet', 0.0)
    bc.u_boundaries['right'] = BoundaryCondition('dirichlet', 0.0)
    bc.u_boundaries['top'] = BoundaryCondition('dirichlet', 1.0)
    bc.u_boundaries['bottom'] = BoundaryCondition('dirichlet', 0.0)

    # -------------------------------------------------------------------------
    # Pre-flight checks
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY - PROGRESSIVE U-NET")
    print("=" * 70)
    print(f"Reynolds Number: {Re}")
    print(f"Fine Mesh: {nx}x{ny}")
    print(f"Coarse Mesh: {lr_dim}x{lr_dim}")
    print(f"Progressive Stages: {lr_dim} -> {' -> '.join(map(str, stage_dims))}")
    print(f"Domain: Lx={lx}, Ly={ly}")
    print(f"Per-sample normalization: ENABLED")
    print(f"Coordinate channels: ENABLED (5-channel input)")
    print(f"Coarse Simulation Mode: {coarse_simulation_mode.upper()}")
    print(f"Normal Simulation Mode: {normal_simulation_mode.upper()}")
    print("=" * 70)

    missing = []
    prev_dim = lr_dim
    for target_dim in stage_dims:
        model_file = model_name_pattern.format(from_dim=prev_dim, to_dim=target_dim)
        if os.path.exists(model_file):
            print(f"  OK model {prev_dim}->{target_dim}: {model_file}")
        else:
            print(f"  MISSING model {prev_dim}->{target_dim}: {model_file}")
            missing.append(model_file)
        prev_dim = target_dim

    if coarse_simulation_mode == 'load' and not os.path.exists(previous_coarse_hdf5):
        missing.append(previous_coarse_hdf5)
    if normal_simulation_mode == 'load' and not os.path.exists(previous_normal_hdf5):
        missing.append(previous_normal_hdf5)

    if missing:
        print("\nPre-flight failed. Missing files:")
        for fpath in missing:
            print(f"  - {fpath}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Execute sequence: normal -> coarse -> ML fine
    # -------------------------------------------------------------------------
    output_dir = create_timestamped_output_dir()
    print(f"\nAll outputs will be saved to: {output_dir}")

    solver_normal = None
    iterations_normal = None
    elapsed_time_normal = None

    if normal_simulation_mode == 'run':
        solver_normal, iterations_normal, elapsed_time_normal = run_normal_simulation(
            Re=Re,
            nx=nx,
            ny=ny,
            dt=dt,
            scheme=scheme,
            convergence_criteria=convergence_criteria,
            max_iterations=max_iterations_normal,
            output_name=os.path.join(output_dir, f"cavity_Re{Re}_{nx}x{ny}_{max_iterations_normal}_NORMAL"),
            bc=bc,
            check_interval=monitoring_interval
        )
    elif normal_simulation_mode == 'load':
        solver_normal = load_solver_from_hdf5(
            filepath=previous_normal_hdf5,
            Re=Re,
            nx=nx,
            ny=ny,
            dt=dt,
            scheme=scheme,
            bc=bc,
            lx=lx,
            ly=ly,
            convergence_criteria=convergence_criteria
        )

    if coarse_simulation_mode == 'run':
        coarse_fields, iterations_coarse, elapsed_time_coarse = run_coarse_simulation(
            Re=Re,
            lr_dim=lr_dim,
            dt=dt,
            scheme=scheme,
            convergence_criteria=convergence_criteria,
            max_iterations=max_iterations_coarse,
            output_dir=output_dir,
            bc=bc
        )
    else:
        coarse_solver = load_solver_from_hdf5(
            filepath=previous_coarse_hdf5,
            Re=Re,
            nx=lr_dim,
            ny=lr_dim,
            dt=dt,
            scheme=scheme,
            bc=bc,
            lx=lx,
            ly=ly,
            convergence_criteria=convergence_criteria
        )
        coarse_fields = {
            'u': coarse_solver.Var[0, 1:-1, 1:-1].T.copy(),
            'v': coarse_solver.Var[1, 1:-1, 1:-1].T.copy(),
            'p': coarse_solver.Var[2, 1:-1, 1:-1].T.copy(),
        }
        iterations_coarse = None
        elapsed_time_coarse = None

    solver_ml, iterations_ml, elapsed_time_ml = run_ml_accelerated_fine_simulation(
        coarse_fields=coarse_fields,
        Re=Re,
        nx=nx,
        ny=ny,
        lr_dim=lr_dim,
        dt=dt,
        scheme=scheme,
        convergence_criteria=convergence_criteria,
        max_iterations_fine=max_iterations_fine_ml,
        output_name=os.path.join(output_dir, f"cavity_Re{Re}_{nx}x{ny}_{max_iterations_coarse}_coarse_{max_iterations_fine_ml}_fine_UNET"),
        stage_dims=stage_dims,
        model_name_pattern=model_name_pattern,
        bc=bc,
        lx=lx,
        ly=ly,
        normal_solver=solver_normal,
        check_interval=monitoring_interval
    )

    if solver_normal is not None:
        ml_centerlines = extract_centerlines(solver_ml, nx, ny)
        normal_centerlines = extract_centerlines(solver_normal, nx, ny)
        plot_centerline_comparison(
            ml_centerlines,
            normal_centerlines,
            Re=Re,
            save_path=os.path.join(output_dir, f"centerline_comparison_Re{Re}_{nx}x{ny}.png"),
            bc=bc
        )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------
    total_ml_time = elapsed_time_ml if elapsed_time_coarse is None else (elapsed_time_coarse + elapsed_time_ml)
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - PYCFD ML-ACCELERATED SIMULATION")
    print("=" * 70)
    print(f"Reynolds Number: {Re}")
    print(f"Mesh: {nx}x{ny}")
    print(f"Coarse Iterations: {iterations_coarse}")
    print(f"ML Fine Iterations: {iterations_ml}")
    print(f"ML Fine Time: {elapsed_time_ml:.2f} s")
    print(f"Total ML Time: {total_ml_time:.2f} s")
    if solver_normal is not None and elapsed_time_normal is not None:
        print(f"Normal Iterations: {iterations_normal}")
        print(f"Normal Time: {elapsed_time_normal:.2f} s")
        print(f"Speedup Factor: {elapsed_time_normal / total_ml_time:.2f}x")
    print(f"Outputs: {output_dir}")
    print("=" * 70)

    summary_info = {
        "Configuration": {
            "Reynolds Number": str(Re),
            "Resolution (Fine)": f"{nx}x{ny}",
            "Resolution (Coarse)": f"{lr_dim}x{lr_dim}",
            "Domain Size": f"{lx} x {ly}",
            "Diff. Scheme": scheme,
            "Time Step": str(dt),
            "Coarse Mode": coarse_simulation_mode,
            "Normal Mode": normal_simulation_mode,
        },
        "ML Acceleration Settings": {
            "Model Pattern": model_name_pattern,
            "Stages": f"{lr_dim} -> {' -> '.join(map(str, stage_dims))}",
            "Normalization": "Per-sample Z-score",
            "Input Channels": "5 (u, v, p, x_bar, y_bar)",
        },
        "Results": {
            "Coarse Iterations": str(iterations_coarse),
            "Coarse Time (s)": f"{elapsed_time_coarse:.2f}" if elapsed_time_coarse is not None else "Loaded from HDF5",
            "ML+Fine Iterations": str(iterations_ml),
            "ML+Fine Time (s)": f"{elapsed_time_ml:.2f}",
            "Total ML Time (s)": f"{total_ml_time:.2f}",
            "Output Directory": output_dir,
        }
    }

    if solver_normal is not None and elapsed_time_normal is not None:
        summary_info["Normal Simulation"] = {
            "Iterations": str(iterations_normal),
            "Time (s)": f"{elapsed_time_normal:.2f}",
            "Speedup": f"{elapsed_time_normal / total_ml_time:.2f}x",
        }

    save_run_summary(os.path.join(output_dir, "run_summary.txt"), summary_info)
