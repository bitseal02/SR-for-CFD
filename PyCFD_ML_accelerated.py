

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import h5py
import os
import sys
from typing import Dict, Optional, Tuple
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
    
    def solve(self, output_base_name: str = "output", verbose: bool = True):
        """Main solver loop"""
        count = 0
        converged = False
        start_time = time.time()
        
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

def standardize_with_stats(arr, mean, std):
    """Standardize array with given mean and std"""
    std = 1e-8 if std == 0 else std
    return (arr - mean) / std


def inverse_standardize(arr, mean, std):
    """Inverse standardization"""
    return arr * std + mean


class SuperResolutionAE(Model):
    """
    A minimal version of the SuperResolutionAE class required for loading
    and inference.
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
# ML-Accelerated CFD Workflow
# ==============================================================================

def run_coarse_simulation(Re: float, lr_dim: int = 10, 
                         dt: float = 0.001, scheme: str = 'QUICK',
                         convergence_criteria: Dict[str, float] = None,
                         max_iterations: int = 100000,
                         output_dir: str = None,
                         bc: Optional[BoundaryConditions] = None) -> Dict[str, np.ndarray]:
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
        Dictionary with 'u', 'v', 'p' fields of shape (lr_dim, lr_dim)
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
    
    return coarse_fields


def ml_super_resolution(coarse_fields: Dict[str, np.ndarray], 
                        lr_dim: int, hr_dim: int,
                        stats_file: str, encoder_file: str, decoder_file: str) -> Dict[str, np.ndarray]:
    """
    Step 2: Use ML models to super-resolve coarse simulation to fine resolution
    
    Args:
        coarse_fields: Dictionary with 'u', 'v', 'p' arrays of shape (lr_dim, lr_dim)
        lr_dim: Low resolution dimension (e.g., 10)
        hr_dim: High resolution dimension (e.g., 100)
        stats_file: Path to standardization statistics file
        encoder_file: Path to encoder model file
        decoder_file: Path to decoder model file
    
    Returns:
        Dictionary with 'u', 'v', 'p' fields of shape (hr_dim, hr_dim)
    """
    print(f"\n{'='*70}")
    print(f"STEP 2: ML Super-Resolution ({lr_dim}x{lr_dim} -> {hr_dim}x{hr_dim})")
    print(f"{'='*70}")
    
    # Load component-specific standardization statistics
    print(f"Loading component-specific standardization stats from '{stats_file}'...")
    stats = {}
    try:
        with open(stats_file, "r") as f:
            for line in f:
                # Skip comments and empty lines
                if line.strip().startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) == 2:
                    key, value = parts
                    stats[key] = float(value)
        
        # Load component-specific statistics
        stats_lr = {
            'u': (stats[f'mean{lr_dim}_u'], stats[f'std{lr_dim}_u']),
            'v': (stats[f'mean{lr_dim}_v'], stats[f'std{lr_dim}_v']),
            'p': (stats[f'mean{lr_dim}_p'], stats[f'std{lr_dim}_p'])
        }
        stats_hr = {
            'u': (stats[f'mean{hr_dim}_u'], stats[f'std{hr_dim}_u']),
            'v': (stats[f'mean{hr_dim}_v'], stats[f'std{hr_dim}_v']),
            'p': (stats[f'mean{hr_dim}_p'], stats[f'std{hr_dim}_p'])
        }
        
        print(f"  ✓ Component-specific stats loaded:")
        print(f"    LR - U: mean={stats_lr['u'][0]:.6f}, std={stats_lr['u'][1]:.6f}")
        print(f"    LR - V: mean={stats_lr['v'][0]:.6f}, std={stats_lr['v'][1]:.6f}")
        print(f"    LR - P: mean={stats_lr['p'][0]:.6f}, std={stats_lr['p'][1]:.6f}")
        print(f"    HR - U: mean={stats_hr['u'][0]:.6f}, std={stats_hr['u'][1]:.6f}")
        print(f"    HR - V: mean={stats_hr['v'][0]:.6f}, std={stats_hr['v'][1]:.6f}")
        print(f"    HR - P: mean={stats_hr['p'][0]:.6f}, std={stats_hr['p'][1]:.6f}")
            
    except FileNotFoundError:
        print(f"❌ FATAL: Stats file '{stats_file}' not found.")
        raise
    except KeyError as e:
        print(f"❌ FATAL: Missing component-specific stats in file. Required keys: mean{lr_dim}_u/v/p, std{lr_dim}_u/v/p, mean{hr_dim}_u/v/p, std{hr_dim}_u/v/p")
        print(f"   Missing key: {e}")
        raise
    
    # Load ML models
    print(f"Loading encoder from '{encoder_file}'...")
    print(f"Loading decoder from '{decoder_file}'...")
    try:
        encoder_lr = tf.keras.models.load_model(encoder_file, compile=False)
        decoder_hr = tf.keras.models.load_model(decoder_file, compile=False)
        inference_model = SuperResolutionAE(encoder_lr, decoder_hr)
        print(f"  ✓ Models loaded successfully")
    except (IOError, OSError) as e:
        print(f"❌ FATAL: Error loading models: {e}")
        raise
    
    # Super-resolve each field component using component-specific stats
    hr_fields = {}
    for component in ['u', 'v', 'p']:
        print(f"  - Super-resolving '{component.upper()}' field...")
        
        # Get coarse field data
        x_lr_raw = coarse_fields[component].astype(np.float32)  # Shape: (lr_dim, lr_dim)
        
        # Get component-specific statistics
        mean_lr_comp, std_lr_comp = stats_lr[component]
        mean_hr_comp, std_hr_comp = stats_hr[component]
        
        # Standardize using component-specific stats
        x_lr_norm = standardize_with_stats(x_lr_raw, mean_lr_comp, std_lr_comp)
        
        # Add batch and channel dimensions
        x_lr_norm_batch = np.expand_dims(x_lr_norm, axis=(0, -1))  # Shape: (1, lr_dim, lr_dim, 1)
        
        # Predict
        pred_hr_norm = inference_model.predict(x_lr_norm_batch, verbose=0)[0, ..., 0]  # Shape: (hr_dim, hr_dim)
        
        # Inverse standardize using component-specific stats
        pred_hr_real = inverse_standardize(pred_hr_norm, mean_hr_comp, std_hr_comp)
        
        hr_fields[component] = pred_hr_real
        print(f"    Shape: {x_lr_raw.shape} -> {pred_hr_real.shape}")
        print(f"    Input range: [{x_lr_raw.min():.6f}, {x_lr_raw.max():.6f}]")
        print(f"    Output range: [{pred_hr_real.min():.6f}, {pred_hr_real.max():.6f}]")
        
        # Check for NaN or Inf values
        if np.isnan(pred_hr_real).any() or np.isinf(pred_hr_real).any():
            nan_count = np.isnan(pred_hr_real).sum()
            inf_count = np.isinf(pred_hr_real).sum()
            print(f"    ⚠️  WARNING: Component '{component.upper()}' contains {nan_count} NaN and {inf_count} Inf values!")
            print(f"    ⚠️  ML model may not generalize well to these boundary conditions")
            print(f"    ⚠️  Replacing NaN/Inf with zeros to prevent solver failure...")
            pred_hr_real = np.nan_to_num(pred_hr_real, nan=0.0, posinf=0.0, neginf=0.0)
            hr_fields[component] = pred_hr_real
    
    print(f"  ✓ Super-resolution complete")
    return hr_fields


def run_fine_simulation_with_ml_init(Re: float, nx: int, ny: int,
                                     ml_initial_fields: Dict[str, np.ndarray],
                                     dt: float = 0.001, scheme: str = 'QUICK',
                                     convergence_criteria: Dict[str, float] = None,
                                     max_iterations: int = 100000,
                                     output_name: str = "cavity_accelerated",
                                     bc: Optional[BoundaryConditions] = None) -> tuple:
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
    iterations, time_elapsed = solver.solve(output_name, verbose=True)
    
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
        Tuple of (coarse_fields, output_dir) where:
            - coarse_fields: Dictionary with 'u', 'v', 'p' fields of shape (lr_dim, lr_dim)
            - output_dir: The directory where outputs were saved
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
    coarse_fields = run_coarse_simulation(
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
    
    return coarse_fields, output_dir


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
    stats_file: str = None,
    encoder_file: str = None,
    decoder_file: str = None,
    bc: Optional[BoundaryConditions] = None
) -> tuple:
    """
    Run ML-accelerated fine simulation using coarse mesh solution
    
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
        stats_file: Path to standardization stats file
        encoder_file: Path to encoder model
        decoder_file: Path to decoder model
        bc: BoundaryConditions object. If None, uses default lid-driven cavity BCs.
    
    Returns:
        (solver, iterations, time_elapsed)
    """
    
    print(f"\n{'#'*70}")
    print(f"# ML-ACCELERATED FINE SIMULATION")
    print(f"# Re={Re}, Target Resolution={nx}x{ny}")
    print(f"# Using coarse solution from {lr_dim}x{lr_dim}")
    print(f"{'#'*70}\n")
    
    # Set default file paths if not provided
    if stats_file is None:
        stats_file = f"standardization_stats_{lr_dim}to{nx}.txt"
    if encoder_file is None:
        encoder_file = f"vanilla_encoder{lr_dim}_to_{nx}.h5"
    if decoder_file is None:
        decoder_file = f"vanilla_decoder{nx}_from_{lr_dim}.h5"
    if output_name is None:
        output_name = f"cavity_Re{Re}_{nx}x{ny}"
    
    # Verify files exist
    print("Checking required ML model files...")
    for fname, desc in [(stats_file, "Stats file"), 
                        (encoder_file, "Encoder model"), 
                        (decoder_file, "Decoder model")]:
        if os.path.exists(fname):
            print(f"  ✓ {desc}: {fname}")
        else:
            print(f"  ✗ {desc}: {fname} NOT FOUND")
            raise FileNotFoundError(f"{desc} not found: {fname}")
    
    # STEP 1: ML super-resolution
    hr_fields = ml_super_resolution(
        coarse_fields=coarse_fields,
        lr_dim=lr_dim,
        hr_dim=nx,  # Assuming nx == ny
        stats_file=stats_file,
        encoder_file=encoder_file,
        decoder_file=decoder_file
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
        bc=bc
    )
    
    print(f"\n{'#'*70}")
    print(f"# ML-ACCELERATED FINE SIMULATION COMPLETE")
    print(f"# Converged in {iterations} iterations ({time_elapsed:.2f} seconds)")
    print(f"# Output saved with '_accelerated' suffix")
    print(f"{'#'*70}\n")
    
    return solver, iterations, time_elapsed


# ==============================================================================
# Normal (Non-accelerated) Simulation
# ==============================================================================

def run_normal_simulation(Re: float, nx: int, ny: int,
                         dt: float = 0.001, scheme: str = 'QUICK',
                         convergence_criteria: Dict[str, float] = None,
                         max_iterations: int = 100000,
                         output_name: str = "cavity_normal",
                         bc: Optional[BoundaryConditions] = None) -> tuple:
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
    
    iterations, time_elapsed = solver.solve(output_name, verbose=True)
    
    print(f"Normal simulation completed in {iterations} iterations ({time_elapsed:.2f} seconds)")
    
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
    """
    Example: Run ML-accelerated simulation for Re=1000, 100x100 mesh
    
    Required files (ensure these are in the same directory):
    - PyCFD (7).py
    - standardization_stats_10to100.txt
    - vanilla_encoder10_to_100.h5
    - vanilla_decoder100_from_10.h5
    """
    
    # Configuration
    Re = 1000
    nx = 400
    ny = 400
    lr_dim = 10  # Coarse mesh dimension
    
    # Separate max_iterations for coarse mesh, fine mesh (ML-accelerated), and normal simulations
    max_iterations_coarse = 100000       # Max iterations for coarse mesh simulation (10x10)
    max_iterations_fine_ml = 200     # Max iterations for fine mesh with ML initialization
    max_iterations_normal = 100000         # Max iterations for normal simulation
    other_details="swish_trained_upto_700_multiBC"  # Suffix for model files
    # =========================================================================
    # OPTIONAL: Define custom boundary conditions
    # If not provided, default lid-driven cavity BCs will be used
    # =========================================================================
    
    # Example 1: Default lid-driven cavity (uncomment to use)
    # bc = None  # Will use default BoundaryConditions()
    
    # Example 2: Custom lid-driven cavity with different lid velocity
    bc = BoundaryConditions()
    bc.u_boundaries = {
        'left': BoundaryCondition('dirichlet', 0.0),
        'right': BoundaryCondition('dirichlet', 0.0),
        'top': BoundaryCondition('dirichlet', 1.0),    # Moving lid with velocity 1.0
        'bottom': BoundaryCondition('dirichlet', 1.0)
    }
    bc.v_boundaries = {
        'left': BoundaryCondition('dirichlet', 0.0),
        'right': BoundaryCondition('dirichlet', 0.0),
        'top': BoundaryCondition('dirichlet', 0.0),
        'bottom': BoundaryCondition('dirichlet', 0.0)
    }
    bc.p_boundaries = {
        'left': BoundaryCondition('neumann', 0.0),
        'right': BoundaryCondition('neumann', 0.0),
        'top': BoundaryCondition('neumann', 0.0),
        'bottom': BoundaryCondition('neumann', 0.0)
    }
    
 
    # PART 1: Generate coarse mesh solution
    print("\n" + "#"*70)
    print("# PART 1A: GENERATE COARSE MESH SOLUTION")
    print("#"*70)
    
    # Create a single timestamped output directory for this run
    output_dir = create_timestamped_output_dir()
    print(f"All outputs will be saved to: {output_dir}")
    
    coarse_fields, output_dir = generate_coarse_mesh_solution(
        Re=Re,
        lr_dim=lr_dim,
        dt=0.001,
        scheme='QUICK',
        convergence_criteria={'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6},
        max_iterations_coarse=max_iterations_coarse,
        output_dir=output_dir,  # Use the timestamped directory
        bc=bc  # Pass boundary conditions
    )
    
    # PART 1B: Run ML-accelerated fine simulation
    print("\n" + "#"*70)
    print("# PART 1B: ML-ACCELERATED FINE SIMULATION")
    print("#"*70)
    solver_ml, iterations_ml, elapsed_time_ml = run_ml_accelerated_fine_simulation(
        coarse_fields=coarse_fields,
        Re=Re,
        nx=nx,
        ny=ny,
        lr_dim=lr_dim,
        dt=0.001,
        scheme='QUICK',
        convergence_criteria={'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6},
        max_iterations_fine=max_iterations_fine_ml,
        output_name=os.path.join(output_dir, f"cavity_Re{Re}_{nx}x{ny}_{max_iterations_coarse}_coarse_{max_iterations_fine_ml}_fine_ML"),  # Use same directory
        stats_file=f"standardization_stats_{lr_dim}to{nx}_{other_details}.txt",
        encoder_file=f"vanilla_encoder{lr_dim}_to_{nx}_{other_details}.h5",
        decoder_file=f"vanilla_decoder{nx}_from_{lr_dim}_{other_details}.h5",
        bc=bc  # Pass boundary conditions
    )
    
    # Run normal simulation
    print("\n" + "#"*70)
    print("# PART 2: NORMAL SIMULATION")
    print("#"*70)
    solver_normal, iterations_normal, elapsed_time_normal = run_normal_simulation(
        Re=Re,
        nx=nx,
        ny=ny,
        dt=0.001,
        scheme='QUICK',
        convergence_criteria={'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6},
        max_iterations=max_iterations_normal,
        output_name=os.path.join(output_dir, f"cavity_Re{Re}_{nx}x{ny}_{max_iterations_normal}_NORMAL"),  # Use same directory
        bc=bc  # Pass boundary conditions
    )
    
    # Extract centerlines
    print("\n" + "#"*70)
    print("# PART 3: EXTRACTING CENTERLINES")
    print("#"*70)
    print("Extracting centerline data from ML-accelerated simulation...")
    ml_centerlines = extract_centerlines(solver_ml, nx, ny)
    print("Extracting centerline data from normal simulation...")
    normal_centerlines = extract_centerlines(solver_normal, nx, ny)
    
    # Plot comparison
    print("\n" + "#"*70)
    print("# PART 4: PLOTTING COMPARISON")
    print("#"*70)
    plot_centerline_comparison(
        ml_centerlines, 
        normal_centerlines, 
        Re=Re,
        save_path=os.path.join(output_dir, f"centerline_comparison_Re{Re}_{nx}x{ny}_coarse{max_iterations_coarse}_ML{max_iterations_fine_ml}_NORMAL{max_iterations_normal}.png"),
        bc=bc
    )
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Reynolds Number: {Re}")
    print(f"Mesh: {nx}x{ny}")
    print(f"\nML-Accelerated Simulation:")
    print(f"  Coarse mesh iterations (10x10): {max_iterations_coarse}")
    print(f"  Fine mesh iterations ({nx}x{ny}): {iterations_ml}")
    print(f"  Total time: {elapsed_time_ml:.2f} seconds")
    print(f"\nNormal Simulation:")
    print(f"  Iterations: {iterations_normal}")
    print(f"  Time: {elapsed_time_normal:.2f} seconds")
    print(f"\nSpeedup Factor: {elapsed_time_normal/elapsed_time_ml:.2f}x")
    print(f"Iteration Reduction (fine mesh): {iterations_normal - iterations_ml} iterations saved")
    print("="*70)
