"""Core CFD solver classes and numerical kernels for BFS workflows."""

import time
from dataclasses import dataclass
from typing import Dict, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

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


