import numpy as np
from numba import njit, prange
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import os

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
            'top': BoundaryCondition('dirichlet', 0.0),  # Moving lid
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
                 scheme: str = 'QUICK',
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

        # Under-relaxation factors
        if relaxation_factors is None:
            # Conservative defaults for SIMPLE stability
            self.relaxation_factors = {
                'u': 0.5,
                'v': 0.5,
                'p': 0.2
            }
        else:
            self.relaxation_factors = relaxation_factors

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
    
    def _apply_bfs_inlet(self, k: int):
        """Apply BFS inlet/wall mixture on the left boundary.
        - For y < step_height: enforce wall (Dirichlet 0) via ghost cell reflection
        - For y >= step_height: apply parabolic U inlet, V = 0
        Only active when self.case_type == 'BFS'.
        """
        if getattr(self, 'case_type', None) != 'BFS':
            return
        if k not in (0, 1):
            return
        ny = self.mesh.ny
        dy = self.mesh.dy
        step_h = getattr(self, 'step_height', 0.94)
        h = getattr(self, 'h', 1.0)
        Ub = getattr(self, 'Ub', 1.0)

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
    
    def solve(self, output_base_name: str = "output", verbose: bool = True, 
              log_convergence: bool = True):
        """Main solver loop"""
        count = 0
        converged = False
        start_time = time.time()
        
        # Initialize convergence log file
        convergence_log_file = None
        if log_convergence:
            convergence_log_filename = f"{output_base_name}_convergence.log"
            convergence_log_file = open(convergence_log_filename, 'w')
            convergence_log_file.write("# Convergence History\n")
            convergence_log_file.write(f"# Reynolds number: {self.fluid.Re}\n")
            convergence_log_file.write(f"# Mesh: {self.mesh.nx}x{self.mesh.ny}\n")
            convergence_log_file.write(f"# Time step: {self.settings.dt}\n")
            convergence_log_file.write(f"# Scheme: {self.settings.scheme}\n")
            convergence_log_file.write("# Iteration\tU_RMS\t\tV_RMS\t\tP_RMS\t\tTime(s)\n")
            convergence_log_file.flush()
        
        if verbose:
            print(f"Starting simulation with Re={self.fluid.Re}, mesh={self.mesh.nx}x{self.mesh.ny}")
            print(f"Time step: {self.settings.dt}, Scheme: {self.settings.scheme}")
            if log_convergence:
                print(f"Convergence history will be saved to: {convergence_log_filename}")
            print("\nIteration\tU-RMS\t\tV-RMS\t\tP-RMS")
            print("-" * 60)
        
        while not converged and count < self.settings.max_iterations:
            count += 1
            self._implicit_solve()
            
            # Check convergence and get residuals
            converged, residuals = self._convergence_check(verbose and count % 100 == 0)
            
            # Write to convergence log file
            if log_convergence and convergence_log_file:
                elapsed_time = time.time() - start_time
                convergence_log_file.write(f"{count}\t{residuals[0]:.6e}\t{residuals[1]:.6e}\t{residuals[2]:.6e}\t{elapsed_time:.3f}\n")
                convergence_log_file.flush()  # Ensure data is written immediately
            
            if verbose and count % 100 == 0:
                print(f"{count}", end="")
        
        # Close convergence log file
        if convergence_log_file:
            convergence_log_file.close()
        
        end_time = time.time()
        
        if verbose:
            print(f"\n\nSimulation completed in {end_time - start_time:.2f} seconds")
            print(f"Total iterations: {count}")
            if log_convergence:
                print(f"Convergence history saved to: {convergence_log_filename}")
        
        # Save results
        self._save_results(output_base_name)
        
        return count, end_time - start_time
    
    def _implicit_solve(self):
        """Implicit solver step using SIMPLE algorithm"""
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
        
        # Correct velocities and compute residuals with a safe parallel reduction
        res_u, res_v, res_p = correct_velocity(self.Var, self.VarOld, self.settings.dt, self.fluid.rho, 
                                               self.mesh.nx, self.mesh.ny, self.mesh.dx, self.mesh.dy)
        self.residual[0] = res_u
        self.residual[1] = res_v
        self.residual[2] = res_p
        
        self._apply_bc_wrapper(0)
        self._apply_bc_wrapper(1)
        
        update_flux(self.Var, self.Ff, self.settings.dt, self.fluid.rho, 
                   self.mesh.nx, self.mesh.ny, self.mesh.dx, self.mesh.dy)
    
    def _convergence_check(self, print_residuals: bool = False) -> bool:
        """Check convergence based on residuals"""
        rms = np.zeros(self.nVar)
        for k in range(self.nVar):
            rms[k] = np.sqrt(self.residual[k] / (self.mesh.nx * self.mesh.ny))
            rms[k] = rms[k] / self.settings.dt
            if print_residuals:
                print(f"\t{rms[k]:.6e}", end="")
        
        if print_residuals:
            print()
        
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
        # Save full field data
        self._save_full_field(f"{output_base_name}_full.dat")
        
        # Save centerline data
        self._save_centerline_data(f"{output_base_name}_centerline.dat")
        
        # Generate plots
        self._generate_plots(output_base_name)
    
    def _save_full_field(self, filename: str):
        """Save full field data"""
        with open(filename, 'w') as f:
            f.write(f"# Reynolds number: {self.fluid.Re}\n")
            f.write(f"# Mesh: {self.mesh.nx}x{self.mesh.ny}\n")
            f.write(f"# Time step: {self.settings.dt}\n")
            
            for k in range(self.nVar):
                var_names = ['U', 'V', 'P']
                f.write(f"\n# ########## {var_names[k]} velocity ############ \n")
                for i in range(self.mesh.nx + 2):
                    for j in range(self.mesh.ny + 2):
                        f.write(f"{self.Var[k, i, j]:.6f} \t")
                    f.write("\n")
    
    def _save_centerline_data(self, filename: str):
        """Save centerline velocity profiles"""
        # Extract centerline data
        u_vertical = self.Var[0, self.mesh.nx//2, 1:-1]  # U along vertical centerline
        v_horizontal = self.Var[1, 1:-1, self.mesh.ny//2]  # V along horizontal centerline
        
        y = np.linspace(0, self.mesh.ly, self.mesh.ny)
        x = np.linspace(0, self.mesh.lx, self.mesh.nx)
        
        with open(filename, 'w') as f:
            f.write(f"# Reynolds number: {self.fluid.Re}\n")
            f.write(f"# Mesh: {self.mesh.nx}x{self.mesh.ny}\n")
            f.write("# Centerline data\n")
            f.write("# y\tu(x=0.5)\tx\tv(y=0.5)\n")
            
            max_len = max(len(y), len(x))
            for i in range(max_len):
                if i < len(y):
                    f.write(f"{y[i]:.6f}\t{u_vertical[i]:.6f}\t")
                else:
                    f.write("\t\t")
                
                if i < len(x):
                    f.write(f"{x[i]:.6f}\t{v_horizontal[i]:.6f}")
                
                f.write("\n")
    
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

# Numba-compiled functions (same as before but with slight modifications)

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

            # Accumulate residuals locally (Numba will parallel-reduce these scalars)
            du = Var[0, i, j] - VarOld[0, i, j]
            dv = Var[1, i, j] - VarOld[1, i, j]
            dp = Var[2, i, j] - VarOld[2, i, j]
            res_u += du * du
            res_v += dv * dv
            res_p += dp * dp
    return res_u, res_v, res_p


# Example usage functions
def create_lid_driven_cavity(Re: float = 100, nx: int = 100, ny: int = 100, 
                           dt: float = 0.001, output_name: str = "cavity_Re100",
                           scheme: str = 'QUICK', 
                           convergence_criteria: Dict[str, float] = None):
    """
    Create and solve a lid-driven cavity problem
    
    Parameters:
    -----------
    Re : float
        Reynolds number
    nx, ny : int
        Number of grid cells in x and y directions
    dt : float
        Time step
    output_name : str
        Base name for output files
    scheme : str
        Numerical scheme ('QUICK' or 'UPWIND')
    convergence_criteria : dict
        Convergence criteria for each variable
    """
    # Create mesh
    mesh = MeshParameters(nx=nx, ny=ny, lx=1.0, ly=1.0)
    
    # Create fluid properties
    fluid = FluidProperties(Re=Re, rho=1.0)
    
    # Set default convergence criteria if not provided
    if convergence_criteria is None:
        convergence_criteria = {'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6}
    
    # Create solver settings
    solver_settings = SolverSettings(dt=dt, scheme=scheme, 
                                   convergence_criteria=convergence_criteria)

    # Create boundary conditions (default is lid-driven cavity)
    bc = BoundaryConditions()
    
    # Create and run solver
    solver = CFDSolver(mesh, fluid, solver_settings, bc)
    iterations, time_elapsed = solver.solve(output_name)
    
    return solver, iterations, time_elapsed


def create_custom_case(mesh_params: Dict, fluid_params: Dict, 
                      solver_params: Dict, bc_params: Dict, 
                      output_name: str = "custom_case"):
    """
    Create a custom CFD case with user-defined parameters
    
    Parameters:
    -----------
    mesh_params : dict
        Dictionary with keys: 'nx', 'ny', 'lx', 'ly'
    fluid_params : dict
        Dictionary with keys: 'Re', 'rho' (optional)
    solver_params : dict
        Dictionary with keys: 'dt', 'scheme' (optional), 'convergence_criteria' (optional)
    bc_params : dict
        Dictionary with boundary condition specifications
    output_name : str
        Base name for output files
    """
    # Create mesh
    mesh = MeshParameters(**mesh_params)
    
    # Create fluid properties
    fluid = FluidProperties(**fluid_params)
    
    # Create solver settings
    solver_settings = SolverSettings(**solver_params)
    
    # Create boundary conditions
    bc = BoundaryConditions()
    
    # Update boundary conditions based on user input
    if 'u_boundaries' in bc_params:
        for wall, condition in bc_params['u_boundaries'].items():
            bc.u_boundaries[wall] = BoundaryCondition(**condition)
    
    if 'v_boundaries' in bc_params:
        for wall, condition in bc_params['v_boundaries'].items():
            bc.v_boundaries[wall] = BoundaryCondition(**condition)
    
    if 'p_boundaries' in bc_params:
        for wall, condition in bc_params['p_boundaries'].items():
            bc.p_boundaries[wall] = BoundaryCondition(**condition)
    
    # Create and run solver
    solver = CFDSolver(mesh, fluid, solver_settings, bc)
    iterations, time_elapsed = solver.solve(output_name)
    
    return solver, iterations, time_elapsed


def create_bfs_case(nx: int = 400, ny: int = 194, dt: float = 2e-3,
                    scheme: str = 'UPWIND', output_name: str = 'bfs_Re400',
                    relaxation_factors: Dict[str, float] | None = None):
    """
    Create and solve a Backward-Facing Step (BFS) baseline case (no custom inlet yet).

    Geometry (non-dimensional with h=1):
      - Step height = 0.94 h
      - Channel length Lx = 20 h
      - Channel height Ly = 1.94 h
    Flow:
      - Re = 100 based on Ub=1 and h=1
    BCs (base):
      - Left: placeholder u=v=0 (custom inlet will be added later)
      - Right: pressure outlet (p=0), u,v Neumann 0
      - Top/Bottom: no-slip walls (u=v=0)
    """
    # Mesh and physics
    mesh = MeshParameters(nx=nx, ny=ny, lx=10.0, ly=3.0)
    fluid = FluidProperties(Re=400, rho=1.0)

    # Solver settings
    convergence_criteria = {'u': 1e-6, 'v': 1e-6, 'p': 1e-6, 'continuity': 1e-6}
    solver_settings = SolverSettings(dt=dt, scheme=scheme, convergence_criteria=convergence_criteria,
                                     relaxation_factors=relaxation_factors)

    # Base boundary conditions
    bc = BoundaryConditions()
    # Velocities
    bc.u_boundaries['left'] = BoundaryCondition('dirichlet', 0.0)   # placeholder; will override later
    bc.u_boundaries['right'] = BoundaryCondition('neumann', 0.0)
    bc.u_boundaries['top'] = BoundaryCondition('dirichlet', 0.0)
    bc.u_boundaries['bottom'] = BoundaryCondition('dirichlet', 0.0)

    bc.v_boundaries['left'] = BoundaryCondition('dirichlet', 0.0)
    bc.v_boundaries['right'] = BoundaryCondition('neumann', 0.0)
    bc.v_boundaries['top'] = BoundaryCondition('dirichlet', 0.0)
    bc.v_boundaries['bottom'] = BoundaryCondition('dirichlet', 0.0)

    # Pressure: outlet Dirichlet at right, others Neumann 0
    bc.p_boundaries['left'] = BoundaryCondition('neumann', 0.0)
    bc.p_boundaries['right'] = BoundaryCondition('dirichlet', 0.0)
    bc.p_boundaries['top'] = BoundaryCondition('neumann', 0.0)
    bc.p_boundaries['bottom'] = BoundaryCondition('neumann', 0.0)

    # Create and run solver
    solver = CFDSolver(mesh, fluid, solver_settings, bc)
    # Store case metadata for later steps
    solver.case_type = 'BFS'
    solver.h = 2.0
    solver.step_height = 1
    solver.Ub = 1.0

    iterations, time_elapsed = solver.solve(output_name)
    return solver, iterations, time_elapsed


# Main execution
if __name__ == "__main__":
    # Run only Backward-Facing Step (Re=100)
    print("Backward-Facing Step (Re=400)")
    print("=" * 60)

    solver, iterations, elapsed_time = create_bfs_case(
        nx=10, ny=10, dt=2e-3, scheme='UPWIND', output_name='bfs_Re400',
        relaxation_factors={'u': 0.5, 'v': 0.5, 'p': 0.2}
    )
    print(f"Converged in {iterations} iterations ({elapsed_time:.2f} seconds)")