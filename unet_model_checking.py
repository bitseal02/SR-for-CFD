"""
Progressive U-Net Model Checking and Visualization Script
==========================================================
Tests the trained progressive U-Net models on BFS Re=100 (rectangular geometry).

Workflow:
1. Load trained progressive U-Net models (5 stages: 10→20→40→80→200→400)
2. Load BFS Re=100 ground truth from H5 file (rectangular 20×1.94 domain)
3. Plot ground truth contours BEFORE aspect ratio correction
4. Apply aspect ratio correction (rectangular → square coordinate system)
5. Plot ground truth contours AFTER aspect ratio correction
6. Run progressive prediction through all 5 stages
7. Plot predicted contours in square coordinates
8. Reverse aspect ratio correction (square → rectangular)
9. Plot final prediction in original rectangular geometry
10. Centerline comparison between ground truth and prediction for u and v

Author: Automated ML-CFD Pipeline
Date: February 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py
import tensorflow as tf
from scipy import interpolate
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
# Model configuration
LR_DIM = 10
STAGE_DIMS = [20, 40, 80, 200, 400]
HR_DIM = 400
OTHER_DETAILS = "progressive_residual_unet_(20-40-80-200-400)_trained_with_LDCs-and-one-bfs-with_aspect_ratio_correction"

# Test data configuration
TEST_FILE = "simulation_result_bfs.h5"  # Update path if needed
TEST_RE = 100  # BFS Reynolds number
BC_TYPE = "BFS(step_height=0.94, h=1.0, Ub=1.0)"

# BFS geometry parameters (rectangular domain)
BFS_LX = 20.0  # Domain length
BFS_LY = 1.94  # Domain height (step_height=0.94 + h=1.0)

# Normalization statistics file
NORM_STATS_FILE = f"norm_stats_10to400_{OTHER_DETAILS}.txt"

# Output configuration
SAVE_PLOTS = True
SHOW_PLOTS = False
OUTPUT_DIR = "model_checking_results"

# Centerline extraction locations (for BFS)
CENTERLINE_Y = 0.5  # Horizontal centerline at mid-height
CENTERLINE_X = 5.0  # Vertical centerline at x=5 (behind step in recirculation zone)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_normalization_stats(filepath):
    """Load normalization statistics from text file."""
    norm_stats = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) == 3:
                    # Format: channel mean std
                    channel, mean, std = parts
                    norm_stats[channel] = (float(mean), float(std))
    
    print("📊 Loaded normalization statistics:")
    for ch in ['u', 'v', 'p']:
        if ch in norm_stats:
            mean, std = norm_stats[ch]
            print(f"   {ch}: mean={mean:.6f}, std={std:.6f}")
        else:
            print(f"   ⚠️  {ch}: NOT FOUND in stats file!")
    
    return norm_stats


def normalize_with_stats(data, norm_stats):
    """Normalize 3-channel data with provided statistics."""
    normalized = data.copy()
    for ch_idx, ch_name in enumerate(['u', 'v', 'p']):
        mean, std = norm_stats[ch_name]
        std = max(std, 1e-8)  # Avoid division by zero
        normalized[..., ch_idx] = (data[..., ch_idx] - mean) / std
    return normalized


def denormalize_with_stats(data, norm_stats):
    """Denormalize 3-channel data with provided statistics."""
    denormalized = data.copy()
    for ch_idx, ch_name in enumerate(['u', 'v', 'p']):
        mean, std = norm_stats[ch_name]
        denormalized[..., ch_idx] = data[..., ch_idx] * std + mean
    return denormalized


def reshape_rectangular_to_square(fields_dict, nx, ny, lx, ly):
    """
    Resample rectangular grid data to square coordinate system for ML model.
    Uses bicubic interpolation to map physical domain to canonical square.
    
    Args:
        fields_dict: Dictionary with 'u', 'v', 'p' arrays of shape (nx, ny)
        nx, ny: Grid dimensions
        lx, ly: Physical domain size
    
    Returns:
        Dictionary with resampled fields in square coordinate (nx, nx)
    """
    # Square coordinate system (use min dimension to preserve aspect ratio)
    L_square = min(lx, ly)
    print(f"  🔄 Aspect ratio correction: ({nx}×{ny}) domain [{lx}×{ly}] → square [{L_square}×{L_square}]")
    
    # Create physical coordinate systems
    x_rect = np.linspace(0, lx, nx)
    y_rect = np.linspace(0, ly, ny)
    
    x_square = np.linspace(0, L_square, nx)
    y_square = np.linspace(0, L_square, nx)
    
    # Resample each field from rectangular to square
    fields_square = {}
    for component in ['u', 'v', 'p']:
        # Use RectBivariateSpline for smooth interpolation
        # Fields are (nx, ny) but RectBivariateSpline expects (ny, nx), so transpose
        f_interp = interpolate.RectBivariateSpline(y_rect, x_rect, fields_dict[component].T)
        fields_square[component] = f_interp(y_square, x_square).T  # Transpose back to (nx, nx)
    
    return fields_square


def reshape_square_to_rectangular(fields_dict, nx, ny, lx, ly):
    """
    Reverse operation: resample from square coordinate system back to rectangular.
    
    Args:
        fields_dict: Dictionary with 'u', 'v', 'p' arrays in square coords (nx, nx)
        nx, ny: Target rectangular grid dimensions
        lx, ly: Physical domain size
    
    Returns:
        Dictionary with resampled fields in rectangular coordinate (nx, ny)
    """
    # Square coordinate system (use min dimension to preserve aspect ratio)
    L_square = min(lx, ly)
    print(f"  🔄 Reverse aspect ratio correction: square [{L_square}×{L_square}] → ({nx}×{ny}) domain [{lx}×{ly}]")
    
    x_square = np.linspace(0, L_square, nx)
    y_square = np.linspace(0, L_square, nx)
    
    # Rectangular coordinate systems
    x_rect = np.linspace(0, lx, nx)
    y_rect = np.linspace(0, ly, ny)
    
    # Resample each field from square to rectangular
    fields_rect = {}
    for component in ['u', 'v', 'p']:
        # Use RectBivariateSpline for smooth interpolation
        # Fields are (nx, nx) but RectBivariateSpline expects (ny, nx), so transpose
        f_interp = interpolate.RectBivariateSpline(y_square, x_square, fields_dict[component].T)
        fields_rect[component] = f_interp(y_rect, x_rect).T  # Transpose back to (nx, ny)
    
    return fields_rect


def load_bfs_data(filepath, Re, lr_dim, hr_dim):
    """
    Load BFS data from H5 file WITHOUT aspect ratio correction.
    Returns raw data with actual grid dimensions from H5 attributes.
    
    Following the loading procedure from bfs_ml_accelerated_unet.py:
    - Data is stored as flattened arrays
    - Reshape as: u_flat.reshape((ny, nx)).T to get shape (nx, ny)
    """
    print(f"\n📂 Loading BFS Re={Re} from: {filepath}")
    
    with h5py.File(filepath, 'r') as f:
        g_lr = f"Re{Re}_mesh{lr_dim}x{lr_dim}"
        g_hr = f"Re{Re}_mesh{hr_dim}x{hr_dim}"
        
        if g_lr not in f or g_hr not in f:
            raise ValueError(f"❌ Re={Re} data not found in file!")
        
        # ===== LR data =====
        grp_lr = f[g_lr]
        # Read actual grid dimensions from attributes
        nx_lr = int(grp_lr.attrs['nx'])
        ny_lr = int(grp_lr.attrs['ny'])
        lx_lr = float(grp_lr.attrs['lx'])
        ly_lr = float(grp_lr.attrs['ly'])
        
        # Load flattened data and reshape correctly
        u_flat_lr = grp_lr['u'][:].astype(np.float32)
        v_flat_lr = grp_lr['v'][:].astype(np.float32)
        p_flat_lr = grp_lr['p'][:].astype(np.float32)
        
        lr_u = u_flat_lr.reshape((ny_lr, nx_lr)).T  # Shape: (nx_lr, ny_lr)
        lr_v = v_flat_lr.reshape((ny_lr, nx_lr)).T
        lr_p = p_flat_lr.reshape((ny_lr, nx_lr)).T
        
        print(f"   LR: nx={nx_lr}, ny={ny_lr}, domain: {lx_lr}×{ly_lr} (aspect: {lx_lr/ly_lr:.2f}:1)")
        print(f"       Data shape: {lr_u.shape}")
        
        # ===== HR data =====
        grp_hr = f[g_hr]
        # Read actual grid dimensions from attributes
        nx_hr = int(grp_hr.attrs['nx'])
        ny_hr = int(grp_hr.attrs['ny'])
        lx_hr = float(grp_hr.attrs['lx'])
        ly_hr = float(grp_hr.attrs['ly'])
        
        # Load flattened data and reshape correctly
        u_flat_hr = grp_hr['u'][:].astype(np.float32)
        v_flat_hr = grp_hr['v'][:].astype(np.float32)
        p_flat_hr = grp_hr['p'][:].astype(np.float32)
        
        hr_u = u_flat_hr.reshape((ny_hr, nx_hr)).T  # Shape: (nx_hr, ny_hr)
        hr_v = v_flat_hr.reshape((ny_hr, nx_hr)).T
        hr_p = p_flat_hr.reshape((ny_hr, nx_hr)).T
        
        print(f"   HR: nx={nx_hr}, ny={ny_hr}, domain: {lx_hr}×{ly_hr} (aspect: {lx_hr/ly_hr:.2f}:1)")
        print(f"       Data shape: {hr_u.shape}")
    
    lr_fields = {'u': lr_u, 'v': lr_v, 'p': lr_p}
    hr_fields = {'u': hr_u, 'v': hr_v, 'p': hr_p}
    
    # Return actual dimensions
    return lr_fields, hr_fields, (nx_lr, ny_lr, lx_lr, ly_lr), (nx_hr, ny_hr, lx_hr, ly_hr)


def plot_contours_3fields(fields_dict, title_prefix, nx, ny, lx, ly, save_path=None):
    """
    Plot contours for u, v, p fields with proper aspect ratio.
    Matches the style from bfs_ml_accelerated_unet.py
    
    Args:
        fields_dict: Dictionary with 'u', 'v', 'p' arrays of shape (nx, ny)
        title_prefix: Title prefix for the plot
        nx, ny: Grid dimensions
        lx, ly: Physical domain size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    
    # Create coordinate grids
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    X, Y = np.meshgrid(x, y)
    
    # U velocity contour
    im1 = axes[0, 0].contourf(X, Y, fields_dict['u'].T, levels=20, cmap='RdBu')
    axes[0, 0].set_title('U Velocity', fontsize=12)
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # V velocity contour
    im2 = axes[0, 1].contourf(X, Y, fields_dict['v'].T, levels=20, cmap='RdBu')
    axes[0, 1].set_title('V Velocity', fontsize=12)
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Pressure contour
    im3 = axes[1, 0].contourf(X, Y, fields_dict['p'].T, levels=20, cmap='viridis')
    axes[1, 0].set_title('Pressure', fontsize=12)
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Velocity magnitude and streamlines
    u_mag = np.sqrt(fields_dict['u']**2 + fields_dict['v']**2)
    im4 = axes[1, 1].contourf(X, Y, u_mag.T, levels=20, cmap='plasma')
    axes[1, 1].set_title('Velocity Magnitude with Streamlines', fontsize=12)
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].set_aspect('equal')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Add streamlines
    axes[1, 1].streamplot(X, Y, fields_dict['u'].T, fields_dict['v'].T, 
                         color='white', linewidth=0.35, density=1.5)
    
    plt.suptitle(f'{title_prefix}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")
    
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def plot_centerline_comparison(gt_fields, pred_fields, nx, ny, lx, ly, save_path=None):
    """
    Plot centerline comparisons for u and v velocity.
    Matches the style from bfs_ml_accelerated_unet.py
    
    Args:
        gt_fields: Ground truth fields dictionary
        pred_fields: Predicted fields dictionary
        nx, ny: Grid dimensions
        lx, ly: Physical domain size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Create coordinate arrays
    x_coords = np.linspace(0, lx, nx)
    y_coords = np.linspace(0, ly, ny)
    
    # Find vertical centerline (x = Lx/2)
    x_center_idx = nx // 2
    # Find horizontal centerline (y = Ly/2)
    y_center_idx = ny // 2
    
    # Extract centerline data - fields have shape (nx, ny)
    # Vertical centerline: x=constant, y varies -> fields[x_idx, :] gives values along y
    # Horizontal centerline: y=constant, x varies -> fields[:, y_idx] gives values along x
    u_vertical = gt_fields['u'][x_center_idx, :]  # Shape: (ny,)
    u_vertical_pred = pred_fields['u'][x_center_idx, :]  # Shape: (ny,)
    v_horizontal = gt_fields['v'][:, y_center_idx]  # Shape: (nx,)
    v_horizontal_pred = pred_fields['v'][:, y_center_idx]  # Shape: (nx,)
    
    # Compute metrics
    u_vertical_diff = np.abs(u_vertical - u_vertical_pred)
    v_horizontal_diff = np.abs(v_horizontal - v_horizontal_pred)
    
    u_l2 = np.sqrt(np.mean(u_vertical_diff**2))
    u_max = np.max(u_vertical_diff)
    u_mean = np.mean(u_vertical_diff)
    
    v_l2 = np.sqrt(np.mean(v_horizontal_diff**2))
    v_max = np.max(v_horizontal_diff)
    v_mean = np.mean(v_horizontal_diff)
    
    # Plot U velocity along vertical centerline (x = Lx/2)
    ax1 = axes[0]
    ax1.plot(u_vertical_pred, y_coords,
             'b-o', linewidth=2, markersize=4, label='ML Prediction (U-Net)', alpha=0.7)
    ax1.plot(u_vertical, y_coords,
             'r--s', linewidth=2, markersize=4, label='Ground Truth', alpha=0.7)
    ax1.set_xlabel('U Velocity', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_title('U Velocity along Vertical Centerline (x=Lx/2)', fontsize=11)
    # Add metrics text box
    metrics_text = f"L2: {u_l2:.2e}\nMax: {u_max:.2e}\nMean: {u_mean:.2e}"
    ax1.text(0.02, 0.98, metrics_text, transform=ax1.transAxes, 
             fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot V velocity along horizontal centerline (y = Ly/2)
    ax2 = axes[1]
    ax2.plot(x_coords, v_horizontal,
             'r--s', linewidth=2, markersize=4, label='Ground Truth', alpha=0.7)
    ax2.plot(x_coords, v_horizontal_pred,
             'b-o', linewidth=2, markersize=4, label='ML Prediction (U-Net)', alpha=0.7)
    ax2.set_xlabel('X Position', fontsize=12)
    ax2.set_ylabel('V Velocity', fontsize=12)
    ax2.set_title('V Velocity along Horizontal Centerline (y=Ly/2)', fontsize=11)
    # Add metrics text box
    metrics_text = f"L2: {v_l2:.2e}\nMax: {v_max:.2e}\nMean: {v_mean:.2e}"
    ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, 
             fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'BFS Centerline Velocity Comparison - Progressive U-Net (Re={TEST_RE})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {save_path}")
    
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    
    # Calculate and print differences
    print("\n" + "="*70)
    print("CENTERLINE COMPARISON STATISTICS")
    print("="*70)
    
    print(f"U Velocity (vertical centerline):")
    print(f"  Max absolute difference: {u_max:.6e}")
    print(f"  Mean absolute difference: {u_mean:.6e}")
    print(f"  RMS difference: {u_l2:.6e}")
    
    print(f"\nV Velocity (horizontal centerline):")
    print(f"  Max absolute difference: {v_max:.6e}")
    print(f"  Mean absolute difference: {v_mean:.6e}")
    print(f"  RMS difference: {v_l2:.6e}")
    print("="*70)


def compute_metrics(gt_fields, pred_fields):
    """Compute MAE and NMAE for each field component."""
    metrics = {}
    
    for comp in ['u', 'v', 'p']:
        gt = gt_fields[comp]
        pred = pred_fields[comp]
        
        mae = np.mean(np.abs(gt - pred))
        data_range = np.max(gt) - np.min(gt)
        nmae = (mae / (data_range + 1e-8)) * 100
        
        metrics[comp] = {'MAE': mae, 'NMAE': nmae}
    
    return metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*80)
    print("🔬 PROGRESSIVE U-NET MODEL CHECKING - BFS Re=100")
    print("="*80)
    
    # Create output directory
    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(OUTPUT_DIR, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        print(f"\n📁 Output directory: {run_dir}")
    
    # ========================================================================
    # STEP 1: Load normalization statistics
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 1: Loading Normalization Statistics")
    print("="*80)
    
    if not os.path.exists(NORM_STATS_FILE):
        print(f"❌ Normalization stats file not found: {NORM_STATS_FILE}")
        print("   Please run the training notebook first to generate statistics.")
        return
    
    norm_stats = load_normalization_stats(NORM_STATS_FILE)
    
    # ========================================================================
    # STEP 2: Load trained models
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 2: Loading Trained Progressive U-Net Models")
    print("="*80)
    
    models = {}
    for stage_idx, target_dim in enumerate(STAGE_DIMS):
        if stage_idx == 0:
            input_dim = LR_DIM
        else:
            input_dim = STAGE_DIMS[stage_idx - 1]
        
        stage_name = f"{input_dim}to{target_dim}"
        model_path = f"unet_stage_{stage_name}_{OTHER_DETAILS}.h5"
        
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            print("   Please run the training notebook first to generate models.")
            return
        
        print(f"\n   Loading stage: {stage_name}")
        print(f"   Path: {model_path}")
        
        # Load model with custom objects
        model = tf.keras.models.load_model(model_path, compile=False)
        models[stage_name] = model
        
        print(f"   ✅ Loaded successfully")
    
    print(f"\n✅ All {len(models)} stage models loaded successfully!")
    
    # ========================================================================
    # STEP 3: Load BFS Re=100 ground truth data (BEFORE aspect ratio correction)
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 3: Loading BFS Re=100 Ground Truth Data")
    print("="*80)
    
    if not os.path.exists(TEST_FILE):
        print(f"❌ Test file not found: {TEST_FILE}")
        print("   Please update TEST_FILE path in the configuration section.")
        return
    
    lr_fields_rect, hr_fields_rect, (nx_lr, ny_lr, lx_lr, ly_lr), (nx_hr, ny_hr, lx_hr, ly_hr) = load_bfs_data(
        TEST_FILE, TEST_RE, LR_DIM, HR_DIM
    )
    
    print("\n✅ Ground truth data loaded")
    
    # ========================================================================
    # STEP 4: Plot ground truth BEFORE aspect ratio correction
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 4: Plotting Ground Truth (BEFORE Aspect Ratio Correction)")
    print("="*80)
    
    print(f"\n📊 Plotting LR ground truth (rectangular)...")
    save_path_lr_rect = os.path.join(run_dir, "1_groundtruth_LR_rectangular.png") if SAVE_PLOTS else None
    plot_contours_3fields(
        lr_fields_rect, 
        f"Ground Truth LR (BEFORE Correction) - BFS Re={TEST_RE}",
        nx_lr, ny_lr, lx_lr, ly_lr,
        save_path=save_path_lr_rect
    )
    
    print(f"\n📊 Plotting HR ground truth (rectangular)...")
    save_path_hr_rect = os.path.join(run_dir, "2_groundtruth_HR_rectangular.png") if SAVE_PLOTS else None
    plot_contours_3fields(
        hr_fields_rect,
        f"Ground Truth HR (BEFORE Correction) - BFS Re={TEST_RE}",
        nx_hr, ny_hr, lx_hr, ly_hr,
        save_path=save_path_hr_rect
    )
    
    # ========================================================================
    # STEP 5: Apply aspect ratio correction (rectangular → square)
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 5: Applying Aspect Ratio Correction (Rectangular → Square)")
    print("="*80)
    
    print("\n🔄 Correcting LR fields...")
    lr_fields_square = reshape_rectangular_to_square(lr_fields_rect, nx_lr, ny_lr, lx_lr, ly_lr)
    
    print("\n🔄 Correcting HR fields...")
    hr_fields_square = reshape_rectangular_to_square(hr_fields_rect, nx_hr, ny_hr, lx_hr, ly_hr)
    
    # ========================================================================
    # STEP 6: Plot ground truth AFTER aspect ratio correction
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 6: Plotting Ground Truth (AFTER Aspect Ratio Correction)")
    print("="*80)
    
    L_square_lr = min(lx_lr, ly_lr)
    L_square_hr = min(lx_hr, ly_hr)
    
    print(f"\n📊 Plotting LR ground truth (square)...")
    save_path_lr_square = os.path.join(run_dir, "3_groundtruth_LR_square.png") if SAVE_PLOTS else None
    plot_contours_3fields(
        lr_fields_square,
        f"Ground Truth LR (AFTER Correction) - BFS Re={TEST_RE}",
        nx_lr, nx_lr, L_square_lr, L_square_lr,
        save_path=save_path_lr_square
    )
    
    print(f"\n📊 Plotting HR ground truth (square)...")
    save_path_hr_square = os.path.join(run_dir, "4_groundtruth_HR_square.png") if SAVE_PLOTS else None
    plot_contours_3fields(
        hr_fields_square,
        f"Ground Truth HR (AFTER Correction) - BFS Re={TEST_RE}",
        nx_hr, nx_hr, L_square_hr, L_square_hr,
        save_path=save_path_hr_square
    )
    
    # ========================================================================
    # STEP 7: Run progressive prediction through all 5 stages
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 7: Running Progressive Prediction (All 5 Stages)")
    print("="*80)
    
    # Prepare initial LR input (stack u, v, p as 3 channels and normalize)
    lr_3channel = np.stack([lr_fields_square['u'], lr_fields_square['v'], lr_fields_square['p']], axis=-1)
    lr_3channel = lr_3channel[np.newaxis, ...]  # Add batch dimension
    lr_3channel_norm = normalize_with_stats(lr_3channel, norm_stats)
    
    print(f"\n🔢 Initial LR input shape: {lr_3channel_norm.shape}")
    
    # Progressive prediction with explicit interpolation
    current_output = lr_3channel_norm
    
    for stage_idx, target_dim in enumerate(STAGE_DIMS):
        if stage_idx == 0:
            input_dim = LR_DIM
        else:
            input_dim = STAGE_DIMS[stage_idx - 1]
        
        stage_name = f"{input_dim}to{target_dim}"
        model = models[stage_name]
        
        print(f"\n   Stage {stage_idx+1}/5: {stage_name}")
        print(f"      Current output shape: {current_output.shape}")
        
        # Step 1: Interpolate current output to target resolution
        # TensorFlow bilinear interpolation
        interpolated = tf.image.resize(
            current_output,
            size=(target_dim, target_dim),
            method='bilinear'
        ).numpy()
        
        print(f"      After interpolation: {interpolated.shape}")
        
        # Step 2: Pass through U-Net to refine
        prediction = model.predict(interpolated, verbose=0)
        
        print(f"      After U-Net refinement: {prediction.shape}")
        print(f"      ✅ Stage {stage_name} completed")
        
        # Update for next stage
        current_output = prediction
    
    # Final prediction (normalized)
    final_pred_norm = current_output[0]  # Remove batch dimension
    
    print(f"\n✅ Progressive prediction completed!")
    print(f"   Final prediction shape: {final_pred_norm.shape}")
    
    # Denormalize prediction
    final_pred_square_3channel = denormalize_with_stats(final_pred_norm[np.newaxis, ...], norm_stats)[0]
    
    # Split into individual fields
    pred_fields_square = {
        'u': final_pred_square_3channel[:, :, 0],
        'v': final_pred_square_3channel[:, :, 1],
        'p': final_pred_square_3channel[:, :, 2]
    }
    
    # ========================================================================
    # STEP 8: Plot prediction in square coordinates
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 8: Plotting Prediction (Square Coordinates)")
    print("="*80)
    
    print(f"\n📊 Plotting predicted HR fields (square)...")
    save_path_pred_square = os.path.join(run_dir, "5_prediction_HR_square.png") if SAVE_PLOTS else None
    plot_contours_3fields(
        pred_fields_square,
        f"Predicted HR (Square Coordinates) - BFS Re={TEST_RE}",
        nx_hr, nx_hr, L_square_hr, L_square_hr,
        save_path=save_path_pred_square
    )
    
    # ========================================================================
    # STEP 9: Reverse aspect ratio correction (square → rectangular)
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 9: Reversing Aspect Ratio Correction (Square → Rectangular)")
    print("="*80)
    
    print("\n🔄 Reversing correction for prediction...")
    pred_fields_rect = reshape_square_to_rectangular(pred_fields_square, nx_hr, ny_hr, lx_hr, ly_hr)
    
    # ========================================================================
    # STEP 10: Plot prediction in original rectangular geometry
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 10: Plotting Prediction (Original Rectangular Geometry)")
    print("="*80)
    
    print(f"\n📊 Plotting predicted HR fields (rectangular)...")
    save_path_pred_rect = os.path.join(run_dir, "6_prediction_HR_rectangular.png") if SAVE_PLOTS else None
    plot_contours_3fields(
        pred_fields_rect,
        f"Predicted HR (AFTER Reverse Correction) - BFS Re={TEST_RE}",
        nx_hr, ny_hr, lx_hr, ly_hr,
        save_path=save_path_pred_rect
    )
    
    # ========================================================================
    # STEP 11: Centerline comparison
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 11: Centerline Comparison (Ground Truth vs Prediction)")
    print("="*80)
    
    print(f"\n📊 Plotting centerline comparisons...")
    save_path_centerline = os.path.join(run_dir, "7_centerline_comparison.png") if SAVE_PLOTS else None
    plot_centerline_comparison(
        hr_fields_rect, pred_fields_rect,
        nx_hr, ny_hr, lx_hr, ly_hr,
        save_path=save_path_centerline
    )
    
    # ========================================================================
    # STEP 12: Compute and display metrics
    # ========================================================================
    print(f"\n{'='*80}")
    print("STEP 12: Computing Error Metrics")
    print("="*80)
    
    metrics = compute_metrics(hr_fields_rect, pred_fields_rect)
    
    print("\n📊 Error Metrics (in original rectangular geometry):")
    print("-" * 60)
    for comp in ['u', 'v', 'p']:
        mae = metrics[comp]['MAE']
        nmae = metrics[comp]['NMAE']
        print(f"   {comp.upper()}: MAE = {mae:.6f}, NMAE = {nmae:.2f}%")
    print("-" * 60)
    
    # Save metrics to file
    if SAVE_PLOTS:
        metrics_file = os.path.join(run_dir, "metrics.txt")
        with open(metrics_file, 'w') as f:
            f.write(f"BFS Re={TEST_RE} Error Metrics\n")
            f.write("="*60 + "\n")
            f.write(f"Model: {OTHER_DETAILS}\n")
            f.write(f"Geometry: {lx_hr}×{ly_hr} (rectangular)\n")
            f.write(f"Resolution: {nx_hr}×{ny_hr}\n")
            f.write("\n")
            f.write("Error Metrics:\n")
            f.write("-"*60 + "\n")
            for comp in ['u', 'v', 'p']:
                mae = metrics[comp]['MAE']
                nmae = metrics[comp]['NMAE']
                f.write(f"{comp.upper()}: MAE = {mae:.6f}, NMAE = {nmae:.2f}%\n")
        
        print(f"\n💾 Metrics saved to: {metrics_file}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*80}")
    print("✅ MODEL CHECKING COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    if SAVE_PLOTS:
        print(f"\n📁 All results saved to: {run_dir}")
        print("\n📋 Generated files:")
        print("   1. 1_groundtruth_LR_rectangular.png  - LR ground truth (before correction)")
        print("   2. 2_groundtruth_HR_rectangular.png  - HR ground truth (before correction)")
        print("   3. 3_groundtruth_LR_square.png       - LR ground truth (after correction)")
        print("   4. 4_groundtruth_HR_square.png       - HR ground truth (after correction)")
        print("   5. 5_prediction_HR_square.png        - Prediction (square coordinates)")
        print("   6. 6_prediction_HR_rectangular.png   - Prediction (original geometry)")
        print("   7. 7_centerline_comparison.png       - Centerline u, v comparison")
        print("   8. metrics.txt                       - Error metrics")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
