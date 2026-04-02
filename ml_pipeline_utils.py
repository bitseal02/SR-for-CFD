"""Shared ML preprocessing and progressive U-Net inference utilities for BFS workflows."""

import os
from typing import Dict, List

import numpy as np
import tensorflow as tf
from scipy import interpolate


def bicubic_interpolate_batch(x, target_size):
    """Bicubic interpolation for batched multi-channel images."""
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    else:
        x_tensor = x

    result = tf.image.resize(x_tensor, target_size, method="bicubic")
    if is_numpy:
        return result.numpy()
    return result


def standardize_with_stats(arr, mean, std):
    """Standardize array with provided mean/std."""
    std = 1e-8 if std == 0 else std
    return (arr - mean) / std


def inverse_standardize(arr, mean, std):
    """Inverse standardization."""
    return arr * std + mean


def normalize_with_stats_3channel(arr_3ch: np.ndarray, stats: Dict) -> np.ndarray:
    """Apply per-channel normalization to [u, v, p] channels."""
    normalized = np.zeros_like(arr_3ch)
    for ch_idx, ch_name in enumerate(["u", "v", "p"]):
        mean, std = stats[ch_name]
        normalized[..., ch_idx] = (arr_3ch[..., ch_idx] - mean) / std
    return normalized


def denormalize_with_stats_3channel(arr_3ch: np.ndarray, stats: Dict) -> np.ndarray:
    """Inverse per-channel normalization for [u, v, p] channels."""
    denormalized = np.zeros_like(arr_3ch)
    for ch_idx, ch_name in enumerate(["u", "v", "p"]):
        mean, std = stats[ch_name]
        denormalized[..., ch_idx] = arr_3ch[..., ch_idx] * std + mean
    return denormalized


def normalize_per_sample(arr: np.ndarray):
    """Per-sample Z-score normalization for (N, H, W, 3) arrays."""
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
    """Inverse of normalize_per_sample."""
    result = np.copy(arr).astype(np.float32)
    n_samples = arr.shape[0]
    for i in range(n_samples):
        for c in range(3):
            mean = stats[i, c, 0]
            std = stats[i, c, 1]
            result[i, :, :, c] = arr[i, :, :, c] * std + mean
    return result


def make_coord_channels_batch(dim: int, lx_arr, ly_arr) -> np.ndarray:
    """Generate normalized coordinate channels [x_bar, y_bar] for a batch."""
    n_samples = len(lx_arr)
    coords = np.zeros((n_samples, dim, dim, 2), dtype=np.float32)
    for i in range(n_samples):
        lx = lx_arr[i]
        ly = ly_arr[i]
        scale = max(lx, ly)
        x = np.linspace(0, lx, dim) / scale
        y = np.linspace(0, ly, dim) / scale
        xx, yy = np.meshgrid(x, y)
        coords[i, :, :, 0] = xx
        coords[i, :, :, 1] = yy
    return coords


def reshape_rectangular_to_square(fields: Dict[str, np.ndarray],
                                  nx_rect: int, ny_rect: int,
                                  lx: float, ly: float) -> Dict[str, np.ndarray]:
    """Resample rectangular grid fields to a square coordinate system."""
    print(f"  Resampling rectangular ({nx_rect}x{ny_rect}) -> square ({nx_rect}x{nx_rect})...")
    print(f"  Physical domain: {lx}x{ly} (aspect ratio: {lx/ly:.2f}:1)")

    x_rect = np.linspace(0, lx, nx_rect)
    y_rect = np.linspace(0, ly, ny_rect)

    l_square = max(lx, ly)
    x_square = np.linspace(0, l_square, nx_rect)
    y_square = np.linspace(0, l_square, nx_rect)

    fields_square = {}
    for component in ["u", "v", "p"]:
        field_rect = fields[component]
        interpolator = interpolate.RectBivariateSpline(y_rect, x_rect, field_rect, kx=3, ky=3)
        field_square = interpolator(y_square, x_square)
        fields_square[component] = field_square
        print(f"    {component.upper()}: {field_rect.shape} -> {field_square.shape}")

    return fields_square


def reshape_square_to_rectangular(fields: Dict[str, np.ndarray],
                                  nx_rect: int, ny_rect: int,
                                  lx: float, ly: float) -> Dict[str, np.ndarray]:
    """Resample square grid fields back to rectangular coordinates."""
    nx_square = fields["u"].shape[0]

    print(f"  Resampling square ({nx_square}x{nx_square}) -> rectangular ({nx_rect}x{ny_rect})...")
    print(f"  Physical domain: {lx}x{ly} (aspect ratio: {lx/ly:.2f}:1)")

    l_square = max(lx, ly)
    x_square = np.linspace(0, l_square, nx_square)
    y_square = np.linspace(0, l_square, nx_square)

    x_rect = np.linspace(0, lx, nx_rect)
    y_rect = np.linspace(0, ly, ny_rect)

    fields_rect = {}
    for component in ["u", "v", "p"]:
        field_square = fields[component]
        interpolator = interpolate.RectBivariateSpline(y_square, x_square, field_square, kx=3, ky=3)
        field_rect = interpolator(y_rect, x_rect)
        fields_rect[component] = field_rect
        print(f"    {component.upper()}: {field_square.shape} -> {field_rect.shape}")

    return fields_rect


def ml_super_resolution_unet(coarse_fields: Dict[str, np.ndarray],
                             lr_dim: int, hr_dim: int,
                             stage_dims: List[int],
                             model_name_pattern: str,
                             lx: float = 1.0, ly: float = 1.0) -> Dict[str, np.ndarray]:
    """Run progressive U-Net super-resolution from lr_dim to hr_dim."""
    print(f"\n{'='*70}")
    print(f"STEP 2: ML Super-Resolution - Progressive U-Net ({lr_dim}x{lr_dim} -> {hr_dim}x{hr_dim})")
    print(f"  Cascaded stages: {lr_dim} -> {' -> '.join(map(str, stage_dims))}")
    print("  Per-sample normalization: ENABLED")
    print(f"  Coordinate channels: ENABLED  (lx={lx}, ly={ly})")
    print(f"{'='*70}")

    x_lr_3ch = np.stack([
        coarse_fields["u"],
        coarse_fields["v"],
        coarse_fields["p"],
    ], axis=-1).astype(np.float32)
    x_lr_batch = x_lr_3ch[np.newaxis]

    print("\nPreparing 5-channel input...")
    print(f"  Flow field shape: {x_lr_batch.shape}")
    print(
        f"  Value ranges: U=[{x_lr_3ch[...,0].min():.4f}, {x_lr_3ch[...,0].max():.4f}], "
        f"V=[{x_lr_3ch[...,1].min():.4f}, {x_lr_3ch[...,1].max():.4f}], "
        f"P=[{x_lr_3ch[...,2].min():.4f}, {x_lr_3ch[...,2].max():.4f}]"
    )

    x_lr_norm, sample_stats = normalize_per_sample(x_lr_batch)
    print("  Per-sample stats computed:")
    for c, name in enumerate(["U", "V", "P"]):
        print(f"    {name}: mean={sample_stats[0,c,0]:.6f}, std={sample_stats[0,c,1]:.6f}")

    coords = make_coord_channels_batch(lr_dim, [lx], [ly])
    x_current = np.concatenate([x_lr_norm, coords], axis=-1)
    print(f"  5-channel input shape: {x_current.shape}")

    print(f"\nLoading and running {len(stage_dims)} U-Net stages...")

    prev_dim = lr_dim
    for stage_idx, target_dim in enumerate(stage_dims):
        stage_name = f"{prev_dim}to{target_dim}"
        model_file = model_name_pattern.format(from_dim=prev_dim, to_dim=target_dim)

        print(f"\n  Stage {stage_idx+1}/{len(stage_dims)}: {stage_name}")
        print(f"    Loading model: {model_file}")

        if not os.path.exists(model_file):
            print("    ERROR: Model file not found!")
            raise FileNotFoundError(f"U-Net stage model not found: {model_file}")

        unet_model = tf.keras.models.load_model(model_file, compile=False)
        print("    U-Net model loaded successfully")

        print(f"    Interpolating: {x_current.shape} -> (1, {target_dim}, {target_dim}, 5)")
        x_interp = bicubic_interpolate_batch(x_current, (target_dim, target_dim))
        print(f"    Interpolated shape: {x_interp.shape}")

        print("    Predicting residual correction...")
        residual = unet_model.predict(x_interp, verbose=0)

        residual = residual * 0.1
        x_flow = x_interp[..., :3] + residual

        print(f"    Flow output shape: {x_flow.shape}")
        print(
            f"    Value ranges: U=[{x_flow[0,...,0].min():.4f}, {x_flow[0,...,0].max():.4f}], "
            f"V=[{x_flow[0,...,1].min():.4f}, {x_flow[0,...,1].max():.4f}], "
            f"P=[{x_flow[0,...,2].min():.4f}, {x_flow[0,...,2].max():.4f}]"
        )

        if np.isnan(x_flow).any() or np.isinf(x_flow).any():
            nan_count = np.isnan(x_flow).sum()
            inf_count = np.isinf(x_flow).sum()
            print(f"    WARNING: Stage output contains {nan_count} NaN and {inf_count} Inf values")
            print("    Replacing with zeros to prevent propagation...")
            x_flow = np.nan_to_num(x_flow, nan=0.0, posinf=0.0, neginf=0.0)

        if stage_idx < len(stage_dims) - 1:
            coords_new = make_coord_channels_batch(target_dim, [lx], [ly])
            x_current = np.concatenate([x_flow, coords_new], axis=-1)
        else:
            x_current = x_flow

        prev_dim = target_dim

    print("\nDenormalizing output using per-sample stats...")
    hr_3ch_real = denormalize_per_sample(x_current, sample_stats)

    hr_fields = {
        "u": hr_3ch_real[0, ..., 0],
        "v": hr_3ch_real[0, ..., 1],
        "p": hr_3ch_real[0, ..., 2],
    }

    print("  Final output ranges:")
    print(f"    U: [{hr_fields['u'].min():.6f}, {hr_fields['u'].max():.6f}]")
    print(f"    V: [{hr_fields['v'].min():.6f}, {hr_fields['v'].max():.6f}]")
    print(f"    P: [{hr_fields['p'].min():.6f}, {hr_fields['p'].max():.6f}]")

    print("\n  Progressive U-Net super-resolution complete")
    return hr_fields
