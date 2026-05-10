import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def create_timestamped_output_dir(base_dir: str = "outputs") -> Path:
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    out_dir = Path(base_dir) / f"{timestamp} (burgers)"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def normalize_per_sample(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    result = np.copy(arr).astype(np.float32)
    n_samples = arr.shape[0]
    for i in range(n_samples):
        for c in range(3):
            mean = stats[i, c, 0]
            std = stats[i, c, 1]
            result[i, :, :, c] = arr[i, :, :, c] * std + mean
    return result


def bicubic_interpolate_batch(x: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    x_tensor = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.image.resize(x_tensor, target_size, method="bicubic")
    return y.numpy()


def make_xt_coord_channels_batch(n_samples: int, nt: int, nx: int) -> np.ndarray:
    """
    Build [x_bar, t_bar] channels in [0, 1].

    Axis convention in this file:
      - H axis (rows) = time index
      - W axis (cols) = space index
    """
    coords = np.zeros((n_samples, nt, nx, 2), dtype=np.float32)
    x_bar = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    t_bar = np.linspace(0.0, 1.0, nt, dtype=np.float32)
    xx, tt = np.meshgrid(x_bar, t_bar)
    coords[..., 0] = xx
    coords[..., 1] = tt
    return coords


# -----------------------------------------------------------------------------
# Burgers PDE definitions
# -----------------------------------------------------------------------------

def initial_condition_common(x: np.ndarray) -> np.ndarray:
    """Common choice: u(0,x) = -sin(pi x)."""
    return -np.sin(np.pi * x)


def burgers_residual_tf(u_fn, t: tf.Tensor, x: tf.Tensor, nu: float = 0.01 / np.pi) -> tf.Tensor:
    """
    TensorFlow PDE residual for:
      u_t + u*u_x - nu*u_xx = 0

    This is the TF2-safe equivalent of the legacy tf.gradients-based formula.
    """
    nu_tf = tf.constant(nu, dtype=t.dtype)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([t, x])
        u = u_fn(t, x)

    u_t = tape.gradient(u, t)
    u_x = tape.gradient(u, x)
    u_xx = tape.gradient(u_x, x)
    del tape

    return u_t + u * u_x - nu_tf * u_xx


def _burgers_one_step_rusanov(
    u: np.ndarray,
    dx: float,
    dt: float,
    nu: float,
    max_abs_u: float = 10.0,
) -> np.ndarray:
    """Single stable step for Burgers using Rusanov convective flux + central diffusion."""
    un = np.nan_to_num(u, nan=0.0, posinf=max_abs_u, neginf=-max_abs_u).astype(np.float64, copy=True)
    un = np.clip(un, -max_abs_u, max_abs_u)
    fn = 0.5 * un * un

    flux = np.zeros(len(un) - 1, dtype=np.float64)
    for i in range(len(un) - 1):
        a = max(abs(un[i]), abs(un[i + 1]))
        flux[i] = 0.5 * (fn[i] + fn[i + 1]) - 0.5 * a * (un[i + 1] - un[i])

    u_next = un.copy()
    for i in range(1, len(un) - 1):
        convection = -(dt / dx) * (flux[i] - flux[i - 1])
        diffusion = nu * dt * (un[i + 1] - 2.0 * un[i] + un[i - 1]) / (dx * dx)
        u_next[i] = un[i] + convection + diffusion

    u_next = np.nan_to_num(u_next, nan=0.0, posinf=max_abs_u, neginf=-max_abs_u)
    u_next = np.clip(u_next, -max_abs_u, max_abs_u)
    u_next[0] = 0.0
    u_next[-1] = 0.0
    return u_next


def _advance_row_adaptive(
    u_row: np.ndarray,
    dx: float,
    dt_target: float,
    nu: float,
    cfl_safety: float = 0.45,
    max_abs_u: float = 10.0,
) -> np.ndarray:
    """Advance one time-row with adaptive internal substeps for stability."""
    u_cur = np.nan_to_num(u_row, nan=0.0, posinf=max_abs_u, neginf=-max_abs_u).astype(np.float64, copy=True)
    u_cur = np.clip(u_cur, -max_abs_u, max_abs_u)

    t_local = 0.0
    while t_local < dt_target - 1e-14:
        umax = max(np.max(np.abs(u_cur)), 1e-8)
        dt_conv = cfl_safety * dx / umax
        dt_diff = cfl_safety * 0.5 * dx * dx / max(nu, 1e-12)
        dt_internal = min(dt_conv, dt_diff, dt_target - t_local)
        u_cur = _burgers_one_step_rusanov(u_cur, dx, dt_internal, nu, max_abs_u=max_abs_u)
        t_local += dt_internal

    return u_cur


def solve_burgers_1d_fd(
    nx: int,
    nt: int,
    t_end: float = 1.0,
    x_min: float = -1.0,
    x_max: float = 1.0,
    nu: float = 0.01 / np.pi,
    cfl_safety: float = 0.45,
    max_abs_u: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Solve viscous Burgers in 1D with stable finite-volume/finite-difference updates:
      u_t + u*u_x = nu*u_xx

    BC: u(t, x_min)=0, u(t, x_max)=0
    IC: u(0, x)=-sin(pi x)

    Returns:
      x (nx,), t (nt,), U (nt, nx)
    """
    if nx < 5 or nt < 5:
        raise ValueError("nx and nt must be >= 5")

    x = np.linspace(x_min, x_max, nx)
    t = np.linspace(0.0, t_end, nt)
    dx = (x_max - x_min) / (nx - 1)
    dt_output = t_end / (nt - 1)

    U = np.zeros((nt, nx), dtype=np.float64)
    U[0, :] = initial_condition_common(x)
    U[:, 0] = 0.0
    U[:, -1] = 0.0

    u_current = np.clip(U[0].copy(), -max_abs_u, max_abs_u)

    for n in range(nt - 1):
        target_time = t[n + 1]
        current_time = t[n]

        u_current = _advance_row_adaptive(
            u_current,
            dx,
            target_time - current_time,
            nu,
            cfl_safety=cfl_safety,
            max_abs_u=max_abs_u,
        )
        current_time = target_time

        U[n + 1, :] = u_current

    return x, t, U


# -----------------------------------------------------------------------------
# U-Net cascade for Burgers (using existing 5 saved models)
# -----------------------------------------------------------------------------

def burgers_scalar_to_3ch_option_a(u_grid: np.ndarray) -> np.ndarray:
    """
    Option A mapping requested by user:
      channel-0 = u
      channel-1 = 0
      channel-2 = 0
    """
    zeros = np.zeros_like(u_grid)
    return np.stack([u_grid, zeros, zeros], axis=-1).astype(np.float32)


def ml_super_resolution_burgers(
    coarse_u: np.ndarray,
    stage_dims: List[int],
    model_name_pattern: str,
) -> np.ndarray:
    """
    Run progressive U-Net cascade from coarse_u (square nt x nx) to final stage dimension.

    The saved U-Net models are reused exactly; only channel mapping is adapted for Burgers.
    """
    if coarse_u.shape[0] != coarse_u.shape[1]:
        raise ValueError("This script currently expects square coarse grids (nt == nx).")

    lr_dim = coarse_u.shape[0]
    if stage_dims[-1] <= lr_dim:
        raise ValueError("Final stage dimension must be larger than coarse dimension.")

    x_lr_3ch = burgers_scalar_to_3ch_option_a(coarse_u)
    x_lr_batch = x_lr_3ch[np.newaxis, ...]

    x_lr_norm, sample_stats = normalize_per_sample(x_lr_batch)
    coords = make_xt_coord_channels_batch(n_samples=1, nt=lr_dim, nx=lr_dim)
    x_current = np.concatenate([x_lr_norm, coords], axis=-1)

    prev_dim = lr_dim
    for idx, target_dim in enumerate(stage_dims):
        model_file = model_name_pattern.format(from_dim=prev_dim, to_dim=target_dim)
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Stage model file not found: {model_file}")

        print(f"[ML] Stage {idx + 1}/{len(stage_dims)}: {prev_dim}->{target_dim}")
        print(f"[ML] Loading model: {model_file}")

        model = tf.keras.models.load_model(model_file, compile=False)
        x_interp = bicubic_interpolate_batch(x_current, (target_dim, target_dim))
        residual = model.predict(x_interp, verbose=0) * 0.1

        x_flow = x_interp[..., :3] + residual
        x_flow = np.nan_to_num(x_flow, nan=0.0, posinf=0.0, neginf=0.0)

        if idx < len(stage_dims) - 1:
            coords_new = make_xt_coord_channels_batch(n_samples=1, nt=target_dim, nx=target_dim)
            x_current = np.concatenate([x_flow, coords_new], axis=-1)
        else:
            x_current = x_flow

        prev_dim = target_dim

    hr_3ch = denormalize_per_sample(x_current, sample_stats)
    hr_u = hr_3ch[0, ..., 0]
    return hr_u


# -----------------------------------------------------------------------------
# ML-initialized Burgers fine refinement
# -----------------------------------------------------------------------------

def enforce_ic_bc(U: np.ndarray, x: np.ndarray) -> None:
    U[0, :] = initial_condition_common(x)
    U[:, 0] = 0.0
    U[:, -1] = 0.0


def burgers_refinement_residual(U: np.ndarray, dt: float, dx: float, nu: float) -> np.ndarray:
    """Compute PDE residual r for interior stencil points."""
    nt, nx = U.shape
    r = np.zeros((nt - 1, nx - 2), dtype=np.float64)

    for n in range(nt - 1):
        for i in range(1, nx - 1):
            u_t = (U[n + 1, i] - U[n, i]) / dt
            u_x = (U[n, i + 1] - U[n, i - 1]) / (2.0 * dx)
            u_xx = (U[n, i + 1] - 2.0 * U[n, i] + U[n, i - 1]) / (dx * dx)
            r[n, i - 1] = u_t + U[n, i] * u_x - nu * u_xx

    return r


def refine_burgers_from_ml_init(
    U_init: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    nu: float = 0.01 / np.pi,
    max_refine_iters: int = 3000,
    omega: float = 0.6,
    tol: float = 1e-5,
    print_every: int = 200,
    cfl_safety: float = 0.35,
    max_abs_u: float = 10.0,
) -> Tuple[np.ndarray, List[float]]:
    """
    Refine ML initialization by repeated forward sweeps toward Burgers-consistent evolution.
    Each sweep updates U[n+1] by blending current value with a stable one-step PDE advance.
    """
    U = np.nan_to_num(U_init, nan=0.0, posinf=max_abs_u, neginf=-max_abs_u).astype(np.float64, copy=True)
    U = np.clip(U, -max_abs_u, max_abs_u)
    nt, nx = U.shape
    dx = (x[-1] - x[0]) / (nx - 1)
    dt = (t[-1] - t[0]) / (nt - 1)

    enforce_ic_bc(U, x)
    history = []

    for it in range(1, max_refine_iters + 1):
        U_prev = U.copy()

        # Causal forward sweep from t_n to t_{n+1}
        for n in range(nt - 1):
            predicted_next = _advance_row_adaptive(
                U[n, :],
                dx,
                dt,
                nu,
                cfl_safety=cfl_safety,
                max_abs_u=max_abs_u,
            )
            U[n + 1, 1:-1] = (1.0 - omega) * U[n + 1, 1:-1] + omega * predicted_next[1:-1]
            U[n + 1, :] = np.nan_to_num(U[n + 1, :], nan=0.0, posinf=max_abs_u, neginf=-max_abs_u)
            U[n + 1, :] = np.clip(U[n + 1, :], -max_abs_u, max_abs_u)

        enforce_ic_bc(U, x)
        U = np.nan_to_num(U, nan=0.0, posinf=max_abs_u, neginf=-max_abs_u)
        U = np.clip(U, -max_abs_u, max_abs_u)

        delta_rms = float(np.sqrt(np.mean((U - U_prev) ** 2)))
        history.append(delta_rms)

        r = burgers_refinement_residual(U, dt, dx, nu)
        residual_rms = float(np.sqrt(np.mean(r * r)))

        if it % print_every == 0 or it == 1:
            print(
                f"[Refine] iter={it:5d} change_rms={delta_rms:.6e} "
                f"residual_rms={residual_rms:.6e}"
            )

        if not np.isfinite(delta_rms) or not np.isfinite(residual_rms):
            raise FloatingPointError(
                "Refinement became non-finite. Try lowering refine_omega or max_abs_u."
            )

        if delta_rms < tol:
            print(f"[Refine] Converged at iter={it} with change_rms={delta_rms:.6e}")
            break

    return U, history


# -----------------------------------------------------------------------------
# Metrics and plots
# -----------------------------------------------------------------------------

def l2_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def save_comparison_plots(
    x_coarse: np.ndarray,
    t_coarse: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    U_coarse: np.ndarray,
    U_ref: np.ndarray,
    U_ml: np.ndarray,
    U_ml_refined: np.ndarray,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap panel (reference style, using native grid resolutions)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor="#dddddd")

    all_fields = [U_coarse, U_ref, U_ml, U_ml_refined]
    vmin = min(np.min(f) for f in all_fields)
    vmax = max(np.max(f) for f in all_fields)

    panel_specs = [
        (axes[0, 0], t_coarse, x_coarse, U_coarse, f"Coarse Input ({len(t_coarse)}x{len(x_coarse)})", "nearest"),
        (axes[0, 1], t, x, U_ref, "Exact", "auto"),
        (axes[1, 0], t, x, U_ml, "Prediction (Upscaled)", "auto"),
        (axes[1, 1], t, x, U_ml_refined, "Prediction (Final)", "auto"),
    ]

    for ax, t_axis, x_axis, U, title, shading in panel_specs:
        m = ax.pcolormesh(t_axis, x_axis, U.T, shading=shading, cmap="viridis", vmin=vmin, vmax=vmax)
        for t_mark in np.linspace(float(t_axis.min()), float(t_axis.max()), 5)[1:-1]:
            ax.axvline(t_mark, color="white", linewidth=1.2, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$x$")
        fig.colorbar(m, ax=ax)
        ax.grid(False)

    fig.suptitle(r"$u(t,x)$", fontsize=16)
    fig.tight_layout()
    fig.savefig(out_dir / "burgers_heatmap_comparison.png", dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)

    # Error heatmaps vs exact (reference style)
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4), facecolor="#dddddd")
    e_ml = np.abs(U_ml - U_ref)
    e_refined = np.abs(U_ml_refined - U_ref)

    m1 = ax2[0].pcolormesh(t, x, e_ml.T, shading="auto", cmap="magma")
    ax2[0].set_title(r"$|Prediction\ (Upscaled)-Exact|$")
    ax2[0].set_xlabel(r"$t$")
    ax2[0].set_ylabel(r"$x$")
    for t_mark in np.linspace(float(t.min()), float(t.max()), 5)[1:-1]:
        ax2[0].axvline(t_mark, color="white", linewidth=1.0, alpha=0.85)
    fig2.colorbar(m1, ax=ax2[0])
    ax2[0].grid(False)

    m2 = ax2[1].pcolormesh(t, x, e_refined.T, shading="auto", cmap="magma")
    ax2[1].set_title(r"$|Prediction\ (Final)-Exact|$")
    ax2[1].set_xlabel(r"$t$")
    ax2[1].set_ylabel(r"$x$")
    for t_mark in np.linspace(float(t.min()), float(t.max()), 5)[1:-1]:
        ax2[1].axvline(t_mark, color="white", linewidth=1.0, alpha=0.85)
    fig2.colorbar(m2, ax=ax2[1])
    ax2[1].grid(False)

    fig2.tight_layout()
    fig2.savefig(out_dir / "burgers_error_heatmaps.png", dpi=200, facecolor=fig2.get_facecolor())
    plt.close(fig2)

    # Profiles at fixed times requested: t = 0.25, 0.50, 0.75
    target_times = [0.25, 0.50, 0.75]
    profile_indices = [int(np.argmin(np.abs(t - t_target))) for t_target in target_times]

    fig3, axes3 = plt.subplots(1, 3, figsize=(12.5, 3.8), facecolor="#dddddd")
    if not isinstance(axes3, np.ndarray):
        axes3 = np.array([axes3])

    for ax, idx, t_target in zip(axes3, profile_indices, target_times):
        ax.plot(x, U_ref[idx], "b-", linewidth=2.6, label="Exact")
        ax.plot(x, U_ml_refined[idx], "r--", linewidth=2.2, label="Prediction")
        ax.set_title(rf"$t = {t_target:.2f}$")
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$u(t,x)$")
        ax.set_xlim(float(np.min(x)), float(np.max(x)))
        ax.grid(False)

    handles, labels = axes3[0].get_legend_handles_labels()
    fig3.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig3.suptitle(r"$t$", y=0.98)
    fig3.tight_layout(rect=[0, 0.03, 1, 0.94])
    fig3.savefig(out_dir / "burgers_profiles_comparison.png", dpi=200)
    plt.close(fig3)


def save_reference_style_xt_plot(
    x: np.ndarray,
    t: np.ndarray,
    U: np.ndarray,
    save_path: Path,
    n_data_points: int = 100,
    seed: int = 123,
) -> None:
    """Save a single u(t,x) heatmap styled like the reference figure."""
    rng = np.random.default_rng(seed)

    # Split displayed data points across initial line and two boundaries.
    n_init = max(1, n_data_points // 3)
    n_top = max(1, (n_data_points - n_init) // 2)
    n_bottom = max(1, n_data_points - n_init - n_top)

    x_init = rng.choice(x, size=n_init, replace=True)
    t_init = np.zeros(n_init, dtype=float)

    t_top = rng.choice(t, size=n_top, replace=True)
    x_top = np.full(n_top, float(np.max(x)))

    t_bottom = rng.choice(t, size=n_bottom, replace=True)
    x_bottom = np.full(n_bottom, float(np.min(x)))

    t_data = np.concatenate([t_init, t_top, t_bottom])
    x_data = np.concatenate([x_init, x_top, x_bottom])

    fig, ax = plt.subplots(figsize=(8.8, 3.1), facecolor="#dddddd")

    mesh = ax.pcolormesh(
        t,
        x,
        U.T,
        shading="auto",
        cmap="viridis",
        vmin=-1.0,
        vmax=1.0,
    )

    for t_mark in np.linspace(float(t.min()), float(t.max()), 5)[1:-1]:
        ax.axvline(t_mark, color="white", linewidth=1.6, alpha=0.9)

    ax.scatter(
        t_data,
        x_data,
        marker="x",
        c="black",
        s=55,
        linewidths=1.8,
        label=f"Data ({n_data_points} points)",
        zorder=3,
    )

    ax.set_title(r"$u(t,x)$", pad=10)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$x$")
    ax.set_xlim(float(t.min()), float(t.max()))
    ax.set_ylim(float(x.min()), float(x.max()))
    ax.legend(loc="upper right", frameon=False)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.01)
    cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_metrics(
    U_ref: np.ndarray,
    U_ml: np.ndarray,
    U_ml_refined: np.ndarray,
    out_path: Path,
) -> Dict[str, float]:
    metrics = {
        "ml_vs_ref_l2": l2_error(U_ml, U_ref),
        "ml_vs_ref_mae": mae(U_ml, U_ref),
        "ml_vs_ref_max_abs": max_abs_error(U_ml, U_ref),
        "ml_refined_vs_ref_l2": l2_error(U_ml_refined, U_ref),
        "ml_refined_vs_ref_mae": mae(U_ml_refined, U_ref),
        "ml_refined_vs_ref_max_abs": max_abs_error(U_ml_refined, U_ref),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("BURGERS 1D COMPARISON METRICS\n")
        f.write("=" * 60 + "\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.8e}\n")

    return metrics


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Burgers 1D workflow with coarse solve, U-Net SR, and ML-initialized fine refinement."
    )

    parser.add_argument("--coarse-nx", type=int, default=10, help="Coarse spatial points (nx).")
    parser.add_argument("--coarse-time-points", type=int, default=10, help="Coarse temporal points (nt).")
    parser.add_argument("--fine-nx", type=int, default=400, help="Fine spatial points (nx).")
    parser.add_argument("--fine-time-points", type=int, default=400, help="Fine temporal points (nt).")
    parser.add_argument("--input-dim", type=int, default=None, help="Backward-compatible alias for square coarse grid (nx=nt).")
    parser.add_argument("--output-dim", type=int, default=None, help="Backward-compatible alias for square fine grid (nx=nt).")
    parser.add_argument("--coarse-dim", type=int, default=None, help="Backward-compatible alias for square coarse grid (nx=nt).")
    parser.add_argument("--fine-dim", type=int, default=None, help="Backward-compatible alias for square fine grid (nx=nt).")
    parser.add_argument(
        "--stage-dims",
        type=int,
        nargs="+",
        default=[20, 40, 80, 200, 400],
        help="Progressive U-Net target dimensions.",
    )

    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--x-min", type=float, default=-1.0)
    parser.add_argument("--x-max", type=float, default=1.0)
    parser.add_argument("--nu", type=float, default=0.01 / np.pi)

    parser.add_argument(
        "--model-suffix",
        type=str,
        default="progressive_residual_unet_(20-40-80-200-400)_trained along with bfs 100,300",
        help="Suffix part of model files.",
    )
    parser.add_argument(
        "--model-pattern",
        type=str,
        default="unet_stage_{from_dim}to{to_dim}_{model_suffix}.h5",
        help="Model filename pattern with {from_dim}, {to_dim}, {model_suffix} keys.",
    )

    parser.add_argument("--refine-iters", type=int, default=3000)
    parser.add_argument("--refine-omega", type=float, default=0.6)
    parser.add_argument("--refine-tol", type=float, default=1e-5)
    parser.add_argument("--out-dir", type=str, default="outputs")

    return parser.parse_args()


def main(
    coarse_nx: int = 10,
    coarse_time_points: int = 10,
    fine_nx: int = 400,
    fine_time_points: int = 400,
    stage_dims: List[int] | None = None,
    t_end: float = 1.0,
    x_min: float = -1.0,
    x_max: float = 1.0,
    nu: float = 0.01 / np.pi,
    model_suffix: str = "burgers2d_finetune",
    #model_suffix: str = "progressive_residual_unet_(20-40-80-200-400)_trained along with bfs 100,300",
    model_pattern: str = "unet_stage_{from_dim}to{to_dim}_{model_suffix}.h5",
    refinement_enabled: bool = False,
    refine_iters: int = 3000,
    refine_omega: float = 0.6,
    refine_tol: float = 1e-5,
    refine_cfl_safety: float = 0.35,
    max_abs_u: float = 10.0,
    display_data_points: int = 100,
    out_dir_base: str = "outputs",
) -> None:
    if stage_dims is None:
        stage_dims = [20, 40, 80, 200, 400]

    coarse_nt = coarse_time_points
    fine_nt = fine_time_points

    if coarse_nx < 5 or coarse_nt < 5 or fine_nx < 5 or fine_nt < 5:
        raise ValueError("coarse/fine nx and time-points must be >= 5")

    if fine_nx <= coarse_nx and fine_nt <= coarse_nt:
        raise ValueError("Fine grid should be larger than coarse grid in nx and/or time-points")

    # Saved stage models are square (dim x dim), so we require square coarse/fine grids.
    if coarse_nx != coarse_nt:
        raise ValueError("Current U-Net pipeline requires coarse_nx == coarse_time_points")
    if fine_nx != fine_nt:
        raise ValueError("Current U-Net pipeline requires fine_nx == fine_time_points")

    if stage_dims[-1] != fine_nx:
        raise ValueError(
            f"Last stage dimension ({stage_dims[-1]}) must equal fine_nx ({fine_nx})"
        )

    model_name_pattern = model_pattern.format(
        from_dim="{from_dim}",
        to_dim="{to_dim}",
        model_suffix=model_suffix,
    )

    out_dir = create_timestamped_output_dir(out_dir_base)
    print(f"[INFO] Output directory: {out_dir}")
    interrupted_during_refinement = False

    # 1) Coarse numerical Burgers
    print("\n" + "=" * 70)
    print("STEP 1: COARSE NUMERICAL BURGERS")
    print("=" * 70)
    x_coarse, t_coarse, U_coarse = solve_burgers_1d_fd(
        nx=coarse_nx,
        nt=coarse_nt,
        t_end=t_end,
        x_min=x_min,
        x_max=x_max,
        nu=nu,
        max_abs_u=max_abs_u,
    )

    # 2) Fine numerical reference
    print("\n" + "=" * 70)
    print("STEP 2: FINE NUMERICAL BURGERS (REFERENCE)")
    print("=" * 70)
    x_fine, t_fine, U_ref = solve_burgers_1d_fd(
        nx=fine_nx,
        nt=fine_nt,
        t_end=t_end,
        x_min=x_min,
        x_max=x_max,
        nu=nu,
        max_abs_u=max_abs_u,
    )

    # 3) ML upscaling using same saved U-Net cascade
    print("\n" + "=" * 70)
    print("STEP 3: ML UPSCALING WITH EXISTING 5 U-NET MODELS")
    print("=" * 70)
    U_ml = ml_super_resolution_burgers(
        coarse_u=U_coarse,
        stage_dims=stage_dims,
        model_name_pattern=model_name_pattern,
    )

    # Save ML-only intermediate outputs immediately so manual interruption still leaves artifacts.
    np.savez_compressed(
        out_dir / "burgers_step3_ml_only.npz",
        x_coarse=x_coarse,
        t_coarse=t_coarse,
        U_coarse=U_coarse,
        x_fine=x_fine,
        t_fine=t_fine,
        U_ref=U_ref,
        U_ml=U_ml,
        nu=nu,
    )

    # 4) Optional ML-initialized fine refinement solve
    if refinement_enabled:
        print("\n" + "=" * 70)
        print("STEP 4: ML-INITIALIZED FINE REFINEMENT")
        print("=" * 70)
        try:
            U_ml_refined, residual_history = refine_burgers_from_ml_init(
                U_init=U_ml,
                x=x_fine,
                t=t_fine,
                nu=nu,
                max_refine_iters=refine_iters,
                omega=refine_omega,
                tol=refine_tol,
                cfl_safety=refine_cfl_safety,
                max_abs_u=max_abs_u,
            )
        except KeyboardInterrupt:
            interrupted_during_refinement = True
            print("\n[WARN] Refinement interrupted by user. Saving available outputs...")
            U_ml_refined = U_ml.copy()
            residual_history = []
    else:
        print("\n" + "=" * 70)
        print("STEP 4: REFINEMENT DISABLED")
        print("=" * 70)
        U_ml_refined = U_ml.copy()
        residual_history = []

    # Save arrays
    np.savez_compressed(
        out_dir / "burgers_fields.npz",
        x_coarse=x_coarse,
        t_coarse=t_coarse,
        U_coarse=U_coarse,
        x_fine=x_fine,
        t_fine=t_fine,
        U_ref=U_ref,
        U_ml=U_ml,
        U_ml_refined=U_ml_refined,
        residual_history=np.array(residual_history, dtype=np.float64),
        nu=nu,
    )

    # Save residual history plot
    if residual_history:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(np.arange(1, len(residual_history) + 1), residual_history, "b-")
        ax.set_yscale("log")
        ax.set_xlabel("Refinement Iteration")
        ax.set_ylabel("Residual RMS")
        ax.set_title("ML-Initialized Refinement Residual History")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "refinement_residual_history.png", dpi=200)
        plt.close(fig)

    # Save visual comparisons using native grid resolutions
    save_comparison_plots(
        x_coarse=x_coarse,
        t_coarse=t_coarse,
        x=x_fine,
        t=t_fine,
        U_coarse=U_coarse,
        U_ref=U_ref,
        U_ml=U_ml,
        U_ml_refined=U_ml_refined,
        out_dir=out_dir,
    )

    save_reference_style_xt_plot(
        x=x_fine,
        t=t_fine,
        U=U_ml_refined,
        save_path=out_dir / "burgers_reference_style_plot.png",
        n_data_points=display_data_points,
    )

    metrics = save_metrics(U_ref, U_ml, U_ml_refined, out_dir / "metrics_summary.txt")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"{k}: {v:.8e}")
    print(f"Artifacts saved in: {out_dir}")
    if interrupted_during_refinement:
        print("Run ended with refinement interruption; U_ml_refined currently equals U_ml.")


if __name__ == "__main__":
    # Edit these values directly for quick manual runs.
    main(
        coarse_nx=10,
        coarse_time_points=10,
        fine_nx=400,
        fine_time_points=400,
        stage_dims=[20, 40, 80, 200, 400],
        refinement_enabled=False,
        refine_cfl_safety=0.35,
        max_abs_u=10.0,
        display_data_points=100,
    )
