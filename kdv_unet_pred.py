import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def create_timestamped_output_dir(base_dir: str = "outputs") -> Path:
	timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
	out_dir = Path(base_dir) / f"{timestamp} (kdv)"
	out_dir.mkdir(parents=True, exist_ok=True)
	return out_dir


def normalize_per_sample(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	arr = arr.astype(np.float32)
	n_samples = arr.shape[0]
	stats = np.zeros((n_samples, 3, 2), dtype=np.float32)
	normalized = np.copy(arr)

	for i in range(n_samples):
		for c in range(3):
			mean = float(arr[i, :, :, c].mean())
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
# KdV PDE definitions
# -----------------------------------------------------------------------------

def _sech(z: np.ndarray) -> np.ndarray:
	return 1.0 / np.cosh(z)


def initial_condition_kdv(
	x: np.ndarray,
	mode: str = "cos",
	amplitude: float = 1.0,
	k: float = 5.0,
	x0: float = 0.0,
) -> np.ndarray:
	if mode == "cos":
		return amplitude * np.cos(np.pi * x)
	if mode == "sin":
		return amplitude * np.sin(np.pi * x)
	if mode == "sech2":
		return amplitude * _sech(k * (x - x0)) ** 2
	raise ValueError("Unknown initial condition mode. Use 'cos', 'sin', or 'sech2'.")


def kdv_rhs(u: np.ndarray, dx: float, beta: float) -> np.ndarray:
	# Periodic finite differences using roll.
	u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
	u_xxx = (
		np.roll(u, -2)
		- 2.0 * np.roll(u, -1)
		+ 2.0 * np.roll(u, 1)
		- np.roll(u, 2)
	) / (2.0 * dx**3)
	return -u * u_x - beta * u_xxx


def kdv_rk4_step(u: np.ndarray, dx: float, dt: float, beta: float) -> np.ndarray:
	k1 = kdv_rhs(u, dx, beta)
	k2 = kdv_rhs(u + 0.5 * dt * k1, dx, beta)
	k3 = kdv_rhs(u + 0.5 * dt * k2, dx, beta)
	k4 = kdv_rhs(u + dt * k3, dx, beta)
	return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_kdv_1d(
	nx: int,
	nt: int,
	t_end: float = 1.0,
	x_min: float = -1.0,
	x_max: float = 1.0,
	beta: float = 0.0025,
	init_mode: str = "sech2",
	init_amplitude: float = 1.0,
	init_k: float = 5.0,
	init_x0: float = 0.0,
	substeps: int = 5,
	max_abs_u: float = 10.0,
	stability_factor: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	if nx < 5 or nt < 5:
		raise ValueError("nx and nt must be >= 5")

	x = np.linspace(x_min, x_max, nx)
	t = np.linspace(0.0, t_end, nt)
	dx = (x_max - x_min) / (nx - 1)
	dt_output = t_end / (nt - 1)

	U = np.zeros((nt, nx), dtype=np.float64)
	U[0, :] = initial_condition_kdv(
		x,
		mode=init_mode,
		amplitude=init_amplitude,
		k=init_k,
		x0=init_x0,
	)
	u_current = np.clip(U[0].copy(), -max_abs_u, max_abs_u)

	max_dt = stability_factor * dx**3 / max(beta, 1e-12)
	min_substeps = int(np.ceil(dt_output / max_dt))
	if min_substeps < 1:
		min_substeps = 1
	effective_substeps = max(substeps, min_substeps)
	if effective_substeps != substeps:
		print(
			f"[KDV] Increasing substeps from {substeps} to {effective_substeps} "
			f"for stability (dt<= {max_dt:.3e})."
		)

	dt = dt_output / effective_substeps
	for n in range(nt - 1):
		for _ in range(effective_substeps):
			u_current = kdv_rk4_step(u_current, dx, dt, beta)
			u_current = np.nan_to_num(u_current, nan=0.0, posinf=max_abs_u, neginf=-max_abs_u)
			u_current = np.clip(u_current, -max_abs_u, max_abs_u)

		U[n + 1, :] = u_current

	return x, t, U


# -----------------------------------------------------------------------------
# U-Net cascade for KdV (using existing saved models)
# -----------------------------------------------------------------------------

def kdv_scalar_to_3ch(u_grid: np.ndarray) -> np.ndarray:
	zeros = np.zeros_like(u_grid)
	return np.stack([u_grid, zeros, zeros], axis=-1).astype(np.float32)


def ml_super_resolution_kdv(
	coarse_u: np.ndarray,
	stage_dims: List[int],
	model_name_pattern: str,
) -> np.ndarray:
	if coarse_u.shape[0] != coarse_u.shape[1]:
		raise ValueError("This script currently expects square coarse grids (nt == nx).")

	lr_dim = coarse_u.shape[0]
	if stage_dims[-1] <= lr_dim:
		raise ValueError("Final stage dimension must be larger than coarse dimension.")

	x_lr_3ch = kdv_scalar_to_3ch(coarse_u)
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
# Plots
# -----------------------------------------------------------------------------

def save_heatmap_comparison(
	x_coarse: np.ndarray,
	t_coarse: np.ndarray,
	x_fine: np.ndarray,
	t_fine: np.ndarray,
	U_coarse: np.ndarray,
	U_fine: np.ndarray,
	U_pred: np.ndarray,
	out_dir: Path,
) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)

	all_fields = [U_coarse, U_fine, U_pred]
	vmin = min(float(np.min(f)) for f in all_fields)
	vmax = max(float(np.max(f)) for f in all_fields)

	fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#dddddd")

	panels = [
		(axes[0], t_coarse, x_coarse, U_coarse, f"Coarse ({len(t_coarse)}x{len(x_coarse)})"),
		(axes[1], t_fine, x_fine, U_fine, f"Fine ({len(t_fine)}x{len(x_fine)})"),
		(axes[2], t_fine, x_fine, U_pred, "Fine Prediction"),
	]

	for ax, t_axis, x_axis, U, title in panels:
		m = ax.contourf(t_axis, x_axis, U.T, levels=20, cmap="rainbow", vmin=vmin, vmax=vmax)
		for t_mark in np.linspace(float(t_axis.min()), float(t_axis.max()), 5)[1:-1]:
			ax.axvline(t_mark, color="white", linewidth=1.1, alpha=0.9)
		ax.set_title(title)
		ax.set_xlabel(r"$t$")
		ax.set_ylabel(r"$x$")
		fig.colorbar(m, ax=ax)
		ax.grid(False)

	fig.suptitle(r"$u(t,x)$", fontsize=16)
	fig.tight_layout()
	fig.savefig(out_dir / "kdv_heatmap_comparison.png", dpi=200, facecolor=fig.get_facecolor())
	plt.close(fig)


def save_time_profiles_comparison(
	x_coarse: np.ndarray,
	t_coarse: np.ndarray,
	x_fine: np.ndarray,
	t_fine: np.ndarray,
	U_coarse: np.ndarray,
	U_fine: np.ndarray,
	U_pred: np.ndarray,
	time_targets: List[float],
	out_dir: Path,
) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)

	profile_indices_coarse = [int(np.argmin(np.abs(t_coarse - tt))) for tt in time_targets]
	profile_indices_fine = [int(np.argmin(np.abs(t_fine - tt))) for tt in time_targets]

	fig, axes = plt.subplots(1, len(time_targets), figsize=(12, 3.8), facecolor="#dddddd")
	if not isinstance(axes, np.ndarray):
		axes = np.array([axes])

	for ax, idx_c, idx_f, t_target in zip(axes, profile_indices_coarse, profile_indices_fine, time_targets):
		ax.plot(x_coarse, U_coarse[idx_c], "k.-", linewidth=1.2, markersize=4, label="Coarse")
		ax.plot(x_fine, U_fine[idx_f], "b-", linewidth=2.0, label="Fine")
		ax.plot(x_fine, U_pred[idx_f], "r--", linewidth=2.0, label="Prediction")
		ax.set_title(rf"$t = {t_target:.2f}$")
		ax.set_xlabel(r"$x$")
		ax.set_ylabel(r"$u(t,x)$")
		ax.set_xlim(float(np.min(x_fine)), float(np.max(x_fine)))
		ax.grid(False)

	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
	fig.tight_layout(rect=[0, 0.06, 1, 0.98])
	fig.savefig(out_dir / "kdv_profiles_comparison.png", dpi=200, facecolor=fig.get_facecolor())
	plt.close(fig)


def save_reference_style_xt_plot(
	x: np.ndarray,
	t: np.ndarray,
	U: np.ndarray,
	save_path: Path,
	n_data_points: int = 100,
	seed: int = 123,
) -> None:
	rng = np.random.default_rng(seed)

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

	mesh = ax.contourf(t, x, U.T, levels=20, cmap="rainbow")

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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="KdV 1D workflow with coarse solve, fine solve, and U-Net SR prediction."
	)

	parser.add_argument("--coarse-nx", type=int, default=10)
	parser.add_argument("--coarse-time-points", type=int, default=10)
	parser.add_argument("--fine-nx", type=int, default=400)
	parser.add_argument("--fine-time-points", type=int, default=400)
	parser.add_argument("--input-dim", type=int, default=None)
	parser.add_argument("--output-dim", type=int, default=None)
	parser.add_argument("--coarse-dim", type=int, default=None)
	parser.add_argument("--fine-dim", type=int, default=None)

	parser.add_argument("--t-end", type=float, default=1.0)
	parser.add_argument("--x-min", type=float, default=-1.0)
	parser.add_argument("--x-max", type=float, default=1.0)
	parser.add_argument("--beta", type=float, default=0.0025)
	parser.add_argument("--substeps", type=int, default=5)

	parser.add_argument("--init-mode", type=str, default="cos")
	parser.add_argument("--init-amplitude", type=float, default=1.0)
	parser.add_argument("--init-k", type=float, default=5.0)
	parser.add_argument("--init-x0", type=float, default=0.0)

	parser.add_argument(
		"--stage-dims",
		type=int,
		nargs="+",
		default=[20, 40, 80, 200, 400],
	)
	parser.add_argument("--model-suffix", type=str, default="kdv_finetune")
	parser.add_argument(
		"--model-pattern",
		type=str,
		default="unet_stage_{from_dim}to{to_dim}_{model_suffix}.h5",
	)

	parser.add_argument("--max-abs-u", type=float, default=10.0)
	parser.add_argument("--stability-factor", type=float, default=0.1)
	parser.add_argument("--display-data-points", type=int, default=100)
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
	beta: float = 0.0025,
	substeps: int = 5,
	init_mode: str = "cos",
	init_amplitude: float = 1.0,
	init_k: float = 5.0,
	init_x0: float = 0.0,
	model_suffix: str = "kdv_finetune",
	model_pattern: str = "unet_stage_{from_dim}to{to_dim}_{model_suffix}.h5",
	max_abs_u: float = 10.0,
	stability_factor: float = 0.1,
	display_data_points: int = 100,
	time_targets: List[float] | None = None,
	out_dir_base: str = "outputs",
) -> None:
	if stage_dims is None:
		stage_dims = [20, 40, 80, 200, 400]

	if time_targets is None:
		time_targets = [0.20, 0.80]

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

	print("\n" + "=" * 70)
	print("STEP 1: COARSE NUMERICAL KDV")
	print("=" * 70)
	x_coarse, t_coarse, U_coarse = solve_kdv_1d(
		nx=coarse_nx,
		nt=coarse_nt,
		t_end=t_end,
		x_min=x_min,
		x_max=x_max,
		beta=beta,
		init_mode=init_mode,
		init_amplitude=init_amplitude,
		init_k=init_k,
		init_x0=init_x0,
		substeps=substeps,
		max_abs_u=max_abs_u,
		stability_factor=stability_factor,
	)

	print("\n" + "=" * 70)
	print("STEP 2: FINE NUMERICAL KDV (REFERENCE)")
	print("=" * 70)
	x_fine, t_fine, U_fine = solve_kdv_1d(
		nx=fine_nx,
		nt=fine_nt,
		t_end=t_end,
		x_min=x_min,
		x_max=x_max,
		beta=beta,
		init_mode=init_mode,
		init_amplitude=init_amplitude,
		init_k=init_k,
		init_x0=init_x0,
		substeps=substeps,
		max_abs_u=max_abs_u,
		stability_factor=stability_factor,
	)

	print("\n" + "=" * 70)
	print("STEP 3: ML UPSCALING WITH EXISTING U-NET MODELS")
	print("=" * 70)
	U_pred = ml_super_resolution_kdv(
		coarse_u=U_coarse,
		stage_dims=stage_dims,
		model_name_pattern=model_name_pattern,
	)

	np.savez_compressed(
		out_dir / "kdv_fields.npz",
		x_coarse=x_coarse,
		t_coarse=t_coarse,
		U_coarse=U_coarse,
		x_fine=x_fine,
		t_fine=t_fine,
		U_fine=U_fine,
		U_pred=U_pred,
		beta=beta,
		init_mode=init_mode,
		init_amplitude=init_amplitude,
		init_k=init_k,
		init_x0=init_x0,
	)

	save_heatmap_comparison(
		x_coarse=x_coarse,
		t_coarse=t_coarse,
		x_fine=x_fine,
		t_fine=t_fine,
		U_coarse=U_coarse,
		U_fine=U_fine,
		U_pred=U_pred,
		out_dir=out_dir,
	)

	save_time_profiles_comparison(
		x_coarse=x_coarse,
		t_coarse=t_coarse,
		x_fine=x_fine,
		t_fine=t_fine,
		U_coarse=U_coarse,
		U_fine=U_fine,
		U_pred=U_pred,
		time_targets=time_targets,
		out_dir=out_dir,
	)

	save_reference_style_xt_plot(
		x=x_fine,
		t=t_fine,
		U=U_pred,
		save_path=out_dir / "kdv_reference_style_plot.png",
		n_data_points=display_data_points,
	)

	print("\n" + "=" * 70)
	print("FINAL SUMMARY")
	print("=" * 70)
	print(f"Artifacts saved in: {out_dir}")


if __name__ == "__main__":
	# Edit these values directly for quick manual runs.
	main(
		coarse_nx=10,
		coarse_time_points=10,
		fine_nx=400,
		fine_time_points=400,
		stage_dims=[20, 40, 80, 200, 400],
		t_end=1.0,
		x_min=-1.0,
		x_max=1.0,
		beta=0.0025,
		substeps=100,
		init_mode="cos",
		init_amplitude=1.0,
		init_k=5.0,
		init_x0=0.0,
		model_suffix="kdv_finetune",
		model_pattern="unet_stage_{from_dim}to{to_dim}_burgers2d_finetune.h5",
		max_abs_u=10.0,
		stability_factor=0.1,
		display_data_points=100,
		time_targets=[0.20, 0.80],
		out_dir_base="outputs",
	)
