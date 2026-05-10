import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


DEFAULT_GRID_SIZE = 400
DEFAULT_TIME_POINTS = 400
DEFAULT_COARSE_SIZE = 10
DEFAULT_COARSE_TIME_POINTS = 10


def initial_condition(x: np.ndarray) -> np.ndarray:
	"""u(0, x) = x^2 cos(pi x)."""
	return x**2 * np.cos(np.pi * x)


def create_timestamped_output_dir(base_dir: str = "outputs") -> Path:
	timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
	out_dir = Path(base_dir) / f"{timestamp} (allen)"
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


def solve_allen_cahn_1d(
	nx: int = DEFAULT_GRID_SIZE,
	time_points: int | None = DEFAULT_TIME_POINTS,
	t_end: float = 1.0,
	dt: float = 1e-3,
	eps: float = 1e-4,
	reaction_strength: float = 5.0,
	x_min: float = -1.0,
	x_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""
	Solve u_t = eps*u_xx + reaction_strength*(u - u^3) with periodic BCs.

	PDE in prompt form:
		u_t - eps*u_xx + reaction_strength*u^3 - reaction_strength*u = 0
	"""
	if nx < 8:
		raise ValueError("nx must be at least 8.")
	if time_points is not None and time_points < 2:
		raise ValueError("time_points must be at least 2.")
	if dt <= 0:
		raise ValueError("dt must be positive.")
	if t_end <= 0:
		raise ValueError("t_end must be positive.")

	# Periodic grid: endpoint=False avoids duplicated point at x_max.
	x = np.linspace(x_min, x_max, nx, endpoint=False)
	dx = (x_max - x_min) / nx

	u = initial_condition(x)

	# Fourier wavenumbers for periodic second derivative.
	k = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
	k2 = k * k

	if time_points is not None:
		n_steps = time_points - 1
		dt = t_end / n_steps
	else:
		n_steps = int(np.round(t_end / dt))
		if n_steps < 1:
			n_steps = 1
		dt = t_end / n_steps

	# IMEX Euler in Fourier space:
	# (u^{n+1} - u^n)/dt = eps*u_xx^{n+1} + reaction(u^n)
	# => u_hat^{n+1} = (u_hat^n + dt*FFT(reaction(u^n))) / (1 + dt*eps*k^2)
	denom = 1.0 + dt * eps * k2

	times = np.linspace(0.0, t_end, n_steps + 1)
	snapshots = np.empty((n_steps + 1, nx), dtype=float)
	snapshots[0] = u

	for n in range(1, n_steps + 1):
		reaction = reaction_strength * (u - u**3)
		u_hat_next = (np.fft.fft(u) + dt * np.fft.fft(reaction)) / denom
		u = np.fft.ifft(u_hat_next).real
		snapshots[n] = u

	return x, times, snapshots


def save_results(
	x: np.ndarray,
	times: np.ndarray,
	snapshots: np.ndarray,
	out_dir: Path,
	prefix: str,
) -> tuple[Path, Path, Path]:
	out_dir.mkdir(parents=True, exist_ok=True)

	data_path = out_dir / f"{prefix}.npz"
	heatmap_path = out_dir / f"{prefix}_heatmap.png"
	profiles_path = out_dir / f"{prefix}_profiles.png"

	np.savez_compressed(data_path, x=x, t=times, u=snapshots)

	# Secondary profile plot (useful for checking specific times).
	plt.figure(figsize=(9, 5))
	plt.plot(x, snapshots[0], label="t=0", linewidth=2)
	plt.plot(x, snapshots[len(times) // 2], label=f"t={times[len(times)//2]:.3f}", linewidth=2)
	plt.plot(x, snapshots[-1], label=f"t={times[-1]:.3f}", linewidth=2)
	plt.title("1D Allen-Cahn Solution")
	plt.xlabel("x")
	plt.ylabel("u(x,t)")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(profiles_path, dpi=150)
	plt.close()

	# Main plot: u(t, x) map with t on x-axis and x on y-axis.
	fig, ax = plt.subplots(figsize=(8, 3))
	mesh = ax.pcolormesh(
		times,
		x,
		snapshots.T,
		shading="auto",
		cmap="seismic",
		vmin=-1.0,
		vmax=1.0,
	)
	ax.set_title(r"$u(t, x)$")
	ax.set_xlabel(r"$t$")
	ax.set_ylabel(r"$x$")

	# Vertical reference lines to match the example style.
	for t_mark in (0.1 * times[-1], 0.9 * times[-1]):
		ax.axvline(t_mark, color="white", linewidth=1)

	cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
	cbar.ax.tick_params(labelsize=8)

	fig.tight_layout()
	fig.savefig(heatmap_path, dpi=150)
	plt.close(fig)

	return data_path, heatmap_path, profiles_path


def allen_scalar_to_3ch(u_grid: np.ndarray) -> np.ndarray:
	zeros = np.zeros_like(u_grid)
	return np.stack([u_grid, zeros, zeros], axis=-1).astype(np.float32)


def ml_super_resolution_allen(
	coarse_u: np.ndarray,
	stage_dims: List[int],
	model_name_pattern: str,
) -> np.ndarray:
	"""
	Run progressive U-Net cascade from coarse_u (square nt x nx) to final stage dimension.

	The saved U-Net models are reused exactly; only channel mapping is adapted for Allen-Cahn.
	"""
	if coarse_u.shape[0] != coarse_u.shape[1]:
		raise ValueError("This script currently expects square coarse grids (nt == nx).")

	lr_dim = coarse_u.shape[0]
	if stage_dims[-1] <= lr_dim:
		raise ValueError("Final stage dimension must be larger than coarse dimension.")

	x_lr_3ch = allen_scalar_to_3ch(coarse_u)
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


def save_comparison_plots(
	x_coarse: np.ndarray,
	t_coarse: np.ndarray,
	x: np.ndarray,
	t: np.ndarray,
	U_coarse: np.ndarray,
	U_ref: np.ndarray,
	U_ml: np.ndarray,
	out_dir: Path,
	t_end: float,
) -> None:
	out_dir.mkdir(parents=True, exist_ok=True)

	def _save_single_panel(
		t_axis: np.ndarray,
		x_axis: np.ndarray,
		U_panel: np.ndarray,
		filename: str,
		title: str,
		shading: str,
	) -> None:
		fig, ax = plt.subplots(figsize=(8, 3), facecolor="#dddddd")
		vmin = float(np.min(U_panel))
		vmax = float(np.max(U_panel))
		m = ax.pcolormesh(t_axis, x_axis, U_panel.T, shading=shading, cmap="seismic", vmin=vmin, vmax=vmax)
		for t_mark in np.linspace(float(t_axis.min()), float(t_axis.max()), 5)[1:-1]:
			ax.axvline(t_mark, color="white", linewidth=1.2, alpha=0.9)
		ax.set_title(title)
		ax.set_xlabel(r"$t$")
		ax.set_ylabel(r"$x$")
		ax.set_aspect("auto")
		fig.colorbar(m, ax=ax, pad=0.02)
		ax.grid(False)

		fig.tight_layout()
		fig.savefig(out_dir / filename, dpi=200, facecolor=fig.get_facecolor())
		plt.close(fig)

	# Single-panel heatmaps (coarse, fine reference, prediction)
	_save_single_panel(
		t_axis=t_coarse,
		x_axis=x_coarse,
		U_panel=U_coarse,
		filename="allen_heatmap_coarse.png",
		title=f"Coarse Input ({len(t_coarse)}x{len(x_coarse)})",
		shading="nearest",
	)
	_save_single_panel(
		t_axis=t,
		x_axis=x,
		U_panel=U_ref,
		filename="allen_heatmap_fine_ref.png",
		title="Exact (Fine)",
		shading="auto",
	)
	_save_single_panel(
		t_axis=t,
		x_axis=x,
		U_panel=U_ml,
		filename="allen_heatmap_prediction.png",
		title="Prediction (Upscaled)",
		shading="auto",
	)

	# Profiles at fixed times (scaled by t_end for robustness)
	target_times = [0.25 * t_end, 0.50 * t_end, 0.75 * t_end]
	profile_indices = [int(np.argmin(np.abs(t - t_target))) for t_target in target_times]

	fig2, axes2 = plt.subplots(1, 3, figsize=(12.5, 3.8), facecolor="#dddddd")
	if not isinstance(axes2, np.ndarray):
		axes2 = np.array([axes2])

	for ax, idx, t_target in zip(axes2, profile_indices, target_times):
		ax.plot(x, U_ref[idx], "b-", linewidth=2.6, label="Exact")
		ax.plot(x, U_ml[idx], "r--", linewidth=2.2, label="Prediction")
		ax.set_title(rf"$t = {t_target:.2f}$")
		ax.set_xlabel(r"$x$")
		ax.set_ylabel(r"$u(t,x)$")
		ax.set_xlim(float(np.min(x)), float(np.max(x)))
		ax.grid(False)

	handles, labels = axes2[0].get_legend_handles_labels()
	fig2.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03))
	fig2.suptitle(r"$t$", y=0.98)
	fig2.tight_layout(rect=[0, 0.03, 1, 0.94])
	fig2.savefig(out_dir / "allen_profiles_comparison.png", dpi=200)
	plt.close(fig2)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Allen-Cahn 1D workflow with coarse solve, U-Net SR, and comparison plots."
	)

	parser.add_argument("--coarse-nx", type=int, default=DEFAULT_COARSE_SIZE, help="Coarse spatial points (nx).")
	parser.add_argument("--coarse-time-points", type=int, default=DEFAULT_COARSE_TIME_POINTS, help="Coarse temporal points (nt).")
	parser.add_argument("--fine-nx", type=int, default=DEFAULT_GRID_SIZE, help="Fine spatial points (nx).")
	parser.add_argument("--fine-time-points", type=int, default=DEFAULT_TIME_POINTS, help="Fine temporal points (nt).")
	parser.add_argument("--input-dim", type=int, default=None, help="Alias for square coarse grid (nx=nt).")
	parser.add_argument("--output-dim", type=int, default=None, help="Alias for square fine grid (nx=nt).")
	parser.add_argument("--coarse-dim", type=int, default=None, help="Alias for square coarse grid (nx=nt).")
	parser.add_argument("--fine-dim", type=int, default=None, help="Alias for square fine grid (nx=nt).")
	parser.add_argument(
		"--stage-dims",
		type=int,
		nargs="+",
		default=[20, 40, 80, 200, 400],
		help="Progressive U-Net target dimensions.",
	)

	parser.add_argument("--grid-size", type=int, default=None, help="Alias for fine-nx.")
	parser.add_argument("--nx", type=int, default=None, help="Alias for fine-nx.")
	parser.add_argument("--time-points", type=int, default=None, help="Alias for fine-time-points.")
	parser.add_argument("--t-end", type=float, default=1.0, help="Final simulation time.")
	parser.add_argument("--dt", type=float, default=1e-3, help="Time step.")
	parser.add_argument("--eps", type=float, default=1e-4, help="Diffusion coefficient.")
	parser.add_argument("--reaction", type=float, default=5.0, help="Reaction coefficient.")

	parser.add_argument(
		"--model-suffix",
		type=str,
		#default="progressive_residual_unet_(20-40-80-200-400)_trained along with bfs 100,300",
		default="burgers2d_finetune",
		help="Suffix part of model files.",
	)
	parser.add_argument(
		"--model-pattern",
		type=str,
		default="unet_stage_{from_dim}to{to_dim}_{model_suffix}.h5",
		help="Model filename pattern with {from_dim}, {to_dim}, {model_suffix} keys.",
	)

	parser.add_argument("--out-dir", type=str, default="outputs", help="Parent output directory.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	coarse_nx = args.coarse_nx
	coarse_time_points = args.coarse_time_points
	fine_nx = args.fine_nx
	fine_time_points = args.fine_time_points

	if args.input_dim is not None:
		coarse_nx = args.input_dim
		coarse_time_points = args.input_dim
	if args.coarse_dim is not None:
		coarse_nx = args.coarse_dim
		coarse_time_points = args.coarse_dim
	if args.output_dim is not None:
		fine_nx = args.output_dim
		fine_time_points = args.output_dim
	if args.fine_dim is not None:
		fine_nx = args.fine_dim
		fine_time_points = args.fine_dim
	if args.nx is not None:
		fine_nx = args.nx
	if args.grid_size is not None:
		fine_nx = args.grid_size
	if args.time_points is not None:
		fine_time_points = args.time_points

	if coarse_nx < 5 or coarse_time_points < 5 or fine_nx < 5 or fine_time_points < 5:
		raise ValueError("coarse/fine nx and time-points must be >= 5")

	if fine_nx <= coarse_nx and fine_time_points <= coarse_time_points:
		raise ValueError("Fine grid should be larger than coarse grid in nx and/or time-points")

	if coarse_nx != coarse_time_points:
		raise ValueError("Current U-Net pipeline requires coarse_nx == coarse_time_points")
	if fine_nx != fine_time_points:
		raise ValueError("Current U-Net pipeline requires fine_nx == fine_time_points")

	stage_dims = args.stage_dims
	if stage_dims[-1] != fine_nx:
		raise ValueError(
			f"Last stage dimension ({stage_dims[-1]}) must equal fine_nx ({fine_nx})"
		)

	model_name_pattern = args.model_pattern.format(
		from_dim="{from_dim}",
		to_dim="{to_dim}",
		model_suffix=args.model_suffix,
	)

	out_dir = create_timestamped_output_dir(args.out_dir)
	print(f"[INFO] Output directory: {out_dir}")

	print("\n" + "=" * 70)
	print("STEP 1: COARSE ALLEN-CAHN")
	print("=" * 70)
	x_coarse, t_coarse, U_coarse = solve_allen_cahn_1d(
		nx=coarse_nx,
		time_points=coarse_time_points,
		t_end=args.t_end,
		dt=args.dt,
		eps=args.eps,
		reaction_strength=args.reaction,
	)

	print("\n" + "=" * 70)
	print("STEP 2: FINE ALLEN-CAHN (REFERENCE)")
	print("=" * 70)
	x_fine, t_fine, U_ref = solve_allen_cahn_1d(
		nx=fine_nx,
		time_points=fine_time_points,
		t_end=args.t_end,
		dt=args.dt,
		eps=args.eps,
		reaction_strength=args.reaction,
	)

	print("\n" + "=" * 70)
	print("STEP 3: ML UPSCALING WITH EXISTING U-NET MODELS")
	print("=" * 70)
	U_ml = ml_super_resolution_allen(
		coarse_u=U_coarse,
		stage_dims=stage_dims,
		model_name_pattern=model_name_pattern,
	)

	np.savez_compressed(
		out_dir / "allen_fields.npz",
		x_coarse=x_coarse,
		t_coarse=t_coarse,
		U_coarse=U_coarse,
		x_fine=x_fine,
		t_fine=t_fine,
		U_ref=U_ref,
		U_ml=U_ml,
		eps=args.eps,
		reaction_strength=args.reaction,
	)

	save_comparison_plots(
		x_coarse=x_coarse,
		t_coarse=t_coarse,
		x=x_fine,
		t=t_fine,
		U_coarse=U_coarse,
		U_ref=U_ref,
		U_ml=U_ml,
		out_dir=out_dir,
		t_end=args.t_end,
	)

	print("\n" + "=" * 70)
	print("FINAL SUMMARY")
	print("=" * 70)
	print(f"Coarse grid: {coarse_nx}x{coarse_time_points}")
	print(f"Fine grid:   {fine_nx}x{fine_time_points}")
	print(f"Artifacts saved in: {out_dir}")


if __name__ == "__main__":
	main()
