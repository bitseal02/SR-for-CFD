import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_GRID_SIZE = 400
DEFAULT_TIME_POINTS = 400


def initial_condition(x: np.ndarray) -> np.ndarray:
	"""u(0, x) = x^2 cos(pi x)."""
	return x**2 * np.cos(np.pi * x)


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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Simulate the 1D Allen-Cahn equation.")
	parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID_SIZE, help="Required grid size in x-direction.")
	parser.add_argument("--nx", type=int, default=None, help="Alias for --grid-size (if provided, it overrides --grid-size).")
	parser.add_argument("--time-points", type=int, default=DEFAULT_TIME_POINTS, help="Number of time samples including t=0 and t=t_end.")
	parser.add_argument("--t-end", type=float, default=1.0, help="Final simulation time.")
	parser.add_argument("--dt", type=float, default=1e-3, help="Time step.")
	parser.add_argument("--eps", type=float, default=1e-4, help="Diffusion coefficient.")
	parser.add_argument("--reaction", type=float, default=5.0, help="Reaction coefficient.")
	parser.add_argument("--out-dir", type=str, default="outputs", help="Parent output directory.")
	parser.add_argument("--prefix", type=str, default="allen_cahn_1d", help="Output file prefix.")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	nx = args.nx if args.nx is not None else args.grid_size

	x, times, snapshots = solve_allen_cahn_1d(
		nx=nx,
		time_points=args.time_points,
		t_end=args.t_end,
		dt=args.dt,
		eps=args.eps,
		reaction_strength=args.reaction,
	)

	data_path, heatmap_path, profiles_path = save_results(
		x=x,
		times=times,
		snapshots=snapshots,
		out_dir=Path(args.out_dir) / f"{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')} (allen)",
		prefix=args.prefix,
	)

	print(f"Simulation complete with nx={nx}, nt={len(times) - 1}.")
	print(f"Time points used: {len(times)} (effective dt={times[1] - times[0]:.6g})")
	print(f"Data saved to: {data_path}")
	print(f"Heatmap saved to: {heatmap_path}")
	print(f"Profiles plot saved to: {profiles_path}")


if __name__ == "__main__":
	main()
