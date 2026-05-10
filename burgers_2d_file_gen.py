import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Burgers2DConfig:
	# Domain
	x_min: float = -1.0
	x_max: float = 1.0
	y_min: float = -1.0
	y_max: float = 1.0
	t_end: float = 0.5

	# Physics
	nu: float = 0.091 / np.pi

	# Numerical
	cfl: float = 0.4
	max_steps: int = 20000
	max_abs_u: float = 10.0

	# Data
	dims: Tuple[int, ...] = (10, 20, 40, 80, 200, 400)
	n_cases: int = 1
	ic_type: str = "sine"
	seed: int = 42
	re_values: Tuple[int, ...] = (100,)
	bc_type: str = "Burgers2D(periodic)"

	# Output
	output_dir: str = "outputs"
	output_tag: str = "burgers2d"
	save_plots: bool = True


def _timestamped_dir(base_dir: str, tag: str) -> Path:
	stamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
	out_dir = Path(base_dir) / f"{stamp} ({tag})"
	out_dir.mkdir(parents=True, exist_ok=True)
	return out_dir


def _make_grid(nx: int, ny: int, cfg: Burgers2DConfig) -> Tuple[np.ndarray, np.ndarray, float, float]:
	x = np.linspace(cfg.x_min, cfg.x_max, nx, dtype=np.float64)
	y = np.linspace(cfg.y_min, cfg.y_max, ny, dtype=np.float64)
	dx = (cfg.x_max - cfg.x_min) / (nx - 1)
	dy = (cfg.y_max - cfg.y_min) / (ny - 1)
	xx, yy = np.meshgrid(x, y, indexing="xy")
	return xx, yy, dx, dy


def _initial_condition(xx: np.ndarray, yy: np.ndarray, ic_type: str, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
	if ic_type == "sine":
		u0 = -np.sin(np.pi * xx) * np.sin(np.pi * yy)
		v0 = np.cos(np.pi * xx) * np.cos(np.pi * yy)
	elif ic_type == "taylor_green":
		u0 = np.sin(np.pi * xx) * np.cos(np.pi * yy)
		v0 = -np.cos(np.pi * xx) * np.sin(np.pi * yy)
	elif ic_type == "gaussian":
		cx, cy = 0.0, 0.0
		sigma = 0.35
		u0 = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
		v0 = -u0.copy()
	elif ic_type == "random_fourier":
		kx = rng.integers(1, 4)
		ky = rng.integers(1, 4)
		phase = rng.uniform(0.0, 2.0 * np.pi)
		u0 = np.sin(kx * np.pi * xx + phase) * np.sin(ky * np.pi * yy)
		v0 = np.cos(kx * np.pi * xx + phase) * np.cos(ky * np.pi * yy)
	else:
		raise ValueError(f"Unknown ic_type: {ic_type}")
	return u0.astype(np.float64), v0.astype(np.float64)


def _upwind_derivative(field: np.ndarray, vel: np.ndarray, dx: float, axis: int) -> np.ndarray:
	if axis == 1:
		f_left = np.roll(field, 1, axis=1)
		f_right = np.roll(field, -1, axis=1)
		d_pos = (field - f_left) / dx
		d_neg = (f_right - field) / dx
	else:
		f_down = np.roll(field, 1, axis=0)
		f_up = np.roll(field, -1, axis=0)
		d_pos = (field - f_down) / dx
		d_neg = (f_up - field) / dx

	return np.where(vel >= 0.0, d_pos, d_neg)


def _laplacian(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
	f_left = np.roll(field, 1, axis=1)
	f_right = np.roll(field, -1, axis=1)
	f_down = np.roll(field, 1, axis=0)
	f_up = np.roll(field, -1, axis=0)
	return (f_right - 2.0 * field + f_left) / (dx * dx) + (f_up - 2.0 * field + f_down) / (dy * dy)


def _compute_dt(u: np.ndarray, v: np.ndarray, dx: float, dy: float, nu: float, cfl: float) -> float:
	umax = max(np.max(np.abs(u)), 1e-8)
	vmax = max(np.max(np.abs(v)), 1e-8)
	dt_adv = cfl * min(dx / umax, dy / vmax)
	dt_diff = cfl * 0.25 * min(dx, dy) ** 2 / max(nu, 1e-12)
	return min(dt_adv, dt_diff)


def solve_burgers_2d(nx: int, ny: int, cfg: Burgers2DConfig, ic_type: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
	xx, yy, dx, dy = _make_grid(nx, ny, cfg)
	rng = np.random.default_rng(seed)
	u, v = _initial_condition(xx, yy, ic_type, rng)

	t = 0.0
	step = 0
	while t < cfg.t_end - 1e-12:
		dt = _compute_dt(u, v, dx, dy, cfg.nu, cfg.cfl)
		if t + dt > cfg.t_end:
			dt = cfg.t_end - t

		u_x = _upwind_derivative(u, u, dx, axis=1)
		u_y = _upwind_derivative(u, v, dy, axis=0)
		v_x = _upwind_derivative(v, u, dx, axis=1)
		v_y = _upwind_derivative(v, v, dy, axis=0)

		u_lap = _laplacian(u, dx, dy)
		v_lap = _laplacian(v, dx, dy)

		u = u + dt * (-(u * u_x + v * u_y) + cfg.nu * u_lap)
		v = v + dt * (-(u * v_x + v * v_y) + cfg.nu * v_lap)

		u = np.clip(u, -cfg.max_abs_u, cfg.max_abs_u)
		v = np.clip(v, -cfg.max_abs_u, cfg.max_abs_u)

		t += dt
		step += 1
		if step > cfg.max_steps:
			raise RuntimeError("Max steps exceeded; reduce dt or cfl, or lower t_end.")

	return u.astype(np.float32), v.astype(np.float32)


def _flatten_for_h5(field: np.ndarray) -> np.ndarray:
	return field.T.flatten()


def save_case_to_h5(
	file_path: Path,
	case_idx: int,
	re_value: int,
	cfg: Burgers2DConfig,
	fields_by_dim: Dict[int, Dict[str, np.ndarray]],
):
	with h5py.File(file_path, "a") as h5f:
		for dim, fields in fields_by_dim.items():
			group_name = f"Re{re_value}_mesh{dim}x{dim}"
			if group_name in h5f:
				del h5f[group_name]

			grp = h5f.create_group(group_name)
			grp.attrs["case_name"] = "burgers2d"
			grp.attrs["bc_type"] = cfg.bc_type
			grp.attrs["reynolds_number"] = float(re_value)
			grp.attrs["nx"] = int(dim)
			grp.attrs["ny"] = int(dim)
			grp.attrs["lx"] = float(cfg.x_max - cfg.x_min)
			grp.attrs["ly"] = float(cfg.y_max - cfg.y_min)
			grp.attrs["total_points"] = int(dim * dim)
			grp.attrs["case_index"] = int(case_idx)
			grp.attrs["nu"] = float(cfg.nu)
			grp.attrs["ic_type"] = cfg.ic_type

			x = np.linspace(cfg.x_min, cfg.x_max, dim, dtype=np.float32)
			y = np.linspace(cfg.y_min, cfg.y_max, dim, dtype=np.float32)
			xx, yy = np.meshgrid(x, y, indexing="xy")
			grp.create_dataset("x", data=_flatten_for_h5(xx.astype(np.float32)))
			grp.create_dataset("y", data=_flatten_for_h5(yy.astype(np.float32)))
			grp.create_dataset("u", data=_flatten_for_h5(fields["u"]))
			grp.create_dataset("v", data=_flatten_for_h5(fields["v"]))
			grp.create_dataset("p", data=_flatten_for_h5(fields["p"]))


def _plot_fields(out_dir: Path, dim: int, fields: Dict[str, np.ndarray], cfg: Burgers2DConfig, tag: str):
	u = fields["u"]
	v = fields["v"]
	speed = np.sqrt(u ** 2 + v ** 2)

	x = np.linspace(cfg.x_min, cfg.x_max, dim)
	y = np.linspace(cfg.y_min, cfg.y_max, dim)
	xx, yy = np.meshgrid(x, y, indexing="xy")

	fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
	levels = 20
	cm = "RdBu"

	im0 = axes[0].contourf(xx, yy, u, levels=levels, cmap=cm)
	axes[0].set_title("u(x,y)")
	fig.colorbar(im0, ax=axes[0])

	im1 = axes[1].contourf(xx, yy, v, levels=levels, cmap=cm)
	axes[1].set_title("v(x,y)")
	fig.colorbar(im1, ax=axes[1])

	im2 = axes[2].contourf(xx, yy, speed, levels=levels, cmap="viridis")
	axes[2].set_title("speed")
	fig.colorbar(im2, ax=axes[2])

	for ax in axes:
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_aspect("equal")

	fig.suptitle(f"Burgers2D @ {dim}x{dim} | {tag}")
	fig.savefig(out_dir / f"burgers2d_{tag}_{dim}x{dim}.png", dpi=200)
	plt.close(fig)


def generate_burgers_2d_dataset(cfg: Burgers2DConfig) -> Path:
	if cfg.n_cases < 1:
		raise ValueError("n_cases must be >= 1")

	if len(cfg.re_values) < cfg.n_cases:
		raise ValueError("re_values must have at least n_cases entries")

	out_dir = _timestamped_dir(cfg.output_dir, cfg.output_tag)
	h5_path = out_dir / f"burgers2d_fields_nu_{cfg.nu:.4f}.h5"

	for case_idx in range(cfg.n_cases):
		re_value = int(cfg.re_values[case_idx])
		fields_by_dim: Dict[int, Dict[str, np.ndarray]] = {}

		for dim in cfg.dims:
			u, v = solve_burgers_2d(dim, dim, cfg, cfg.ic_type, cfg.seed + case_idx)
			p = np.zeros_like(u, dtype=np.float32)
			fields_by_dim[dim] = {"u": u, "v": v, "p": p}

			if cfg.save_plots:
				tag = f"case{case_idx}_nu{cfg.nu:.4f}_{cfg.ic_type}"
				_plot_fields(out_dir, dim, fields_by_dim[dim], cfg, tag)

		save_case_to_h5(h5_path, case_idx, re_value, cfg, fields_by_dim)

	return h5_path


def main():
	cfg = Burgers2DConfig(
		x_min=-1.0,
		x_max=1.0,
		y_min=-1.0,
		y_max=1.0,
		t_end=0.4,
		nu=0.01 / np.pi,
		cfl=0.4,
		max_steps=20000,
		max_abs_u=10.0,
		dims=(10, 20, 40, 80, 200, 400),
		n_cases=1,
		ic_type="sine",
		seed=42,
		re_values=(100,),
		bc_type="Burgers2D(periodic)",
		output_dir="outputs",
		output_tag="burgers2d",
		save_plots=True,
	)

	h5_path = generate_burgers_2d_dataset(cfg)
	print(f"Saved H5: {h5_path}")


if __name__ == "__main__":
	main()
