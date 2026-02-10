#!/usr/bin/env python3
"""Compute and plot CL from LBM force CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute CL from force CSV and plot.")
    p.add_argument("--spec", type=Path, default=Path("specs/flapping_right_wing.yaml"))
    p.add_argument("--csv", type=Path, default=Path("outputs/lbm_static/forces.csv"))
    p.add_argument("--out", type=Path, default=Path("outputs/lbm_static/cl.png"))
    p.add_argument("--show", action="store_true", help="Show plot interactively.")
    p.add_argument("--use-pressure", action="store_true", help="Plot pressure-only CL if available.")
    return p.parse_args()


def _load_spec(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _get_dx(spec: dict) -> float:
    solver = spec.get("solver", {})
    lattice = solver.get("lattice", {})
    if "dx" in lattice:
        return float(lattice["dx"])
    units = spec.get("units", {})
    c = float(units.get("reference", {}).get("c", 1.0))
    n_c = int(lattice.get("N_c", 40))
    return c / n_c


def _read_forces(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return list(r)


def main() -> None:
    args = parse_args()
    spec = _load_spec(args.spec)
    rows = _read_forces(args.csv)

    if not rows:
        raise SystemExit(f"No rows found in {args.csv}")

    units = spec.get("units", {})
    ref = units.get("reference", {})
    c = float(ref.get("c", 1.0))
    b = float(spec["geometry"][0]["primitive"]["dimensions"]["b"])
    dx = _get_dx(spec)

    # Use lattice units for CL since forces are in lattice units.
    U_lat = float(spec["solver"]["lbm"]["U_lattice"])
    rho_lat = 1.0
    S_lat = (c / dx) * (b / dx)
    q = 0.5 * rho_lat * U_lat**2 * S_lat

    steps = np.array([int(r["step"]) for r in rows])
    fz_m = np.array([float(r["fz_m"]) if r["fz_m"] not in (None, "None", "") else np.nan for r in rows])
    cl_m = -fz_m / q

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, cl_m, label="CL (momentum)")

    if args.use_pressure and "fz_p" in rows[0]:
        fz_p = np.array([float(r["fz_p"]) if r["fz_p"] not in (None, "None", "") else np.nan for r in rows])
        cl_p = -fz_p / q
        plt.plot(steps, cl_p, label="CL (pressure)")

    plt.xlabel("Step")
    plt.ylabel("C_L")
    plt.title("Lift Coefficient (lattice units)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")

    print(f"Last CL (momentum): {cl_m[-1]:.6e}")
    if args.use_pressure and "fz_p" in rows[0]:
        print(f"Last CL (pressure): {cl_p[-1]:.6e}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
