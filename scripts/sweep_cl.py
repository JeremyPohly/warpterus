#!/usr/bin/env python3
"""Grid refinement sweep for CL using run_lbm_static.py."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from warp_fwuav.lbm import build_domain, estimate_cells


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep grid refinement and compute CL.")
    p.add_argument("--spec", type=Path, default=Path("specs/flapping_right_wing.yaml"))
    p.add_argument("--n-cs", type=str, default="20,25,30", help="Comma-separated chord resolutions.")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--output-interval", type=int, default=50)
    p.add_argument("--avg-last", type=int, default=3, help="Average CL over last N outputs.")
    p.add_argument("--out", type=Path, default=Path("outputs/sweep_cl"))
    p.add_argument("--pressure", action="store_true", help="Also compute pressure-only CL.")
    p.add_argument("--force", action="store_true", help="Pass --force to run_lbm_static.")
    return p.parse_args()


def compute_cl(csv_path: Path, U: float, c: float, b: float, dx: float, avg_last: int) -> dict:
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        raise ValueError(f"No force rows found in {csv_path}")

    tail = rows[-avg_last:]
    rho_lat = 1.0
    S_lat = (c / dx) * (b / dx)
    q = 0.5 * rho_lat * U**2 * S_lat

    def _cl_from(row, key):
        fz = float(row[key])
        return (-fz) / q

    out = {}
    out["cl_m_last"] = _cl_from(rows[-1], "fz_m")
    out["cl_m_mean"] = float(np.mean([_cl_from(r, "fz_m") for r in tail]))

    if "fz_p" in rows[-1] and rows[-1]["fz_p"] not in (None, "None"):
        out["cl_p_last"] = _cl_from(rows[-1], "fz_p")
        out["cl_p_mean"] = float(np.mean([_cl_from(r, "fz_p") for r in tail]))

    return out


def main() -> None:
    args = parse_args()
    spec = yaml.safe_load(args.spec.read_text())

    units = spec.get("units", {})
    ref = units.get("reference", {})
    c = float(ref.get("c", 1.0))
    U = float(ref.get("U_ref", 1.0))
    b = float(spec["geometry"][0]["primitive"]["dimensions"]["b"])
    U_lat = float(spec["solver"]["lbm"]["U_lattice"])

    # Use lattice U for CL in lattice units
    U = U_lat

    n_cs = [int(x.strip()) for x in args.n_cs.split(",") if x.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    summary_path = args.out / "sweep_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "n_c",
                "dx",
                "cells",
                "cl_m_last",
                "cl_m_mean",
                "cl_p_last",
                "cl_p_mean",
            ]
        )

    for n_c in n_cs:
        run_out = args.out / f"nc_{n_c}"
        run_out.mkdir(parents=True, exist_ok=True)
        force_csv = run_out / "forces.csv"

        cmd = [
            sys.executable,
            "scripts/run_lbm_static.py",
            "--spec",
            str(args.spec),
            "--n-c",
            str(n_c),
            "--steps",
            str(args.steps),
            "--output-interval",
            str(args.output_interval),
            "--out",
            str(run_out),
            "--momentum-force",
            "--force-csv",
            str(force_csv),
            "--no-vtk",
        ]
        if args.pressure:
            cmd.append("--pressure-force")
        if args.force:
            cmd.append("--force")

        print(f"Running n_c={n_c} ...")
        subprocess.run(cmd, check=True)

        domain = build_domain(spec, n_c_override=n_c)
        dx = domain.dx
        cells = estimate_cells(domain)
        cl = compute_cl(force_csv, U, c, b, dx, args.avg_last)

        row = [
            n_c,
            f"{dx:.6f}",
            cells,
            f"{cl.get('cl_m_last', float('nan')):.6e}",
            f"{cl.get('cl_m_mean', float('nan')):.6e}",
            f"{cl.get('cl_p_last', float('nan')):.6e}",
            f"{cl.get('cl_p_mean', float('nan')):.6e}",
        ]
        with summary_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"n_c={n_c} -> {row}")

    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
