#!/usr/bin/env python3
"""Preview wing geometry + kinematics by exporting a VTK frame sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from warp_fwuav.geometry import RectangularWing, transform_vertices
from warp_fwuav.kinematics import KinematicsParams, right_wing_rotation


def _write_vtk_polydata(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_pts = vertices.shape[0]
    n_faces = faces.shape[0]
    total_indices = n_faces * 4
    with path.open("w", encoding="utf-8") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("wing\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n_pts} float\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        f.write(f"POLYGONS {n_faces} {total_indices}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def _load_spec(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_wing(spec: dict) -> RectangularWing:
    geom = next(g for g in spec["geometry"] if g["name"] == "right_wing")
    if geom["type"] != "primitive":
        raise ValueError("Only primitive geometry supported in preview script.")
    prim = geom["primitive"]
    if prim["shape"] != "rectangle":
        raise ValueError("Only rectangle supported in preview script.")
    dims = prim["dimensions"]
    root_to_le = prim.get("root_to_le", 0.0)
    return RectangularWing(
        chord=float(dims["c"]),
        span=float(dims["b"]),
        thickness=float(dims.get("t", 0.0)),
        root_to_le=float(root_to_le),
    )


def _build_kinematics(spec: dict) -> KinematicsParams:
    kin = spec.get("kinematics", {})
    qs = kin.get("qs")
    if qs is None:
        raise ValueError("spec.kinematics.qs block is required for QS_model kinematics.")
    return KinematicsParams(
        frequency=float(qs["frequency"]),
        phi_m=float(qs["phi_m"]),
        phi_0=float(qs["phi_0"]),
        phi_K=float(qs["phi_K"]),
        theta_m=float(qs["theta_m"]),
        theta_0=float(qs["theta_0"]),
        theta_C=float(qs["theta_C"]),
        theta_a=float(qs["theta_a"]),
        psi_m=float(qs["psi_m"]),
        psi_0=float(qs["psi_0"]),
        psi_N=float(qs["psi_N"]),
        psi_a=float(qs["psi_a"]),
        beta=float(qs.get("beta", 0.0)),
    )


def _origin_from_spec(spec: dict) -> np.ndarray:
    geom = next(g for g in spec["geometry"] if g["name"] == "right_wing")
    frame = geom.get("frame", {})
    return np.array(frame.get("origin", [0.0, 0.0, 0.0]), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export wing motion as VTK frames.")
    parser.add_argument("--spec", type=Path, default=Path("specs/flapping_right_wing.yaml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/preview"))
    parser.add_argument("--n-chord", type=int, default=40)
    parser.add_argument("--n-span", type=int, default=120)
    parser.add_argument("--frames-per-period", type=int, default=60)
    parser.add_argument("--periods", type=int, default=None)
    parser.add_argument("--solid", action="store_true", help="Export thin solid mesh.")
    args = parser.parse_args()

    spec = _load_spec(args.spec)
    wing = _build_wing(spec)
    kin = _build_kinematics(spec)
    origin = _origin_from_spec(spec)

    sim = spec.get("simulation", {})
    periods = args.periods if args.periods is not None else int(sim.get("periods", 1))
    frames = periods * args.frames_per_period

    if args.solid:
        mesh = wing.solid_mesh(args.n_chord, args.n_span)
    else:
        mesh = wing.surface_mesh(args.n_chord, args.n_span)

    period = 1.0 / kin.frequency
    dt = period / args.frames_per_period

    for i in range(frames):
        t = i * dt
        R = right_wing_rotation(t, kin)
        verts = transform_vertices(mesh.vertices, R, origin)
        _write_vtk_polydata(args.out / f"wing_{i:04d}.vtk", verts, mesh.faces)

    print(f"Wrote {frames} frames to {args.out}")


if __name__ == "__main__":
    main()
