"""LBM setup helpers for XLB integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import yaml

from .geometry import RectangularWing
from .so3 import rot_y


@dataclass(frozen=True)
class DomainConfig:
    origin: np.ndarray  # physical origin (x_min, y_min, z_min)
    shape: tuple[int, int, int]  # (nx, ny, nz)
    dx: float  # lattice spacing


def load_spec(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_wing_from_spec(spec: dict) -> RectangularWing:
    geom = next(g for g in spec["geometry"] if g["name"] == "right_wing")
    prim = geom["primitive"]
    dims = prim["dimensions"]
    root_to_le = prim.get("root_to_le", 0.0)
    return RectangularWing(
        chord=float(dims["c"]),
        span=float(dims["b"]),
        thickness=float(dims.get("t", 0.0)),
        root_to_le=float(root_to_le),
    )


def get_wing_root(spec: dict) -> np.ndarray:
    geom = next(g for g in spec["geometry"] if g["name"] == "right_wing")
    frame = geom.get("frame", {})
    return np.array(frame.get("origin", [0.0, 0.0, 0.0]), dtype=float)


def build_domain(spec: dict, n_c_override: int | None = None) -> DomainConfig:
    solver = spec.get("solver", {})
    lattice = solver.get("lattice", {})
    n_c = int(n_c_override) if n_c_override is not None else int(lattice.get("N_c", 40))

    units = spec.get("units", {})
    c = float(units.get("reference", {}).get("c", 1.0))
    dx = c / n_c

    domain = solver.get("domain", {})
    bounds = domain.get("bounds", {})
    x_min, x_max = bounds.get("x", [-2.0, 6.0])
    y_min, y_max = bounds.get("y", [-1.0, 3.5])
    z_min, z_max = bounds.get("z", [-2.0, 2.0])

    nx = int(np.ceil((x_max - x_min) / dx))
    ny = int(np.ceil((y_max - y_min) / dx))
    nz = int(np.ceil((z_max - z_min) / dx))

    return DomainConfig(origin=np.array([x_min, y_min, z_min], dtype=float), shape=(nx, ny, nz), dx=dx)


def estimate_cells(domain: DomainConfig) -> int:
    nx, ny, nz = domain.shape
    return int(nx) * int(ny) * int(nz)


def wing_solid_indices(
    wing: RectangularWing,
    domain: DomainConfig,
    pitch_rad: float,
    root_pos: np.ndarray,
) -> np.ndarray:
    """Return solid cell indices (3, N) for a pitched rectangular wing."""
    # Ensure thickness is at least one cell to avoid empty solids.
    thickness = max(wing.thickness, domain.dx)
    wing = RectangularWing(chord=wing.chord, span=wing.span, thickness=thickness, root_to_le=wing.root_to_le)

    x_te, x_le = wing.chord_limits()
    y_min, y_max = 0.0, wing.span
    z_min, z_max = -0.5 * wing.thickness, 0.5 * wing.thickness

    # Rotation: pitch about +y
    R = rot_y(pitch_rad)
    R_T = R.T

    # Compute AABB of rotated wing for loop bounds
    corners = np.array(
        [
            [x_te, y_min, z_min],
            [x_te, y_min, z_max],
            [x_te, y_max, z_min],
            [x_te, y_max, z_max],
            [x_le, y_min, z_min],
            [x_le, y_min, z_max],
            [x_le, y_max, z_min],
            [x_le, y_max, z_max],
        ],
        dtype=float,
    )
    corners_world = (R @ corners.T).T + root_pos
    bb_min = corners_world.min(axis=0)
    bb_max = corners_world.max(axis=0)

    # Convert physical bounds to index bounds
    i_min = np.floor((bb_min - domain.origin) / domain.dx - 0.5).astype(int)
    i_max = np.ceil((bb_max - domain.origin) / domain.dx - 0.5).astype(int)

    nx, ny, nz = domain.shape
    i_min = np.maximum(i_min, 0)
    i_max = np.minimum(i_max, np.array([nx - 1, ny - 1, nz - 1]))

    indices = []
    for i in range(i_min[0], i_max[0] + 1):
        for j in range(i_min[1], i_max[1] + 1):
            for k in range(i_min[2], i_max[2] + 1):
                pos = domain.origin + (np.array([i, j, k], dtype=float) + 0.5) * domain.dx
                local = R_T @ (pos - root_pos)
                if (
                    (x_te <= local[0] <= x_le)
                    and (y_min <= local[1] <= y_max)
                    and (z_min <= local[2] <= z_max)
                ):
                    indices.append([i, j, k])

    if not indices:
        raise ValueError("No solid cells were generated for the wing. Increase thickness or resolution.")

    return np.array(indices, dtype=int).T


def face_indices(
    shape: tuple[int, int, int],
    face: str,
    exclude_x: Iterable[int] | None = None,
    exclude_y: Iterable[int] | None = None,
    exclude_z: Iterable[int] | None = None,
) -> np.ndarray:
    nx, ny, nz = shape
    exclude_x = set(exclude_x or [])
    exclude_y = set(exclude_y or [])
    exclude_z = set(exclude_z or [])

    if face == "x0":
        i_vals = [0]
        j_vals = [j for j in range(ny) if j not in exclude_y]
        k_vals = [k for k in range(nz) if k not in exclude_z]
    elif face == "x1":
        i_vals = [nx - 1]
        j_vals = [j for j in range(ny) if j not in exclude_y]
        k_vals = [k for k in range(nz) if k not in exclude_z]
    elif face == "y0":
        i_vals = [i for i in range(nx) if i not in exclude_x]
        j_vals = [0]
        k_vals = [k for k in range(nz) if k not in exclude_z]
    elif face == "y1":
        i_vals = [i for i in range(nx) if i not in exclude_x]
        j_vals = [ny - 1]
        k_vals = [k for k in range(nz) if k not in exclude_z]
    elif face == "z0":
        i_vals = [i for i in range(nx) if i not in exclude_x]
        j_vals = [j for j in range(ny) if j not in exclude_y]
        k_vals = [0]
    elif face == "z1":
        i_vals = [i for i in range(nx) if i not in exclude_x]
        j_vals = [j for j in range(ny) if j not in exclude_y]
        k_vals = [nz - 1]
    else:
        raise ValueError(f"Unknown face: {face}")

    ii, jj, kk = np.meshgrid(i_vals, j_vals, k_vals, indexing="ij")
    return np.vstack([ii.ravel(), jj.ravel(), kk.ravel()])
