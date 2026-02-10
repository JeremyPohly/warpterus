#!/usr/bin/env python3
"""Run a static right-wing LBM case using XLB (JAX backend)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import xlb
from xlb.compute_backend import ComputeBackend
from xlb.precision_policy import PrecisionPolicy
from xlb.velocity_set import D3Q19, D3Q27
from xlb.grid import grid_factory
from xlb.operator.stepper import IncompressibleNavierStokesStepper
from xlb.operator.boundary_condition import RegularizedBC, HalfwayBounceBackBC, ExtrapolationOutflowBC, EquilibriumBC
from xlb.operator.macroscopic import Macroscopic
from xlb.operator.force import MomentumTransfer
from xlb.helper.initializers import initialize_eq
from xlb.utils import save_fields_vtk

from warp_fwuav.geometry import transform_vertices
from warp_fwuav.lbm import build_domain, estimate_cells, get_wing_from_spec, get_wing_root, load_spec, wing_solid_indices, face_indices
from warp_fwuav.so3 import rot_y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Static wing LBM run with XLB (JAX).")
    p.add_argument("--spec", type=Path, default=Path("specs/flapping_right_wing.yaml"))
    p.add_argument("--out", type=Path, default=Path("outputs/lbm_static"))
    p.add_argument("--n-c", type=int, default=None, help="Override chord resolution.")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--output-interval", type=int, default=50)
    p.add_argument("--pitch-deg", type=float, default=35.0)
    p.add_argument("--flap-deg", type=float, default=0.0)
    p.add_argument("--psi-deg", type=float, default=0.0)
    p.add_argument("--force", action="store_true", help="Run even if grid is large.")
    p.add_argument("--export-wing", action="store_true", help="Export wing surface mesh VTK.")
    p.add_argument("--wing-n-chord", type=int, default=60)
    p.add_argument("--wing-n-span", type=int, default=160)
    p.add_argument("--wing-solid", action="store_true", help="Export a thin solid wing mesh.")
    p.add_argument("--vorticity", action="store_true", help="Compute and output vorticity.")
    p.add_argument("--q-criterion", action="store_true", help="Compute and output Q-criterion.")
    p.add_argument("--deriv-spacing", type=float, default=1.0, help="Spacing for derivatives (lattice units).")
    p.add_argument("--pressure-force", action="store_true", help="Integrate pressure force on wing.")
    p.add_argument("--momentum-force", action="store_true", help="Compute momentum-exchange force on wing.")
    p.add_argument("--force-csv", type=Path, default=None, help="Optional CSV to store forces.")
    p.add_argument("--no-vtk", action="store_true", help="Skip VTK output (forces still computed).")
    return p.parse_args()


def _as_indices_list(arr):
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


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


def _compute_vorticity_and_q(u: np.ndarray, spacing: float) -> dict:
    # u shape: (3, nx, ny, nz)
    u = np.asarray(u)
    du_dx, du_dy, du_dz = np.gradient(u[0], spacing, spacing, spacing, edge_order=1)
    dv_dx, dv_dy, dv_dz = np.gradient(u[1], spacing, spacing, spacing, edge_order=1)
    dw_dx, dw_dy, dw_dz = np.gradient(u[2], spacing, spacing, spacing, edge_order=1)

    omega_x = dw_dy - dv_dz
    omega_y = du_dz - dw_dx
    omega_z = dv_dx - du_dy
    omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

    # Strain-rate tensor components
    s_xx = du_dx
    s_yy = dv_dy
    s_zz = dw_dz
    s_xy = 0.5 * (du_dy + dv_dx)
    s_xz = 0.5 * (du_dz + dw_dx)
    s_yz = 0.5 * (dv_dz + dw_dy)

    # Rotation tensor components
    o_xy = 0.5 * (du_dy - dv_dx)
    o_xz = 0.5 * (du_dz - dw_dx)
    o_yz = 0.5 * (dv_dz - dw_dy)

    s_norm2 = s_xx**2 + s_yy**2 + s_zz**2 + 2.0 * (s_xy**2 + s_xz**2 + s_yz**2)
    o_norm2 = 2.0 * (o_xy**2 + o_xz**2 + o_yz**2)
    qcrit = 0.5 * (o_norm2 - s_norm2)

    return {
        "omega_x": omega_x,
        "omega_y": omega_y,
        "omega_z": omega_z,
        "omega_mag": omega_mag,
        "q_criterion": qcrit,
    }


def _pressure_force(rho, bc_mask, missing_mask, no_slip_id, velocity_set, dx):
    # rho shape: (1, nx, ny, nz)
    boundary = bc_mask[0] == no_slip_id
    boundary_f = boundary.astype(rho.dtype)

    main_idx = velocity_set.main_indices
    c_main = jnp.array(velocity_set.c[:, main_idx], dtype=rho.dtype)
    m = missing_mask[main_idx]  # (n_main, nx, ny, nz)
    normals = -jnp.tensordot(c_main, m, axes=(1, 0))  # (3, nx, ny, nz)
    normals = normals * boundary_f
    norm = jnp.linalg.norm(normals, axis=0) + 1.0e-12
    n_unit = normals / norm

    cs2 = velocity_set.cs2
    p = cs2 * (rho[0] - 1.0)
    p = p * boundary_f
    force = -p * n_unit * (dx * dx)
    return jnp.sum(force, axis=(1, 2, 3))


def main() -> None:
    args = parse_args()
    spec = load_spec(str(args.spec))

    print(f"JAX version: {jax.__version__}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"JAX devices: {[str(d) for d in jax.devices()]}")

    domain = build_domain(spec, n_c_override=args.n_c)
    n_cells = estimate_cells(domain)
    print(f"Grid shape: {domain.shape} (cells={n_cells:,}, dx={domain.dx:.4f})")
    if n_cells > 25_000_000 and not args.force:
        raise SystemExit(
            f"Grid has {n_cells:,} cells. Re-run with --force or reduce N_c/domain bounds."
        )

    solver = spec.get("solver", {})
    lbm = solver.get("lbm", {})
    lattice = solver.get("lattice", {})
    units = spec.get("units", {})
    U_ref = float(units.get("reference", {}).get("U_ref", 1.0))

    # Lattice parameters
    U_lattice = float(lbm.get("U_lattice", 0.05))
    dx = domain.dx
    dt = float(lattice.get("dt", dx * U_lattice / U_ref))
    nu = float(spec.get("fluid", {}).get("nu", 0.001))
    nu_lattice = nu * dt / (dx * dx)
    tau = float(lbm.get("tau", 0.5 + nu_lattice / (1.0 / 3.0)))
    if tau <= 0.5:
        raise SystemExit(f"tau={tau:.4f} <= 0.5 is unstable. Adjust nu/U_lattice/dx.")
    omega = 1.0 / tau

    # Configure XLB
    precision = PrecisionPolicy.FP32FP32
    backend = ComputeBackend.JAX
    model = str(lbm.get("model", "D3Q19")).upper()
    velocity_set = D3Q27(precision, backend) if model == "D3Q27" else D3Q19(precision, backend)
    xlb.init(velocity_set=velocity_set, default_backend=backend, default_precision_policy=precision)

    grid = grid_factory(domain.shape, compute_backend=backend)

    # Build wing and solid indices
    wing = get_wing_from_spec(spec)
    root_pos = get_wing_root(spec)
    pitch_rad = np.deg2rad(args.pitch_deg)
    _ = np.deg2rad(args.flap_deg)  # placeholder for future use
    _ = np.deg2rad(args.psi_deg)
    solid_idx = wing_solid_indices(wing, domain, pitch_rad, root_pos)

    if args.export_wing:
        if args.wing_solid:
            mesh = wing.solid_mesh(args.wing_n_chord, args.wing_n_span)
        else:
            mesh = wing.surface_mesh(args.wing_n_chord, args.wing_n_span)
        R = rot_y(pitch_rad)
        verts = transform_vertices(mesh.vertices, R, root_pos)
        _write_vtk_polydata(args.out / "wing_static.vtk", verts, mesh.faces)
        # Convert to lattice coordinates to match the VTK volume grid (origin=0, spacing=1)
        verts_lattice = (verts - domain.origin) / domain.dx
        _write_vtk_polydata(args.out / "wing_static_lattice.vtk", verts_lattice, mesh.faces)

    # Boundary conditions
    nx, ny, nz = domain.shape
    inlet_idx = face_indices(domain.shape, "x0")
    outlet_idx = face_indices(domain.shape, "x1")
    # Exclude inlet/outlet edges from farfield faces
    exclude_x = {0, nx - 1}
    far_y0 = face_indices(domain.shape, "y0", exclude_x=exclude_x)
    far_y1 = face_indices(domain.shape, "y1", exclude_x=exclude_x)
    far_z0 = face_indices(domain.shape, "z0", exclude_x=exclude_x)
    far_z1 = face_indices(domain.shape, "z1", exclude_x=exclude_x)
    far_idx = np.hstack([far_y0, far_y1, far_z0, far_z1])
    far_idx = np.unique(far_idx, axis=1)

    inlet_idx = _as_indices_list(inlet_idx)
    outlet_idx = _as_indices_list(outlet_idx)
    far_idx = _as_indices_list(far_idx)
    solid_idx = _as_indices_list(solid_idx)

    u_in = np.array([-U_lattice, 0.0, 0.0], dtype=np.float64)
    inlet_bc = RegularizedBC("velocity", prescribed_value=u_in, indices=inlet_idx)
    outlet_bc = ExtrapolationOutflowBC(indices=outlet_idx)
    far_bc = EquilibriumBC(rho=1.0, u=(u_in[0], 0.0, 0.0), indices=far_idx)
    wing_bc = HalfwayBounceBackBC(indices=solid_idx)

    boundary_conditions = [far_bc, inlet_bc, outlet_bc, wing_bc]

    stepper = IncompressibleNavierStokesStepper(
        omega=omega,
        grid=grid,
        boundary_conditions=boundary_conditions,
        collision_type=str(lbm.get("collision", "BGK")).upper(),
    )

    # Initialize fields with uniform inflow
    def initializer(_grid, _velocity_set, _precision, _backend):
        rho = _grid.create_field(cardinality=1, fill_value=1.0, dtype=_precision.compute_precision)
        u = _grid.create_field(cardinality=_velocity_set.d, fill_value=0.0, dtype=_precision.compute_precision)
        # Set u_x to inflow velocity
        u = u.at[0].set(u_in[0])
        return initialize_eq(None, _grid, _velocity_set, _precision, _backend, rho=rho, u=u)

    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields(initializer=initializer)

    macroscopic = Macroscopic()
    momentum_transfer = MomentumTransfer(no_slip_bc_instance=wing_bc)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.force_csv:
        args.force_csv.parent.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps + 1):
        if step % args.output_interval == 0:
            need_macro = (not args.no_vtk) or args.vorticity or args.q_criterion or args.pressure_force
            if need_macro:
                rho, u = macroscopic(f_0)

            if not args.no_vtk:
                # Convert to numpy for VTK output
                rho_np = np.array(rho[0])
                u_np = np.array(u)
                fields = {
                    "rho": rho_np,
                    "u_x": u_np[0],
                    "u_y": u_np[1],
                    "u_z": u_np[2],
                }
                if args.vorticity or args.q_criterion:
                    derivs = _compute_vorticity_and_q(u_np, args.deriv_spacing)
                    if args.vorticity:
                        fields["omega_x"] = derivs["omega_x"]
                        fields["omega_y"] = derivs["omega_y"]
                        fields["omega_z"] = derivs["omega_z"]
                        fields["omega_mag"] = derivs["omega_mag"]
                    if args.q_criterion:
                        fields["q_criterion"] = derivs["q_criterion"]
                save_fields_vtk(fields, step, output_dir=str(args.out), prefix="fields")
                print(f"[step {step}] wrote fields")

            if args.pressure_force or args.momentum_force:
                fx = fy = fz = None
                mfx = mfy = mfz = None
                if args.pressure_force:
                    fvec = _pressure_force(rho, bc_mask, missing_mask, wing_bc.id, velocity_set, domain.dx)
                    fx, fy, fz = [float(v) for v in np.array(fvec)]
                    print(f"[step {step}] pressure force (lattice): fx={fx:.6e}, fy={fy:.6e}, fz={fz:.6e}")
                if args.momentum_force:
                    fgrid = momentum_transfer(f_0, bc_mask, missing_mask)
                    fvec = jnp.sum(fgrid, axis=(1, 2, 3))
                    mfx, mfy, mfz = [float(v) for v in np.array(fvec)]
                    print(f"[step {step}] momentum force (lattice): fx={mfx:.6e}, fy={mfy:.6e}, fz={mfz:.6e}")

                if args.force_csv:
                    header = "step,fx_p,fy_p,fz_p,fx_m,fy_m,fz_m\n"
                    line = f"{step},{fx},{fy},{fz},{mfx},{mfy},{mfz}\n"
                    if not args.force_csv.exists():
                        args.force_csv.write_text(header)
                    with args.force_csv.open("a", encoding="utf-8") as f:
                        f.write(line)

        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, step)
        f_0, f_1 = f_1, f_0


if __name__ == "__main__":
    main()
