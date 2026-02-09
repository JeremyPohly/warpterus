**Phase 0 — Requirements + Scoping**

**Goal**
Define the initial 3D prescribed-motion flapping-wing case (rectangular planform, AR=2.5) and the lattice/flow scaling for XLB.

**Primary Decisions**
1. Solver core: XLB with NVIDIA Warp backend or JAX backend. Start with JAX (GPU already working) and keep Warp as a switchable backend.
2. Flow regime: Re = 1000 based on chord `c` and reference velocity `U_ref`.
3. Geometry: rectangular wing, AR = 2.5, thin plate.
4. Motion: prescribed kinematics (FSI later).
5. Units: dimensionless primary spec; map to lattice units for LBM.

**Geometry (Dimensionless)**
1. Chord `c = 1`.
2. Span `b = AR * c = 2.5`.
3. Thickness `t = ?` (thin plate; initial suggestion `t = 0.02c` unless you prefer another value).
4. Leading edge at `x = 0`, trailing edge at `x = 1`, mid-span at `y = 0`.

**Flow (Dimensionless)**
1. Reference velocity `U_ref = 1`.
2. Kinematic viscosity `nu = U_ref * c / Re = 0.001`.
3. Reference time `t_ref = c / U_ref = 1`.

**Kinematics (Prescribed)**
1. Choose a kinematic model: heave + pitch about 1/4 chord.
2. Alternate kinematic model: pure pitch.
3. Alternate kinematic model: pure heave.
4. Specify amplitude, frequency, and phase. I can propose defaults once you pick a model.

**Domain (Initial Guidance)**
1. Upstream: 5c
2. Downstream: 15c
3. Lateral/spanwise: 6c total
4. Vertical: 6c total
These are placeholders and can be tightened after we see wake extent.

**LBM Lattice Mapping (Example Draft)**
This is just to ground the parameter ranges; we will finalize in Phase 1.
1. Choose chord resolution `N_c = 100` (cells across chord)
2. `dx = c / N_c = 0.01`
3. Choose lattice velocity `U_lattice = 0.05` (keeps Mach number small)
4. `dt = dx * U_lattice / U_ref = 0.0005`
5. `nu_lattice = nu * dt / dx^2 = 0.005`
6. LBM relaxation time `tau = 0.5 + nu_lattice / cs^2 = 0.515` (with `cs^2 = 1/3`)

**Boundary Conditions (Initial)**
1. Inlet: uniform velocity `U_ref`
2. Outlet: fixed pressure
3. Far-field: slip or pressure (depending on XLB support)
4. Wing surface: bounce-back or interpolated bounce-back

**Data + Validation**
1. FVM reference format: `.p3d` (we will add a conversion step; let me know preferred target, e.g., VTK or CSV).
2. Metrics: lift, drag, moment time histories; phase-averaged force coefficients.

**Open Questions**
1. Thickness `t/c` for the thin plate.
2. Kinematics model and parameters (amplitude, frequency, phase).
3. Domain extents you prefer for the baseline.
4. Preferred output format for comparison (VTK, CSV, HDF5).

**Next**
Once you answer the open questions, I’ll scaffold Phase 1 (XLB + JAX setup, geometry mask, and a stationary wing baseline).
