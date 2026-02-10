**Case Spec Schema (Draft)**

This file describes the draft schema for a flapping-wing case specification.

**Top-Level**
1. `meta`: identifiers and versioning
2. `units`: unit system and normalization
3. `geometry`: list of bodies/parts
4. `kinematics`: motion definitions
5. `fluid`: material properties
6. `solver`: LBM/XLB backend and lattice parameters
7. `simulation`: time stepping and outputs

**Fields**

`meta`
1. `case_id` (string)
2. `schema_version` (string)

`units`
1. `system` (string): `dimensionless` or `SI`
2. `reference`
   - `c` (float): chord
   - `U_ref` (float): reference velocity
   - `rho_ref` (float): reference density (optional)
3. `time_normalization`
   - `type` (string): `period` or `c_over_U`
   - `T_ref` (float): reference period if `period`

`geometry` (list)
1. `name` (string): `right_wing`, `left_wing`, `body`, etc.
2. `type` (string): `primitive` or `mesh`
3. `primitive`
   - `shape` (string): `rectangle`, `square`, `triangle`, `polygon`
   - `dimensions`
     - `c` (float): chord
     - `b` (float): span
     - `t` (float): thickness
4. `mesh`
   - `path` (string): file path
   - `format` (string): `stl`, `obj`, etc.
   - `units` (string): `dimensionless` or `SI`
5. `frame`
   - `origin` (list[3]): body frame origin in global coordinates
   - `axes` (list[3][3]): 3x3 basis vectors

`kinematics`
1. `definitions` (list)
   - `name` (string)
   - `type` (string): `cosine`, `sine`, `fourier`, `table`
   - `variable` (string): `pitch`, `flap`, `heave`
   - `axis` (list[3]): axis of rotation or translation direction
   - `about` (list[3]): point of rotation (global or body frame)
   - `amplitude` (float): radians or length
   - `frequency` (float): cycles per unit time
   - `phase` (float): radians
   - `coeffs` (list): Fourier coefficients if `fourier`
   - `table` (string): path to time series if `table`
2. `bindings` (list)
   - `geometry` (string): name of geometry
   - `motions` (list): list of kinematic definition names
3. `qs` (object): QS_model (doc/QS_model.pdf) parameters
   - `frequency` (float): flapping frequency (Hz)
   - `phi_m` (float): flapping amplitude (rad)
   - `phi_0` (float): flapping offset (rad)
   - `phi_K` (float): flapping waveform shape (0 < K <= 1)
   - `theta_m` (float): pitch amplitude (rad)
   - `theta_0` (float): pitch offset (rad)
   - `theta_C` (float): pitch waveform sharpness
   - `theta_a` (float): pitch phase offset (rad)
   - `psi_m` (float): deviation amplitude (rad)
   - `psi_0` (float): deviation offset (rad)
   - `psi_N` (float): deviation cycles per flap period (1 or 2)
   - `psi_a` (float): deviation phase offset (rad)
   - `beta` (float): stroke plane angle (rad)

`fluid`
1. `rho` (float)
2. `nu` (float)
3. `Re` (float): optional consistency check

`solver`
1. `backend` (string): `jax` or `warp`
2. `lbm`
   - `model` (string): `D3Q19` or `D3Q27`
   - `collision` (string): `BGK` or `KBC`
   - `U_lattice` (float)
   - `tau` (float): optional; can be derived from `nu_lattice`
3. `lattice`
   - `N_c` (int): chord resolution
   - `dx` (float): optional; derived from `c/N_c`
   - `dt` (float): optional
4. `domain`
   - `bounds`
     - `x` (list[2]): [x_min, x_max]
     - `y` (list[2]): [y_min, y_max]
     - `z` (list[2]): [z_min, z_max]

`simulation`
1. `periods` (int)
2. `steps_per_period` (int)
3. `output`
   - `format` (string): `vtk`, `csv`, `hdf5`
   - `interval_steps` (int)
   - `path` (string)
