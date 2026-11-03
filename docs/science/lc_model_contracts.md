# Liquid-Crystal Model Contracts

This document is the canonical Product-level scientific contract for LCProp's
liquid-crystal (LC) workflows. It records the current equations,
normalization, boundary conditions, coupling cadence, supported numerical
choices, and scope limitations without reproducing historical development or
validation narratives.

## Coordinates, state, and units

The physical runtime grid uses micrometres. The LC director solvers use
dimensionless transverse coordinates

```text
u = 2 x / d
v = 2 y / d
```

where `d = x_aperture_um` is the LC thickness. The established discrete
spacings are `du = 2/(Nx - 1)` and `dv = du*(dy_um/dx_um)`. They are solver
spacings and are not inferred from the cell-centred physical coordinate
arrays.

The material state is the director angle `theta` in radians. Static
self-consistent results retain one accepted director plane per longitudinal
slice. TD results retain a director stack `theta(z, x, y)` at each accepted
material time. Retained LC volume coordinates identify slice midpoints,
`z_k = (k + 1/2) dz`.

## Static director equation

At one longitudinal slice, the normalized fully nonlinear static equation is

```text
0 = laplacian_uv(theta) + (b + bi I) sin(2 theta).
```

The x boundary rows have the prescribed Dirichlet value `theta_bc`; y is
periodic. The director remains within the declared `theta_min`/`theta_max`
clamp. The dark-bias initial condition is the exact one-dimensional
Jacobi-elliptic equilibrium, tiled in y.

The electrical coefficient is dimensionless:

```text
b = delta_epsilon epsilon_0 V_bias^2 / (8 K),
```

unless an explicit internal `b_override` is used. `K` is in N, voltage is in
V, and `delta_epsilon` is dimensionless.

The optical coefficient has units of square micrometres:

```text
bi = (ne^2 - no^2) d^2 P / (8 c K),
```

with SI conversion applied in the implementation. `P` is the total enabled
incident power. The launched channel fields are normalized so the director
intensity `I` is in `1/um^2`; consequently `bi I` is dimensionless.

## Time-dependent director equation

LC material time is nondimensional. With the production mobility equal to
one, the implemented continuous equation corresponding to the CN/Picard
update is

```text
partial_t theta = laplacian_uv(theta)
                + (b + bi I) sin(2 theta)
                + gamma_z partial_zz theta.
```

The optional longitudinal term uses the centred neighbor expression
`gamma_z*(theta_prev - 2 theta + theta_next)/dz^2`; endpoint slices repeat
their missing neighbor. `gamma_z=0` is the GUI/default contract. `dt` and
`Nt` are explicit fixed material-time controls. The solver does not infer a
physical time unit or stop automatically at steady state.

Each accepted material interval uses one complete optical pass from the
entrance field through the frozen prior director stack. Every slice is then
updated from the corresponding midpoint optical intensity. Cancellation is
accepted only between complete material intervals, so a checkpoint always
contains an accepted state.

## Optical response and propagation

For extraordinary-polarized scalar propagation, the effective index is

```text
                  ne no
n_eff(theta) = ------------------------------.
               sqrt((ne cos theta)^2 +
                    (no sin theta)^2)
```

Over distance `dz`, the LC phase response is

```text
exp(i (2 pi / wavelength) dz (n_eff - n_ref)),
```

with `n_ref = no`. The shared angular-spectrum split-step propagator applies
this response with its established Strang ordering. Optical substeps refine a
fixed physical propagation increment; they do not change material-z sampling
or the optical/material refresh cadence.

Current LC propagation uses one diffraction and material-response wavelength.
All enabled channels must therefore declare exactly the same wavelength.
Per-channel-wavelength LC propagation is not implemented.

## Coherence and director-driving intensity

Fields sharing an explicit coherence group are summed before intensity is
formed. Intensities from different groups are then added. The same grouped
intensity convention drives the LC director and is used for power and result
products. Legacy request-level coherence is migrated to explicit groups.

## Transverse optical boundaries

Propagation workflows support the shared material-neutral boundary policies:

- `periodic`: exact no-op on the FFT cell;
- `sponge`: a smooth field-amplitude absorption rate
  `exp(-gamma(x,y)*abs(delta_z))`, accumulated with physical distance and
  invariant to optical-substep subdivision;
- `tukey`: a discrete separable square-root Tukey window, applied as an
  apodization rather than interpreted as an absorption rate.

These boundaries reduce optical wraparound. They do not replace aperture,
grid, `dz`, or optical-substep convergence studies. Director x boundaries
remain Dirichlet and director y boundaries remain periodic regardless of the
optical boundary selection.

Stationary LC soliton and existence-curve workflows have a periodic-only
transverse optical contract. Headless nonperiodic stationary requests are
rejected, and the GUI fixes and labels the stationary selection as Periodic.
No nonperiodic stationary eigenproblem is implied.

## Static workflows and convergence

`fixed_theta` propagates through a prepared dark/bias director and performs no
self-consistent material solve.

`local_self_consistent` marches in z. At each slice it seeds from the last
accepted director plane, alternates CN/Picard director relaxation with a
refreshed optical midpoint calculation, and carries only the accepted optical
field into the next slice. The selected optical boundary is applied to every
actual midpoint propagation.

Scientific convergence requires both configured static residual RMS and
maximum-residual qualifications. Optional director-update thresholds do not
replace residual acceptance. Execution may finish while one or more slices
remain nonconverged; the result retains `all_slices_converged=False`,
per-slice termination reasons, and residual diagnostics, and the GUI reports
the run as scientifically nonconverged rather than as a successful solve.

## Stationary solitons and existence curves

The stationary solver alternates the existing director relaxation and
periodic optical fixed-point map, with optional transverse eigenpair
polishing. It reports propagation constant/eigenvalue, residuals, director
maximum, optical profile, and convergence. Existence curves sweep physical
power and may reuse the preceding accepted member as a continuation seed.

These workflows do not provide a spectral stability calculation. TD
propagation from a retained stationary result is a separate finite-time
evolution, not an eigenvalue-stability proof.

## Backend and precision

LC production workflows currently execute with NumPy. They do not dispatch
scientific work through CuPy or an automatic backend. GPU Slurm profiles are
therefore not applicable to LC production requests.

Headless local static and TD workflows support NumPy `float64/complex128` and
`float32/complex64`; focused Product regressions compare both paths. The LC
GUI represents the conservative `float64` choice. Users remain responsible
for problem-specific grid, timestep, and precision convergence before
publication use.

## Persistence, continuation, and remote scope

Experiment files support static and TD requests. They preserve beam/coherence
intent, boundary selection, precision, solver controls, and compatible
LaunchPane presentation state. Soliton and existence requests are not
experiment-file workflows.

Static and TD checkpoints contain accepted continuation state. Compatibility
validation rejects changes to scientific request fields, including the
optical boundary; TD may change only the number of additional requested
steps. Experiment files describe a request and are not continuation
checkpoints.

All displayed workflows execute locally. Slurm transport is currently limited
to canonical static propagation and remains NumPy-based. LC has no Fast/Full
retention selector, CuPy production path, TD/soliton Slurm path, input-screen
launch-element request, runtime estimator, or spectral stability solver.

## Implementation sources

The executable contracts are owned by:

- `src/lcprop/lc/requests.py`, `normalization.py`, `bias.py`, and
  `coupling.py`;
- `src/lcprop/lc/theta_cn.py`, `theta_picard.py`, and
  `theta_cn_zcoupled.py`;
- `src/lcprop/lc/optical_response.py`, `propagation.py`, and `source.py`;
- `src/lcprop/lc/workflows/`;
- the LC-owned persistence, transport, products, and GUI modules.

Historical derivations, validation campaigns, performance studies, and
rejected approaches are Research records, not dependencies of this Product
contract.
