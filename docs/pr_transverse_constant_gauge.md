# Constant-gauge transverse equilibrium discretization

`constant_gauge_v1` is an opt-in discrete potential space for the nonlinear
zero-flux material solver and causal transverse marcher/streaming adapter.
The historical default remains `derivative_resolved_v1`. It is not a new
continuum material model or a change to the three Figure 11 reference models.

```python
from dataclasses import replace
from lcprop.pr.transverse.static import PRTransverseStaticMaterialSolverOptions

material = PRTransverseStaticMaterialSolverOptions(potential_space="constant_gauge_v1")
# For an existing causal request, preserve its tolerances:
material = replace(request.solver.material_solver, potential_space="constant_gauge_v1")
request = replace(request, solver=replace(request.solver, material_solver=material))
```

The public frozen-intensity solver accepts this option directly. Pass the same
`potential_space` to `state_from_potential` or `static_equilibrium_residual`
when reconstructing its result manually. The streaming adapter does this
consistently. The volume optical workflow and experimental TD-discrete
corrector reject this option; they have not been integrated or qualified.

## Discrete equations

Retain the existing real periodic Fourier derivative symbols, including zero
Nyquist first derivatives. Set L=-Dx²-h_y Dy², G=I-11ᵀ/N,
p=exp(-Gψ)/I divided by its mean. The residual is

    R = 1 + L Gψ - p
    J v = G [L(Gv) + p(Gv) - p mean(p Gv)]

On the mean-zero space, J=L+diag(p)-ppᵀ/N is positive definite for positive p.
Only the constant is gauge. Even-axis checkerboard potential modes are retained
as nonlinear unknowns; their nonlinear carrier response lifts the derivative
nullspace. Their derivative contribution to E=-Dψ and P=1+Lψ remains zero.
The same derivative symbols thus define electric field, elliptic charge,
curl, and Gauss-law diagnostics. No inconsistent Nyquist Laplacian is added.

Newton directions, the elliptic-plus-identity preconditioner, trial potentials,
state reconstruction and causal evaluations all use G. Line search and
convergence use the **full residual**. Existing positivity checks, tolerances,
damping, PCG forcing and rollback remain unchanged. The historical derivative
projection is retained for reporting resolved/null residual components, not
for excluding checkerboard unknowns in the new solve.

This is a deliberate change in finite-grid numerical semantics. Checkerboard
potentials are not resolved physical electric-field oscillations. Grid-refinement
qualification remains necessary before physical interpretation. Conservative
flux diagnostics and the equilibrium residual are not interchangeable under
spectral product-rule aliasing; the existing qualification policy is unchanged.

## Reproducibility and limits

Selected mode appears in causal solver provenance and its restart fingerprint.
Cross-mode restart is rejected. Default fingerprints preserve historical
serialization; other options metadata now exposes the default mode explicitly.
CPU odd/even manufactured cases, retained synthetic failure data, replay,
rollback and independent Wolfram references pass locally. Existing one-dimensional
reference tests still pass; finite-grid spectral equations are not asserted to
be identical to the separate one-dimensional finite-difference solver.

Retained synthetic H200 qualification (job 6585850) covers the reviewed source
on bounded frozen-intensity and coupled two-cell cases, replay and rollback.
This integration is being checked separately on CPU and as an installed package;
it does not extend that GPU evidence to arbitrary grids or physical Figure 11
calculations. See `development/constant_gauge_integration.json` for exact source
identities and qualification scope. No new physical-domain run is implied.
