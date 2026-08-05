# Minimal photorefractive vertical slice

## State, normalization, and boundaries

The material state is the normalized space-charge field `E` with shape
`(Nz, Nx, Ny)`. The hopping model is

```text
E_t = E_app I_b - (E I - I_x)(1 + E_x) + I E_xx
```

where `x_normalized = k0*x`, `I = I_optical/I_peak_reference + I_b`, and
`I_b` is dark intensity plus the optional uniform background. The positive
`I E_xx` term is the diffusive sign in the paper and trusted implementation.

The material derivatives are periodic in x and act on array axis `-2`.
There are no y derivatives in this scalar hopping equation, so each y column
has independent material evolution; the optical FFT propagation remains
periodic in both x and y. Numerical periodicity is not a claim that a finite
crystal is physically periodic. A finite-beam calculation should use an
aperture large enough that optical intensity and material perturbations are
negligible at the transverse edges. PRProp3D additionally apodized the optical
field to suppress FFT wraparound. This minimal slice deliberately does not.

Centered differences were selected for the first slice because they are
local, inexpensive on NumPy and CuPy, converge at second order, and leave a
simple path to implicit treatment of the stiff diffusion term. Their Fourier
symbols are

```text
k1_modified = sin(k*dx)/dx
k2_modified_squared = 4*sin(k*dx/2)^2/dx^2
```

Thus they attenuate resolved high-frequency derivatives relative to spectral
derivatives, especially the first derivative near Nyquist. Frequency-sweep
tests characterize that tradeoff explicitly. A PR-owned spectral reference
path can be added later if benchmark work needs it; LC optics does not need to
change.

## Integrator timestep guards

Paper Equation (15) chooses one quarter of the shortest linearized time
constant:

```text
dt_paper = 1 / (4*(1 + k_max^2))
```

The trusted PRProp3D source actually computes `1/(4*k_max^2)`, omitting the
reaction term. Both use a spectral wavenumber and effectively take the
normalized uniform optical intensity as one.

For the implemented centered differences, linearization about uniform `I0`
and `Ebar` gives

```text
mu(k) = -I0*(1 + k2_modified_squared)
        - 1j*I0*Ebar*k1_modified
```

and forward Euler is absolutely stable when

```text
dt <= 2*(-Re(mu))/abs(mu)^2.
```

The workflow guard evaluates this boundary over every representable mode with
`I0 = 1 + I_b` and `Ebar = E_app*I_b/I0`, then divides it by eight. For zero
drift this is `1/[4*I0*(1 + k2_modified_squared)]`: the paper's quarter-time-
constant margin, adapted to the implemented operator and total plane-wave
intensity. It is a conservative uniform-state linearization, not a proof of
stability for arbitrary nonlinear, spatially concentrated intensity.

The selected semi-implicit method assigns `I*E_xx` to the implicit operator
and retains reaction, drift, source, and nonlinear terms explicitly. Its
uniform-state mode split is

```text
lambda_L = -I0*k2_modified_squared
lambda_N = -I0 - 1j*I0*Ebar*k1_modified
```

The semi-implicit guard finds the first absolute-stability boundary of the
implemented predictor/corrector amplification factor for every representable
mode, then applies the same factor-of-eight conservative margin. Grid-scale
diffusion no longer controls this limit, but explicit reaction and drift still
do. Both guards remain local uniform-state estimates rather than nonlinear
stability proofs.

## Optical and material stepping

The improved optical coupling remains a frozen-E Strang pass:

```text
half PR response -> linear hop -> half PR response
```

for every optical substep through `advance_prepared_response()`. After a full
optical pass, all z slices of E receive one synchronous material-time update.
This intentionally differs from PRProp3D's interleaved full-hop/full-response
Lie ordering.

Two named material integrators are available:

- `euler` is the original first-order reference and preserves existing
  programmatic request semantics.
- `semi_implicit_trapezoidal` is the selected production method. It uses an
  IMEX-Euler predictor and a linearly implicit trapezoidal corrector, treats
  variable-coefficient `I*E_xx` implicitly, and evaluates the complete optical
  mapping at both accepted and predicted states.

The method is one-step and requires no multistep history. It is second order
for prescribed intensity, a state-dependent intensity proxy, and the actual
frozen-E optical mapping. The standalone PR GUI defaults explicitly to the
semi-implicit method; the request dataclass retains Euler as its compatibility
default. CNAB2 and adaptive control remain possible future performance work.
The full derivation and validation evidence are recorded in
`docs/architecture/pr_second_order_numerics_design.md`.

## Strict fixed-intensity reference solve

`solve_pr_static_intensity()` solves the complete discrete equation

```text
hopping_rhs(E, prescribed_intensity) = 0
```

with a PR-owned CPU `float64` damped-Newton method. The solver uses the exact
cyclic Jacobian of the centered-difference residual and accepts an optional
initial `E` for continuation-friendly validation. Convergence requires both
residual RMS and residual maximum tolerances; a small Newton update alone is
never treated as convergence.

This is a strict material-static reference for prescribed intensity, not a
self-consistent propagation-static workflow. It intentionally favors an
independent and transparent root calculation over production-scale throughput.
Uniform and spatially modulated cases agree with long-time semi-implicit
transients in the focused validation suite.

## Image-amplification numerical readiness

`run_image_amplification_readiness()` is a bounded headless acceptance case,
not the substantive image-amplification benchmark. It combines a broad pump
and weak localized signal in one coherence group. Opposite integer Fourier
tilts make their interference grating exactly periodic on the x aperture, and
the default grid resolves each grating period with 16 samples.

The default normalized material timestep is 0.05, compared with a conservative
explicit-Euler limit of 0.0140056022409 and a semi-implicit limit of
0.238095238095. After 250 material steps, the reference run produced a final
coupled residual of `6.22445031137e-7` RMS and `6.78252578409e-6` maximum. Its
physical E state agreed with an independent zero-start Newton root for the
prescribed final optical source to `9.66995080703e-6` relative L2 and
`8.36204123975e-6` maximum absolute error. Normalized optical power drift was
`-2.33146835171e-15`, and matched-field output diagnostics showed a nontrivial
signal/pump interaction. These results clear the numerical gate for beginning
a separate, physically benchmarked image-amplification study without adding a
new workflow or architecture layer.

## Image amplification

`run_image_amplification()` implements the PRProp3D measurement chain for a
real intensity transparency supplied as a NumPy array. The transparency's
square root multiplies the signal field. After coherent propagation, a
nearest-carrier Fourier partition isolates the signal angular region and the
complex field is propagated backward through the inverse linear kernel to the
input image plane. Reported diagnostics include absolute carrier-region gain,
the paper's finite-ratio plane-wave prediction, image-intensity correlation,
normalized image RMSE, and channel-normalized power drift.

The default 64 × 32 CPU case is a scaled validation analogue with
`G_sat=10`; it is not presented as a reproduction of the paper's Figure 4.
`paper_figure4_spec()` preserves the reported 16,384 × 2,048 × 2,175 geometry,
`G_sat=4000`, beam ratio `1e-5`, 4 mm aperture, 4.35 mm interaction length,
2 µm step, 0.514 µm wavelength, 3.4 mm waists, and ±7.56° external carriers.
One float64 E state at that grid is 543.75 GiB, so the current unbatched PR
workflow cannot responsibly run it. The exact comparison is deferred to a
separate batching/boundary-condition milestone. The benchmark contract,
scaled Air Force chart results, image provenance, and limitations are recorded
in `docs/pr_image_amplification_validation.md`.

## In-memory continuation

`run_pr_timedependent()` returns a `PRTimeDependentCheckpoint` at the latest
complete material-time boundary, including for pre-cancelled and partially
completed runs. The checkpoint preserves the original normalized state
`E_initial`, the latest accepted state `E_current`, and the entrance optical
field `A0`. An intermediate propagated optical field is not checkpoint state:
every material update reconstructs its optical source by propagating `A0`
through the accepted `E_current`.

`continue_pr_timedependent()` resumes for an explicit number of additional
steps. Completed steps, requested steps, normalized material time, and progress
records remain cumulative, while `E_initial` continues to mean the initial
state of the complete run. Grid, material, beams, backend and precision,
material timestep, material integrator, and optical substeps must match. The
solver's original `Nt` is excluded from compatibility because continuation
supplies its own additional-step count.

The checkpoint contains detached host copies of the physical state. The
PR-owned `save_pr_checkpoint()` and `load_pr_checkpoint()` functions encode it
as a versioned, portable `request.json`, `checkpoint.npz`, and
`provenance.json` directory. Both JSON documents identify the material as
`pr`, the workflow as `pr_timedependent`, and the PR schema version. The NPZ
payload preserves `E_initial`, `E_current`, and `A0`; it does not substitute an
optical phase or index perturbation for the physical PR state.

Checkpoint schema version 2 records the material integrator explicitly.
Version-1 checkpoints remain loadable and are deterministically interpreted as
Euler checkpoints, which preserves the only integrator available when that
schema was written.

The codec is deliberately material-owned and directly callable. It is also
registered with the shared checkpoint-composition layer using its stable
material and workflow identifiers. The shared layer delegates payload details
back to this codec; it does not interpret `E` or reconstruct PR requests. The
existing LC checkpoint formats remain unchanged and are composed through
adapters with explicit legacy aliases for their material-less metadata.

## Two-beam geometry and measurements

For a requested internal polar angle `theta` and azimuth `phi`, the PR helper
uses

```text
k_medium = 2*pi*n/wavelength
kx = k_medium*sin(theta)*cos(phi)
ky = k_medium*sin(theta)*sin(phi)
```

The actual LCProp paraxial kernel is
`exp(-i*pi*dz*wavelength*(fx^2+fy^2)/n)`. Differentiating its spectral phase
with respect to transverse spatial frequency gives the numerical envelope
slopes

```text
dx/dz = kx/k_medium = sin(theta)*cos(phi)
dy/dz = ky/k_medium = sin(theta)*sin(phi).
```

Consequently, launch centers for a crossing at `(xc, yc, zc)` are
`x0=xc-zc*kx/k_medium` and `y0=yc-zc*ky/k_medium`. They deliberately do not
use the exact-ray `tan(theta)` slope because that is not the trajectory
implemented by the selected paraxial kernel.

The periodic plane-wave benchmark uses integer FFT modes `+m` and `-m`, so
both fields and their `2m` interference grating are continuous across x. With
`kg=(kx_signal-kx_pump)/k0`, paper Equation (7) predicts

```text
gamma_p*L = gamma*L * 2*kg / (cos(theta)*(1 + kg^2)).
```

The sign follows the ordered signal-to-pump ratio. Output modal powers are
computed by complex Fourier projection, never by spatially partitioning the
interference pattern.

For finite Gaussians, the frozen final E state is replayed through the same
Strang seam. At every z plane the coherent field is fitted to the two
pure-diffraction reference fields by solving their 2-by-2 complex Gram system.
Reported matched powers are `|coefficient|^2` times the reference norm. Since
overlapping references need not be orthogonal, these matched powers are mode
diagnostics; conserved total power is always measured directly from the full
coherent field. Channel-contribution centroids are recorded separately.
