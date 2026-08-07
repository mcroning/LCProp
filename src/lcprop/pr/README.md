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

## Self-consistent static workflow

`run_pr_static()` solves the coupled optical/material steady problem with a
slice-local z march. For each slice it holds the accepted incoming optical
field fixed, evaluates the same before/after midpoint source used by the
time-dependent workflow, and uses `solve_pr_static_intensity_batched()` for
the prescribed-source material correction. Every damped correction is then
checked against a newly propagated source. Only the refreshed complete PR
residual can establish convergence, and both its RMS and maximum must pass.

The structured material solve uses a PR-owned batched cyclic-tridiagonal
kernel along x. It supports NumPy and CuPy without changing the discrete
equation, while `solve_pr_static_intensity()` retains the dense NumPy solve as
the small-system oracle. The completed E volume is independently replayed
from the original launch field; sequential and replayed optical fields,
sources, and residuals must agree before the workflow reports convergence.
Results are returned as detached host arrays, consistent with the existing PR
time-dependent result boundary.

Coupled-static tolerances use explicit precision-aware defaults when their
option value is `None`. Float64 retains material residual RMS/max defaults of
`1e-10`/`1e-9`, coupled residual RMS/max defaults of `1e-8`/`1e-7`, and replay
relative/absolute defaults of `1e-11`/`1e-12`. Float32 uses
`2e-6`/`1e-5`, `2e-6`/`1e-5`, and `2e-6`/`2e-7`, respectively. Every numeric
override is preserved exactly. Supplying a `PRStaticSolverOptions` object
likewise makes its material tolerances explicit; omitting it selects the
precision default without changing the fixed-intensity solver API. Results
record every resolved value and whether it came from the precision policy, an
explicit workflow override, or an explicitly supplied material-solver object.

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

## Streaming static image benchmarks

`run_pr_static_streaming()` provides a PR-owned bounded-memory z march for
paper-scale static comparisons. The production mode uses the complete
nonlinear hopping residual, refreshed midpoint source, cyclic material solve,
and prepared-response Strang optical advancement used by `run_pr_static()`.
It retains only the current and previous material slices, scalar per-slice
moments and summaries, and explicitly requested x-z or y-z cross-sections.

Because no E volume exists to replay, its independent validation is a second
complete streaming solve from the original launch field and zero material
seed. No first-pass E slice is reused. Final optical fields and per-slice
state/source/residual moments must reproduce within the resolved precision
tolerances. This deterministic recomputation is not the frozen-volume replay
performed by `run_pr_static()`, and the result provenance states that
difference explicitly.

Three research modes remain separate:

- `legacy_linearized_spectral_lie` implements the trusted PRProp3D static
  spectral solution of paper Equation (5), exact angular-spectrum hop, and
  full-hop/full-response Lie ordering;
- `full_nonlinear_lie_reference` replaces only the linearized material solve
  with LCProp's complete nonlinear steady residual;
- `production_nonlinear_strang` uses the complete nonlinear coupled solve and
  the shared prepared-response Strang interface.

The optional square-root two-dimensional Tukey window reproduces PRProp3D's
window definition and is disabled unless requested. Optional correlated
volume phase noise follows the trusted amplitude, Gaussian-filter, and
correlation-length scaling. It accepts either an explicit seed for every z
slice or a base seed that deterministically derives each slice seed. Explicit
sequences are length-checked against `Nz` and recorded by count, endpoints,
and SHA-256 digest, so independent recomputation is meaningful without
copying a long seed list into each result.

`paper_figure6_spec()` records the published large-signal image-amplification
contract: 16,384 × 2,048 × 2,175 samples, 4,000 µm × 4,000 µm aperture,
4,350 µm length, 2 µm step, 0.514 µm wavelength, 3,400 µm waists, equal
incident peak intensities, external half-angles ±7.56 degrees,
`G_sat=4000`, `Id=0.01`, Tukey alpha 0.05, and no scattering noise. The caller
supplies the checksummed Air Force chart. Figure 6 uses the inverted real-image
transparency: inversion precedes square padding and nearest-neighbor placement,
giving the published dark chart field with bright bars while the exterior
remains transparent. This polarity belongs to the benchmark spec; general
production defaults and the preprocessing API are unchanged.

The paper supplement's file named `Figure_3_4_6.json` instead contains an
unrelated no-image, noisy 3,000 µm × 1,000 µm saved run. Its 1,970 historical
per-slice noise seeds remain preserved in `figure6_noise_seeds.py` as useful
fanning/reference data, but `paper_figure6_spec()` does not select them. Their
little-endian uint32 SHA-256 remains
`ade77c0e678bf3c2836131c4771e9df22774eba3cc3eb30e17150107adf2f32f`.

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
