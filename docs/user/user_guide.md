# LCProp User Guide

LCProp provides separate LC and PR applications over shared beam launch,
optical propagation, execution, progress, and result-presentation services.
This guide describes current Product behavior. Equations and immutable PR
model contracts are in [PR Model Contracts](../science/pr_model_contracts.md).

## Beam and input controls

Both applications embed the separate LaunchPane Product. A beam definition
contains profile, wavelength, power, position, transverse phase gradients,
phase, and coherence group. LCProp converts that intent into fields on its
runtime grid without making LaunchPane material-aware.

Channels with the same explicit coherence group are summed as fields before
intensity is formed; different groups add as intensities. A two-beam coupling
study must therefore set coherence groups deliberately. The Request summary
records the launched interpretation.

Profiles are:

- **Collimated Gaussian**: entrance-plane waists are specified directly.
- **Focused Gaussian**: x/y waists are defined at the focus. The signed focus
  position is relative to the interaction entrance plane and may be upstream.
- **Uniform**: the consuming grid must make each nonzero transverse phase
  gradient an exact periodic Fourier mode.

LC currently disables shared input-screen editing because LC requests do not
yet carry launch-element plans. PR exposes the supported PR input modes and
launch elements.

## Grid and optical boundaries

`Nx` and `Ny` set transverse samples. Apertures set the physical periodic FFT
cell. `dz` is the optical/material longitudinal sampling interval; optical
substeps further subdivide propagation within it where supported. These are
independent convergence controls.

The material-neutral transverse optical boundary choices are:

- **Periodic / no absorber**: exact no-op; fields wrap on the FFT cell.
- **Sponge absorber**: a smooth amplitude rate
  `exp(-gamma(x,y) * |delta_z|)` applied during propagation. Accumulated
  attenuation depends on physical distance, so subdividing a fixed distance
  into arbitrary optical substeps does not strengthen the sponge.
- **Tukey window**: a discrete separable square-root Tukey apodization. It is
  not a per-distance absorption law and must not be interpreted as a sponge.

Absorbers reduce wraparound; they do not establish that the aperture, profile,
or grid is converged.

## LC application

The LC application supports:

- fixed-director and local self-consistent static propagation;
- time-dependent director evolution;
- stationary solitons and soliton-existence sweeps;
- retained static or soliton initial states for TD where compatible;
- experiment persistence for static and TD requests;
- checkpoints and continuation;
- local execution of all displayed workflows;
- Slurm execution of canonical static propagation only.

**Physics** controls ordinary/extraordinary indices, elastic constant,
dielectric anisotropy, bias voltage, and x-boundary director angle. **Solver**
shows the static strategy, coupled-pass limit, TD step count/timestep, and
soliton controls applicable to the selected experiment.

The LC GUI currently fixes represented runtime precision according to its
request adapter and does not expose PR's backend, Fast/Full, or quantitative
runtime-estimator controls. These are explicit Product limitations, not hidden
automatic settings.

## PR model matrix

PR choice has three independent physical axes:

```text
Evolution × Transport × Material response
```

All eight combinations are production model choices. “Linearized” describes a
material approximation; it does not mean “Experimental.” Validation and H200
commissioning are separate evidence and are not uniform across the matrix.

| Evolution | Transport | Response | Applicable material update | Local guidance | Validation/commissioning note |
| --- | --- | --- | --- | --- | --- |
| Static | Reduced x-only | Fully nonlinear | Coupled damped-Newton static solve | Local-friendly to moderate | Production; extensive local regression coverage |
| Static | Reduced x-only | Linearized | Analytic reduced Fourier response inside coupled passes | Local-friendly | Production; local float32/float64 coverage |
| Static | Full transverse | Fully nonlinear | Zero-flux Newton/Krylov solve with coupled globalization | Small smoke cases only | Production; selected GPU/static paths commissioned, convergence remains problem-dependent |
| Static | Full transverse | Linearized | Analytic full-transverse Fourier response inside coupled iterations | Moderate local grids | Production; isolated operator commissioned and coupled path locally validated |
| Time dependent | Reduced x-only | Fully nonlinear | Semi-implicit trapezoidal or explicit Euler reference | Local-friendly to moderate | Production; longstanding local regression coverage |
| Time dependent | Reduced x-only | Linearized | Exact modal evolution | Local-friendly | Production; local float32/float64 coverage, GPU conditional |
| Time dependent | Full transverse | Fully nonlinear | Spectral IMEX Euler or explicit Euler reference | Slurm recommended | Production; bounded H200/CuPy commissioning exists |
| Time dependent | Full transverse | Linearized | Exact modal evolution | Moderate local grids; Slurm for scale | Production; bounded H200/CuPy commissioning exists |

Static selections do not use a material-time integrator. Exact-modal TD cells
do not expose a fake Euler choice. Full-transverse linearized requests require
an explicit positive complete-transport reference intensity `I₀`. The
full-transverse periodic biased mean field belongs to its electrical boundary
profile; the reduced material applied field is a different parameter.

## PR material and scattering

The PR Material tab controls dark intensity, uniform background, reduced-model
applied field, gain-length product, refractive index, and advanced
normalization. Dark and uniform background contributions are already included
in the complete transport intensity and must not be added again elsewhere.

Canonical volume scattering records:

- enable state;
- accumulated phase-variance strength `epsilon`;
- transverse correlation length in micrometres;
- unsigned realization seed;
- canonical physical-z slab width;
- algorithm identity.

Scattering is available for reduced TD and full-transverse static/TD. Reduced
static has no canonical scattering request field, so the GUI hides those
controls rather than inventing support.

For fanning work, record scattering parameters, beam/focus, coherence groups,
boundary treatment, material-z sampling, and optical substeps separately.
Material sampling and optical propagation substeps are not interchangeable.

## Execution target, backend, and precision

Execution target answers *where* the job runs; backend answers *which array
implementation* performs the scientific calculation.

- Local starts with the implicit NumPy default.
- A GPU-capable Slurm resource may supply an automatic CuPy default.
- Returning to Local restores the prior implicit local default while the
  choice remains automatic.
- Explicit user selections and experiment-loaded backends are never silently
  overwritten by target/resource changes.
- `float64` is the conservative default for reference and publication-facing
  work. `float32` is supported only where the selected workflow validates it;
  users must still perform problem-specific convergence comparisons.

Slurm configuration uses system SSH/agent authentication. Test a profile
before submission. Structured `progress.json` telemetry is atomically replaced
when possible and is always best-effort: progress-write failure cannot change
the scientific result, packaging, cancellation, or original exception.

## Runtime and resource planning

The PR **Run Planning** panel is an advisory estimator, not an accuracy oracle
or benchmark. It reports a range, confidence, dominant cost, memory, and
Fast/Full package-size estimates from committed measurements plus transparent
scaling. Missing or weak calibration lowers confidence.

Full-transverse nonlinear static timing is especially sensitive to
initialization, continuation, line search, and iteration counts. Its estimate
is intentionally conservative. Local-versus-H200 recommendations never modify
the request, backend, grid, tolerances, retention policy, or execution target.

A publication-facing “Rolls Royce” configuration may reasonably use
full-transverse nonlinear transport, float64, an open/sponge optical boundary,
fine material-z sampling, sufficient optical substeps, and H200 execution. It
is not a universal preset: each observable requires its own aperture, grid,
longitudinal-step, timestep, boundary, and model-convergence study.

## Fast and Full result retention

PR remote execution exposes:

- **Fast / Exploratory**: optical endpoints, compact diagnostics/provenance,
  exact full-resolution longitudinal intensity cuts nearest zero, and a
  bounded float32 MPR preview.
- **Full**: complete supported scientific arrays and volumes in addition to
  compact products.

Fast arbitrary MPR slices are block-averaged visualization products. Metadata
records original/preview grids, actual retained z planes, coordinates,
reduction, normalization, dtype, and byte budget. They are not quantitative
replacements for the exact Fast cuts. Fast TD retains the final accepted 3-D
preview plus a bounded fixed-scale material-time movie, not a 4-D volume
history.

Older supported payloads may predate cuts or previews; the GUI reports those
products unavailable instead of synthesizing them.

## Understanding results

- **Fields**: transverse fields and the x-y member of a linked MPR volume.
- **Longitudinal/MPR view**: linked x-y, x-z, and y-z views with a common
  physical crosshair when a volume is available.
- **Curves**: convergence, accepted-state TD quantities, and workflow-specific
  histories. Linearized modes do not fabricate nonlinear iteration curves.
- **Diagnostics**: convergence, power, backend, retention, carrier-power, and
  workflow provenance.
- **Request**: immutable request summary for the actual run.
- **Console**: throttled progress, warnings, completion, or failure details.

Carrier-resolved Fourier-space power is the quantitative two-beam coupling
diagnostic. Peak intensity or apparent brightness in independently autoscaled
images is not energy transfer: focusing, beam-shape change, and spatial
redistribution can change peaks without a one-to-one carrier-power change.

## Persistence and compatibility

Experiment files store representable user intent. Supported older versions
migrate missing additive fields to their historical defaults, including
periodic optical boundaries and nonlinear PR material response where
appropriate. Current files round-trip nonperiodic boundaries and PR model
choices exactly. Unsupported future versions are rejected.

Checkpoints contain continuation state and are validated against the current
request. Fast transport results explicitly mark omitted fields unavailable.
Neither experiment loading nor result reconstruction silently invents missing
scientific arrays.

## Warnings and limitations

- A GUI “production model” label identifies a supported physical choice, not
  uniform validation quality across all hardware and parameter regimes.
- Image Amplification has separate mode-specific validation badges. A
  linearized base can therefore be Experimental for that composite experiment
  without making the underlying linearized material model undefined.
- Full-transverse nonlinear Profile v1 is unbiased. Nonzero periodic mean bias
  is supported by the declared linearized current-carrying profile, not by
  silently extending the nonlinear zero-flux workflow.
- Local cost warnings are advisory and do not prevent an explicitly confirmed
  run.
- Cancellation preserves the last accepted state where the workflow supports
  continuation; incomplete candidates are discarded.

For a first run, return to the [Quick Start](quick_start.md).
