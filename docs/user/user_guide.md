# LCProp User Guide

LCProp provides separate LC and PR applications over shared beam launch,
optical propagation, execution, progress, and result-presentation services.
This guide describes current Product behavior. Equations and immutable PR
model contracts are in [PR Model Contracts](../science/pr_model_contracts.md).
The corresponding LC equations and numerical semantics are in
[LC Model Contracts](../science/lc_model_contracts.md).

## Beam and input controls

Both applications embed the separate LaunchPlane Product, requiring schema 3
or newer and Python 3.11 or newer for the complete GUI installation. A beam
definition contains profile, wavelength, power, position, transverse phase
gradients, phase, and coherence group. LCProp converts that intent into fields
on its runtime grid without making LaunchPlane material-aware.

Channels with the same explicit coherence group are summed as fields before
intensity is formed; different groups add as intensities. A two-beam coupling
study must therefore set coherence groups deliberately. The Request summary
records the launched interpretation.

Profiles are:

- **Collimated Gaussian**: entrance-plane waists are specified directly with no
  focusing curvature there. A Gaussian still has a waist; “collimated” does not
  imply a physically waist-free beam.
- **Focused Gaussian**: x/y waists are defined at the focus. The signed focus
  position is relative to the interaction entrance plane and may be upstream.
- **Uniform**: the consuming grid must make each nonzero transverse phase
  gradient an exact periodic Fourier mode.

LC currently disables shared input-screen editing because LC requests do not
yet carry launch-element plans. PR exposes the supported PR input modes and
launch elements.

## Grid and optical edge treatment

`Nx` and `Ny` set transverse samples. Apertures set the physical periodic FFT
cell. `dz` is the optical/material longitudinal sampling interval; optical
substeps further subdivide propagation within it where supported. These are
independent convergence controls.

FFT propagation always uses a periodic computational domain. Optical edge treatment choices are:

- **None**: exact no-op; fields wrap on the FFT cell.
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
- implemented Slurm execution of canonical static propagation only, with real
  CPU/NumPy scheduler commissioning still pending.

LC propagation currently uses one optical wavelength kernel. All enabled LC
channels must therefore use the same wavelength; unequal-wavelength requests
are rejected rather than approximated. Stationary soliton and existence
workflows are periodic-only. Propagation workflows apply their selected
None, Sponge, or Tukey edge treatment to the actual optical march.

**Physics** controls ordinary/extraordinary indices, elastic constant,
dielectric anisotropy, bias voltage, and x-boundary director angle. **Solver**
shows the static strategy, coupled-pass limit, TD step count/timestep, and
soliton controls applicable to the selected experiment.

LC production workflows currently execute with NumPy. Headless static and TD
requests support NumPy float64/complex128 and float32/complex64; the LC GUI
represents the conservative float64 choice. The GUI does not expose PR's
backend, Fast/Full, or quantitative runtime-estimator controls. GPU Slurm
profiles are disabled for LC execution because selecting a GPU resource does
not change the NumPy scientific backend. These are explicit Product
limitations, not hidden automatic settings.

Static execution completion and scientific convergence are separate. If any
slice fails the configured residual qualifications, the retained result and
GUI report the static solution as nonconverged even though execution and
result construction completed.

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

Those automatic backend rules apply to the PR application. LC does not expose
an automatic or CuPy production backend: it is NumPy-based, as described in
the LC section above.

LC Slurm execution is implemented only for canonical NumPy static propagation
on a CPU resource profile, and real scheduler commissioning is pending. TD,
soliton, and soliton-existence remote execution are unsupported.

Slurm configuration uses system SSH/agent authentication. Test a profile
before submission. Structured `progress.json` telemetry is atomically replaced
when possible and is always best-effort: progress-write failure cannot change
the scientific result, packaging, cancellation, or original exception.

Automatic Slurm source deployment is a source-checkout workflow: it archives
the committed `src/` tree and `pyproject.toml` from a clean Git revision. An
installed wheel alone is not a remote-source deployment mechanism.

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
full-transverse nonlinear transport, float64, sponge optical edge treatment,
fine material-z sampling, sufficient optical substeps, and H200 execution. It
is not a universal preset: each observable requires its own aperture, grid,
longitudinal-step, timestep, boundary, and model-convergence study.

## Fast and Full result retention

PR remote execution exposes:

- **Fast / Exploratory**: optical endpoints, compact diagnostics/provenance,
  exact full-resolution longitudinal intensity cuts nearest zero, and a
  bounded float32 orthogonal-slice preview.
- **Full**: complete supported scientific arrays and volumes in addition to
  compact products.

Fast arbitrary orthogonal slices are block-averaged visualization products. Metadata
records original/preview grids, actual retained z planes, coordinates,
reduction, normalization, dtype, and byte budget. They are not quantitative
replacements for the exact Fast cuts. Fast TD retains the final accepted 3-D
preview plus a bounded fixed-scale material-time movie, not a 4-D volume
history.

Older supported payloads may predate cuts or previews; the GUI reports those
products unavailable instead of synthesizing them.

## Understanding results

- **Fields**: transverse fields and the x-y member of a linked orthogonal-slice volume.
- **Longitudinal/orthogonal-slice view**: linked x-y, x-z, and y-z views with a common
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

## Inspecting requests and identifying displayed results

In either LC or PR, **Inspect Request** captures pending editor values,
constructs and validates the selected request, and opens Results → Request.
It does not start a worker, propagate fields, relax a material state, or contact
Slurm. Capturing the request commits pending beam-editor values and updates
optical context; specialized image validation can also prepare its raster/launch
inputs. These are request-preparation side effects, not a scientific run.

The preview uses the same scientific summary as execution. It includes
precision, execution target, requested backend, and the selected cluster/resource
for Slurm. A backend requiring runtime resolution is explicitly unresolved;
preview does not probe a GPU. Comparison resource estimates in Run Planning
are separate from the configured execution path. LC continues to use NumPy
and its existing float64 GUI settings.

A visible Results banner identifies the displayed request and its state:
**Previous run**, **Current accepted state** or **Current progress state** when
available, **Completed result**, or **State at stop/cancellation**. New attempts
mark retained results as previous before validation, so a rejected request cannot
claim an older result. When the inspected/requested configuration differs, the
Request tab also retains the summary that owns the displayed result. A failed
run with a retained current progress product labels it **State at failure**.
Completed result means execution returned a result; it does not assert solver
convergence. Inspect the existing convergence diagnostics separately.
Unavailable fields or curves are hidden when a new product replaces the old one.
If a result update fails, all result panes become unavailable and the banner
reports that no result is displayed; a later successful update restores them.

Reduced PR time-dependent **Local** runs and Local Continue display bounded
**Current accepted state** snapshots in the existing Results tabs. They contain
sampled output intensity, one accepted material-field plane (its z position is
shown), and fixed nearest-zero x-z/y-z optical cuts. Diagnostics records the
accepted segment/cumulative step, normalized time, existing scalar diagnostics,
and point-sampling coordinates. Cuts use the existing slice-average optical
intensity convention. These visualization-only samples are not a checkpoint or
a material-time history; final scientific results retain their existing policy.

The first and final accepted steps are displayed; intermediate scientific frames
are limited to one per 0.5 seconds after the previous delivery. Status may advance
between frames, while the displayed frame retains its own step/time. Stop makes
the last accepted step available. No accepted update means no new live frame.
Changing display limits cannot change the calculation. Full-transverse PR live
Results and remote scientific previews are not part of this capability; Slurm
continues to report scalar progress.

PR checkpoint **Continue** is Local-only, even when Slurm is selected for Run.
Its execution summary and Console identify Local execution. Continuation does
not validate Slurm resources or require an automatically deployable Git source;
the selected target remains available for the next ordinary Run.

Results refreshes preserve your selected Fields, Curves, Diagnostics, Request,
or Console subtab. Timestamped Console boundaries separate Run, Continue,
experiment/checkpoint operations, and remote dispatch. Configuration errors show
the cause above Results; full tracebacks remain in Console. Scientific field
names are shortened in the views while original names and provenance remain
in Diagnostics and persisted products.

For PR, **General beams / two-beam coupling** is the ordinary beam route.
**Image amplification — specialized setup** requires the signal screen,
pump/signal roles, and symmetric carriers in the x-z plane. Those specialized
constraints do not apply to general beams. The selected model still matters:
reduced PR material transport acts along x with y as a batch axis; rotating
beam crossings is not generally physically equivalent.

Automatic Slurm source deployment requires a clean deployable LCProp Git
checkout. A normal non-editable installation remains suitable for Local runs.
Inspect Request and Run check the automatic source locally before staging or
submission. If it is not a checkout, configure a runner with the supported
`local_source` checkout argument, or use the existing pre-staged source route
with both `LCPROP_SLURM_SOURCE_PATH` and `LCPROP_SLURM_SOURCE_SHA`. The profile's
Remote source root is the remote destination, not a local checkout selector.
This preflight does not certify remote connectivity, environment, or device
availability; those checks remain part of remote execution.

## Results and numerical transparency

### Solitons and existence sweeps

**Solver converged within configured limits** means the applicable strict
convergence checks passed together in one completed outer iteration. A returned
nonconverged solution remains inspectable; nonconvergence does not establish
physical nonexistence or instability, and it is distinct from worker execution
failure or cancellation.

The member selector above the Results tabs identifies each retained sweep
member by power, solver status and failed gates. **Convergence details** opens
its execution status, termination, iteration usage and exact failed values
without replacing the selected Results subtab. Execution completion does not
mean solver convergence.

Open **Results → Samples / Tables** and select **Samples**, **Sweep members**,
or **Convergence gates**. Available member values include power, beta, residuals,
field relative change, mode overlap, theta update, completed iterations, budget,
solver status, and termination reason. The gates table shows exact retained
outer-iteration values, request tolerances, the strict `<` comparison, and
Pass/Fail. Missing evidence is labeled **Unavailable**, not reconstructed from
rounded plots. Cells use concise quantity-aware text; tooltips retain exact
round-trip float values. Near-unity overlap retains its meaningful precision.
Results without retained tables show an explicit explanation.

Convergence curves show applicable tolerance lines and guidance beside the plot;
mode overlap has no additive axis offset. **Transverse RMS widths** has separate
x/y legend entries. A near-unity overlap alone does not establish convergence.
For transverse eigenpair refinement, qualification uses the outer history and
includes an optical-residual gate. Final polishing does not update that outer
convergence flag; final returned-field diagnostics can differ from gate values.

The GUI existence path uses `ParameterSweepRequest` with default
`SolitonRequest`, not the separate `SolitonExistenceRequest` API defaults.
Continuation initializes a subsequent member from a previously converged member
and can improve branch following. It does not waive any convergence criterion.

### Configured execution and comparison estimates

Inspect Request and Run Planning identify the configured Local/Slurm target,
requested backend, precision, and (for Slurm) cluster, resource and retrieval
policy. Backend resolution remains explicitly unknown until execution where it
is not yet known; planning does not probe devices. Completed-run diagnostics
and remote status provide execution evidence.

**Run Planning → Configured execution** is separate from **Comparison estimates**.
Local Mac/NumPy and H200/CuPy estimates do not select a backend or resource.
Selecting a GPU allocation with explicit NumPy produces a warning: scientific
computation will not use the allocated GPU. The explicit choice is preserved.
Full-transverse PR requires explicit NumPy or CuPy; Auto is unsupported there.
Configuration changes invalidate the displayed planning estimate.

### Display scaling and temporal locking

Image and longitudinal panes provide **Auto each frame**, **Manual limits**, and
**Lock scale across frames**, with **Minimum**, **Maximum**, and **Apply**.
Concise displayed limits keep their exact values when applied without editing;
tooltips show those exact limits. Auto remains the default. Transverse x-y
scaling is independent of longitudinal scaling. The x-z and y-z views of the
same selected 3-D quantity share one longitudinal scale.
Limits must be finite and lower must be less than upper. Invalid entries leave
the applied scientific-image mapping unchanged. Select Auto to resume automatic
scaling, even after an invalid manual entry.

Auto preserves the existing presentation policies: transverse intensity uses a
robust positive-intensity percentile, other transverse quantities use their
finite range, and longitudinal cuts use their shared volume range (or the
combined retained-cut range). Auto can therefore rescale across frames and
across transverse versus longitudinal views.

Fixed and locked limits share a mapping between x-z/y-z views of the same
source quantity. Transverse x-y limits remain independent, even when the plane
comes from that volume. Unrelated quantities have separate settings.
Lock captures the limits currently displayed in the pane where it is selected;
identical values then have identical brightness across subsequent updates.
A new-run scale reset captures fresh limits for locked quantities; manual limits
persist during the session. Flat or entirely nonfinite fields get finite display
bounds. These controls work with existing frame/progress updates; they do not
add movie history or alter already encoded MP4 previews.

All scaling is display-only. Rendering may saturate outside the chosen range;
stored arrays, powers, curves, checkpoints and transport artifacts are unchanged.
Settings are session-only and do not change experiment/checkpoint schemas.

## User-local configuration and Beam workspace

Cluster profiles belong to you, not to Product defaults. **Configure Remote
Execution** shows the catalog location and explicit cluster/resource default
checkboxes. Check one to make it the default when saving; uncheck the current
default to clear it. Editing another resource preserves the existing default.
Without an explicit default, the GUI selects the first available entry; this
fallback does not write a default. Multiple clusters/resources remain supported.
The catalog lives outside the package/repository and survives ordinary upgrades.
A new catalog is empty: create your first profile to configure remote execution.

For a profile-only clean-user session on macOS/Linux, without touching your real
catalog:

```bash
LCPROP_CLUSTER_CONFIG="$(mktemp -d)/clusters.toml" python -m lcprop.pr.gui.app
```

The absent temporary catalog starts empty. This override does not isolate native
macOS file-dialog history or other operating-system preferences. Experiment
Save/Open supplies an empty initial directory; the native dialog may remember a
previous location. LCProp does not impose a developer experiment directory.

**Beam → Beams** gives the LaunchPlane editor the workspace. Secondary controls
remain available in **Optical edge treatment**, and in PR **Input Screen**.
LC does not support input screens and does not allocate a permanent disabled pane.
Beam-center and tilt-handle dragging remain LaunchPlane's existing interactions.
