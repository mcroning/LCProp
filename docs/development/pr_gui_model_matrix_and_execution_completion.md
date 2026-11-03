# PR GUI model matrix and execution completion

## Scope and baseline

This GUI-only milestone starts from commit
`c69f017bab38d1b1ff05ccfa9de8ee354d66b78d`, where all eight intended PR
model cells are already production capabilities. It changes presentation,
request composition, and experiment persistence only. No material equation,
material solver, optical propagation, beam-launch physics, coherence rule, or
scattering realization algorithm changes here.

## Independent model axes

The Evolution panel now exposes the three physical choices independently:

- evolution: **Static** or **Time dependent**;
- transport: **Reduced x-only (x drift/diffusion)** or **Full transverse
  (x-y drift/diffusion)**;
- material response: **Fully nonlinear** or **Linearized**.

Those controls map onto the existing production requests without introducing
a ninth model:

| Evolution | Transport | Response | Existing production request |
| --- | --- | --- | --- |
| Static | Reduced x-only | Fully nonlinear | `PRStaticRunRequest` |
| Static | Reduced x-only | Linearized | `PRStaticRunRequest` |
| Static | Full transverse | Fully nonlinear | `PRTransverseStaticRunRequest` |
| Static | Full transverse | Linearized | `PRTransverseStaticRunRequest` |
| Time dependent | Reduced x-only | Fully nonlinear | `PRRunRequest` |
| Time dependent | Reduced x-only | Linearized | `PRRunRequest` |
| Time dependent | Full transverse | Fully nonlinear | `PRTransverseRunRequest` |
| Time dependent | Full transverse | Linearized | `PRTransverseRunRequest` |

Physical model labels contain no validation badge. A separate status row and
request-summary field identify every cell as a production model selection;
commissioning remains separate evidence rather than an alias for a physical
assumption. Image Amplification retains its separate, mode-specific capability
badges and warnings, including Experimental where specialized IA validation is
pending.

## Conditional time integrators

Static choices hide the material-time controls. Time-dependent choices present
only the integrators applicable to the selected production model:

| Transport and response | Presented material integrators |
| --- | --- |
| Reduced x-only, fully nonlinear | Semi-implicit trapezoidal; Explicit Euler (reference) |
| Reduced x-only, linearized | Exact modal evolution |
| Full transverse, fully nonlinear | Spectral IMEX Euler; Explicit Euler (reference) |
| Full transverse, linearized | Exact modal evolution |

The full-transverse linearized workflow already executes its committed exact
frozen-source modal update. Its stored transverse solver token remains a
backward-compatible request field and is not reinterpreted as the material
algorithm. Loading an older linearized request preserves that token while the
GUI truthfully presents the exact update that production executes.

## Canonical scattering controls and persistence

The GUI exposes the existing `PRCanonicalScatteringSpec` fields: enable,
accumulated phase-variance strength `epsilon`, transverse correlation length in
micrometres, unsigned seed, canonical physical-z slab width in micrometres,
and the supported v1/v2 algorithm identity. The controls are available for
reduced TD and full-transverse static/TD requests, the three request families
that already carry canonical scattering. Reduced static has no scattering
request field, so the controls are hidden and ignored for that cell rather
than inventing support.

PR experiment request schema version 6 adds the previously omitted reduced-TD
`scattering` field. Version-5 reduced-TD payloads migrate deterministically to
`scattering=None`; current payloads round-trip the complete scattering spec.
The transverse codecs already retained this field, and versions 1--5 remain
accepted according to their existing migration rules. Unsupported future
versions remain rejected.

## Backend defaults and run-cost guidance

Local execution retains the NumPy default. Selecting a Slurm resource that
declares a GPU or requires CuPy selects `cupy` only when the user has not made
an explicit backend choice. Explicit `numpy`, `auto`, or `cupy` selection is
preserved. Backend origin is tracked as one of three states:

- `implicit_local_default`: the untouched local NumPy default;
- `automatic_target_default`: a reversible backend selected from the current
  Slurm CPU/GPU resource;
- `explicit_user_or_loaded`: a direct GUI selection or an experiment-loaded
  backend that target changes must never overwrite.

Consequently Local -> GPU Slurm -> Local changes NumPy -> CuPy -> NumPy, and
GPU -> CPU -> GPU Slurm resource changes select CuPy -> NumPy -> CuPy while the
choice remains automatic. Repeated transitions are deterministic and never
promote an automatic value to explicit intent. An explicit user choice,
including reselecting the already visible NumPy item, survives all target and
resource changes. Loading an experiment establishes explicit intent before
any later target-derived default can run, whether the load occurs under Local
or GPU Slurm. This default changes request composition only; backend validation
and remote submission remain unchanged.

The GUI keeps the existing quantitative local-run cost guard and adds concise
model guidance: reduced models are local-friendly or moderate,
full-transverse linearized models are local-friendly at moderate grids, and
full-transverse nonlinear models recommend Slurm except for small smoke cases.
The recommendation does not disable local execution.

## Validation

Focused tests cover all eight request cells, explicit x-only/x-y labels,
response/status separation, exact conditional integrator lists, canonical
scattering GUI and experiment round trips, version-5 migration, GPU Slurm
defaulting, reversible Local/Slurm and GPU/CPU resource transitions, explicit
backend overrides, experiment-loaded backend preservation, and the existing
Image Amplification selector. Existing workflow, persistence, and run-cost
regressions are also run. Syntax compilation and `git diff --check` complete
the local gate.
