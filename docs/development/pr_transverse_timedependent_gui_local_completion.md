# PR Transverse Time-Dependent GUI and Local Completion

**Date:** 2026-08-28

**Branch:** `feature/pr-second-order-static`

**Implementation base:** `f44ea5923ac9f8531c195bed67da3d08ac8d914e`

## Purpose

This record documents the completion of the local application layers around
the existing canonical `pr_transverse_timedependent` workflow. The milestone
made the full-transverse, two-dimensional zero-flux PR time-dependent model a
first-class choice in the standalone PR GUI and in the Image Amplification
composite experiment. It did not change the transverse transport equations,
electrostatic closure, optical response, time integrators, or scientific
diagnostic definitions.

## Previous Missing Layers

Before this milestone, the canonical request/result types, production solver,
registered workflow operation, NumPy/CuPy numerical paths, and shared product
adapter already existed under `lcprop.pr.transverse`. The PR GUI nevertheless
disabled the transverse-TD selector because it could not construct the
canonical request. The GUI's `LocalRunner` did not register the operation,
ordinary completion handling did not recognize the result type, experiment
persistence had no transverse-TD codec, and Image Amplification used a
prepared-field compatibility adapter with validation still marked pending.

## Canonical Request Mapping

The GUI now constructs `PRTransverseRunRequest` directly. Its shared controls
map to:

- the BeamPanel `BeamStack` and declarative channel launch elements;
- `GridSpec` and `PRMaterialSpec`;
- the frozen Profile-v1 transport, dielectric, boundary, and scalar optical
  projection records;
- `PRTransverseSolverOptions`, including material steps, normalized timestep,
  optical substeps, and the selected transverse integrator;
- the existing `BackendSpec` precision and backend choice;
- no runtime initial optical or potential arrays.

Profile v1 retains its established restrictions: the applied field is zero
and execution requires an explicit NumPy or CuPy backend. The common Auto
choice remains visible because the backend selector is shared, but canonical
transverse-TD preflight rejects it actionably rather than silently changing
the requested backend. The current GUI has no canonical scattering editor, so
ordinary GUI construction produces `scattering=None`; headless scattering
support in the workflow is unchanged.

## GUI Controls and Local Execution

The selector label is `Time dependent (2D transverse zero-flux)` in ordinary
mode and `Transverse TD [Validated]` in Image Amplification mode. Selecting it
exposes the finite-time controls and changes the integrator choices to:

- `spectral_imex_euler`, the production default;
- `explicit_euler_reference`, the transparent reference method.

The PR window registers the existing
`PR_TRANSVERSE_TIMEDEPENDENT_OPERATION` with its `LocalRunner`. Execution uses
the same generic `WorkflowWorker`, cancellation token, progress signal, and
result workspace as the other PR workflows. No transverse-specific worker or
second operation identity was added.

## Progress and Completion Semantics

Progress comes directly from accepted material steps in
`run_pr_transverse_timedependent()`. The coordinate is normalized material
time, `completed_steps * dt_normalized`. Finite integration reports
`completed`, not `converged`; accepted-boundary cancellation reports
`cancelled`. The GUI presents these as Completed and Stopped, respectively,
and does not retain a reduced-TD checkpoint for this workflow.

Cancellation was verified both before the first accepted step and after an
accepted step. The final optical replay remains complete, and a cancelled run
returns only the last accepted potential state.

## Product Behavior

Ordinary completion passes the canonical result through
`pr_transverse_result_to_run_data()`. The GUI receives input and output optical
intensity, full longitudinal volumes for potential, normalized carrier
density, `E_x`, `E_y`, and the optically active field, plus physics-profile,
backend, power, carrier, gauge, curl, Gauss-law, integrator, and cancellation
diagnostics. Image Amplification adds its common carrier-isolated signal,
back-propagated reconstruction, gain, fidelity, and power products.

The existing ordinary transverse-TD adapter does not synthesize sparse x-z or
y-z optical histories because the canonical finite-time result does not retain
the intermediate source-intensity volume. No new history allocation was added
to this GUI milestone.

## Declarative Launch Elements

`PRTransverseRunRequest` now carries the same immutable
`ChannelLaunchElements` tuple used by the other PR workflows. The production
workflow validates the channel assignments, rejects a prepared `initial_A`
combined with declarative elements, and passes the elements once to the shared
`build_launch()` function. Tests cover no-screen launch, a screen on an
arbitrary channel, an unchanged unscreened channel, exact agreement with one
shared launch application, and rejection of a possible double application.

This eliminates the former Image Amplification prepared-field exception. The
composite now supplies declarative elements to the ordinary registered
transverse-TD operation.

## Experiment Persistence

The existing versioned `.lcprop.json` experiment envelope now registers a
material-owned codec for `pr_transverse_timedependent`. It preserves grid,
beams, material, all four transverse profiles, solver policy, backend and
precision, canonical scattering specification when present, and portable
launch elements. Runtime `initial_A` and `initial_psi` arrays are rejected.

The Image Amplification codec dispatch also recognizes transverse TD. Its
round trip preserves the selected base workflow, base request, two coherent
channels, passive signal screen, and explicit pump/signal roles. Actual PR GUI
Save/Open tests rebuild requests with exact dataclass equality. Remote-result
transport and experiment persistence remain separate formats.

The shared experiment envelope still has no independent user-visible
experiment name/title field. That pre-existing usability limitation was not
expanded into this milestone.

## Bounded Image Amplification Validation

A deterministic NumPy/float64 calculation used two coherent channels, an
explicit pump and signal, one passive raster screen on the signal only, no
pump screen, a `32 x 8` grid, two material steps, and
`dt_normalized = 1e-4`. The same declarative launch was also run through the
reduced TD workflow for a bounded comparison of common infrastructure, not as
a claim that the two material models should be physically identical.

| Metric | Transverse TD | Reduced TD |
|---|---:|---:|
| Composite status | completed | completed |
| Base status | completed | completed |
| Analysis status | completed | completed |
| Measured absolute signal gain | 1.0000064969375924 | 1.0000056526241667 |
| Image-intensity correlation | 0.9999999999793246 | 0.9999999999902769 |
| Normalized image RMSE | 6.40306305177292e-06 | 4.588854245587175e-06 |
| Normalized power drift | 0.0 | -1.1206373046081234e-16 |

The initial complex optical fields were exactly equal, establishing common
launch construction and exact-once signal-screen application. The final-field
relative L2 difference was `1.1006735765452054e-06`, which is recorded only as
a bounded comparison between different material equations. Both
reconstructions were finite and recognizable. These results support promoting
the local transverse-TD Image Amplification capability from validation-pending
to validated for this bounded path; they are not a production-scale image
amplification benchmark.

## Backend Evidence

This milestone validated NumPy/float64 locally and retained the existing
float32/float64 and NumPy/CuPy transverse-TD regression coverage. CuPy tests
remain skipped when no local CUDA device is available. Earlier project H200
commissioning is relevant evidence for the canonical numerical backend, but
no cluster job or new GPU claim was made here.

## Remaining Remote-Execution Gap

The PR GUI does not advertise Slurm execution for ordinary or composite
transverse TD because no registered remote transport codec exists for this
workflow. Capability-based operation checks reject that selection before
submission. Implementing and commissioning transverse-TD remote transport is
a separate milestone.

## Validation

Focused validation covered canonical request construction, GUI operation
registration and worker dispatch, progress and completion, products,
cancellation, launch elements, ordinary and composite persistence, and the
bounded Image Amplification calculation: **155 passed**. The complete PR suite
then passed with **449 passed, 57 skipped** under the repository's established
local environment. The skips are existing optional-backend cases, including
CuPy tests where a local CUDA device is unavailable.

## Conclusion

The canonical full-transverse PR time-dependent workflow is now complete for
ordinary local GUI use and bounded local Image Amplification use. The work
filled application, launch, and persistence gaps around the established
solver without changing its physics. Remote/Slurm enablement remains
deliberately unavailable pending a dedicated transport and commissioning
milestone.
