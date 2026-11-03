# Standalone Photorefractive GUI Vertical-Slice Design

**Date:** 2026-08-04

**Branch:** feature/pr-gui-vertical-slice

**Status:** Implemented and validated

## Purpose

This document defines the first interactive photorefractive interface for
LCProp. The interface is intentionally a standalone PR application rather than
a material mode added to the existing LC window. It uses the validated PR
workflow, operation, product adapter, checkpoint codec, and shared optical
engine without changing LC physics or asking LC users to navigate PR concepts.

The vertical slice is deliberately small. It proves that the existing
headless material-composition boundaries can support a focused application,
while preserving the current LC application and its user experience.

## Product Decision

LC and PR should have separate user-facing applications backed by shared
infrastructure:

```text
Shared GUI infrastructure
├── background worker and cancellation
├── compatible beam editing
├── result workspace and plotting
├── RunData presentation
└── common checkpoint dispatch

LC application                    PR application
├── LC physics                    ├── PR material physics
├── director solvers              ├── normalized E evolution
├── LC experiments                ├── PR propagation
└── theta continuation            └── E continuation
```

There will be no material selector in either main window. The existing LC
window and application module remain the LC interface. The new application is
identified as **LCProp PR** and is launched independently. A lightweight
launcher may be considered later, but it is not part of this milestone.

## Goals

- Construct a valid `PRRunRequest` from an interactive PR-specific interface.
- Execute `PR_TIMEDEPENDENT_OPERATION` through `LocalRunner` in a Qt worker.
- Support progress, cooperative stop, and safe application shutdown.
- Present completed and cancelled results through the existing PR `RunData`
  adapter and shared result workspace.
- Retain, save, load, validate, and continue `PRTimeDependentCheckpoint`.
- Preserve beam phase-gradient units and coherence groups exactly.
- Keep the current LC GUI, LC requests, LC persistence formats, and LC tests
  unchanged.

## Non-Goals

The first PR GUI will not expose:

- the plane-wave coupling benchmark;
- finite two-beam crossing construction;
- image amplification;
- scattering or fanning;
- screening solitons;
- noise sources;
- parameter sweeps or batching;
- material selection inside one window;
- external plugin discovery;
- a universal material panel, request, result, or state interface.

Those capabilities remain headless research or future product milestones.

## Current GUI Assessment

### Reusable without physics changes

`lcprop.gui.workers.WorkflowWorker` is workflow-neutral. It invokes a callable
with a request, cancellation token, and progress callback, and returns the
callable's result without interpreting material state.

`lcprop.gui.workspace.Workspace`, `ResultsPanel`, and the image, longitudinal,
curve, diagnostics, request, and console views consume `RunData`. They can
display PR final products already produced by `pr_result_to_run_data()`. The
workspace's LC-specific time indicator logic does not prevent PR field display;
the PR window can set its material-time indicator explicitly.

`BeamPanel` uses the separate LaunchPane package and converts enabled beams to
`BeamStack`. Its wavelength, power, waist, center, transverse phase gradients,
phase, and coherence-group fields are applicable to PR. Its built-in beam and
aperture defaults are LC-oriented and must be replaced when used by the PR
application. The compatibility-only `theta_weight=1` added by the current
adapter is ignored by PR physics and is not presented as a PR control.

The numerical spin-box helpers and most result views are also reusable.

### Not reusable as PR composition

`LCPropMainWindow` directly owns LC experiment names, LC requests, LC
initial-condition modes, director continuation, soliton state, LC live-state
conversion, and LC result summaries. It should not be subclassed by the PR
application.

`PhysicsPanel`, `SolverPanel`, `ExperimentPanel`, `SweepPanel`, and
`RetainedResults` model LC concepts. They remain part of the LC application.

`GridPanel` produces a generic `GridSpec`, but its defaults were selected for
LC. The first PR application should own its grid defaults and validation rather
than silently inheriting them.

## Proposed Package Layout

```text
src/lcprop/pr/gui/
    __init__.py
    app.py
    main_window.py
    material_panel.py
    solver_panel.py
    grid_panel.py
    request_adapter.py
```

The PR GUI imports shared widgets from `lcprop.gui` where they are already
material-neutral. PR panels and request mapping stay under `lcprop.pr.gui`.
The existing `src/lcprop/gui/main_window.py` and `src/lcprop/gui/app.py` remain
the LC application.

No inheritance hierarchy is required. `PRMainWindow` composes ordinary Qt
widgets and callables.

## Application Entry Point

The application module will support:

```text
python -m lcprop.pr.gui.app
```

The packaging configuration may additionally expose:

```text
lcprop-pr
```

Adding the PR entry point must not change how the existing LC application is
started. The PR GUI uses the window title `LCProp PR` so users cannot confuse
its normalized material controls with LC director parameters.

## Window Composition

The initial window contains a header and five tabs:

1. **PR Material**
2. **Beam**
3. **Grid**
4. **Evolution**
5. **Results**

The header contains **Run**, **Continue**, **Stop**, **Save Checkpoint**, and
**Load Checkpoint** actions plus a concise runner/status label. There is no
experiment selector because this vertical slice has one workflow:
`pr_timedependent`.

### PR Material panel

The primary controls map one-to-one to `PRMaterialSpec`:

| GUI label | Model field | Units/semantics |
|---|---|---|
| Dark intensity | `dark_intensity` | normalized, nonnegative |
| Uniform background | `uniform_background_intensity` | normalized, nonnegative |
| Applied field | `applied_field` | normalized signed field |
| Gain-length product | `gain_length_product` | signed dimensionless coupling |
| Refractive index | `refractive_index` | positive scalar |

An advanced section exposes relative permittivity, mobile charge density,
temperature, and the optional characteristic-wavenumber override. The
override uses an explicit enable checkbox so an absent value remains `None`;
zero is not used as a sentinel.

Labels must say **normalized** wherever the value is not a laboratory unit.
The panel must not use LC labels such as bias voltage, director angle, or
dielectric anisotropy.

### Beam panel

The PR application reuses LaunchPane beam editing but installs PR-owned
defaults. Enabled beams map to `BeamChannel` without changing the public phase
gradient semantics:

```text
tilt_x_rad_per_um = kx
tilt_y_rad_per_um = ky
launch phase = kx*x + ky*y + phase_rad
```

The GUI labels remain transverse phase gradients in `rad/µm`; they are not
relabeled as geometric angles. Multiple enabled beams and coherence groups are
allowed, but all enabled channels must have one shared wavelength because the
minimal PR workflow enforces that restriction.

Loading a checkpoint requires the inverse mapping from `BeamStack` to
LaunchPane definitions. That mapping must preserve channel order, names,
wavelengths, powers, waists, centers, phase gradients, phase, and coherence
groups. It belongs in the LaunchPane adapter, is independently tested, and
does not interpret PR physics.

### PR grid panel

The panel constructs the existing `GridSpec`, but PR owns the displayed
defaults. LC aperture and spacing defaults must not be copied implicitly.
Before implementation, the selected default request must be checked by the
same headless PR stability and sampling rules used in tests.

The panel explains that optical propagation is periodic transversely. The PR
material derivative is along x in the present hopping model; there is no PR
material derivative along y. The UI must not imply absorbing material
boundaries or apodization.

Preflight validation must report, rather than conceal:

- insufficient samples per beam waist;
- a propagated beam envelope approaching a periodic boundary;
- an incompatible wavelength set;
- an invalid or unstable normalized material timestep;
- invalid grid sizes or spacings.

Physics formulas should remain in PR-owned pure helpers. The GUI may present
their results but must not duplicate propagation, diffraction-width, or
timestep equations. A general finite-beam aperture helper may be added under
`lcprop.pr` if the existing two-beam-specific report cannot represent the
selected beam stack.

### Evolution panel

The controls map directly to `PRSolverOptions` and `BackendSpec`:

| GUI label | Model field |
|---|---|
| Material steps | `Nt` |
| Normalized timestep | `dt_normalized` |
| Optical substeps per z slice | `optical_substeps` |
| Backend | `backend` |
| Precision | `precision` |

`Nt` means steps in the next fresh or continuation segment. The UI displays
cumulative accepted steps and cumulative normalized material time separately.
The timestep preflight uses the implemented discrete stability guard rather
than a duplicated paper or legacy-code expression.

### Results panel

The shared workspace displays the fields and diagnostics already defined by
`pr_result_to_run_data()`:

- input and output optical intensity;
- initial and final normalized `E`;
- initial and final `E(z,x,y)` stacks;
- final PR-driving intensity stack;
- power, status, grid, launch, backend, and stability diagnostics.

The initial vertical slice reports progress text and normalized material time
at every accepted material boundary. It does not require live conversion of
partial PR fields to `RunData`; completed or cancelled results are displayed
when the worker returns. A PR live-state adapter may be added later if users
need live field visualization.

## Request Construction

`request_adapter.py` constructs exactly:

```text
PRRunRequest(
    grid=<PR grid panel GridSpec>,
    beams=<enabled LaunchPane BeamStack>,
    material=<PR material panel PRMaterialSpec>,
    solver=<evolution panel PRSolverOptions>,
    backend=<evolution panel BackendSpec>,
    initial_A=None,
    initial_E=None,
)
```

Fresh runs always use `initial_A=None` and `initial_E=None`. Continuation state
comes only from `PRTimeDependentCheckpoint`; the GUI must not copy `E_current`
into an unrelated fresh request.

The request summary records all physical and numerical values, beam phase
gradients and coherence groups, backend/precision, normalized timestep, and
the conservative timestep limit reported by PR validation.

## Execution Flow

The PR window creates:

```text
LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,))
```

A fresh run is dispatched by exact material/workflow identity through
`run_registered()`. `WorkflowWorker` runs that callable in a `QThread`, and
`CancellationToken` provides cooperative stop at accepted material-time
boundaries. The finished handler consumes the returned `RunnerResult.run_data`
directly; it does not call the LC-oriented generic result converter.

Only one worker may run at a time. Configuration and checkpoint actions are
disabled during execution. Closing the window requests cancellation and joins
the worker thread using the existing cooperative shutdown pattern.

## Checkpoint and Continuation Flow

Every completed or cancelled PR result supplies a
`PRTimeDependentCheckpoint`. The window retains the latest checkpoint and
enables **Save Checkpoint**. Saving and loading use the shared
`save_run_checkpoint()` and `load_run_checkpoint()` functions, which select the
PR-owned codec through `pr/pr_timedependent` identity.

On load, the window must:

1. reject any non-PR checkpoint;
2. populate grid, material, solver, backend, and beam controls from the saved
   request;
3. retain `E_initial`, `E_current`, and `A0` only inside the loaded checkpoint;
4. show completed steps and normalized material time;
5. enable continuation after `validate_pr_continuation()` succeeds.

Changing a control does not silently delete the loaded checkpoint. The window
revalidates it, disables **Continue** while incompatible, and shows the first
incompatible field. Reverting the control may make the checkpoint usable
again.

Continuation uses the current **Material steps** value as
`additional_steps`. A small PR-owned continuation callable invokes
`continue_pr_timedependent()` and applies `pr_result_to_run_data()` before
returning to the GUI. This does not require adding continuation fields to the
generic `WorkflowOperation` contract.

## Error Handling

Request construction and preflight errors are shown in the console and a
concise status message; no worker is started. Worker exceptions preserve their
traceback in the console. Unsupported checkpoint identity, schema errors,
incompatible continuation, unavailable CuPy, and stability-guard failures are
reported distinctly.

The GUI must never automatically reduce `dt_normalized`, alter an aperture,
recenter a beam, change a coherence group, or convert phase gradients to
geometric angles.

## Test Strategy

### Request and panel tests

- Default panels construct a valid `PRRunRequest`.
- Every material, solver, backend, grid, and beam field maps with the correct
  units and sign.
- Nonzero x/y transverse phase gradients survive GUI construction exactly.
- Multiple coherence groups and the shared-wavelength restriction are tested.
- Applying a saved request to the controls and rebuilding it is lossless.
- The PR defaults pass stability, aperture, and sampling preflight.

### Window and execution tests

- The PR window starts without constructing `LCMaterial`, `BiasSpec`, or an LC
  workflow request.
- The window registers only `PR_TIMEDEPENDENT_OPERATION`.
- A tiny CPU run completes in the worker and displays PR `RunData`.
- Stop returns a cancelled result and a resumable checkpoint.
- Progress updates occur on the GUI thread and remain monotonic.
- Window shutdown cancels and joins an active worker.

### Persistence and continuation tests

- Save/load uses the shared dispatcher and PR codec identity.
- A loaded request repopulates every control without unit or sign changes.
- Incompatible edits disable continuation without discarding the checkpoint.
- Disk-loaded continuation matches an uninterrupted run exactly.
- Loading an LC checkpoint is rejected by the PR window with a clear message.

### Compatibility tests

- Existing `LCPropMainWindow` smoke, request, execution, stop, continuation,
  persistence, and result-view tests remain unchanged and pass.
- Existing LC application startup behavior remains unchanged.
- Shared `Workspace`, `WorkflowWorker`, and LaunchPane behavior remain covered
  independently of either main window.

## Incremental Implementation Sequence

1. Add PR panels, request construction, PR defaults, inverse beam mapping, and
   request round-trip tests. Do not start background execution yet.
2. Add `PRMainWindow`, registered fresh execution, progress text, stop,
   shutdown, and final/cancelled `RunData` display.
3. Add retained checkpoint state, shared save/load dialogs, control hydration,
   compatibility indication, and continuation.
4. Add the standalone application module and optional `lcprop-pr` packaging
   entry point, then run both PR and LC GUI suites.

Each step should be independently reviewable. No step should refactor
`LCPropMainWindow` merely to make the two windows look structurally alike.

## Acceptance Criteria

The vertical slice is complete when a user can launch the PR application,
configure a physically valid request, run and stop it, inspect PR fields and
diagnostics, save the accepted `E` checkpoint, close the application, reload
that checkpoint, continue it, and obtain the same final state as an
uninterrupted headless run.

The existing LC GUI must behave identically before and after the milestone.
The implementation must preserve the architectural physics boundary:

```text
Beam definitions -> generic launch -> optical fields A
    -> PR source -> PR state E -> PR response
    -> advance_prepared_response()
```

The PR GUI is a consumer of that boundary, not a new implementation of PR or
optical physics.
