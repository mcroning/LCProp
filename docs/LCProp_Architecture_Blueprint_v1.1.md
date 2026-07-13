# LCProp Architecture Blueprint

## Version 1.1

**Status:** Authoritative architecture document and source of truth

**Current implementation checkpoint:** July 2026

## 1. Purpose and vision

LCProp is a research framework for optical beam propagation in liquid-crystal
media. It separates immutable experiment description, runtime numerical state,
physics, numerical algorithms, workflow orchestration, scientific products,
and user interfaces.

The same computational core is intended to support the PySide6 application,
scripts, notebooks, automated sweeps, and future command-line interfaces
without duplicating physics or bypassing workflows.

## 2. Architectural rules

1. Experiment descriptions are immutable dataclasses.
2. Runtime arrays and backend resources are separate from experiment
   descriptions.
3. Physics modules implement physical relationships; numerical modules
   implement algorithms.
4. Workflows orchestrate complete computations and return result objects.
5. User interfaces build request objects and call runners or workflows rather
   than numerical kernels directly.
6. A single optical beam is represented as a one-channel `BeamStack`.
7. User-facing inputs are physical quantities. Derived quantities such as
   `b` and `bi` are computed unless explicitly overridden.
8. Migrated physics and numerical behavior must remain covered by regression,
   comparison, and workflow tests.

## 3. Current package structure

```text
src/lcprop/
    core/        immutable specifications, requests/results, grids, backends
    lc/          liquid-crystal bias and coupling physics
    optics/      launch-field construction and split-step propagation
    algorithms/  FFT, Thomas, CN/Picard, relaxation, and z-march algorithms
    workflows/   static, time-dependent, soliton, and sweep orchestration
    products/    workflow-independent data products and diagnostics
    runners/     execution boundary, currently LocalRunner
    gui/         PySide6 application, panels, workspace, and views
```

Persistence services, a command-line interface, and additional execution
backends remain future extensions; they are not represented as implemented
packages in this layout.

## 4. Core experiment model

### 4.1 Geometry and material

- `GridSpec` owns `Nx`, `Ny`, `dz_um`, `z_length_um`,
  `x_aperture_um`, and `y_aperture_um`.
- `LCMaterial` owns intrinsic liquid-crystal constants.
- `BiasSpec` owns voltage, director boundary conditions, clamps, and optional
  derived-parameter overrides.

### 4.2 Optical launch model

`BeamChannel` describes one scalar input field: wavelength, power, waists,
position, tilt, phase, director-coupling weight, and `coherence_group`.
`BeamStack` is an ordered tuple of channels.

LCProp's native coherence rule is:

```text
total intensity = sum over groups(abs(sum of fields in that group) ** 2)
```

Channels with the same `coherence_group` interfere coherently. Channels with
different group strings are mutually incoherent. The legacy stack-wide
`BeamStack.coherence` flag remains a compatibility input: an old coherent
stack normalizes to one shared group, and an old incoherent stack normalizes to
one distinct group per channel. Explicit per-channel groups are authoritative.

`theta_weight` is an intensity-side relative optical-coupling multiplier.
Channels in one coherent group must have equal weights so that the coherent
cross terms have an unambiguous director coupling.

### 4.3 Requests and results

Concrete requests include `StaticRunRequest`, `TimeDependentRunRequest`,
`SolitonRequest`, `SolitonExistenceRequest`, and
`ParameterSweepRequest`. They compose grid, material, bias, beams, solver,
runtime, and output choices as appropriate.

In this document, *RunRequest* is architectural shorthand for these concrete
LCProp request types; it does not name a separate Python class.

Concrete workflow results carry scientific arrays and metadata. The products
layer converts them to workflow-independent `RunData` for visualization and
diagnostics.

## 5. Runtime and numerical layers

Runtime builders validate requests and construct ephemeral objects such as
`RuntimeGrid`, `BiasResult`, `LaunchResult`, and `RuntimeComponents`.
These contain derived arrays, coefficients, launch fields, kernels, and
algorithm operators. They are not the serialized experiment description.

The numerical layers currently include:

- Gaussian multichannel launch and power normalization;
- grouped coherent intensity and weighted director-driving intensity;
- split-step optical propagation;
- FFT and Thomas-solver utilities;
- Crank-Nicolson and Picard director updates;
- static-relaxation orchestration primitives;
- time-dependent z-marching;
- a z-coupled CN algorithm used by the current time-dependent machinery.

Having an algorithm module does not by itself mean that every proposed
high-level workflow using that algorithm is complete.

## 6. Workflow layer

### 6.1 Fixed-theta propagation

The `fixed_theta` static strategy propagates the optical channel stack through
a prepared director field without self-consistent director updates.

### 6.2 Local self-consistent static propagation

The `local_self_consistent` strategy alternates optical propagation and local
static director relaxation through the workflow boundary.

### 6.3 Time-dependent propagation

`run_timedependent()` builds runtime components and advances the optical field
and director state with the time-dependent z-march.

### 6.4 Soliton workflow

`run_soliton()` solves the stationary optical/director problem. The
`SolitonRequest.refine_transverse` option lets `LocalRunner` apply the
optional transverse eigensoliton polishing stage. That refinement currently
supports exactly one optical channel.

### 6.5 Soliton-existence workflows

LCProp provides a dedicated soliton-existence request/workflow and a
parameter-sweep path currently specialized for soliton power and used by the
GUI. Continuation and sequential or parallel execution are available where
supported by the sweep request.

### 6.6 Stability workflow

Soliton stability analysis is planned. It must consume a request and a prior
solution through the same workflow/result architecture rather than calling
algorithms from a user interface.

## 7. User-interface and execution boundary

All user interfaces construct identical LCProp request objects. External GUI
packages such as LaunchPane must be integrated through LCProp-owned adapters
and must never bypass the request/workflow layer.

The current PySide6 application builds requests in `LCPropMainWindow`, sends
them through `LocalRunner`, converts results to `RunData`, and displays them
in the workspace. A future background runner may replace synchronous local
execution without changing request or result contracts.

## 8. Package boundaries: LaunchPane and LCProp

LaunchPane remains a separate standalone reusable package and repository.
The integration dependency will be one-way: LCProp will import LaunchPane;
LaunchPane must never import LCProp. The LaunchPane-to-LCProp adapter belongs
in LCProp.

### LaunchPane owns

- launch-plane visualization;
- beam editing;
- `BeamDefinition` and `BeamStackDefinition`;
- beam-stack serialization;
- graphical beam placement and editing.

### LCProp owns

- `GridSpec` and numerical discretization;
- LC material and bias;
- propagation engines;
- workflows;
- *RunRequest* and result objects;
- diagnostics, persistence policy, and application orchestration.

### Integration contract

- LCProp `GridPanel` is authoritative for `x_aperture_um` and
  `y_aperture_um`.
- LCProp will pass those two physical aperture dimensions to LaunchPane.
- LaunchPane does not need `Nx` or `Ny`.
- LaunchPane will return a `BeamStackDefinition` to the host application.
- An LCProp-side adapter will convert enabled LaunchPane beams into
  `lcprop.core.beams.BeamChannel` and `BeamStack` objects.
- Disabled LaunchPane beams will remain in the editor but will be filtered out
  by the adapter.
- The same `coherence_group` string means mutually coherent beams; different
  strings are mutually incoherent.
- LCProp supports this grouped-coherence model natively.
- All application interfaces still construct the same LCProp request objects
  and do not bypass the workflow layer.

Planned dependency flow:

```text
LaunchPane
    |
    | BeamStackDefinition
    v
LCProp adapter
    |
    | BeamStack
    v
LCProp requests, workflows, and engines
```

This contract keeps LaunchPane reusable and prevents physics-application
dependencies from leaking into the editor package.

## 9. Products, diagnostics, and persistence

The products layer separates scientific fields, curves, geometry, and
diagnostics from GUI widgets. The current GUI consumes these neutral products
through image, longitudinal, and curve views.

`OutputOptions` expresses output intent, but complete save/reload,
reproducible-run manifests, and workspace persistence protocols remain planned.
Persistence must serialize request and result data, not ephemeral runtime
objects.

## 10. Validation strategy

Validation proceeds from the narrowest stable boundary outward:

1. core-object and validation tests;
2. physics and algorithm regression tests;
3. launch and propagation tests;
4. workflow integration tests;
5. products and GUI smoke tests;
6. representative research-scale checks where practical.

Physics changes require explicit numerical evidence. Architectural cleanup must
not silently alter validated equations or normalization conventions.

## 11. Planned extensions

Near-term implementation order is tracked only in
[`development_plan.md`](development_plan.md); the current checkpoint is tracked
only in [`STATUS.md`](STATUS.md). Planned extensions must preserve the request →
workflow → result boundary. The presence of a low-level algorithm does not, by
itself, mark a corresponding high-level workflow as implemented.

## Appendix A. System data flow

```text
User interface or script
          |
          v
LCProp request object
          |
          v
Runner / workflow
          |
          v
Runtime builders
          |
          v
Physics + numerical algorithms
          |
          v
LCProp result object
      |                 |
      v                 v
Products,          Planned request/result
diagnostics,       persistence layer
and display
```

This document is the architectural contract for LCProp. Significant structural
changes should be reflected here and reconciled with the current implementation.
