# LCProp Status

**Checkpoint:** July 2026

**Architecture source of truth:** [`LCProp_Architecture_Blueprint_v1.1.md`](LCProp_Architecture_Blueprint_v1.1.md)

## Current implementation

LCProp has a working request/workflow/result architecture with separated core
models, liquid-crystal physics, optics, numerical algorithms, products,
runners, and a PySide6 application.

### Core and optics

- Immutable grid, material, bias, beam, and request models are present;
  workflow-specific result models carry the computed products.
- Optical launch uses channel stacks for scalar and multichannel cases.
- Native grouped coherence is implemented: fields within one
  `coherence_group` interfere coherently and group intensities add
  incoherently.
- Legacy stack-wide coherent/incoherent inputs normalize into grouped
  coherence.
- Split-step propagation, launch normalization, director coupling, and
  NumPy/CuPy-style backend handling are covered by tests.

### Workflows

Implemented and exercised by automated tests:

- fixed-theta propagation;
- local self-consistent static propagation;
- time-dependent propagation;
- soliton solving;
- soliton-existence curves and the soliton-power parameter-sweep path.

The soliton request has optional transverse refinement. `LocalRunner` invokes
the transverse polishing stage when requested; that stage currently supports
one optical channel.

### PySide6 application

The GUI currently provides experiment, physics, beam, grid, solver, sweep, and
results tabs. It builds the same LCProp requests used by non-GUI callers,
dispatches them through `LocalRunner`, and displays neutral `RunData`
products in image, longitudinal, and curve views.

Current limitations:

- the Beam tab still contains the original single-beam controls;
- runs execute synchronously on the GUI thread;
- there is no durable workspace save/reload protocol;
- robust background-job, progress, cancellation, and spinning-wheel handling
  remain planned.

## Validation checkpoint

The complete LCProp test suite passed on 2026-07-13:

```text
86 passed
```

Coverage includes core models, grouped coherence, optics, numerical algorithms,
runtime builders, all implemented workflows, products, GUI panels/views,
workspace behavior, precision policy, and sequential/parallel sweep paths.

The repository also contains `scripts/checks/run_3mm_workflows.py` for
representative 3 mm static, time-dependent, and soliton-existence checks. Its
presence is confirmed; this documentation update does not claim a new execution
of that manual check script.

## LaunchPane integration checkpoint

LaunchPane exists as a separate reusable package with
`BeamDefinition`, `BeamStackDefinition`, serialization, and
`LaunchPlaneWidget`. LCProp now has the grouped-coherence semantics needed to
represent LaunchPane beam groups.

Still planned:

- thin LCProp-owned LaunchPane adapter;
- filtering disabled editor beams during adaptation;
- embedding `LaunchPlaneWidget` in the Beam tab;
- one-way authority for aperture dimensions from LCProp `GridPanel`;
- one-beam equivalence and multibeam end-to-end request tests.

## Other remaining major work

- save/reload protocols and workspace persistence;
- soliton stability workflow;
- responsive/background execution, progress, cancellation, and spinning-wheel
  handling;
- dual-grid support;
- complete global z-coupled workflow.

Detailed implementation order is maintained in
[`development_plan.md`](development_plan.md).
