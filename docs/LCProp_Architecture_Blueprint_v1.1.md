# LCProp Architecture Blueprint

## Version 1.1

**Status:** Design baseline for implementation

## 1. Vision

LCProp is a research-grade framework for optical beam propagation in
liquid-crystal media. The architecture separates:

-   experiment description
-   numerical implementation
-   workflows
-   user interfaces

The same computational core should be usable from:

-   PySide6 GUI
-   command line
-   notebooks
-   automated parameter sweeps

------------------------------------------------------------------------

## 2. Design Principles

1.  Experiment description is immutable.
2.  Runtime state is separate from experiment description.
3.  Physics is separated from numerical algorithms.
4.  Workflows orchestrate; they do not implement physics.
5.  Every migrated physics module is validated against the trusted
    reference implementation.
6.  A single beam is a BeamStack containing one BeamChannel.
7.  User-facing quantities are physical quantities (V_bias, P_mW, θ_bc);
    derived quantities (b, bi) are computed unless explicitly
    overridden.

------------------------------------------------------------------------

## 3. Package Layout

``` text
src/lcprop/
    core/
    lc/
    optics/
    workflows/
    io/
    gui_pyside/
    cli/
```

### core

Foundational dataclasses and runtime objects.

-   context.py
-   beams.py
-   requests.py
-   results.py
-   derived.py
-   backend.py

### lc

Liquid-crystal physics.

-   bias.py
-   coupling.py
-   static.py
-   timedependent.py
-   residuals.py

### optics

Optical propagation.

-   launch.py
-   kernels.py
-   propagate.py

### workflows

High-level orchestration.

-   static.py
-   timedependent.py
-   soliton_existence.py
-   soliton_stability.py (planned)

### io

Persistence and replay.

### gui_pyside

Desktop GUI.

### cli

Command-line interface.

------------------------------------------------------------------------

## 4. Core Data Model

### GridSpec

Defines simulation geometry and discretization.

-   Nx, Ny, Nz
-   xaper_um
-   yaper_um
-   dz_um

### LCMaterial

Intrinsic material constants.

-   ne
-   no
-   K
-   delta_epsilon

### BiasSpec

Electrical and director boundary conditions.

-   V_bias
-   theta_bc
-   theta_min
-   theta_max
-   theta_center
-   b_override

### BeamChannel

One optical beam.

-   wavelength
-   waists
-   launch position
-   launch angles
-   phase
-   power_mW
-   coherence group

### BeamStack

Ordered collection of BeamChannel objects.

### SolverOptions

Algorithm choice and tolerances.

### OutputOptions

Persistence options.

### RunRequest

Complete immutable experiment specification.

------------------------------------------------------------------------

## 5. Runtime Objects

### LCContext

Constructed from a RunRequest.

Contains:

-   derived coordinate arrays
-   backend
-   FFT plans
-   propagation plans
-   derived parameters
-   temporary work arrays

LCContext is never serialized as an experiment description.

------------------------------------------------------------------------

## 6. Derived Physics

`core/derived.py`

Responsibilities:

-   compute_b_from_voltage
-   compute_freedericksz_voltage
-   compute_neff
-   resolved_b
-   theta_center

------------------------------------------------------------------------

## 7. LC / Optical Coupling

`lc/coupling.py`

Responsibilities:

-   compute_bi_from_power
-   resolved_bi

Validated against the trusted reference implementation.

------------------------------------------------------------------------

## 8. Workflow Architecture

### Static

    RunRequest
    → build LCContext
    → launch BeamStack
    → self-consistent split-step propagation
    → RunResult

Static propagation means repeated optical propagation with local static
director updates. It does **not** imply frozen intensity or frozen
director.

### Time-dependent

    RunRequest
    → initialize theta
    → time loop
        → propagate optics
        → update director
    → RunResult

### Soliton Existence

Parameterized sweep in optical power and/or bias using the static
workflow.

### Soliton Stability

Launches perturbations from an existence solution and analyzes
propagation stability.

------------------------------------------------------------------------

## 9. Numerical Strategy

Current:

-   local self-consistent split-step propagation

Planned:

-   dual-grid acceleration
-   phase-conjugate bidirectional relaxation (experimental)
-   additional global relaxation strategies

------------------------------------------------------------------------

## 10. Testing Strategy

Each migrated module must satisfy:

1.  unit tests
2.  physics tests
3.  comparison tests against `lc_reference`

Migration pattern:

    reference module
    → port
    → unit test
    → physics test
    → comparison test

------------------------------------------------------------------------

## 11. User Interfaces

All interfaces construct identical RunRequest objects.

-   PySide6 GUI
-   CLI
-   notebooks
-   scripted parameter sweeps

No interface bypasses workflows.

------------------------------------------------------------------------

## 12. Migration Strategy

Reference packages:

-   lc_reference --- frozen migration reference
-   lc --- current validated implementation
-   lcprop --- clean implementation

Architecture improvements are allowed. Physics changes require
validation.

------------------------------------------------------------------------

## 13. Development Roadmap

1.  Core dataclasses ✅
2.  Derived physics ✅
3.  LC coupling ✅
4.  Optics launch
5.  Propagation kernels
6.  Split-step propagation
7.  Static workflow
8.  Save/load/replay
9.  PySide6 GUI
10. Time-dependent workflow
11. Soliton existence
12. Soliton stability

------------------------------------------------------------------------

## 14. Long-Term Goals

-   Modular research framework
-   Reproducible experiments
-   Extensible optics engine
-   Support for future vector propagation
-   Shared architecture with future PRProp
-   Clean separation of physics, numerics, workflows, and UI

This document is the architectural contract for LCProp. Significant
structural changes should be reflected here before implementation.

# Appendix A. System Data Flow

The guiding architectural principle is that **all user interfaces
construct the same immutable experiment description**, and **all
numerical work is performed through workflows operating on an
LCContext**.

``` text
                    USER INTERFACES
     ┌─────────────────────────────────────┐
     │  PySide6 GUI   CLI   Notebook   API │
     └─────────────────────────────────────┘
                      │
                      ▼
               StaticRunRequest
          TimeDependentRunRequest
         SolitonExistenceRequest
                      │
                      ▼
              build_context()
                      │
                      ▼
                  LCContext
      (derived arrays, backend, plans,
       derived parameters, work buffers)
                      │
                      ▼
                 WORKFLOWS
      ┌──────────────┼─────────────────┐
      │              │                 │
      ▼              ▼                 ▼
   Static      Time Dependent   Soliton Existence
      │              │                 │
      └──────┬───────┴─────────────────┘
             ▼
      Physics + Optics Engines
      ┌──────────────────────────────┐
      │ optics.launch                │
      │ optics.propagate             │
      │ lc.static                    │
      │ lc.timedependent             │
      │ lc.coupling                  │
      └──────────────────────────────┘
                     │
                     ▼
                 RunResult
                     │
         ┌───────────┼────────────┐
         ▼           ▼            ▼
      Save/Load   Diagnostics    GUI Display
```

## Architectural Rules

1.  User interfaces never call numerical algorithms directly.
2.  Workflows orchestrate the computation.
3.  Physics modules implement equations.
4.  Numerical modules implement algorithms.
5.  RunResult is the only product returned to the user interface.

------------------------------------------------------------------------

# Appendix B. Core Class Relationships

``` text
                 GridSpec
                     │
                 LCMaterial
                     │
                  BiasSpec
                     │
                 BeamStack
                     │
             SolverOptions
                     │
             OutputOptions
                     │
                     ▼
                RunRequest
                     │
             build_context()
                     │
                     ▼
                 LCContext
                     │
                     ▼
                  Workflow
                     │
                     ▼
                 RunResult
```

## Responsibilities

### GridSpec

Defines simulation geometry and discretization only.

### LCMaterial

Contains intrinsic liquid-crystal constants only.

### BiasSpec

Contains electrical bias and director boundary conditions only.

### BeamChannel / BeamStack

Describe the optical experiment independently of the propagation
algorithm.

### RunRequest

Immutable description of an experiment. It is suitable for
serialization, replay, and reproducibility.

### LCContext

Ephemeral runtime object constructed from a RunRequest. It owns derived
arrays, backend resources, FFT plans, and temporary work buffers.

### Workflow

Coordinates the computation by invoking optics and liquid-crystal
modules. It contains essentially no physics itself.

### RunResult

Owns the scientific products of the computation together with sufficient
metadata to reproduce or replay the experiment.

------------------------------------------------------------------------

These relationships define the primary architectural boundaries of
LCProp. Future extensions (vector propagation, dual-grid methods,
photorefractive modules, additional workflows) should preserve these
boundaries whenever possible.
