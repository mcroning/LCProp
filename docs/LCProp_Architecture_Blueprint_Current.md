# LCProp Architecture Blueprint (Current Status)

## Core Package Layout

``` text
src/lcprop/

    core/
        backend.py
        context.py
        beams.py
        requests.py
        results.py
        derived.py

    lc/
        bias.py
        coupling.py
        static.py
        timedependent.py
        residuals.py

    optics/
        launch.py
        kernels.py
        propagate.py

    workflows/
        static.py
        timedependent.py
        soliton_existence.py
        soliton_stability.py      # planned

    io/
        save.py
        load.py
        replay.py

    gui_pyside/
        ...

    cli/
        ...
```

------------------------------------------------------------------------

# Core Data Model

The architecture separates the experiment description from the runtime
numerical state.

## Experiment description (immutable)

These dataclasses completely describe the experiment requested by the
user.

### GridSpec

Describes the computational grid.

Owns:

-   Nx
-   Ny
-   Nz
-   xaper_um
-   yaper_um
-   dz_um

Includes validation of numerical dimensions and physical sizes.

### LCMaterial

Describes intrinsic liquid-crystal properties.

Owns:

-   material name
-   ne
-   no
-   K
-   delta_epsilon

Contains no numerical state.

### BiasSpec

Describes the electrical bias and director boundary conditions.

Owns:

-   V_bias
-   theta_bc
-   theta_min
-   theta_max
-   theta_center
-   optional b_override

Contains no grid information.

### BeamChannel

Describes one optical beam.

Owns:

-   wavelength
-   waist_x
-   waist_y
-   launch position
-   launch angle
-   phase
-   power_mW
-   optional power_fraction
-   coherence group

### BeamStack

An ordered collection of BeamChannel objects.

A single beam is represented as:

``` text
BeamStack(channels=[beam])
```

The package is therefore multichannel by construction.

### SolverOptions

Describe numerical tolerances and algorithm choices.

They specify:

-   tolerances
-   iteration limits
-   solver method

They do not contain physics.

### OutputOptions

Describe output and persistence choices.

Examples:

-   run directory
-   save slices
-   save full arrays

### RunRequest

A RunRequest is the complete immutable description of one experiment.

Example:

``` text
StaticRunRequest

    GridSpec
    LCMaterial
    BiasSpec
    BeamStack
    StaticSolverOptions
    OutputOptions
```

The GUI, CLI, and notebooks all construct these same request objects.

No workflow should modify a RunRequest.

## Runtime Objects

### LCContext

LCContext is **not** part of the experiment description.

It is constructed from a RunRequest and contains runtime numerical
objects such as:

-   derived grid arrays
-   backend arrays
-   FFT plans
-   propagation plans
-   derived physical parameters
-   temporary work arrays

LCContext exists only while a workflow is executing.

# Derived Physics

General liquid-crystal relationships belong in:

``` text
core/derived.py
```

Responsibilities include:

-   electrical bias parameter `b`
-   Freedericksz voltage
-   effective refractive index
-   resolved electrical bias
-   default theta center

These relationships are independent of any workflow.

# LC / Optical Coupling

Optical-material coupling belongs in:

``` text
lc/coupling.py
```

Responsibilities:

-   `compute_bi_from_power()`
-   `resolved_bi()`

This module contains the validated optical normalization and LC coupling
relationships.

# Architecture Principle

``` text
Experiment description
        ↓
RunRequest
        ↓
build_context()
        ↓
LCContext
        ↓
Workflow
        ↓
RunResult
```

This separation is a fundamental design rule.

# Implemented Components

-   GridSpec
-   LCMaterial
-   BiasSpec
-   BeamChannel
-   BeamStack
-   StaticRunRequest
-   StaticSolverOptions
-   OutputOptions
-   `core/derived.py`
-   `lc/coupling.py`

These are covered by unit tests, including comparison against the
trusted `lc_reference` implementation for the optical coupling
coefficient.

# Planned Next Steps

1.  Port `optics/launch.py`.
2.  Port propagation kernels.
3.  Port split-step propagation.
4.  Port the static LC solver.
5.  Build the static workflow.
6.  Add save/load/replay.
7.  Build the PySide6 GUI.
8.  Port the time-dependent workflow.
9.  Port the soliton existence workflow.
10. Implement the soliton stability workflow.

Future solver strategies should include:

-   local self-consistent split-step propagation
-   dual-grid acceleration
-   phase-conjugate bidirectional relaxation (experimental)
