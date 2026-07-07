# LCProp Status
Version 1.0 Architecture
July 2026

## Overview

LCProp has completed migration of the numerical engine from the validated
research code into the new architecture.

The architecture now separates:

- Core data model
- LC physics
- Optical propagation
- Numerical algorithms
- Workflows

The numerical algorithms are largely unchanged from the validated code.
Most migration effort has consisted of API cleanup and architectural
separation rather than numerical modification.

---

# Current Package

```
lcprop/

    core/
        backend.py
        context.py
        derived.py
        grid.py
        beams.py
        requests.py
        results.py

    lc/
        bias.py
        coupling.py

    optics/
        launch.py
        splitstep.py

    algorithms/
        fft_y.py
        thomas.py
        thomas_fast.py
        theta_cn.py
        theta_picard.py
        theta_cn_zcoupled.py
        static_relax.py
        td_zmarch.py

    workflows/
        static.py
```

---

# Implemented

## Core

✓ Experiment data model

- GridSpec
- LCMaterial
- BiasSpec
- BeamChannel
- BeamStack
- RuntimeGrid

---

## Optical system

✓ Multichannel launch

- channel stacks
- coherent / incoherent propagation
- arbitrary beam count
- normalized optical power

✓ Split-step propagation

- nonlinear phase
- Fourier propagation
- midpoint intensity
- multichannel support

---

## LC system

✓ Derived coefficients

- b
- bi
- Freedericksz voltage
- effective refractive index

✓ Bias preparation

- cosine seed
- theta stack builder

---

## Numerical algorithms

Imported directly from validated code.

- FFT operators
- Thomas solvers
- CN theta solver
- Picard correction
- Static relaxation
- TD z-march
- Z-coupled CN

---

## Workflows

Implemented

✓ fixed_theta

✓ local_self_consistent

using the new workflow architecture.

---

# Validation

Current automated tests:

27 passing

Coverage includes

- core objects
- derived physics
- beam objects
- launch normalization
- split-step propagation
- bias generation
- workflow smoke tests
- self-consistent static workflow

---

# Architecture

Current workflow configuration

```python
StaticWorkflowOptions(
    strategy="fixed_theta" |
             "local_self_consistent",

    theta_solver="none" |
                 "picard_cn",

    optics_solver="splitstep",

    coupling="frozen" |
             "self_consistent",
)
```

This is intended to grow naturally into future workflow strategies without
changing the public API.

---

# Remaining Work

## Immediate

- Time-dependent workflow
- Existence-curve workflow
- Stability workflow

---

## Physics

- Dual-grid workflow
- Global z-coupled static workflow
- Bidirectional static solver
- Newton static solver

---

## User Interface

- PySide6 GUI
- Job management
- Progress monitoring
- Result browser

---

# Guiding Principles

LCProp separates

- experiment description
- numerical algorithms
- workflows

Numerical algorithms remain experiment-independent.

Workflows coordinate algorithms but contain minimal numerical logic.

The public API is centered on request objects and workflow execution rather
than low-level numerical routines.

# Design Goals

LCProp is intended to be a long-lived research platform.

Primary goals are:

- Preserve validated numerical algorithms unchanged whenever possible.
- Isolate physics from numerical implementation.
- Make all workflows reproducible from serializable request objects.
- Treat multichannel propagation as the fundamental optical representation.
- Allow future extensions (dual grid, vector optics, multiple wavelengths, global z-coupled solvers) without redesigning the core architecture.



# LCProp Status (July 2026)

## Core
✓ Context objects
✓ Request/result objects
✓ Grid builder
✓ Derived physics
✓ Backend abstraction

## LC
✓ Bias builder
✓ Coupling coefficients

## Optics
✓ Beam model
✓ Multichannel launch builder
✓ Split-step propagation
✓ Optical eigenmode utilities

## Algorithms
✓ FFT operators
✓ Thomas solvers
✓ CN theta solver
✓ Picard corrector
✓ Static relaxation
✓ TD z-march
✓ Z-coupled CN

## Products
✓ Diagnostics

## Runtime
✓ Runtime component builder

## Workflows
✓ Fixed-theta propagation
✓ Static relaxation
✓ Time-dependent
✓ Soliton
✓ Soliton existence curve

## Tests

**33 automated tests passing**

Coverage includes:

- Core objects
- Runtime builders
- Optics
- Algorithms
- Workflow integration
- Soliton existence workflow

## Sanity validation

The migrated LCProp package successfully executes representative research workflows:

✓ 3 mm static self-consistent propagation (3 μm waist)

✓ 3 mm time-dependent propagation

✓ Soliton existence curve (0.5, 1.0, 2.0 mW)

These reproduce the expected workflow behavior of the validated LC package.

## Remaining work

- Stability workflow
- Dual-grid support
- Global z-coupled workflow
- PySide6 GUI