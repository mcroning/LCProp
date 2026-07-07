# LCProp

**LCProp** is a research library for self-consistent optical beam propagation in nonlinear liquid crystals.

LCProp was conceived as a general liquid-crystal beam-propagation tool. Its first major research application is liquid-crystal spatial solitons, but the architecture remains general.
It provides a clean, modular architecture for developing, validating, and running numerical simulations of liquid-crystal beam propagation, stationary self-consistent modes (solitons), and time-dependent nonlinear evolution.

The package is designed for scientific research, reproducibility, and long-term maintainability.

---

## Features

Current capabilities include

- **Multichannel optical beam model**
  - arbitrary number of optical channels
  - independent wavelength per channel
  - coherent or incoherent propagation
  - physically normalized optical power

- **Liquid-crystal model**
  - Fréedericksz bias field generation
  - optical coupling coefficients
  - derived physical parameters

- **Optical propagation**
  - split-step Fourier propagation
  - channel-stack architecture
  - arbitrary propagation distance

- **Theta solvers**
  - Crank–Nicolson
  - Picard correction
  - z-coupled CN solver

- **Scientific workflows**
  - fixed-theta propagation
  - static self-consistent relaxation
  - time-dependent evolution
  - stationary soliton solver
  - soliton existence curves using continuation

---

# Architecture

LCProp separates physics, algorithms, and workflows.

```
                Request
                   │
                   ▼
        Runtime Component Builder
                   │
      ┌────────────┴────────────┐
      ▼                         ▼
 Optics Engine             Theta Engine
      │                         │
      └────────────┬────────────┘
                   ▼
              Workflows
                   │
                   ▼
          Diagnostics / Results
```

Package layout

```
src/lcprop/

    core/
        context
        requests
        results
        backend
        grids
        derived physics

    lc/
        bias
        coupling

    optics/
        launch
        split-step propagation

    algorithms/
        CN solvers
        Picard correction
        TD z-march
        FFT
        Thomas solvers

    products/
        diagnostics

    workflows/
        static
        timedependent
        soliton
        soliton_existence
```

---

# Example

```python
from lcprop.core.context import GridSpec, LCMaterial, BiasSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.requests import (
    StaticRunRequest,
    StaticSolverOptions,
    OutputOptions,
)

from lcprop.workflows import run_static

request = StaticRunRequest(
    grid=GridSpec(
        Nx=256,
        Ny=256,
        dz_um=5.0,
        x_aperture_um=75.0,
        y_aperture_um=100.0,
        z_length_um=3000.0,
    ),
    material=LCMaterial(
        ne=1.7,
        no=1.5,
        K=7e-12,
        delta_epsilon=13.0,
    ),
    bias=BiasSpec(theta_bc=0.0),
    beams=BeamStack(
        channels=(
            BeamChannel(
                wavelength_um=0.633,
                power_mW=1.0,
                waist_x_um=3.0,
                waist_y_um=3.0,
            ),
        )
    ),
    solver=StaticSolverOptions(),
    output=OutputOptions(),
)

result = run_static(request)
```

---

# Current status

The current release includes

- clean layered architecture
- multichannel optics engine
- runtime component builder
- static workflow
- time-dependent workflow
- soliton workflow
- soliton existence workflow

The package currently contains **33 automated tests**, including workflow integration tests covering all major computational workflows.

---

# Roadmap

Planned additions include

- stability-analysis workflow
- dual-grid implementation
- global z-coupled solver
- PySide6 graphical interface
- documentation and tutorials
- benchmark and validation suite

---

# Design philosophy

LCProp was developed around several guiding principles.

- Separate **physics** from **numerical algorithms**.
- Separate **algorithms** from **scientific workflows**.
- Treat multichannel propagation as the fundamental optical representation.
- Keep workflows readable and easy to modify.
- Make scientific computations reproducible and testable.
- Prefer clear architecture over short-term convenience.

---

# Project status

This repository represents the first complete implementation of the LCProp architecture.

The package is intended to serve as the foundation for future liquid-crystal beam-propagation research and related graphical tools.