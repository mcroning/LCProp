# LCProp Status

**Checkpoint:** September 2026

**Architecture source of truth:**
[`architecture/LCProp_Target_Architecture.md`](architecture/LCProp_Target_Architecture.md)

## Current architecture

LCProp has completed the ownership migration from an LC-centric package to a
peer-material architecture:

- `lcprop.core`, `lcprop.optics`, `lcprop.products`, `lcprop.runners`,
  `lcprop.persistence`, and the shared parts of `lcprop.gui` provide
  material-neutral infrastructure.
- `lcprop.lc` canonically owns LC specifications, director physics,
  algorithms, workflows, persistence, products, operations, and application.
- `lcprop.pr` independently owns PR specifications, material state and
  evolution, workflows, persistence, products, operations, and application.
- Historical LC import locations remain compatibility shims where required;
  canonical definitions are not duplicated there.

The shared optical contract is a prepared multiplicative complex response
consumed by `advance_prepared_response()`. Material packages remain
responsible for source construction, material evolution, and conversion of
their state to that optical response.

## Implemented capabilities

### Shared platform

- physical runtime grids and Fourier coordinates;
- multichannel Gaussian launch and grouped coherence;
- focus-defined, collimated, and uniform launch intent through LaunchPane;
- NumPy/CuPy backend and precision selection;
- angular-spectrum propagation with explicit optical substepping;
- periodic, distance-scaled sponge, and discrete Tukey optical boundaries;
- shared execution, cancellation, progress, and presentation products;
- explicit workflow-operation and checkpoint-codec composition;
- reusable GUI workers, workspace, beam editing, and result views.

### Liquid crystal

- fixed-director and local self-consistent static propagation;
- time-dependent director evolution;
- stationary soliton and soliton-existence workflows;
- sequential and parallel parameter sweeps;
- LC checkpoint persistence and continuation;
- standalone LC PySide6 application;
- in-application Help plus current Quick Start and User Guide.

### Photorefractive

- all eight production choices formed by static/time-dependent evolution,
  reduced x-only/full-transverse transport, and fully nonlinear/linearized
  material response;
- exact-modal time evolution for both linearized transport models;
- coupled nonlinear and analytic linearized static solvers;
- canonical partition-independent scattering where request schemas support it;
- Image Amplification and carrier-resolved two-beam diagnostics;
- PR experiment/checkpoint persistence and continuation;
- Local and Slurm execution with explicit NumPy/CuPy semantics;
- Fast/Full remote retention, exact Fast longitudinal cuts, bounded MPR
  previews, TD accepted-state curves, and compact preview movies;
- structured progress and advisory runtime/resource planning;
- standalone PR PySide6 application with in-application Help.

Physical model choice is independent of validation and hardware commissioning
status. The [User Guide](user/user_guide.md) records those distinctions and
the [PR model contracts](science/pr_model_contracts.md) define the current
equations and profile boundaries.

## Validation policy

The repository contains focused tests for shared boundaries and both material
packages. Run the complete local suite with:

```bash
python -m pytest -q
```

GPU execution is validated separately on CUDA hardware. Test counts are not
embedded in this status document because they change as coverage grows;
committed validation records and continuous local review should report the
exact command and result for each milestone.

## Current boundaries

- LC supports Local execution for displayed workflows; LC Slurm execution is
  currently limited to canonical static propagation. PR exposes Local and
  Slurm execution for its registered production operations.
- LC and PR have separate user applications rather than one material-selector
  window.
- Prepared responses currently cover the supported scalar multiplicative
  optical interaction. Future nonlocal, vector, or operator-valued responses
  require their own justified design.
- External dynamic plugin discovery, a universal material-state hierarchy,
  and a public third-party plugin SDK are intentionally deferred.
- Research-scale runs, raw outputs, unpublished references, and historical
  experiment notebooks are maintained outside the public package.

See [`development_plan.md`](development_plan.md) for forward-looking package
work.
