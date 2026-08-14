# LCProp Status

**Checkpoint:** August 2026

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
- NumPy/CuPy backend and precision selection;
- angular-spectrum propagation with explicit optical substepping;
- shared execution, cancellation, progress, and presentation products;
- explicit workflow-operation and checkpoint-codec composition;
- reusable GUI workers, workspace, beam editing, and result views.

### Liquid crystal

- fixed-director and local self-consistent static propagation;
- time-dependent director evolution;
- stationary soliton and soliton-existence workflows;
- sequential and parallel parameter sweeps;
- LC checkpoint persistence and continuation;
- standalone LC PySide6 application.

### Photorefractive

- time-dependent normalized hopping-model workflow;
- fixed-intensity and coupled-static solvers;
- memory-bounded streaming static propagation;
- partition-independent scattering representation;
- image-amplification and coherent two-beam validation helpers;
- PR checkpoint persistence and continuation;
- standalone PR PySide6 application;
- isolated transverse-reference model for multidimensional transport studies.

The transverse-reference model is deliberately separate from the production
scalar PR workflows. Its presence does not change the production PR equation.

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

- Execution is local and explicitly composed; LC compatibility methods remain
  available.
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
