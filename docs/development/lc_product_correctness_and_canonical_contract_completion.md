# LC Product Correctness and Canonical-Contract Completion

## Scope

This bounded completion milestone starts from Product commit
`42be41a4b488b5100991c30f4443fb8d6e35826a`. It closes the release-critical
LC correctness and presentation findings from the read-only Product audit
without adding equations, models, backends, remote workflows, retention
policies, or research capabilities.

## Corrected Product contracts

- The existing selected optical boundary now reaches every actual midpoint
  propagation in local self-consistent static execution. Periodic remains the
  established no-op; Sponge and Tukey retain their shared definitions.
- Stationary soliton and existence requests now reject nonperiodic boundaries.
  The GUI displays a disabled Periodic boundary during those workflows and
  restores the propagation-workflow selection afterward.
- TD continuation compatibility now includes the optical boundary.
- LC execution rejects unequal enabled channel wavelengths because the current
  propagation and material-response path has one wavelength kernel.
- LC volume z coordinates now identify the accepted slice midpoints retained
  in those arrays.
- Static execution completion is distinguished from scientific convergence in
  Product diagnostics and GUI status.
- Solver, persistence, stationary-boundary, and Slurm-resource controls are
  enabled only when applicable. LC rejects GPU Slurm resources because its
  production scientific backend is NumPy.
- Bounded end-to-end static and TD regressions qualify the existing NumPy
  float32/complex64 path against float64/complex128. The GUI remains float64.

The canonical current scientific contract is
`docs/science/lc_model_contracts.md`.

## Scientific non-change

No LC governing equation, material coefficient, bias relation, effective-index
mapping, director update, TD integrator, eigensoliton equation, branch
continuation rule, split-step operator, boundary definition, tolerance, or
launch/focus calculation changed. The source changes route an already-declared
boundary, reject combinations the existing algorithms cannot represent,
correct retained coordinates, and make status/applicability truthful.

## Explicit exclusions

The milestone does not add Fast/Full retention, an LC runtime estimator, CuPy
production dispatch, TD/soliton Slurm transport, per-channel wavelengths,
input-screen editing, spectral stability, nonperiodic stationary physics,
result export, Research migration, or PR/fanning work.

## Validation

Validation is local and CPU-only. Focused regressions cover every corrected
defect, followed by affected LC workflow/GUI/persistence/product suites, syntax
compilation, Product-relative documentation links, whitespace, and the full
Product suite.

Recorded validation:

- initial focused completion regressions: `9 passed`;
- affected LC workflow, GUI, persistence, product, soliton, and existence
  suites after final corrections: `222 passed, 1 skipped`;
- persistence/transport and focused contract rerun: `113 passed`;
- complete LCProp Product suite: `1665 passed, 77 skipped`;
- syntax compilation: passed;
- Product documentation/link tests: passed;
- candidate whitespace scan and `git diff --check`: passed.

The skips are existing conditional backend/platform tests. No CUDA device,
cluster, remote system, or scheduler was accessed.
