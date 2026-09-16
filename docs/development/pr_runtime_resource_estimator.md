# PR Runtime and Resource Estimator

## Scope and status

This milestone adds a planning-only estimator to the PR GUI. It does not run a
scientific kernel, alter a request, select a backend, submit a job, or make a
convergence claim. The estimator covers all eight production model cells and
reports broad ranges for local NumPy and H200/CuPy computation where a useful
calibration or conservative projection exists.

The implementation is based on Product commit
`82452729703e78eec9838ac063c0ffd581187fe7`. No material equation, optical
propagator, solver tolerance, retention policy, transport schema, or execution
default changed.

## User interface

The PR window has a **Run Planning** tab. **Estimate current configuration**
builds the same immutable request used by execution and passes it to a pure
estimator. Image Amplification is reduced to its prepared base PR request for
planning, matching the existing local-cost guard. The output reports:

- model cell and `Nx × Ny × Nz`;
- local Mac/NumPy-proxy and H200/CuPy runtime ranges;
- confidence and dominant cost;
- peak GPU and host-memory ranges;
- Fast and Full result-package ranges;
- a local/H200 recommendation and explicit qualifications.

Changing a scientific control marks a completed estimate stale. Request
construction failures are displayed without starting a worker. Runtime ranges
exclude scheduler queueing, source staging, retrieval, and local product
conversion. The displayed local number is deliberately called a NumPy proxy,
not a promise for every Mac generation.

## Calibration v1

`src/lcprop/pr/assets/runtime_estimator_calibration_v1.json` is installed as
Product package data. It is a small, human-readable, versioned calibration,
not an opaque fitted model. Every measured scalar carries the exact committed
source artifact and a deterministic JSON-pointer or CSV-reduction extraction
rule. Tests reconstruct all 20 measured values from those sources.

| Coverage | Retained measurement |
|---|---|
| Reduced static, NumPy | 256² × 100, 20 configured passes: nonlinear 5.043 s from Gaussian x and 5.530 s from Gaussian 45°; linearized 2.852 s from Gaussian x and 3.445 s from the broad-beam x fixture |
| Full-transverse linearized TD, NumPy | 256² × 2, three intervals: 0.30389 s total; 0.08201 s material; 0.11718 s optical |
| Full-transverse linearized TD, H200 | Same request: 0.03357 s warmed scientific total; 0.01050 s material; 0.00883 s optical; 576 MiB sampled peak |
| Full-transverse nonlinear TD, NumPy | 256² × 2, three intervals: 0.34524 s total; 0.09641 s material; 0.10742 s optical |
| Full-transverse nonlinear TD, H200 | Same request: 0.42977 s initialization-bearing total; 0.02015 s material; 0.01533 s optical; 572 MiB sampled peak |
| Full-transverse nonlinear static memory | Documented 1024² × 400 float64 bounded-memory projection: 1.5–4 GiB GPU and 25–45 GiB host |

The two H200 totals have different timing semantics. Linearized Case B is a
warmed measurement. Nonlinear TD is the first scientific measurement after
explicit CUDA/CuPy initialization. The model retains that overhead instead of
silently treating both as warmed throughput.

The asset separates three categories deliberately:

- **Measured evidence:** timings, sampled GPU process peaks, and retained Fast
  package sizes, each with an exact source and extraction rule.
- **Derived projections:** FFT-work, grid, precision, direct/continuation stage,
  and retained-array scaling. The nonlinear-static memory anchor is itself a
  committed architecture projection rather than a measured H200 peak.
- **Planning policy:** range safety factors, compression/container allowances,
  confidence labels, and the 2-minute/15-minute Local/H200 advisory thresholds.

Those policy constants widen or classify estimates; they are not presented as
measurements or scientific tolerances.

## Transparent scaling model

For two-dimensional optical and full-transverse Fourier work the estimator
uses

`Nx Ny log2(Nx Ny) × Nz × passes × optical_substeps`.

Material intervals and optical passes are counted separately. A TD request
with `Nt` accepted intervals plans `Nt + 1` complete optical passes, including
the final replay. Static requests use the configured maximum coupled-pass
count as a direct-stage budget, not as a workflow-wide upper bound. Float32
uses the explicit derived 0.65 timing factor and half-width real/complex
storage.

Reduced static timing scales the retained end-to-end NumPy measurements.
Reduced TD lacks a direct retained workflow calibration; it uses one quarter
of the full-transverse material component while retaining the complete optical
component and is labeled **Low** confidence. Full-transverse linearized static
is likewise projected from fixed-count TD components and labeled Low.

Full-transverse static requests receive a scenario decomposition. The direct
stage includes one initial optical evaluation, up to `max_backtracks + 1`
optical trials per coupled iteration, and a final replay. If two or more
channels share a coherence group, production may restart after a specific
direct line-search failure through visibility `(0, 0.25, 0.5, 0.75, 1)`.
The three intermediate stages require both incoherent and coherent optical
propagations for every logical optical evaluation. The planning envelope
therefore includes the direct attempt plus all five stages and their optical
retry factors, while stating explicitly that activation is not predicted.
Distinct coherence groups and single-channel requests are marked direct-only
because continuation is inapplicable.

Full-transverse nonlinear static remains the least predictable cell. Its
lower bound represents ordinary direct work. Its upper bound applies the
commissioned nonlinear TD component cost through the configured coupled,
Newton, PCG, continuation-stage, and optical-backtracking envelope. It is
intentionally broad because initialization, continuation, and convergence
statistics can dominate runtime. This is a defensive planning envelope, not a
benchmark, convergence oracle, or assertion that the maximum work will occur.

## Memory and result size

Fast and Full sizes are derived from retained-array shapes and precision:

- two complex optical endpoints per channel;
- the three ordinary transverse optical products;
- exact Fast longitudinal cuts;
- at most the established 4 MiB Fast MPR preview;
- three retained real material volumes for TD, four for reduced static, and
  five for full-transverse static;
- bounded metadata/package allowance and a broad compression/container range.

Full-transverse nonlinear static GPU memory follows the plane-local
bounded-memory architecture and therefore has no `Nz` multiplier in its GPU
workspace. Host working memory does retain the full-volume factor. At the
1024² × 400 float64 anchor, direct/non-continuation planning uses the committed
25–45 GiB host projection. A request with shared coherence groups instead uses
the 50–61 GiB direct-fallback/continuation-overlap projection. Both are scaled
by retained-volume bytes for other grids and precisions and are explicitly
identified as extrapolations. Other GPU
ranges include a conservative CUDA/CuPy process floor, plane workspace, and
the material volumes required by the selected workflow. These are planning
ranges; allocator state and FFT plans remain machine dependent.

## Holdout checks

Executable tests treat retained historical measurements as holdouts rather
than rerunning science:

- both reduced-static 256² × 100 measurements lie inside their projected
  ranges;
- the NumPy and H200 linearized TD totals lie inside their ranges;
- the NumPy and H200 nonlinear TD totals lie inside their ranges;
- the 1024² × 400 nonlinear-static estimate reproduces the documented
  1.5–4 GiB GPU and 25–45 GiB host projection exactly;
- increasing optical substeps increases, rather than silently ignoring, the
  optical cost;
- float32 and Fast retention reduce estimated storage monotonically.
- distinct coherence groups produce a direct-only static plan;
- shared groups show both direct-success and continuation-capable scenarios;
- nonlinear and linearized full-transverse static upper ranges include all
  configured stages and line-search optical retries;
- coherent nonlinear-static host memory selects the continuation-overlap
  projection, including precision scaling;
- estimation leaves the request, backend, grid, controls, retention, and
  execution target unchanged.

The static nonlinear holdout is a memory-architecture projection, not a
completed performance measurement. Consequently its runtime confidence stays
Low and H200 is recommended without claiming a precise completion time.

## Optional planning benchmark design

A future **Benchmark this configuration** action is feasible, but is not
implemented here. The safe design is one explicitly requested, cancellable
planning run in a separate temporary output directory. It would time one
optical pass and one model-appropriate material unit, synchronize the selected
backend, sample memory, and project the remaining immutable request. It must
never overwrite or masquerade as a scientific result, submit remotely without
the ordinary approval gate, or use a reduced fixture without labeling the
change. Until that workflow and its provenance are reviewed, the GUI uses only
the retained versioned calibration.

## Accuracy-preset audit

No Preview, Standard, or Publication preset was added. A universal
Publication preset would be scientifically misleading: spatial convergence,
material timestep, coupled-pass convergence, Newton/PCG limits, optical
substeps, and retention needs depend on the model and problem. Preview and
Standard could eventually be per-model convenience actions only if they
populate visible controls and preserve explicit provenance. Publication/high
accuracy must remain a documented convergence study, not a hidden GUI token.

## Local validation

Validation used the repository Python 3.12 environment with the isolated
worktree on `PYTHONPATH`:

- focused estimator and GUI tests: **23 passed**;
- affected estimator, GUI, and main-window tests: **39 passed**;
- complete PR suite: **776 passed, 76 skipped**;
- calibration JSON parsing and source/checksum reconstruction passed;
- candidate syntax compilation passed;
- whitespace inspection and `git diff --check` passed.

The complete suite was run before the final measured-memory/package holdout
assertions were added to the already passing focused test; the final focused
rerun passed all 23 tests. No scientific solver, benchmark, cluster, or CUDA
calculation was run for this remediation.

## Candidate boundary and exclusions

The candidate consists only of the packaged calibration, estimator and GUI
presentation, focused tests, package-data declaration, and this record. It
does not include new Research evidence. Existing committed evidence is read as
calibration provenance; no solver or cluster job was rerun.

Explicitly excluded are production physics, request/persistence/transport
schemas, backend selection semantics, result retention, Slurm configuration,
cluster access, fanning, soliton work, and Milestone 5.
