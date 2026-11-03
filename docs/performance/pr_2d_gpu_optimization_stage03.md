# PR Full-Transverse 2D GPU Optimization — Stage 03

**Date:** 2026-08-26

**Branch:** `feature/pr-second-order-static`

**Baseline Git SHA:** `8b5b303bf493d773d7a6f53c4e9354400b35a43f`

**Status:** Complete; clean benchmark, detailed profile, and focused CuPy gate passed

## Objective and Frozen Boundary

Stage 03 profiles and optimizes the canonical optical pass used by the full
transverse PR workflows. The accepted Stage-01 material solver is frozen. No
Newton, PCG, zero-flux closure, residual, tolerance, convergence, material
cadence, or replay semantics are changed. Stage 02 made no retained production
change; its exact but unproductive Newton/PCG candidate was fully reverted.

The repository was already substantially dirty at entry. Stage 03 is limited
to the optical/scattering replay changes and focused tests identified below.
No commit or push has been made.

## Repository and Canonical Call Path

Entry provenance was:

- branch: `feature/pr-second-order-static`;
- HEAD: `8b5b303bf493d773d7a6f53c4e9354400b35a43f`;
- canonical registered static workflow: `run_pr_transverse_static()`;
- canonical static stage: `_run_pr_transverse_static_at_visibility()`;
- canonical optical march: `lcprop.pr.transverse.workflow._optical_pass()`;
- shared PR slice coupling: `advance_pr_slice_with_midpoint_source()`;
- material-neutral propagation seam: `advance_prepared_response()`;
- backend abstraction: NumPy/CuPy-compatible `xp` selected by `get_backend()`.

The production static call graph is:

```text
PR_TRANSVERSE_STATIC_OPERATION
  -> run_pr_transverse_static
     -> _run_pr_transverse_static_at_visibility
        -> construct RuntimeGrid, launch, and one linear diffraction kernel
        -> construct one deterministic scattering-phase stack (Stage 03)
        -> repeated coupled optical_pass(psi)
           -> _optical_pass
              -> reconstruct E_x/E_y and project E_active
              -> for each material interval z
                 -> advance_pr_slice_with_midpoint_source
                    -> construct entrance PR-driving intensity when needed
                    -> construct one frozen-E half-step response
                    -> advance_prepared_response
                       -> response half step
                       -> FFT -> spectral kernel multiply -> IFFT
                       -> response half step
                    -> construct exit PR-driving intensity
                    -> midpoint source = (I_before + I_after) / 2
                 -> apply canonical scattering phase for this interval
        -> material solve and coupled acceptance
        -> independent final optical replay through accepted psi
  -> pr_transverse_static_result_to_run_data (host presentation products)
```

For the representative request, `Nz=80`, `Nsub=1`, and one optical pass
therefore performs 80 FFT/IFFT propagation pairs, 160 response-screen
applications, 80 response constructions, 80 scattering applications, and 81
PR-driving intensity constructions. The exact pass count is determined by the
unchanged coupled iteration and replay path.

The linear kernel and runtime frequency grid were already constructed once per
workflow stage. No Tukey/window operation exists in this canonical path.
Far-field, logarithmic far-field, and carrier-masked products are constructed
after the canonical result crosses the established host/result boundary; they
are not part of the workflow's `optical_pass_seconds` timer.

## Performance Hazard Found

Before Stage 03, every optical pass regenerated every deterministic canonical
scattering field. In the representative case, each 50 µm material interval is
the sum of twenty-five 2 µm canonical slabs. With 80 intervals, one replay
regenerated 2,000 primitive slabs and correlated 80 full transverse fields.
The same physical realization was regenerated for every coupled trial and the
independent replay even though it depends only on the scattering specification,
physical z interval, grid geometry, dtype, and backend.

This was not an FFT issue: canonical V2 correlation uses a PR-owned separable
Gaussian implementation. It nevertheless incurred repeated full-grid random
field construction, accumulation, filtering, allocation, and Python dispatch.

## Retained Optimization

Stage 03 precomputes the real canonical phase increment for every material
interval once per workflow stage and reuses those phases in all optical passes.
The cached object has shape `(Nz, Nx, Ny)` and the runtime grid's real dtype.

The optimization intentionally does **not** cache a complex phase screen.
Every propagation still evaluates `exp(1j * phase)` and multiplies it into the
field at the same accepted slice boundary. Thus the nonlinear response,
diffraction, scattering ordering, arithmetic at coupling boundaries, and
independent replay remain unchanged. Callers that do not provide a cached
phase retain the former generation path, and no-scattering requests allocate
no cache.

The cache costs:

| Grid | Shape | float32 bytes | MiB |
|---:|---:|---:|---:|
| 256² | 80×256×256 | 20,971,520 | 20 |
| 512² | 80×512×512 | 83,886,080 | 80 |
| 1024² | 80×1024×1024 | 335,544,320 | 320 |

At 1024² this is about 1.75% of the Stage-01 sampled peak of
19,141,623,808 bytes. The H200 clean run measured the cache-construction
scratch and persistent-pool effect separately below.

## Local Numerical Equivalence

Focused tests explicitly compare the cached workflow against the old uncached
path. For deterministic NumPy cases they establish:

- every cached phase plane is bit-for-bit equal to independently regenerated
  canonical phase for the same z interval;
- static final complex field, final potential, midpoint source volume,
  equilibrium residual volume, status, coupled-iteration count, and complete
  iteration records are bit-for-bit identical;
- TD final complex field, final potential, status, and completed material-step
  count are bit-for-bit identical;
- only `Nz` phase increments are generated for the cached run, independent of
  the number of optical replays;
- no-scattering construction returns `None` and preserves the old path.

The focused optical/static suite passed **119 tests with 3 skips** in 18.03 s.
The complete local PR suite passed **348 tests with 57 skips** in 57.30 s.
Skips are unavailable CuPy devices or external saved-state fixtures. Python
syntax compilation, both Slurm shell syntax checks, and scoped
`git diff --check` also pass. The clean H200 production benchmark subsequently
passed the float32 bitwise trajectory gate at all three grid sizes.

## Optical Optimization Opportunity Ranking

| Opportunity | Expected payoff | Complexity | Trajectory risk | Disposition |
|---|---|---|---|---|
| Cache deterministic scattering phases | High for repeated replay with scattering | Small | Very low | Retained; H200 timing and trajectory passed |
| Diffraction propagator caching | None currently | Small | Low | Already once per workflow stage |
| Frequency-grid caching | None currently | Small | Low | Already once in `RuntimeGrid` |
| Window/Tukey caching | None | Small | Low | No window in canonical path |
| FFT plan/workspace specialization | Unknown | Medium | Low–medium | Measure; defer |
| Reuse current spectral field | Potentially medium | Medium | High | Rejected for this stage |
| Eliminate redundant FFT/IFFT pairs | No redundant optical pair identified | Medium | High | Not implemented |
| Reuse far-field diagnostic transform | Small, outside solver timer | Medium | Medium | Deferred |
| Lazy diagnostics/products | Small, outside solver timer | Small–medium | Low | Deferred |
| Work-buffer reuse | Small–medium | Medium | Aliasing risk | Deferred; remaining optical fraction is small |
| Channel batching | Request dependent | Medium | Medium | Existing FFT already batches channels |
| Reduce Python-loop overhead | Unknown | Small–medium | Low | Measure after cache |
| CuPy expression fusion | Unknown | Medium | Medium | Not justified yet |
| `ElementwiseKernel` | Unknown | Medium | Medium | Not justified yet |
| `RawKernel` | Unknown | Large | High | Not justified |
| Custom propagation kernel | Low against FFT cost | Large | High | Rejected |
| Different optical algorithm | Potentially high | Large | Blocking | Out of scope |

## H200 Profiling and Validation Package

The bounded package was uploaded after separate approval. It contained only:

- the three Stage-03 production source files;
- the frozen Stage-01 `static.py` and its focused test as baseline overlay
  dependencies;
- the three focused regression files;
- `pr_2d_gpu_optimization_stage03.py`;
- the accepted Stage-01 profiler helper;
- one 256²/512²/1024² clean-plus-1024²-detailed Slurm script;
- one focused CuPy validation Slurm script.

The clean profiler records synchronized total, material, optical, product, and
memory values. The detailed profiler records optical FFT/IFFT counts and time,
response construction/application, diffraction hop, intensity construction,
scattering application, phase-cache construction, longitudinal slices, and
substeps. Inclusive categories intentionally overlap and are labeled as such.

Two approved jobs ran on an NVIDIA H200 on `pax008`, with CUDA 12.9.0,
CuPy 13.6.0, and Python 3.10.4:

| Purpose | Job | State | Runtime | Exit | Scientific payload |
|---|---:|---|---:|---:|---|
| Clean benchmark + detailed profile | 2846626 | FAILED | 3:21 | 1:0 | Clean 256²/512²/1024² completed; detailed report serialization failed |
| Focused CuPy validation | 2846627 | FAILED | 0:08 | 2:0 | Collection stopped before tests because `PySide6` is absent |
| Corrected detailed-only profile | 2846794 | COMPLETED | 1:04 | 0:0 | 1024² detailed optical profile completed |
| Corrected focused CuPy validation | 2846795 | COMPLETED | 0:12 | 0:0 | 147 passed, 6 saved-state-only skips |

Both jobs passed source checksums, exact import-path verification, H200/CuPy
preflight, and scheduler provenance. Neither failure was in production physics
or propagation. Job 2846626 completed and serialized all three clean cases,
then the detailed profiler used nonexistent `GridSpec.Nz` while calculating a
report-only cache-size field. Job 2846627 included a Qt-dependent presentation
test even though the commissioned GPU environment intentionally lacks PySide6.
No automatic retry occurred. The two harness-only corrections were reviewed
and separately authorized before jobs 2846794 and 2846795 were submitted.

The successful clean results are:

| Grid | Stage-01 total | Stage-03 total | Stage-01 optical | Stage-03 optical | Material | Sampled/pool memory |
|---:|---:|---:|---:|---:|---:|---:|
| 256² | 35.2409 s | 28.3144 s | 8.5316 s | 0.4432 s | 26.3775 s | 1.077 GB pool reserved after run |
| 512² | 34.1921 s | 30.3847 s | 5.1371 s | 0.2420 s | 28.7022 s | 4.308 GB pool reserved after run |
| 1024² | 80.1498 s | 52.4582 s | 30.1687 s | 1.3298 s | 45.1275 s | 20,483,801,088 bytes sampled device use |

Relative to Stage 01, total synchronized time fell by 19.65%, 11.14%, and
34.55% at 256², 512², and 1024². Timed optical passes fell by 94.81%, 95.29%,
and 95.59%. Material time changed by only +0.17%, +0.29%, and +0.33%,
consistent with ordinary run-to-run noise and the frozen material path.

The 1024² sampled device use increased by exactly 1,342,177,280 bytes
(1.25 GiB, 7.01%) relative to the Stage-01 sampled peak. The persistent cache
is 320 MiB; the remaining increase is CuPy-pool retention of temporary arrays
used during cache construction. Straight quadratic projection adds about
5 GiB to the earlier 2048² estimate, raising it from approximately 76.2 GB to
approximately 81.2 GB. That remains feasible on the 141 GiB H200 but should
not be assumed safe on an 80 GB device without a separate measurement.

The unchanged optical propagation count is determined exactly from the
accepted pass histories:

| Grid | Optical passes | Slices/substeps | FFT2 | IFFT2 | Response applications |
|---:|---:|---:|---:|---:|---:|
| 256² | 9 | 720 | 720 | 720 | 1,440 |
| 512² | 5 | 400 | 400 | 400 | 800 |
| 1024² | 28 | 2,240 | 2,240 | 2,240 | 4,480 |

The corrected 1024² detailed profile records the following inclusive values:

| Operation | Calls | Inclusive time |
|---|---:|---:|
| Complete optical pass | 28 | 2.0947 s |
| Midpoint slice advancement | 2,240 | 1.4717 s |
| Prepared-response advancement | 2,240 | 0.5864 s |
| PR-driving intensity construction | 2,268 | 0.5754 s |
| FFT2 within optical-pass context | 2,296 | 0.1983 s |
| IFFT2 within optical-pass context | 2,352 | 0.3456 s |
| Linear diffraction hop | 2,240 | 0.3856 s |
| Half-response construction | 2,240 | 0.1726 s |
| Response-screen application | 4,480 | 0.1786 s |
| Scattering-screen application | 2,240 | 0.1078 s |
| One-time scattering-cache construction | 1 | 1.0803 s |
| Per-interval phase construction inside cache | 80 | 1.0783 s |

These timers deliberately synchronize around nested operations and therefore
overlap; their sum is not a wall-time decomposition. The 56 extra FFTs and 112
extra IFFTs beyond the 2,240 propagation pairs are exactly the two FFT/four
IFFT field reconstructions performed once per optical pass. The clean 1.3298 s
optical timer is the authoritative performance number; the instrumented
2.0947 s value is for attribution only.

Code-path audit and the detailed counters confirm no host scalar extraction
inside the optical z loop. Device-to-host conversion remains at the established
result boundary. The clean 1024² product adapter took 17.3424 s after the
52.4582 s workflow, for 69.8006 s through product construction. This aggregate
includes state reconstruction, far field, logarithmic far field,
carrier-masked output, fields, curves, and diagnostics; the far-field portion
was not isolated independently.

All four stored arrays at every grid size are bit-for-bit equal to the accepted
Stage-01 H200 references: complete output complex field, selected potential,
selected midpoint source, and selected equilibrium residual. Status,
termination reason, coupled iterations, backtracks, Newton/PCG counts,
authoritative diagnostics, replay diagnostics, and power drift are also
identical. This directly validates the production CuPy trajectory despite the
separate focused-test collection failure.

Retrieved, checksum-verified artifacts are stored under:

- `results/pr_2d_gpu_optimization_stage03/pr-2d-gpu-optimization-stage03-2846626`;
- `results/pr_2d_gpu_optimization_stage03/pr-2d-gpu-optimization-stage03-validation-2846627`;
- `results/pr_2d_gpu_optimization_stage03/pr-2d-gpu-optimization-stage03-detailed-2846794`;
- `results/pr_2d_gpu_optimization_stage03/pr-2d-gpu-optimization-stage03-validation-2846795`.

The local profiler now uses the physical `z_length_um / dz_um` interval count,
and the validation launch list no longer includes the Qt-only presentation
test. These are harness-only corrections. The corrected exact H200 test list
passed locally with **119 passed and 34 unavailable-CuPy skips**, then passed on
the H200 with **147 passed and 6 saved-state-only skips**. Python and all three
Slurm scripts pass syntax checks. All corrected-job artifacts were retrieved
and checksum-verified.

## Files in the Stage-03 Boundary

- `src/lcprop/pr/workflow.py` — optional cached phase application and phase-stack construction;
- `src/lcprop/pr/transverse/workflow.py` — one-cache TD reuse;
- `src/lcprop/pr/transverse/static_workflow.py` — one-cache coupled-static reuse;
- `tests/test_pr_scattering.py` — exact phase-stack construction;
- `tests/test_pr_transverse_production.py` — exact TD trajectory and generation count;
- `tests/test_pr_transverse_static_workflow.py` — exact static trajectory and generation count;
- `scripts/checks/pr_2d_gpu_optimization_stage03.py` — clean/detailed H200 profiler;
- `scripts/checks/pr_2d_gpu_optimization_stage03_h200.sbatch` — bounded benchmark launcher;
- `scripts/checks/pr_2d_gpu_optimization_stage03_detailed_h200.sbatch` — corrected detailed-only retry launcher;
- `scripts/checks/pr_2d_gpu_optimization_stage03_validation_h200.sbatch` — focused CuPy gate;
- `docs/performance/pr_2d_gpu_optimization_stage03.md` — this record.

`src/lcprop/pr/workflow.py` already contained unrelated interpolation research
changes before Stage 03. Only the phase-cache hunks belong to this milestone;
future review/staging must preserve that distinction.

## Stage-04 Decision Gate

The retained cache is strongly supported: it removed 95.59% of timed optical
replay cost at 1024² while preserving the complete H200 trajectory bit for bit.
The remaining clean optical fraction is only 1.33 s, or about 2.5% of workflow
time. Custom `ElementwiseKernel`, `RawKernel`, custom FFT, mixed precision, and
an alternative propagation architecture are therefore not justified.

If another performance stage is desired, it should profile the 17.34 s host
product-conversion path and distinguish state reconstruction, far-field FFT,
field packaging, and avoidable diagnostic traffic. That work should preserve
the canonical result and requested products; it is a presentation/product
optimization, not another optical-propagator or material-solver stage.
