# PR Product-Conversion Optimization — Stage 01

**Date:** 2026-08-26

**Branch:** `feature/pr-second-order-static`

**Base Git SHA:** `d6043b61cb3ba9e19071ba583d0b2eabfc9cdeda`

**Status:** Local and H200 validation complete; ready for pre-commit review

## Objective and Frozen Boundary

This stage profiles and optimizes the canonical conversion from completed
full-transverse PR results to material-neutral `RunData`. The material solver,
zero-flux closure, convergence rules, optical propagation, deterministic
scattering cache, replay, diagnostics, and scientific product definitions are
unchanged.

The accepted Stage-03 H200 result established a 1024² product-conversion
baseline of **17.3424 s**, after a **52.4582 s** workflow. The objective here is
to reduce that post-workflow latency without changing any product value.

## Canonical Call and Data Flow

```text
PR_TRANSVERSE_STATIC_OPERATION
  -> run_pr_transverse_static(request)
     -> PRTransverseStaticRunResult
        (A, psi, source, and residual arrays cross the GPU/host result boundary)
  -> pr_transverse_static_result_to_run_data(result)
     -> reconstruct P, E_x, E_y from psi
     -> project E_active
     -> construct input/output intensity
     -> construct x-z and y-z products and selected material planes
     -> construct one coherence-aware far-field FFT
     -> construct log and carrier-masked far-field products
     -> assemble fields, curves, diagnostics, coordinates, and metadata
     -> RunData
```

The TD adapter follows the same host-side state-reconstruction path through
`pr_transverse_result_to_run_data()` but does not construct the extended static
presentation products.

At workflow completion the canonical result arrays are NumPy arrays. The
post-workflow adapter therefore performs **zero device-to-host transfers, zero
transferred bytes, and no device synchronization**. For the representative
single-channel 80×1024² float32/complex64 static result, the preceding result
boundary performs seven array conversions: two 8 MiB optical fields and five
320 MiB real volumes, approximately **1,616 MiB** in total. Those transfers are
not part of the measured 17.3424 s product-conversion interval.

## Local Profile

The local benchmark uses Python 3.12.13, NumPy 2.5.1, and SciPy 1.18.0. It
constructs deterministic 80-plane float32/complex64 result objects at the
requested transverse size and runs the complete canonical static adapter. It
measures the state reconstruction, far field, intensity construction, and
remaining assembly separately. The scattering-on control changes the
deterministic state but not the conversion contract.

### 512², 80 material slices, three repetitions

| Case | Path | Median total | State reconstruction | Far field | Intensity | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Scattering off | baseline | 2.1971 s | 2.1305 s | 0.0063 s | 0.0039 s | — |
| Scattering off | candidate | 2.0278 s | 1.9543 s | 0.0075 s | 0.0042 s | 1.083× |
| Scattering on | baseline | 2.2099 s | 2.1359 s | 0.0065 s | 0.0038 s | — |
| Scattering on | candidate | 1.9012 s | 1.8645 s | 0.0073 s | 0.0039 s | 1.162× |

State reconstruction accounts for roughly 97% of the local baseline. Far-field
and input/output-intensity construction together account for about 0.5%.
Remaining field, curve, coordinate, log, mask, and diagnostic assembly is
small. The variation between scattering-on and scattering-off speedups is
ordinary local timing variability; the optimization does not inspect the
scattering specification.

### 256² control

The candidate deliberately retains the original direct reconstruction below
512². Three-repeat medians differed by approximately ±4%, with interleaved
samples overlapping. This is a no-change control rather than a speedup claim.

## Large-Array and Temporary Audit

For 80×1024² float32 data, each real volume is 320 MiB and each complex64
volume is 640 MiB. The adapter necessarily retains reconstructed `psi`, `P`,
`E_x`, and `E_y` products (1.25 GiB total), plus the separately constructed
`E_active` product (320 MiB). The direct whole-volume reconstruction also
creates full-volume complex FFT work arrays and real inverse-transform
temporaries. These work arrays are the dominant avoidable peak-host-memory
pressure.

The retained implementation processes independently defined z planes using a
chunk target of 2 MiB of real input data, subject to a minimum of one complete
plane. A 512² float32 plane is approximately 1 MiB, so two planes achieve the
2 MiB target. A 1024² float32 plane is approximately 4 MiB, so the one-plane
chunk necessarily exceeds the target because the implementation does not
subdivide a transverse plane. The four required output volumes remain, but FFT
temporaries are bounded to one small longitudinal chunk rather than the full
80-plane volume. Exact peak RSS is allocator- and FFT-implementation-dependent
and was not claimed from the local process; the temporary-size reduction
follows directly from the array shapes.

## Retained Optimization

`lcprop.pr.transverse.products` now reconstructs presentation state through a
private chunked wrapper around the unchanged authoritative
`state_from_potential()` function for planes of at least 512² cells.

The wrapper preserves:

- the same per-plane FFT implementation and operation order;
- original float32/float64 dtypes;
- exact axes, shapes, normalization, units, and metadata;
- the whole-volume uniform-field shortcut;
- the direct path for 2-D, invalid, small, and locally uniform inputs.

If any proposed chunk is spatially uniform while the complete volume is not,
the wrapper falls back to the original whole-volume call. This avoids changing
the transport helper's deliberate uniform-field roundoff behavior.

Random float32 and float64 comparisons, the mixed-uniform fallback, and a
complete static-adapter comparison are bit-for-bit equal for `psi`, `P`,
`E_x`, `E_y`, all 21 field products, far-field products, coordinates, curves,
dtypes, and diagnostic keys/values.

## Opportunity Ranking

| Opportunity | Payoff | Complexity | Semantic risk | Disposition |
| --- | --- | --- | --- | --- |
| Single-transfer reuse | None in adapter | Low | Low | Already at host boundary |
| Slice before transfer | None in adapter | Low | Low | Result is already host-backed |
| Lazy/requested-only products | Potentially high | Medium | Medium | Deferred; requires request/API policy |
| Far-field transform reuse | Negligible locally | Low | Low | One FFT already |
| Intensity reuse | Negligible locally | Low | Low | No important duplication found |
| Profile before transfer | None in adapter | Low | Low | Already host-backed and sparse |
| Selected plane before transfer | None in adapter | Low | Low | Already host-backed and views used |
| Avoid duplicate host copies | Medium memory | Low | Medium | `E_active` alias rejected due writable-alias risk |
| Dtype preservation | Correct today | Low | Low | No promotions found |
| Low-copy dataclass assembly | Small | Low | Medium | Existing arrays passed without constructor copies |
| Asynchronous transfers | None here | High | Medium | Rejected |
| Pinned host memory | None here | High | Medium | Rejected |
| Batched transfers | Outside adapter | Medium | Medium | Deferred to result-boundary work |
| Serialization changes | Potential artifact benefit | High | High | Out of scope |
| Custom kernels | No host-product justification | High | High | Rejected |
| Bounded z-chunk reconstruction | High time/memory | Low | Low | Retained |

Using SciPy worker-thread FFTs, changing FFT libraries, omitting material
volumes, and replacing `E_active` with a writable alias were rejected because
they would change numerical values, public product availability, or ownership
semantics.

## Validation

The retained change has passed:

- focused chunk/equivalence tests: **6 passed**;
- presentation, transverse workflow, static workflow, and transport tests:
  **72 passed**;
- additional PR product, execution, persistence, marching, and GPU-prepared
  tests: **58 passed, 32 skipped**;
- complete PR selection excluding one unrelated timing-sensitive GUI assertion:
  **353 passed, 57 skipped, 1 deselected**.

The skips are unavailable CuPy hardware or external saved-state fixtures. No
tolerances were changed. No scientific workflow output was altered. Complete
suite invocations intermittently failed one timing-sensitive GUI cancellation
assertion: cancellation was observed in the predictor/corrector rather than the
optical z-march. The same test passed in isolation, and an earlier complete
candidate run passed **354 tests with 57 skips**. The final complete selection
with only that assertion deselected passed with the counts above. No GUI or
cancellation code was modified, and the product adapter is not on the failing
execution path.

## H200 Commissioning

The bounded GPU Validation Benchmark completed on 2026-08-26. Benchmark job
`2900754` completed on `pax009` with exit `0:0` in 4:21. Focused validation job
`2900772` completed on the same node with exit `0:0` in 0:16.

The allocated device was an NVIDIA H200 with 143,771 MiB reported memory. The
jobs used CUDA 12.9, CuPy 13.6.0, and requested and resolved the CuPy backend.

### 1024² performance

| Case | Baseline | Candidate | Speedup | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Accepted Stage-03 result | 17.3424 s | 10.3527 s | 1.675× | 40.30% |
| Controlled scattering off | 17.5747 s | 10.6729 s | 1.647× | 39.27% |
| Controlled scattering on | 17.6492 s | 10.9181 s | 1.617× | 38.14% |

The same-job controlled comparisons used three repetitions and showed narrow,
nonoverlapping timing ranges. The improvement therefore clearly exceeded
run-to-run noise and was not specific to scattering-enabled data.

These values measure post-workflow product conversion. They are not
device-transfer timings and must not be interpreted as whole-workflow
speedups. The product adapter received the canonical NumPy result and performed
zero device-to-host transfers, zero transferred bytes, and no device
synchronization. The preceding workflow-result boundary remained separate, as
described above.

### Numerical equivalence and memory

All public product fields, coordinates, curves, dtypes, metadata, and
diagnostic values were bit-for-bit identical between the direct and chunked
paths. All four accepted workflow array hashes matched. Workflow status,
iteration counts, residuals, replay diagnostics, and optical-power drift were
unchanged. The focused H200 validation suite passed **44 tests**.

Peak GPU use after the workflow was 19.08 GiB. Peak RSS for the complete
benchmark job was 10.17 GiB.

The accepted physical fixture retained its existing
`not_converged / coupled_line_search_failed` status identically. This
commissioning validated product conversion; it did not establish scientific
convergence of that particular fixture.

The checksum-verified local commissioning record is retained at
`results/pr_product_conversion_gpu_validation/commissioning_report.md`. The
retrieved result artifacts are not part of the source candidate. macOS
extended-attribute warnings during archive extraction were harmless packaging
hygiene and did not affect source identity, execution, or numerical results.
