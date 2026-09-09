# H200 Commissioning: Biased Full-Transverse Linearized PR Reference

**Classification:** Commissioned\
**Date:** 2026-09-09\
**Source SHA:** `482acef47f677c9c23faffc4bc6aa6005ee06ee9`\
**Source subject:** `Generalize biased transverse PR reference backend`
**Slurm job:** `3470108`

## Scope and immutable source

This run commissioned only the isolated frozen-intensity reference operator.
It did not invoke or change production static/time-dependent PR workflows,
Image Amplification, soliton code, GUI, persistence, transport codecs, or
workflow registration.

The source was packaged with `git archive --format=tar.gz` from the exact
committed SHA. The archive was unpacked inside the scheduler-created run
directory and placed first on `PYTHONPATH`.

| Artifact | SHA-256 |
| --- | --- |
| Source archive | `913bea57f0f5b65780642de688bf5c32ba83bb419846f5250da93177c1e7c622` |
| Commissioning harness | `af781927151d5aeaf2ab409ef8bf1210d3677f39abe15d76c02b4c43f00da168` |
| Scheduler script | `7dbe6111475b50db4979f2b233647c6502c5b40c1f4ca1a0a150d9bc2ecc14c3` |
| Retrieved metrics | `80db97300f8256ea143dbaffb48bbe21461ce3eba75e3a4b6fffe42f385f2e88` |
| Retrieved provenance | `d95f37bae2fcc5667dbf9fb84ea2d47b54a6cb1f5d7ae8993bf116419dc7b251` |

Remote launch archive:

`/cluster/tufts/cglab/mcroning/lcprop_runs/pr-biased-linearized-reference-482acef/launch/lcprop-482acef47f677c9c23faffc4bc6aa6005ee06ee9.tar.gz`

Local retrieval:

`results/pr_biased_linearized_reference_h200_commissioning_2026-09-09/job_3470108`

After checksum-verified retrieval, the temporary extracted remote source tree
was removed. The compact metrics/provenance evidence was retained, and the
immutable launch inputs and source archive were retained according to
commissioning policy. Remote commissioning evidence was therefore preserved;
only the temporary extracted source tree was deleted.

## Scheduler and environment

The one authorized job completed without retry.

| Item | Value |
| --- | --- |
| State / exit | `COMPLETED` / `0:0` |
| Elapsed | 14 seconds |
| Node | `pax008` |
| GPU | NVIDIA H200, 143771 MiB |
| Device | `cuda:0`; `CUDA_VISIBLE_DEVICES=0` |
| Driver | 575.57.08 |
| CUDA module/runtime/driver API | 12.9.0 / 12090 / 12090 |
| Python | 3.10.4 |
| NumPy | 2.1.0 |
| CuPy | 13.6.0 |
| Backend | requested `cupy`, resolved `cupy` |

CuPy FFT smoke returned `complex128` from float64 and `complex64` from
float32. Standard error was empty.

## Deterministic multi-mode equivalence

The fixture contained nonzero mean, x variation, y variation, an oblique
mode, and nonzero applied bias. Acceptance was declared before submission:

- float64: `rtol=3e-13`, `atol=3e-13`;
- float32: `rtol=2e-5`, `atol=2e-6`.

### CuPy versus NumPy float64

| Quantity | Relative L2 | Maximum absolute |
| --- | ---: | ---: |
| Response kernel | `1.020e-16` | `6.206e-17` |
| Denominator | `0` | `0` |
| delta psi | `2.611e-16` | `2.602e-18` |
| delta E_x | `3.242e-16` | `5.204e-18` |
| delta E_y | `2.914e-16` | `1.735e-18` |
| delta P | `3.796e-16` | `2.429e-17` |
| Mean-current perturbation | `0` | `0` |
| Mean-intensity perturbation | `0` | `0` |

The reconstructed potential mean was `-5.034e-21`. Returned dtypes were
float64/complex128 on both backends.

### CuPy versus NumPy float32

| Quantity | Relative L2 | Maximum absolute |
| --- | ---: | ---: |
| Response kernel | `9.465e-8` | `6.664e-8` |
| Denominator | `0` | `0` |
| delta psi | `1.463e-7` | `1.397e-9` |
| delta E_x | `1.519e-7` | `2.794e-9` |
| delta E_y | `1.918e-7` | `7.567e-10` |
| delta P | `2.085e-7` | `9.313e-9` |
| Mean-current perturbation | `0` | `0` |
| Mean-intensity perturbation | `0` | `0` |

The reconstructed potential mean was `1.476e-11`. Returned dtypes were
float32/complex64 on both backends; no CuPy float32 promotion occurred.

NumPy float32 also passed against the float64 reference. The largest output
relative L2 difference was `2.002e-6` for delta P, and the largest output
maximum-absolute difference was `5.566e-8`, also for delta P. The denominator
had a `7.238e-8` relative L2 difference; its larger `0.0580` maximum absolute
difference reflects the magnitude of high-frequency denominator entries and
passed the combined tolerance.

## Analytic modes and limits

Both precisions passed all five analytic cases:

1. zero-bias x mode;
2. zero-bias y mode;
3. positive-bias x mode;
4. negative-bias x mode;
5. biased oblique mode.

For float64, the largest maximum-absolute field error was `1.481e-17`. For
float32 it was `1.505e-8`. The relative metrics are less informative near
analytic zeros, so acceptance used the declared combined relative/absolute
tolerances.

The float64 CuPy response kernel passed bias reversal by bitwise
`cp.array_equal` equality against the conjugated opposite-bias kernel. This
was an exact equality check, not a tolerance comparison. The unbiased
screened-Poisson limit passed with `5.551e-17` maximum-absolute difference.

## Gauge, Nyquist parity, and batch

NumPy and CuPy masks were identical, every unresolved response coefficient
was identically zero, and each expected joint derivative-null count matched:

| Shape | Expected/actual unresolved | Reconstructed mean psi |
| --- | ---: | ---: |
| 8 x 10 | 4 / 4 | `-6.353e-23` |
| 8 x 9 | 2 / 2 | `0` |
| 7 x 10 | 2 / 2 | `-9.680e-23` |
| 7 x 9 | 1 / 1 | `-1.613e-22` |

The batch input and output shapes were both `(2, 32, 35)`. FFT axes were
`(-2, -1)`, so the leading dimension remained a batch of independent frozen
planes. All batch fields passed float64 equivalence; the largest relative L2
difference was `3.862e-16` and the largest maximum-absolute difference was
`2.429e-17`.

## Operation and synchronization audit

Instrumentation observed exactly one forward `fft2` and four inverse
`ifft2` calls. Result provenance independently reported the same counts.
There was no Newton, line search, PCG, or iterative material solve.

Wrapping the solver module's `asnumpy` boundary observed exactly one call: the
documented Boolean input-validity transfer. Static source inspection found no
`.item()` or `.get()` calls in the solver. Comparison code necessarily copied
results after each solve, and benchmark timing explicitly synchronized the
null stream; neither is part of the solver hot path.

## Runtime

The reported first-call timings are the first measurements for each grid and
precision after CUDA/CuPy initialization and after the earlier scientific
commissioning gates. They are not process-cold CUDA/CuPy startup measurements.
Before each grid/precision timing block, the harness cleared the CuPy FFT plan
cache and freed all blocks held by the default device and pinned memory pools.
Warmed times are medians of five repeats. Every interval used
`cupy.cuda.Stream.null.synchronize()` before and after timing.

| Precision | Grid | First call (ms) | Warm median (ms) |
| --- | ---: | ---: | ---: |
| float64 | 256 x 256 | 18.062 | 1.162 |
| float64 | 512 x 512 | 17.464 | 1.162 |
| float64 | 1024 x 1024 | 16.324 | 1.170 |
| float32 | 256 x 256 | 22.036 | 1.213 |
| float32 | 512 x 512 | 15.215 | 1.185 |
| float32 | 1024 x 1024 | 17.942 | 1.217 |

The near-flat warmed values indicate launch/backend-resolution overhead is a
large fraction of these small isolated runs; no production-speedup claim is
made.

## GPU memory

CuPy pool total is a high-water-style measure because the pool retains freed
intermediate allocations. `nvidia-smi` process residency also contains the
CUDA context and libraries.

| Precision | Grid | Pool total after | Pool growth | Live pool used | `nvidia-smi` process |
| --- | ---: | ---: | ---: | ---: | ---: |
| float64 | 256 x 256 | 18.51 MiB | 17 MiB | 6.50 MiB | 552 MiB |
| float64 | 512 x 512 | 74.02 MiB | 68 MiB | 26.00 MiB | 608 MiB |
| float64 | 1024 x 1024 | 296.03 MiB | 272 MiB | 104.00 MiB | 830 MiB |
| float32 | 256 x 256 | 9.25 MiB | 8.5 MiB | 3.25 MiB | 542 MiB |
| float32 | 512 x 512 | 37.01 MiB | 34 MiB | 13.00 MiB | 570 MiB |
| float32 | 1024 x 1024 | 148.02 MiB | 136 MiB | 52.00 MiB | 682 MiB |

The float64 pool totals are about 1.48 times the development live-array
estimates of 12.5, 50, and 200 MiB. That modest excess is consistent with FFT
workspace and pool behavior. Pool growth increased exactly fourfold for each
doubling of transverse grid size in both precisions, and float32 used half the
float64 pool growth. This validates `O(Nx Ny)` scaling. No unexpected
multi-GiB or longitudinal-history allocation appeared.

## Nonlinear Taylor oracle

The low-level nonlinear CuPy residual produced RMS errors

`[1.89289e-7, 4.73222e-8, 1.18306e-8, 2.95764e-9]`

under successive epsilon halving. Observed orders were

`[2.0000000006, 1.9999999988, 2.0000000011]`.

Classification: **CuPy Taylor validated**.

## Verdict and next lifecycle

All blocking gates passed on an actual NVIDIA H200. The isolated operator is
commissioned for NumPy/CuPy float64 and float32 execution. Production
integration may now be proposed as a separate, narrowly reviewed development
milestone; it is not performed or authorized by this record.
