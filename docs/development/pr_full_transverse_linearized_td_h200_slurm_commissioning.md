# Full-Transverse Linearized PR TD H200 / Slurm Commissioning

**Date:** 2026-09-11
**Classification:** Commissioned
**Production commit:** `f865c5e30c5233c8f5b89e478f0bde68b28b2763`
**Slurm job:** `3593204` (`COMPLETED`, exit `0:0`, elapsed `00:00:12`)
**Node / GPU:** `pax009` / NVIDIA H200

## Provenance and boundary

The production source was cloned from a complete Git bundle, checked out
detached at the exact production commit, and required an empty
`git status --porcelain` before execution. The separately supplied launch
inputs were:

- source bundle SHA-256:
  `3a5cee151b63dcfe5a494e02233a49a5855f214fe0809077b5043e449e88c15d`;
- commissioning harness SHA-256:
  `0e58fce9bb6d4558c2ff3631eac82b0575592abbca68ab4c29416adc83072fbf`;
- one-off scheduler script SHA-256:
  `1758e2cba66115de4448575faf21c5763ae6596476c0775da6b07c20c86838df`;
- launch checksum manifest SHA-256:
  `299d9f3fdedcbaaed883b5a63aeaa13da2ba186dca71a5c93904fecdcfdee305`.

The run changed no production source. Nonlinear TD, soliton behavior,
optimization, production GUI behavior, and other material models were not
commissioned by this job.

## Environment and request

The job used Python 3.10.4, NumPy 2.1.0, CuPy 13.6.0, CUDA runtime 12.9,
and CUDA driver 12.9. The requested and resolved scientific backend was CuPy;
the resolved dtypes were `float64` and `complex128`, and backend provenance
reported `is_gpu=true`.

Both production cases used a 256 x 256 even transverse grid over a
64 x 64 micrometre aperture, two longitudinal planes (`dz=5` micrometres,
length 10 micrometres), two coherent 1 mW Gaussian channels, explicit
reference intensity `I0=1.5`, gain-length product `0.025`, three accepted
material intervals, normalized interval `dt=0.2`, and one optical substep.
Case A used zero bias and Case B used `E_app=0.35`. The canonical Fast package
described below was created from the biased Case B result.

The two launch tilts were exact aperture-frequency bins `(3, 2)` and
`(-2, -3)`, with Gaussian waists `(22, 19)` and `(21, 20)` micrometres and
a relative phase of 0.37 radians. This supplies a deterministic resolved
two-dimensional frozen source while keeping the workload bounded.

## Production execution and NumPy/CuPy parity

Both cases completed all three accepted intervals. Each made four complete
optical passes (three source passes plus final replay) and six plane-local
linearized material calls. Source cadence remained one complete optical pass
per accepted material interval. The predeclared parity thresholds were
`rtol=3e-11` and `atol=3e-12`.

| Case | Quantity | Relative L2 | Maximum absolute |
| --- | --- | ---: | ---: |
| Unbiased | potential | 3.7513e-16 | 2.2204e-16 |
| Unbiased | E_x / E_active | 3.8404e-14 | 1.1807e-14 |
| Unbiased | E_y | 3.5840e-14 | 1.1657e-14 |
| Unbiased | final optical field | 9.2851e-16 | 4.9065e-17 |
| Unbiased | final source intensity | 2.5114e-16 | 2.6645e-15 |
| Biased | potential | 3.9152e-16 | 2.2204e-16 |
| Biased | E_x / E_active | 4.9559e-15 | 1.1879e-14 |
| Biased | E_y | 3.8238e-14 | 9.9643e-15 |
| Biased | final optical field | 9.4324e-16 | 4.4329e-17 |
| Biased | final source intensity | 2.4363e-16 | 2.6645e-15 |

Compact diagnostics and accepted-step metadata also passed. Several
diagnostics are numerically near zero, so their relative errors are not useful;
their absolute CPU/GPU differences passed the declared absolute tolerance.
For example, the largest RHS-maximum difference was `3.4758e-13`, gauge-mean
differences remained below `3.0765e-18`, and optical power drift was
`6.6613e-16` unbiased and `4.4409e-16` biased. No fake nonlinear iteration
diagnostics were emitted.

## Bias, static limit, and null modes

For a controlled resolved `(5,5)` Fourier mode, the positive-bias transient
coefficient was `-131.39880303907705 - 1.3960932393256031 i`; reversing the
bias produced `-131.398803039077 + 1.3960932393259216 i`. Conjugation passed
with relative L2 `2.4614e-15` and maximum absolute difference `3.2345e-13`.
The imaginary part therefore reversed direction while the decay envelope was
preserved.

An H200 operator-level long-time diagnostic at normalized time 30 compared
the exact production TD propagator with the accepted static linearized
operator. Potential, E_x, E_y, and E_active were identical in the retained
float64 arrays. This was a bounded static-limit sanity check, not a separate
coupled-static commissioning campaign.

The constant and joint derivative-null/Nyquist coefficients on the 256 x 256
grid remained zero to a maximum `2.9030e-17`. No division regularization,
non-finite state, or null-mode contamination was observed.

## Fast package and products

The canonical Fast request/result package was 4,207,592 bytes (4.013 MiB).
It retained both `A_initial` and `A_final`, compact TD diagnostics, backend and
resolved-profile provenance, and accepted-step metadata. It omitted exactly
`psi_initial` and `psi_final`, as specified by the Fast policy; omitted
material volumes decoded as unavailable and were not synthesized.

Remote conversion produced input intensity, output intensity, and far-field
intensity in 0.0318 seconds. Independent local reconstruction from the
retrieved package took 0.0149 seconds and reproduced those fields, the carrier
power and transverse-PR diagnostics, 256-element x/y axes, two z coordinates,
the linearized material specification, and `Experimental` validation status.
Fast mode intentionally cannot present material endpoint fields because its
documented policy omits both material volumes.

## Runtime and GPU memory

Scheduler elapsed time was 12 seconds. The synchronized production scientific
times were 0.7990 seconds for the first, unbiased CuPy case and 0.03357 seconds
for the subsequently warmed biased case. Across those two cases, instrumented
optical passes used 0.03040 seconds and exact plane-local material evolution
used 0.24662 seconds. The first case includes first use after explicit
CUDA/CuPy initialization and therefore must not be interpreted as a warmed
steady-state measurement. No claim of process-cold CUDA startup is made.

The pre-science `nvidia-smi` baseline was 0 MiB. Process sampling every
0.2 seconds observed a peak of 576 MiB. The CuPy memory pool ended with
53,494,272 total allocated bytes (51.016 MiB) and zero used bytes. Memory
samples were copied into the compact evidence package before cleanup and were
checksum-verified after retrieval.

## Retrieval and cleanup

All retrieved remote evidence files passed the job-generated retrieval
checksum manifest before permanent-evidence curation. The consolidated
permanent manifest checksums every retained Fast-package, metrics, memory,
final-scheduler, reconstruction, cleanup, and provenance artifact. Final
scheduler accounting, local reconstruction evidence, and cleanup evidence are
retained locally; the raw stdout and stderr remain in the immutable remote
launch directory.

After successful retrieval, the transient remote directory
`launch/runs/3593204` was removed and verified absent. The immutable source
bundle, harness, scheduler input, checksum manifest, and stdout/stderr remain
under the remote launch directory according to commissioning policy. Thus the
cleanup removed temporary extracted source and run products, not the retained
launch/provenance evidence.

## Conclusion and limitations

The production full-transverse linearized time-dependent PR path is
**Commissioned** for this bounded H200/CuPy float64/Fast configuration. Backend
execution, production coupling, NumPy parity, biased transient direction,
static-limit behavior, even-grid null modes, Fast retrieval, local product
conversion, provenance, memory evidence, and cleanup all passed.

This result does not commission CuPy float32, nonlinear full-transverse TD,
larger grids, throughput scaling, optical self-consistency studies, or soliton
work. The timing data consist of one initialization-bearing case followed by
one warmed case and are commissioning measurements rather than a performance
benchmark series.
