# Full-Transverse Nonlinear PR TD H200 / Slurm Commissioning

**Date:** 2026-09-13

**Production-path classification:** Commissioned

**Production commit:** `1a129dc3cb93f6c8a005a198f40d86a31792f223`

**Authoritative commissioning job:** `3672475` (`COMPLETED`, exit `0:0`,
elapsed `00:00:13`)

**Historical failed job:** `3662453` (`FAILED`, exit `1:0`, elapsed
`00:00:13`; permanently **Not Commissioned**)

Both jobs ran on `pax009` with an NVIDIA H200. Job `3672475` is the sole
authoritative basis for the production path's **Commissioned** classification.
Job `3662453` remains a separate failed commissioning record and is not
reclassified by the successful retry.

## Provenance and launch

For the initial attempt, exactly one job was submitted and no automatic retry
was attempted. The job cloned the
complete source bundle, checked out the production commit detached, and
required an empty `git status --porcelain`. Remote launch checksum verification
passed before submission and again inside the allocation.

Launch inputs were:

- source bundle: `dbba0dcf54da3da9cff41c98928507aa4f51ec4461d1d940580c1becb77f3ba0`;
- commissioning harness: `57ac722051b3b4c775c2440a76db6c8da855b51f7c2f4f1c181af06e4dd86807`;
- one-off scheduler script: `a439aa3888bf1cc5d174cd98ddf44f138af6ea57ea6bcc6ec7cc84390063c72e`;
- launch manifest: `53a568f99f7001d420b5a8634d02e35a5abf5c2b3706820d995e2e28b1b1b110`;
- launch checksum file: `00d027b2b20b9472fc087eab434e81bef60a83375cc671a0954a894563d68638`.

The prescribed primary request used the nonlinear default material response,
zero applied field, a 256 x 256 transverse grid over 64 x 64 micrometres, two
longitudinal planes, three material intervals of normalized duration 0.05, one
optical substep, two coherent 1 mW beams, float64/complex128 CuPy, and Fast
retention. The predeclared NumPy/CuPy thresholds were `rtol=3e-10` and
`atol=3e-11`.

## Execution reached before the harness failure

The traceback location establishes that the primary NumPy and CuPy production
runs, parity assertions, 32 x 32 static sanity, 32 x 32 timestep sanity,
cancellation sanity, and Fast package construction all returned without an
earlier exception. This is useful diagnostic evidence, but the corresponding
quantitative comparison dictionaries remained only in process memory.

The retrieved Fast result independently establishes actual CuPy GPU execution:

- resolved backend `cupy`, `is_gpu=true`, float64/complex128;
- nonlinear material response and `Validated` response metadata;
- three accepted material intervals and three actual IMEX material calls;
- four complete optical passes;
- zero-mean potential diagnostic `3.3881317890172014e-20`;
- physical-state validity true;
- minimum carrier density `0.5760850973270508`;
- final nonlinear TD RHS RMS `0.17935435855301135`;
- final nonlinear TD RHS maximum `0.6710893841846177`;
- frozen-source cadence recorded correctly.

GPU process sampling at approximately 0.2 second cadence recorded a 0 MiB
pre-science baseline and a peak process allocation of 572 MiB. The H200
reported 143771 MiB total memory. Component timing and CuPy-pool measurements
were not serialized, so only the scheduler elapsed time of 13 seconds is
retained.

## Fast retrieval and local reconstruction

The complete nine-file Fast request/result package passed its internal request
and output checksum manifests. Its size is 4,207,630 bytes. Independent local
reconstruction from the retrieved package produced the 256-element x and y
axes, two z coordinates, and the canonical input-intensity, output-intensity,
and far-field-intensity products.

Fast retention omitted exactly `psi_initial` and `psi_final`. They reconstructed
as unavailable and were not synthesized. Optical endpoints, nonlinear compact
diagnostics, backend provenance, status, and accepted-step metadata remained
available.

## Initial failure and job 3662453 classification

After all bounded scientific cases and Fast packaging, the harness attempted to
serialize `scientific_rows.json`. Its canonical JSON helper did not pass the
existing NumPy-aware JSON encoder, and a retained NumPy array caused:

```text
TypeError: Object of type ndarray is not JSON serializable
```

This is a commissioning-harness evidence-serialization defect, not evidence of
a production-physics or GPU-execution failure. Nevertheless, it prevented
creation of:

- the quantitative NumPy/CuPy comparison rows;
- static and timestep numerical values;
- cancellation evidence details;
- runtime-component and CuPy-pool measurements;
- `scientific_rows.json` and its canonical checksum;
- `commissioning_metrics.json` and the in-job evidence manifest.

The commissioning prompt requires retained quantitative CPU/GPU parity and a
canonical scientific-data checksum. Because both are missing, the correct
classification for job `3662453` is **Not Commissioned**. The successful assertions cannot be
promoted into authoritative numerical evidence without their retained values.
No tolerance was changed and, at that stopping point, no retry had been
submitted.

## Cleanup and non-change

The partial evidence, memory samples, immutable logs, and final terminal
scheduler accounting were retrieved. The transient remote run directory was
then removed and independently verified absent. Immutable launch inputs and
logs remain under the remote launch directory.

No production source was modified. Nonlinear bias remains unsupported and was
not attempted. The commissioned linearized TD path was neither modified nor
rerun. No optimization, soliton work, staging, commit, or push occurred.

At that stopping point, a future retry required a separately reviewed,
checksummed harness fix and new explicit submission authorization. The failed
job did not itself authorize that work or another submission.

## Harness remediation and authorized retry

The failed job and its **Not Commissioned** classification remain unchanged.
The retained `job_3662453` directory was not overwritten or reinterpreted.

The root cause was confined to the harness evidence boundary: the canonical
writer called standard `json.dumps` on a nested structure containing NumPy
arrays. The other report writers happened to use a NumPy-aware fallback, but
the canonical writer did not, so the first retained array raised the reported
`TypeError` after the scientific work had completed.

The remediated harness now sends every JSON artifact through one recursive,
strict normalizer. It converts NumPy scalars to Python scalars, NumPy arrays to
lists, CuPy scalars and arrays to host values at the evidence boundary, tuples
to lists, string-keyed dictionaries recursively, paths to filesystem strings,
and enums through their declared values. It rejects non-string mapping keys,
non-finite floats, and unsupported objects with a path-qualified exception;
there is no arbitrary string fallback. JSON writes use a temporary sibling and
atomic replacement.

Completed quantitative stages are also retained before final canonical
assembly as `primary_quantitative.json`, `static_sanity.json`,
`timestep_sanity.json`, `cancellation_sanity.json`,
`fast_package_summary.json`, and `cupy_memory_pool.json`. A later schema or
manifest failure therefore cannot erase already completed parity, timing,
scientific, Fast-package, or memory-pool values. The canonical scientific rows
now explicitly include CPU/GPU timings and the CuPy-pool measurements while
preserving the previous schema names and scientific quantities.

Local validation used an isolated clean source tree at exact production commit
`1a129dc3cb93f6c8a005a198f40d86a31792f223`. The focused serialization suite
passed (`4 passed, 1 skipped`; the skip was the conditional no-GPU CuPy case).
It reproduces the exact `ndarray` serialization failure, constructs the full
final scientific-evidence object, exercises normalization and canonical JSON
write/read, verifies strict rejection, and confirms that an earlier durable
artifact survives a later formatting failure. The clean 32 x 32 preflight
passed, including request codec round trips. A local NumPy smoke using the
unchanged 256 x 256, two-plane, three-interval request completed three material
calls and four optical passes with a physical state and minimum carrier density
`0.5760850973270796`.

A separate retry launch package was prepared under
`results/pr_full_transverse_nonlinear_td_h200_commissioning_2026-09-13/retry_pre_submission_launch/`
and, after explicit authorization, uploaded to
`mcroning@login.pax.tufts.edu:/cluster/tufts/cglab/mcroning/lcprop_runs/pr-nonlinear-td-1a129dc-retry1/launch/`.
Its checksums are:

- source bundle: `fd963b1242e16055b38f5269e508e7bb6484dc532b312e87b13034845f2b68b7`;
- remediated harness: `1e28f04a9891b0852ab2a8dd88ce09a6db18889bcb012811e1f508692460e667`;
- one-off scheduler script: `7fd971939d79d36f7eba9fde330b7178d45f647cd58efd5cbb79f6ac6478c091`;
- launch manifest: `ae833a1683486e0b8a5f35bff5217d74c0f12d3c7c266faba971858f3c5652e4`;
- launch checksum file: `fe7847473ed8a84bbda25f29e009e8c332b741ad5a5ef78e8e83905e245eb452`.

The bundle verifies as a complete bundle whose `HEAD` is the exact production
commit, and all four entries in the launch checksum file pass. The retry keeps
the original request, controls, `rtol=3e-10`, `atol=3e-11`, and 0.2-second
GPU-memory sampling unchanged. Exactly one explicitly authorized retry was
submitted as job `3672475`; no second retry was submitted.

## Successful H200 commissioning

Job `3672475` completed `0:0` in 13 seconds on `pax009`. The retained
environment records Python 3.10.4, NumPy 2.1.0, CuPy 13.6.0, CUDA runtime and
driver 12.9, and an NVIDIA H200. The requested and resolved backend was CuPy,
with `is_gpu=true`, float64 real arrays, and complex128 optical arrays; there
was no CPU fallback. The production checkout was the exact clean commit
`1a129dc3cb93f6c8a005a198f40d86a31792f223`, with empty
`git status --porcelain`.

The unchanged primary request completed three accepted nonlinear material
intervals, three IMEX material calls, and four optical passes. It used unbiased
nonlinear Profile v1 and did not exercise unsupported nonlinear bias. The final
state was finite and physical, with minimum carrier density
`0.5760850973270508`, maximum absolute potential mean
`3.3881317890172014e-20`, nonlinear TD RHS RMS `0.17935435855301135`, and
nonlinear TD RHS maximum `0.6710893841846177`.

### NumPy/CuPy parity

All comparisons passed the predeclared `rtol=3e-10` and `atol=3e-11`:

| Quantity | Relative L2 | Maximum absolute |
| --- | ---: | ---: |
| `psi` | `2.02427243004672e-15` | `2.7755575615628914e-16` |
| `E_x` | `2.5031225311598255e-14` | `3.755676325489787e-15` |
| `E_y` | `2.3087898613296795e-14` | `3.4017927363905187e-15` |
| `E_active` | `2.5031225311598255e-14` | `3.755676325489787e-15` |
| Final optical field | `8.559603838294974e-16` | `4.654751537338637e-17` |
| Final source intensity | `2.353337060546019e-16` | `2.6645352591003757e-15` |

Array-valued carrier integrals agreed exactly. Differences in near-zero scalar
diagnostics are interpreted through their retained maximum-absolute values and
the declared absolute tolerance, rather than their ill-conditioned relative
ratios.

### Bounded scientific controls

The 32 x 32 static sanity fixture converged, remained physical, and changed by
only `1.3316381643190637e-13` in relative L2 after one TD step. The timestep
refinement ratio was `1.9715451064890337`, within the predeclared `[1.7, 2.3]`
interval and consistent with the locally established first-order integrator;
it is not evidence of higher-order convergence.

Cancellation observed after construction of a nonlinear material candidate
returned status `cancelled` with zero completed steps. The candidate was
discarded, and the last accepted potential and final optical replay were
preserved bitwise.

### Runtime and memory

The initialization-bearing first scientific CuPy measurement after explicit
CUDA/CuPy initialization was `0.42976761795580387 s`. This is not a
process-cold startup measurement or a warmed steady-state benchmark. Within
that measurement, three nonlinear material updates took
`0.02014520694501698 s` and four optical passes took
`0.01533179497346282 s`.

GPU process memory was sampled at the requested approximately 0.2-second
cadence. Seven retained samples reached an observed peak of 572 MiB; this is an
observed bound, not an extrapolation. The H200 reported 143771 MiB total
memory. The CuPy pool measurement changed from zero allocated bytes before the
case to 49,299,968 total pool bytes afterward, with zero live used bytes at the
final sample.

### Fast package and retained evidence

The canonical Fast request/result package is 4,207,630 bytes
(`4.012709 MiB`). Its internal request and result checksums pass. It retains
both required optical endpoints and the `input_intensity`, `output_intensity`,
and `far_field_intensity` products. Fast policy omits `psi_initial` and
`psi_final`; local reconstruction confirmed that both remain unavailable and
were not synthesized. The nonlinear material model, `Validated` status,
physical diagnostics, three completed steps, backend provenance, and product
geometry survived decoding and canonical product conversion.

The canonical scientific-data checksum is
`46dfb4797a0872da9a375589ccd9c63e3e3618e66583e4974151c928b3015738`.
The remote retrieval checksum manifest passed before cleanup, and the local
post-retrieval checksum manifest passed after scheduler, reconstruction,
provenance, and cleanup evidence were added. Final terminal scheduler
accounting is retained as `SCHEDULER_ACCOUNTING_FINAL.txt`; no unlabeled
nonterminal scheduler snapshot is used as authoritative evidence.

The transient remote run tree for job `3672475` was removed and independently
verified absent after retrieval. Immutable launch inputs and stdout/stderr logs
remain remotely according to commissioning policy. The compact permanent
evidence is checksum-closed by
`results/pr_full_transverse_nonlinear_td_h200_commissioning_2026-09-13/retry_job_3672475/evidence_manifest.json`.

## Failed-attempt audit trail

Job `3662453` remains permanently **Not Commissioned**. Its scientific work
substantially completed, but the harness failed while serializing canonical
evidence, so quantitative parity rows and the canonical scientific checksum
were not retained. The separately reviewed remediation led to job `3672475`;
it does not retroactively change the failed job's classification.

The failed-job manifest remains byte-for-byte unchanged with SHA-256
`cf4035696f276d4ee6eef013f189413d88473cc6db6d44098639a41941ac4c38`.
Its embedded report checksum `439bbbb302906212fe351633b3ab5eae2629d0741f02266c8f4d58e88cce7e2e`
identifies the historical pre-remediation report snapshot that accompanied the
failed record. It is intentionally not a checksum of this current combined
report.

No production source, nonlinear equation or integrator, linearized TD path,
static solver, optics, persistence/transport logic, GUI, run-cost behavior,
nonlinear bias behavior, or soliton code changed during commissioning or
harness remediation.
