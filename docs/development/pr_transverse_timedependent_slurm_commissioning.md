# PR Transverse Time-Dependent Slurm Commissioning

**Date:** 2026-08-28

**Classification:** Commissioned

**Implementation SHA:** `615ea73a73ff3c2ab5072fe0401249356bbeb713`

**Commit subject:** `Add transverse TD PR Slurm transport`

## Objective

This record commissions the canonical full-transverse time-dependent
photorefractive operation through the material-neutral Slurm runner and
portable transport path used by the PR GUI. It also verifies Image
Amplification as a local composite analysis over exactly one remotely executed
ordinary `pr_transverse_timedependent` operation.

The two bounded gates were:

1. ordinary `pr_transverse_timedependent` Local-versus-Slurm equivalence;
2. Image Amplification over `pr_transverse_timedependent`, with remote base
   propagation followed by local post-processing.

This is operational and cross-backend numerical-path commissioning. It is not
a production-scale PR calculation or a comparison between reduced and
full-transverse PR physics.

## Immutable Source and Environment

The dirty primary development checkout was not deployed. A clean detached
local clone at the implementation SHA supplied both local commissioning imports
and the committed source archive.

| Item | Value |
| --- | --- |
| Local branch | `feature/pr-second-order-static` |
| Implementation SHA | `615ea73a73ff3c2ab5072fe0401249356bbeb713` |
| Clean local source | `/private/tmp/lcprop-transverse-td-commissioning.M4X7jn/repo` |
| Remote snapshot | `/cluster/tufts/cglab/mcroning/lcprop_sources/git-615ea73a73ff3c2ab5072fe0401249356bbeb713` |
| Source archive SHA-256 | `71cc8a5ad21061887e2513a73f9f87c1f0c79fa9f3a30e1e4a2630fdd0070665` |
| Cluster/login | PAX, `mcroning@login.pax.tufts.edu` |
| Resource profile | `H200 Large` |
| Slurm request | `gpu`, QOS `normal`, 30 min, 2 CPUs, 64 GiB, `gpu:h200:1` |
| CUDA setup | `module load cuda/12.9.0` |
| Remote Python | `/cluster/tufts/cglab/mcroning/condaenv/prenv/bin/python`, Python 3.10.4 |
| CuPy | 13.6.0 |
| CUDA driver/runtime | 12090 / 12090 |
| Allocated GPU | NVIDIA H200 on `pax009` |
| Local Python | `/Users/mcroning/miniforge3/envs/lcprop/bin/python`, Python 3.12.13 |

Full-transverse TD currently requires an explicit NumPy or CuPy backend. The
local and remote requests therefore differed only in the execution-backend
field: NumPy float32 locally and CuPy float32 remotely. The harness verified
that replacing the remote backend field with the local backend produced the
same canonical request. All scientific inputs were identical.

Gate 1 created and verified the immutable source snapshot. Gate 2 reused the
same snapshot without another source upload.

## Commissioning Fixture

Both gates used the same deterministic base calculation:

- grid `64 x 48 x 4` over `120 x 80 x 20` µm;
- `dz=5` µm;
- three production spectral-IMEX material steps;
- normalized timestep `1e-4`;
- two optical substeps;
- full-transverse isotropic transport and dielectric profiles, `m_y=h_y=1`;
- unbiased periodic transverse electrostatics with zero-mean potential gauge;
- optical projection `(g_x,g_y)=(1,0)`;
- normalized dark and uniform-background intensities `0.2` and `0.1`;
- gain-length product `0.05`;
- refractive index `2.4`;
- characteristic-wavenumber override `0.1` rad/µm;
- two coherent 0.633 µm channels in group `transverse-td-image`;
- pump/signal powers `1.0` and `0.2` mW;
- pump/signal phase gradients `+0.15` and `-0.15` rad/µm;
- waists `24 x 20` µm, centered on the optical axis;
- one passive intensity raster on the signal channel and no pump screen;
- no scattering.

The deterministic `8 x 6` raster had SHA-256
`a919630df6e63ab288c13ce0875429a60cea131dd20496fe4ec1eff861ca4a9f`
and occupied `42 x 32` µm. Pump and signal roles were explicit.

The clean-checkout local preflight completed all three material steps. Its
relative optical-power drift was `1.250e-7`. Local Image Amplification also
completed, with measured gain `0.99992737`, correlation `0.9999999769`, and
normalized RMSE `5.4304e-5`.

## Acceptance Criteria

The predeclared cross-backend tolerance was `rtol=3e-5`, `atol=3e-6`, with a
relative-L2 alternative of `3e-5` for transformed fields. Boolean masks and
categorical state were required to agree exactly. Independent wall-clock
timings and backend identity were not treated as scientific-result equality.

The remote result envelope was required to report:

- requested and resolved backend `cupy`;
- `scientific_status=completed`;
- `cancelled=false`;
- `converged=null`;
- termination reason `requested_material_steps_completed`.

## Scheduler Summary

| Gate | Submitted operation | Job | Node | State | Exit | Runtime |
| --- | --- | ---: | --- | --- | --- | ---: |
| 1 | `pr / pr_transverse_timedependent` | 2992820 | pax009 | COMPLETED | `0:0` | 7 s |
| 2 | `pr / pr_transverse_timedependent` | 2992938 | pax009 | COMPLETED | `0:0` | 9 s |

Each gate submitted exactly one job. No retry or replacement job was used. No
remote `pr_image_amplification` operation was submitted.

## Gate 1 — Ordinary Transverse TD

The remote result completed all three requested steps at normalized material
time `0.0003`. Array comparison was:

| Canonical array | Relative L2 | Maximum absolute difference |
| --- | ---: | ---: |
| `A_initial` | `7.045e-8` | `5.890e-9` |
| `A_final` | `1.365e-6` | `7.181e-8` |
| `psi_initial` | `0.0` | `0.0` |
| `psi_final` | `1.126e-6` | `1.135e-9` |

Initial optical power agreed exactly. Final power differed by `8.941e-7`, a
relative difference of `9.374e-7`. The remote relative power drift was
`-8.124e-7`; both local and remote drift remained at the expected float32
level.

Scientific diagnostics, grid geometry, launch provenance, resolved transverse
profile, ordinary RunData field identities, coordinates, arrays, and recursive
diagnostic values all passed. Very small gauge/curl/Gauss quantities could
have large relative differences because their denominators were near zero;
their absolute differences remained below the declared absolute tolerance.

The lifecycle progressed through submission, pending, running, scientific
completion, retrieval, verification, reconstruction, product conversion, and
only then GUI-ready `COMPLETED`. The final status reported the verified CuPy
backend and scheduler-visible NVIDIA H200.

### Gate 1 Harness Correction

The first local comparison report returned `passed=false` after the remote job
had completed successfully. The cause was exact Python dictionary equality for
float32 `launch_summary` and `resolved_profile` values. Canonical arrays and
transport checks had already passed. The already downloaded, checksum-verified
result was re-evaluated locally with the predeclared float32 tolerance and
passed. No job was retried and no production source was changed.

## Gate 2 — Image Amplification over Transverse TD

Gate 2 submitted one ordinary `pr_transverse_timedependent` request. The
canonical result was downloaded, checksum-verified, reconstructed, and passed
through the ordinary transverse-TD product adapter. Only afterward did the
local process report the six Image Amplification stages:

1. carrier isolation;
2. output back-propagation;
3. zero-response propagation;
4. reference back-propagation;
5. metric construction;
6. product augmentation.

This confirms the intended architecture:

```text
Image Amplification orchestrator
    -> one remote ordinary pr_transverse_timedependent operation
    -> verified canonical PRTransverseRunResult
    -> local Image Amplification analysis and products
```

The analysis continued to use carrier isolation, one full-length linear `-L`
back-propagation, and zero-response full-length `+L` and `-L` propagation. No
per-z complex optical history was transported.

The principal derived-field comparisons were:

| Image product | Relative L2 | Maximum absolute difference |
| --- | ---: | ---: |
| `image_transmission` | `0.0` | `0.0` |
| signal-carrier mask | `0.0` | `0.0` |
| input isolated signal | `2.153e-7` | `4.165e-9` |
| output isolated signal | `2.069e-6` | `3.957e-8` |
| back-propagated amplified image | `2.071e-6` | `3.613e-8` |
| zero-response reconstruction | `1.623e-7` | `3.434e-9` |

Scalar differences were:

| Metric | Absolute difference |
| --- | ---: |
| Analytic gain | `0.0` |
| Measured gain | `1.000e-6` |
| Image correlation | `3.486e-9` |
| Zero-response correlation | `3.496e-9` |
| Normalized RMSE | `4.033e-7` |
| Relative optical-power drift | `9.374e-7` |
| Signal throughput | `0.0` |

All sixteen augmented product fields passed. The largest product relative-L2
difference was `8.955e-6` for the logarithmic far field. Field identities,
geometry, coordinates, arrays, presentation metadata, curves, and recursive
scientific diagnostic values agreed within the declared criterion. Base,
analysis, and overall status were all `completed`.

## Artifact and Integrity Evidence

Successful remote run directories were removed only after retrieval, checksum
verification, canonical reconstruction, and ordinary product conversion. The
verified immutable source snapshot remains reusable.

| Gate | Local run ID | Result JSON SHA-256 | Result arrays SHA-256 | Local bytes |
| --- | --- | --- | --- | ---: |
| 1 | `lcprop-4bfd40cfe0ee4db19b5f4a6f2f9a4eb8` | `d7f0c584...448ac` | `25ee91dd...6fb7` | 214,616 |
| 2 | `lcprop-a86bc664cc4d4d11bd1d949139c85974` | `849c1e85...ed80` | `25ee91dd...6fb7` | 214,615 |

The result-array archives are byte-identical because both gates submitted the
same ordinary base physics request. Gate 2 transported no image-analysis data.
Post-scheduler retrieval, verification, reconstruction, and cleanup took
approximately 11.66 s for Gate 1 and 10.65 s for Gate 2.

Complete local artifacts and compact comparison JSON are under:

`/private/tmp/lcprop-transverse-td-commissioning.M4X7jn/`

The successful cleanup policy did not retain stdout, stderr, or GPU-memory
sampling files in the downloaded compact package. Scheduler accounting and
execution provenance were retained, but peak device memory is therefore not
claimed.

## Four-Algorithm Capability Matrix

| PR algorithm | Local | Slurm | Image Amplification over Slurm |
| --- | ---: | ---: | ---: |
| Static 1D | commissioned | commissioned | commissioned |
| Static 2D | commissioned | commissioned | commissioned |
| TD 1D | commissioned | commissioned | commissioned |
| TD 2D | commissioned | commissioned | commissioned |

Remote TD continuation and checkpoint recovery are not claimed for
full-transverse TD. The canonical transverse-TD result contains no checkpoint,
restart record, retained source volume, or time-history volume.

## Conclusion

Both commissioning gates passed at implementation SHA
`615ea73a73ff3c2ab5072fe0401249356bbeb713`. The canonical full-transverse TD
workflow now has direct evidence that the same scientific request can execute
locally with NumPy and remotely with CuPy on an H200 while preserving canonical
state, diagnostics, and presentation products within declared float32
tolerances.

Image Amplification also passed through the intended composition boundary:
one remote ordinary solve, verified canonical reconstruction, and local-only
analysis. This completes bounded Local/Slurm/Image-Amplification commissioning
for all four currently exposed PR algorithms.

The recommended next step is a bounded review and commit of this commissioning
record. Sparse retrieval, remote continuation, and production-scale scientific
runs remain separate future work.
