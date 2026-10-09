# Stage 5 — hosted nonlinear LC CPU results review

**PASS — the tested self-consistent nonlinear LC CPU example is qualified in the
reported hosted Colab environment.** This is a bounded 128×128×100 qualification,
not all LC workflows, large-grid performance, stationary soliton formation or stability.
No calculation was rerun during review; only retained-array postprocessing occurred.

## Integrity and provenance

Received archive: `../lcprop-colab-nonlinear-lc-cpu-stage-5-preparation-v1/lc-nonlinear-colab-results.zip`.
SHA-256 `a04fde6758533c0d428e5fe5767beb512d34166c5bcd6d91b25e675f5951be3f`.
Evidence-manifest SHA-256 `874e64f47beb73c56d3952d5de655ff064ab921d82f34bba265dd48691f1dc02`.
All 16 manifest entries and exact archive file membership verified;
inner result manifest and exact output membership verified independently. No failure
receipt; process exit 0 and empty stderr. Retrieved copies are in `retrieved/`.

- Engine source binding: `52b00928978a9ec7862a357d38327fd0b9ae857c`.
- Installed wheel receipt: `1322400b31fac3a28d439f281cc189fb790f1526940dc36af9bf50971fdc5c20`.
- Executed script bytes: `2ffed40dfc82d47b73bd1f9770c2ef4a12a2918e32bd2c137f6a252ed07f90a3`.
  The actual copied script is retained, matching the validated uncommitted example;
  it is not claimed to have an example Git revision.
- Input manifest: `9be0fdb38372a9112aa9b91a10374df5dee3065fdab4fefa4d2f5b1c4a938a15`.
- All 264 recorded installed package-file hashes match the reference. Execution used
  `/usr/bin/python3 -I` and `/usr/local/lib/python3.13/dist-packages/lcprop/__init__.py`.
- Both scientific request files are byte-identical to Stage 4. No grid, solver,
  convergence, launch, bias, normalization or boundary changes detected.

Environment: Python 3.13.16; NumPy 2.1.3; SciPy 1.16.3; Matplotlib 3.10.0;
Linux x86-64, glibc 2.39. NumPy records OpenBLAS 0.3.27; complete NumPy/SciPy
build configurations and pip inventory are retained. Reference: macOS arm64,
Python 3.12.13, NumPy 2.5.1, SciPy 1.18.0. Recorded installed-file attestations
and receipts are internally consistent; the remote filesystem is not independently
accessible in this review. No signature-based attestation is implied.

## Independent scientific comparison

[Independent comparison JSON](independent-comparison.json) contains both SHA-256
values for every array, differing counts, shapes checked against the pinned inventory,
and all numerical metrics. The prepared comparator was rerun locally on the downloaded
arrays, without scientific execution. Independently recomputed hashes, maximum absolute
errors and differing-element counts agree with the hosted comparator.

| Array | Bitwise | Max absolute | RMS absolute | Relative L2 | Max relative* |
|---|---|---:|---:|---:|---:|
| nonlinear/A_initial | No | 1.38778e-17 | 1.53633e-19 | 1.33066e-17 | 3.32727e-16 |
| nonlinear/A_final | No | 1.19374e-13 | 6.34482e-15 | 5.49541e-13 | 1.60758e-09 |
| nonlinear/theta | No | 3.33067e-15 | 4.55356e-16 | 7.86367e-16 | 5.05851e-15 |
| nonlinear/theta_bias | No | 1.11022e-16 | 2.47768e-17 | 4.39626e-17 | 1.96456e-16 |
| nonlinear/intensity_stack | No | 4.39579e-15 | 4.06675e-17 | 3.55873e-14 | 1.21184e-08 |
| nonlinear/widths_um | No | 6.39488e-14 | 2.61779e-14 | 4.276e-15 | 7.83021e-15 |
| nonlinear/x_um | Yes | 0 | 0 | 0 | 0 |
| nonlinear/y_um | Yes | 0 | 0 | 0 | 0 |
| nonlinear/z_um | Yes | 0 | 0 | 0 | 0 |
| fixed/A_initial | No | 1.38778e-17 | 1.53633e-19 | 1.33066e-17 | 3.32727e-16 |
| fixed/A_final | No | 6.4548e-15 | 7.5825e-16 | 6.56739e-14 | 1.20501e-05 |
| fixed/theta | No | 1.11022e-16 | 2.47768e-17 | 4.39626e-17 | 1.96456e-16 |
| fixed/theta_bias | No | 1.11022e-16 | 2.47768e-17 | 4.39626e-17 | 1.96456e-16 |
| fixed/intensity_stack | No | 8.69096e-16 | 2.46906e-17 | 2.94447e-14 | 5.80591e-09 |
| fixed/widths_um | No | 4.44089e-14 | 1.4256e-14 | 2.64311e-15 | 6.06649e-15 |
| fixed/x_um | Yes | 0 | 0 | 0 | 0 |
| fixed/y_um | Yes | 0 | 0 | 0 | 0 |
| fixed/z_um | Yes | 0 | 0 | 0 | 0 |

*Maximum relative error uses reference magnitudes >= max(1e-300, 1e-12×reference peak).
This reporting threshold is not an acceptance tolerance. Small dark-tail denominators
make some maximum relative errors larger (fixed endpoint ~1.2e-5), despite tiny absolute
errors. Acceptance uses the predeclared global absolute and relative-L2 criteria,
not pointwise relative error alone. No phase alignment, rescaling or rounding applied.

All 18 arrays pass the predeclared max-absolute <=1e-12+1e-9×reference peak and
relative-L2 <=1e-9 screen, with exact coordinate bytes and matching shapes/dtypes.
Six coordinate arrays are bitwise identical; twelve scientific arrays have small
floating-point differences. The largest relative L2 is ~5.50e-13 in nonlinear A_final,
well inside the predeclared screen. No thresholds changed after seeing results.

Differences already appear in the launch A_initial (26 elements; max 1.39e-17) and
dark-bias director (max 1.11e-16), before nonlinear evolution. They accumulate modestly
in propagation; nonlinear theta differs by at most 3.33e-15 rad. This pattern,
identical requests/engine bytes, unchanged iteration schedule and much-smaller-than-gate
errors are consistent with cross-platform floating-point effects. Specific causation
by BLAS, FFT, compiler or a particular library is **not established** by this evidence.

## Convergence and diagnostics

- Both calculations completed **100/100 slices**; nonlinear all-slice convergence true.
- Maximum final slice RMS residual: **0.0049350853571164107**, gate 0.005.
- Maximum final slice maximum residual: **0.019985181428802123**, gate 0.02.
- All 100 slice gates, flags, termination reasons and discrete iteration structure
  match. **6275** relaxation records retained. All scientific
  scalar/history floats pass the predeclared comparison; this is not bitwise scalar identity.
- Maximum reorientation: **0.080025811245331369 rad**, independently
  recomputed from saved theta/bias. Reference 0.0800258112453327 rad.
- Nonlinear last midpoint widths: **[7.121488748678269, 6.3070605625133345] µm**;
  fixed: **[5.885449242055482, 6.013741921381347] µm**. Complete curves compared and
  independently recomputed with the Product RMS diagnostic from saved intensities.
- Nonlinear normalized power: **0.99976911219201126 → 0.99976911219207953**;
  independently reconstructed from complex arrays. Dimensionless, not mW.
- Separate physical power: **0.99972300271064385 → 0.99497326085679128 mW**.
  No normalized-power conservation rule was imposed on this separate flux diagnostic.
- Fixed-director convergence is not applicable; its theta equals dark bias exactly.

Original tolerances and boundary warning remain unchanged. Transient narrowing,
subsequent broadening and director reorientation agree with Stage 4. No stationary
soliton or stability claim follows from self-consistent slice convergence.

## Plotting, export and resources

The actual `retrieved/result/comparison.png` was visually inspected: four correct
panels, physical coordinates, radian director labels, normalized intensity units and
width legends render. Endpoint z=500 µm and last material/width midpoint z=497.5 µm
remain distinct. Image bytes are bound by the result manifest alongside the arrays.
The exact executed script creates both from the same completed result objects within
one invocation; no stale/mismatched artifact evidence exists. Pixel identity across
Matplotlib versions is not required. Manifest consistency is not cryptographic proof
of process causality.

Nonlinear workflow including products: **69.038 s**;
fixed **1.003 s**; supervisor total **76.037 s**.
Linux child MaxRSS **309108 KiB (~301.9 MiB)**. This includes child processes and is
not an isolated sampled solver-memory profile. Runtime stayed below original 120 s
per-calculation and 300 s outer limits. Compared with local ~14 s/~292 MiB, hosted
execution is slower but remains within the planned budget. No benchmark claim.

## Scope, limitations and preservation

Stage 5 is accepted for this exact CPU fixture, engine and tested hosted environment:
installation, isolated headless execution, self-consistent convergence, numerical
agreement, plotting and export verified. Future workloads/platforms require their
own evidence. Selected exports omit some internal arrays and are not checkpoints.
The retained Stage 4 installed reference is backed by original canonical parity tests;
no missing reference arrays were regenerated.

Only this new review directory was written. Input archive, preparation package,
reference evidence and scientific source remain unchanged. Tracked-file preservation
checked; index empty and unrelated tracked modifications retained. No installations,
simulations, GPU/cluster access, commits or push.
