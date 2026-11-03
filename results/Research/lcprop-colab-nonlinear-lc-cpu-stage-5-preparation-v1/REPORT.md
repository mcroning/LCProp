# Stage 5 — nonlinear LC hosted CPU qualification preparation

**Prepared for manual hosted Colab CPU execution; not yet hosted-qualified.**
No simulation, installation, scientific-source change, reference regeneration,
commit or push was performed. Only this new preparation directory was written.

## Portable inputs and identities

Upload [qualification-inputs.zip](qualification-inputs.zip), **31,111,686 bytes**.
SHA-256:
`86be5478f788b68405684e63ab1c953d1f53bc66bfdf428fd4fea2cf66bcb23a`.

Use [PROCEDURE.md](PROCEDURE.md); replace its `BUNDLE_SHA256` placeholder with the
above value. It contains minimal Colab cells for verification, explicit installation,
one CPU execution, plot inspection, integrity checks and download. No new notebook
framework, editable checkout, GUI, GPU or cluster dependency was introduced.

| Authority | Identity |
|---|---|
| Engine source revision | `52b00928978a9ec7862a357d38327fd0b9ae857c` |
| Retained Stage 4 wheel | `1322400b31fac3a28d439f281cc189fb790f1526940dc36af9bf50971fdc5c20` |
| Exact standalone example | `2ffed40dfc82d47b73bd1f9770c2ef4a12a2918e32bd2c137f6a252ed07f90a3` |
| Nonlinear scientific request file | `b620ee2ebd878b89993dbe9ee66645da3aaabe6f793ff94089542ef19ebbbaa2` |
| Fixed scientific request file | `7aa94ff4e54b94105ad82ccb0a0e946e7031366e18c1b8c93e20ac3d2d570718` |
| Nonlinear reference NPZ | `8cfddcfc2c870fbfa5f3c4ffb020f2756e220f2638e22a3b7cc173971fd857e3` |
| Fixed reference NPZ | `fb3dcffbea72bbd3016c1d4d09694d0ed305110ac1929a97e96c4c6144c87c14` |
| Input manifest | `9be0fdb38372a9112aa9b91a10374df5dee3065fdab4fefa4d2f5b1c4a938a15` |

The example remains **uncommitted and content-pinned**, not part of the engine
revision. The wheel was copied from retained Stage 3 preparation inputs, verified
against the Stage 4 identity, **not rebuilt**. Its 264 package files were checked
against the retained installed-file inventory. Requests, NPZs, summary, PNG and
original manifest are untouched copies of Stage 4 `installed-output/`.

[expected-arrays.json](qualification-inputs/expected-arrays.json) records all 18
C-order stored-dtype array hashes, shapes and dtypes. [inputs-sha256.json](qualification-inputs/inputs-sha256.json)
records every portable payload file. The archive itself is independently hashed
above; manifests exclude their own bytes to avoid self-reference.

## Preserved science and acceptance

Unchanged 128×128×100 request: aperture 75×100 µm; length 500 µm, dz 5 µm;
one 1 mW, 10 µm-radius beam at x=-20 µm, wavelength 0.633 µm; original dark bias,
material parameters, periodic boundary, float64 NumPy, local self-consistent
picard_cn/splitstep workflow and all solver settings. RMS/max director gates
remain **0.005/0.02**. Fixed comparison freezes the dark-bias director with the same
launch, not the nonlinear final director. Original launch-boundary warning retained.

Both modes must complete 100 slices; all nonlinear slices must converge. Compare
all scientific scalar history, 100 residual summaries, reference 6,275 relaxation
records, complete widths, director reorientation and both power conventions.
Reference reorientation is 0.0800258112453327 rad. Nonlinear final midpoint widths
are (7.1214887487, 6.3070605625) µm; fixed widths (5.8854492421, 6.0137419214) µm.
The full width curves, rather than only endpoints, are included in array comparison.
Normalized power and separately reconstructed physical mW remain distinct.

The procedure defines a **provisional, predeclared consistency screen** for
cross-platform differences: max error <=1e-12+1e-9×reference peak and relative
L2 <=1e-9; exact coordinates/shapes/dtypes/requests and discrete histories.
This permits tiny accumulated floating-point differences while staying orders of
magnitude below scientific residual thresholds/response. It is not an already
qualified backend tolerance. The earlier fixed-LC hosted evidence showed ~1e-15
scale differences but cannot establish nonlinear tolerances by itself.
All metrics are saved. Passing the screen plus unchanged physical gates requires
final provenance/plot review before qualification; exceeding it requires diagnosis,
not automatic rejection of the science or retroactive threshold adjustment.
No stationary soliton, stability or spatial-convergence claim is made.

## Execution provenance and plotting

The hosted supervisor copies the executed example into the evidence archive and
hashes it before/after execution, verifies installed package bytes before/after,
records Python/NumPy/SciPy/Matplotlib/platform, package receipt/path, numerical
library configuration, thread environment, pip inventory, command, logs and timing.
The isolated CLI produces plots and arrays from the same completed result objects;
its manifest is checked before comparison. The outer evidence manifest binds the
script, environment, result and comparison. This avoids relying on notebook cell
order or an unrecorded script, a limitation identified in Stage 3.

Manual inspection of the actual exported four-panel PNG remains required on Colab;
rendering pixels need not match across Matplotlib versions. Selected NPZ exports
are not restart checkpoints/full transport archives. Reference lineage is the
retained installed Stage 4 run with canonical-workflow parity tests, not an invented
second canonical archive. Missing unexported internal arrays remain outside scope.

## Resource and failure handling

Local nonlinear time ~14 s, fixed ~0.18 s, measured process peak ~292 MiB. Plan at
least 1 GiB free memory and several minutes including import/plot/comparison overhead;
these are planning allowances, not hosted measurements. Scientific example timeout
remains 120 s per calculation at slice boundaries. Outer hard process cap is 300 s.
Any timeout, dependency failure, hash mismatch or failed gate stops the procedure
and preserves available evidence; no fallback request or automatic retry.

## Preparation checks and preservation

- Original three-file Stage 4 candidate hashes and reference manifest verified.
- Existing wheel hash and all 264 package-file identities verified.
- All 18 reference hashes independently recomputed; ZIP payload bytes verified.
- Saved-output comparator exercised against `new-nonlinear-lc-run/` without running
  science: 18/18 bitwise identities, all metric differences zero, residual and
  full-history screen passed. See [preparation-comparison-check.json](preparation-comparison-check.json).
- New scripts parse under Python 3.10 grammar. Hosted execution supervisor was
  **not executed locally**, since doing so would run the calculation. Colab and
  failure-path runtime behavior remain to be exercised in the hosted procedure.
- All 815 tracked file hashes preserved; index remains empty. Two unrelated tracked
  modifications remain untouched. Existing source/evidence paths were only read;
  all new files are within this preparation directory.

Machine-readable preparation receipt: [preparation-validation.json](preparation-validation.json).
Authority: [Stage 4 report](../lcprop-scientific-package-stage-4-nonlinear-lc-example-v1/REPORT.md);
prior hosted evidence: [Stage 3 review](../lcprop-colab-lc-cpu-stage-3-results-review-v1/REPORT.md).

**STOPPED BEFORE HOSTED EXECUTION — READY FOR MANUAL COLAB CPU QUALIFICATION.**
