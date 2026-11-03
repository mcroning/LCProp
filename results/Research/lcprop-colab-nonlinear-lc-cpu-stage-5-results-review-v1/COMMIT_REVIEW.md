# Stage 4–5 accepted commit scope

Accepted for commit: exact nonlinear LC example, usage documentation, focused tests,
Stage 4 validation report and small receipts, Stage 5 procedure/checking scripts,
request/hash inventories and hosted qualification report/receipts.

Acceptance rechecked before staging: original three-file candidate hashes exact;
portable input manifest and sealed bundle identities exact; returned archive and
all manifest-bound files exact; 18/18 comparisons pass; 100/100 nonlinear slices
converged under unchanged 0.005/0.02 gates; 6,275 iteration records retained;
264 installed package hashes, executed script and both requests verified.
Retained validation is 4 passed in 30.86 seconds with exact canonical parity.
No simulations or benchmarks were rerun for this commit. Python 3.10 grammar
checks pass. No engine/API/source arithmetic changes are part of this commit.

## Deliberate evidence exclusions

ZIP bundles, wheel, NPZ arrays, PNG figures, full multi-megabyte scalar histories,
large preservation snapshots, temporary scripts and duplicated copied examples are
not committed. Existing copies remain unchanged locally. Committed manifests retain
their hashes; references in the historical reports intentionally point to these
external artifacts. A fresh clone alone cannot execute the exact archived Colab
procedure without obtaining the hash-matched portable bundle/reference arrays.
No report or sealed package was rewritten to conceal that limitation. The standalone
example can run with the documented pinned engine installation independently.
Historical statements that the example was uncommitted describe validation time;
this commit now versions it without changing its validated bytes.

The unrelated fixed-LC Stage 2 example and other research work are excluded. Local
filesystem paths in receipts document provenance; no credential/private-key material
was found in selected files. All selected files are small UTF-8 text; largest is
under 40 KiB. No binary or generated simulation array is staged.

## Staged review

Exact staged inventory and file bytes were verified against the selected hashes.
The sole `git diff --cached --check` notice is the original trailing blank line in
`retrieved/numerical-config.txt`. This raw hosted library-config receipt is preserved
byte-for-byte to retain its manifest identity; it is not reformatted. All other
staged files pass whitespace verification. No unrelated or scientific-engine file
is staged.

## Exact committed-file inventory

- `examples/lc_static_nonlinear_cpu.py`
- `examples/lc_static_nonlinear_cpu.md`
- `tests/test_lc_static_nonlinear_example.py`
- `results/Research/lcprop-scientific-package-stage-4-nonlinear-lc-example-v1/REPORT.md`
- `results/Research/lcprop-scientific-package-stage-4-nonlinear-lc-example-v1/candidate-inventory.json`
- `results/Research/lcprop-scientific-package-stage-4-nonlinear-lc-example-v1/execution-receipt.json`
- `results/Research/lcprop-scientific-package-stage-4-nonlinear-lc-example-v1/pre-execution-budget.json`
- `results/Research/lcprop-scientific-package-stage-4-nonlinear-lc-example-v1/resource.json`
- `results/Research/lcprop-scientific-package-stage-4-nonlinear-lc-example-v1/tests.txt`
- `results/Research/lcprop-scientific-package-stage-4-nonlinear-lc-example-v1/installed-parity-review/comparison.json`
- `results/Research/lcprop-scientific-package-stage-4-nonlinear-lc-example-v1/installed-parity-review/verification.json`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-preparation-v1/REPORT.md`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-preparation-v1/PROCEDURE.md`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-preparation-v1/preparation-validation.json`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-preparation-v1/qualification-inputs/expected-arrays.json`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-preparation-v1/qualification-inputs/expected-installed-files.json`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-preparation-v1/qualification-inputs/inputs-sha256.json`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-preparation-v1/qualification-inputs/compare_saved.py`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-preparation-v1/qualification-inputs/run_hosted.py`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-preparation-v1/qualification-inputs/reference/nonlinear.lcprop.json`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-preparation-v1/qualification-inputs/reference/fixed.lcprop.json`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-preparation-v1/qualification-inputs/reference/sha256.json`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-results-review-v1/REPORT.md`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-results-review-v1/verification.json`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-results-review-v1/independent-comparison.json`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-results-review-v1/retrieved/execution.json`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-results-review-v1/retrieved/installation.json`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-results-review-v1/retrieved/numerical-config.txt`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-results-review-v1/retrieved/process.json`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-results-review-v1/retrieved/stdout.txt`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-results-review-v1/retrieved/qualification-manifest.json`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-results-review-v1/retrieved/result/sha256.json`
- `results/Research/lcprop-colab-nonlinear-lc-cpu-stage-5-results-review-v1/COMMIT_REVIEW.md`
