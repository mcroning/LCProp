# Stage 3K — Hosted scientific execution and snapshot qualification preparation

**PASS: hosted single-run qualification and independent exported-evidence review are complete for this bounded preset.** See the dated closure below. The following preparation record describes the state before execution authorization. No hosted calculation, parameter sweep or local propagation was performed. Current hosted artifact/origin/request gates must still pass in the target kernel. Previous graphical and origin acceptance do not establish the state of a new session.

## Exact candidate and request

Release source: `2e195579dc3d76f6477bf6c885fd537043acdf9d`. Package: 0.4.0rc1. Wheel SHA-256: `2cc0ede68970b2d7de08dc385597861ab1706ad7b08fa4fa2169c5539f8e396e`; sdist: `7f3367ea5a2940b8cd2256ec9c93ab44bfae2f3907798dda1158c96b1fdf2738`. Artifacts are unchanged.

The full proposed request is retained in `request.json`. Its canonical sorted-JSON SHA-256 (excluding the file's final newline) is `46448f7f3dec0af49faf053c8b94426c2ff8c4c6e772e66aba54a71440298ffc`.

The installed RC1 `request_for_controls()` default was compared with the retained Stage 1B experiment request payload and matches exactly. No request was approximated from UI labels.

- Single collimated Gaussian: 0.633 µm wavelength, 1 mW, center (-20, 0) µm, radii (10, 10) µm, zero incidence/azimuth/roll/phase, external index 1; saved group A, incoherent stack setting.
- Bias 0.9144 V; material no=1.5, ne=1.7, K=7e-12, dielectric anisotropy 13; all remaining values preserved in the JSON.
- Grid 128×128, apertures 75×100 µm, length 500 µm, dz=5 µm: 100 slices. Float64 CPU; periodic optical boundary; canonical optical substeps enabled with unchanged limits.
- Local self-consistent Picard-CN/split-step workflow; static RMS residual tolerance 0.005 and maximum residual tolerance 0.02. Coupled/relaxation limits remain 3 and 200. Record iteration history. No convergence or normalization changes.
- Existing controller wall-time budget: **120 seconds**, checked at progress callbacks; this is not a hard preemptive process timeout and can overrun until the next callback. No automatic retry or larger-budget rerun is authorized.

## Reference and comparison policy

Reference: `results/Research/lcprop-colab-interactive-stage-1b-v1/kernel-result/`. Its retained checksum manifest was verified before preparation. The historical 0.3.0 hosted review used these same arrays, achieving 9/9 comparisons and 100/100 converged slices. Reference arrays were not regenerated.

All nine arrays are required: A_initial, A_final, theta, theta_bias, intensity_stack, widths_um, x_um, y_um, z_um. Shapes and dtypes must match; all values finite. Coordinates must be bitwise exact. For each other array require BOTH maximum absolute error <= `1e-12 + 1e-9 * max(abs(reference))` AND relative L2 error <= `1e-9`. These are the original thresholds. No relaxation for a newer stack is allowed.

Convergence requires completed status, 100/100 converged slices, 100 slice summaries and original maximum RMS/max residual gates. Preserve the entire iteration history and metadata, not merely scalar summaries. The previous reference has 6,275 iteration records; record the new count and investigate differences, without inventing a new convergence gate. The complete summary and request are exported. The operational 120-second check is also recorded separately from scientific convergence.

## Prepared files and integrity

- `request.json`: complete proposed scientific request.
- `runtime-sha256.json`: 269 runtime file identities derived from the qualified wheel.
- `preparation.json`: request and retained reference-file hashes, budget and unexecuted status.
- `qualification_helpers.py`: preflight and post-run checks; no function calls Run, execute, LocalRunner or another propagation entry point.
- Local transfer bundle: `.codex-work/beam-stage3k/qualification-inputs.zip`, including those inputs/helper and unchanged reference arrays, experiment and summary. SHA-256: `3bc5b3389ec867050841ed07cebf1cee8dbb2b2700280a025c73563a5279c3dc`.

Nothing was uploaded. Review and transfer this bundle manually only when ready for hosted preparation. Keep the accepted RC1 and production notebooks unchanged.

## Hosted preflight — permitted before execution authorization

1. Open the accepted corrected RC1 qualification notebook in a CPU runtime. Use the qualified install → manual session restart → setup verification → initialization sequence. Retain exact artifact checks, 269-file verification, import ownership and pip consistency checks.
2. Run Stage 3J's origin check in the same kernel. Retain all metadata records without name flattening. Active NumPy/SciPy must agree with selected same-origin metadata. The prior accepted stack was NumPy 2.1.3/SciPy 1.16.3; report any change before execution. Additional system metadata alone is not a fault.
3. Use a controller with no completed record for this one-run trial; do not discard an existing completed record to force that condition. If one exists, export it and prepare a separate session instead.
4. Click **Restore preset**, confirm Beam 2 is off, and choose 120 seconds. Do not press Run. Upload the reviewed input bundle using Colab's Files panel. Run this preparation cell with its actual uploaded path:

```python
from pathlib import Path
import hashlib, zipfile, json
bundle = Path('/content/qualification-inputs.zip')
assert hashlib.sha256(bundle.read_bytes()).hexdigest() == '3bc5b3389ec867050841ed07cebf1cee8dbb2b2700280a025c73563a5279c3dc'
with zipfile.ZipFile(bundle) as z:
    qualification = {}
    exec(compile(z.read('qualification_helpers.py'), 'qualification_helpers.py', 'exec'), qualification)
preflight_record = qualification['preflight'](app_rc1, bundle)
print(json.dumps(preflight_record, indent=2))
```

This rechecks installed bytes, numerical origins, exact request and budget without propagation. It refuses a mismatched draft or existing completed snapshot. **Stop here and supply the preflight output for separate execution authorization.** Do not edit draft controls after this check; if edited, rerun preflight. The one-click controller captures the kernel-accepted draft; no browser-visible atomicity beyond synchronized Python controls is claimed.

## After separate authorization only — one scientific run

Recheck preflight immediately before the authorized run, then press the existing packaged **Run** button exactly once. Do not call a helper to simulate a click, and do not run twice. Retain status/progress screenshots and final elapsed time. If timeout, convergence, rendering or identity fails, preserve any prior completed record and stop; no retry is preauthorized. Do not alter the package's failure handling.

On successful completion, before editing the draft:

```python
checkpoint = qualification['capture_completed'](app_rc1, preflight_record,
                                              '/content/lcprop-stage3k-evidence')
comparison = qualification['compare_arrays'](checkpoint, bundle)
print(json.dumps(comparison, indent=2))
```

The first call retains snapshot references, complete arrays/request/summary, displayed output and executed-parameter HTML, and creates the initial export using the packaged export function. A comparison failure leaves the saved record intact. The evidence folder must be new; do not overwrite another run.

Then manually change the draft only: x=-15 µm, y=2 µm, principal radii 11 and 9 µm, power=0.8 mW, voltage=0.9 V, incidence=0.01 rad, azimuth=0.2 rad, and phase=0.3 rad. Keep Beam 2 off. These are post-run edit probes, not additional scientific requests to execute. Verify the draft preview/controls change while completed plots and View executed parameters remain fixed. **Do not press Run again.**

```python
preservation = qualification['check_after_draft_edits'](app_rc1, checkpoint)
print(json.dumps(preservation, indent=2))
```

Checks require the same completed object, byte-identical arrays/request/summary (including histories and execution metadata), unchanged completed plot output and presentation, unchanged prior export files and identical re-export file hashes. The second export uses a new directory. Any mismatch is retained and reported; it is not normalized away. Export-generated PNG identity is tested within this same runtime, not across different Matplotlib versions.

Retain the whole evidence directory, setup/environment/origin outputs and screenshots for independent review. It contains before/after exports, frozen request, complete diagnostics/history, visualization metadata, PNGs, checksum manifests, preflight, numerical comparisons and draft-preservation results. Do not infer browser painting from Python output equality.

## Local bounded validation

Existing disposable RC1 environment was used without installation. Preflight verified all 269 installed files and numerical origins; exact default request passed and a shifted request was rejected. Using the retained CompletedRun arrays/summary as a fixture (not new execution), all nine self-comparisons passed and capture/re-export preservation checks passed after a fake draft edit. This validates helper plumbing and unchanged export behavior on retained data, not new numerical parity or real hosted control interaction. Temporary checks: `.codex-work/beam-stage3k/check.py` and `retained-fixture-check/`.

## Current results and gate

Hosted execution, convergence, elapsed time, numerical comparison and post-edit snapshot/export acceptance are **PENDING**. There is no new scientific result to report. Preparation establishes local request/reference parity and a bounded procedure only. **GO for hosted preflight; HOLD scientific execution until its current-session gates pass and the user separately authorizes exactly one Run.** No commits, pushes, installations, package changes or simulations occurred.

## Hosted execution and preservation — user-reported completion

Following the prepared preflight, the user separately authorized one run and executed it through the packaged Run button. Preflight reported the exact expected request SHA-256, all 269 runtime identities and matching NumPy 2.1.3/SciPy 1.16.3 origins. The supplied UI screenshot reported all 100 slices converged and displayed the expected request identity.

The hosted comparison reported **9/9 scientific arrays passing the original tolerances**, coordinates bitwise exact, convergence passed, **35.377490939999916 seconds** within the 120-second budget and **6,275 iteration records**. No new run or tolerance change was requested. The reported floating-point differences match the magnitudes of the retained 0.3.0 comparison (including A_final max absolute error 1.1937414660825033e-13 and relative L2 5.495412631679486e-13).

After changing draft x/y, both radii, power, voltage, incidence, azimuth and phase to the prescribed values, all nine snapshot checks reported true: snapshot object identity, frozen request, array bytes, diagnostics/metadata, completed plots, summary presentation, executed parameters, prior export integrity and re-export identity. The user additionally confirmed: “Plots did not change, draft image did change.” The changed draft retained a single beam and the original solver settings; it was not executed.

This establishes **user-observed hosted PASS for the bounded single-beam calculation and post-completion draft-edit isolation**. It does not qualify additional scientific beam configurations, altered numerical settings or a second execution. The complete exported arrays, request, histories, metadata, PNGs and before/after checksums remain to be transferred and independently reviewed. Summary evidence is retained in `hosted-user-reported-results.json`; do not treat that summary as an independently verified archive or replace the original exports with it.

Next: preserve `/content/lcprop-stage3k-evidence` as a ZIP and retrieve it before the Colab session is deleted. No further calculation is required or authorized by this evidence-retention step. Earlier preparation and current-pending statements above are historical; current status is hosted reported PASS, archive review PENDING.

## Closure — independent archive review

Reviewed the user-supplied `lcprop-stage3k-evidence.zip` in place. SHA-256: `65aa3736fd8e08783ce70f76396c7761f30197d76b4f81ef5665ce642b547199`. The archive contains 15 files plus two directory entries, with unique safe names. Both five-file export checksum manifests verify. Every one of the six before/after export files, including manifests and PNGs, is byte-identical. Original archive bytes remain unchanged.

The exported request payload equals both preflight and the prepared reference exactly. The request digest is `46448f7f3dec0af49faf053c8b94426c2ff8c4c6e772e66aba54a71440298ffc`. Environment metadata records RC1, the qualified wheel digest in direct_url and the expected controller source digest; numerical versions match the accepted hosted origins. The preflight records 269 runtime files verified. That remote verification is retained evidence rather than a new remote filesystem inspection.

Independently loaded arrays with pickle disabled: **9/9 PASS** for original shape/dtype/finiteness and numerical gates, coordinates bitwise exact. All 100 slice summaries report convergence under RMS <=0.005 and max <=0.02. Summary maxima are **0.004935085357116411** and **0.019985181428802123**. Elapsed time is **35.377490939999916 seconds**, within 120 seconds. All discrete slice and iteration identity fields match the retained reference, with **6,275 iteration records**. Floating history equality was not asserted.

An initial review assertion compared locally recomputed metric JSON for exact equality with hosted metric JSON; platform-dependent norm rounding made that assertion overly strict. The scientific array thresholds themselves passed unchanged. The review now checks metric-report agreement to relative 1e-12 (zero absolute tolerance) separately from the original, unmodified scientific gates. This is not a change to array acceptance tolerances.

Visualization metadata was independently reconstructed from the frozen request and retained coordinates and matches both exported visualization.json and summary metadata. The exported PNG was inspected: completed voltage 0.914 V and single-beam labels, slices at the retained x/y sample positions, intensity/reorientation views and width curves are present and agree with the supplied completed-screen view. Before/after PNG byte identity independently confirms unchanged exported plot content.

The archive retains the intended changed draft and all nine true preservation assertions. Independently verified evidence establishes unchanged request, arrays, full diagnostics/metadata/history and exports. In-memory object identity and widget callback/presentation history remain recorded hosted assertions, supported by the user's visual confirmation; they cannot be re-observed from a ZIP alone. No new calculation was executed during review.

Detailed member hashes, numerical metrics and verification flags: `independent-archive-review.json`. Review script: `.codex-work/beam-stage3k/review_archive.py`. No package/controller/request-schema changes, installations, commits or pushes occurred.

**Final result: PASS for the single authorized hosted CPU preset, original nine-array parity gates, convergence, frozen-request/metadata preservation and unchanged exports following the prescribed draft edits.** This does not qualify additional runs, multibeam scientific cases, different numerical settings or a general release/publication action. The hosted manual restart requirement remains. Earlier pending statements above describe the chronological preparation/review stages and are superseded by this closure.
