# PR Image Amplification D1 GUI Completion Regression

**Date:** 2026-08-27
**Branch:** `feature/pr-second-order-static`
**Baseline:** `7cc0144a34e18f8e4f743f5b429410174e50aa48`

## Purpose

This record documents the bounded correction of a GUI completion regression
introduced when the D1 Image Amplification path began returning
`PRImageAmplificationCompositeResult`. The scientific calculation and product
construction completed successfully; the PR main window then failed while
rendering its final power summary.

## Symptom and Root Cause

The completed local calculation produced the ordinary PR products, all six
Image Amplification post-processing stages, and the specialized gain and image
products. `PRMainWindow._on_finished()` nevertheless attempted to read
`power_initial` and `power_final` directly from the composite result. Those
quantities belong to the preserved ordinary base result available through
`PRImageAmplificationCompositeResult.run_result`.

The exception was therefore a presentation-interface mismatch, not a physics,
propagation, reconstruction, or product-adapter failure.

## Completion-Interface Correction

For an Image Amplification runner result, the completion handler now resolves
one authoritative ordinary result from `result.run_result`. For every other PR
workflow, the runner result remains its own ordinary result. The resolved
ordinary result supplies:

- material-time completion information for the current reduced-TD base;
- normalized propagated optical power before and after the PR propagation.

The composite remains authoritative for the experiment-level status,
analysis status, analysis message, and specialized Image Amplification result.
No convenience fields were added to the composite and no scientific quantity
is recomputed in the GUI.

The displayed normalized optical-power line continues to describe the base
workflow's prepared launch and propagated output. It is distinct from the
incident channel powers and post-screen launch-power diagnostics retained in
the Image Amplification products.

## Status Handling

The existing D1 distinctions remain intact:

- cancellation is presented as stopped;
- analysis failure is presented as failed;
- successful base execution and analysis are presented as completed;
- a preserved base `not_converged` status is presented as not converged and is
  not relabeled as successful or as an analysis failure.

## Regression Coverage

The focused GUI regression obtains a real D1 composite result, passes it
through `PRMainWindow._on_finished()`, and verifies:

- no post-completion exception or `ERROR` entry;
- completed status and workspace installation;
- retention of the ordinary output-intensity product;
- retention of the reconstructed amplified and zero-response images;
- retention of the Image Amplification diagnostic product;
- exact use of the base result's normalized initial and final power values;
- equality of the presented reconstructed-image product and the common
  postprocessor output.

A second focused regression verifies that an analysis completed over a base
result marked `not_converged` remains visibly not converged. Existing ordinary
TD, legacy-static, and transverse-static GUI completion tests provide the
ordinary-result compatibility gate.

## Scientific Non-Change

This correction changes only PR GUI completion reporting and its regression
coverage. It does not change material physics, optical propagation, launch
screens, carrier isolation, back-propagation, gain, correlation, NRMSE,
tolerances, solver behavior, or D1 progress/cancellation behavior. The
coherent-grating undersampling warning remains unchanged.

## Validation

- Focused composite completion/status tests: `7 passed`.
- Affected ordinary and Image Amplification GUI suite: `63 passed`.
- Complete PR suite: `408 passed, 57 skipped`; the skips are existing
  optional-backend gates.

## Deferred Work

Multi-algorithm Image Amplification GUI selection and transverse-static Image
Amplification validation remain paused. Remote execution and experiment
persistence are also outside this correction.
