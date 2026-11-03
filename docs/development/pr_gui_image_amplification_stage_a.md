# PR GUI Image-Amplification Stage A

**Date:** 2026-08-26
**Branch:** `feature/pr-second-order-static`
**Baseline HEAD:** `4360b4c4714996d635d6b308018d86d5a2d85915`

## Purpose

Stage A exposes the existing validated reduced time-dependent PR image-
amplification calculation through the canonical PR GUI and local registered-
operation path. It adds no propagation or material physics and does not use the
legacy static-streaming, transverse-static, or full-transverse TD solvers.

## Architecture Mapping

The implemented path is:

```text
PRImageSource
    + PRImageLaunchSpec
    + existing GridSpec / PRMaterialSpec / PRSolverOptions / BackendSpec
                         |
                         v
PRImageAmplificationRunRequest
                         |
                         v
PR_IMAGE_AMPLIFICATION_OPERATION
                         |
                         v
prepare_image_amplification_workflow_request()
                         |
                         v
existing run_pr_timedependent()
                         |
                         v
existing carrier isolation, back-propagation, gain, and fidelity metrics
                         |
                         v
pr_image_amplification_result_to_run_data()
```

The GUI does not call solver internals. The composite operation derives the
runtime `initial_A` immediately before the existing TD workflow. The scientific
request instead owns the decoded source and immutable launch policy.

## Source and Grid Separation

`PRImageSource` owns source pixels and provenance: kind, display name, basename,
SHA-256, original pixel dimensions, encoded format, decoded mode, preprocessing
policy, and an immutable conventional `(row_y, column_x)` grayscale array.

`GridSpec` independently owns `Nx`, `Ny`, transverse apertures, and longitudinal
sampling. Loading or previewing a source never changes the grid. Only the
resampled `(Nx, Ny)` transmission and two-channel launch enter numerical
execution.

## Decoding and Preprocessing

User raster files are inspected and decoded with Qt `QImageReader`, already in
the GUI dependency set. Stage A supports the intersection of Qt's installed
formats with BMP, JPEG, PGM, PNG, PPM, TIFF, and WebP. Source files are limited
to 512 MiB and decoded images to 100 million pixels. Preview decoding requests a
bounded 320 by 220 image from Qt.

Scientific grayscale conversion is explicit BT.601 integer luminance:

```text
L = (19595 R + 38470 G + 7471 B + 32768) >> 16
```

This matches historical Pillow `convert("L")` for the tested representative
colors. Alpha is discarded and is not an optical mask.

The existing `prepare_image_transmission()` policy remains authoritative:

1. normalize maximum intensity to one;
2. optionally invert;
3. pad to an even square with transmission one;
4. rotate conventional image axes into LCProp `(x, y)`;
5. resize with SciPy nearest-neighbor interpolation;
6. use a square physical footprint centered on the signal launch;
7. retain transmission one outside the footprint;
8. map intensity transmission to field transmittance with
   \(t=\sqrt{T}\), then apply \(A_{\mathrm{out}}=tA_{\mathrm{in}}\).

The headless compatibility path retains historical clipping behavior. GUI image
requests set an explicit full-footprint policy and reject placement outside the
aperture rather than clipping silently.

## Power Semantics

The GUI power controls specify the physical powers of the pump and signal
incident on the image element. The ordinary launch builder first prepares the
two incident fields using those powers and its established normalization:

\[
\sum_c\int |A_c(x,y)|^2\,dx\,dy=1.
\]

The image-bearing signal is then transformed without any subsequent channel or
stack renormalization. For the current grayscale intensity transparency
\(T(x,y)\),

\[
t(x,y)=\sqrt{T(x,y)},\qquad
A_{\mathrm{signal,out}}=tA_{\mathrm{signal,in}}.
\]

Consequently, absorption reduces the signal power and total power entering the
PR medium. The post-element powers are computed from the actual discrete field
using \(\sum |A|^2\,dx\,dy\), scaled by the incident total power. The pump is
unchanged because the element acts only on the signal.

The underlying element operation accepts any passive complex field
transmittance

\[
t(x,y)=a(x,y)e^{i\phi(x,y)},\qquad |t|\leq 1.
\]

A pure phase element changes the complex field while preserving pointwise and
integrated intensity. Active values with \(|t|>1\) are rejected. Stage A does
not interpret image files as phase or complex masks; that GUI capability is
deferred.

The historical `make_image_amplification_request()` and
`run_image_amplification()` benchmark APIs retain their established peak-ratio
rescaling and unit-total normalization for published/research comparisons. The
new registered GUI operation instead uses physical incident-power semantics.
The two paths are deliberately explicit rather than silently conflated.

## GUI Controls

The PR application now has an `Input` tab with:

- Gaussian beams or Image amplification input mode;
- Standard/User source selector;
- native user-file chooser, filename, and preview;
- inversion;
- square physical image size;
- pump incident power;
- signal incident power;
- wavelength, x/y beam waist, and periodic positive carrier mode.

The signal/pump incident power ratio is derived from the two power controls; it
is not an additional amplitude constraint. This avoids overdetermining an
already power-normalized signal. Result provenance records incident and
post-element channel powers, total incident power, power entering the PR
medium, signal throughput, and transparency policy.

Selecting Image amplification fixes Evolution to the validated reduced time-
dependent workflow and disables the ordinary free-form Beam tab. Material,
Grid, Evolution-step, backend, runner, and Results components remain shared.
Stage A supports Local execution only.

## Standard Assets

The installed, data-driven catalog is currently empty. Historical names include
MNIST 0 through MNIST 9 and the AF Resolution Chart, but no exact source file in
the current repository has sufficient approved redistribution provenance for
production packaging. No research rendering or checksum-mismatched copy was
substituted. The Standard selector is disabled with an actionable explanation;
User Image remains fully available.

Future approved assets belong in `src/lcprop/pr/assets/images/`, with stable ID,
display name, filename, checksum, dimensions, decoded mode, and default
inversion in `catalog.json`. They are loaded with `importlib.resources`.

## Result Products

The image operation adds these products to the ordinary reduced-TD PR products:

- decoded source image;
- simulation-grid transmission;
- image-bearing input signal intensity;
- isolated output signal intensity;
- back-propagated amplified image;
- zero-response reconstruction;
- signal-carrier mask;
- linear and logarithmic far fields;
- existing measured/analytic gain, image correlation, normalized RMSE, power
  drift, and source/launch provenance diagnostics.

Rectangular-grid far-field coordinates remain independent in x and y. The
generic image view now preserves equal-coordinate aspect for x-y, source-pixel,
and direction-cosine planes without changing far-field mathematics.

## Deferred Work

- approval and packaging of exact standard source assets;
- image-source experiment persistence;
- remote transport and Slurm image execution;
- image-amplification checkpoint continuation;
- transverse-static and full-transverse TD scientific promotion;
- phase images, alpha-mask physics, crop/stretch choices, and image editing.

These are separate bounded milestones. Stage A intentionally does not redesign
the shared experiment or remote-result envelopes.
