# PR Image Amplification Results Framing and Diagnostics

## Purpose

This record documents the presentation-only refinement of photorefractive
Image Amplification results. The change improves the initial image framing and
surfaces the existing post-processing metrics through canonical `RunData`
products. It does not change launch fields, optical propagation, PR material
physics, carrier isolation, reconstruction, or solver behavior.

## Default Image Framing

Image Amplification fields remain stored on the complete runtime grid with
their original physical coordinates. The product adapter now supplies a
separate initial display extent for the transverse image products, and the
shared image viewer uses that extent only for its initial axes limits. The
image artist and canonical product retain the full-aperture extent and data;
the presentation change therefore does not crop or discard any region.

The selected region is the union of:

- the signal beam envelope `x0 +/- 2*waist_x` and
  `y0 +/- 2*waist_y`; and
- the exact rectangular raster-screen footprint defined by its physical
  center, width, and height.

`BeamChannel` waists are the 1/e^2 intensity radii. The two-waist envelope is
the same convention used by the established PR aperture validation. A margin
equal to 15% of the union width is added independently along each transverse
axis. The result is clamped to the full numerical aperture. Consequently the
rule supports offset beams, offset screens, unequal beam waists, rectangular
screens, and rectangular runtime grids without imposing a square view.

If the signal-channel or screen-placement provenance is absent or invalid, no
display extent is emitted and the viewer retains its previous full-aperture
default. The Back-Propagated Amplified Image is the initially selected image
product for a completed Image Amplification experiment. Related physical-plane
image products receive the same initial extent; source-pixel and Fourier-plane
products do not.

## Scalar Diagnostics

The PR Image Amplification product adapter now adds a dedicated
`image_amplification_metrics` diagnostic block containing canonical values
already computed by the postprocessor:

- measured and analytic signal gain;
- reconstructed-image correlation and normalized RMSE;
- incident and post-screen signal power;
- passive screen throughput;
- carrier-isolated reference and output signal power;
- total power entering the PR medium; and
- normalized optical-power drift.

The earlier, more detailed `image_amplification` diagnostic block remains for
compatibility. Ordinary base-workflow diagnostics and convergence curves are
passed through unchanged.

## Measured-Gain Convention

The existing scalar measured gain is preserved exactly. Let `A_in` be the
total coherent field after the launch screen at z=0, `A_out` the final total
coherent field, and `C(.)` the established signal-carrier isolation operation
using the canonical Fourier mask. With the runtime-grid measure `dx*dy`,

```text
P_reference = sum(|C(A_in)|^2) * dx * dy
P_output    = sum(|C(A_out)|^2) * dx * dy
G_measured  = P_output / P_reference
```

Thus the gain reference is the carrier-isolated post-screen field entering the
PR medium, not the pre-screen incident signal power. The diagnostic block
reports both this reference power and the isolated output power. Their mW
values use the same launch normalization already used by the postprocessor.

## Gain Versus z Decision

No measured gain-versus-z curve is added in this milestone. The canonical
Reduced TD, legacy static, Transverse Static, and Transverse TD base results
retain input and output complex optical fields, plus selected real material or
intensity volumes, but they do not retain the complex optical field at every z
plane. Total intensity alone cannot separate coherently overlapping pump and
signal carriers using the established carrier-isolation convention.

Computing a scientifically consistent curve would therefore require a second
optical replay or a new retained complex-field product. Both are explicitly
outside this presentation milestone. The diagnostics record
`measured_gain_vs_z_available = false` with this reason. No curve is fabricated
from total intensity or independent spatial masks.

An analytic gain-versus-z curve is also omitted. Adding an analytic curve
without the corresponding measured curve would not answer how gain accumulates
in the computed field, and no new length-dependent convention is introduced.

## Computational Cost

The framing calculation operates on scalar request metadata. The new scalar
diagnostics reuse powers and metrics already computed during Image
Amplification analysis. No propagation, material solve, FFT, field copy, or
large retained volume is added. The only new per-result storage is four scalar
power values and small presentation metadata.

## Validation Scope

Focused validation covers:

- centered and offset signal screens;
- a signal beam away from the optical origin;
- rectangular screens and grids;
- screen-dominated and beam-dominated framing;
- aperture clamping and missing-provenance fallback;
- preservation of full stored arrays and coordinates;
- initial selection and axes limits in the shared image pane;
- exact agreement between reported power ratios and the established scalar
  measured gain;
- dedicated Diagnostics-tab presentation;
- preservation of base curves; and
- Reduced TD and Transverse Static Image Amplification paths.

## Scientific Non-Change

This work changes presentation metadata and exposes existing analysis values.
It does not alter PR equations, optical or material stepping, convergence
criteria, screen application, carrier sign or mask, back-propagation,
reconstructed fields, gain values, correlation, normalized RMSE, or power
semantics.
