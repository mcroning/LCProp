# Shared Beam Input-Screen Primitive — Stage B1

**Date:** 2026-08-26

**Branch:** `feature/pr-second-order-static`

**Baseline commit:** `bb866d21db07520d3dd9c368ca9ab8aa0bb96236`

## Purpose

Stage B1 extracts the numerically generic launch-plane screen behavior proven
by the PR image-amplification milestone into shared optics ownership. It does
not change either material GUI, add persistence, or expose a new user-facing
workflow. PR continues to own image-amplification roles, propagation,
reconstruction, and metrics.

## Ownership

`src/lcprop/optics/screens.py` owns the material-neutral scientific contract:

- `RasterSource` stores immutable decoded raster samples and portable source
  provenance;
- `ScreenPlacement` stores physical placement, resampling, exterior, and
  boundary policies;
- `IntensityRasterScreen` interprets a raster as intensity transmission;
- `ChannelLaunchElements` assigns an ordered tuple of elements to one
  canonical enabled-channel index;
- runtime-grid preparation and passive field multiplication;
- validation of passive transmittance and channel assignments.

The shared screen module imports neither `lcprop.lc` nor `lcprop.pr`.
`lcprop.optics.launch.build_launch()` consumes the ordered assignments and
reports both incident and post-element powers.

PR retains `PRImageSource` as a compatibility subclass of `RasterSource` with
the established PR preprocessing identity. The public PR functions
`prepare_image_transmission()`,
`intensity_transmission_to_field_transmittance()`, and
`apply_passive_field_transmittance()` remain available through compatibility
wrappers or re-exports. The PR image-amplification request builder now delegates
screen preparation and application to the shared implementation.

## Scientific Contract

For an intensity-transmission raster

\[
T(x,y) \in [0,1],
\]

the field transmittance is

\[
t(x,y)=\sqrt{T(x,y)},
\]

and the launch element acts as

\[
A_{\mathrm{out}}(x,y)=t(x,y)A_{\mathrm{in}}(x,y).
\]

The lower-level field operation also accepts passive complex transmittance
with \(|t|\leq 1\). A pure phase screen therefore changes phase without
changing integrated power.

The application order is fixed:

1. construct each incident channel from its `BeamChannel` definition;
2. normalize incident channels to the requested physical-power fractions;
3. apply each channel's launch elements in tuple order;
4. measure post-element channel powers and throughput;
5. pass the transformed fields to the material workflow.

There is no post-screen renormalization. `BeamChannel.power_mW`,
`LaunchResult.physical_powers_mW`, and
`LaunchResult.physical_total_power_mW` retain their incident-power meaning.
The explicit post-element fields report transmitted channel powers, transmitted
total power, and per-channel throughput.

Channel indices refer to the canonical channel sequence after disabled editor
beams have been filtered by the existing LaunchPlane adapter. Assignments are
explicit, unique per channel, range-checked, and ordered deterministically.
Shared metadata does not acquire PR-specific pump or signal roles.

## Historical Raster Policy

The named policy `even_square_nearest_transparent_v1` preserves the Stage A
operation order:

1. require a finite, nonnegative raster and normalize by its maximum;
2. apply optional intensity inversion;
3. pad to an even square using the declared exterior transmission;
4. rotate from image row/column orientation to LCProp x/y orientation;
5. resample with nearest-neighbor interpolation;
6. place on the selected runtime grid;
7. reject an out-of-aperture footprint by default, or clip only when requested
   explicitly.

This is one versioned compatibility policy, not a claim that every future
screen must use the same preprocessing.

## Source/Grid Invariant

Raster dimensions never determine `Nx`, `Ny`, physical aperture, or z
sampling. Sources remain host-side decoded rasters and are prepared only after
the caller supplies a `RuntimeGrid`. Rectangular sources and rectangular
runtime grids are independent.

## Stage A Equivalence

A deterministic 32 by 8, float64, two-material-step image-amplification case
was captured before extraction and replayed after extraction. The following
were bit-for-bit identical (`numpy.array_equal`, maximum absolute difference
zero):

- prepared intensity transmission;
- initial and final optical channel volumes;
- input and output signal fields;
- reconstructed and zero-response back-propagated fields;
- incident and post-screen channel powers;
- gain, correlation, normalized RMSE, and normalized power drift.

The comparison artifact was kept outside the repository at
`/private/tmp/lcprop-stage-a-screen-b1-baseline.npz`; it is validation scratch
data and is not part of the proposed change.

## Validation Scope

Focused tests cover no-screen compatibility, identity, attenuation, opacity,
nonuniform intensity, pure-phase and complex passive transmittance, active-mask
rejection, channel isolation, ordered multiplication, enabled-channel index
resolution, power and throughput reporting, source/grid independence,
rectangular geometry, boundary rejection, PR compatibility, and the
material-neutral import boundary.

The complete validation commands and exact counts are recorded in the Stage B1
implementation report produced before pre-commit review.

## Deferred Work

Stage B1 deliberately defers:

- B2 BeamPanel screen-editor and source-selection work;
- B3 PR Image Amplification GUI rewiring and removal of duplicated image
  controls;
- pump/signal GUI role selection;
- persistence and remote-transport schemas for screens;
- phase-image loading, lenses, generalized apertures, and active masks;
- LC GUI exposure.
