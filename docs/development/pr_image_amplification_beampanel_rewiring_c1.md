# PR Image Amplification BeamPanel Rewiring (Stage C1)

## Purpose

Stage C1 converts the specialized PR image-amplification GUI from a second
beam-launch editor into an experiment and analysis layer over LCProp's shared
`BeamPanel`. The numerical backend remains the validated reduced
time-dependent PR workflow.

## Ownership

The shared Beam tab is authoritative for enabled beam channels, wavelength,
incident power, waist, position, transverse phase gradient or launch angle,
phase, coherence group, and ordered channel input screens. It produces one
immutable `LaunchConfiguration` containing the canonical enabled-channel
`BeamStack` and its `ChannelLaunchElements`.

The PR Image Amplification input panel now owns only two experiment roles:

- Pump channel
- Signal channel

The selectors display the canonical enabled-channel index and beam name. They
track uniquely named beams through reorder operations and clear a role when
its selected beam is disabled or otherwise becomes stale. The roles are not
added to the generic `BeamChannel` model.

The duplicated Stage A power, wavelength, waist, Fourier-mode, image-source,
inversion, footprint, and preview controls were removed from the specialized
panel. Image loading, placement, inversion, preprocessing, transmission
preview, and transmitted-power preview remain in the shared input-screen
editor on the Beam tab.

## Physical Request

`PRBeamPanelImageAmplificationRunRequest` contains:

- the runtime grid;
- PR material and reduced-TD solver settings;
- backend and precision;
- the shared `LaunchConfiguration`;
- canonical pump and signal channel indices.

Stage C1 deliberately requires exactly two enabled channels. The roles must
be distinct, have positive incident powers, share a coherence group and
wavelength, and use symmetric carriers in the x-z plane. The pump must have
no image screen. The signal must have exactly one `IntensityRasterScreen`.
These restrictions preserve the validated carrier-isolation and analytic-gain
contract while producing actionable errors for ambiguous configurations.

The shared `build_launch()` function normalizes the incident channels from
their configured physical powers and then applies the signal screen once.
The prepared field is passed as `initial_A` to the existing
`run_pr_timedependent()` workflow; declarative launch elements are not passed
again, so the screen cannot be double applied.

## Power and Geometry Semantics

Pump and signal powers are the incident `BeamChannel.power_mW` values. Passive
screen attenuation reduces the signal power entering the PR medium, and the
field is not renormalized afterward. The analytic input ratio is therefore
the configured incident signal power divided by configured incident pump
power. Transmitted powers and signal throughput are measured from the shared
prepared launch.

Carrier gradients come directly from the selected channels. The signed
normalized grating wave number is

\[
K = \frac{q_{x,\mathrm{signal}}-q_{x,\mathrm{pump}}}{k_0}.
\]

This is the existing Stage A sign convention: for pump \(+q_x\) and signal
\(-q_x\), the grating is \(-2q_x/k_0\). No parallel PR beam geometry is
reconstructed.

## Compatibility and Equivalence

The historical `PRImageAmplificationRunRequest`,
`make_image_amplification_request()`, and `run_image_amplification()` APIs are
unchanged. They retain the established paper/research benchmark behavior,
including the historical unit-normalized peak-ratio path where applicable.
The registered `PR_IMAGE_AMPLIFICATION_OPERATION` accepts both the historical
physical Stage A request and the new BeamPanel-composed request.

An equivalent Stage A configuration produces bit-for-bit identical prepared
transmission and input channel fields. A complete reduced-TD comparison also
produces identical material and optical fields, isolated signal,
back-propagated reconstruction, powers, gain, correlation, normalized RMSE,
power drift, status, iteration count, and diagnostics.

## GUI Behavior and Results

The Beam tab remains enabled in Image Amplification mode. Switching between
ordinary PR and Image Amplification does not mutate beam definitions or input
screens. The specialized workflow continues to use the canonical PR result
workspace and preserves source/transmission fields, input and isolated output
signal fields, amplified and zero-response reconstructions, carrier mask,
far-field products, gain and fidelity metrics, power accounting, and runtime
diagnostics. Product provenance now records the selected pump and signal
canonical indices and the shared screen preprocessing policy.

## Persistence, Remote Execution, and Checkpoints

Stage C1 is local-only. Image-amplification experiment persistence remains
explicitly rejected by the PR GUI, and Slurm execution remains limited to the
already supported canonical transverse-static workflow. Image-amplification
checkpoint continuation is unchanged and unavailable. No screen, role, or
source state is silently dropped.

## Validation

Focused validation covers:

- shared screen primitives and BeamPanel editing;
- ordinary PR TD/static/transverse-static screen consumption;
- Beam tab availability and mode-switch preservation;
- pump/signal role validation, reorder tracking, and stale-role clearing;
- exact two-channel, shared-coherence, matched-wavelength, x-z carrier, and
  symmetric-carrier request constraints;
- specialized pump-screen and multiple-signal-screen rejection;
- signal-screen ownership and single application;
- a signal assigned to canonical channel zero;
- prepared-launch and complete Stage A numerical equivalence;
- historical image-amplification APIs and result products;
- explicit persistence rejection.
- explicit pre-dispatch Slurm rejection without role or screen downgrade.

The complete PR suite is the final local regression gate for this milestone.
The validation-completion pass added nine targeted cases; the combined
B1/B2/B3 and Image Amplification suite passes 62 tests, and the complete PR
suite passes 394 tests with 57 CuPy-dependent skips on the local CPU host.

## Deferred Work

Stage C1 does not add specialized static or full-transverse image
amplification, phase or complex-image input, standard-image packaging,
experiment persistence, remote image transport, LC image amplification, or
material-solver changes.
