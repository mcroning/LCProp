# General PR beams and carrier-power diagnostics

Status: **Accepted future Product contract; implementation NOT AUTHORIZED.**

Date: 2026-09-20. Source baseline:
`bbb42dd3bf5367adc411f2764cd251eb2e2f6e00` on
`feature/pr-second-order-static`.

This P2B-1 decision preserves the source-qualified N-beam architecture audit
and the relevant qualification from the subsequent intensity-reference audit.
It defines propagation and diagnostic contracts, not a new gain convention,
solver, schema, or claim of scientific qualification for every supported input.
It follows the [Product ownership architecture](LCProp_Target_Architecture.md).

## Existing implemented capability

- [BeamStack](../../src/lcprop/core/beams.py) accepts at least one channel
  without a two-channel ceiling. The four ordinary reduced/full-transverse,
  static/TD requests already carry beams and declarative launch elements;
  [GUI request construction](../../src/lcprop/pr/gui/request_adapter.py) selects
  the material workflow independently of ordinary screen presence.
- [LaunchConfiguration](../../src/lcprop/optics/launch_configuration.py) and
  [ChannelLaunchElements](../../src/lcprop/optics/screens.py) support optional
  assignments to arbitrary enabled channels, multiple screened channels, and
  ordered intensity-raster elements. The shared
  [screen editor](../../src/lcprop/gui/panels/input_screen_editor.py) already
  supports multiple screened beams, but only one editable screen per channel
  and its supported sampling/boundary policies. General declarative phase
  screens or lenses are not thereby implemented.
- [Launch construction](../../src/lcprop/optics/launch.py) normalizes incident
  channels using requested powers before applying passive elements, without
  renormalizing afterward. Coherent groups are combined as fields within each
  group and as intensities between groups by
  [total_intensity](../../src/lcprop/optics/splitstep.py).
- The current [Image Amplification composite](../../src/lcprop/pr/image_amplification.py)
  prepares an ordinary base request and dispatches one ordinary operation,
  then adds specialized analysis. Its validator still requires two channels,
  selected Pump/Signal roles, one signal screen, no pump screen, positive
  powers, common coherence/wavelength, and symmetric x-z carriers. These
  experiment restrictions do not define general PR physics.
- The existing [carrier-power diagnostic](../../src/lcprop/pr/carrier_power.py)
  measures entrance/output spectral powers for exactly two channels in one
  coherence group. It is not an implemented N-carrier longitudinal or
  material-time history facility. Its `carrier_gain` is `P_out/P_in`, and
  `carrier_delta_power` is `P_out-P_in`, as asserted by the
  [carrier tests](../../tests/test_pr_carrier_power.py).
- [Experiment codecs](../../src/lcprop/pr/experiment_codec.py) already preserve
  ordinary beam/screen requests and specialized composite experiments. Current
  PR request schema version is 6. Historical development records describe
  earlier milestones and must not override current source capability.

These are source-level capabilities, not qualification of arbitrary N-beam
physics. The [PR model contracts](../science/pr_model_contracts.md) remain
authoritative for shared-wavelength requirements, reduced x-directed transport,
transverse electrical profiles, and other genuine model restrictions.

## Canonical propagation and optional analysis

**Ordinary PR propagation is the canonical scientific workflow and accepts
N >= 1 supported optical beams.** Each beam may independently have no screen
or an intensity raster screen, together with its ordinary supported power,
position, tilt, phase, waist/profile/focus, coherence group, and launch settings.
Multiple beams may carry screens simultaneously.

The accepted configuration space includes one beam with or without an image,
two or more beams without images, an image on any supported beam, and multiple
image-bearing beams. Image presence or absence does not select a material
equation. A missing screen must not invalidate otherwise valid propagation.
Supported geometry must not be rejected merely because a selected image
analysis assumes special geometry. Model, launch, sampling, and physical
admissibility restrictions remain in force; this is not blanket acceptance of
arbitrary parameters or all-zero illumination.

Keep three responsibilities separate:

1. Propagation through the selected ordinary material workflow.
2. Optional presets/configuration, including Two-Beam Coupling and Image
   Amplification.
3. Optional specialized image analysis.

Pump/Signal roles, carrier resolvability, reconstruction geometry, analytic
two-wave assumptions, and screens needed for image metrics may remain analysis
requirements. Their failure or inapplicability must not invalidate successful
ordinary propagation. Preserve propagation status and separately report
analysis status and reasons. A screen-free Image Amplification preset must
still permit the equivalent ordinary calculation.

This separation is a future contract. It does not claim that today's composite
validation/status behavior already implements it, nor does it redefine
historical API results. The
[composite execution record](../development/pr_image_amplification_composite_execution_d1.md)
documents the existing delegation and historical compatibility obligations.

## Carrier powers and existing Fourier convention

For coherence group g, form the total coherent optical field at plane z:

```text
A_g(x,y,z) = sum_{j in g} A_j(x,y,z).
```

Carrier power is attributed from this group field, not from the norm of an
individually propagated input-lineage array. Lineage norms can remain unchanged
while the coherent field redistributes angular power. Never coherently sum
mutually incoherent groups to perform this measurement.

For launched carrier centers k_j within each group, the intended extension is
a qualified N-carrier Voronoi partition of transverse spatial-frequency space:

```text
P_j(z) = integral_{V_j} |A_tilde_g(k,z)|^2 dk
```

The integral notation denotes the existing Parseval-normalized measurement,
not a new Fourier convention. Preserve the current discrete implementation:

```text
kx = 2*pi*fftfreq(Nx, dx_um)
ky = 2*pi*fftfreq(Ny, dy_um)
P_j = dx_um*dy_um/(Nx*Ny)
      * sum_k w_j(k) * |fft2(A_g)(k)|^2
```

The forward FFT is unnormalized and unshifted. Carrier centers are launch
phase gradients in rad/um. Regions remain anchored to the launched centers;
do not silently recenter them on output maxima. Normalized integrals and mW
must remain separately labeled. Existing physical conversion multiplies the
normalized integral by the positive incident total launch power in mW when
that scale is available. It does not renormalize the attenuated output.

The two-carrier diagnostic splits exact bisector ties equally; specialized
Image Amplification assigns its historical ties to the signal. Preserve both
existing behaviors. N-way ties need a deterministic, documented policy whose
weights partition the plane without double counting. That policy and its
qualification are unresolved implementation decisions, not permission to
change historical measurements. See the
[carrier diagnostic record](../development/pr_carrier_resolved_power_diagnostic.md).

## Entrance reference, ratios, and Results

**P_j(0) is the actual attributed carrier power entering propagation after all
launch screens/elements.** Requested source power before a screen is a
different quantity. Retain/report, where meaningful:

- requested/source beam power;
- post-element channel power/throughput from launch provenance;
- attributed entrance carrier power P_j(0);
- attributed power P_j(z), including output P_j(L).

Post-element lineage power and entrance spectral carrier power also need not
be identical for overlapping coherent fields. Preserve their measurement
definitions rather than relabeling one as the other. Passive-screen attenuation
must not be reported as energy transfer inside the PR medium.

For reliably resolvable carriers with usable entrance power, the primary
normalized curve is the **carrier power ratio**:

```text
R_j(z) = P_j(z) / P_j(0).
```

No fractional-gain definition is introduced. Existing historical fields named
`carrier_gain` retain their established ratio meaning and names.

| Evolution | Desired presentation, subject to availability |
|---|---|
| Static | P_j(z) and R_j(z) versus z; P_j(0), P_j(L), and R_j(L) for each resolvable carrier. |
| TD | At accepted material time t, P_j(z,t) and R_j(z,t)=P_j(z,t)/P_j(0,t); current accepted-state ratio versus z; output ratio R_j(L,t) versus material time; raw entrance/output powers and group totals. |

Each TD profile must belong to the same accepted material state and consistent
optical propagation/replay. Record material time, units/normalization, segment
identity where applicable, and actual z coordinates. Use the entrance for that
state/segment; never normalize a changed launch by an unrelated previous one.

This contract does not authorize unlimited z-by-material-time storage.
Retention, sampling, presentation cadence, resource limits, result policy,
and serialized availability require a bounded implementation design. Do not
claim unavailable interior planes from endpoint-only results or interpolate
missing measurements without explicitly declaring that presentation.

## Conservation, attribution, and unavailable values

Within a coherence group, complete partition weights imply

```text
sum_j P_j(z) ~= integral |A_g(x,y,z)|^2 dx dy = P_group(z).
```

This Parseval/partition accounting applies at each measured plane. In
power-conserving propagation with appropriate boundaries, group power should
also remain constant along z within numerical error. Boundary absorption or
other declared losses must be distinguished from partition error and carrier
redistribution. Report group totals, partition balance, and relevant power
change/loss diagnostics; combine incoherent groups as powers.

**Power conservation does not prove unique carrier attribution.** A complete
Voronoi partition can conserve power while individual regions are unreliable
as beam measurements. Qualification must account for overlapping or coincident
carriers, broad spectra, finite-beam diffraction, screen-induced broadening,
sampling/aliasing, insufficient separation, and negligible entrance power.
Entrance separation alone is not a proof of attribution at every later plane.

Current two-carrier quality is `1-max(wrong_side_fraction)` of the isolated
input spectra and requires at least 0.9 for a reported ratio. Its input-power
floor is `max(1e-12, 64*eps(real_dtype))*P_group(0)`. These are existing rules
to preserve for compatibility, not automatically validated N-channel rules.
The generalization must establish its own explicit applicability and quality
contract while retaining historical output semantics.

Where attribution fails, retain measurable group/total power. Individual
partition integrals may be retained as explicitly qualified bookkeeping, but
must not be presented as reliable beam attribution merely because a cell can
be assigned. Mark individual powers/ratios unavailable or qualified with an
explicit reason. Analysis unavailability must not invalidate propagation.

Zero or negligible post-screen entrance power makes R_j unavailable: do not
divide by zero, report infinity, or substitute an arbitrary ratio. Absolute
output power or power change may remain meaningful if attribution is adequate.
Distinguish an unlit named input from another beam's spectral leakage into its
nominal region.

## Identity, ownership, and compatibility

Future Product-owned stable beam/channel identity must survive rename, reorder,
enable/disable, duplicate display names, persistence/reload, screen assignment,
diagnostic attribution, and later intervention provenance. List index and name
alone are insufficient; disabled channels must not silently transfer identity
or screens to another enabled channel.

Shared Product launch infrastructure should own material-neutral identity and
screen association. PR owns carrier-analysis configuration, applicability,
products, and material-specific provenance. The
[LaunchPlane adapter](../../src/lcprop/adapters/launchplane.py) maps editor
definitions to Product launch data; LaunchPlane must not own PR scientific
identity or diagnostic semantics. This ownership decision does not prescribe a
UUID format, a new dataclass field, or an immediate shared-schema migration.
The exact persistent representation and legacy-ID assignment require separate
authorization and compatibility review; no obvious schema-free addition is
established by the current strict codecs.

Preserve historical Image Amplification requests/results, ratio-valued
`carrier_gain`, delta-power fields, analytic/coupling benchmark metrics, and
saved-experiment interpretation. Old specialized experiments must remain
loadable under their historical contracts. Do not silently convert their
stored quantities or analysis assumptions into new diagnostics. Future general
experiments may use ordinary requests plus optional analysis configuration.
Retain historical direct APIs and compatibility adapters until a separately
reviewed migration establishes equivalent meaning. See
[experiment persistence history](../development/pr_image_amplification_persistence_and_transverse_zoom.md)
and the current codec cited above; the historical schema number is not current.

## Continuation and scientific qualification

Follow the [continuation state/intervention decision](continuation_state_and_interventions_decision.md):
accepted material state must not be permanently bound to its original optical
configuration. Stable identities and source/screen provenance should support
future segment-boundary records. Current validators and checkpoint behavior
remain unchanged. Channel-count, coherence, wavelength, and related changes
remain scientifically unclassified where that decision says so.

Fully nonlinear PR uses complete local transport intensity and is not implicated
by the uniform-reference linearization concern. Current production linearized
PR is a documented tangent about an explicitly declared spatially uniform
reference intensity. Architectural support for finite beams and screens does
not establish that their illumination lies in that approximation's validity
regime. This records the qualification only; it does not reopen the physics
investigation or change normalization. See [PR model contracts](../science/pr_model_contracts.md).

## Deferred decisions and non-authorization

Separate authorized milestones must resolve stable-ID representation/migration,
N-carrier tie and resolvability rules, qualification along z and material time,
bounded history/transport representation, and optional-analysis persistence and
GUI exposure. Source capability alone supplies no blanket scientific validation.

This document authorizes no diagnostic code, GUI/preset changes, Image
Amplification migration, stable-ID schema changes, continuation changes,
P2A-2 full-transverse live Results, solver/integrator/default changes, new gain
definitions, or LaunchPlane changes. It authorizes no scattering/fanning work
or remote continuation.

PR scattering equivalence remains **SCIENTIFIC HOLD / UNESTABLISHED**.
Remote continuation remains **DEFERRED / NOT AUTHORIZED**.
