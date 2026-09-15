# BeamPlane focus and transverse-boundary completion

## Scope and baseline

This milestone was developed on
`32023c0fe9471d347f7334e9f6ab3d9b0f601be1` on
`feature/pr-second-order-static`. It changes material-neutral launch intent,
launch construction, propagation-boundary primitives, compatibility codecs,
and their focused tests. It does not change LC or PR material equations,
nonlinear-index or electro-optic mappings, coherence grouping, power
normalization, material solvers, or static/time-dependent integrators.

LaunchPane remains independent of LCProp material models. It records beam
intent; each LCProp workflow supplies an `OpticalLaunchContext` containing the
runtime transverse grid, linear propagation index `n_ref`, interaction length,
and propagation sign/convention.

## Focus-defined launch

New launches distinguish four explicit profiles:

- `focused_gaussian`: `waist_x_at_focus_um` and
  `waist_y_at_focus_um` are 1/e field radii at `focus_z_um`, or at half the
  supplied interaction length when midpoint focus is selected;
- `collimated_gaussian`: a finite entrance-plane Gaussian with flat wavefront;
- `uniform`: constant amplitude over the periodic aperture;
- `legacy_gaussian`: the pre-milestone entrance-plane Gaussian semantics.

For one axis with focus radius `w0`, wavelength `lambda`, and propagation
index `n_ref`, the Rayleigh distance is

```text
z_R = pi n_ref w0^2 / lambda.
```

The coordinate origin for `focus_z_um` is the crystal entrance plane. Positive
values are downstream along the selected propagation coordinate and negative
values are upstream. The context's `propagation_sign` maps that coordinate to
the signed z increment used by the propagator; it does not change the
user-facing upstream/downstream meaning. At the entrance, `d = -s z_f`, where
`s` is the propagation sign. The launch radius and curvature are

```text
w(d) = w0 sqrt(1 + (d/z_R)^2)
R(d) = d [1 + (z_R/d)^2].
```

LCProp's forward angular-spectrum hop is
`exp(-i q_perp^2 dz/(2 k_ref))`. Its matching entrance curvature factor is
`exp(+i k_ref x^2/(2 R_x) + i k_ref y^2/(2 R_y))`. Second-moment tests
propagating to positive and negative requested foci verify this sign directly,
including a negative-focus elliptical beam, an elliptical midpoint focus, and
the negative propagation convention.

Uniform illumination is spatially constant before the requested carrier
phase. To remain continuous under periodic propagation, any uniform-profile
phase gradient must be an exact transverse Fourier mode. Collimated and
uniform profiles add no artificial quadratic phase.

All profiles use the established per-channel integral normalization. The
legacy profile and all old payloads retain their old entrance-waist behavior;
new dataclass fields are appended so positional construction remains stable.
LaunchPane schema 1 and 2 payloads migrate to `legacy_gaussian`, while schema
3 retains explicit profile/focus intent. LCProp performs the lightest runtime
compatibility check available by requiring LaunchPane schema 3 or newer.

## Propagation-boundary treatment

This milestone adds a sponge boundary treatment informed by the earlier
successful use of sponge boundaries. It is a newly defined and independently
validated propagation absorber; no prior implementation was found in
reachable repository history.

For aperture half-width `a_x` and fractional sponge width `w`, define

```text
s_x(x) = clip((|x| - a_x(1-w)) / (a_x w), 0, 1),
s_y(y) = clip((|y| - a_y(1-w)) / (a_y w), 0, 1).
```

With profile order `p` and field-amplitude attenuation coefficient
`alpha` in inverse micrometres, the local absorption rate and one-increment
mask are

```text
gamma(x,y) = alpha [s_x(x)^p + s_y(y)^p],
S_Delta_z(x,y) = exp(-gamma(x,y) |Delta_z|).
```

Thus `alpha` is the side-edge field-amplitude e-folding rate; the x and y
rates add in corners. Intensity attenuation is the square of the field mask.
The `TransverseBoundarySpec` defaults used when sponge mode is selected are
`w=0.15`, `p=2`, and `alpha=0.05 um^-1`.

The sponge is applied after every complete symmetric split-step optical
substep, after the second half material screen. For a total optical increment
`Delta_z` divided into `Nsub` pieces, each application uses
`|Delta_z|/Nsub`. Consequently,

```text
product_j S_(Delta_z/Nsub) = exp(-gamma |Delta_z|),
```

so changing optical steps from 10 um to 2 um does not multiply the physical
sponge strength by five. The attenuation-operator identity is exact for every
positive integer subdivision. Complete propagated fields need not be identical
under subdivision because diffraction, material response, and absorption need
not commute. This distinction is explicit in the executable coverage.

The three propagation policies are:

- `periodic`: exact no-op;
- `sponge`: the incremental distance-rate absorber above;
- `tukey`: the established square-root separable Tukey amplitude window
  `sqrt(T_x T_y)`, applied once after a complete prepared-response call rather
  than interpreted as a rate per z.

The bounded wraparound fixture launches an edge-directed packet and verifies
that the sponge reduces wrapped power to less than 10% of the periodic result.

`TransverseBoundarySpec` is appended to the ordinary LC, reduced PR, and
full-transverse PR static/time-dependent request types. The selected policy is
validated and applied by their production optical marches, retained by
experiment, continuation/checkpoint, persistence, and transport codecs, and
selected in the shared LCProp Beam panel. The historical default is an exact
`periodic` no-op; older payloads decode to that same default. The specialized
legacy PR streaming solver keeps its existing solver-level Tukey semantics and
is not silently reinterpreted by this milestone.

The additive boundary field is accompanied by explicit persistence-version
migrations:

- LC experiment request schema 2 accepts schema 1 and supplies `periodic` when
  the older payload has no `optical_boundary`;
- PR experiment request schema 4 accepts the immediately previous schema 3
  (and retains the already-supported schema 1/2 migrations), with a missing
  schema-3 boundary decoded as `periodic`;
- LC static transport codec 2, reduced PR static codec 2, reduced PR TD codec
  2, full-transverse PR static codec 3, and full-transverse PR TD codec 2 each
  accept their immediately previous codec version and migrate a missing
  boundary to `periodic`;
- LC static and TD checkpoint schemas are both version 2, accept schema 1, and
  normalize loaded legacy checkpoints to current schema 2 with a `periodic`
  boundary.

Current experiment schemas require the explicit field. Current experiment,
transport, and checkpoint encoders retain sponge and Tukey selections exactly;
unsupported future experiment, transport-codec, and checkpoint versions remain
rejected. These are schema/compatibility changes only and do not alter launch
or propagation behavior.

## Focused-beam preview and aperture visualization

Focused input-screen previews now receive the same material-neutral
`OpticalLaunchContext` used by production launch realization. LC and PR supply
only the runtime grid, propagation index, interaction length, and propagation
sign; LaunchPane imports no LC or PR material/workflow code.

For a focused beam, LaunchPane's solid ellipse is the derived entrance-plane
1/e field footprint. Its dotted, concentric ellipse is labelled as a focus
waist-size reference and does not represent another field at the entrance.
Tilt and center indicators remain attached to the solid entrance field. The
inspector reports the requested focus radii, signed focus position, derived
entrance radii, and entrance curvatures. It also estimates the largest
peak-normalized Gaussian intensity at the nearest x or y aperture edge as
`exp[-2 min((g_x/w_x)^2,(g_y/w_y)^2)]`. Significant edge intensity is a
prominent wraparound warning for periodic propagation and informational for a
sponge, where absorption may be intentional.

## Cancellation diagnosis

The broad-suite cancellation-stage failure was a fixed-delay race in the test:
depending on suite load, its 0.2 s request could arrive on either side of the
intended optical-stage boundary. Focus/context setup does not participate in
cancellation or accepted-state mutation. The regression now uses a test-only
handshake at entry to the real reduced-PR optical-slice function, then exercises
the unchanged production cancellation path and latency limits. It
deterministically verifies `material_source_optical_z_march`, zero accepted
material steps, the unchanged last accepted state, and the unpropagated-launch
fallback.

## Compatibility and validation

The reviewed LaunchPane source baseline is
`d55528c798f6b520dc95f41ff70b29e494d2e9d1`. The compatible candidate revision
is the seven-file patch over that baseline (`README.md`, four files under
`src/launchplane/`, and two focused test files) whose canonical
`git diff --binary` SHA-256 is
`c5969533dd4904a8f7c95933df053eea3dec1741f1712b927fb9c12e18c847fd`.
That reviewed patch is committed as LaunchPane revision
`e7b61de7e4ddbc8fbe28be1477115d178fa2793d`.

The required commit order is LaunchPane first, followed by verification that
its committed patch has that digest. LCProp is committed second with the
compatible LaunchPane commit SHA recorded above. LCProp also
checks at runtime and in tests that the installed LaunchPane understands schema
3. A heavier packaging/version mechanism is deliberately outside this
developmental integration.

The focused launch context is supplied by LC and PR production entry points,
including reduced/full-transverse static and time-dependent workflows and the
image-amplification/coupling preparation paths. Legacy fields remain
bitwise-identical when a context is supplied. PR aperture diagnostics use
focus-plane waists and focus-relative radii for focused profiles, and treat a
valid periodic uniform profile as continuous rather than as an oversized
clipped Gaussian.

Local validation covered:

- exact focused circular and elliptical second-moment radii;
- positive/negative focus positions, midpoint convenience, and propagation
  sign;
- collimated/uniform phase and periodic continuity;
- normalized power and legacy bitwise equivalence;
- sponge rate/profile, arbitrary-integer subdivision invariance, and bounded
  wraparound reduction;
- one-time Tukey behavior;
- LaunchPane schema migrations, editor behavior, and LCProp adapter round trip;
- LCProp request, experiment, checkpoint/persistence, and transport round trips,
  including old-payload periodic defaults;
- focused input-screen preview and focused entrance-footprint/aperture display;
- deterministic cancellation during the modified reduced-PR optical path;
- affected LC, PR, IA, and aperture-preflight regressions;
- syntax compilation and `git diff --check`.

The final focus/boundary/GUI group passed `121 passed`; persistence, checkpoint,
workflow, and cancellation coverage passed `72 passed, 1 skipped`; transport
and static-workflow coverage passed `121 passed, 1 skipped`; affected non-GUI
workflow and IA coverage passed `100 passed, 3 skipped`; the IA GUI suite
passed `71 passed`; and coupling/carrier diagnostics passed `30 passed`. The
complete LaunchPane suite passed `55 passed`.

After the schema/codec-version remediation, the focused experiment,
checkpoint, PR-persistence, and remote-transport set passed `122 passed`. The
broader affected persistence, transport, checkpoint, GUI-persistence, and
workflow set passed `274 passed, 1 skipped`. Syntax compilation and final
whitespace/`git diff --check` validation passed, and the unchanged LaunchPane
suite again passed `55 passed`.

The complete LCProp audit reached `1537 passed, 74 skipped` with three failures.
An isolated archive of the exact LCProp and LaunchPane baselines had already
reproduced those same three deterministic pre-existing failures: two stale
image-pane field-list expectations and one stale Fast-result longitudinal
availability expectation. None of those files or product paths is changed by
this milestone. The former fourth, timing-sensitive cancellation-stage failure
passed in the complete audit after the deterministic test remediation above.
Syntax compilation and `git diff --check` passed in both repositories.

No CUDA or cluster resource was accessed. No authoritative numerical evidence
package was generated; this is ordinary deterministic local validation.
