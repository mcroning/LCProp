# BeamPlane native-acceptance amendment: signed tilt and input-face rays

Status: candidate for review; implementation and native acceptance are separate gates.

This bounded amendment supersedes the unsigned-theta and bounded external-arrow
interaction decisions in the [physical launch contract](beamplane_physical_launch_contract.md).
It does not supersede that contract's interface, footprint, power, capture,
screen, scalar qualification, or deferred focused-launch decisions.

## External intent

External tilt is signed, with `-pi/2 < theta_ext < pi/2`; grazing endpoints remain
unsupported. Phi remains crystal-frame azimuth, and psi remains independent roll.
The entered theta sign is retained in the model, editor, and saved experiment.

```text
kx = k_ext * sin(theta_ext) * cos(phi)
ky = k_ext * sin(theta_ext) * sin(phi)
kz_ext = k_ext * cos(theta_ext)
```

`(-theta, phi)` and `(+theta, phi+pi)` have the same direction. This redundancy
is intentional; the editor must not silently canonicalize signed input. Positive
and negative theta at phi=0 give positive and negative kx respectively. The
existing resolver's reported internal polar angle is a nonnegative magnitude;
internal direction is determined by signed kx/ky and positive kz. Magnitude-only
Snell and positive fringe-period statements in the original contract use
`abs(sin(theta_ext))` when theta is signed. No resolver equation is changed.

## Host-resolved internal ray

The figure is titled **Input face**, uses x-right/y-up coordinates and numeric
axes in micrometres, and labels beams only by number. Rich identity and status
remain outside the plotting area.

For each enabled beam, Product supplies the physical internal ray:

```text
entrance = (x0, y0)
exit = (x0 + L*kx/kz_internal, y0 + L*ky/kz_internal)
```

L is the configured interaction length. The wavevector comes from Product's
material-aware resolver. LaunchPlane renders these coordinates without inferring
refraction or scaling a ray to fit the viewport. The ray indicates the scalar
central-direction prediction; it is not a simulation of nonlinear bending,
walk-off, or broad-spectrum centroid evolution. Existing scalar qualifications
remain applicable. Without host geometry the ray is explicitly unavailable.

External intent and internal ray geometry remain distinct. Changing material
index preserves external theta/phi, tangential k and ideal entrance footprint,
but changes internal angle, kz and exit position. Normal incidence retains a
coincident center and arrowhead with zero shaft length; disabled beams have no ray.
The arrowhead remains interactive at normal incidence.

## Interactive handles and inverse ownership

There is one Input face view. Drag the entrance center to translate the beam
without changing direction intent. Drag the device-sized arrowhead to approximately
set the physical exit/direction; use numeric signed theta/phi for exact values.
The tip stays at the physical exit regardless of view scale. Both handles have
constant device-sized hit targets. All arrowheads stack above all centers,
including different beams; footprints and shafts do not intercept handle clicks.
At zero tilt the arrowhead wins over its coincident center. Drag it away to expose
the center. Objects-list selection remains available but is not required between
canvas manipulations. Selected beams and the focused handle are highlighted.

LaunchPlane sends desired exit coordinates through a host callback
`set_inverse_resolver`. Product owns the scalar inverse:

```text
sx = (exit_x - x0)/L; sy = (exit_y - y0)/L
u_int = (sx, sy, 1)/sqrt(1 + sx*sx + sy*sy)
(kx, ky) = k_internal * u_int.xy
abs(theta_ext) = asin(hypot(kx, ky)/k_external)
```

Unreachable or numerically grazing directions are rejected without changing
intent; no clipping or alternative interface model is used. Product's existing
forward interface, footprint, propagation, power and capture calculations are
unchanged. LaunchPlane has no fallback refraction model. Missing inverse support
makes direction editing explicitly unavailable.

Among the two equivalent signed-theta representations, choose the azimuth
closest modulo 2*pi to the current phi. At an equal-distance tie retain the
current theta sign (positive for +0); at zero displacement retain phi and signed
zero. Crossing zero on a fixed meridian therefore changes theta sign without a
180-degree phi jump. Small successive drags preserve the current branch; a
large jump is resolved by this explicit nearest-azimuth rule.

With canvas focus, arrows nudge the selected handle by one device pixel; Shift
uses ten. The current inverse view transform supplies physical displacement.
Center nudges translate the entrance; arrowhead nudges use the same Product
inverse. Spinbox arrow keys retain their ordinary editor behavior.

Manual reciprocal crossing is supported: drag beam A's exit onto B's entrance,
then B's exit onto A's entrance. This is transient direction editing, not an
automatic or persistent Target mode. Endpoints are recomputed on material or
length changes and are not saved as constraints. Finite-spectrum propagation
centroids remain subject to sampling and scalar qualifications.

## Ordinary interaction controls

Product exposes x/y **full aperture widths** beside the input-face view, linked
to its existing Grid controls. This is one scientific aperture definition,
including request creation and experiment persistence. A view spanning -100 to
+100 µm has full width 200 µm, not 100 µm. Changing the aperture does not move
beams, clip their geometric previews, or renormalize capture.

`Fit` reveals the complete aperture, footprints, entrance centers, and physical
ray endpoints. `Full Aperture` returns to the input-face extent. Neither changes
physical geometry. Number labels are separated in display space, including
coincident beams. Dashed 1/e footprints remain above the aperture and below
rays/center markers.

History qualification: LaunchPlane already had `Fit aperture`; the distinct
`Fit` / `Full Aperture` pair existed in Product image views. It is exposed here
with corresponding geometry/full-extent behavior. Aperture dimensions were
already editable in Grid, but not beside the Beam canvas. This amendment does
not assert that an earlier LaunchPlane aperture dropdown existed. None/Sponge/
Tukey remain optical edge policies, not aperture-size definitions.

## Serialization and coordinated delivery

The pre-release schemas remain LaunchPlane/Product/LC/PR = **4/2/3/7**. Existing
angle fields already store a real number, so this extends the accepted domain
without changing field meaning, structure, units, or old positive-angle data.
Obsolete launch definitions remain rejected. Negative theta round-trips without
canonicalization. Old binaries may reject new negative-angle input; there is no
compatibility conversion into obsolete launch semantics.

Host preview endpoints are derived transient data and are not persisted into
beam intent. LaunchPlane advertises `supports_resolved_internal_rays`; Product
requires this capability and `set_inverse_resolver` at import, so an old contour-only package cannot
silently satisfy the new preview contract. The two candidates require
coordinated integration and installation after review. This task authorizes
neither integration nor installation.

## Validation and exclusions

Independent development regressions compare launch and ordinary PR propagation
centroids with `L*kx/kz_internal, L*ky/kz_internal` for both signs, cardinal and
arbitrary azimuth. Host tests independently check the same physical endpoints,
material-index dependence, aperture/request wiring, signed persistence, and
unchanged footprint oracles. These are local regression checks, not GPU
commissioning or completed human native acceptance.

No interface coefficients, beam-normal screens, focused launch, automatic persistent Target mode,
normalization, propagation kernels, material equations, carrier diagnostics,
continuation, generic reconstruction, power curves, scattering or fanning work
is included. Remote continuation remains deferred and unauthorized.
