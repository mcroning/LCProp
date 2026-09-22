# BeamPlane physical launch: external direction, beam-normal profiles and power

Date: 2026-09-22. Status: **Future design candidate; implementation NOT
AUTHORIZED.** This document makes design decisions and analytic derivations;
it does not claim implemented or numerically validated high-angle physics.

Source baselines: LCProp `5131648f7a7560b0a4f635d2c6a6bb1e70ffeacf` on
`feature/pr-second-order-static`; LaunchPlane
`add5f6bdbfcceaea7a8f2fd132e4ebf70b6eb5a8`; read-only Research context
`b947fac51722eafa971e756a2cc2053be065294d`.

## Ownership and precedence

**LCProp owns this single canonical integration/scientific contract.** It spans
material-index selection, sampling, power normalization and LC/PR consumers;
the independent LaunchPlane widget cannot own those material decisions.
"BeamPlane" here denotes the intended physical-beam editor contract, not a
package rename or a second implementation. Current package naming remains
LaunchPlane. Its future public intent/API documentation should reference this
decision and describe its own interface, without duplicating the scientific
contract. No LaunchPlane file or cross-reference is changed in this task.

This is an intentional future breaking migration, not an additional legacy/new
waist mode or paraxial/scalar selector. Upon a separately authorized migration,
the semantics here replace development-era laboratory-waist and tilt-input
semantics. Until then, current source remains authoritative. The existing
[general PR/carrier contract](pr_general_beams_and_carrier_power_contract.md)
still governs ordinary propagation versus optional analysis, coherence groups,
post-element entrance references and attribution qualification. Its historical
unweighted-integral-to-mW formulas must not be applied to the new field
normalization without the explicit power migration below. No carrier curves
are implemented by this decision.

The [continuation decision](continuation_state_and_interventions_decision.md)
remains future guidance; no current compatibility check is loosened. Follow
the [architecture standard](../codex/01_Architecture/architecture_review.md)
and [Development/evidence gates](../codex/README.md) in later milestones.

## Current source facts and the archival lesson

- LaunchPlane's public `README.md`, `src/launchplane/model.py` and
  `src/launchplane/serialization.py` at the baseline above already support
  external **projected-plane** angles: `tan(angle_x)=s_x/s_z` and
  `tan(angle_y)=s_y/s_z`. They are not polar theta and azimuth phi. The model
  derives conserved transverse wavevectors using an explicit launch-medium
  index. Schema 3 persists compatibility wavevector fields, profile/focus and
  launch-medium provenance; versions 1/2 are currently migrated.
- Current LaunchPlane `canvas.py` places `(x,y)` at scene `(y,-x)` and draws
  tilt handles with that convention. Physical x-horizontal/y-up arrows require
  migrating canvas positions, labels, dragging and hit testing together.
- The [adapter](../../src/lcprop/adapters/launchplane.py) copies wavevectors,
  waists, centers and focus fields to [BeamChannel](../../src/lcprop/core/beams.py).
  Its reverse path reconstructs wavevector-mode editor definitions; it does
  not recover original external-angle intent.
- [Shared launch construction](../../src/lcprop/optics/launch.py) evaluates
  legacy/collimated Gaussians directly on laboratory x/y. Focused launches use
  one propagation-coordinate distance for the whole entrance plane. A carrier
  `exp(i*(kx*x+ky*y+phase))` is applied separately. Sampled per-channel
  integrals are normalized to incident power fractions before screens.
  `reconstructed_physical_powers_mW` multiplies those unweighted integrals by
  incident total power. Neither an obliquity flux factor nor Fresnel interface
  transmission is currently applied.
- The committed [laboratory screen correction](../development/image_input_display_coordinate_parity.md)
  maps top-down raster `[row,column]` to increasing physical `[x,y]`; Qt row
  reversal is display-only. [Screens](../../src/lcprop/optics/screens.py) are
  specified on the laboratory entrance face and applied without post-screen
  renormalization. This contract is preserved.
- The preceding source-only archival audit recovered PRProp3D, not an earlier
  original PRProp. Its `genrot`/`gaus` rotate an internal beam-frame Gaussian,
  aim through the crystal center, and include point-dependent width, curvature
  and Gouy phase. Its raster remains a laboratory-grid patch. Its amplitude
  ratio is not a general physical power normalization. The recovered file was
  61,640 bytes, SHA-256
  `0adb5c824459ebaa31adbf9cad541c9ef634ac63f2b83bf42e8c3ee5a62ed28f`,
  also present at historical Product commit
  `7578997903fd02e70c6053e1a117594b5ca3b6b0:reference/prprop/prprop3d.py`.
  These conclusions are preserved here without a runtime/private-Research
  dependency or adoption of the archive's angle signs, centering or images.

## Canonical physical intent

The first consumer-ready contract carries vacuum wavelength in micrometres,
requested incident power in mW, external medium index, external polar angle and
azimuth, external beam-normal collimated Gaussian radii, beam roll, entrance
intersection `(x0,y0)`, phase, coherence group and existing laboratory-plane
launch elements. Lengths are physical micrometres, internal angular values
are radians; the GUI may display clearly labelled degrees.

**Radii refer to the external incident beam**, at a beam-normal reference plane
through its axis/entrance intersection immediately before the interface. They
are 1/e field (1/e-squared intensity) radii. For the collimated launch
prescription the envelope is constant along that incident axis. "Collimated"
describes this entrance prescription, not a finite Gaussian that never
diffracts. No focused radius or source-plane distance is inferred.

Use explicit finite positive `n_ext`, default 1, in the first intent schema:
current LaunchPlane already supports launch-medium provenance, so dropping it
would introduce ambiguity. It means the medium immediately adjacent to the
entrance, not an upstream lens or an arbitrary earlier glass layer. A single
homogeneous external region must have a consistent index for the same
wavelength. Layer stacks and dispersion models are not part of this milestone.

Include roll initially because unequal radii are already useful. A circular
beam is roll-independent. Do not add polarization support through a roll field.
The ordinary GUI should not require phase gradients; derived k values may be
shown read-only. Analytic benchmark plane waves remain a separately labelled
test/periodic-cell prescription, not a finite physical beam with an unexplained
"total power" over an infinite cross-section.

## Direction, refraction and admissibility

Use a right-handed laboratory frame: x to the right, y upward, material
entrance at z=0, forward into the material along +z. Define

```text
0 <= theta_ext < pi/2
phi modulo 2*pi, measured from +x toward +y
s_ext = (sin(theta_ext)*cos(phi), sin(theta_ext)*sin(phi), cos(theta_ext))
k_ext = 2*pi*n_ext/lambda0
kx = k_ext*sin(theta_ext)*cos(phi)
ky = k_ext*sin(theta_ext)*sin(phi)
```

Thus phi=0 points toward +x, phi=pi/2 toward +y, and opposite transverse
directions use phi+pi. At theta=0, phi has no directional meaning; display a
normal-incidence symbol, not a meaningful arrow. Beam roll still orients an
elliptical cross-section at normal incidence.

For a forward scalar mode at a planar isotropic interface:

```text
k_int = 2*pi*n_int/lambda0
kz_int = sqrt(k_int**2 - kx**2 - ky**2) > 0
s_int = (kx,ky,kz_int)/k_int
n_ext*sin(theta_ext) = n_int*sin(theta_int)
```

Tangential k is conserved; phi is the same on both sides. Reject a carrier
with `kx**2+ky**2 >= k_int**2`: negative radicand means no propagating
transmitted carrier (total internal reflection where applicable); equality is
grazing, with no positive face-normal flux for the proposed finite-power beam.
Do not clip theta, replace a square root by an absolute value, or invent an
evanescent launch. This launch restriction is stricter than the unchanged
scalar kernel, which retains grazing modes. Validate finite inputs, positive
wavelength/index, grid Nyquist support, aperture and resolved spectral width.
An admissible carrier alone does not qualify its entire spectrum.

For PR, `n_int` is its declared scalar optical reference index, not an
electrical transport coefficient. PR's electrical-profile isotropy does not
establish optical crystal isotropy. LC currently supplies `no` as reference
index and uses extraordinary effective-index phase response. LC director
anisotropy, polarization and Poynting walk-off do not follow an isotropic
Snell/ray formula in general. Use the shared scalar-reference construction
with explicit qualification in the existing small-angle LC regime; do not
claim a high-angle extraordinary interface solution. An anisotropic optical
interface needs a separately reviewed dispersion/energy-flow model.

## Gratings and trajectories

For coherent beams at the same vacuum wavelength through the same planar
isotropic interface, define `K_perp = k2_perp - k1_perp`. The cosine intensity
fringes have transverse period `2*pi/|K_perp|` along K_perp. For equal polar
angles theta_ext and opposing azimuths in one external medium:

```text
Lambda_perp = lambda0 / (2*n_ext*sin(theta_ext)).
```

There is no additional division/multiplication by n_int: writing the same
expression using `n_int*sin(theta_int)` gives the same answer. At zero
separation there is no finite transverse fringe period. Unequal angles can
also produce a longitudinal grating component; the statement is about K_perp.
It is not a blanket reflection-grating, unequal-wavelength, anisotropic or
nonplanar-interface result.

For the [current scalar PR kernel](../../src/lcprop/optics/splitstep.py),
`H(d)=exp(i*kz*d)` on retained propagating support. Narrow-packet trajectories
obey `dx/dz=kx/kz_int`, `dy/dz=ky/kz_int`, consistent with
[PR geometry](../../src/lcprop/pr/geometry.py), not kx/k_int and ky/k_int.
Index strongly changes these slopes, crossing distance and overlap even when
the external-geometry transverse fringe period is fixed. This design changes
neither dispersion nor response/substep ordering.

## Transverse basis, roll and projected footprint

For either medium's angle theta, let `u=(cos(phi),sin(phi),0)`,
`v=(-sin(phi),cos(phi),0)`. Let R be the proper Rodrigues rotation about v
through +theta, mapping +z to s. Set `e1=R ex`, `e2=R ey`, then

```text
e1_psi = cos(psi)*e1 + sin(psi)*e2
e2_psi = -sin(psi)*e1 + cos(psi)*e2
```

Positive roll turns e1 toward e2 about +s; it never changes s or k. At theta=0
R is the identity independent of phi, so roll is measured from laboratory +x
without an azimuth singularity. This avoids using phi as a hidden beam roll.

For r=(x-x0,y-y0,0), beam-local coordinates are
`xi=e1_psi dot r`, `eta=e2_psi dot r`, `zeta=s dot r`. A collimated Gaussian
has envelope `g=exp(-xi**2/w1**2-eta**2/w2**2)`. The phase at the axis/entrance
intersection is explicitly alpha; the entrance carrier is
`exp(i*(kx*(x-x0)+ky*(y-y0)+alpha))`. Do not also add a second copy of k*zeta.
Changing phase reference from the old global-origin convention must be covered
by preset/coherence migration tests.

Let M be the two-by-two matrix of the x/y components of e1_psi and e2_psi.
The entrance quadratic form is

```text
Q_lab = M.T @ diag(1/w1**2, 1/w2**2) @ M
g = exp(-r_xy.T @ Q_lab @ r_xy).
```

Its eigenvectors give footprint axes and inverse square-root eigenvalues give
1/e field radii. For a circular profile in that medium, the laboratory radii
are `w/cos(theta)` along u and w perpendicular to u. For general ellipses/roll,
do not merely stretch the laboratory x radius. Use the full quadratic form.

### External waist versus internal waist: an essential distinction

The requested external incident beam and an arbitrarily specified internal
circular beam are not the same boundary condition. A planar interface does
not relocate the transverse envelope at z=0. In the first narrow-angular-band,
ideal-transmission approximation, the external envelope determines that face
footprint. For a circular external beam of radius w_ext:

```text
w_parallel,face = w_ext / cos(theta_ext)
w_perpendicular,face = w_ext
w_parallel,int = w_ext*cos(theta_int)/cos(theta_ext)
w_perpendicular,int = w_ext.
```

The transmitted beam-normal section is generally elliptical, even though the
incident section is circular. Conversely, **if w denotes an internal circular
beam-normal radius**, the requested familiar formula is exactly
`w_parallel,face=w/cos(theta_int)`, `w_perpendicular,face=w`. It must not be
applied to w_ext without conversion. Doing so would silently redefine the
user's physical incident beam or assume an undeclared upstream reshaping optic.

Decision: external beam-normal radii are canonical; internal radii and the face
footprint are derived, clearly labelled quantities. There is no external/internal
waist toggle. For general profiles, derive the internal transverse quadratic
form from the shared face form: `Q_int = inv(M_int).T @ Q_lab @ inv(M_int)`.
This is an explicit refinement of the shorthand internal-circle example,
necessary to satisfy the physical incident-beam contract. Equality holds without
that distinction at normal incidence or equal indices. No code changes here.

## Physical power and scalar field normalization

**Requested P_j is the power in incident beam j through its external
beam-normal plane, after any separately accounted upstream preparation and
before the material interface and the Product's laboratory-face screens.**
It is not the numerical integral over whatever aperture happens to be chosen.
For the first contract upstream optics are represented by the resulting incident
beam, not an additional hidden loss. If future explicit upstream elements are
introduced, distinguish source-before-preparation power from this requested
incident P_j to prevent double counting. Coherent source powers are not silently
renormalized after superposition.

Use B as a scalar **irradiance amplitude**, so `|B|^2` has units mW/um^2
per area perpendicular to a narrow beam's energy direction. It is not an
electric field in V/m. For a collimated narrow beam with C=cos(theta)>0:

```text
dS_perp = C * dx*dy
J_z = C * |B|**2                         # normal flux density
P = integral |B|**2 dS_perp
  = integral J_z dx*dy
  = C * integral |B|**2 dx*dy.
```

Consequently the unweighted laboratory integral is P/C, not P. The circle's
larger footprint and smaller normal flux component cancel. **Geometric
projection is not optical gain.** This ray-direction relation is exact for the
adopted collimated geometry, approximate for a diffracting/broad-spectrum field.

For an external Gaussian on its full transverse plane:

```text
N_perp = integral |g|**2 dxi*deta = pi*w1*w2/2
B_ext(x,y,0) = sqrt(P_j/N_perp) * g_ext(x,y) * exp(i*carrier_phase).
```

This defines P_j independently of the simulation grid. In the initial ideal
interface model with unit power transmission and no interface phase shift:

```text
B_int(x,y,0) = sqrt(C_ext/C_int) * B_ext(x,y,0)
C_int*|B_int|**2 = C_ext*|B_ext|**2.
```

The footprint is continuous and total incident/transmitted face flux agrees.
The internal beam-normal area changes by C_int/C_ext, consistent with the
derived internal ellipse. If B were instead an actual scalar electric-field
amplitude in a nonmagnetic medium, irradiance would carry an index/impedance
factor; ideal power transmission would require
`E_int/E_ext=sqrt(n_ext*C_ext/(n_int*C_int))`. Neither expression is a Fresnel
coefficient. The chosen B convention has already absorbed the irradiance
conversion, so applying that index factor again would be an error.

To retain manageable normalized scientific arrays, let `P_scale=sum_j P_j>0`
and store `A_j=B_int,j/sqrt(P_scale)` in 1/um. Then, for adequately captured
unscreened beams under the narrow-band approximation,

```text
C_int,j * integral |A_j|**2 dx*dy = P_j/P_scale
sum_j C_int,j * integral |A_j|**2 dx*dy = 1.
```

The current `sum_channel_integrals_equals_one` invariant is deliberately
replaced. Do not multiply A by sqrt(C_int) merely to preserve that old invariant:
that would make A a normal-flux amplitude while material code still interprets
its square as irradiance. Per-channel zero power produces zero field; retain
the positive-total-power requirement until dark-relaxation normalization is
separately resolved.

**Finite aperture:** use the analytically power-normalized physical beam, then
sample it. Do not boost a truncated beam until the captured grid integral equals
the requested power. Record requested power, captured pre-screen face flux,
quadrature/tail error and post-screen flux separately. An implementation must
set and validate explicit sampling/tail tolerances; inadequate coverage must
request a larger/finer grid or be explicitly treated as physical aperture loss,
not hidden normalization. Broad/unresolved spectra cannot be certified by a
good scalar integral alone.

For a laboratory intensity mask T(x,y), apply `A_post=sqrt(T)*A_pre` without
renormalization. Under the declared narrow-direction approximation:

```text
P_post,j = P_scale*C_int,j*sum_xy(T*|A_pre,j|**2)*dx*dy
P_post,j - P_pre,j = P_scale*C_int,j*sum_xy((T-1)*|A_pre,j|**2)*dx*dy.
```

Report both requested-to-post and captured-pre-to-post ratios, with explicit
denominators. For resolved capture they coincide; otherwise they distinguish
aperture loss from screen loss. Never silently rename one as the other.

### Limits of a scalar physical-flux claim

Large **central tilt** alone is not large numerical aperture; a narrow spectrum
can use the cosine formula at substantial tilt. Conversely a tight waist or a
sharp image can require many directions even at normal incidence. Qualification
must consider relative variation of `kz/k` over the retained spectrum, support
near grazing/cutoff, and coherent overlap, not theta alone. Future implementation
must declare numerical acceptance tolerances before claiming physical power.

For a forward homogeneous isotropic scalar Helmholtz field with this irradiance
amplitude convention, the integrated scalar axial current is proportional to
the angular-spectrum weighted norm. With Product's unnormalized FFT:

```text
P_z,scalar = P_scale * dx*dy/(Nx*Ny)
             * sum_propagating_k ((kz(k)/k_int)*|fft2(A_g)(k)|**2).
```

Form A_g coherently within each group; add powers across mutually incoherent
groups. This is the appropriate scalar reference for qualifying the constant
carrier-cosine approximation. It is not a universal vector Poynting formula:
polarization, anisotropy, interference-local flux and interface angular response
need more information. A lossless ideal interface over a broad spectrum would
need angle-dependent amplitude matching, not just the central cosine ratio.
Those broad-spectrum/interface capabilities are not claimed by the first launch.

Coherent local material intensity remains the declared scalar
`P_scale*sum_g |sum_j A_j|**2`, not J_z. For overlapping different directions,
one cannot get total coherent flux by summing separately measured lineage
powers or multiplying the group's intensity by an arbitrary single cosine.
Record source powers separately from group interference and flux diagnostics.

The present scalar split-step diffraction preserves unweighted L2 norms on
retained support; a homogeneous hop also preserves the weighted norm because
its phase is diagonal in k. A spatial material phase screen can redistribute
angles, so preservation of unweighted L2 across the full solver does **not**
prove conservation of the weighted physical-flux proxy. This task does not
change the material solver to repair that approximation. High-angle output
power must be qualified; do not advertise legacy L2 conservation as exact
physical energy conservation. Preserve useful norm diagnostics with truthful
labels. A later optical-model qualification is required wherever the difference
is material.

First implementation: implement the explicit narrow-band irradiance/flux
convention and its applicability diagnostics; reject unsupported launch claims
or mark physical power unavailable outside qualification. Existing laboratory
screens still retain their coordinate and multiplication semantics, including
sharp masks, but their resulting broad spectra do not inherit a narrow-band
power qualification automatically. Do not implement a new full-vector solver
or general carrier-power history as part of launch migration.

## Interface transmission is a separate model

Initial contract: planar scalar refraction, ideal unit **power** transmission,
no reflection, no polarization-dependent Fresnel coefficients and no added
interface phase. Display this assumption. Refraction, projected area, irradiance
normalization and Fresnel transmission are four separate concepts. A future
interface model would multiply transmitted flux by an explicit T_s/T_p and
track reflection, polarization and phase; it must not be smuggled into a radius
projection or an unexplained amplitude normalization.

## Center, phase and arrow presentation

Canonical center intent is the beam-axis intersection `(x0,y0)` with z=0.
No crystal-center targeting or focus-induced center shift is implicit. A later
common-target convenience may explicitly derive

```text
x0 = x_target - z_target*kx/kz_int
y0 = y_target - z_target*ky/kz_int.
```

It must show the resulting entrance centers and require deliberate application;
changing an index must not silently move a manually specified center. An
external source-plane center, an internal target and a focus center are
different concepts. Initial implementation needs only explicit entrance centers.

Draw arrows at those centers in laboratory x-horizontal/y-up coordinates.
Use `L=L_max*sin(theta_ext)` in bounded display units and an explicit theta_ext
annotation. The arrow points `(cos(phi),sin(phi))`; Qt's scene adapter maps
that direction to `(cos(phi),-sin(phi))`, without changing physical data.
Normal incidence uses a dot/normal symbol. Large angles saturate naturally
below L_max; zoom must not turn arrows into physical trajectory lengths.
Retain readable per-beam colors/names and distinguish disabled selections.

One solid external-direction arrow is the default. Show derived theta_int and
slopes in the inspector. If a contextual internal-trajectory overlay is later
enabled, draw it dashed over a stated physical z span with a separate legend;
at an isotropic planar interface its azimuth is identical, not a second
rotated direction. Footprint ellipses must use the same Product-supplied
quadratic form as launch construction, not duplicate material-aware geometry
inside the widget.

## Focus and screen dispositions

Defer focused beam-frame support to a separate qualified milestone. Evaluating
a focused Gaussian on an oblique face gives a point-dependent beam-local zeta,
and therefore point-dependent radius, curvature and Gouy phase. The archive
illustrates a rotated paraxial beam-frame model, not an exact Maxwell solution.
An external focus refracting into an internal focus also raises astigmatism
and focus-location questions. These must not be hidden behind the collimated
formula or a constant face-wide propagation distance. Old focused requests
must be explicitly unsupported after the breaking migration until replaced;
do not silently flatten them or retain a legacy-mode selector.

Keep all current screens laboratory-plane masks on the material entrance face,
in the existing order after ideal interface launch realization. Preserve the
committed raster orientation, placement, sampling and passive-amplitude rules.
Roll applies to the beam profile, not to a laboratory mask. Beam-normal images
are deferred: they require their own plane location, basis, projection,
resampling, phase/support and flux contract. Archival lab-grid raster pasting
is not a precedent for that capability.

## Integration, LC/PR implications and persistence

| Owner | Future responsibility |
| --- | --- |
| BeamPlane/LaunchPlane | Physical external intent, units, roll, center/phase, coherence, bounded arrows; consume resolved geometry for previews |
| LCProp adapter and shared launch | Preserve intent in headless requests; select host-provided scalar reference index; derive k and footprint, sample and normalize field, apply laboratory elements, emit power/qualification provenance |
| LC and PR workflows | Consume one resolved launch; keep their material equations, cadence and accepted-state rules; do not derive a second geometry |

Keep physical intent and resolved runtime launch quantities distinct. The
adapter cannot discard n_ext/theta/phi and later fabricate them from k without
provenance. GUI previews, headless requests and workers must invoke the same
resolver. Array storage remains `(channel,x,y)` with ascending x/y, and dtype/
backend behavior must remain explicit.

LC's historical small soliton launch angles made projection errors negligible;
that was an intended scope/accuracy approximation, not a different physical
meaning of a beam. Shared construction benefits LC without a PR-only fork.
However, [LC's extraordinary scalar contract](../science/lc_model_contracts.md)
and [power coupling](../../src/lcprop/lc/coupling.py) require care:
`bi` uses total physical P and multiplies normalized irradiance. Keeping
`A=B/sqrt(P_scale)` permits that coefficient formula to remain unchanged;
using a normal-flux amplitude would underdrive the material at oblique angle.
This does not qualify the existing torque/polarization model at large angle.

[LC stationary solitons](../../src/lcprop/lc/workflows/soliton.py) repeatedly
normalize to unit unweighted field integral. That is an eigenproblem constraint,
not merely launch setup. Do not automatically change it or claim tilted
stationary solitons are fixed by a shared launch edit. Retain qualified
normal-incidence stationary behavior and explicitly gate new tilted stationary
interpretations pending a separate scientific decision. No director, soliton
or existence-curve physics change is authorized.

PR is the immediate geometry driver. Respect the
[current PR material contracts](../science/pr_model_contracts.md), shared
wavelength requirements, group interference and the peak-reference definition.
Changed irradiance distribution can change a material solution physically;
that is not permission to change transport equations or gain definitions.

Migration touchpoints include:

- LaunchPlane `model.py`, `canvas.py`, `launchpane.py`, `serialization.py`,
  its README and editor/serialization tests. Introduce one new physical-intent
  schema; obsolete developmental schemas should be rejected clearly instead
  of permanently keeping dual waist/angle semantics.
- [BeamChannel](../../src/lcprop/core/beams.py),
  [adapter](../../src/lcprop/adapters/launchplane.py),
  [shared launch](../../src/lcprop/optics/launch.py), shared beam-panel previews
  and host optical contexts. Derived k must agree across GUI and headless paths.
- [Shared experiment beam encoding](../../src/lcprop/persistence/experiments.py),
  [LC experiment schema 2](../../src/lcprop/lc/experiment_codec.py),
  [PR experiment schema 6](../../src/lcprop/pr/experiment_codec.py), and the
  material transport/request codecs. Request fingerprints and runtime launch
  provenance must distinguish the new normalization and reference planes.
  Exact replacement schema numbers are implementation decisions; no schema
  changes are made now.
- Existing `normalized_power`, physical-power reconstruction, results labels
  and [carrier-power analysis](../../src/lcprop/pr/carrier_power.py) must stop
  treating the new unweighted norm as mW. This is a required integration audit,
  not authorization for new carrier curves. Record the physical-power model,
  requested/captured/post-element powers, resolved indices/k, coordinate basis,
  aperture/support qualifications and code/schema identity.
- [Launch tests](../../tests/test_launch.py),
  [adapter tests](../../tests/test_launchplane_adapter.py),
  [beam-panel tests](../../tests/test_beam_panel_launchplane.py),
  [experiment persistence](../../tests/test_experiment_persistence.py),
  [coordinate parity](../../tests/test_image_input_display_parity.py),
  [PR coupling tests](../../tests/test_pr_two_beam_coupling.py), LC static/TD
  launch fixtures and [soliton](../../tests/test_soliton_trans.py)/
  [existence](../../tests/test_soliton_existence.py) assumptions need deliberate
  migration. GUI presets and examples described in the
  [user guide](../user/user_guide.md) must declare external angles and waist
  plane; do not merely bless changed numerical outputs.

Reject obsolete development experiment inputs with a clear recreate/re-export
instruction. A targeted offline converter is warranted only with real data
needs and enough provenance; unknown external medium, waist plane or phase
reference cannot be guessed. Preserve archived source/results as history.
Do not reinterpret saved checkpoint A0 or resume it under the new contract.
Gate incompatible checkpoint/request combinations using the existing
continuation boundary, without implementing interventions or remote continuation.
The intentionally unstaged reconstruction-contract candidate is excluded and
unmodified; no inferred migration is applied to it.

## Reconstruction distinction

Generic backward scalar propagation returns a laboratory z=0 complex field
on retained support. That differs from the beam-normal external reference
field, the post-screen entrance field, and the raw raster. A future beam-frame
view needs propagation to a tilted plane and appropriate spectral resampling,
basis/flux handling and support qualification; it is not a 2-D image stretch.
Neither inverse propagation nor a coordinate rotation undoes absorption,
nonlinear material interaction or cutoff losses. No reconstruction contract
or implementation is amended here.

## Recommended bounded implementation sequence

1. **Physical intent and UI contract:** new external theta/phi/n_ext, radii,
   roll and center/phase semantics, x-horizontal/y-up arrows and explicit schema
   rejection. Develop with a resolver interface; do not connect a new UI to
   old launch semantics and call the migration complete.
2. **Shared collimated resolver and power qualification:** external/internal
   basis, refraction, footprint, analytic incident normalization, captured and
   post-screen flux, angular-support diagnostics. Resolve numeric applicability
   tolerances before claiming power accuracy. No focus or beam-normal screen.
3. **Coordinated Product integration:** adapters, PR and LC propagation,
   preview/results power interpretation, presets and persistence/transport
   boundaries. Complete all three stages before enabling the new contract by
   default. Gate unqualified LC stationary/anisotropic and broad-spectrum claims;
   do not silently alter material/eigenproblem physics.
4. **Focused beams:** separately review external/internal focus planes,
   astigmatism, point-dependent longitudinal coordinate and approximation limits.
5. **Beam-normal screens:** separately authorize plane geometry, resampling,
   phase/support and power semantics.
6. **Beam-frame reconstruction views:** only after their separate scientific
   contract and generic reconstruction authorization exist.

Future validation must independently derive zero-angle/equal-index limits,
Snell signs/quadrants, grating period independent of n_int, scalar trajectories,
external-to-internal anamorphism, arbitrary roll/elliptical quadratic forms,
phase reference and stable normal-incidence arrows. Check one-mW flux against
both analytic beam-normal integrals and sampled face flux; distinguish aperture
capture from screen transmission; verify cosine and spectral-current agreement
within a declared narrow-band regime, and explicit failure of that claim
outside it. Preserve asymmetric raster parity, no-screen behavior, precision,
coherence and all normal-incidence LC/PR regression contracts. These are future
tests, not results claimed by this document.

## Unresolved qualifications and non-authorization

Focused/refraction astigmatism, beam-normal screens, vector/interface Fresnel
models, anisotropic energy direction, broad-spectrum physical flux and
high-angle stationary LC modes require later qualification. They are explicitly
separate from the accepted physical-intent and narrow-band collimated design.
No source, schema, LaunchPlane, material equation, scalar propagator, checkpoint,
carrier diagnostic or continuation implementation is changed by this candidate.
No numerical tests, simulations, remote queries or GPU work were performed.

Remote continuation remains **DEFERRED / NOT AUTHORIZED**.
PR scattering equivalence remains **SCIENTIFIC HOLD / UNESTABLISHED**.
No P2A-2, scattering/fanning or follow-on implementation is authorized.
