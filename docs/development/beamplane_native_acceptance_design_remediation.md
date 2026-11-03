# BeamPlane native-acceptance design remediation

Status: Development + Local Validation candidate. Human native acceptance is
not complete. Nothing is staged, committed, integrated or installed by this task.

## Preflight and scope

- LCProp baseline: `dc3cc668d756762917e8070e421a36a3bb3143b5`.
- LaunchPlane baseline: `7b10bc1963c5b83dcf2eab866216f02746e1ea40`.
- Both isolated branches: `feature/native-design-remediation`.
- Scope: signed external intent; host-resolved internal rays; input-face controls
  and presentation; persistence checks; small architecture amendment.
- Existing forward scientific equations and algorithms are unchanged. The accepted
  angle domain extends to negative theta; a Product-owned inverse geometry helper
  now maps requested physical exits into that same interface contract.
- Validation: Python 3.12, NumPy local regressions and Qt offscreen; focused,
  affected and complete suites; syntax/docs/links/whitespace checks. No GPU or
  cluster access is authorized or used.
- Fresh protection inventory includes authoritative Product/LaunchPlane/Research,
  reconstruction candidate, historical files, and installed new-user environment.
  Historical Product inventory is 4,345 paths, not an older inventory count.

The [architecture amendment](../architecture/beamplane_native_acceptance_amendment.md)
records the revised interaction contract. The scientific sections of the
original physical-launch contract remain unchanged.

## Findings and bounded corrections

### Direction and signed intent

No PR launch-sign defect was found in independent signed-direction development
checks. The existing interface phase is `exp(i*(kx*x + ky*y + phase))` and the
ordinary PR scalar propagator predicts positive displacement `L*kx/kz`,
`L*ky/kz`. The old canvas drew a bounded `80*sin(theta)` device-pixel glyph:
it described external direction, not travel through the material. This was an
interaction-contract mismatch, not evidence of reversed propagation.

LaunchPlane and Product now accept `-pi/2 < theta_ext < pi/2`; grazing rejection
is preserved. Entered signed theta and phi are retained. Roll is independent.
No eigenmode, material, propagation, power or footprint formula is changed.

### Host geometry and presentation

Product sends exit endpoints from its existing resolved kx, ky, positive internal
kz and actual interaction length. Disabled beams retain stack index positions
but have no ray. Both material-index and length controls refresh host geometry
immediately, before Run; no fabricated one-micrometre/material-index fallback is
shown as a resolved ray. Errors invalidate the complete preview.

LaunchPlane draws physical ray segments in scene coordinates. Only the
arrowhead's styling is fixed in device pixels, so it remains legible at long-ray
Fit scales; the endpoint never moves or shortens. The earlier center-only candidate removed the old external direction-drag handle.
The authorized interactive remediation below supersedes that decision with a
physical-exit handle and explicit Product-owned inverse. Missing host geometry
remains explicit.

The caption is `Input face`; verbose scene labels become beam numbers. Number
labels are separated in device space, including coincident multibeam centers.
Normal symbols, x-right/y-up parity, dashed footprints and stacking order
(aperture < footprints < rays/centers) remain. Rich status stays outside the plot.

### Aperture and view-control history

Inspection of pre-Milestone-1 Product and LaunchPlane history showed:

- Grid already owned editable x/y aperture dimensions. The Beam canvas had no
  size controls; it was not a scientifically fixed-size aperture. Human follow-up
  clarified that missing controls on that view were the usability problem.
- The screenshot's -100 to +100 µm extent is a 200 µm full width. New controls
  explicitly say **x/y full width** to avoid a half-width interpretation.
- The new Beam controls directly mirror canonical Grid editors, including their
  ranges and precision. Requests and saved experiments consume the same Grid;
  no second aperture definition or capture renormalization is introduced.
- None/Sponge/Tukey were not removed: they remain optical edge-treatment policies.
  No fictitious historical aperture dropdown is claimed or recreated.
- LaunchPlane `add5f6b...` already had `Fit aperture`; it did not have a separate
  historical companion. Product's `ImagePane` had the `Fit` / `Full Aperture`
  pair (present before this redesign). This candidate exposes that distinction
  on Input face: Fit reveals complete physical geometry, Full Aperture returns
  to the aperture extent. Neither changes beam intent or shortens rays.

### Persistence and package boundary

Schemas remain LaunchPlane/Product/LC/PR = 4/2/3/7. The real-valued theta field
already preserves sign, so widening validation requires no new wire structure
or reinterpretation. Negative values round-trip through LaunchPlane and a real
PR request codec. Obsolete semantics remain rejected. Old binaries can reject
new negative intent; this task does not build a legacy conversion layer.

Product requires LaunchPlane's explicit `supports_resolved_internal_rays`
capability and `set_inverse_resolver` API. The candidates must be reviewed and eventually delivered together;
the installed environment is untouched.

## Earlier candidate validation evidence

The development tests are regression evidence, not commissioned scientific runs.
They use the two isolated source trees ahead of installed modules on PYTHONPATH,
`PYTHONDONTWRITEBYTECODE=1`, `QT_QPA_PLATFORM=offscreen`, a temporary Matplotlib
cache, and `pytest -p no:cacheprovider`. No source is installed for validation.

Focused checks include five signed launch/kernel centroid cases, five ordinary
PR workflow centroid cases, five real-host signed endpoint cases, Grid/request/
codec round-trip, disabled-beam index alignment, and real LC host wiring. Cardinal
positive/negative directions and arbitrary theta/phi use independently calculated
wavevector slopes, not preview geometry as a propagation oracle. Finite spectral
width uses 0.001 µm absolute / 0.001 relative centroid tolerance.

Retained footprint regressions independently require the 20 µm circle,
28.284271 by 20 µm oblique ellipse, azimuth rotation, roll, scene refresh and
material-index invariance. LaunchPlane additionally checks physical zoom scaling,
coincident number labels, unscaled off-aperture exits, fit behavior and fixed-size
arrowhead readability.

Final totals are appended after completion of validation. The complete suite
identified one stale P1 tooltip assertion requiring the old unsigned-angle
wording. It now requires signed external tilt, fixed phi, and grazing rejection,
while retaining every other selector/accessibility assertion. This is a bounded
contract-test update, not a production workaround.

Development corrections retained in the record: an initial test used a nonexistent
generic PR codec name and was corrected to the real timedependent codec; a label
spacing test exposed a 20-pixel assumption smaller than actual font height, fixed
by measuring label height. A full-suite attempt was superseded during self-review
when arrowhead readability at long-ray Fit scales was improved. A later
self-review aligned aperture editor keyboard tracking with canonical Grid, so
typed widths reach a request even before focus leaves the field; its regression
types the value rather than only calling setValue. Final validation uses both
corrections. No production physics workaround was added.

## Self-review and exclusions

All production changes have the bounded reasons above. No forward resolver equations,
propagators, power/current algorithms, capture, screens, material integrators,
carrier diagnostics, continuation or schemas were changed. Internal ray positions
are transient, Product-owned derived data; they are not new persisted intent.
Normal/disabled/unavailable states remain explicit. Signed intent is the only
new accepted physical-input domain; scalar qualifications remain unchanged.

Reconstruction, installed packages, authoritative working files, Research,
active prompts and prior worktrees are outside the candidate. Remote continuation
remains deferred/not authorized. PR scattering equivalence remains on scientific
hold/unestablished. No automatic persistent Target mode, focused launch, beam-normal screens,
reconstruction, power curves, P2A-2, scattering or fanning work is included.

## Remaining human native acceptance

After separate review, commits, coordinated integration and installation approval:

1. Change positive/negative theta without changing phi; confirm intuitive opposing
   directions and motion toward the predicted internal exit.
2. Confirm material-index/length changes update ray endpoints and leave external
   intent and ideal entrance footprints unchanged.
3. Change x/y full widths directly on Input face and confirm Grid agrees.
4. Use Fit on opposing/crossing/multiple beams with exits outside the aperture;
   then Full Aperture. No ray should shorten or change physical endpoint.
5. Confirm only beam numbers appear on the plot, caption is Input face, and no
   Laboratory x/y title or verbose label overlap remains.
6. Retain the normal 20 µm circle, 45° x ellipse, phi=90° rotation and independent
   roll behavior; inspect readability through resize and zoom.

Automated validation does not complete this human gate.

## Earlier candidate check commands

Run under Python 3.12.13 with the environment described above:

```text
python -m pytest -p no:cacheprovider tests/test_native_design_remediation.py -q
python -m pytest -p no:cacheprovider tests/test_native_design_remediation.py tests/test_host_footprint_preview.py tests/test_physical_launch_resolver.py tests/test_beam_panel_launchplane.py tests/test_launchplane_adapter.py tests/test_input_screen_editor.py tests/test_image_input_display_parity.py tests/test_pr_gui_request.py tests/test_experiment_gui.py tests/test_pr_scalar_diffraction.py -q
python -m pytest -p no:cacheprovider tests/ -q
```

Focused remediation: **18 passed**. Affected Product set: **194 passed**.
LaunchPlane final complete suite: **106 passed**. Syntax compilation,
Markdown/local links, whitespace and `git diff --check` pass for both candidates.

## Earlier candidate full-suite result

Complete final Product suite: **1923 passed, 77 skipped**, no failures
(533.65s (0:08:53)). The separate corrected P1 tooltip/accessibility module has
**26 passed**; all are included in the final complete suite.

Development self-review passes for scope, scientific non-change, signed-intent
validation, derived-data ownership, schema/transport compatibility, public API
qualification, executable acceptance coverage, documentation and whitespace.
Human native acceptance and coordinated delivery remain separate pending gates.

The earlier candidate protected Product inventory was 4,345 historical paths with
fingerprint `5dba5b06f9d2ee6d062e8f3f81fb0b27404a9e7c69b236ee8fb8fb2c4818a747`.
The reconstruction candidate, authoritative repositories, Research and installed
new-user environment remain unchanged. Nothing is staged or committed.

## Interactive ray-handle review remediation

The read-only review identified a real hit-testing defect: the visible center was
device-sized but noninteractive, while the parent beam used a scene-sized hit
shape. The old center test used setPos, bypassing the viewport. The review also
found stale user guidance about tilt dragging. The subsequent human instruction
explicitly superseded center-only editing and authorized approximate physical-exit
dragging with Product-owned inversion and manual reciprocal crossing.

Handles are now scene-owned device-sized items, with global stacking bands:
aperture/axes (0), footprints (1), physical shafts/number labels (10), centers
(20), arrowheads (30). This gives arrowhead priority across different beams,
not only within each beam. Shafts have no interactive shape and footprints
accept no mouse buttons. The enlarged arrowhead tip remains at the exact
physical exit. At zero tilt it stays visible and interactive over the center;
no artificial shaft length is added. Objects-list beam selection and a focus
ring identify the selected beam/handle.

Product's `external_direction_for_exit` converts displacement/L into a normalized
forward internal direction, then conserved tangential k, then external theta/phi.
It rejects unreachable/grazing exits without clamping. Among equivalent signed
representations it chooses the nearest current azimuth; equal-distance ties retain
theta sign and zero displacement retains phi/signed zero. This avoids gratuitous
180-degree azimuth jumps through normal incidence. No existing forward function
is edited. LaunchPlane dispatches physical coordinates through a callback and
has no duplicated optical inversion or persisted endpoint constraint.

Mouse drags preserve the cursor-to-handle offset. Center drags retain direction.
Keyboard arrows use one device pixel, Shift ten, mapped by the inverse current
view transform; only the focused canvas handle consumes these keys. Spinboxes
retain their own arrow-key behavior. Synchronous host geometry redelivery is
atomic for the focused/mouse-grabbed handle: hiding it between clear/redelivery
would lose Qt's mouse grab. When the host does not redeliver geometry, stale rays
are invalidated rather than retained under new intent. This is production
interaction lifecycle handling, not a test-owned QObject workaround.

Real viewport tests press, move repeatedly, and release; they assert the actual
Qt mouse grabber at every move. Coverage includes ordinary/low/high zoom, long-ray
Fit, zero overlap, exposed centers, cross-beam overlap, footprints, keyboard
scale/Shift/focus transfer and unavailable callbacks. Product uses the real PR
window for reciprocal drags, checks endpoint placement within screen-pixel
quantization, then independently samples/propagates each edited narrow beam.
Propagated centroids agree with the selected physical exits within 0.006 µm and
with opposite entrances within manual pixel tolerance plus 0.006 µm. Analytic
inverse tests separately cover both signs, quadrant/wrap boundaries, arbitrary
azimuth, high-index refraction, zero crossing, tie rule and invalid exits.

Initial new test failures were fixture issues: a small default aperture clamped a
50 µm center move; viewport layout had not settled after zoom; two default beam
names collided. Tests now use a sufficiently large explicit aperture, process
pending layout before input, and use unique names. No production workaround was
added for these failures. All completed test runs are retained separately from
earlier candidate evidence.

Directly affected guide/README text and the architecture amendment now describe
both handles, overlap priority, zero tilt, inverse ownership, signed branch,
keyboard focus and manual crossing. No automatic persistent Target mode exists.

### Current validation and protection

Focused Product: **35 passed**. Affected Product: **237 passed**.
Focused LaunchPlane: **21 passed**. Affected LaunchPlane: **114 passed**.
Full-suite and final static/protection results are recorded below after completion.

Validation uses Python 3.12.13 in the existing development environment, both
isolated src directories on PYTHONPATH, bytecode and pytest caches disabled,
Qt offscreen and an external temporary Matplotlib cache. No package is installed
or native application launched. The fresh Product historical inventory has
**4,346 paths**, fingerprint
`2560a2e10bedf87efe1674e9b06f55413333ab126cc6aee6590d999e31fabfd3`.
The additional active instruction is preserved; an older count is not imposed.
Authoritative repositories, reconstruction candidate, Research and installed
new-user environment are protected by before/after byte/size/mode/mtime inventory.

### Next human checks (still pending)

After separately authorized coordinated delivery:

1. Grab the zero-tilt arrowhead, drag it away, then grab the exposed center.
2. Drag the center and verify direction is retained.
3. Drag an arrowhead onto another beam's entrance; make reciprocal crossing.
4. Use arrows and Shift+arrows on each handle; confirm numeric controls update.
5. Confirm spinbox focus edits only the spinbox, then return focus to the canvas.
6. Compare the ray exit with propagated position within sampling qualifications.
7. Check Fit, Full Aperture, resize and zoom, including long rays and overlap.

Automated local results do not mark this human acceptance complete. No stage,
commit, integration, installation, cleanup or follow-on work is authorized here.

### Current commands and self-review scope

Under the validation environment above, current Product commands are:

```text
python -m pytest -p no:cacheprovider tests/test_inverse_ray_geometry.py tests/test_native_design_remediation.py -q
python -m pytest -p no:cacheprovider tests/test_inverse_ray_geometry.py tests/test_native_design_remediation.py tests/test_host_footprint_preview.py tests/test_physical_launch_resolver.py tests/test_beam_panel_launchplane.py tests/test_launchplane_adapter.py tests/test_input_screen_editor.py tests/test_image_input_display_parity.py tests/test_pr_gui_request.py tests/test_experiment_gui.py tests/test_pr_scalar_diffraction.py tests/test_residual_polish_p1.py -q
python -m pytest -p no:cacheprovider tests/ -q
```

Development self-review confirms scope/API ownership, numerical signs/units,
independent inverse/propagation oracles, explicit rejection paths, focus/mouse-grab
lifecycle, preserved schema/persistence behavior, documentation and candidate
exclusions. Every pre-existing function/class in physical_launch.py is source-
identical to the baseline; its only addition is external_direction_for_exit.
Existing forward Snell/tangential-k, scalar propagation, footprint, power/capture,
screens, coherence, carrier diagnostics, material and continuation code remain
unchanged by this interactive remediation. No forward inconsistency was found.

Relative to the earlier ten-file Product candidate, the only new paths are
src/lcprop/optics/physical_launch.py and tests/test_inverse_ray_geometry.py.
The architecture amendment, this record, user guide, beam_panel.py and
native-design test module change. The other five earlier candidate files remain
byte-identical. Relative to the earlier nine-file LaunchPlane candidate,
tests/test_ray_handles.py is added; README, its development record, canvas.py
and launchpane.py change; the other five files remain byte-identical.

## Completed interactive validation

Complete suite: **1940 passed, 77 skipped in 567.47s (0:09:27)**. No implementation changes occurred during
these final full-suite runs. Syntax compilation, Markdown/local-reference checks,
candidate whitespace and `git diff --check` pass. Fresh protection verification
confirms authoritative repositories, reconstruction, historical files and the
installed new-user environment unchanged. Both indexes remain empty.

This completes Development + Local Validation, not human native acceptance or
independent pre-commit review. Nothing is staged or committed.
