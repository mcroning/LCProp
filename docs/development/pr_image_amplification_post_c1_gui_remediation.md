# PR Image Amplification Post-C1 GUI Remediation

## Purpose

This bounded remediation corrects three usability defects found during manual
testing after the BeamPanel-based Image Amplification Stage C1 milestone. It
does not change PR physics, propagation, reconstruction, or result products.

## Canonical Role Refresh

The C1 role selectors listened directly to LaunchPlane's interactive
`beamStackChanged` signal. Programmatic replacement through
`BeamPanel.set_beam_stack_definition()` updated the visible BeamPanel but did
not emit that signal, leaving the role selectors stale until request
construction forced a refresh.

`BeamPanel` now exposes one canonical `beamStackChanged` relay covering both
interactive LaunchPlane edits and programmatic immutable stack replacement.
The PR role selectors consume that relay and always rebuild their choices from
the enabled-channel ordering that also produces `LaunchConfiguration`.
Selections follow uniquely identifiable beams through reorder and a single
immutable rename; ambiguous or removed roles are cleared rather than silently
reassigned.

## Screen Placement

The screen editor now initializes a newly attached screen at the selected
beam's current `x_um`, `y_um` center. This happens only on the transition from
no screen to a new intensity screen. Once the screen exists, its position is
independent: later beam motion does not drag an intentionally offset screen.

The compact **Center on beam** action explicitly resets both screen-center
coordinates from the currently selected beam. It does not change the source,
width, height, inversion, or any other screen property, and it refreshes the
preview and power accounting through the normal editor path.

## Compact Editor Layout

The two vertically stacked previews are retained as two data views but shown
in one preview area selected by **Transmission** or **Post-screen beam**. The
screen source, geometry, power, throughput, and actionable status remain
available. Controls that have no meaning when `Screen = None` are hidden until
an intensity screen is selected, and the empty standard-image notice has a
bounded height.

## Scientific Non-Change

Explicitly configured `LaunchConfiguration` objects are unchanged. Screen
application, passive power semantics, reduced-TD Image Amplification,
carrier isolation, back-propagation, gain, correlation, normalized RMSE, and
power drift are untouched. The only changed numerical input is the corrected
default center for a newly created GUI screen.

Remote execution and experiment persistence remain deferred exactly as in
Stage C1.

## Validation

Focused GUI coverage verifies live add, duplicate, rename, enable, disable,
delete, and request-role refresh; new-screen centering and independent manual
offsets; the Center on beam action; per-beam attachment defaults; compact
preview selection; power visibility; and Screen=None compaction. Existing
B1/B2/B3 and C1 scientific-equivalence and reconstructed-image product tests
remain the regression gates.

The focused shared-screen and Image Amplification suite passes 65 tests. The
complete PR suite passes 395 tests with 57 CuPy-dependent skips on the local
CPU host.
