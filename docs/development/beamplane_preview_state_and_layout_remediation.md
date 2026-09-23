# BeamPlane preview state and layout remediation

## Scope and preflight

Product baseline: `59eaf50d519d2ea0476c6ebbff5e00c2f35fa02a`.
LaunchPlane baseline: `d5e064c4a24475362345fc43f2d55191086f2a12`.
Both candidates use isolated `fix/preview-state-layout` branches.

This is local presentation development and regression validation. Equations/model
changed: N/A. Product production changes are confined to the shared BeamPanel
editor/request synchronization boundary. The forward and inverse
physical launch resolver, interface quadratic form, internal geometry, flux,
propagation, material evolution, screens, coherence, carrier diagnostics,
continuation and persistence schemas remain unchanged. The latent direct-import
persistence cycle is explicitly outside this task. No remote or GPU work is
permitted. The installed native-acceptance environment is not used for tests or
modified. Nothing is staged, committed or integrated.

## Native reproduction and established root cause

The native sequence is now established: with w1=w2=20 micrometres and theta=0,
typing 30 without Return, Tab or focus loss leaves a circle representing committed
zero-angle intent. Direct Run consumes 30 degrees; returning to Beam shows the
correct ellipse. Tab commits 30 immediately and produces the ellipse before Run.
This is pending editor text versus committed model/preview state, not a mutation
of the resolver or worker geometry. The earlier unresolved diagnosis is superseded
by this native evidence. Human acceptance of the correction remains pending.

Numeric editors use keyboard tracking disabled. Intermediate typing therefore
emits no physical-intent update. A valid focus/Return commit emits `valueChanged`,
updates the immutable LaunchPlane beam through `_apply_selected_changes`, then
emits `beamStackChanged`. Product synchronously rebuilds its resolved preview.
At baseline, `BeamPanel.beams()` separately interpreted pending spinbox text during
request construction. Execution could consequently consume newly committed intent
while the last preview the user saw still represented the earlier committed state.
The native observation does not establish that propagation used a wrong angle.

The correction makes this boundary explicit and shared. LaunchPlane owns
`commit_pending_edits()`: validate all beams' pending numeric text using Qt's
existing validators, parse it with the same editor conversion, collect changes,
and publish one fully validated canonical stack update. Unchanged rounded editor values do not
replace full-precision model values. A reentrancy guard permits synchronous host
request inspection without a duplicate update. Names and editable laser labels
also join this boundary; enabled is an immediate checkbox update, and the physical
profile is not an editable pending-text control.

`BeamPanel.beams()` calls that API before adaptation. The synchronous sequence is:

```text
pending text -> validate/commit -> immutable LaunchPlane stack
-> beamStackChanged -> Product screen binding/context preview refresh
-> resolved Input-face geometry -> request adaptation -> worker dispatch
```

The Input Screen preview callback now reads committed state without flushing
pending editors. Passive preview refresh must not act as a second request boundary.
Both LC and PR request builders use the same BeamPanel boundary; no main-window
fork or scientific parser is introduced. Product requires the companion pending-
edit API rather than silently falling back to old editor semantics.

Invalid or out-of-range text blocks request construction through existing Product
error presentation. Qt otherwise silently replaces invalid text with its previous
value on focus loss **or Hide**: Run switches to Results before constructing the
request. `IntentSpinBox` suppresses that invalid-text fixup for focus transitions,
Hide and Return, retaining the visible text so validation can reject it. Valid
text retains ordinary Qt behavior. No grazing-angle or signed-theta rule changes.

## Production path and ownership

- [BeamPanel](../../src/lcprop/gui/panels/beam_panel.py) owns optical context and
  adapts the current immutable LaunchPlane stack. `_boundary_changed` builds a
  launch and diagonalizes each `interface_quadratic`; it does not use
  `internal_quadratic` or a result-derived fit.
- Enabled channels retain their stack order. Ray endpoints explicitly map back
  through enabled stack indices. Contours follow enabled-channel order.
- `LaunchPlaneWidget.set_resolved_preview` copies coordinate pairs before making
  independent scene paths. It does not resolve optical geometry.
- `set_aperture` preserves the stack, rebuilds the scene, then redelivers host
  geometry. Programmatic stack replacement also redelivers geometry.
- Selection synchronizes the inspector; Fit/Full Aperture/resize change view
  presentation. None replaces launch intent with Results arrays.
- [PR main window](../../src/lcprop/pr/gui/main_window.py) constructs the request
  through `BeamPanel.beams`, disables/re-enables configuration around execution,
  delivers results to Results, and refreshes checkpoint readiness. The result
  path does not supply a new footprint or material index to the Beam tab.

No beam association error was demonstrated. The tests retain each stack index,
immutable beam definition, optical context, quadratic form and scene contour.
This is a local association check, not a new persisted identity mechanism.
Duplication intentionally offsets the new center by 5 micrometres in x and y;
its shape remains equal and deleting it restores the original associations.
Disabled beams do not consume enabled-contour positions.

## Executable invariance and geometry evidence

[Preview lifecycle regressions](../../tests/test_preview_state_lifecycle.py)
use the real PR main window, ordinary 200 by 200 micrometre aperture, two enabled
coherent beams with separated centers, and real viewport arrowhead events through
the existing Product inverse. The real Run button handler starts the normal
worker/operation and delivers a genuine small local PR result. Results rendering
is observed with a forwarding wrapper, not replaced with a synthetic result.

The test compares exact beam definitions, material index, interaction length,
quadratic forms and all 65 physical contour points before/after each lifecycle
boundary. It verifies the actual worker request uses the displayed launch stack.
Quadratic equality implies unchanged principal radii/orientation; point equality
also checks scene association and centers. Ellipse conic membership uses a
2e-14 absolute floating-point tolerance. The test detects circle-to-ellipse
mutation or exchanging either beam's contour.

Independent oracles require:

- 20 micrometre beam-normal radii at normal incidence: 20 micrometre circle;
- 45-degree external incidence: 20/cos(45 degrees) = 28.284271... micrometres
  along incidence and 20 micrometres perpendicular;
- 90-degree azimuth: long axis along y;
- equivalent negative incidence: the same footprint sizes;
- changing internal index from 2 to 2.4: identical interface contour;
- duplication, deletion and disabled channels: correct association.

These are development regressions, not commissioned propagation evidence. The
small worker run verifies presentation lifecycle, not high-angle numerical
accuracy on its deliberately coarse grid.

## Companion LaunchPlane layout change

The figure title remains `Input face`; adjacent layout-managed `x (µm)` and
`y (µm)` designators restore axis identity without adding laboratory-coordinate
prose or zoom-dependent text anchors. Only beam numbers remain inside the plot.

Qualification text remains intact in a wrapping, vertically scrollable region
outside the plot. Its height bounds derive from font metrics, not screenshot
widths. Long multibeam prose cannot consume the entire plot height. The inspector
uses more row spacing, Qt form wrapping and a separate full-width derived-k row;
vertical scrolling keeps the lower values reachable rather than compressing
rows. No width-specific branches or scientific-value changes are introduced.

Old tests requiring zero vertical inspector scrolling are replaced by checks
that derived values can be brought into the viewport without horizontal
scrolling. This implements the explicitly authorized scrolling-based layout,
not a relaxation of readability requirements. Existing ray-handle, keyboard,
zero-tilt, zoom, priority and inverse tests remain in validation.

## Earlier layout validation

Python 3.12 development environment; isolated source trees on `PYTHONPATH`,
`QT_QPA_PLATFORM=offscreen`, `PYTHONDONTWRITEBYTECODE=1`, temporary Matplotlib cache,
and pytest cache provider disabled. No package installation or native launch.

- Initial baseline lifecycle/oracle/pending-text checks: 6 passed.
- Product focused preview/geometry/interaction set: 45 passed before the added
  pending-text qualification; final totals recorded below.
- Product affected PR/GUI/adapter/physical-launch set: 176 passed before that
  added qualification; final totals recorded below.
- Companion LaunchPlane focused preview/layout/handle set: 17 passed.
- Companion LaunchPlane complete suite: 118 passed after layout assertions were
  aligned with accessible vertical scrolling.

Preliminary development failures: the chosen `lcprop-test` interpreter lacked
pytest; no environment was changed, and validation uses the existing `lcprop`
Python 3.12 environment. The first duplicate test incorrectly expected no center
offset; it was corrected to the existing documented 5-micrometre displacement.
The initial companion full suite exposed stale no-scroll expectations and a
read-only label assertion. Read-only wording was retained; vertical scrolling
assertions now require accessibility. No physics change addressed any failure.

## Pending-editor regressions

[Pending editor tests](../../tests/test_pending_beam_editor_sync.py) use the real
PR window and `QTest` keyboard/mouse events. They type 30 into the focused theta
editor without Return/Tab, verify the old committed circle, click the real Run
button, and observe the immutable request and actual ellipse before forwarding
to the ordinary worker. They test both focus-taking and non-focus-taking Run
buttons. Exactly one beam update occurs. Real result delivery and return to Beam
preserve the synchronized ellipse. With n=2, an adequately sampled zero-gain run
has centroid displacement `L * sin(30) / sqrt(2^2 - sin(30)^2)`, independently
checking tilted propagation (0.005 micrometre absolute tolerance).

Separate Tab/Return tests establish the same intent and ellipse; Tab advances to
phi. Direct Run covers phi, both radii, both centers, roll, external index, power,
wavelength and phase. Invalid intermediate minus and out-of-range 90/100 degree
text cannot dispatch or silently restore the old value. The real LC window uses
the same request boundary for valid radius edits and invalid theta rejection.

Companion LaunchPlane tests cover an atomic multiple-field commit, reentrant
inspection, invalid batch rejection without partial mutation, invalid Tab/Return,
and preservation of full-precision model values when rounded text is unchanged.
Existing lifecycle, independent footprint oracles and layout/interaction tests
remain intact. No test-generated presentation callback substitutes for the normal
worker path in the real propagation regression.

## First pending-editor correction validation (superseded)

- product-focused: **24 passed in 6.16s**.
- product-affected: **207 passed in 42.47s**.
- product-full: **1964 passed, 77 skipped in 540.21s**.
- launchplane-focused: **22 passed in 0.95s**.
- launchplane-full: **123 passed in 1.54s**.

Preliminary failures exposed Qt's additional Hide-triggered invalid-text fixup;
focus-only interception was insufficient. The production fix and real Run tests
now cover it. One test expected suffix-free raw text although Qt appended the unit;
it now checks the editor's `cleanText`, without changing validation or beam values.

## Remaining native acceptance

After separately approved review, coordinated delivery and installation:

1. Start with the 20-micrometre circle, type theta=30 without Return/Tab, click Run.
2. Confirm preview/request commit before execution and the same ellipse after Run.
3. Repeat Tab and Return commits; check invalid text blocks Run visibly.
4. Check reciprocal rays, center/arrowhead dragging, keyboard/Shift nudges,
   Fit/Full Aperture, axes, status and inspector readability at laptop sizes.

The native cause is established and the synchronization correction is locally
validated; human acceptance is not complete. No installed-environment change,
source integration, remote/GPU execution, persistence-cycle fix or follow-on
scientific work is included.

## First correction self-review and protection (superseded)

Syntax compilation in memory, Markdown fences/local references, candidate
whitespace and `git diff --check` pass. Complete validation commands and logs are
retained externally as `pending-editor-validation-commands.txt` and
`pending-editor-*.log`; they are not candidate files.

Product scope is one shared BeamPanel source file, two regression modules and
this record. LaunchPlane scope is its editor source, three regression modules and
its record. The earlier lifecycle, editor-layout and preview-layout test files
remain byte-identical to the prior candidate. No unrelated hunks or changes to
scientific source, schemas, numerical tolerances or experimental-status labels
are included. The new editor API requires coordinated Product/LaunchPlane
delivery; no existing public positional argument changes. Numerical probes are
local regression evidence, not commissioned scientific results.

Authoritative baselines, Research, reconstruction candidate and all 19,733
installed-environment files retain their fresh pre-task content/metadata inventory.
The fresh historical Product inventory is 4,348 paths, fingerprint
`929fe98e9a4162687e4ce21b140f85d11c5233f3e4f5897f10ed3de3b011bc68`.
Both isolated indexes remain empty. No stage, commit, integration, installation,
native application, remote/GPU access or follow-on work occurred.

Development and Local Validation pass. The next gate is an independent read-only
review of the complete combined candidates, including the former native-cause
blocker and Qt invalid-text Hide handling. Human native acceptance remains pending.

## Read-only review P2 follow-up: unresolved edit ownership

The review found that preserving invalid text against Qt fixup was insufficient:
`_sync_inspector()` still replaced every field from the model when an unrelated
field committed. The real theta-minus -> Tab -> phi change/Return sequence
reproduced the lost theta text before this correction. The center case also
reproduced it. Initial positive-minimum radius/index stimuli used a minus key,
which Qt rejects without replacing their valid text; final regressions instead
use real Select All/Backspace and assert that the resulting text is unacceptable.
This separates a genuine unresolved edit from a keystroke Qt never accepted.

LaunchPlane now distinguishes three states:

- committed valid physical intent in the immutable beam model;
- valid pending text, recognized by Qt but not yet equal to its committed editor
  value;
- invalid/intermediate text, rejected by the editor's existing Qt validator.

`_editor_drafts` owns numeric draft text per beam for this editor session.
Entries retain the actual immutable beam object; object identity is a local
ownership key, not a UUID, schema field or persisted carrier identity. Keeping
the reference prevents object-ID reuse. `_editor_owner` identifies the beam whose
controls are currently displayed. Before model-to-widget refresh or selection
replacement, `_capture_pending_edits()` saves only differing or unacceptable
text. Clean controls can refresh normally. After refresh, each draft is restored
verbatim. This applies uniformly to all numeric/angle fields, not just theta.

Ordinary field commits still update their valid model field and preview. Other
unresolved drafts survive. Immutable beam replacements from field commits,
center moves or ray edits transfer draft ownership to the replacement object.
Selection changes retain independent drafts per beam and restore them on return.
Scene/aperture rebuilds retain the same beam objects; host rebuild, preview
redelivery, Fit, Full Aperture and layout refresh preserve the drafts. Duplication
copies committed intent without attaching the original beam's draft to the new
beam. There is no automatic discard or new global cancel action. Deleting a beam
with pending edits is blocked with an explanatory status; wholesale replacement
that would lose an owner raises a validation error instead of dropping its draft.
Existing Escape still cancels placement only.

Before request construction, `commit_pending_edits()` captures the displayed
controls and validates drafts on **all** beams, including unselected beams.
An invalid draft selects its owning beam, scrolls to/focuses the editor and raises
the existing field-qualified error. Product's established error presentation
blocks dispatch. Text remains available on return to Beam. Normal focus commits
of another valid field may still occur; no request containing unresolved input
is dispatched. Preview continues to represent committed model intent, not a
claim that invalid text has been corrected.

After all draft and model validation succeeds, `_apply_beam_updates()` assigns
the complete validated stack before refreshing scene items and publishing one
stack signal. Normal single-beam commits use the same helper. Synchronous host
reentry therefore sees a complete stack, never a partially applied multi-beam
batch. Unchanged rounded controls do not rewrite full-precision model values.
Correcting a field to a valid new value or explicitly back to its original value
clears that draft through normal commit/request capture. Invalid text on other
fields or beams continues to block Run until corrected.

## P2 regressions and retained behavior

The real PR-window cross-field regressions cover:

- invalid theta -> valid phi;
- invalid radius -> valid center;
- invalid center -> valid roll;
- invalid external index -> valid radius.

Each confirms preserved invalid text, a successful valid neighboring commit,
committed-model preview consistency, actual Run-button rejection with the invalid
field identified, and successful request dispatch after explicit correction.
A two-beam real-window case additionally changes selection, rebuilds the host
aperture, redelivers preview, fits/resizes, and invokes Run while the other beam
is selected. Run returns ownership to the invalid beam and blocks until repaired.

Companion tests cover multiple unresolved fields/beams, sequential correction,
local owner transfer, duplicate/delete/replacement safeguards, explicit correction
to the original value, and one complete stack publication for valid drafts on
multiple beams with a synchronous reentrant observer. Existing direct theta Run
real-worker propagation/result tests, Tab/Return, lifecycle invariance, geometry
oracles, LC shared boundary, axes/status/inspector and mouse/keyboard/ray tests
remain in validation. No new Product production change was needed for this P2
follow-up; the retained shared BeamPanel boundary consumes the corrected API.

No resolver, propagation, normalization, screens, schema, material or continuation
change is included. Selection preservation required no destructive UX decision:
independent drafts remain attached to their original beams. Native acceptance
remains pending; only offscreen Development + Local Validation is performed.

## Final P2 remediation validation and self-review

- product-focused: **30 passed in 7.94s**.
- product-affected: **213 passed in 44.29s**.
- product-full: **1970 passed, 77 skipped in 544.05s**.
- lp-focused: **10 passed in 0.47s**.
- lp-affected: **53 passed in 1.03s**.
- lp-full: **128 passed in 1.28s**.

Syntax compilation in memory, Markdown/local-link checks, whitespace and
`git diff --check` pass. The external command record remains
`pending-editor-validation-commands.txt`; final logs are `invalid-drafts-*.log`.
An initial full Product run was deliberately interrupted to include the added
center/aperture-range regression, then restarted in full. It is not counted as
completed evidence. The final complete run above includes the entire candidate.

Preliminary test corrections were limited to using actual unacceptable text in
positive-minimum controls (Select All/Backspace; minus alone is rejected by Qt),
and allowing a normal valid neighboring commit when validation transfers focus.
The contract blocks unresolved requests, not independent valid model edits.
Direct batch validation without focus transfer still checks no partial mutation.
The added host center/range-refresh check passes without another source change.

Self-review: the P2 production correction is confined to LaunchPlane's editor.
The existing Product BeamPanel source, prior lifecycle test, LaunchPlane layout
and inspector tests remain unchanged from the earlier candidate. The combined
candidate still has exactly four Product and five LaunchPlane files. Session
editor ownership does not introduce scientific state, persistent IDs or schema
changes. All physical mappings, algorithms, power/capture, screens, material and
continuation behavior remain unchanged. Coordinated delivery is still required.

Fresh protection verification matches authoritative Product/LaunchPlane/Research,
the reconstruction candidate, all 4,348 historical Product paths and all 19,733
installed-environment files. Historical Product fingerprint remains
`929fe98e9a4162687e4ce21b140f85d11c5233f3e4f5897f10ed3de3b011bc68`.
Both isolated indexes are empty. No commit, integration, installation, native
application launch, remote/GPU access or follow-on work occurred.

Development + Local Validation closes the P2 failure: unresolved edits survive
unrelated commits/refreshes, Run rejects them across beam selections, and explicit
correction permits canonical synchronization. Independent read-only review is
next. Human native acceptance remains pending.
