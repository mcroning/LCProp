# Stage 2C — scientific run isolation and persistent canvas safety review

Date: 2026-10-16. Reviewed HEAD: `9151e6bc7db8adb1a6788942e7be5bf315063fa2`, including the existing uncommitted Stage 2A/2B candidate.

## Decision

**CONDITIONAL GO.** Persistent canvas editing is scientifically defensible under the revised draft-only delayed-event policy. It does not require rejecting every old complete mouse gesture. However, the current implementation does not establish that a one-click Run executes exactly the values visually present in the browser at click time. Do not implement checkbox removal alone and claim the entire proposed contract is satisfied.

Proceed only with explicit run-request capture, lifetime/busy guards, atomic completed-record presentation, and a reviewed-request execution gate described below. If literal single-click browser-state equality is nonnegotiable, that portion remains **NO-GO pending a qualified frontend state/acknowledgment protocol**. No such protocol is established by the current source or prior tests. This review authorizes no implementation or interface-policy change.

## Evidence and scope

Read-only source review; no simulations, installations, browser sessions, or new test execution. Historical test evidence remains historical, not a fresh pass claim. No Product, notebook, tests, saved outputs, or earlier evidence edited. Only this report was added.

Inspected:

- `src/lcprop/notebook/interactive_lc.py`: request construction (27–68, 461–477), `CompletedRun` (91–108), execution (111–146), presentation/export (149–291), lifecycle and Run/load/reset (404–605).
- `src/lcprop/notebook/_beam_position_editor.py`: gesture capture, live previews, release, cancellation and disposal (142–295).
- `src/lcprop/notebook/lc_static_demo.py:65–82`: widths and result-derived diagnostics.
- `src/lcprop/lc/requests.py:127–146`, `core/beams.py`, `lc/experiment_codec.py`: frozen request models and serialization restrictions.
- `src/lcprop/runners/local.py:40–85`: synchronous operation dispatch and result conversion.
- Existing tests: `test_lc_static_interactive.py`, `test_lc_static_interactive_presentation.py`, `test_interactive_lc_dragging.py`, `test_interactive_lc_move_tilt.py`, `test_interactive_lc_preview_lifecycle.py`, `test_interactive_lc_display_publication.py`.
- Prior Stage 2C report and its bounded dispatch experiment. That experiment proves an ambiguity under synthetic delivery ordering, not actual Safari event reordering.
- Installed ipywidgets source in `lcprop-interactive-test`: `widgets/widget_button.py:90–109` dispatches a click without attaching the other widgets' values or a run revision; `widgets/widget_float.py` declares FloatText `continuous_update=False` (submission/blur, not every keystroke). No dependency changes or imports were necessary to inspect these files.

## What is already isolated

| Object/path | Actual ownership | Assessment |
|---|---|---|
| Draft request | `request()` constructs replacement BeamChannel/BeamStack/bias/request objects from controls and base | Ordinary later trait edits do not mutate the previously returned request |
| Running workflow | `_run` calls `execute(self.request(), ...)` once; LocalRunner receives that argument | No subsequent control lookup for scientific propagation is evident |
| Request runtime state | General request model permits `initial_A`/`initial_theta`; experiment codec rejects populated runtime state and machine-local output paths | Frozen dataclasses alone are not deep immutability; current bounded serializable UI scope avoids these mutable-array inputs |
| Completed request | Canonical JSON string; property decodes anew | Detached from draft and caller modifications |
| Scientific arrays | NPZ bytes; `arrays()` returns freshly loaded arrays | Subsequent edits to decoded arrays cannot change stored bytes |
| Histories/diagnostics | `diagnostics(completed,...)` serializes result slice summaries, iteration records, residuals and powers | Derived from the executed result, not current controls |
| Other metadata | Sampling warnings and slice metadata use the execution argument; environment collected during execution | No current-control reread found; provenance collection is not itself an independent installed-file audit |
| Completed plots/summary | Functions read only the supplied snapshot | No plot-data dependence on editable controls |
| Export | Request, NPZ, summary, slice metadata and regenerated figure all read one snapshot argument | Path control chooses destination only; scientific content remains snapshot-owned |

The current scope therefore supports strong isolation from **ordinary widget edits**, not an adversarial Python process mutating arbitrary internals. `self.completed` is replaceable; users can programmatically construct inconsistent CompletedRun values or modify modules. Those are outside the proposed browser interaction contract. Snapshot arrays are selected scientific products, not a promise to preserve every internal solver object.

## Gaps and required safeguards

1. **Capture before execution, explicitly.** Canonically encode and decode the bounded request once at the run admission boundary, store its exact serialized bytes/digest, and capture the time budget separately. Execute that detached request; all result metadata, slice selection and request export must use it. Verify the result request agrees with it. Current encoding occurs at the end of `execute`; current frozen scalar models are reassuring but do not justify a general deep-freeze claim.
2. **Guard all entry points.** `_run` and `_restore` check busy; `load()` and `_changed()` do not. Widget `disabled` is a UI property, not a backend authorization check. Add consistent busy/disposed guards for load/reset, gesture handlers and draft processing. Keep draft state separate from the running record even if queued trait updates arrive. A trait observer runs after a value changes; merely returning from `_changed()` does not undo that change. Either restore authoritative values under a notification-suppression guard while busy, or explicitly define such arrivals as deferred draft edits after completion. Do not claim they were rejected solely because the browser control was disabled.
3. **Permanent disposal.** `close()` currently closes only the editor; old Run/load/widget callbacks are not globally retired. Add a disposed state checked by controller entry points, including Run. A retired controller must not launch science or regain an interaction endpoint. Disconnect editor callbacks once on final disposal, never reconnect that retired editor.
4. **Atomic publication of a completed record.** `_run` builds the figure before assigning `self.completed`, but assigns it and the summary before `display_figure` serializes/displays the image. A rendering/display failure can leave a new snapshot with stale/cleared visible output while reporting that the previous result was retained. Pre-render the PNG/widget payload and summary before publishing the new record. Tie displayed result identity to snapshot identity; distinguish successful calculation with display failure from failed calculation. Keep previous record intact until preparation succeeds. Browser rendering cannot be made infallible; any view error must not silently mislabel ownership.
5. **One-shot execution admission.** A queued second click delivered after synchronous Run finishes sees `busy=False` and can start another run. Disabled buttons plus busy checks do not prove suppression of queued clicks. Require a consumed preparation token, not automatic rearming in `finally`. Failure/timeout also consumes the token. Explicitly prepare again for another run.
6. **Errors and invalid drafts.** Maintain a persistent canvas even on validation failure, but visibly mark any retained launch image as stale/invalid and disable run admission. Do not silently use the last valid request. Current `_changed()` closes the editor on ValueError/TypeError; this must change for lifetime canvas identity.

No identified ordinary draft path rewrites existing snapshot bytes. These gaps concern admission, lifecycle, display consistency and strength of guarantees, not a discovered numerical mutation or altered scientific model.

## The next Run: browser-visible versus kernel-received state

The strongest user requirement is not satisfied simply by freezing the request in `_run`.

- Numeric text can be visible before submission; FloatText is not continuously committed by default.
- A graphical drag shows provisional geometry/intensity before release commits controls. That preview is intentionally not yet the scientific draft used by Run.
- A mouse release can update kernel controls and send their new values back while the browser still displays older values. A Run click received in that interval can capture the newer kernel request.
- Conversely a browser edit may not yet have reached Python when a click is processed. The review has not established ordering across all relevant event, widget and rendering paths.
- A click message contains neither an atomic multibeam form snapshot nor an acknowledgment that a particular kernel revision was displayed.

These are possible consistency gaps, not claims that each occurred in Safari. Sleep/debounce, `continuous_update=True`, a generic “validated” label, queue draining, or a hash stored only in Python is not a proof of browser-visible equality. Changing the draft after a completed run is safe; running an unintended next draft is still scientifically consequential.

### Smallest defensible no-custom-JavaScript policy

Use ordinary widgets for a **Prepare/Review → Run reviewed configuration** transaction:

1. Prepare cancels provisional gestures, validates and freezes the kernel-accepted request, creates an immutable review record and a fresh one-shot confirmation endpoint.
2. Display that record read-only, with all editable scientific values (both beams, absolute/relative phase, effective groups, voltage), preset/hidden parameter context, request digest/revision, and a way to inspect the full serialized request. A digest alone is not a meaningful review.
3. The user explicitly confirms this displayed record. Run executes the frozen review object, never rereads editable controls. Confirmation endpoint is bound to that specific record and is consumed before executing.
4. Any accepted draft edit, load or reset invalidates the pending review and its endpoint; old queued confirmations cannot confirm a later preparation. Preparing again creates a new identity. Do not reconnect an old confirmation callback to a new request.
5. After run/failure, require a new explicit preparation. Delayed gestures may update only the visibly identified draft and must refresh controls/preview; they cannot arm execution.

This changes the promise to **“Run executes exactly the reviewed configuration you explicitly confirmed.”** It avoids relying on unsynchronized editable fields at click time, without a custom frontend or science changes. It requires user acceptance of that interaction contract; do not silently add a two-stage flow while claiming unchanged one-click semantics. It still requires real-browser qualification and assumes the user reviews the indicated immutable record. Strict automatic equality with every live editable field at the instantaneous click requires a separately qualified frontend snapshot/acknowledgment mechanism.

## Persistent canvas under the revised contract

Keep one canvas/model and original handlers until controller disposal. Replace boundary retirement with gesture cancellation plus busy/loading/disposed guards. After Run/load/reset, resynchronize marker positions, arrows and canonical intensity from the accepted draft. No callback reconnection is needed. A trailing old release with no active gesture is ignored. A complete delayed gesture after resumption is an allowed draft edit, visible through numeric controls and preview, and invalidates pending review. Editing never calls propagation. Existing canonical launch, geometry, release batching and live-preview caps remain.

The relaxation has no inherent effect on previous numerical records. Risks remain user surprise, delayed overwrite of a freshly loaded draft, selection/action interpretation at event delivery, and accidental subsequent execution. Explicit review, invalidation and one-shot Run protect execution; UI messaging addresses surprise. Do not describe delayed events as user-intended fresh gestures or guarantee their original action/beam without frontend provenance.

A draft-versus-completed indicator is not needed to keep bytes immutable, but is strongly recommended for scientific usability. A draft-versus-reviewed indicator and invalidation state are required for the proposed admission gate. Mark provisional drag preview separately; use full request identity, not only beam-position comparison. In-memory revision counters identify accepted kernel state, not browser event age.

## Bounded acceptance tests (no propagation)

Use synthetic coordinate-coded CompletedRun data and a fake LocalRunner/execute recording exact inputs; solver calls outside explicit confirmation must fail the test. Existing retained numerical parity remains the scientific reference, not a new comparison claim.

1. Capture request JSON/digest and budget; inject numeric, move, tilt, load and reset attempts during fake execution/progress. Assert request and result-associated request, histories, residuals, warnings, coordinates and environment record remain bound to capture. Assert busy policy explicitly.
2. Hash request_json, arrays_npz and summary_json before/after post-run edits; mutate decoded copies and verify stored bytes unchanged. Verify completed labels, axes, mesh values, color scales and widths remain snapshot-derived.
3. Export before/after edits to separate temporary folders; compare all scientific JSON/NPZ identities and deterministic rendered figure bytes in the same environment. Test timeout/failure preserving prior snapshot and image. Inject plot construction, PNG serialization and publication failures to check truthful ownership/status.
4. Verify one canvas and one visible host throughout normal editing and Run/load/reset; callbacks stay attached to the same live endpoint. Final disposal rejects every event and Run/load/reset. No extra Jupyter display publications; actual Safari visual acceptance still required.
5. Deliver old motion/release after cancellation: no mutation. Deliver a complete delayed gesture after resumption: only draft changes, visible controls/preview synchronize, pending review invalidates, no solver call and completed hashes unchanged.
6. Deliver delayed numeric updates and duplicate/stale Run clicks before/during/after execution. Each preparation authorizes at most one call; consumed/stale confirmations never execute or pick up another request.
7. Prepare request A, change draft to B, deliver confirmation A: reject. Prepare B, deliver old confirmation A: reject. Confirm B: exactly B is passed. Modify all draft values during fake execution: outputs/metadata still describe B.
8. Preserve canonical numeric/graphical request and launch parity for Move/Tilt, coherent/incoherent beams, live-preview bounds, invalid-gesture recovery and optional static/headless operation. No propagation required.
9. Real Safari/Jupyter manual tests: rapid typing then Prepare, pending blur, rapid drag/Prepare, repeated Run clicks, load/reset after drag, kernel busy delays, reconnect, two displays of one model. Inspect reviewed values against captured request. Kernel mocks cannot establish DOM visibility or human intent.

## Bounded implementation recommendation

First agree to reviewed-request confirmation versus literal one-click visible-draft equality. Then implement only notebook-controller capture/admission/guards, atomic completed presentation, and editor cancel/resynchronize lifecycle; retain one optional canvas and static fallback. Add the tests above and update notebook-interface documentation. No engine, codec, numerical normalization, solver, dependency or request-schema changes are needed. Do not broaden to multithreaded execution or a new event protocol in this milestone.

**Stage 2C should proceed only after these safeguards and the revised execution UX are approved.** Scientific isolation is achievable with existing APIs. Browser-visible one-click equality is not presently demonstrated, and persistent canvas reuse alone cannot provide it.
