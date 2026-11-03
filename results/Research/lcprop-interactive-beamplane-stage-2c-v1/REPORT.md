# Stage 2C — controlled Beam 2 transition and frame-delivery isolation

Date: 2026-10-16. **NO-GO for shipping a rendering correction on the present evidence.** Actual InteractiveLC callbacks produce correct two-beam arrays, normalization and ordered PNG streams. Deliberately omitted or stale-painted transition frames reproduce persistent corruption. A delivered full-frame recovery repairs it, but a skipped recovery followed by an unchanged differential frame does not. No speculative Product correction was implemented.

## Reproduction method and scope

Added `tests/test_interactive_lc_frame_delivery.py`, using the existing real ipympl Canvas/Matplotlib renderer and actual `InteractiveLC` trait callbacks. It blocks the execution entry point, simulates the normal widget initialization messages, intercepts actual outgoing PNG payloads/mode notifications, and records incoming draw/refresh events. It does not substitute an alternate launch calculation. This is an actual Python/ipympl backend test with a controlled PNG compositor, **not a real browser or Jupyter frontend reproduction**.

Sequence, without sleeps: initialize one beam → enable Beam 2 → Beam 2 radius 12 µm → disable → enable → radius 7 µm → disable → enable. Each accepted edit sends its committed frame; a simulated pending ordinary draw reply follows it. These replies exercise the coexistence of direct committed draws and deferred draw requests. No material evolution, propagation, parameter sweep or installation.

For every state, the test retains the exact canonical request JSON, canonical intensity, Matplotlib mesh, color limits and renderer RGBA. It asserts exact canonical/mesh equality, min/max color-limit equality, and the correct marker count. Every ordered frame prefix is composited and checked pixel-for-pixel against its corresponding renderer. Delayed delivery **with order retained** is equivalent to replaying that same sequence and is not a failure by itself.

## Identical-state comparison

Captured arrays are float64, 128×128; renderer images are uint8 RGBA, 375×375×4. These entrance-plane arrays and rendered pixels are paired with the same two-beam request, avoiding the earlier ambiguity between a one-beam screenshot and a two-beam UI.

| Captured state | Beams | Intensity limits | Canonical vs mesh | Ordered PNG composition vs renderer |
|---|---:|---|---|---|
| First Beam 2 enable | 2 | 9.201157830653937e-28 to 0.003170471428964832 | Exact | Exact |
| Final two-beam state (Beam 2 radius 7 µm) | 2 | 1.0778456747969344e-29 to 0.0045248951242747954 | Exact | Exact |

Final executed-draft request serialization SHA-256: `fa1ef8a6661c0e1c7d7d94abc5a626eaab68d12fabc559e66412dc3872166eed` (sorted canonical request JSON, default JSON separators). No calculation was executed: “request identity” here identifies the sampled draft, not a completed scientific run.

The canonical model remains responsible for combined-beam normalization. No assumption that adding equal beam powers must double the numerical normalized peak was introduced.

## Transmission sequence and reproduced failure mechanism

There are 15 frames before the recovery experiment:

- Frame 0: initialization, full PNG, 19,756 bytes.
- Frames 1/2: Beam 2 enable, changed diff (9,446 bytes), then unchanged diff (626 bytes).
- Frames 3/4, 5/6, 7/8, 9/10, 11/12, 13/14: radius/toggle transitions, each a changed diff followed by an unchanged diff.
- Frame 13 is the last Beam 2 enable transition; frame 14 contains no new painted information relative to the sender's already advanced buffer.

| Delivery experiment at the identical final two-beam state | Incorrect pixels |
|---|---:|
| All frames, ordered | 0 |
| Omit frame 13, deliver frame 14 | 13,580 |
| Deliver stale frame 11 after the final frame | 13,580 |
| Further unchanged draw after the omitted transition | Still 13,580 |
| Deliver one forced full refresh afterward | 0 |
| Skip that recovery and deliver its following unchanged diff | Still 13,580 |

This deterministically reproduces a **persistent stale/incomplete displayed state without changing any scientific array**. The omitted-transition image retains a one-beam-looking view although the recorded request/mesh/renderer have two beams. It does not reproduce the user's exact white-band morphology spontaneously. The earlier omitted-initial-base simulation produced white regions; neither injection proves which frame Safari did not paint.

“Reordered” here models delayed/stale **painting**, not a claim that WebSocket transport reorders messages. The browser may receive every buffer yet supersede an asynchronous image decode. The installed ipympl frontend uses one Image object and replaces its source before asynchronous onload painting; the backend advances its diff base when generating a frame. The actual failing Safari sequence remains uncaptured.

## Marker, color-limit and image isolation

Three additional tests start from the actual two-beam state and separately exercise:

1. Marker/arrow highlight and recreation with unchanged canonical image.
2. A diagnostic color-limit change with unchanged image/markers.
3. A newly sampled canonical changed-radius image assigned to the mesh while retaining the prior normalization/markers (diagnostic component isolation only).

All three produce valid ordered PNG streams exactly matching their renderers; the array matches its intended component input. No component independently reproduced corruption without imposed delivery faults. The integrated callback test also confirms normal marker count and canonical dynamic limits together. These results rule out the tested deterministic artist/update path, not every Safari layout/decoding interaction.

## Full-frame recovery assessment

The supported refresh request through `canvas.manager.handle_json({'type':'refresh'})`, followed by `canvas.draw()`, sends a self-contained full PNG without writing backend private state. In the deterministic compositor it completely repairs an arbitrary stale prior view **when that frame is applied**.

A single refresh at a committed-edit boundary is not yet a reliable production fix. The test proves that if the full recovery frame is not painted, a subsequent unchanged diff cannot repair the state. Refresh itself can also enqueue a draw request. The previous investigation established that `set_image_mode('full')` alone is not a persistent full-frame policy, so simply setting it is not a valid alternative.

**Recommendation:** do not change Product to emit a one-shot recovery and claim resolution. A narrow future correction must establish final-frame delivery/paint behavior or ensure subsequent eligible frames are independently decodable, using supported APIs and bounded work. That design must first pass the new dropped/stale-frame tests as well as real Safari acceptance. No private flags, custom JavaScript, timer, backend replacement or speculative retry loop was added.

## Tests, retained artifacts and preservation

Focused deterministic regression: **4 passed**, existing ipympl 0.10.0 environment, no installation. The optional backend is required for this test; ordinary core-only environments skip it. Four pre-existing dependency deprecation warnings concern Toolbar initialization in the installed ipympl/Matplotlib combination; they were recorded, not suppressed or treated as the cause.

Log: `.codex-work/beam-stage2c-two-beam/qualified-tests.txt`.
Evidence root: `.codex-work/beam-stage2c-two-beam/qualified-output/test_two_beam_callbacks_delive0/`.

- `capture/*-request.json`: each exact draft request.
- `capture/*-arrays.npz`: canonical/mesh/limits/renderer for each state.
- `capture/events.json`, `capture/frames.json`, `capture/frame-*.png`: event order, modes, sizes and PNG hashes.
- `ordered.png`, `dropped.png`, `reordered.png`, `repaired.png`: identical-state compositor comparisons.
- `metrics.json`: mismatch counts and final request identity.

Other parametrized test folders retain the component-isolation streams. `two-beam-evidence.json` alongside this report indexes artifact SHA-256 identities, array shapes/dtypes and hashes. Array hashes are SHA-256 of C-contiguous bytes, with shape/dtype stored separately; no claim of interchangeability with other historical hash conventions.

No Product/runtime source was modified. Only the new regression-test file and report/metadata are candidate changes in this turn; disposable diagnostic evidence is under `.codex-work/`. Existing candidate source identities were rechecked; saved notebook outputs, unrelated modifications and protected evidence remain untouched. Syntax/Python 3.10 grammar and whitespace checks pass. No propagation simulations, dependency upgrades, commits or pushes.

## Smallest remaining browser evidence

A natural failure still has not been reproduced outside Safari; only explicit delivery/painting faults reproduce the mechanism. Do not ask the user to discover an exact repeatable gesture sequence. Attach the existing bounded `safari_capture.py` recorder before ordinary synthetic editing and leave it running. At the next failure, after mouse release, one capture must show **two channels in the recorded request**, canonical/mesh equality, the renderer and recent PNG/mode sequence. Take the screenshot at that same state. Then request one full refresh and record whether that same view repairs and stays repaired.

This is a single opportunistic incident, not repeated manual stress testing. If no incident is captured, keep the issue unresolved. If the renderer is correct and a full refresh repairs Safari, that confirms display recovery for that incident but still does not prove a one-shot committed refresh is universally durable. If refresh fails, obtain the first relevant frontend image/console error before proposing more rendering changes.

**NO-GO for a shipping correction; GO for further bounded delivery/recovery qualification using the deterministic regression.**

---

# Stage 2C — persistent image corruption investigation

Date: 2026-10-16. **Differential-frame dependency failure reproduced synthetically; actual Safari loss/paint sequence not yet captured. No Product correction applied.** The user's persistent white regions/bands are unresolved browser evidence, and supersede earlier readiness statements. Canonical propagation/scientific source was not changed or executed.

## What was measured

A bounded probe used the real installed ipympl canvas and current editor, a simulated normal initialization handshake, four synthetic Move positions with live entrance sampling, release, and final redraw. Every outgoing PNG was decoded and compared with the corresponding Matplotlib RGBA renderer. No optical propagation or material evolution occurred.

- All four live mesh intensities exactly equaled the canonical launch intensity for their provisional requests.
- After release, the mesh exactly equaled the canonical intensity for the committed numeric request. Final canonical contiguous-array SHA-256: `452c928300451372ce1ebeccb90317bbeac2eccdb678ee2064cc7ae6ffacdc4b` (raw array bytes only, not the scientific-results hashing convention).
- Compositing **all** transmitted full/diff PNGs in order reproduced the final renderer pixel-for-pixel.
- Omitting the first full frame, then applying every subsequent differential frame, left **60,624 differing pixels**, including persistent white areas and visible beam markers. A subsequent unchanged normal redraw did not repair the omitted background.
- A normal backend refresh request followed by draw sent a full PNG, which exactly matched the renderer and repaired the synthetic missing-base composition.
- Calling public `canvas.set_image_mode('full')` alone did **not** force the next image full: it was automatically switched to `diff` during image generation.

Evidence: `.codex-work/beam-stage2c-corruption/probe.py`, `probe.txt`, `entrance-check.npz`, `renderer.png`, `ordered-transmission.png`, `missing-base-simulation.png`, `refresh-full.png`. The simulated missing-base image was visually inspected and contains white regions with beam/marker content, but it is not a screenshot of the user's corruption. These are small diagnostic artifacts, not regenerated scientific references.

This establishes canonical correctness in the bounded reproduction, not yet in the user's currently corrupted controller. The supplied capture helper below measures that directly after mouse release. No actual browser screenshot or browser compositor buffer was available for pixel comparison during this turn.

## Full/diff protocol and overlap risk

Inspected installed `matplotlib/backends/backend_webagg_core.py:216–287`: `get_diff_image()` chooses full output for a forced refresh, changed buffer size or renderer transparency; otherwise it emits changed pixels with transparent unchanged pixels. Its reference buffer advances at generation time, not after confirmed browser paint. `handle_ack()` is a no-op for synchronization. Ordinary opaque draws therefore depend on the receiver having applied preceding frames.

Inspected bundled ipympl `src/mpl_widget.ts` in `nbextension/index.js.map:291–309, 344–396`: each binary message replaces the source of a single Image object, revokes the previous blob URL, and clears the waiting flag before `onload` draws into the offscreen canvas. The full/diff mode is read when `onload` runs. Image decoding and painting are asynchronous. Rapid direct draws, pending idle-draw replies, and model/view readiness can therefore create a dependency risk if a frame is superseded or not painted. This is a source-backed risk analysis, **not evidence that Safari dropped a particular message or decoded a particular frame out of order**.

The current editor still limits live canonical sampling to four times/second and 20 samples/gesture, and motion draw requests to 30/second. Cancellation/synchronization can also queue idle requests, while committed release draws directly. A synchronous Python loop does not prove browser image loads cannot overlap. The bounded probe intentionally delivered draws quickly but applied all frames correctly; it did not spontaneously produce incorrect PNGs or arrays. The persistent corruption was reproduced only by explicitly omitting a required base frame.

## Supported full-frame experiment and bounded recommendation

A standard WebAgg refresh request through `canvas.manager.handle_json({'type': 'refresh'})`, followed by `canvas.draw()`, forces the next generated frame full without assigning any private flag. The probe verified the resulting full image. This is a diagnostic refresh of a notebook view, not a scientific replay. It is not a persistent full-frame policy, and additional pending draw replies may subsequently generate diffs.

Do **not** ship `set_image_mode('full')` as a fix: this installed backend overrides it automatically. Do not set private `_force_full`/image buffers, add JavaScript, or assume a sleep solves transport ordering. No such change was made.

Smallest next step: capture the actual corrupted state, compare canonical array/mesh/renderer, and request one supported full refresh in the same Safari view. If the mesh and renderer are correct and full refresh repairs the live view, the next bounded correction to evaluate is a self-contained full refresh at the committed gesture/edit boundary, retaining ordinary throttled provisional updates. That would provide final-state recovery, not guarantee corruption-free intermediate frames; queued draw and real Safari acceptance tests are required before adopting it. A permanent all-full policy needs a genuinely supported backend extension strategy and bandwidth measurement, not an ineffective mode setter. A 375×375 full RGBA frame is 562,500 uncompressed bytes; actual PNG size varies. Do not multiply that cost at pointer-event rate without measurement.

## Safari capture procedure (no propagation)

Use the existing synthetic acceptance probe in its fresh local kernel. Before reproducing corruption, run:

```python
%run .codex-work/beam-stage2c-corruption/safari_capture.py
capture, full_refresh, detach = attach(probe)
```

1. Drag until corruption appears, then release. Do not press Run. Take a screenshot of the corrupted interactive view before requesting another draw.
2. Run `capture('safari-corrupted-1')`. It samples only the canonical entrance field, compares it to the displayed mesh, saves both arrays, the existing renderer and up to eight recent transmitted PNGs, and displays a **separate explicitly diagnostic static renderer image**. The extra image is intentional, not another interactive canvas. Compare this static image with the corrupted live view. The helper refuses capture while a gesture is active.
3. Run `full_refresh()`. Report whether the original interactive image repairs immediately and remains correct after movement stops.
4. Run `capture('safari-after-refresh-1')` and take an after screenshot. Use unique folder names; the helper refuses overwrite. It never writes the user's saved notebook or scientific result files.
5. Run `detach()` when finished. It restores the two instrumented send methods. Close the synthetic test kernel afterward.

The helper records bounded image payloads, modes and renderer copies (eight frames maximum), not credentials or browser tokens. It does not modify backend private state, install dependencies, run propagation or implement a production workaround. Captured per-frame renderer images permit checking what the sender generated, while screenshots establish what Safari actually displayed. Browser receipt/decode order still needs frontend evidence if a full refresh does not repair the view.

## Preservation and status

All Product candidate hashes in `validation.json` were rechecked unchanged. No source or Product test edits, dependency upgrades, scientific simulations, commits or pushes. Existing saved notebook outputs and unrelated work are untouched. Diagnostic scripts pass Python 3.10 grammar checks. The real-canvas bounded probe's assertions pass; no browser qualification or new full scientific-parity claim is made.

**Recommendation: collect the bounded Safari comparison before applying a Product fix.** We have demonstrated how a missing full base causes persistent differential-image corruption and that a full refresh repairs that simulated state. We have not yet established that this is the exact failure occurring in the user's browser.

---

# Stage 2C — initial canvas transmission correction

Date: 2026-10-16. **Correction implemented; local protocol/regression checks pass. Final Safari acceptance of this exact candidate is pending.** This supersedes the blocked initial-display investigation below; all historical evidence is retained.

## Confirmed cause and evidence boundary

The user reports Safari case A (minimal deferred canvas) visible, B (ordinary current controller) absent, and C (constructor transmissions deferred) visible. B sent two raw 562,500-byte RGBA buffers before initialization; C sent no early image buffers and later sent valid PNGs. These are user-reported browser observations, consistent with the independently retained real-kernel payload probe. Together they isolate the introduced constructor-time transmission path, rather than a missing child or intentional visibility setting.

The two premature calls were `synchronize(immediate=True)` and `_freeze_layout()` from the first `BeamPositionEditor.update()`. Both called the ipympl canvas's direct `draw()` while constructing the controller. In the tested ipympl version, that occurs before its frontend handshake has switched binary transmission into the normal PNG path. Attaching the model to a VBox is not equivalent to frontend initialization.

## Smallest correction

Only the editor draw policy changes; the controller and scientific behavior are byte-identical to the pre-fix candidate.

- The editor records its own initially-false frontend-ready flag.
- It registers an observer using the **public ipywidgets `on_msg` API** and recognizes ipympl's ordinary `initialized` message. It does not modify backend private state, synthesize messages in production, replace the backend's handler, or introduce JavaScript or timers.
- Before that notification, `_draw_committed()` uses `FigureCanvasAgg.draw(figure.canvas)` to calculate/render the layout locally without WebAgg image transmission, then requests the normal supported `draw_idle()` path. This is the local-render approach discriminated by Safari case C, without globally monkeypatching Canvas methods.
- After initialization, `_draw_committed()` calls direct `canvas.draw()` as before, preserving responsive committed-edit transmission.
- Initialization, layout freeze and genuine resize all use the guarded draw helper. Thus programmatic commits and resize events before initialization cannot reintroduce the two premature image transmissions.
- Disposal removes the observer alongside the existing interaction callbacks; an inactive editor cannot reactivate itself. No callback is reconnected to a retired canvas.

The initial axes/colorbar layout is still computed and frozen. Moving labels stay outside layout; genuine resize still recomputes and freezes readable margins. No figure/canvas/axes/colorbar replacement. Move/Tilt geometry, launch normalization, live-preview throttling, FloatText commit policy, one-click Run, request capture, provenance, completed records and exports remain unchanged. Static fallback remains available when a compatible active ipympl backend is absent.

The observer recognizes the initialization message used by the tested ipympl 0.10.0 frontend. This is protocol-specific observation through a supported public callback, not a claim of compatibility with arbitrary future frontend protocols. If that notification is absent, the editor continues to use deferred drawing instead of guessing readiness. It does not force backend state or promise instantaneous Safari painting.

## Exact files changed in this correction

1. `src/lcprop/notebook/_beam_position_editor.py` — editor-owned readiness observation, local pre-handshake layout drawing, post-handshake direct drawing, observer retirement.
2. `tests/test_interactive_lc_dragging.py` — Agg widget double explicitly receives the initialization notification before testing ready-state interaction.
3. `tests/test_interactive_lc_display_publication.py` — initialization/payload decoding regression through a real kernel and actual ipympl message dispatch.
4. This `REPORT.md` and `validation.json` — updated evidence, identities, acceptance instructions.

No `interactive_lc.py`, engine, request schema, dependency, notebook or saved output changes in this follow-up. The earlier standalone diagnostic that expected malformed frames is historical defect evidence, not the corrected acceptance test; its assertions intentionally describe the old state and should not be rerun as a success criterion.

## Validation

**92 passed, 2 skipped, 2 deselected**, 23.46 s, in the existing focused Stage 2C suite. Two optional LaunchPlane adapter cases remain skipped; two propagation cases are explicitly deselected. All geometry, live preview, redraw, resize, static fallback, request/launch parity, busy/disposed, duplicate-click, frozen-record and export checks pass. No propagation simulation or reference regeneration.

**1 real ipympl/ipykernel initialization test passed**, 17.61 s. This now verifies:

- Image payload interception starts before ordinary `InteractiveLC()` construction.
- Zero image buffers during construction and additional pre-initialization Beam 2/radius changes.
- Zero image buffers during a real dispatched pre-initialization resize.
- Normal refresh/image-mode/initialized messages are delivered through Jupyter comm messages and the actual ipympl handler, not by writing backend flags.
- The first subsequent frontend-style draw request produces a decodable 375×375 PNG.
- All later binary payloads decode as PNG; committed numeric edits and Beam 2 enable/disable transmit updated images without a further frontend draw request or pointer movement.
- Stable canvas identity and axes positions; one preview child; one initial display and no additional display publications during edits.

This is real-kernel protocol verification with a simulated frontend message sequence, not a browser rendering test. It closes the earlier test gap that counted malformed buffers as valid frames. No scientific execution is possible in this test; the execution entry point is a rejecting sentinel.

Logs: `.codex-work/beam-stage2c-init-fix/tests.txt`, `iopub.txt`, `checks.txt`. Existing local environments only. Python syntax, Python 3.10 grammar, whitespace and preservation checks pass. Index remains empty.

## Final Safari acceptance

Use a fresh kernel to avoid retaining old controller classes/canvas models. Do not clear or overwrite the user's saved notebook outputs. Open the existing single-controller synthetic harness `.codex-work/beam-stage2c/safari_acceptance.ipynb`, which imports the current development source and blocks real scientific execution by returning explicitly synthetic records. Do not use the A/B/C monkeypatch diagnostic as the final acceptance path.

1. Initialize and keep the pointer still: one interactive BeamPlane should appear without explicit draw/flush commands.
2. Enable/disable Beam 2 and commit a numerical radius change: the same canvas should repaint without cursor movement.
3. Move and Tilt both beams; confirm fixed plotting/colorbar geometry and synchronized committed numeric positions/angles. Resize once, then verify stable geometry during subsequent gestures.
4. In the synthetic harness only, click Run and repeat editing; confirm stable canvas identity, consumed-click rejection and frozen executed-parameter/result ownership. Its fields are synthetic, not scientific evidence.
5. Close the diagnostic kernel afterward. Report any absent/blank initial view, first relevant console error, or unexpected extra image. Do not install dependencies or apply private-state workarounds.

The previously supplied Safari A/B/C evidence supports the correction, but successful rendering of this final exact implementation must still be reported by the user. No claim is made that the agent independently opened Safari.

## Preservation

Pre-fix SHA-256 checks confirm unchanged controller `src/lcprop/notebook/interactive_lc.py` (including Run/isolation/export implementation), unrelated `src/lcprop/pr/published_static.py` and `tests/test_pr_published_static.py`, and saved `notebooks/lc_static_local.ipynb`. Protected evidence and prior report sections retained. No scientific-engine/schema/dependency changes, simulations, installations, commits, pushes or publication. Candidate remains unstaged.

**INITIAL CANVAS TRANSMISSION CORRECTED — READY FOR FINAL SAFARI ACCEPTANCE.**

---

# Stage 2C — missing initial BeamPlane display investigation

Date: 2026-10-16. **BLOCKED pending browser discrimination; no speculative Product correction applied.** The user reports the current canvas is absent in Safari despite an active editor, one Canvas child and explicit draw/flush attempts. Earlier backend test passes do not establish successful initial browser rendering. The readiness statements in earlier sections are superseded by this unresolved regression.

## Established evidence

The rendering refinement changed construction from deferred `draw_idle()` to synchronous `draw()` inside `update()`/`synchronize()`, followed by a second synchronous draw when freezing layout. This happens during `InteractiveLC.__init__`, after attaching the canvas to the preview-host model but before the complete controller is displayed. A Python child relationship does not prove a frontend view exists or has initialized.

In the installed ipympl 0.10.0 backend:

- `Canvas.syncing_data_url` starts True. The frontend `initialized` message switches it False.
- Before that message, `Canvas.send_binary()` replaces its PNG argument with `_last_buff` raw RGBA values, writes a bare base64 PNG to `_data_url`, then sends the raw values through the binary-image transport.
- After initialization it sends the PNG argument normally.
- The bundled frontend `MPLCanvasModel.handle_binary` constructs a Blob with MIME `image/png`. `_init_image` assigns a non-null `_data_url` directly to `image.src`; it expects a usable URL. Initial model messages include refresh, image mode and initialized; view rendering/toolbar creation are separate asynchronous steps.

Sources inspected without dependency modification: `ipympl/backend_nbagg.py:248–332`; `matplotlib/backends/backend_webagg_core.py:189–201, 360–382, 501–509`; bundled `ipympl/nbextension/index.js.map`, source `src/mpl_widget.ts` model initialization (61–100), initialization messages (130–141), binary handling (291–309), image initialization (344–402), view rendering (435–464). These sources describe the locally installed package; the actual Safari-loaded frontend version has not been independently recovered.

A bounded initial-frame probe, repeated in a **real disposable Jupyter kernel**, observed:

| Stage | Actual payload |
|---|---|
| Controller construction, before frontend initialization | Two 562,500-byte payloads; neither decodes as PNG |
| Initial `_data_url` | Base64 string without `data:image/png;base64,` prefix |
| After synthetic refresh/image-mode/initialized/draw messages | Valid 19,756-byte PNG |
| Python ownership | One preview child; 375×375 figure; active canvas |

This establishes a concrete initialization/transmission incompatibility exposed by the new early draws. It does **not** establish the complete causal chain for Safari's missing view: no real browser participated, no frontend exception was captured, and the simulated handshake recovered a valid image. In particular, the user's explicit redraw not restoring the view needs explanation beyond simply “the first image was early.” A frontend image/view error is plausible, not proven.

## Scope: ordinary controller versus diagnostic notebook

The probe constructs ordinary `InteractiveLC()` with execution blocked; it does not use the earlier synthetic-run notebook. The invalid pre-initialization transmissions therefore are not exclusive to that diagnostic. The original diagnostic helper replaces only execution and embeds `probe.widget` in the same way as ordinary usage. This does not rule out different notebook output/view state contributing to the Safari symptom.

The previous real-kernel display test counted binary transmissions and IOPub publications but did not decode the initial image payloads or execute browser model/view creation. It was inadequate for qualifying initial display. Its “frame sent” evidence must not be read as “valid initial frame rendered.”

## Tests and stop decision

Temporary diagnostic artifacts, under `.codex-work/beam-stage2c-initial-display/`:

- `probe.py`, `protocol.txt`: construction and synthetic-handshake payload inspection; all diagnostic assertions pass.
- `kernel_probe.py`, `kernel.txt`: same assertions inside a real Jupyter kernel; pass. Only that disposable kernel was started and stopped.
- `test_initial_frame_contract.py`, `regression.txt`: focused **known failing** regression requiring constructor transmissions to be valid PNGs (or no transmission). It fails on the current candidate, exposing the gap rather than suppressing it. It is kept outside the Product suite pending the supported correction.
- `safari_initial_display.py`, `safari_initial_display.ipynb`: three-case fresh-kernel browser diagnostic described below.

No propagation/material simulation, installation, dependency change, source/test-suite modification, commit or push. All current candidate file hashes match `validation.json`; saved notebook bytes and unrelated tracked modifications remain unchanged. No assertion is made that the full Stage 2C suite now qualifies browser initialization.

Because the actual missing-view failure has not been reproduced or conclusively attributed, the requested stop condition applies. No private ipympl field edits, delayed timers, JavaScript workaround or speculative initialization rewrite was added to Product. A supported deferral strategy should be chosen only after the browser diagnostic distinguishes transmission timing from model/view creation failure.

## Minimal Safari diagnostic

Open `.codex-work/beam-stage2c-initial-display/safari_initial_display.ipynb` in a **fresh** local kernel from the repository root, with the existing ipympl environment. Do not overwrite or clear the user's saved notebook. It displays three explicitly labeled areas, intentionally, with propagation blocked:

- **A — minimal deferred canvas:** ordinary embedded ipympl canvas using the earlier deferred draw pattern, no LCProp controller.
- **B — ordinary current controller:** current `InteractiveLC()` initialization unchanged.
- **C — diagnostic constructor deferral:** temporarily performs constructor drawing locally through Agg without transmitting those early buffers, then restores the real canvas drawing method and requests the normal draw. This is an isolated diagnostic manipulation, **not** an approved Product fix or a deployment instruction.

The harness records up to 100 image/initialization/resize event summaries, valid-PNG flags and model IDs. It does not record tokens or image bytes, access browser credentials, or run science. Run buttons are disabled and the execution entry point is replaced with a rejecting sentinel. Close this diagnostic kernel afterward; do not use it for scientific work.

After the first cell settles, run `readout()` and report which of A/B/C is visible, blank, or absent, without moving the cursor first. If available, include the first relevant Safari JavaScript Console error, omitting sensitive URLs/tokens. The output establishes whether initialized/refresh/draw messages arrive; a screenshot alone cannot do that.

Interpretation:

- A and C visible, B absent: strongly localizes the regression to constructor-time transmission; proceed to a supported deferral correction and promote the initial-frame regression into the Product tests.
- A visible, both B and C absent: investigate controller embedding/layout/view creation beyond early transmission.
- All absent: inspect ipympl/frontend view initialization first; do not assume an LCProp scientific problem.
- All visible: compare the failing notebook's frontend state and loaded identities before changing Product.

**Current status: initial-image protocol defect reproduced; Safari disappearance unresolved; Product unchanged pending this bounded diagnostic.**

---

# Stage 2C — Safari rendering and committed-control update follow-up

Date: 2026-10-16. **Supported backend corrections implemented; Safari paint timing remains unverified.** This section supersedes the rendering behavior and test totals in the earlier Stage 2C implementation report retained below. Scientific execution/isolation contract is unchanged.

## Findings and correction

### Committed changes and repaint

Previously, Beam 2 and numeric edits reached `_changed()`, canonical launch sampling, `BeamPositionEditor.update()`, and `synchronize()`. That path ended in `canvas.draw_idle()`. In the installed Matplotlib WebAgg backend (used by ipympl), `draw_idle()` sends a `draw` event to the frontend; it does not itself render and transmit the updated image. `draw()` renders then calls the manager's `refresh_all()`, which sends the image through ipympl's binary transport. Source inspected: installed `matplotlib/backends/backend_webagg_core.py:189–201, 501–509` and `ipympl/backend_nbagg.py:318–332` in `lcprop-interactive-test`.

Committed updates now call supported `canvas.draw()` after updating the existing mesh and beam decorations. Real-kernel tests confirm an image frame is sent for numeric edits and Beam 2 enable/disable **without a frontend draw response or pointer event**. Initialization, first layout and genuine resize can require an additional draw; ordinary committed edits perform one direct final draw. Motion retains its existing bounded `draw_idle()` scheduling/throttling.

This removes the deferred frontend draw round trip from committed updates. It does **not** prove why Safari previously waited for cursor motion: browser comm delivery, frontend render scheduling and DOM layout were not independently instrumented. No guarantee of immediate browser pixels is possible from these backend APIs; no unsupported frontend workaround was added.

### Stable image and colorbar geometry

Constrained layout previously ran on every draw while colorbar tick strings and moving annotation extents changed. That permits axes/colorbar positions to change independently of scientific data. The editor now excludes moving decorations from layout, solves the initial layout once, stores the resulting axes positions and disables further automatic layout for ordinary draws. Figure, canvas, mesh, axes and colorbar objects are retained. Dynamic intensity limits remain unchanged.

A genuine Matplotlib `resize_event` cancels any gesture, recalculates readable margins and freezes the new layout. It does not resize the figure itself or introduce a timer. Existing resize tests now dispatch the actual resize event after changing figure size, matching the backend lifecycle. During Move/Tilt/live sampling the figure size, physical x/y bounds, axes position and colorbar position stay unchanged to the test's 1e-12 absolute tolerance. Genuine window/container resize is intentionally allowed to change layout.

The initial frozen-layout implementation exposed a narrow-canvas colorbar label clipping failure at 320 px. It was corrected by the resize-event layout pass, not by weakening the label-containment assertions. Tests cover 320, 375 and 450 px figure widths. Very small browser containers, extreme tick-label lengths and browser CSS/device-pixel behavior still need visual review; no universal viewport claim is made.

### Numeric commit semantics

All scientific FloatText controls explicitly set `continuous_update=False`, retaining the previous ipywidgets default rather than enabling per-keystroke work. Kernel observers respond to accepted `value` trait changes. One committed Beam 2 radius value causes one final canonical preview update/direct draw. Enter is not the sole possible commit: leaving the field or using a numeric stepper may submit a value. The reported before-Enter update therefore does not alone establish per-keystroke recalculation. Exact Safari frontend commit timing remains a manual check.

## Exact changes in this follow-up

- `src/lcprop/notebook/_beam_position_editor.py`: direct committed redraw, layout exclusion/freeze, resize-event recalculation, retained artists/canvas.
- `src/lcprop/notebook/interactive_lc.py`: explicit FloatText commit-only policy; no Run, capture, request construction, snapshot, diagnostics or export changes.
- `tests/test_interactive_lc_redraw.py`: new committed-draw/FloatText checks and repeated Move/Tilt geometry/object-identity tests, including live sampling and changing intensity limits.
- `tests/test_interactive_lc_dragging.py`: Agg-backed widget test double exposes supported `draw()` as well as `draw_idle()`.
- `tests/test_interactive_lc_preview_lifecycle.py`: resize tests exercise the real resize-event lifecycle; containment assertions retained.
- `tests/test_interactive_lc_display_publication.py`: real ipympl image transmission assertions for committed changes, while retaining zero additional display-publication checks.
- This report and `validation.json`: follow-up evidence and current identities.

## Validation and limits

- **92 passed, 2 skipped, 2 deselected**, 21.83 s. Same focused Stage 2C suite plus `test_interactive_lc_redraw.py`. Two LaunchPlane adapter cases skipped because the optional dependency is absent. Two propagation tests deliberately deselected.
- **1 real ipympl/ipykernel test passed**, 17.88 s. One initial widget display; subsequent edits transmit binary image frames with no additional Jupyter display publications, no cursor input, stable canvas identity and one host child.
- Layout subset: **9 passed**, 5.46 s (overlapping).
- Syntax, Python 3.10 grammar and `git diff --check`: passed.
- Existing request/launch parity, geometry, lifecycle, frozen-request, failed-publication rollback and snapshot/export identity tests remain passing. No new propagation parity is claimed; retained scientific references were not regenerated.

Logs: `.codex-work/beam-stage2c-render/final-tests.txt`, `iopub.txt`, `layout.txt`, `checks.txt`. Existing environments only, no installations. No propagation/material solves, custom JavaScript, dependency changes or extra display framework. No scientific engine/schema edits. Saved `notebooks/lc_static_local.ipynb` and the two unrelated tracked modifications match pre-follow-up SHA-256 identities. Nothing staged, committed or pushed.

## Safari acceptance still required

Use the previously prepared `.codex-work/beam-stage2c/safari_acceptance.ipynb` in a fresh kernel importing this candidate; it replaces execution with synthetic results, so do not use its output as scientific evidence.

1. Enable/disable Beam 2 and keep the pointer still. Confirm the new preview appears without cursor movement.
2. Type a Beam 2 radius without committing; then Enter, blur and stepper changes separately. Confirm accepted values repaint and compare `probe.beam2_controls['w1_um'].value` with the visible number. Record which action submitted the value if timing remains surprising.
3. Repeatedly Move/Tilt both beams with live preview on/off. Observe image/colorbar geometry and titles. Confirm dynamic colors change while axes placement does not.
4. Resize the browser once; verify readable labels, then stable layout during subsequent gestures. Check no duplicate launch images.
5. Run only the synthetic harness, edit again, and verify completed parameters/plots/export ownership remains separate.

If image delivery passes but Safari still delays paint or changes canvas dimensions without an intentional resize, collect frontend resize/render evidence before further changes. The stop condition prohibits claiming an immediate-paint guarantee or adding unsupported frontend workarounds; this implementation makes only the verified backend rendering/transmission guarantee.

---

# Stage 2C — persistent canvas with one-click Run

Date: 2026-10-16. HEAD remains `9151e6bc7db8adb1a6788942e7be5bf315063fa2`. Uncommitted Stage 2A/2B work retained and extended. **Implemented and locally validated; real Safari visual acceptance remains pending.** The historical safety-gate report is retained below and is superseded by the user's approved kernel-accepted-draft contract.

## Approved contract and implementation

Run remains one click, with no Prepare/Review or confirmation step. It executes a validated detached copy of the **kernel-accepted** draft. Browser edits not yet synchronized to Python are explicitly outside the atomicity guarantee. A provisional drag image is not a committed request until release. Users can inspect the exact executed parameters afterward.

- Removed the graphical-positioning checkbox and toggle lifecycle. Controller initialization makes one optional editor attempt. A compatible, explicitly active ipympl backend creates one canvas, retained through ordinary edits, Run, Load, Reset and invalid drafts. Otherwise the existing static image/numeric controls remain available. No automatic backend switch or dependency installation.
- Active Beam, Move/Tilt, canonical launch construction, geometry, numeric release batching and bounded live-preview sampling are unchanged.
- Run/Load/Reset cancel active gestures without retiring the canvas or reconnecting callbacks. A trailing old release is ignored. A complete delayed gesture after editing resumes may update only the draft, through the same numeric controls and validation path.
- Busy handlers restore attempted scientific-control changes to captured values. Load, Reset and Run check busy/disposed state; graphical callbacks also check owner, busy/loading and disposed state. Final `close()` permanently retires editing and execution for that controller.
- Run admission serializes/decodes the complete canonical request before execution and captures the budget separately. The execution helper defensively detaches again. Both returned result-request and completed snapshot-request identities are checked. Diagnostics, slices, environment, plots and exports derive from that executed request/result, never later controls.
- Completed metadata adds canonical `request_sha256` and `wall_time_budget_seconds`; this is notebook summary metadata, not a request-schema or solver change. Completed arrays remain immutable serialized NPZ bytes. Accessors return detached decoded values.
- Completed summary shows the request SHA-256. **View executed parameters** is a collapsed disclosure containing full frozen canonical parameters; no mandatory pre-run listing or review. Export retains the complete executed request, scientific arrays, metadata, plot and file-hash manifest.
- Figure construction, PNG serialization, summary and parameter disclosure are prepared before replacing completed ownership. Publication errors roll back the prior snapshot, output and labels. A timeout, request mismatch or preparation failure preserves the previous record. The result output is assigned an already prepared PNG payload, with the existing 900 px width intent, rather than publishing a figure from a callback.

## Duplicate Run clicks: single-use button endpoint

Busy checking alone cannot suppress a queued click delivered after synchronous execution finishes. Each Run consumes its button model. After completion/failure a fresh **Run** button replaces it in the same compact slot. Old callback registration is removed, its model closed, and explicit calls carrying an old button identity are rejected. The canvas is never replaced by this operation. There is always one visible Run control and no extra user step.

This protects queued clicks directed at the old endpoint. A new explicit click on the newly displayed Run button is a new authorized run and captures a new request. Programmatic `_run()` calls are private API calls, not browser click messages. No guarantee is made that a browser has rendered a kernel value before the click; the request shown afterward is authoritative. Real-browser responsiveness and button replacement require Safari acceptance.

## Exact files changed in this Stage 2C turn

1. `src/lcprop/notebook/interactive_lc.py` — persistent initialization, request isolation/provenance, guards, one-shot Run endpoint, prepared completed presentation.
2. `src/lcprop/notebook/_beam_position_editor.py` — disposed guard; existing geometry/launch implementation retained.
3. `tests/test_interactive_lc_run_isolation.py` — new synthetic run/result/provenance/export/publication tests.
4. `tests/test_interactive_lc_dragging.py` — automatic canvas initialization, revised delayed-event semantics, static fallback and disposal tests.
5. `tests/test_interactive_lc_preview_lifecycle.py` — stable canvas/host identity rather than obsolete mode-toggle tests.
6. `tests/test_interactive_lc_display_publication.py` — real-kernel persistent canvas edits, including browser-style comm updates, without additional display publications.
7. `tests/test_interactive_lc_multibeam_presentation.py` — two fake-result tests now submit the same request as their returned synthetic result.
8. `docs/notebook_package.md` — setup, lifecycle, one-click/kernel-state contract and executed parameters.
9. This report and `validation.json` — current evidence; prior gate retained.

The existing `tests/test_lc_static_interactive_presentation.py` modification predates this turn and is preserved. Existing Stage 2A/2B untracked files remain uncommitted. No notebook source or saved notebook outputs changed.

## Validation

No propagation or material-evolution simulations ran. Tests use synthetic coordinate-coded arrays, fake execution/runner responses, and bounded canonical entrance-field sampling. No dependency installation.

| Check | Result |
|---|---|
| Focused controller/drag/tilt/lifecycle/multibeam/presentation suite | **89 passed, 2 skipped, 2 deselected**, 23.71 s |
| Skips | Two desktop adapter checks require unavailable `launchplane.model`; not claimed passed |
| Deselections | Two `bounded_execution_exact_canonical_arrays` propagation tests deliberately excluded |
| Real ipympl/ipykernel IOPub test | **1 passed**, 16.38 s; one initial display, no additional display publications during programmatic and browser-style numeric edits; canvas identity stable |
| Isolation suite, separately repeated after publication rollback coverage | **9 passed**, 4.49 s; overlaps the 89 |
| Syntax and Python 3.10 grammar | Passed |
| Whitespace / index | `git diff --check` passed; index empty |

Focused suite: `tests/test_interactive_lc_run_isolation.py`, `test_interactive_lc_dragging.py`, `test_interactive_lc_move_tilt.py`, `test_interactive_lc_preview_lifecycle.py`, `test_interactive_lc_multibeam_presentation.py`, `test_lc_static_interactive_presentation.py`, `test_interactive_lc_multibeam.py`, with `-k 'not bounded_execution_exact_canonical_arrays'`.

Logs: `.codex-work/beam-stage2c/final-tests.txt`, `iopub.txt`, `isolation-final.txt`, `checks.txt`. Main tests used the existing `.codex-work/pypi-stage2a/colab-env` Python environment. Real IOPub test used the existing `lcprop-interactive-test` Python/ipympl kernel, with pytest borrowed through a process-local trailing sys.path; no environment modification. Only its disposable test kernel was started/stopped.

Coverage includes captured request detachment, busy mutations, fake progress metadata, result/request mismatch rejection, delayed draft gestures, new-run capture, stale/duplicate button clicks, post-disposal events, export and same-environment PNG identities, invalid-draft recovery, timeout/plot/PNG/publication failures, optional-import isolation, coherent/incoherent launch parity and forward/inverse Move/Tilt parity. Tests mutate decoded arrays and source fixture arrays after serialization to verify completed bytes remain unchanged.

Two intermediate failures were test-fixture incompatibilities exposed by the new identity assertion: their fake executor returned a coherent two-beam request while the form submitted the single-beam preset. The fixtures now load their synthetic request before Run; no assertions were weakened. A dedicated mismatch-rejection test confirms the guard. No unresolved focused failures. The historical saved-notebook-output baseline failure was not rerun or “fixed”; saved user outputs were preserved.

### Scientific parity scope

Exact canonical request/launch comparisons pass in the focused tests. Completed synthetic request/NPZ/diagnostic/export/figure identity survives later editing. Existing retained full propagation parity evidence remains applicable to the unchanged engine and canonical scientific construction, but **no new propagation parity claim is made**. New request digest/budget summary fields are intentional provenance additions; scientific arrays and request schemas are not changed.

## Memory and display limitations

One interactive canvas, no per-edit canvas models or callback reconnection. Static fallback reuses its image and Agg figure. Live sampling retains existing throttling/caps; no optical volumes or frame history are added. Request detachment copies bounded serialized configuration, not simulation arrays. Prepublication rendering temporarily holds a figure/PNG alongside the previous record; completed output storage remains bounded to the current record. Run button models are closed after use, not retained in a history.

Kernel state and rollback are tested. Frontend trait rendering is asynchronous; this is not an atomic browser DOM transaction or proof that Safari releases every obsolete view. One persistent launch model avoids the previously observed replacement pattern, but actual Safari acceptance remains necessary. Static mode requires creating a fresh controller/kernel with a non-widget backend; there is no runtime toggle. A compatible active backend is required, not merely an installed ipympl package.

## Local Safari acceptance without science

Prepared disposable harness: `.codex-work/beam-stage2c/safari_acceptance.ipynb`, backed by `safari_probe.py`. Open it in a **fresh** local Jupyter kernel from the repository root with the existing ipympl environment. It activates `%matplotlib widget`, imports development source, replaces execution with a synthetic record factory, and displays `probe`. It does not install packages. Do not mix it with scientific sessions or preserve its fake outputs as numerical evidence.

1. Confirm the printed module path is this development candidate and interactive canvas is available. There is no graphical checkbox; one launch preview is shown.
2. Move and Tilt Beam 1/2; test overlapping centers, numeric edits, invalid bounds, reset and selection. Confirm marker/control synchronization and no extra launch images.
3. Click Run rapidly, including double clicks. All results are explicitly synthetic. A consumed button's queued clicks must not add a second fake run. Each deliberately clicked newly displayed Run button may add one.
4. Confirm stable canvas identity and one host child in the readout. Inspect **View executed parameters**; change draft controls and confirm completed parameters, labels and plots stay unchanged. Click the new Run once and verify it captures the new draft.
5. Optional load test: save a bounded draft with canonical persistence to a disposable path and load it; confirm the same canvas remains. Do not use or overwrite protected experiment files.
6. Close the test kernel afterward. For fallback, use a separate fresh kernel without widget backend activation; numeric/static operation must remain usable.

No Safari success is claimed until user observations are supplied. The synthetic status may use normal “converged” interface wording; its arrays/metadata are deliberately fabricated for presentation testing, not convergence evidence.

## Preservation

SHA-256 checks confirm unrelated `src/lcprop/pr/published_static.py`, `tests/test_pr_published_static.py`, and saved `notebooks/lc_static_local.ipynb` bytes match the pre-turn inventory. No scientific-engine, request-schema, dependency or version edits. No simulations, installations, commits, pushes or publication. Earlier reports/evidence remain preserved; changes are unstaged.

**STAGE 2C IMPLEMENTED — READY FOR LOCAL SAFARI ACCEPTANCE under the approved kernel-accepted-draft contract.**

---

# Historical safety gate — superseded contract, retained evidence

# InteractiveLC Stage 2C — single-canvas safety gate

Date: 2026-10-16. Committed HEAD: `9151e6bc7db8adb1a6788942e7be5bf315063fa2`. Existing uncommitted Stage 2A/2B candidate retained.

**BLOCKED before implementation.** Removing the mode toggle does not establish event freshness across Run/load/reset while the same interaction endpoint remains reusable. No Product source, tests, notebook, dependencies or previous evidence were changed. This report and temporary safety-probe artifacts are the only additions.

## Design evaluated

The proposed one-canvas interface is otherwise feasible: lazy optional ipympl initialization, static fallback, persistent widget identity, Active Beam and Move/Tilt, current canonical launch/geometry, throttled provisional previews and snapshot ownership can all remain. Avoiding repeated model destruction addresses the user-confirmed browser retention mechanism without requiring new physics.

The safety question is separate: can the same canvas continue accepting fresh edits after a Run/load/reset boundary while rejecting every delayed pre-boundary gesture? No demonstrated event-freshness contract currently supports that distinction. Stage 2C explicitly requires establishing it before implementation.

## Existing source boundaries

- `interactive_lc.InteractiveLC._run` currently retires the canvas before execution, then restores ordinary controls when the synchronous call completes.
- `load` and `_set_request_controls` retire it before loading/resetting scientific draft state.
- `_beam_position_editor.BeamPositionEditor.ready` checks active ownership, busy/loading and navigation state. These are current Python states, not the generation/creation time of a browser event.
- `press` captures the request at **delivery** time and begins a new gesture. Cancelling the prior gesture protects against a trailing old motion/release, but cannot identify a complete delayed press/release received after the boundary.
- `close` permanently disconnects callbacks and closes the canvas; explicit controller disposal remains safe. Retaining the model and permanently retiring its edit callbacks is also safe, but it would disable mouse editing after the first Run/load/reset until a new controller is created. That restriction was not silently substituted for the requested usable single-canvas design.

The earlier Stage 2B investigation tested callback disconnect/reconnect. This investigation additionally tested **unchanged handlers and the same canvas throughout**, as requested by Stage 2C. Thus the blocker is not merely reconnection to a retired callback object.

## Bounded real-ipympl experiment

Temporary script: `.codex-work/beam-stage2c-safety/probe.py`.
Output: `.codex-work/beam-stage2c-safety/probe.txt`.

Used the already installed ipympl 0.10.0 and Matplotlib 3.11.2 in `lcprop-interactive-test`, without installation, browser access, kernel/server access, LCProp execution or scientific arrays. A private Figure/Canvas/FigureManager and persistent handlers model the current busy/gesture guards. Browser-style messages pass through the actual `FigureManager.handle_json` and Matplotlib event registry. No callbacks are reconnected between boundaries.

| Synthetic boundary | Events during guard | Old release alone after guard | Complete old sequence delivered after guard |
|---|---|---|---|
| Run completion | Rejected | Rejected | Accepted as a new gesture |
| Load completion | Rejected | Rejected | Accepted as a new gesture |
| Reset completion | Rejected | Rejected | Accepted as a new gesture |
| Permanent disposal | All subsequent events ignored | All subsequent events ignored | All subsequent events ignored |

All probe assertions passed. The accepted count is an inert Python counter, not a scientific control mutation. This establishes an ambiguity if such a delivery order occurs; it does **not** establish that Safari actually generated that ordering, nor that ordinary observed edits are wrong. Transport/frontend ordering guarantees sufficient to rule it out have not been established. Passing normal toggle/child-count tests cannot prove that stronger guarantee.

## Precise unresolved issue

The existing ipympl event interface does not provide a guaranteed interaction-generation token linking a pointer event to the scientific draft state before/after Run/load/reset. The inspected frontend's GUI-event serialization does not guarantee a timestamp field. Python callback epochs, resetting the gesture, a timeout, or requiring a press all operate after message delivery and cannot alone prove that press is fresh. Assuming FIFO behavior across every relevant frontend/kernel queue without qualification would weaken the accepted stale-event contract.

A safe permanent controller disposal is demonstrable; continued editing through a retained endpoint across state transitions is not yet demonstrable under the stated guarantee. Preserving snapshots is necessary but does not fix stale edits to the current draft.

## Stop decision and smallest next decision

No checkbox removal or lifecycle rewrite was performed. Existing Move/Tilt behavior, canonical request/launch parity, optional static operation and historical validation remain unchanged. There is no new numerical qualification claim and no new propagation test.

To proceed, one of the following requires an explicit design decision and validation:

1. Establish an event-freshness/ordering contract supported by the actual frontend and kernel, or introduce a generation/acknowledgment protocol. A custom frontend protocol is outside this task's no-custom-JavaScript constraint.
2. Accept a persistent **display** canvas whose editing callbacks are permanently disabled after the first Run/load/reset; numeric editing continues, and a fresh controller is needed for graphical editing again. This preserves retirement safety but changes the desired usability.
3. Explicitly redefine the guarantee to cancellation of already-started gestures and rejection while busy, rather than rejection of every delayed complete gesture. This was not assumed or implemented.

## Safari acceptance instructions

There is no Stage 2C candidate to accept in Safari. Do not expect the current source to have lost the graphical toggle or acquired persistent-canvas reuse. After a safe design is approved and implemented, acceptance must separately check one visible view, stable identity during ordinary edits, state-transition event safety, Move/Tilt numeric parity and snapshot retention. The previous publication probe remains available for collecting browser evidence; no saved notebook output should be cleared by the implementation audit.

## Preservation

Recorded Stage 2B candidate file hashes were rechecked unchanged. No scientific-engine, request-schema, dependency, notebook-output or unrelated tracked changes were made. No simulations, installations, commits, pushes or publication. Existing dirty worktree and protected evidence remain intact.

**Final status: STAGE 2C BLOCKED — event freshness across Run/load/reset on a retained interactive canvas is not established. Current implementation retained.**
