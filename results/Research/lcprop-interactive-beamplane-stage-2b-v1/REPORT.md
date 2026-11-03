# InteractiveLC Stage 2B — Move/Tilt and live launch preview

**Latest status: browser view retention is user-confirmed. Persistent-canvas reuse is blocked at the stale-event safety gate; see the final section. Product source remains unchanged in that investigation. Earlier tests do not establish browser resolution.**

Date: 2026-10-15. Committed base `9151e6bc7db8adb1a6788942e7be5bf315063fa2`, extended from the uncommitted Stage 2A candidate. **Implemented and locally validated: 77 passed, two propagation tests deliberately deselected. Ready for Safari/Jupyter visual testing, not browser-qualified.** No propagation/material simulations, installations, dependency changes, commits, pushes or publication.

## Authoritative geometry recovered

- LaunchPlane `canvas.py:97`, `set_resolved_exit`, receives a host-owned physical endpoint. `tilt_tip_offset` returns its physical displacement in µm; it is not an arbitrary pixel-length direction arrow. The Qt canvas negates physical y for scene display.
- LaunchPlane `canvas.py:251`, handle `_move_to`, emits `exitRequested(index, x, -scene_y)` for arrow drags. `launchpane.py:436–461` delegates inverse resolution to the host and changes only external incidence/azimuth; unreachable endpoints are rejected.
- LCProp `src/lcprop/gui/panels/beam_panel.py:59`, `_direction_for_exit`, calls `external_direction_for_exit`. Its preview at lines 329–332 places the endpoint at `(x0 + L*kx/kz_internal, y0 + L*ky/kz_internal)`.
- `src/lcprop/optics/physical_launch.py:65`, `resolve_beam_geometry`, computes the external transverse wavevector and the forward internal refracted wavevector. The unchanged canonical equations are `kx = (2π n_ext / λ) sin(theta_ext) cos(phi)` and analogously `ky` with sin(phi); the internal forward kz follows the material reference index. Transverse phase gradients are kx/ky in rad/µm; these are not angles or arrow lengths.
- `external_direction_for_exit` at line 92 owns the inverse. It chooses the equivalent signed-theta branch whose azimuth is nearest the current azimuth; a tie preserves theta sign, and zero displacement preserves phi and signed zero. It rejects unreachable/grazing incidence rather than clamping.

Thus the mapping is authoritative and unambiguous under the existing branch-selection convention. Notebook arrows use the exact same internal-ray endpoint formula and canonical resolver/inverse functions. They use `request.material.no`, the same reference index already used by this controller's canonical launch context, and `request.grid.z_length_um`. No Qt scene-y negation is applied to Matplotlib physical data coordinates. No LaunchPlane import or duplicate optical model is needed.

Tilt changes external incidence/azimuth, not internal angles directly. Radii and roll stay fixed, although the canonical projected launch profile can legitimately change with incidence. These are straight reference-ray geometry arrows, not predicted nonlinear beam trajectories. Off-view endpoints are not rescaled or clamped; use numeric controls to bring them into view. At zero incidence the endpoint sits on the center and remains selectable in Tilt mode.

## Exact candidate changes

1. `src/lcprop/notebook/interactive_lc.py`: replace Move beam with independent **Active Beam** and **Action** selectors; add **Live draft launch (throttled)**; observe interaction-state changes and include new controls in Run disabling. Request construction, scientific execution, snapshots and exports unchanged from Stage 2A.
2. `src/lcprop/notebook/_beam_position_editor.py`: canonical ray endpoint/inverse adapters, active-arrow highlighting, action-aware gestures and atomic numeric commits, bounded live preview, restoration of committed draft image on cancellation.
3. `tests/test_interactive_lc_dragging.py`: update selector references; explicitly disable live previews in the fixture to retain all marker-only Stage 2A assertions.
4. `tests/test_interactive_lc_move_tilt.py` (new): 16 geometry/live/tilt checks with no propagation.
5. `docs/notebook_package.md`: updated interaction semantics and local test instructions.
6. This report and `validation.json`.

Stage 2A reports remain untouched as historical candidate evidence; their hashes describe the earlier candidate, superseded here for the listed files. Existing local notebook outputs, unrelated work and protected evidence are not included or edited. All changes remain unstaged/uncommitted.

## Interaction and scientific ownership

Active Beam selects Beam 1 or enabled Beam 2 independently of Move/Tilt. Both internal-ray arrows are shown; the active arrow is heavier/opaque. Left-button dragging starts only near the selected center (Move) or endpoint (Tilt), preserving the grab offset. Overlap never selects an intensity maximum or implicitly changes beam identity.

Motion updates existing artists. A gesture captures its starting immutable request representation for proposed changes; controls remain unchanged until release. Tilt uses the canonical inverse and commits only theta_ext_rad/phi_rad. Move commits only x/y. One final `_changed` call performs the canonical launch preview and sampling advisory refresh. No second final validation preview or optical replay is added. Impossible tilt endpoints leave controls untouched with a rejection message.

During live preview, a detached request carries proposed geometry into the unchanged `launch_preview`. Nothing is written to `base`, scientific controls, completed snapshots or retained result arrays. Live plots are explicitly provisional and identify the sampled target coordinates to distinguish them from newer marker positions. Relative phase and coherence-group handling stay canonical. The existing Run button remains the sole propagation entry point.

Run/load/reset/disposal retire the old canvas and its callbacks. Action/selection/live-toggle/numeric edits cancel any gesture; cancel restores the committed draft image, not the last provisional frame. Reusing the same controller retains ownership. Explicit controller replacement still requires `demo.close()` first: Python rebinding cannot automatically disable arbitrary old visible widget instances. Core/static use and missing-ipympl fallback remain available.

## Bounded live work and measured smoke evidence

Live mode is enabled by default for the optional editor but can be unchecked. Limits are presentation settings only:

- no live launch until 250 ms after press, then at most one sample per 250 ms after the preceding sample completes;
- at most 20 live samples per gesture, plus one unconditional final release refresh;
- a sample taking over 100 ms switches to marker-only for the remainder of that gesture;
- ValueError/TypeError from proposed geometry or launch sampling disables live updates for that gesture with an explanatory message;
- no timers, background tasks, pending sample queues, frame histories or retained volumes;
- marker redraw requests are limited to 30 per second during motion; live samples update the existing mesh/colorbar rather than replacing the canvas.

Only one committed draft intensity and the current mesh/provisional data are retained, plus existing transient canonical launch allocations. The introductory 128×128 float64 intensity is 128 KiB; channel launch construction retains its existing bounded NumPy allocation behavior. ipympl owns image/differential transport. The application does not resend raw scientific volumes or construct a new PNG/widget per motion. Live changes can still require image-diff transmission; frontend/network message queue behavior is outside this Python-side cap and is not claimed bounded by a browser test.

Real ipympl 0.10.0 smoke check used the already extracted temporary wheel from Stage 2A, with no installation. Two gestures, Move and Tilt, each made three provisional canonical samples and one final refresh (eight launch calls total). Individual launch construction took **1.80–2.15 ms** locally in that check. This measures only launch sampling, not figure rendering, network latency or Safari responsiveness; it is not a performance guarantee. A synthetic clock controlled scheduling, while wall-time measurement recorded each real launch call. No solver ran.

Evidence: `.codex-work/beam-drag-stage2b/canvas_smoke.py`, `canvas-smoke.txt`. Actual ipympl MouseEvents traveled through its callback registry, controls stayed unchanged during motion, final controls changed appropriately, disposal succeeded and the automatic-show queue remained empty. This is headless canvas evidence, not hosted or browser qualification.

## Validation

Existing disposable Python environment: Python 3.12.13, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2, ipywidgets 8.1.9. No environment changes.

Final focused command:

```sh
PYTHONPATH=src:/Users/mcroning/LaunchPane/src MPLBACKEND=Agg \
MPLCONFIGDIR=.codex-work/beam-drag-stage2a/mpl \
.codex-work/pypi-stage2a/colab-env/bin/python -B -m pytest -q -p no:cacheprovider \
 tests/test_interactive_lc_move_tilt.py tests/test_interactive_lc_dragging.py \
 tests/test_interactive_lc_multibeam_presentation.py tests/test_notebook_lazy_interface.py \
 tests/test_lc_static_interactive_presentation.py tests/test_interactive_lc_multibeam.py \
 -k 'not bounded_execution' --basetemp=.codex-work/beam-drag-stage2b/validated
```

**77 passed, 2 deselected in 13.61 seconds.** Log: `.codex-work/beam-drag-stage2b/validated.txt`. No unexpected failures. This is the Stage 2A 61-check selection plus 16 new cases, with marker-only fallback coverage retained explicitly. The two excluded tests perform scientific propagation; retained Stage 1 scientific parity evidence was not regenerated. Historical unrelated baseline failures were not rerun or concealed.

New tests cover signed/zero/negative-incidence forward/inverse round trips; meridian zero crossing; active-arrow emphasis; independent selectors; coherent/incoherent exact request and canonical complex-field/intensity parity for graphical/numeric tilt; preservation of all other beam properties; unreachable inverse and release rejection; live scheduling cap (100 immediate motion events cause no launch); final refresh count; slow fallback/cancel/recovery; rejected live samples; action-change/stale-canvas cancellation; and snapshot-bound export after live tilt. Existing tests cover Move parity, overlap, physical coordinates, bounds, snapshot preservation, static fallback and optional-import isolation.

Syntax/Python 3.10 grammar and whitespace checks passed. Top-level scientific/request/execution/export functions and `InteractiveLC.request` remain unchanged from the Stage 2A input. Full numerical propagation parity is retained evidence, not a new claim from mouse-event tests.

## Local Safari/Jupyter test instructions and limitations

Use the candidate in the disposable local kernel, not unchanged production 0.3.0. The optional dependency/setup procedure remains Stage 2A's: ipympl 0.10.0 in the kernel environment with matching frontend; in a fresh kernel run `%matplotlib widget` before constructing the controller. Do not install/reload around live results. Export snapshots before a restart. No notebook/dependency installation was performed here.

1. Run the reusable initialization cell; enable graphical positioning. Confirm Active Beam and Action are independent, with Move initially selected.
2. Drag Beam 1's center and watch the provisional launch update at a limited rate. Numeric x/y must remain unchanged until release. Disable Live draft launch and verify marker-only behavior.
3. Select Tilt. At normal incidence grab the square endpoint at the center, drag a small distance, and release. Verify numeric external incidence/azimuth change while center/radii/power/phase/groups stay unchanged. The image may change because the canonical phase gradient and projection changed.
4. Enable Beam 2 and repeat both actions; use explicit Active Beam for coincident centers. Confirm active-arrow emphasis and group-aware combined intensity. Do not press Run to test interaction.
5. Test Escape/canvas leave/out-of-aperture cancellation, unreachable direction rejection, action switch during drag, live off/on and static fallback. The image must return to committed draft data after cancellation.
6. Rerun the same initialization cell and confirm one reused controller. Retire it explicitly before intentional replacement. Completed data/exports must remain unchanged throughout.

Actual Safari performance, high-DPI picking, frontend loss/latency and hosted Colab interaction remain unqualified. Browser input queues are not controlled by these Python callbacks. Endpoints outside the launch axes require numeric editing. This milestone adds no beam resizing, roll handles, channel count, GPU/Slurm support, propagation or scientific model.

**Conclusion: Move/Tilt and bounded live launch previews are implemented and ready for local visual acceptance. Scientific behavior remains canonical; publication and hosted qualification are separate decisions.**

## Launch-preview display lifecycle remediation (2026-10-15)

### User observation and root-cause scope

The user reports that the active preview correctly followed numeric controls; the apparent synchronization failure arose from retained earlier images. Repeated graphical checkbox changes exposed multiple launch images in local Safari/Jupyter. This is user-reported browser evidence, not an independently captured frontend trace.

The inspected implementation had two separate presentation owners: static PNGs were published with `display(image)` inside an `Output` context on every `_changed`, while interactive canvases were children of another VBox. It cleared/hid the former and replaced the latter. That design relies on callback output capture and kernel parent-message routing; any static publication outside the intended Output cannot be removed by clearing that Output. Hiding it also leaves retained image/widget state. The code-level defect is this split ownership plus repeated publication. The exact Safari/kernel message sequence causing the supplied observation was not independently recorded, so this report does not assert a Safari or widget-manager bug.

### Correction

The launch area now has one stable `preview_host` VBox. Its single child is either the controller's reusable `ipywidgets.Image` or the active ipympl canvas. Switching modes replaces `children`; it never publishes a new launch display output. Static rendering writes PNG bytes into the same Image widget. It uses a persistent private FigureCanvasAgg figure/mesh, independent of the global ipympl backend, with no pyplot registration. Numeric static edits reuse that figure and image.

Interactive edits retain the same canvas and artists as before. `_changed` no longer clears launch Output widgets or recreates canvases. On disabling/retiring the editor, the committed draft intensity already held by it is rendered to the reusable static image; no extra canonical sampling is needed for that switch. This avoids showing a stale pre-interactive static image. Invalid drafts remove the preview rather than leaving an apparently current old plot. Completed result plotting/display/export are untouched.

Disabled canvases are closed, callbacks disconnected, and their ownership checks continue rejecting retained events. Move/Tilt, live-preview throttling and atomic final numeric updates are unchanged. Explicit `close()` is still required before deliberate controller replacement; reuse remains the notebook initialization policy.

This fixes future displays within the current controller. It does not erase saved user outputs or historical images that were already published into other notebook cells. Load the corrected source in a fresh kernel after exporting any in-memory results, then rerun the existing initialization cell. Do not rewrite/clear saved notebook evidence as part of source validation.

### Layout finding and correction

A 375×375 px Agg reproduction measured the previous single-line title extending about **5.98 px left of the figure**. Colorbar overlap was not independently reproduced, but layout bounds were checked. The interactive title is now two lines, “Draft launch / Internal rays to exit,” with explicit compact title/tick/axis-label fonts and colorbar padding. Numerical data, physical aspect ratio, angular geometry, normalization and color limits are unchanged.

At 320, 375 and 450 px figure widths, tests confirm title/axis/colorbar labels fit inside the figure and image/colorbar axes remain separated. The rendered 375 px real-ipympl figure was visually inspected: full title, axes and colorbar are readable with no overlap. Evidence `.codex-work/beam-preview-lifecycle/interactive.png` is a launch-only rendering, not a propagation result or browser screenshot. Safari responsive/high-DPI rendering still needs user confirmation.

### Focused tests and retained behavior

Added `tests/test_interactive_lc_preview_lifecycle.py` (six cases): repeated toggles, exactly one preview child, no display publication, static image/figure reuse, persistent active canvas during numeric edits, stale-event rejection, bounds at three figure sizes, and snapshot/export byte preservation through Move/Tilt and mode changes.

Updated directly affected assertions in `tests/test_interactive_lc_dragging.py` to inspect the single preview owner and static renderer. Updated `test_compact_widgets_and_preview` in `tests/test_lc_static_interactive_presentation.py`: its old expectation of an Image in captured `display()` output intentionally no longer applies. The test now asserts **no display output**, exactly one Image child, and retains the original 375 px, maximum-width and square-raster assertions. No assertion was removed to hide a scientific or compatibility failure.

The real ipympl 0.10.0 smoke check performed three enable/disable cycles and Move/Tilt gestures through actual Matplotlib MouseEvent callbacks. It forbade `IPython.display.display`, verified one preview child, static reuse, retired canvases, and an empty automatic-show queue. It passed without browser access or installation. Script/log: `.codex-work/beam-preview-lifecycle/real_canvas.py`, `real-canvas.txt`.

The new lifecycle plus existing Move/Tilt/dragging selection passed **37 tests**. **Final affected selection: 83 passed, 2 propagation tests deselected in 19.04 seconds.** It includes multibeam presentation, import isolation, compact presentation and canonical launch tests; see the final results in `validation.json` and `.codex-work/beam-preview-lifecycle/accepted.txt`. Propagation comparisons remain deliberately excluded. No new scientific simulation or numerical qualification is claimed.

### Exact additional scope and preservation

Modified `src/lcprop/notebook/interactive_lc.py`, `src/lcprop/notebook/_beam_position_editor.py`, `tests/test_interactive_lc_dragging.py`, `tests/test_lc_static_interactive_presentation.py`; added `tests/test_interactive_lc_preview_lifecycle.py`; updated this report and `validation.json`. The earlier Stage 2A/2B candidate remains incorporated. All other recorded starting files, including unrelated PR changes and `notebooks/lc_static_local.ipynb`, remain byte-identical. Scientific functions, request construction, canonical launch, snapshots, export serialization and dependencies remain unchanged. Syntax/Python 3.10 grammar/whitespace checks pass. No commit, push or installation.

**Local acceptance remaining:** after loading corrected source, toggle graphical mode repeatedly and edit numeric coordinates; confirm one current launch image, persistent active canvas, readable title/colorbar, working Move/Tilt, and unchanged completed results. Do not interpret old saved images from another cell/kernel as the current preview.

## Follow-up: duplicate display still reported — unresolved (2026-10-15)

**Current status overrides the earlier resolution implication: Safari duplication is not yet reproduced or resolved.** The user confirms development-source imports and controller class identity, but still sees additional launch images on repeated toggles. No speculative Product correction was applied during this follow-up. The previous 83 passes did not establish browser display-output behavior.

### Actual hierarchy and publication inventory

The current Configure VBox contains `preview_host` exactly once. The host contains either the reusable `_preview_image` or the current editor's canvas. `_close_editor` retires/disconnects that canvas and restores the static Image. `_render_static_preview` sets Image bytes and host children directly; it calls neither display nor clear_output.

Inspected all figure and output sites:

| Location | Behavior / relationship to toggling |
|---|---|
| `interactive_lc.display_figure` | Calls `IPython.display.display(image)`; now used for completed results, not launch preview |
| `InteractiveLC._run` | Clears `result_output`, displays completed figure, closes it; no toggle calls this path |
| `plot_completed` | Creates pyplot figure within `plt.ioff()`; only completed plotting/export |
| `_render_static_preview` | Persistent direct Figure + FigureCanvasAgg; updates one Image; no pyplot registry or display publication |
| `_beam_position_editor.widget_canvas` | Direct Figure, ipympl Canvas and FigureManager; embeds canvas in preview_host; no pyplot factory/show call |
| Notebook initialization | Explicit `display(controller.widget)`; repeated/copied cell outputs can create additional views independent of the controller |
| ipympl backend | Cached `display` aliases in show/export methods; `post_execute` flush can display queued pyplot figures |

The editor factory does not use `plt.ioff()`, but it also does not use pyplot's manager factory, register its figure in Gcf, or add it to ipympl's automatic-show queue. Actual `%matplotlib widget` interactive-mode testing below found no automatic publication. Therefore adding ioff speculatively would not establish a root cause. Completed pyplot creation is already scoped correctly.

### Why earlier tests were insufficient

Earlier tests asserted a single child and/or patched `IPython.display.display` after ipympl import. That does not intercept ipympl's previously imported display alias. Their standalone interpreter backend switch also did not reproduce a real IPython magic/post-execute lifecycle. These were genuine coverage gaps; a one-child widget assertion alone cannot rule out separately published images or frontend view duplication.

### Real-kernel publication regression

Added `tests/test_interactive_lc_display_publication.py`. It starts and shuts down only a disposable kernel, invokes actual `%matplotlib widget`, and captures Jupyter IOPub messages rather than monkeypatching display calls. It counts `display_data`, `update_display_data`, `execute_result` and `clear_output`, including messages whose parent is a widget comm rather than the current execute request.

Coverage includes three programmatic on/off cycles and three browser-style `comm_msg` on/off cycles, verifies checkbox and active-editor state, and explicitly invokes post-execute hooks. The initial controller produces one widget display publication. All twelve toggle transitions produce zero additional display/clear outputs. No solver is permitted. This establishes Python/kernel publication behavior in this test, **not Safari DOM behavior**.

Environment: existing `lcprop-interactive-test`, Matplotlib 3.11.2, ipympl 0.10.0, IPython 9.17.1, ipykernel 7.4.0, jupyter_client 8.10.0. Pytest was borrowed via a process-local trailing sys.path entry from the existing disposable test environment; nothing was installed or changed in either environment. The test requires ipympl/ipykernel and reports a skip when absent; it was actually executed, not skipped, here. Temporary local kernel sockets required sandbox approval; no existing Jupyter server/kernel was accessed or terminated. Connection files/tokens were not displayed.

Evidence: `.codex-work/beam-preview-publisher/probe.txt` and `iopub-test-final.txt`. The initial probe and final regression are distinct; final status is recorded in validation metadata. No simulations or dependency changes.

### Minimal Safari diagnostic — next required evidence

Prepared `.codex-work/beam-preview-publisher/safari_preview_probe.ipynb` without modifying any user notebook. It uses a separate diagnostic controller with Run disabled, installs no packages, and reuses that controller when its same setup cell is rerun. In a fresh kernel after exporting existing results:

1. Open the probe in Safari and run setup once. One controller should appear.
2. Click graphical positioning on/off three times. If images accumulate, retain a screenshot showing the extra images and their cell boundaries.
3. Run the readout cell. It reports capped publication records with MIME keys and Python call stacks, toggle transitions, child model IDs, pyplot manager/automatic queue counts, dependency versions, and loaded-code versus on-disk fingerprints for the controller and editor factory. It records no kernel authentication data.
4. Share the readout and screenshot. Optional cleanup removes its recorder and retires only its own canvas.

The recorder wraps the actual IPython display publisher, so cached backend display aliases still reach it. Zero publication records with visible duplicates suggests frontend view retention or old cell output; nonzero records provide source call stacks. Class identity alone does not prove the helper's loaded function bytecode matches disk. The fingerprint comparison tests that possibility without asserting stale code as the cause. Saved outputs from earlier kernels remain another possibility, not an established diagnosis.

### Preservation and stop condition

No controller/editor/scientific/dependency/notebook source was changed in this follow-up. Only the new publication test, this report and validation metadata were added/updated; the temporary diagnostic is under `.codex-work`. Existing candidate identities remain unchanged. Saved local notebook outputs and unrelated work remain untouched. No commit or push.

**Stop condition reached:** current Python-side display accumulation was not reproduced even using real kernel widget messages. A frontend/browser observation is necessary to distinguish remaining causes. The issue remains open; no claim of resolution or speculative scientific/presentation redesign is made.

## Safari persistent-canvas safety gate — blocked (2026-10-16)

### Confirmed browser evidence

The user now confirms duplicate graphical views despite matching loaded source identities, one Python preview-host child, no extra Jupyter display publications, and no pyplot managers/automatic-show entries. This supersedes stale-source/publication hypotheses for the reported session. The observed problem is retained browser views after mode changes. The precise Safari/frontend view-removal mechanism has not been independently inspected.

### Persistent reuse feasibility

One ipympl canvas can be kept in a stable widget hierarchy and hidden/shown through layout traits. Static and graphical children could remain mounted with mutually exclusive visibility. Numeric/canonical preview updates, live throttling, completed ownership and explicit Run do not intrinsically require canvas replacement. Explicit controller disposal can still close it permanently.

However, this changes the stale-event boundary. The current implementation closes the old canvas/comm and rejects its callbacks after ownership changes. Reusing the same canvas/model/comm after off/on means messages generated before hiding can reach newly connected callbacks after showing. Disconnecting callbacks while hidden protects the off interval; a Python callback generation counter protects old callback objects. Neither attaches a generation to browser messages that have not yet reached the callback registry.

### Reproduced safety conflict

An isolated real-ipympl 0.10.0 probe (no LCProp controller, no solver) retained one canvas, disconnected all edit callbacks while hidden, and reconnected new-generation callbacks on re-enable. It then delivered a previously constructed press/motion/release sequence through the real FigureManager.handle_json path:

- canvas identity stayed fixed;
- messages delivered while hidden invoked zero edit callbacks;
- all three old messages delivered after re-enable reached the **new** callback generation;
- their physical coordinates and event metadata were indistinguishable from a new gesture.

This is a synthetic queued-message test establishing a protocol ambiguity, not a claim that Safari delivered such a sequence in the user's session. It uses real ipympl/Matplotlib event dispatch and no scientific arrays or controls. Evidence: `.codex-work/safari-retention-safety/probe.py` and `probe.txt`.

The inspected ipympl frontend serializes GUI-event metadata using `Object.keys(event)` and includes no explicit interaction-generation token. DOM timestamps are not a guaranteed serialized own property; treating an optional timestamp as a reliable scientific-edit epoch would be unsupported. The backend converts each arriving message into a new Matplotlib MouseEvent and sends it to currently connected handlers. A timeout, requiring a fresh press, or reconnecting handlers alone does not establish freshness of a complete delayed press/motion/release sequence.

### Stop decision

**Stopped without changing Product source.** The requested persistent reuse is visually feasible, but implementing only visibility plus callback reconnect would weaken the accepted stale-event guarantee after re-enable. This triggers the user's explicit stop condition. No persistent-canvas fix or Safari resolution is claimed; the current candidate remains unchanged.

A subsequent implementation needs a validated event-freshness contract across the browser/kernel boundary—for example a frontend-acknowledged interaction generation echoed with pointer events—or explicit agreement on a narrower stale-event guarantee. No such protocol, custom frontend code, new dependency or relaxed guarantee was introduced here. Controller replacement must continue closing the old interaction endpoint regardless.

The previous real-kernel publication regression remains evidence that the current candidate emits no extra display outputs during the tested toggles. It does not qualify an unimplemented persistent-canvas design. Safari confirmation would still be required after a safe implementation to prove old views no longer accumulate.

### Preservation

All existing candidate source/test/documentation hashes recorded before this investigation remain unchanged. Only this report and its small validation metadata were updated; the disposable probe is under `.codex-work`. No saved notebook output, scientific engine, request schema, dependency, unrelated work or protected evidence was changed. No simulation, installation, commit or push.
