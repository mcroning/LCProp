**Final local acceptance:** see [ACCEPTANCE.md](ACCEPTANCE.md) for the final candidate, 61 retained passing tests, user-reported browser pass, limitations and commit inventory. Historical addenda below are preserved.

# Draft-update redraw ordering correction

User capture `.codex-work/beam-first-toggle/20261016T224402.411769Z`: canonical intensity equals mesh and committed intensity exactly, yet cached backend pixels lack the second spot. Fresh canonical and mesh static PNGs have the same hash. This establishes stale cached pixels at capture time, not incorrect scientific entrance data. It does not by itself establish which browser event rendered them.

New deterministic redraw-boundary tests reproduced an inconsistent intermediate state before the correction: the first Beam 2 draw request had two markers but a mesh unequal to the new request's canonical intensity. `_changed()` called `cancel()`, which restored previous committed intensity and rebuilt markers from new controls, then requested a draw before `update()` installed the new intensity. Valid Move/Tilt release also requested a restored old-draft frame before committing new coordinates. All three new tests failed on the previous implementation; logs retained in `.codex-work/beam-atomic-draw/before.txt`. This identifies a concrete ordering defect matching the observed image, not proof of the exact browser scheduling mechanism.

Correction: `cancel` and `synchronize` accept an internal redraw flag. A draft commit cancels/restores locally without requesting an intermediate draw; the final `update()` requests the normal idle draw after data and decorations are installed. Active-beam option resets are guarded by the existing loading flag to avoid a selection-callback redraw mid-commit. Valid gesture releases likewise defer cancellation redraw; invalid gestures and explicit cancellation retain restoration drawing. Failed tilt conversion requests the restored image. No direct draw or forced refresh was reintroduced; live throttling, request construction, propagation, snapshots and exports are unchanged. Initial layout and explicit resize remain separate from this ordinary draft-commit guarantee.

Validation: **56 bounded tests passed** (22.56 s), including the three new ordering regressions, static/interactive parity, invalid gestures, Move/Tilt, lifecycle and frozen-run/export checks. **5 real-ipympl frame tests passed** (2.58 s), with five existing Toolbar deprecation warnings. The new tests require exactly one draw request per ordinary Beam 2 transition, including disabling the selected Beam 2, and exactly one final-request draw on valid Move/Tilt release. Every draw request in those transitions sees canonical mesh data. No propagation was run. Logs and source hashes are recorded in `atomic-draw-validation.json`.

**Browser acceptance pending:** restart the kernel in `notebooks/lc_static_local.ipynb`, run `%matplotlib widget` before creating the development-source controller, then enable Beam 2 once without touching the plot. Repeat off/on and numeric edits; test Move/Tilt release. Do not run a manual refresh. If the first image is still stale, use the existing non-refreshing capture cell before toggling again. This is a bounded correction to a reproduced inconsistent redraw request, not a claim of guaranteed frontend delivery.

Changed this turn: `interactive_lc.py`, `_beam_position_editor.py`, new `tests/test_interactive_lc_atomic_draw.py`, the redraw-count regression, this report and its small validation manifest. No scientific-engine, dependency or notebook-output edits; no installations, commits or pushes. Earlier addenda below remain historical evidence.

---

# Idle redraw correction — supersedes forced refresh

User browser evidence: the standalone synthetic diagnostic failed in both image-only and expanding-controls cases with refresh/direct draw. Both cases worked after replacing that operation with `canvas.draw_idle()`. This implicates the scheduling path; it does not identify a specific dropped frame or compositor defect.

Applied the same bounded correction to `_draw_committed` in `_beam_position_editor.py`. Before frontend initialization, retain local Agg layout rendering with no image transport. All committed updates then use `draw_idle()`; no forced refresh dispatch or synchronous backend draw. Live motion already uses idle scheduling. No scientific, request, controller snapshot, export, dependency or notebook changes in this correction.

Tests updated to the asynchronous contract, not immediate image transmission:
- Real-ipympl transport: **5 passed** (2.81 s), five existing Toolbar deprecation warnings. Edits cannot call direct draw or forced refresh, send draw requests but no images until the frontend draw response, then produce pixels matching the ordered composite and canonical mesh. Stable canvas identity and no premature constructor frames retained.
- Bounded presentation/geometry/isolation suite: 52 passed initially; one old synchronous-draw count assertion failed. Updated that test to count both existing cancellation and committed idle requests; the complete redraw file then **3 passed** (3.44 s). Together all 53 distinct cases pass; no propagation was performed.
- Real ipykernel test: **1 passed** (17.27 s). Zero preinitialization image buffers, valid initial PNG, deferred edit images through frontend draw messages, no additional display publications, stable canvas/layout. The sandbox initially blocked loopback sockets; the same bounded test passed with local socket permission.

Logs: `.codex-work/beam-idle-fix/{frames,regressions,redraw,kernel}.txt`; current identities in `idle-validation.json`. The older full-refresh files below are historical evidence, not the current implementation or browser acceptance status. Tests changed: `test_interactive_lc_full_refresh.py` (now explicitly guards against forced refresh), `test_interactive_lc_frame_delivery.py`, `test_interactive_lc_redraw.py`, `test_interactive_lc_display_publication.py`.

**GO for real-controller Safari/Chrome acceptance.** Export any needed completed results, restart the kernel, run `%matplotlib widget` before normal development-source initialization, then toggle Beam 2 on/off/on and try numeric radius changes, Move and Tilt. Do not run the manual refresh cell. Check immediate updating without extra pointer interaction, complete axes/colorbar/labels, and stable images after release. Minimal diagnostic success is user-reported evidence; the updated full LCProp controller has not yet been browser-qualified.

No propagation, installations, commits or pushes. Scientific source and saved notebook outputs remain untouched. The known differential-frame fault-injection limitation remains; this correction uses normal frontend scheduling rather than promising delivery guarantees.

---

# Committed full-frame recovery addendum

User-confirmed Safari evidence: enabling Beam 2 damaged axes, colorbar and labels; the manual `manager.handle_json({'type': 'refresh'}); canvas.draw()` restored the full two-beam visualization without scientific edits. This supports recovery, not a diagnosis of the browser reception/compositing failure.

Implemented the same supported refresh dispatch immediately before post-initialization committed draws in `_beam_position_editor.py`. Constructor drawing remains deferred: no refresh dispatch or image transmission before the frontend initialized message. Canonical draft updates (Beam 2, numeric edits, final gesture release) now send an independent full PNG. Existing initial-layout/resize committed draws also use this path. Ordinary marker/live motion and idle draws retain existing throttling and differential transport. No timers, retries, private backend-state changes, new dependencies or scientific changes. The Agg test double has no manager and uses its existing local draw.

Validation: **5 real-ipympl tests passed** (3.20 s, five existing Toolbar deprecation warnings); **28 focused redraw, Move/Tilt and Run-isolation tests passed** (11.44 s). New `tests/test_interactive_lc_full_refresh.py` asserts full PNGs after enable/radius/disable/reenable, exact renderer pixels even when every previous frame is discarded, stable canvas identity, and ordinary differential draws afterward. Existing Recorder asserts zero constructor image transmissions. Existing fault-injection tests still show that dropping a full recovery itself can leave corruption: this is bounded recovery, not guaranteed delivery or browser painting.

Logs: `.codex-work/beam-stage2d-refresh/frames.txt` and `regressions.txt`. Current changed-file/log hashes: `refresh-validation.json`. The earlier `validation.json` describes the pre-recovery candidate and is retained as historical evidence.

**GO for Safari acceptance, not yet browser-qualified:** export any needed completed records, restart the kernel, initialize the development-source controller normally, then enable/disable Beam 2 and change its radius without touching the plot or running the manual refresh cell. Repeat several transitions; verify complete axes, colorbar, beam labels and intensity. Then test Move/Tilt release. If corruption persists, retain a screenshot and use the existing manual refresh; do not claim success from backend transmission alone. No propagation is required. Restarting loses in-memory snapshots; initialization in the same loaded kernel would still use old class code, so do not use live module reload.

No simulations, installations, notebook-output edits, commits or pushes. This addendum supersedes only the previous no-recovery implementation status; scientific invariants and unresolved browser transport limitations remain.

---

# Stage 2D — unified static and interactive launch preview

2026-10-16. **GO for local visual review.** Static rendering is now scientifically complete to the same extent as interactive rendering. This is not a resolution or qualification of Safari differential-frame delivery.

## Implementation and scope

Changed this turn:

- `src/lcprop/notebook/_launch_preview.py` (new): shared private intensity artists and decorations; canonical physical `resolve_beam_geometry` determines internal-ray endpoints at the exit plane. No independent scientific beam model or widget dependency.
- `src/lcprop/notebook/_beam_position_editor.py`: delegates intensity/decorations to the shared functions. Existing `ray_endpoint` import remains available. Gesture handling, frontend handshake, transmission, live throttling, persistent canvas and lifecycle remain unchanged.
- `src/lcprop/notebook/interactive_lc.py`: static Agg renderer uses the same artists, title, axes, colorbar, centers, arrows and active selection; reuses its figure and single full-PNG Image widget. Layout is frozen after initialization. Active Beam selection refreshes static highlighting. Busy/disposed/loading selection callbacks return without editing presentation. Static PNG uses figure-native DPI (375×375 at default 100 DPI), matching the default interactive size instead of the previous 110-DPI export.
- `tests/test_interactive_lc_unified_preview.py` (new): four bounded tests.
- `docs/notebook_package.md`: shared preview semantics and static fallback guidance.
- This report and `validation.json`.

Both modes obtain intensity from existing `launch_preview(request)`. Numeric edits still construct/validate canonical requests before rendering. Beam colors remain white/red; the active arrow is thicker and its endpoint square larger. A zero-angle arrow has zero length; its square remains at the center. Off-aperture centers and ray endpoints can be clipped by the existing axes; coordinates are not silently clamped. No schema, propagation, normalization, dependency, Run admission, snapshot or export algorithm changes.

## Validation

| Check | Result |
| --- | --- |
| Unified preview, lifecycle, Move/Tilt, Run isolation, redraw and dragging suite | **53 passed**, 25.49 s |
| Real ipympl frame-delivery suite | **4 passed**, 2.61 s; four existing Toolbar deprecation warnings |
| Preserved-group, coherent and incoherent two-beam requests | Exact equality between static and editor controllers |
| Combined canonical intensity versus both meshes | Exact array equality for each tested state, including nonzero relative phase |
| Center and tilted endpoint coordinates | Centers exact; endpoint agreement with direct canonical geometry within 1e-14 µm |
| Physical axes limits, active highlighting | Equal limits; correct active/inactive arrow widths and endpoint styles |
| Renderer RGBA parity | Pixel-exact after aligning backend layout boxes; see qualification below |
| Static PNG versus static renderer RGBA | Pixel-exact |
| Numeric radius, phase, tilt, position, Beam 2 and active selection callbacks | Correct preview updates, no propagation |
| Completed snapshot/export after static draft edits | Same snapshot object; byte-identical arrays.npz, summary.json and experiment.lcprop.json |
| Fresh-process static fallback | No ipympl, Qt, LaunchPlane or CuPy imports; execution helper forbidden |
| Diff whitespace | `git diff --check` passes |

The editor test uses an Agg-backed widget double. Its synchronous draw_idle performs an extra initial constrained-layout pass compared with the static renderer. Initial raw pixel comparison failed on layout (6.16% of RGBA components), while intensity and geometry assertions already passed. The final parity test explicitly aligns only the axes/colorbar layout boxes, keeps data, norms, physical limits and artists unchanged, and then requires exact pixels. This is normalized-layout renderer parity, not a claim that native browser screenshots are byte-identical. No scientific or production drawing behavior was changed to accommodate that test.

Real ipympl tests exercise actual controller callbacks, normal initialization messages, PNG transport and controlled composition: ordered streams match renderer pixels; simulated missing/stale frames remain corrupt and a delivered full frame repairs them. They do not establish Safari reception or painting. No propagation simulations or new numerical qualification were performed. Existing scientific evidence is retained; current tests cover canonical entrance sampling and frozen-record behavior using bounded fake execution/retained fixtures.

Logs: `.codex-work/beam-stage2d/regressions.txt`, `frames.txt`. Early diagnostic test failures are retained in `tests.txt` and `debug.txt`; the final combined suite supersedes them. Candidate and log SHA-256 identities are in `validation.json`.

## Local acceptance and limitations

In a fresh local kernel, select the inline backend before creating the current candidate controller for static mode. No ipympl is needed. Verify both beam labels, enable Beam 2, select each active beam, change a small external incidence and azimuth, and compare the ray endpoint and numeric controls. Check coherent/incoherent modes and phase. Editing must not start Run. For interactive mode, select `%matplotlib widget` before controller creation and repeat those checks, including Move/Tilt. Do not run propagation merely to qualify presentation.

Static mode offers the same scientific launch content with numeric editing, but no mouse editing. Action/Live controls only affect an available interactive editor. Its complete PNG delivery avoids dependence on differential canvas history; this has not been newly qualified in a real Safari session. The intermittent interactive Safari corruption remains unresolved and mouse support remains optional. Browser transport, fonts and display scaling can differ despite identical scientific state and renderer geometry.

## Preservation

No installs, dependency upgrades, simulations, commits or pushes. No notebook files or saved outputs were edited. SHA-256 checks confirm the unrelated tracked `src/lcprop/pr/published_static.py`, `tests/test_pr_published_static.py`, and saved `notebooks/lc_static_local.ipynb` are unchanged from this turn's entry. Existing Stage 2C evidence and other untracked inputs remain untouched. All candidate changes remain unstaged.
