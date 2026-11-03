# Stage 2C — final Safari diagnostic feasibility gate

Date: 2026-10-16.

**NO-GO for determining Safari's frame-reception/painting sequence through the existing Python-facing APIs. Stop condition reached.** No new diagnostic notebook or recorder was built, no Product changes made, and no further reproduction requested from the user. The recommended operational fallback is the existing static launch preview with numeric controls.

## Why another sender-side diagnostic cannot answer the question

The installed ipympl 0.10.0 frontend receives binary PNG data without a per-frame sequence identifier echoed to Python. It replaces the source of one asynchronous Image object; the image's `onload` callback composites pixels into an offscreen canvas and updates views. Neither binary reception nor successful image load/paint sends a corresponding frame acknowledgment to the kernel.

Its outgoing normal initialization, draw, resize and interaction messages can establish that some frontend code is active. They cannot establish which PNG was received, superseded, decoded, composited, or painted. A kernel timestamp on outgoing data is a sender observation, not a browser reception/paint timestamp. A subsequent mouse or draw event is not a receipt for an identifiable frame.

The inherited Matplotlib `handle_ack()` has an explanatory comment about acknowledgments but its body is `pass`. The inspected ipympl frontend does not send an image `ack` at all. Even an uncorrelated acknowledgment would not prove paint completion. Therefore the required distinction cannot be obtained by adding more Python transmission logs, widget child counts, or timing instrumentation.

The inspected code is the local installed frontend bundle; the exact assets loaded in the user's Safari session have not been independently identified. This further limits any inference about that session.

## Source evidence

Read locally, without installations or scientific imports:

- `ipympl/nbextension/index.js.map`, embedded `src/mpl_widget.ts`: initialization at 61–100 and 130–141; binary receipt at 291–309; image load/composition at 344–396. Binary receipt sets Image.src and clears waiting_for_image; onload paints but sends no receipt/paint event.
- Matching compiled JupyterLab assets `ipympl/labextension/static/643.717608b8dea13aad5293.js` and `688.7f16beda883825078f42.js`: normal initialized-message send present; no image-ack send. The source-map finding was checked against compiled assets, not assumed solely from comments.
- `matplotlib/backends/backend_webagg_core.py:295–305`: inherited no-op acknowledgment handler; `handle_draw()` invokes draw.
- Existing Stage 2C frame-delivery evidence: `results/Research/lcprop-interactive-beamplane-stage-2c-v1/REPORT.md`, `two-beam-evidence.json`, and `.codex-work/beam-stage2c-two-beam/qualified-output/`.

Installed files inspected under `/Users/mcroning/miniforge3/envs/lcprop-interactive-test/lib/python3.12/site-packages/`. This path identifies the reviewed environment, not a notebook runtime requirement.

## Established versus missing evidence

| Question | Status |
|---|---|
| Actual one-beam → two-beam callbacks and exact two-beam request | Already captured in retained tests |
| Canonical intensity equals Matplotlib mesh, exact color limits | Already verified for those recorded states |
| Ordered PNGs reproduce corresponding renderer | Verified, including each ordered prefix |
| Outgoing modes, PNG dimensions/hashes and event sequence | Retained; not browser reception records |
| Kernel transmit timing for a new Safari incident | Not newly recorded; cannot close the receiver-side gap |
| Dropped/stale-painted frame dependency can cause persistent corruption | Deterministically demonstrated with explicit fault injection |
| All received Safari frames decoded/painted in order | Unknown; supported API does not expose this |
| A delivered full frame repairs synthetic compositor corruption | Verified |
| Manual refresh repairs the user's corrupted Safari view | Not established by retained user evidence |
| Production full-frame recovery is durable | Not established; a skipped recovery frame can still leave corruption |

Retained transition tests passed four cases; no new numerical or browser qualification is claimed here. The earlier one-beam static screenshot cannot substitute for a matching two-beam browser-state comparison.

## Supported manual refresh: limited observation, not a fix

If the user chooses to try a single refresh in the already running synthetic probe, the existing backend supports:

```python
canvas = probe._position_editor.canvas
canvas.manager.handle_json({'type': 'refresh'})
canvas.draw()
```

This code depends only on the existing controller, not the working directory. It requests a full next frame without changing scientific parameters or writing private backend fields. It does not invoke propagation. A visible repair would show that a refresh repaired that incident; it would not reveal the preceding frame-reception order or guarantee future repairs. No user refresh outcome is inferred or claimed in this report. This optional two-line observation is not a request to reproduce the intermittent failure repeatedly.

The previous deterministic recovery test establishes identical request/array state across refresh and a full PNG identical to the renderer. Those acceptance checks apply to the retained test, not a newly observed Safari incident.

## Recommendation and reliable fallback boundary

**NO-GO:** do not implement an automatic full-frame retry, arbitrary delay, dependency upgrade, or frontend workaround on the claim that Safari's specific reception failure has been proved. Do not treat `set_image_mode('full')` as a persistent policy; the installed backend automatically reselects differential mode.

**GO for the existing static-preview/numeric-control fallback.** In a fresh local kernel, select the ordinary inline Matplotlib backend before creating the controller:

```python
%matplotlib inline
from lcprop.notebook import InteractiveLC
probe = InteractiveLC()
display(probe.widget)
```

Use the existing environment containing the reviewed development candidate. Do not replace an active scientific controller without exporting needed results; completed snapshots are in-memory. The initializer above is independent of Jupyter's working directory when LCProp is available in that kernel environment. It performs no installation and does not select source by relative path. For production 0.3.0, use its existing long import path instead; it does not contain this development candidate.

The current editor rejects incompatible non-ipympl backends and automatically retains numeric controls with the static preview. That preview sends a complete PNG through the existing Image widget rather than an incremental WebAgg image stream. It preserves canonical scientific construction and avoids the demonstrated differential-frame dependency. This is the already implemented and tested fallback, not a new interface or a new Safari qualification. Graphical dragging is unavailable in this mode; numeric beam position, radii, angles, coherence and power remain usable. No Run is needed to edit or view the launch.

For a future graphical fix, receiver/decode/paint evidence would require browser-side inspection or a frontend capability that is not exposed by the currently supported Python API. Custom JavaScript and dependency changes are outside this task. Per the explicit stop condition, no additional recorder system or working-directory-dependent notebook was created to mask that gap.

## Preservation and activity

Read-only inspection plus this report and its small source-evidence manifest. Current candidate source hashes were checked against Stage 2C validation metadata and remain unchanged. Existing unrelated changes, user inputs, saved notebook outputs and protected evidence were not edited. No scientific simulations, tests that execute propagation, installations, commits, pushes, custom JavaScript or Product modifications.

**Final decision: NO-GO for causal Safari frame-sequence qualification under the current API constraints; use the existing static-preview fallback for dependable exploratory work while graphical rendering remains unqualified.**
