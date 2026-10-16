# Stage 2 final local acceptance

**Accepted for local commit.** This closure supersedes provisional GO/NO-GO and uncommitted-status statements in the historical Stage 2A–2D reports. Earlier reports/manifests are retained as development history, not claims about the final implementation.

## Evidence and scope

The final candidate matches `atomic-draw-validation.json` for controller, editor, ordering tests and both retained test logs. Shared renderer, unified preview tests and usage documentation match Stage 2D `validation.json`; deferred-redraw tests match `idle-validation.json`. No accepted implementation was changed during this commit review.

Retained automated evidence: **56 passed** (22.56 s) in `accepted-regressions.txt`, plus **5 passed** (2.58 s) in `accepted-ipympl-tests.txt`: **61 passing tests**. The latter records five existing Toolbar deprecation warnings. These tests use canonical entrance sampling, synthetic mouse/frontend messages, fake execution and snapshot fixtures; no propagation simulation was performed for final acceptance. Earlier real-kernel validation remains documented separately and is not counted again in 61.

User-reported local browser acceptance: after restarting with the development source and widget backend, the first Beam 2 activation rendered correctly, and the user reported **pass** for repeated toggles, Move/Tilt and numeric edits. This is user-reported visual evidence, supported by the conversation screenshot; not an independently automated browser trace or hosted-Colab qualification.

Static and interactive previews share `_launch_preview.py` intensity/decorations and the canonical physical geometry resolver. Coherent/incoherent launch construction remains in the existing scientific APIs. Backend-layout-normalized pixel parity is documented precisely in Stage 2D; native browser screenshots need not be byte-identical.

The final correction requests idle drawing only after ordinary draft updates install both intensity and decorations. It removes the reproduced intermediate old-intensity/new-markers redraw. Forced refresh/direct drawing is not the final correction. Initialization defers image transmission until the frontend handshake. Persistent canvas, optional ipympl, static fallback, Move/Tilt, live throttling and cancellation remain supported.

One-click Run captures/detaches the kernel-accepted request. Tests cover busy/disposed guards, retired Run endpoints, immutable arrays/request/metadata, later draft edits, byte-identical exports, and presentation preparation/publication failures retaining the previous completed record. Browser edits not yet synchronized to Python are outside the guarantee. Complete delayed gestures after editing resumes may change only the visible draft. User replacement of a controller must close the old one.

## Limitations

No new scientific-array propagation qualification is claimed. Existing scientific evidence remains unchanged. No package release, hosted-Colab graphics qualification, dependency upgrade or browser-delivery guarantee is implied. Static Action/Live controls still have no mouse effect without an interactive backend. Full numerical execution remains synchronous. Kernel restart discards in-memory snapshots unless exported. Protected evidence and local notebooks are not part of this commit.

## Scope and review

Only the notebook controller/private helpers, directly associated tests/docs, and Stage 2 Markdown/JSON reports are included. No scientific-engine, request-schema, dependency or version files. No notebooks, saved outputs, PNGs, NPZs, ZIPs, wheels, diagnostics or `.codex-work` files are staged. Two unrelated tracked PR modifications remain untouched. The added test-fixture adjustments align fake result requests with the enforced captured-request identity and test the new single preview widget instead of callback display output.

Explicit committed-file inventory (including this closure and identity manifest):

- `src/lcprop/notebook/interactive_lc.py`
- `src/lcprop/notebook/_beam_position_editor.py`
- `src/lcprop/notebook/_launch_preview.py`
- `docs/notebook_package.md`
- `tests/test_interactive_lc_multibeam_presentation.py`
- `tests/test_lc_static_interactive_presentation.py`
- `tests/test_interactive_lc_atomic_draw.py`
- `tests/test_interactive_lc_display_publication.py`
- `tests/test_interactive_lc_dragging.py`
- `tests/test_interactive_lc_frame_delivery.py`
- `tests/test_interactive_lc_full_refresh.py`
- `tests/test_interactive_lc_move_tilt.py`
- `tests/test_interactive_lc_preview_lifecycle.py`
- `tests/test_interactive_lc_redraw.py`
- `tests/test_interactive_lc_run_isolation.py`
- `tests/test_interactive_lc_unified_preview.py`
- `results/Research/lcprop-interactive-beamplane-stage-2-audit-v1/REPORT.md`
- `results/Research/lcprop-interactive-beamplane-stage-2a-v1/REPORT.md`
- `results/Research/lcprop-interactive-beamplane-stage-2a-v1/validation.json`
- `results/Research/lcprop-interactive-beamplane-stage-2b-v1/REPORT.md`
- `results/Research/lcprop-interactive-beamplane-stage-2b-v1/validation.json`
- `results/Research/lcprop-interactive-beamplane-stage-2c-v1/REPORT.md`
- `results/Research/lcprop-interactive-beamplane-stage-2c-v1/two-beam-evidence.json`
- `results/Research/lcprop-interactive-beamplane-stage-2c-v1/validation.json`
- `results/Research/lcprop-interactive-beamplane-stage-2c-safety-review-v1/REPORT.md`
- `results/Research/lcprop-interactive-beamplane-stage-2c-final-safari-v1/REPORT.md`
- `results/Research/lcprop-interactive-beamplane-stage-2c-final-safari-v1/source-evidence.json`
- `results/Research/lcprop-interactive-beamplane-stage-2d-v1/REPORT.md`
- `results/Research/lcprop-interactive-beamplane-stage-2d-v1/atomic-draw-validation.json`
- `results/Research/lcprop-interactive-beamplane-stage-2d-v1/idle-validation.json`
- `results/Research/lcprop-interactive-beamplane-stage-2d-v1/refresh-validation.json`
- `results/Research/lcprop-interactive-beamplane-stage-2d-v1/validation.json`
- `results/Research/lcprop-interactive-beamplane-stage-2d-v1/accepted-regressions.txt`
- `results/Research/lcprop-interactive-beamplane-stage-2d-v1/accepted-ipympl-tests.txt`
- `results/Research/lcprop-interactive-beamplane-stage-2d-v1/ACCEPTANCE.md`
- `results/Research/lcprop-interactive-beamplane-stage-2d-v1/acceptance-identities.json`
