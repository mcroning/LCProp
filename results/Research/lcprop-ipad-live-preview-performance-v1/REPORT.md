# iPad bounded live-preview performance assessment

**GO for a bounded hosted experiment; NO-GO for enabling live preview by default in a future release yet.** Retain the accepted release-only notebook until iPad rendering and latency are checked. No production changes or publication were made.

## Baseline and method

Accepted notebook: `3d58d04acda30faa4d8c8057ee9a644f20484088`. User reports finger/Pencil Move and Tilt work, with noticeable but acceptable approximately 200 ms lag. Beam intensity updates on release because that notebook disables live preview.

Verified the target-installed `0.4.1.dev0+stage5c` runtime against all 269 wheel hashes. The existing editor already implements canonical live launch computation at at most 4 Hz, with no timer queue, a 20-update per-gesture cap, and a 100 ms computation fallback. Release always recomputes the canonical launch. No full request serialization is added to motion.

Compared release-only, existing 4 Hz and experimental 5 Hz settings for Move/Tilt with one/two beams: 12 cases, each with 75 motion events on a synthetic 25 Hz/3-second schedule. Measured Python handler, canonical launch, Agg raster and full-frame PNG encoding separately. Full PNG decode was checked against the exact rendered RGBA buffer on every frame. These are local sequential cost measurements, not real-time Colab latency or browser paint tests. Scheduling used a controlled clock; the unchanged slow-update guard is covered by the regression tests, not timed by that clock.

## Measurements

Median milliseconds, ranges across the four beam/action cases:

| Preview setting | Motion handler (includes live work when due) | Canonical launch when due | Raster | Full-frame PNG encode |
|---|---:|---:|---:|---:|
| Release only | 0.02–0.12 | 0.00–0.00 | 18.20–20.48 | 2.72–2.98 |
| 4 | 0.02–0.10 | 1.96–3.52 | 17.88–19.33 | 2.69–2.90 |
| 5 | 0.02–0.11 | 1.95–3.55 | 18.60–19.42 | 2.71–2.92 |

The 5 Hz schedule produced 14 provisional updates per 3-second gesture; the existing 4 Hz schedule produced 10. The schedule sets the next deadline after work completes, so it is an upper bound, not a guaranteed update frequency. Markers remain on the unchanged 25 Hz adapter path. Each synthetic motion was followed by a render to measure cost separately; this does not model frontend draw coalescing. Full PNG encoding is explicitly not a measurement of ipympl's differential image protocol.

Adapter mock-DOM tests passed: 240 input moves produced 24 delivered moves and 216 skipped moves. 100 complete mock test suites took 71 ms (0.71 ms/suite, including fixtures/assertions). This measures only synthetic JavaScript processing, not actual browser dispatch, transport or rendering. The adapter has no deferred motion timers; release and cancellation remain immediate. Finger/Pencil event ordering, duplicate suppression, exact final coordinates, native mouse pass-through and framing refusal passed.

## Correctness and preservation

- All 12 cases preserved the frozen request throughout motion, committed the exact release coordinates, and refreshed an intensity array byte-identical to a fresh canonical preview of the final request.
- Cancellation after a provisional update restored the exact committed request and intensity. No scientific execution occurred; the execution entry point was guarded to fail if called.
- **63 installed-wheel regression tests passed in 33.99 seconds**, covering hit tolerance, inactive/overlapping beams, Move/Tilt, cancellation, slow-live fallback, final refresh, snapshots, metadata and export isolation. Completed records in these tests are synthetic fixtures, not new simulations.
- The 30-pixel global target and adapter bytes remain unchanged. Desktop defaults and mouse pass-through are untouched. Live intensity adds rendering work for any input in the experimental instance; desktop latency with live preview is not newly browser-qualified here.
- No backend RGBA/PNG corruption was found in the local trials. This does **not** exclude the previous frontend image-corruption issue: no hosted browser or asynchronous comm queue was available for inspection.

## Experimental implementation and remaining gate

`live-preview-experiment.ipynb` is a local-only copy of the immutable acceptance notebook. Its installation/hash verification and entire synthetic adapter cell are identical. It enables the existing public live-preview checkbox and sets `editor.LIVE_INTERVAL=0.2` on that disposable instance only. The latter is a disclosed private diagnostic constant override, not a proposed production API. No controller, engine, request schema or installed package files were edited. Existing update cap, slow fallback, final refresh and disabled/disconnected Run remain intact.

Local results show modest additional canonical-computation cost; they do not establish improvement in the reported 200 ms lag. At 5 Hz, beam-image sampling itself may wait approximately 200 ms before computation/transport, even when marker delivery is faster. Rendering and frontend transport may dominate perceived responsiveness. Do not raise the 25 Hz limit.

Before enabling a future release default, one bounded hosted trial should compare the accepted release-only notebook with this experimental setting using finger/Pencil Move/Tilt, cancellation, outside scrolling and mouse input where available. Record perceived marker lag, image lag, whether image updates lag behind release, and any canvas distortion. Confirm final numeric values and image settle at the release target. Any substantial responsiveness degradation, stale queued redraws or image corruption is a stop: retain release-only optical refresh. The current 100 ms safeguard measures launch calculation only; it cannot detect slow frontend paint or transport.

## Retained evidence

`measure.py`, `measurements.json`, `measurement-log.txt`, `adapter-measurements.json`, `regressions.txt`, `environment.json`, and the local experimental notebook. No dependency upgrades, builds, simulations, commits or publication. The accepted notebook and unrelated working files are preserved.
