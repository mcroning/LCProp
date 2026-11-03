# InteractiveLC BeamPlane Stage 2 — graphical interaction feasibility

Date: 2026-10-15. Inspected HEAD: `9151e6bc7db8adb1a6788942e7be5bf315063fa2` (Stage 1B), based on two-beam commit `681f669b5ddb41d6bfd829856933d0ad0e23eb83`.

**Recommendation: GO for a bounded, optional ipympl launch-position editor; NO-GO for claiming hosted/browser qualification before actual event-path testing.** No scientific architectural gap was found. Keep numeric controls authoritative, retain the static preview fallback, and update scientific draft positions only on mouse release. No code, dependency, environment, notebook, or scientific evidence was changed in this audit. Only this report was created. No simulations, installations, tests, builds, commits or pushes were run.

## 1. Existing reusable components and boundaries

| Source / symbol | Verified responsibility | Reuse / boundary |
|---|---|---|
| `src/lcprop/core/beams.py:56`, `BeamChannel`; `:81`, `validate` | Entrance-plane x0/y0 in µm, radii, phase, physical angle conventions; finite positions required | Keep these exact fields; no graphical beam model |
| Same file, `BeamStack.validate`, `coherence_groups` | Ordered channels and canonical group resolution | Selection is a channel index, never a group or intensity peak |
| `src/lcprop/core/grid.py:97` | Cell-centered physical x/y coordinates | Use canonical grid and axes transforms, including half-cell placement |
| `src/lcprop/optics/physical_launch.py:21`, `transverse_frame`; `:65`, `resolve_beam_geometry`; `:92`, `external_direction_for_exit`; `:130`, `sample_resolved_beam` | Principal axes, refraction, projected profiles, inverse exit geometry, infinite-plane normalization | Reuse indirectly through launch builder. Position-only milestone needs no ellipse, angle, exit-handle or new geometry implementation |
| `src/lcprop/optics/launch.py:159`, `build_launch`; group-aware `total_intensity` used by controller | Channel sampling and power normalization | Recompute preview through existing `launch_preview`, never translate image pixels as a physical field |
| `src/lcprop/optics/sampling.py:44`, `qualify_launch_sampling` | Sampling validation plus aperture/periodic-edge advisories | Preserve distinction between rejection and advisory |
| `src/lcprop/notebook/interactive_lc.py:27`, `request_for_controls`; `:59`, `require_bounded_request`; `:71`, `launch_preview` | Existing one/two-beam draft, preset support guard, NumPy launch construction | All drag edits flow through these functions unchanged |
| Same file `:403`, `request`; `:421`, `_changed` | Widget-to-request mapping, canonical preview and advisories | Reuse existing first-beam x/y and second-beam x0_um/y0_um controls; batch both values |
| Same file `:447`, `_run`; `CompletedRun`, `export_completed` | Explicit solver launch, busy state, immutable completed ownership | No mouse callback may invoke execution or replace completed data |
| Same file `:263`, `display_figure` | PNG encoded into `ipywidgets.Image` | Static: no Matplotlib mouse events reach it. Do not change completed plots/exports merely to enable launch interaction |
| `src/lcprop/adapters/launchplane.py`; desktop beam panel | Desktop model conversion and launch resolution | No need to import Qt/LaunchPlane into notebook |
| `/Users/mcroning/LaunchPane/src/launchplane/canvas.py:159`, `BeamItem.itemChange` | Qt scene coordinate conversion, bounds, move signal | Qt-specific. It negates scene y; a Matplotlib data-space event must NOT repeat that negation |

Position validation does not impose an aperture bound in `BeamChannel.validate`. Finite off-aperture beam centers remain representable; boundary-envelope checks are not a universal prohibition. A limited dragging region is an interaction policy, not a new scientific support envelope. Preserve numeric entry/load of existing positions and canonical warnings.

The preview currently draws `pcolormesh(grid.x_um, grid.y_um, intensity.T, shading='nearest')` with equal aspect. The scientific array order is x,y; display transposes it. Mouse `xdata/ydata` are already physical x/y in µm, not pixel indices, and must not be swapped, inverted or snapped to grid cells.

## 2. ipympl feasibility and dependencies

Official ipympl documentation lists Jupyter notebooks, JupyterLab and Colab support. Its installation page specifies Colab's `output.enable_custom_widget_manager()` and warns that Python backend and browser widget frontend versions must match. The published compatibility table currently lists ipympl 0.10.0 / jupyter-matplotlib 0.12.0 with JupyterLab >=2,<5 and Matplotlib >=3.5. This supports a candidate pairing, not qualification of the user's exact Safari/Notebook environment. [Official installation and compatibility](https://matplotlib.org/ipympl/installing.html), [supported environments](https://matplotlib.org/ipympl/index.html).

The current `pyproject.toml:16` colab extra contains Matplotlib, ipywidgets and IPython, but no ipympl. Production 0.3.0 cannot acquire this feature merely by rerunning its pinned notebook. A future implementation/release must provide optional ipympl installation and explicit backend/widget-manager setup; do not silently switch an already-running kernel's backend or reload live controllers. Initial qualification should pin a tested ipympl version separately before deciding whether to extend the existing colab extra. Core imports and ordinary static/numeric use must remain independent of ipympl, Qt and CuPy. No installation or compatibility test was performed here.

Use an explicitly requested interactive preview mode, initialized before displaying the controller. Embed a persistent `fig.canvas` in its widget layout. Create the figure inside a scoped `plt.ioff()` context to avoid automatic duplicate display. The official example distinguishes interactive canvas display from static figure rendering, recommends artist updates, and documents callback-error capture. [ipympl widget embedding example](https://matplotlib.org/ipympl/examples/full-example.html).

Local Jupyter server and kernel may use different environments; inspect both widget frontend and kernel backend at qualification. Colab requires its custom widget manager and a live comm connection. A checkbox working in Safari establishes ipywidgets communication, not ipympl pointer-event compatibility. Test Safari locally and hosted Colab explicitly. This audit does not establish a version incompatibility or justify replacing the visualization stack.

## 3. Smallest interaction contract

1. Add an optional persistent launch canvas, two labeled center markers, and a **Move beam: Beam 1 / Beam 2** selector. Beam 2 is selectable only when enabled. Keep all scientific settings in existing controls.
2. Drag only the explicitly selected marker. Start on left-button press near that marker, measured in display pixels so hit tolerance survives aspect changes and resizing. Do not infer a beam from bright spots: coherent interference can create many peaks, and zero-power beams still have a valid position.
3. Preserve pointer-to-center offset to avoid jumping when grabbing the marker edge. During motion, move only a proposed-position marker; retain the last validated image with a clear pending-position indication. Do not run launch sampling, recreate a figure, or send scientific arrays on each motion.
4. On valid release, update the appropriate pair of numeric controls atomically while suppressing intermediate `_changed` callbacks, then invoke exactly one existing request/launch/advisory refresh. Use the existing loading-guard pattern with exception-safe restoration; holding notifications on each separate widget alone does not guarantee a single refresh across both widgets.
5. Reject nonfinite coordinates, wrong axes/colorbar events and non-left drags. Define the drag region as the physical aperture rectangle derived from grid metadata/cell edges, not extrema of sample centers or zoomed limits. Cancel release outside that region; do not wrap, clamp silently or alter saved off-aperture values. Explain numeric editing for off-view positions. This is a proposed UI policy for implementation review.
6. Disable dragging during toolbar pan/zoom, loading and Run. Escape, leaving the canvas, disabling Beam 2, load/reset, and invalidated gesture ownership cancel the drag and restore its marker. No click-to-add, resize handles, tilt handles, multiselect, dragging of results, or extra channels in this milestone.
7. Keep snapshot, completed plots, summaries, export paths and time budget untouched. A draft position change is exploratory under existing logic. Execution remains exclusively the existing Run callback.

For exact overlap the explicit selector is sufficient: select Beam 2 then drag its coincident marker; do not arbitrarily choose the top artist. Use distinct marker outlines/labels and selection emphasis, not altered intensity. Marker coordinates represent entrance centers, not exit trajectories. Dragging an oblique beam preserves incidence, roll, radii, absolute phase and group fields; canonical launch sampling handles the phase reference at the displaced center. Never add a compensating phase or renormalize capture.

Matplotlib supplies axes-aware mouse coordinates and event connection/disconnection. Use those existing transformations, not browser CSS pixel formulas. Store callback IDs and strong ownership of handlers for the canvas lifetime. [Matplotlib event handling](https://matplotlib.org/stable/users/explain/figure/event_handling.html).

## 4. Synchronization, latency and ownership risks

Current `_changed` clears output, creates a figure, serializes PNG and closes the figure for every edit. That lifecycle must remain for static mode but cannot be reused literally for an interactive canvas: clearing/closing it mid-drag disconnects its frontend. Interactive mode should update existing mesh/markers and colorbar using the same computed intensity and existing limits policy. Completed plotting/export remains static and unchanged.

Use one controller-owned canvas and one event-registration set, reused when the same initialization cell is rerun. Disconnect callbacks and close only owned resources when explicitly disposing/replacing a canvas. Never leave an older displayed canvas capable of editing a newer controller. Tests must cover multiple displayed views and stale instance events; Python variable rebinding cannot retarget old widget models.

The retained local synchronization report shows a working Safari probe and evidence of multiple controller creations/stale views in the original notebook, while noting that exact historical click-to-model association was unavailable. Treat this as a known lifecycle risk, not a diagnosed ipympl defect. Keep the current compatible-instance reuse launcher. Saved widget output after kernel restart is not a live canvas; export completed results before restarting.

A simple `busy` guard alone is insufficient for a synchronous solver: browser messages can arrive after Run completes. At Run/load/reset start, cancel the gesture and change the canvas interaction generation or disconnect/recreate the interaction endpoint; require a fresh press after explicit re-enabling. Test queued old press/motion/release sequences against generation ownership. A conservative first milestone can leave graphical editing disarmed after Run until the user explicitly enables it. No scientific run is needed to test these transitions.

Cloud latency remains unmeasured. Marker-only motion and `draw_idle()` reduce work, but Python-backed interaction still crosses the comm link; do not promise desktop-rate dragging. No unbounded motion queues, retained image sequences or full volumes are needed. Canonical preview allocation remains the existing bounded grid launch cost, once per committed release. If a release is lost, cancel on leave/new interaction rather than leaving the controller stuck.

## 5. Alternatives

No concrete evidence establishes ipympl incompatibility in either intended environment. Therefore do not introduce Plotly, a browser canvas framework, custom JavaScript or a second beam model. The existing numeric controls plus static PNG are the reliable fallback when ipympl or its widget manager is unavailable. A click-to-position mode could reduce motion latency if hosted testing reveals an actual problem, but is a later interaction simplification, not a new dependency. Static inline Matplotlib cannot itself provide mouse dragging.

## 6. Proposed bounded files and implementation sequence

- `src/lcprop/notebook/interactive_lc.py`: optional launch-canvas ownership, selection/gesture callbacks, paired-control transaction, busy/load/reset integration; no changes to request/execute/export/scientific helpers.
- `tests/test_interactive_lc_dragging.py` (new): synthetic Matplotlib events, stubbed launch data and solver-forbidden tests.
- `tests/test_notebook_lazy_interface.py` and existing multibeam/presentation tests: import isolation and compatibility checks only as needed.
- `docs/notebook_package.md`: activation, selected-beam/overlap rules, numeric fallback, qualified environment limitations.
- A dedicated local/Colab qualification notebook or instructions: backend setup and no-solver interaction probe. Preserve current production-pinned notebooks; do not imply they load unreleased code.
- `pyproject.toml`: optional notebook dependency addition only after choosing and qualifying the version set. No mandatory core dependency changes.

First milestone: opt-in one/two-beam entrance-center dragging with explicit selection, marker-only motion, release-to-commit, one canonical preview refresh, and unchanged numeric/static fallback. No physical-profile outlines are required. Split local event qualification from hosted qualification and publication; source implementation alone cannot establish either browser result.

## 7. Focused validation requirements

| Area | Required evidence |
|---|---|
| Coordinate mapping | Synthetic mouse events on unequal/asymmetric axes, nonzero centers, cell-centered half-offsets, resize, zoom and aspect changes; known physical positions yield exact numeric values within display transformation precision; no y reversal or x/y swap |
| Bounds | Aperture edges vs center extrema; outside axes/colorbar; NaN/None; release outside canvas; preserve existing off-aperture numeric requests and warnings |
| Selection | One beam, enabled/disabled second, overlapping centers, zero-power beam, distinct groups and coherent interference; only selected channel moves |
| Atomic updates | Both coordinates update under one guard and exactly one canonical preview call occurs after release; no previews on motion; exceptions restore guard/gesture state |
| Scientific equivalence | Same final numeric inputs produce identical encoded request and canonical launch fields for coherent/incoherent and oblique/rolled cases; do not run propagation in event tests |
| Phase/normalization | All fields other than selected x0/y0 remain exact, including phases/groups/powers; canonical intensity and sampling diagnostics unchanged |
| Snapshot/export | Synthetic completed snapshot retains exact serialized bytes, summary, plot data and export identity through drag, rejection, reset, timeout and load; Run/execute replaced with fail-on-call sentinel |
| Lifecycle | Repeated init yields one canvas and event set; stale canvas/generation events ignored; busy/load/reset cancel; queued events cannot mutate post-run state; pan/zoom never edits coordinates |
| Dependency isolation | Core and static InteractiveLC import/use without ipympl, Qt, LaunchPlane or CuPy; missing backend fails clearly and leaves numeric fallback usable |
| Real frontend | Local Safari/Jupyter and hosted Colab: drag each beam and overlap, inspect actual Python numeric values, numeric-to-marker update, repeated init, one display, narrow layout, no solver activity; record exact frontend/backend versions |

Stage 1 retained scientific evidence: 195 passes and its documented historical bootstrap-test failure, including single-beam and bounded two-channel incoherent numerical comparisons. Stage 1B retained evidence: 50 passes and its separate unchanged saved-notebook-output failure; canonical data unchanged, synthetic plot/snapshot checks. Neither establishes new drag behavior or hosted ipympl compatibility. No numerical evidence was regenerated in this audit.

## 8. Decisions and final gate

There is no scientific blocker to position-only graphical editing through existing controls. Remaining decisions are presentation/deployment choices: optional ipympl version and activation, explicit selection policy, cancellation at bounds, and stale-event rearming. Proposed defaults above avoid silent scientific changes. Arbitrary multibeam support, exit/angle handles, changing launch normalization, desktop editor payload compatibility, or new scientific retention are outside scope.

**GO: bounded opt-in implementation and synthetic event tests. NO-GO: mandatory/default graphical mode or hosted qualification claims until local and Colab pointer/comm lifecycle checks pass.** Preserve numeric-only operation regardless of graphical availability.

Preservation: both pre-existing tracked modifications (`src/lcprop/pr/published_static.py`, `tests/test_pr_published_static.py`) remain outside this audit. No user notebooks, saved outputs, protected reports or evidence were edited. This report is the sole deliverable; it does not authorize implementation, installation or publication.
