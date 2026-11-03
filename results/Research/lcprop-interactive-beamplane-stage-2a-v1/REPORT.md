# InteractiveLC Stage 2A — optional graphical beam positioning

Date: 2026-10-15. Base: `9151e6bc7db8adb1a6788942e7be5bf315063fa2`.

**Implemented; 61 focused checks passed, ready for local Safari/Jupyter review.** No propagation/material simulations, environment installations, commits, pushes or publication. Real ipympl canvas validation was headless; browser interaction is not yet qualified.

## Exact candidate scope

- `src/lcprop/notebook/interactive_lc.py`: opt-in controls/host, optional editor ownership and disposal, paired-position callback integration, lifecycle cancellation, scoped suppression of automatic figure display.
- `src/lcprop/notebook/_beam_position_editor.py` (new): private presentation component; persistent ipympl canvas, markers and synthetic-event-testable gesture handling. No duplicate BeamChannel model or physics.
- `tests/test_interactive_lc_dragging.py` (new): 15 focused test cases.
- `docs/notebook_package.md`: optional setup, interaction contract and manual verification.
- This report and `validation.json`: evidence and source identities.

No changes to dependency specifications, published 0.3.0, notebook files, existing scientific helpers, engine, request schemas or normalization. Existing unrelated PR modifications and saved local notebook outputs were preserved.

## Behavior

Static PNG preview remains the default. Explicitly enabling graphical positioning creates one persistent approximately 375 px launch canvas. Move beam offers Beam 1 and, only while included, Beam 2. Labeled center markers represent entrance-plane physical positions, including zero-power or overlapping beams. Selection is explicit, not inferred from optical maxima.

Left-button press within 12 display pixels of the selected center starts a gesture, preserving the pointer-to-center offset. Motion changes only a proposed marker and label. The old image remains visible with a pending-position explanation; no request/control update or canonical launch calculation occurs during motion. Valid release writes both numeric coordinates under the existing loading guard, then calls `_changed` once. That path invokes the unchanged canonical `launch_preview` and sampling advisories exactly once.

Coordinates use Matplotlib xdata/ydata in µm, with no sign reversal, transposition or sample snapping. Physical aperture bounds come from the existing grid specification, not sample-center extrema. Outside/nonfinite/wrong-axes/wrong-button gestures cancel. Escape or leaving the canvas cancels. Numeric off-aperture inputs remain permitted, with existing validation/advisories. Coherence, phase, radii, orientation, wavelength, power and all other request fields are unchanged by dragging.

The canvas is retired on Run, load (including rejected load), reset, explicit `close()`, or disabling graphical mode. Old callback connections are removed, and retained callback objects also reject events when their editor is inactive/no longer controller-owned. Graphical mode must be explicitly re-enabled afterward; it is not silently rearmed after a synchronous run. Busy/loading and pan/zoom state reject dragging. Draft changes or beam selection changes cancel an in-flight gesture. Disabling Beam 2 removes its selection/marker.

Controller replacement requires `demo.close()` before deliberate replacement. Reusing the same compatible controller remains the recommended notebook initialization behavior. Arbitrary rebinding of Python variables cannot automatically discover or disable old visible controllers; this limitation is explicitly documented, not claimed solved. Disposed canvases cannot mutate either old or replacement controls.

Completed snapshots, summaries, plots, exports and time budgets remain separately owned. No mouse callback calls `execute` or `_run`. Existing numerical controls still provide full precision and remain the authoritative draft model.

## Optional dependency and coexistence finding

The actual ipympl 0.10.0 package initializer calls `matplotlib.use` on import. The first smoke check detected the global backend change; the final implementation therefore requires explicit `%matplotlib widget` activation in a fresh kernel **before controller creation**, rather than silently selecting a backend while results exist. If this precondition or ipympl is unavailable, the controller reports the setup requirement and retains static/numeric operation.

After explicit activation the editor constructs its own Figure, Canvas and FigureManager, without registering it in pyplot's automatic display queue. Static launch and completed figure creation use scoped `plt.ioff()` so their existing PNG rendering does not also produce automatic widget views. Data, plots, diagnostics and export formats are unchanged. Disabling graphical mode regenerates the static preview. Canvas and toolbar widgets are closed on disposal.

No mandatory dependency was added. The optional local procedure is documented in `docs/notebook_package.md`: in the disposable kernel environment, install `ipympl==0.10.0`, restart after exporting results, run `%matplotlib widget`, then initialize the candidate controller. Do not install/reload around live snapshots or alter the existing `lcprop-new-user` environment. Server/kernel frontend versions must match. Official compatibility documentation lists the ipympl 0.10.0 / jupyter-matplotlib 0.12.0 pairing with JupyterLab >=2,<5 and Matplotlib >=3.5. Hosted Colab also needs its custom widget manager; hosted qualification is deferred.

References: [ipympl installation/compatibility](https://matplotlib.org/ipympl/installing.html), [widget embedding and duplicate-display guidance](https://matplotlib.org/ipympl/examples/full-example.html). These document support; they are not evidence of this candidate passing a real browser session.

## Validation

Final focused command used the existing disposable Python environment, without modifying it:

```sh
PYTHONPATH=src:/Users/mcroning/LaunchPane/src MPLBACKEND=Agg \
MPLCONFIGDIR=.codex-work/beam-drag-stage2a/mpl \
.codex-work/pypi-stage2a/colab-env/bin/python -B -m pytest -q -p no:cacheprovider \
 tests/test_interactive_lc_dragging.py \
 tests/test_interactive_lc_multibeam_presentation.py \
 tests/test_notebook_lazy_interface.py \
 tests/test_lc_static_interactive_presentation.py \
 tests/test_interactive_lc_multibeam.py -k 'not bounded_execution' \
 --basetemp=.codex-work/beam-drag-stage2a/validated
```

**61 passed, 2 deselected in 9.99 seconds.** The two omitted cases perform scientific propagation; no propagation was run. New tests use synthetic pointer events, real widget controls, a deterministic Agg-backed widget adapter, canonical entrance sampling and synthetic completed snapshots. Coverage includes:

- exact paired-coordinate commit and one launch refresh; no motion refresh;
- coherent/incoherent overlap selection and exact canonical request/launch-array equality;
- off-center physical mapping, resize/zoom, grab offset, bounds/NaN/axes/button rejection;
- cancellation, recovery, numeric-to-marker synchronization, disabled Beam 2 and pan mode;
- stale callbacks after reset, failed Run, rejected load, disposal and re-enabling;
- completed snapshot/summary preservation and byte-identical arrays/experiment exports;
- missing backend fallback, explicit backend requirement, static reactivation;
- core/static initialization with ipympl, Qt, LaunchPlane and CuPy imports blocked.

Earlier development test failures were corrected (test-fixture import path and using the actual toolbar-mode contract instead of deprecated Matplotlib navigation-setting behavior). No failing final checks remain. The historical saved-output lifecycle test was not rerun or altered; its prior baseline-identical failure remains documented in Stage 1B. No wider-suite pass is claimed.

### Real optional-backend smoke check

Downloaded the public `ipympl-0.10.0-py3-none-any.whl` to `.codex-work/beam-drag-stage2a/`, verified its PyPI-reported SHA-256, and extracted it only there for temporary PYTHONPATH use. Nothing was installed into an environment. Wheel SHA-256:

`a09c4f0ff86490cc62aed45e53b912fb706e3ec3506c4a51ce4a670d6667f5ce`

With explicit widget backend activation, the actual ipympl Canvas/FigureManager rendered, processed Matplotlib MouseEvents via its callback registry, updated numeric positions, rendered a synthetic completed figure, and disposed successfully. The pyplot/ipympl automatic-show queue stayed empty, and the chosen backend remained unchanged. Evidence: `.codex-work/beam-drag-stage2a/real-canvas.txt`. This is Python-side canvas validation, **not** a browser/websocket or ipykernel-session test. The test environment does not contain ipykernel metadata; no claim of an installed Jupyter runtime qualification follows.

Environment: Python 3.12.13, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2, ipywidgets 8.1.9. Final test log: `.codex-work/beam-drag-stage2a/validated.txt`.

Syntax compilation, Python 3.10 grammar and `git diff --check` passed. AST comparison confirms all top-level scientific/request/snapshot/export functions are unchanged; the only top-level plot-function change is scoped `ioff()` during figure creation. Existing scientific parity evidence is retained, not regenerated. New equality checks concern launch fields and scientific exports, not a fresh nonlinear convergence qualification.

## Preservation and limitations

The starting preservation manifest covers tracked files and the saved local notebook. All 902 recorded files outside the two intended existing source/documentation edits remain byte-identical, including the two unrelated PR modifications and local notebook outputs. No protected evidence was overwritten; only new report/evidence files were added. The index remains unchanged and empty.

Browser rendering, pointer latency, Safari high-DPI behavior, missing frontend-extension messages, and hosted Colab comm behavior still require manual acceptance. The editor hides its toolbar in this first milestone, while rejecting pan/zoom modes if activated programmatically. It intentionally does not add dragging of radii/angles, click-to-add beams, new scientific models or automatic execution. It keeps one image/mesh and bounded center-marker state, not histories or scientific volumes. Graphical mode requires candidate source plus an optional backend; published 0.3.0 and production-pinned notebooks remain unchanged.

## Local Safari/Jupyter acceptance procedure

1. Export any completed result before restarting. In `lcprop-interactive-test`, separately install the optional ipympl version if desired; no such installation was performed by this task. Confirm Jupyter uses that kernel and imports the candidate controller, not production 0.3.0.
2. In a fresh kernel run `%matplotlib widget`, then the existing reusable local initialization cell. Do not clear or rewrite saved user notebook output merely for this test.
3. Confirm static mode initially; enable graphical positioning. Select Beam 1 and drag its marker. During motion numeric values must remain unchanged; after release both must match the physical destination. Do not press Run.
4. Enable Beam 2. Set coincident centers numerically, select Beam 2, drag, and verify Beam 1 remains fixed. Toggle coherence/phase and confirm their values survive dragging.
5. Test Escape, outside-aperture release, canvas leave, numeric edits, Beam 2 disable, and graphical off/on. Confirm cancelled gestures do not change controls and static fallback renders.
6. Rerun the same initialization cell: preserve the same controller/controls and one view. For deliberate replacement, call `demo.close()` first. Check stale old canvas callbacks cannot update controls.
7. If a completed snapshot is already available, confirm it and its export remain unchanged by edits. Do not run a simulation solely to validate this feature. Real-run cancellation transitions were covered synthetically; browser timing remains an acceptance limitation.

**Conclusion:** optional positioning coexists with existing static previews through explicit backend setup, separate canvas ownership and retired callback endpoints. Candidate is unstaged/uncommitted and ready for bounded local GUI testing; hosted qualification and any publication remain separate steps.
