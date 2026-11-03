# Interactive LC Stage 1B — controls and longitudinal visualization

**Implemented and validated locally; ready for hosted GUI review.** No hosted run,
scientific-engine edit, public API, commit or push. The four existing uncommitted
Stage 1 candidate files were updated in place; their prior versions are retained
under `baseline/`. Final SHA-256 identities are in `candidate-inventory.json`.

## Exact changed files

- `examples/lc_static_interactive.py`
- `notebooks/lc_static_interactive_cpu.ipynb`
- `notebooks/lc_static_interactive_cpu.md`
- `tests/test_lc_static_interactive.py`

The committed qualified nonlinear example and all Product source remain unchanged.

## Required data verified before implementation

The previous completed-run NPZ retains `intensity_stack` and `theta`, each
(100,128,128), with x/y arrays length 128 and z length 100. These are the existing
retained slice-average optical intensity and director angle on (z,x,y), not inferred
volumes. No new retention, replay, interpolation or scientific reconstruction was
needed. Missing/incorrectly shaped volumes now raise a specific presentation error.
Director views show theta, not a mislabeled reorientation approximation.

## Voltage and initialization

A sixth numeric control, **Applied voltage (V)**, starts at the qualified 0.9144 V.
Request creation replaces only BiasSpec.V_bias and the supported beam controls.
Existing codec/bias validation applies; frontend additionally rejects nonfinite
voltage. All other material/bias/solver/grid settings remain bounded to the preset.
Changes are labeled exploratory, not qualified parameter ranges. Restore qualified
preset resets the draft (including voltage and beam settings) without executing or
changing an existing completed snapshot.

Every new run has initial_theta=None and initial_A=None. It calls the existing
LC_STATIC_OPERATION/LocalRunner; canonical run_static invokes build_bias from the
new BiasSpec, rather than reusing the previous accepted illuminated director. A
focused test intercepts the actual workflow after its original build_bias computes
fresh fields at 0.9144 V and 1.0 V: both requests arrive correctly, fields differ,
and execution stops before optical propagation. This is an initialization test,
not a new qualification of nonlinear convergence at 1.0 V. No voltage sweep or
modified-voltage full solve was performed.

## Layout and advisories

Configure, Run and Results are separate vertical groups. Controls have full-width
inputs with labels above them, eliminating fixed description-width truncation.
Flex rows wrap; groups/outputs/images have 100% maximum width and zero minimum
width constraints. Images are responsive ipywidgets.Image objects, not fixed-width
inline canvases. Preview figure grew from 4×3 to 7×5 inches. Path/export controls
stack rather than forcing a wide row. No custom CSS/JavaScript or frontend framework.

Draft changes only call canonical launch validation/sampling. Existing sampling
warnings are escaped and visibly displayed, and are forwarded to completed-run
diagnostics. The preset's periodic-boundary aperture advisory is now both visible
and exported. Invalid changes clear the draft preview; previous completed results
remain separately owned. An unsupported `Layout(gap=...)` initially emitted a
traitlets deprecation warning; it was removed. Final layout checks are warning-free.

Actual browser width/scrollbar behavior is not measurable from a headless kernel
alone. Responsive widget traits and image sizing are tested; narrow-window hosted
Colab inspection remains required. No universal browser-layout claim is made.

## Longitudinal views and provenance

From the completed request's x0/y0, choose argmin(abs(coordinates-center)) independently
for x and y. Ties deterministically choose the first stored index. Render:

- intensity[:, :, iy] — optical XZ;
- intensity[:, ix, :] — optical YZ;
- theta[:, :, iy] — director XZ;
- theta[:, ix, :] — director YZ.

All use stored z coordinates horizontally and transverse x/y vertically. Transpose
is presentation-only. Titles show the actual selected transverse coordinates, not
the desired beam center. The preset selects ix=29, iy=63: x=-20.21484375 µm,
y=-0.390625 µm. z coordinates are retained slice midpoints. The width curves span
the row below all four views. Figure title records completed voltage and requested
beam center. Colorbars retain normalized optical-intensity and radian director units.

Immutable completed snapshot bytes/strings remain authoritative. Summary now includes
visualization metadata; exported `visualization.json` separately records voltage,
requested beam position, actual sampled positions, indices, axes and quantity identity.
The existing canonical request serialization is unchanged. The export manifest binds
this extra presentation metadata with request, arrays, diagnostics and regenerated
figure. Later edits, loads or preset restoration never alter completed plots/export.
Old compatible snapshots can derive slice metadata from their saved request/arrays.

## Validation results

**12 focused tests passed in 34.85 s**, including the existing Stage 1 tests and:

- Modified-voltage request, finite/nonnegative input rejection and actual fresh-bias
  construction through the canonical workflow, stopped before propagation.
- Coordinate-coded (3,4,5) unequal-dimensional retained volumes, off-center inputs,
  exact four-slice values, correct pcolormesh orientation/axes, actual-coordinate labels
  and explicit missing-volume rejection.
- Wrapped/full-width layout traits, warning visibility, voltage round trip and Restore
  qualified preset, with any scientific execution during edits wired to fail the test.
- Subsequent voltage/position edits leave snapshot request, saved voltage and slice
  metadata unchanged; advisories persist in diagnostics.
- Existing exact canonical scientific-array/history parity, accepted 1–100 progress,
  immutable export, out-of-order edit/save/repeated-run/failure tests, no-save-before-run,
  headless imports and missing-dependency messaging remain passing.

The first 12-test run had five deprecation warnings from the unsupported layout gap
keyword. After that presentation-only correction, **8 relevant non-simulation tests
passed in 1.76 s**, 4 deselected, without warnings. These overlap the 12; do not add
them as distinct tests. **9 selected existing request/persistence/width regressions
passed**, 52 deselected, in 0.60 s. No new failing test IDs or scientific failures.

Commands use the existing disposable installed-wheel environment:

```sh
MPLCONFIGDIR=/tmp/lc-interactive-mpl PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /tmp/lc-interactive-stage1-env/bin/python -B -m pytest -q -p no:cacheprovider --import-mode=importlib tests/test_lc_static_interactive.py
MPLCONFIGDIR=/tmp/lc-interactive-mpl PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /tmp/lc-interactive-stage1-env/bin/python -B -m pytest -q -p no:cacheprovider --import-mode=importlib tests/test_lc_static_interactive.py -k 'layout_warnings or offcenter or voltage_fresh or no_save or unsupported_desktop or missing_dependency or headless or request_and_launch'
MPLCONFIGDIR=/tmp/lc-interactive-mpl PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /tmp/lc-interactive-stage1-env/bin/python -B -m pytest -q -p no:cacheprovider --import-mode=importlib tests/test_lc_static_nonlinear_example.py tests/test_experiment_persistence.py -k 'initial_request_codecs_round_trip_exactly or fixed_baseline_changes_only_workflow or request_matches_existing_converged_regression or width_postprocessing_does_not_change_arrays'
```

### Clean-kernel run and final layout check

The actual notebook ran from copied helpers outside the checkout in a fresh local
Jupyter kernel, installed LCProp required, with Qt/LaunchPlane/CuPy imports blocked.
Actual Run and Export button callbacks completed:

- **100/100 slices converged**; unchanged RMS/max residuals
  0.0049350853571153985 / 0.019985181430640875.
- **15.4287 s**, including products and snapshot creation.
- All **nine scientific arrays bitwise identical** to retained Stage 4 nonlinear
  outputs; shapes/dtypes/hash comparisons in `kernel-parity.json`.
- Export manifest verified; actual new four-view-plus-width figure visually inspected.
- Original sampling advisory retained in summary; `visualization.json` verified.

This scientific run preceded only removal of the unsupported layout keyword. Its
original helper hash/evidence remains unchanged. The final source then passed a
separate fresh-kernel no-solver form/preview/advisory smoke check, preserved in
`executed-final-layout-notebook.ipynb`. No science rerun was needed for that single
layout-argument deletion. Actual run evidence is in `executed-local-notebook.ipynb`
and `kernel-result/`; logs preserve local kernel transport warnings. These are local
kernel tests, not hosted browser commissioning.

Python 3.10 grammar, notebook cell syntax, candidate whitespace and git diff --check
pass; distributable notebook outputs are empty and helper hashes updated.

## Remaining limitations and preservation

Synchronous execution still has no responsive Stop guarantee. The introductory UI
remains single-beam with bounded grid/material/solver options; voltage changes and
beam edits are exploratory. No promise of nonlinear convergence for arbitrary edits.
Full desktop LaunchPlane metadata remains unsupported without its model-validation
dependency; no silent stripping/migration. Selected exports are not restart archives.
Snapshots decode temporary array copies for plotting, as in Stage 1; no extra stored
scientific history. No new performance or memory benchmark.

No simulations beyond the bounded canonical comparison pair and single preset kernel
run; two voltage bias-initialization checks terminate before propagation. No sweeps,
GPU/Slurm/cluster access, installation, commit or push in this task. Existing disposable
environment was reused. All previously tracked hashes—including the two unrelated
modifications—are unchanged. Original Stage 1/hosted and Stage 4–5 protected evidence
is untouched. Candidate remains unstaged and uncommitted.

**STAGE 1B CONTROLS AND VISUALIZATION VALIDATED LOCALLY — READY FOR GUI REVIEW.**
