# Colab interactive LC — Stage 1 minimal interface

**Implemented and validated locally; ready for local review and subsequent hosted
Colab interaction qualification. Stopped before hosted execution.** This is a small
presentation example, not a public API or frontend framework. No Product engine,
desktop GUI, physics, normalization, solver gates or persistence format changed.

## Candidate inventory

Four new unstaged files (SHA-256 in `candidate-inventory.json`):

- `notebooks/lc_static_interactive_cpu.ipynb`
- `notebooks/lc_static_interactive_cpu.md`
- `examples/lc_static_interactive.py`
- `tests/test_lc_static_interactive.py`

The unchanged committed `examples/lc_static_nonlinear_cpu.py` supplies the qualified
request, width and diagnostic helpers. Both helper scripts must be copied beside the
notebook; their exact hashes are checked by its setup cell. No source checkout is
required to execute: the notebook requires LCProp to import from site-/dist-packages.
No installation is performed automatically. Explicit setup uses the retained wheel
`1322400b31fac3a28d439f281cc189fb790f1526940dc36af9bf50971fdc5c20`, associated with
engine `52b00928978a9ec7862a357d38327fd0b9ae857c`. Current repository baseline is
`8c2e8b1f4bc4477787c2e686c38d4636180f3d75`.

## Scientific and interaction boundaries

The initial request equals the qualified Stage 4–5 128×128×100 nonlinear LC request:
75×100 µm aperture, L=500 µm, dz=5 µm; 1 mW beam at (-20,0) µm, 10 µm beam-normal
radii; original material/bias, wavelength, phase/group/orientation and float64 solver
settings. Original 0.005 RMS / 0.02 maximum residual gates and 120 s between-slice
operational limit remain. Modified beams are explicitly exploratory; successful
preflight is not a promise of convergence or scientific adequacy.

Numeric position, radii and power controls construct immutable dataclass replacements.
Edits call `build_launch` with the same OpticalLaunchContext, material.no reference
index, grid and complex128 dtype as canonical LC static execution. Preview intensity
uses Product `total_intensity` with authoritative coherence groups. There is no
frontend Gaussian normalization, refraction, FFT kernel or material algorithm.
Sampling a launch may perform Product preflight/diagnostic FFTs, but no optical
propagation or director solve occurs from a control edit. The preview is normalized
1/µm², not physical irradiance; a single beam's normalized preview need not change
when power changes. The physical power remains in the canonical material request.

Run explicitly invokes `LocalRunner.run_operation(LC_STATIC_OPERATION, request)`.
RunProgress updates the 100-slice bar and elapsed time. The synchronous UI has no
Stop guarantee; kernel/browser event scheduling during a solve remains a hosted
qualification concern. No custom JavaScript, drag canvas or background framework.
One run at a time, no parameter sweeps and no automatic solver execution in notebook
setup. All scientific arrays remain in the Python kernel; only small plots reach
browser output.

## Completed-result ownership and persistence

CompletedRun is a frozen dataclass containing **immutable serialized bytes/strings**:
encoded scientific request, selected NPZ arrays and diagnostic/environment JSON.
Accessors decode detached copies. The snapshot is created only after completed,
converged execution. Subsequent edits/loads do not rewrite it; failed requests leave
the previous completed result available with an explicit status. The UI plots a new
snapshot successfully before replacing previous completed ownership.

Plots show output intensity, final director reorientation relative to dark bias and
full RMS-width curves. Plot/export functions use only the snapshot. Exports regenerate
the figure from that same snapshot; saving before success fails. A new destination
folder is required; existing outputs cannot be silently overwritten. Files:
canonical `experiment.lcprop.json`, `arrays.npz` (nine arrays), `summary.json`,
`fields.png`, and `sha256.json`. These are selected diagnostics, not a restart
checkpoint or complete transport archive. Export does not call the solver.

Canonical LC experiment loading uses the existing public loader and preserves exact
supported floats without control rounding. This bounded interface rejects requests
outside its single-beam qualified grid/material/solver preset, rather than discarding
unknown scientific fields. Desktop LaunchPlane presentation metadata is not reproduced
or silently stripped. If its validation dependency is absent, the load reports a
bounded error and preserves the source file. Full Qt editor-state compatibility and
multibeam UI are intentionally deferred; canonical scientific experiment compatibility
is retained within the supported subset.

## Validation and execution evidence

A new disposable environment `/tmp/lc-interactive-stage1-env` was created with
`--system-site-packages` from the existing lcprop Python environment. The exact retained
wheel was installed non-editably into that disposable environment, alongside ipywidgets
8.1.9, nbclient/nbformat and ipykernel. NumPy/SciPy/Matplotlib were reused from the
existing numerical environment. Existing user environments were not installed into or
changed. Independently verified all **264 installed LCProp files** against the retained
wheel source inventory.

**8 new tests passed** (six main tests in 45.87 s, two additional tests in 0.96 s):

1. Exact typed request and canonical launch-field/intensity parity.
2. Real nonlinear execution versus direct canonical run_static: exact A_initial,
   A_final, theta, theta_bias, intensity_stack and full iteration/slice histories;
   truthful progress 1 through 100 and original convergence gates.
3. Snapshot immutability/detached access, actual plot/export paths, exact persisted
   request/arrays, repeat plot identity, manifest integrity and overwrite rejection.
4. Actual widget events (solver substituted only for UI state-machine coverage):
   no implicit run on editing, saved old result after draft changes, repeated Run,
   failure preservation and invalid-input status.
5. No save before execution; exact no-op loaded request, unsupported grid and zero
   total power rejection.
6. No Qt/LaunchPlane/CuPy imports in the tested process.
7. Actual notebook dependency-check cell gives a bounded missing-ipywidgets message.
8. Unsupported desktop scientific request rejected without changing draft/file.

**9 selected existing regression checks passed**, 52 deselected, in 0.81 s:
existing canonical experiment round trips and non-simulation qualified-example
request/baseline/width checks. No unexpected numerical/regression failures.

Commands (logs retained):

```sh
MPLCONFIGDIR=/tmp/lc-interactive-mpl PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /tmp/lc-interactive-stage1-env/bin/python -B -m pytest -q -p no:cacheprovider --import-mode=importlib tests/test_lc_static_interactive.py
MPLCONFIGDIR=/tmp/lc-interactive-mpl PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /tmp/lc-interactive-stage1-env/bin/python -B -m pytest -q -p no:cacheprovider --import-mode=importlib tests/test_lc_static_interactive.py -k 'missing_dependency or unsupported_desktop'
MPLCONFIGDIR=/tmp/lc-interactive-mpl PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg /tmp/lc-interactive-stage1-env/bin/python -B -m pytest -q -p no:cacheprovider --import-mode=importlib tests/test_lc_static_nonlinear_example.py tests/test_experiment_persistence.py -k 'initial_request_codecs_round_trip_exactly or fixed_baseline_changes_only_workflow or request_matches_existing_converged_regression or width_postprocessing_does_not_change_arrays'
```

The first logged main-test run contained six tests; the two additional tests were
added afterward and run separately. Thus the final suite has eight distinct passing
tests; no extra scientific execution was repeated just for aggregate reporting.

**Fresh local Jupyter kernel:** nbclient executed the actual notebook, with helpers
copied outside the repository and installed LCProp imports enforced. A test-only
import blocker rejects PySide/PyQt, LaunchPlane and CuPy. A test-only appended cell
clicked the actual Run and Export widgets. Complete notebook output is preserved as
`executed-local-notebook.ipynb`; the distributed notebook retains no execution outputs.
Kernel export manifest verified; figure visually inspected with correct labels/units.

- Nonlinear workflow including products/snapshot: **15.1654 s**.
- **100/100 slices converged**, RMS maximum 0.0049350853571153985 and residual maximum
  0.019985181430640875, unchanged gates.
- **All nine exported arrays bitwise identical to retained Stage 4 nonlinear arrays**,
  with matching shapes/dtypes. See `kernel-parity.json`; no new reference generation.
- Actual exported experiment, diagnostics and figure are bound by `kernel-result/sha256.json`.
- Two scientific runs in the shared parity fixture plus one in the clean kernel;
  no sweeps, GPU or hosted execution. Each is the original modest CPU configuration.

An initial collection attempt exposed an existing package import-order sensitivity
when importing the LC codec before persistence. The example now follows the already
qualified persistence-first import order. No Product change or scientific workaround
was made. The local kernel log also contains a TCP-transport warning and shutdown
KeyboardInterrupt message; notebook cells, export and independent comparison all
completed successfully. These do not qualify remote transport or a Stop feature.

## Memory, limitations and preservation

One completed snapshot is retained, not a frame history. Its NPZ bytes are ~26 MiB;
plotting/export decode temporary array copies, and solver results coexist briefly at
snapshot construction. Plan at least 1 GiB process headroom as in Stage 5; this task
did not benchmark peak memory. Widgets render 2-D images only. No scientific computation
is repeated to plot/export. Scalar progress includes Product-created bounded live
state objects transiently, but the frontend does not retain a sequence of them.

Actual hosted ipywidgets rendering, upload/download and event responsiveness still
need a separately authorized hosted interaction check. This is local validation only.
The exact qualified wheel is a retained binary artifact, not provided in Git; the
notebook gives its hash and explicit setup guidance. Arbitrary edited requests are
not newly qualified, and no stationary-soliton claim is made.

Python 3.10 grammar, notebook cell syntax, clean notebook outputs, candidate whitespace
and git diff --check passed. All previously tracked file hashes remain unchanged,
including both unrelated modifications. Existing Stage 4–5 evidence, bounded TD,
notebooks and source remain untouched. All candidates are new, unstaged and uncommitted.
No push, cluster/GPU activity or existing-user-environment installation.

**LCPROP COLAB INTERACTIVE STAGE 1 VALIDATED LOCALLY — READY FOR REVIEW.**
