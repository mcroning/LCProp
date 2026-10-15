# InteractiveLC BeamPlane integration — Stage 1

Date: 2026-10-15. Source baseline: `89152f65e8565208f44cd5ea9c703d191482e112`.

## Implementation

The packaged `InteractiveLC` now supports one or two optical channels through the existing native models, launch functions, LC static operation and LocalRunner. No scientific engine, request schema, numerical tolerance, package dependency, version, notebook bootstrap, or published artifact was changed.

Candidate files:

- `src/lcprop/notebook/interactive_lc.py`: controls, native request construction, bounded one/two-beam validation, loading, and aggregate-result labels.
- `tests/test_interactive_lc_multibeam.py`: focused request, launch, geometry, persistence, ownership, import and numerical parity tests.
- `docs/notebook_package.md`: development-candidate usage and limitations; explicitly distinguishes these controls from published 0.3.0.
- This report and `validation.json`: validation and source identities.

All changes remain unstaged and uncommitted. The pre-existing dirty PR implementation and its test remain unchanged. No installation, GPU/Slurm work, commit or push was performed.

### Controls and scientific semantics

The qualified one-beam preset is unchanged. Existing position, radii, power and voltage controls remain, with numeric beam incidence, azimuth, profile roll, wavelength, external index and phase controls. Orientation angles are in radians to avoid degree-conversion changes on load. Radii retain the canonical beam-normal 1/e field / 1/e² intensity definition.

**Include Beam 2** adds a draft at x=20 µm, y=0, radii 10 µm and power 1 mW. This is an explicit exploratory configuration, not a newly qualified preset; Beam 1 power is not redistributed. Beam 2 has the same kinds of numeric controls, with phase expressed relative to Beam 1. Both phase references are at the respective beam-axis/entrance-plane intersections. Unchecking Beam 2 removes it from the scientific draft but retains its control values for re-enabling; zero power instead retains a zero-field channel.

The coherence selector offers preservation of saved groups, a shared explicit group, or distinct explicit groups. The UI shows the resolved effective groups. For newly added channels, distinct explicit groups are the normal default; all-legacy requests retain their legacy stack interpretation. Explicit-group semantics override the legacy stack coherence word. Loaded group identities remain exact in preserve mode. Changing the coherence selection is an explicit scientific edit.

LC's existing common-wavelength restriction remains enforced. A user must set both wavelengths consistently; one is never silently copied onto the other. Grazing incidence, invalid radii/powers, all-zero launch, partially migrated groups and sampling violations are rejected by existing validation. The preview calls `build_launch` with the material's reference index and actual grid, then `total_intensity` with the resolved groups. There is no new Gaussian formula, propagation, refraction, physical-power conversion, or interference renormalization.

The Stage 0 recommendation initially deferred orientation/wavelength editing. This implementation includes those controls as explicitly requested in Stage 1, using existing validation and marking modified configurations exploratory. The fixed grid, material and solver envelope remains unchanged.

### Execution and result ownership

Only Run invokes `execute`, which still delegates to `LC_STATIC_OPERATION` and LocalRunner. Initialization/editing performs launch sampling only. Voltage still causes fresh canonical bias construction on a new run. Progress and 120/180/300-second between-slice wall-time budgets are unchanged and independent of solver tolerances/iteration limits.

New controls are disabled while running. Snapshot replacement still requires completed, converged execution and successful plotting. Editing, loading, restoring the preset, invalid requests and timeout preserve the previous completed snapshot. Export regenerates its plots from that snapshot, not current controls.

Longitudinal cuts remain through the nearest sampled coordinates to **Beam 1 in the completed request**. Intensity is the retained combined slice-average optical intensity; director reorientation uses the shared retained director minus that run's bias director. Width axes now read **Combined RMS width**. These are not per-beam widths or per-beam reconstructed longitudinal fields. Visualization metadata records the slice-beam index/name and aggregate width meaning; scientific persistence schemas and NPZ array definitions are unchanged.

## Persistence

Canonical scientific-only one/two-beam static requests round-trip exactly within the fixed grid/material/solver envelope, including names/order, angles, wavelengths, power, phase and explicit or legacy groups. Relative phase is displayed without changing saved absolute phases: when phase controls remain unchanged, the original second-channel phase is used directly, avoiding subtraction/addition round-off. A regression uses a large first phase and small second phase to exercise this case.

The public loader still validates optional LaunchPlane editor metadata. Neither its validation nor its dependency requirements are bypassed. A headless environment lacking that validator receives an explicit error; source files, draft request and completed snapshot remain intact. Disabled desktop beams/editor-only state are not recreated by the notebook. Desktop representability is narrower than general native requests, so universal desktop GUI round trips are not claimed.

## Numerical validation scope

Two bounded deterministic configurations were compared against an independent call to canonical `lcprop.lc.workflows.static.run_static` in the same environment:

1. The exact unchanged qualified single-beam preset.
2. Two coincident incoherent channels at the preset position, each 0.5 mW with distinct groups, sharing the same qualified grid/material/solver settings. This exercises multiple channels and combined material driving without adding a parameter sweep or claiming a new separated-beam scientific demonstration.

All nine arrays are checked for exact value, dtype and **C-order byte identity**: `A_initial`, `A_final`, `theta`, `theta_bias`, `intensity_stack`, `widths_um`, `x_um`, `y_um`, `z_um`. Slice summaries and iteration histories also match exactly. Per-case array SHA-256 values, shapes, dtypes and scientific request hashes are retained in `validation.json`; the hash convention is SHA-256 of C-order array bytes, with shape/dtype recorded separately.

Coherent and incoherent launch arrays are additionally compared exactly with canonical launch construction for off-center elliptical, rolled and oblique beams. Phase tests verify constructive/destructive interference without renormalization and incoherent phase invariance within floating-point rounding. Explicit groups are tested against a contradictory legacy stack flag. No new coherent nonlinear-propagation qualification is claimed.

## Regression and preservation interpretation

The unchanged baseline was exported from committed HEAD into `.codex-work/interactive-beamplane-stage1/baseline/src`; no checkout or existing work was overwritten. Selected affected controller/presentation/package tests were run against both that source and the candidate. Both have the same pre-existing failure:

`tests/test_lc_static_interactive.py::test_notebook_missing_dependency_is_clear`

This test expects an older notebook's first-cell missing-dependency exception. The retained Git-bootstrap notebook instead reports a bootstrap retrieval failure and does not raise that expected exception in this environment. The historical notebook and test were not modified to hide the failure. Five older overlapping expensive parity/snapshot cases were deselected from this comparison; the new bounded parity and snapshot/export tests cover this candidate directly. No wider-suite pass is implied.

An initial test invocation used the shell's Python 3.9 and failed collection because Product requires Python >=3.10. Validation was then run with an existing disposable Python 3.12 environment; no packages were installed. Temporary test outputs are under `.codex-work/interactive-beamplane-stage1/`.

Tracked-file SHA-256 comparisons against the start-of-task snapshot establish that only the controller and package guide changed among previously tracked files. All other 895 tracked files, including 270 other tracked source files, remain identical to the starting worktree. In particular, both unrelated PR modifications retain their starting hashes. Existing untracked evidence was not edited. Tests used explicitly bounded temporary directories; no reference arrays were overwritten.

## Readiness and remaining limits

The source candidate is ready for local GUI review and preparation of a future hosted-Colab qualification build. It is **not installed, published, or hosted-qualified**. The published 0.3.0 wheel and immutable notebooks still provide their existing single-beam controller; merely opening them does not expose this candidate.

Hosted acceptance should check wrapping controls, explicit wavelength mismatch messages, phase/coherence feedback, repeated initialization, single interface display, Run-only execution, timeout ownership and exported snapshot provenance. A future authorized package build/version and deployment are necessary first. No scientific engine change is needed.

Execution remains synchronous; arbitrary exploratory two-beam powers, angles and interference patterns may fail sampling, convergence or the wall-time budget. No per-beam longitudinal reconstruction, dragging, general workflow support, GPU use, stationary-soliton claim or universal desktop editor-state compatibility was added.

## Final validation results

| Suite/check | Result |
|---|---|
| New multibeam tests | 22 passed (79.14 seconds), including both bounded canonical comparison pairs |
| Affected existing tests, candidate | 26 passed; 1 unchanged baseline failure; 5 overlapping cases deselected |
| Same selection, committed baseline | 26 passed; same 1 failure; same 5 deselected |
| Existing launch/sampling/physical resolver/LaunchPlane adapter/experiment persistence | 147 passed |
| Syntax, Python 3.10 grammar, scoped whitespace | Passed |
| Staged changes | None |

These suites provide 195 passing test cases and one known unchanged failure. Canonical adapter tests used the existing LaunchPlane model source without importing Qt; no dependency was installed. The environment and complete candidate hashes are in `validation.json`.

| Case | Completed slices | Max residual RMS | Max residual | Controller execution time |
|---|---|---|---|---|
| 1 beam(s) | 100/100 converged | 0.004935085357115399 | 0.01998518143064087 | 15.58 s |
| 2 beam(s) | 100/100 converged | 0.004935085357114844 | 0.01998518142737504 | 15.57 s |

Residual gates remain 0.005 RMS and 0.02 maximum. Timings are measurements of these bounded validation runs, not a runtime guarantee or a benchmark. Canonical comparison calls are additional work within the reported test-suite time.

Environment: Python 3.12.13, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2, ipywidgets 8.1.9, macOS arm64. Tests imported the candidate from `src` using PYTHONPATH; installed-package and hosted execution of this candidate are deferred.

Reproduction command (existing local test environment):

```sh
PYTHONPATH=src:/Users/mcroning/LaunchPane/src MPLBACKEND=Agg \
  .codex-work/pypi-stage2a/colab-env/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_interactive_lc_multibeam.py \
  --basetemp=.codex-work/interactive-beamplane-stage1/review-tests
```

The optional LaunchPlane source path is needed only for the two desktop-adapter identity tests. Core/controller import-isolation tests explicitly block Qt, LaunchPlane, CuPy and optional display imports in a subprocess.

## Stage 1 acceptance review — 2026-10-15

**Recommendation: GO for the scoped implementation commit.** No blocking correction was identified. This is acceptance of the reviewed source candidate, not authorization to publish a package or a claim of hosted multibeam qualification.

Method: fresh source/diff review and checksum verification of retained evidence. No tests, launch calculations, simulations, builds or installations were run in this acceptance review. Only this report was updated. Numerical conclusions below rely on the retained execution evidence and the actual assertions in the reviewed tests, not a newly generated reference.

### Exact candidate and preservation

HEAD remains `89152f65e8565208f44cd5ea9c703d191482e112`. All three candidate SHA-256 values match `validation.json`:

| File | SHA-256 | Reviewed scope |
|---|---|---|
| `src/lcprop/notebook/interactive_lc.py` | `2d467e928567bafbcdb236f1a961ea57b6256404f59d5d6d85d40a61886af5de` | 108 added / 26 removed lines: one/two-channel controls, request construction/guard, load/reset, effective groups, result labels and visualization metadata |
| `tests/test_interactive_lc_multibeam.py` | `1ec204fcb8869a95d0167ae81d32950217dba03353a14970ec202e2d1aa3a40d` | New focused test module, including independently constructed requests/fields, real widget events, synthetic failures and bounded canonical comparisons |
| `docs/notebook_package.md` | `30b72dca938653e3b685193164b48294c3e4321181cdcc6e05d57be4b68ed456` | 54 added lines explaining the development candidate and its limits |

The report and existing `validation.json` are associated evidence, separate from these three implementation files. There are no staged changes. Comparison with the start-of-implementation tracked-file manifest again shows only the controller and guide changed among previously tracked files. The two unrelated PR modifications match their recorded starting hashes. `pyproject.toml`, the shared preset/diagnostic helper and scientific engine received no changes from this candidate. No new mandatory or optional dependencies were introduced; BeamChannel/BeamStack are existing core imports and frontend imports remain deferred.

### Scientific correctness: accepted within the stated evidence scope

- **Single-beam identity:** request construction with default controls remains exactly the shared qualified preset. The unchanged `lc_static_demo.build_request` supplies the same grid, bias, material, beam, precision and solver options. Retained encoded request hash is `46448f7f3dec0af49faf053c8b94426c2ff8c4c6e772e66aba54a71440298ffc` (the sorted-JSON convention recorded by Stage 1). Its nine array hashes/shapes/dtypes agree with the original retained test metadata. Tests compare array bytes to a separate canonical workflow call and compare iteration/slice diagnostics.
- **Two-beam request and optical-field identity:** both coherent and incoherent cases are covered. `test_desktop_adapter_request_identity` independently constructs LaunchPlane definitions for shared/distinct groups and checks exact encoded native requests. `test_geometry_and_launch_parity` compares the preview fields and group-aware intensity to canonical construction for both modes, with unequal radii, offsets, roll and incidence. These are optical launch comparisons, not two full nonlinear propagation qualifications.
- **Coherence/phase:** the selector constructs explicit canonical groups, rather than trusting the legacy stack flag. Preserve mode retains saved explicit or all-legacy semantics. Relative phase is converted to the second channel's absolute axis-reference phase; unchanged loaded phases avoid floating-point subtraction/addition loss. Tests cover constructive/destructive interference, incoherent relative-phase invariance and explicit groups overriding a contradictory stack word.
- **Normalization:** `build_launch` retains incident power fractions and finite-grid sampling; the controller uses canonical `total_intensity` and performs no additional normalization. Zero power retains a channel; zero total incident power fails. Physical power and normalized intensity are not conflated.
- **Execution evidence:** the second complete propagation comparison is two coincident, distinct-group 0.5 mW channels, not a separated-beam or coherent nonlinear study. Its request hash is `055da1bb1f810cbe5d1c1aa6044245cdff5d3e07333d454c4918003128377db8`. Both retained propagation cases report 100/100 converged slices and nine byte-identical arrays against their respective canonical calls. The original test metadata agrees with the summarized acceptance metadata. Arbitrary coherent configurations remain exploratory.

### Snapshot/export and persistence compatibility: accepted with explicit bounds

CompletedRun remains a frozen serialized request/array/summary snapshot with detached access. Execute still routes through LC_STATIC_OPERATION/LocalRunner and checks completion/convergence. Edits do not execute the solver; timeout or failure does not replace the previous snapshot. The new controls join the existing busy/disable handling. Exports use the completed snapshot, including regenerated plots and checksum metadata, not the current draft.

Native one/two-beam files within the bounded envelope preserve names/order, group identities, geometry and phases on load. The large-first-phase regression explicitly exercises exact restoration of a small second absolute phase. Unsupported settings are validated before replacing controls. The public experiment loader is still used, including optional editor-payload validation. No source files or optional payloads are silently rewritten to pass validation.

Limitations are deliberate: notebook exports are scientific-only, not a lossless archive of disabled desktop beams/editor UI state; the selected NPZ is not a continuation checkpoint. LaunchPlane editor payloads require the existing validator, and desktop representability is a separate constraint. These limits are documented and do not block this bounded commit.

### Interface usability: suitable for local acceptance; hosted rendering pending

Source inspection finds useful distinctions: radians and micrometers are labeled; radii are defined as beam-normal field radii; Beam 2 phase is explicitly relative; resolved groups are visible; modified/multibeam drafts are marked exploratory; wavelength incompatibility is rejected rather than corrected silently. Combined RMS-width labels and the figure's Beam 1 slice-center wording avoid assigning total-field diagnostics to an individual beam.

No blocking misleading diagnostic was found. Non-blocking refinements for later work:

1. The UI says "Phase is referenced at each beam axis"; the guide gives the more precise entrance-plane intersection definition. A tooltip could put the full definition beside the control.
2. In incoherent mode relative phase remains editable although it cannot affect the combined intensity between distinct groups; an explanatory hint would help new users without changing semantics.
3. The generic load-error suffix recommends a scientific file without Qt editor metadata even for other load failures. The actual exception is retained, but a reason-specific suffix would be clearer. This behavior predates this candidate.

No browser screenshot or hosted rendering was independently inspected in this review. Wrapping and visibility are source/test-supported; readability in Colab remains a hosted acceptance task. Additional controls may require vertical scrolling. The development documentation correctly states that production 0.3.0 does not contain them.

### Baseline failure and evidence integrity

Recomputed hashes match all four retained logs (`final-tests.txt`, `baseline-tests.txt`, `regression-tests.txt`, `canonical-tests.txt`). The 22 new passes, 26 affected passes and 147 canonical-regression passes remain applicable to these exact candidate bytes. This review did not rerun them.

Both baseline and candidate logs show the same sole failure, `test_notebook_missing_dependency_is_clear`, at the same expected-exception assertion, with the same bootstrap retrieval diagnostic. The corresponding historical notebook/test are unchanged. It is unrelated to two-beam construction and is not a new candidate failure. Five older overlapping cases were deselected as disclosed; this is not a claim that the entire repository suite passes.

### Required actions before commit or hosted use

No source correction is required by this acceptance review. A later authorized commit should stage explicit paths for the three candidate files and approved evidence only, excluding both unrelated tracked modifications and existing unrelated artifacts. Preserve the documented scope of numerical qualification. A future authorized build/release and hosted review are still required to expose and qualify the new controller through a public Colab notebook. No commit, push, upload or scientific execution was performed here.
