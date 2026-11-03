# InteractiveLC Stage 1B — multibeam presentation

Date: 2026-10-15. Accepted base: `681f669b5ddb41d6bfd829856933d0ad0e23eb83`.

**Status: accepted for the Stage 1B presentation commit following local visual review.** The implementation-stage record below is retained; the final acceptance addendum supersedes its pending-review and uncommitted status.

## Exact scope

Changed:

- `src/lcprop/notebook/interactive_lc.py`: coherence layout, relative-phase label/tooltip, a snapshot-derived beam-description helper, completed-summary HTML and concise two-line plot titles.
- `docs/notebook_package.md`: explains presentation and grouping/provenance limits.

Added:

- `tests/test_interactive_lc_multibeam_presentation.py`: 11 presentation tests with synthetic completed fields and mocked execution.
- This report and `validation.json`: candidate hashes, test evidence, preservation and retained scientific identities.

The earlier local-launcher notebook/test and user-run notebooks were not edited. Both unrelated tracked PR modifications and protected evidence remain unchanged. No notebook bootstrap, package version, optional dependency, request schema or scientific support envelope was changed. Changes are unstaged and uncommitted.

## Presentation changes

### Beam coherence

A bold **Beam coherence** label and selector now appear above Beam 1/bias controls. The existing three options and underlying values remain exactly:

- Preserve saved groups (`preserve`, still the default).
- Coherent — same group (`coherent`).
- Incoherent — distinct groups (`incoherent`).

The dropdown is 420 px wide with a 100% container cap. Native dropdown selections cannot wrap reliably at arbitrarily narrow widths, so a normal wrapping HTML line also displays the complete selected option. This preserves readable selection text even when the native select becomes narrower. The label and live full-text line update without constructing or changing a scientific request.

### Relative phase

Beam 2's label is **Relative phase (rad)**. Its tooltip specifies Beam 2 minus Beam 1, each referenced at its beam-axis/entrance-plane intersection, and explains that interference is affected only within a common group. The control remains available when Beam 2 is included, in both coherent and incoherent modes. No values, angle conversions, phase wrapping, loaded-value handling or canonical phase conventions changed.

### Completed request identification

A separate **Completed run** summary appears above completed fields. It is assigned from the successfully completed snapshot at the same ownership transition as the result and is not updated by later edits, reset, load, or a failed run. Plot/export titles independently derive their description from that same snapshot request.

Descriptions resolve `BeamStack.coherence_groups`:

| Completed grouping | Description |
|---|---|
| One beam | Single beam; relative phase not applicable |
| All beams share a group | Coherent |
| All beams have distinct groups | Incoherent between groups |
| Some share groups and others do not | Mixed coherence groups |

The detail line identifies **Explicit saved groups** or **Legacy stack** and lists the resolved group assignment. With explicit groups it also labels the stored legacy stack flag and states that explicit groups take precedence. Beam 2 − Beam 1 phase is displayed to three significant figures. For distinct groups it explicitly says that this relative phase has no intensity effect between those groups. Formatting does not alter full-precision phase values in the request or export.

**Semantic limit / requested ambiguity:** a scientific request does not record whether the user clicked “Preserve saved groups” or explicitly selected a mode. Identical completed requests cannot be labeled with different editing histories without adding provenance. Therefore this change reports saved group identities and effective physics, not an invented selector history. True partially coherent/mixed grouping requires at least three channels; the generic description can accurately label such snapshot metadata, but the interface still admits only one or two beams. With two beams, different saved group names mean incoherent *between* groups, not a newly supported partially coherent model. No request/provenance schema was changed to resolve this ambiguity.

### Plots and diagnostics

Existing XZ/YZ optical-intensity and director-reorientation panels, two shared colorbars, axis orientation, nearest-coordinate selection, color limits and **Combined RMS width** labels remain unchanged. The figure title is now two lines: completed voltage/effective grouping/phase, then Beam 1's requested input coordinates used to select slices. Actual sampled coordinates remain in panel titles. Single-beam titles omit relative phase.

Group names and full detail stay out of plot titles to avoid excessive length; summary HTML escapes saved names. Displayed coherence/phase is not read from the live dropdown or widgets. Scientific diagnostics, serialized summary data, NPZ definitions and result ownership remain unchanged; only rendered summary/title text is added.

## Validation (no scientific execution)

| Check | Result |
|---|---|
| New presentation tests | 11 passed |
| Existing affected presentation/import/lifecycle tests | 19 passed; one pre-existing user-notebook-state failure |
| Stage 1 multibeam tests excluding the two propagation comparisons | 20 passed; 2 deselected |
| Total candidate checks | **50 passed; 1 baseline-identical failure; no new failures** |
| Committed baseline comparison for the failing lifecycle suite | 4 passed; identical 1 failure |
| Syntax / Python 3.10 grammar / scoped whitespace | Passed |

New tests cover selector position/labels/full-option text, live selection updates, actual visibility callbacks with synthetic preview data, phase accessibility, coherent/incoherent/single/mixed labels, legacy-group resolution, explicit groups overriding contradictory legacy flags, snapshot labels surviving edits/failure/reset, snapshot-bound exports, HTML escaping and unchanged plotted arrays/scales/coordinates/widths.

The mixed-group test uses synthetic three-channel request metadata only; it does not run a workflow, relax the one/two-beam guard or qualify three-beam execution. Test controller Run callbacks return synthetic snapshots, never call the solver. Existing non-propagation multibeam regressions include bounded canonical launch sampling, not material evolution or propagation simulations.

For the plot comparison, the accepted commit's `plot_completed` function was extracted via AST and evaluated against the same synthetic snapshot as the candidate. All seven axes, field arrays, color limits, coordinate limits/labels, per-panel titles and width curves match. Only the overall title differs. Exported scientific NPZ bytes and canonical experiment requests remain identical to the completed snapshots after draft edits.

### Unrelated test failure retained

`tests/test_local_notebook_lifecycle.py::test_notebook_has_one_initialization_and_no_saved_widget_output` expects an unexecuted `notebooks/lc_static_local.ipynb`. That user notebook already contained saved executed output at the start of this task. It fails identically using an isolated export of the accepted Stage 1 source and using this candidate. The original notebook's starting hash remains unchanged; its user-provided output was not removed to make the test pass. This failure is distinct from Stage 1's older Git-bootstrap dependency-test failure, which was not rerun here.

Logs:

- `.codex-work/interactive-beamplane-stage1b/final-tests.txt`
- `.codex-work/interactive-beamplane-stage1b/baseline-tests.txt`
- `.codex-work/interactive-beamplane-stage1b/multibeam-regressions.txt`

## Scientific identity and preservation evidence

AST comparison with accepted Stage 1 establishes that only the new display helper, `plot_completed`, controller initialization and the completed-summary assignment within `_run` changed. Request construction, bounded validation, launch preview, `execute`, environment reporting, CompletedRun, slice selection, longitudinal data extraction, export serialization, and controller request/load/reset/preview logic remain unchanged.

The Stage 1 test-log hashes were reverified. Its retained nine-array byte-identity comparisons for the qualified single-beam preset and the bounded two-channel incoherent case remain the scientific evidence; they were not rerun and are not new coherent nonlinear qualification. Their per-array/request hashes are copied into `validation.json` for reference. No new numerical parity claim is based on synthetic presentation fixtures.

The start-of-task preservation manifest confirms that only the intended controller and package guide changed among tracked and specifically recorded user files. All other source, dependency declarations, earlier reports, unrelated PR modifications, user notebooks and prior lifecycle work retain their starting bytes. New source/test/documentation identities are recorded in `validation.json`. The index remains empty.

## Visual evidence and remaining checks

Two synthetic completed-result figures were rendered with Matplotlib (not simulation data):

- `.codex-work/interactive-beamplane-stage1b/coherent-synthetic.png`
- `.codex-work/interactive-beamplane-stage1b/incoherent-synthetic.png`

The incoherent figure was visually inspected: its two-line title fits; the no-inter-group-effect qualifier is visible; four field panels and shared colorbars remain present; combined-width labeling is retained. Figure hashes are in `validation.json`. These are plot-rendering evidence, **not browser screenshots or scientific results**.

Browser layout remains to be checked manually in the local reusable launcher. After preserving any completed result and using a fresh kernel to load this source, inspect Beam coherence above Beam 1, select all three options at normal and narrow notebook widths, and confirm that the wrapping selection line and Relative phase label are readable. No scientific run is needed to inspect controls. Completed-label tests use synthetic snapshots; hosted/local real-result appearance has not been independently captured for this candidate. Do not change global browser configuration or reload a live controller module around in-memory results to test it.

**Conclusion:** presentation refinement is ready for local visual review. Canonical semantics and scientific implementation are unchanged. The absent selector-history provenance is explicitly reported rather than inferred or added to the schema.


## Final acceptance and commit review

The user supplied a local browser screenshot and accepted the updated interface. The screenshot shows Beam coherence prominently above Beam 1, the complete Preserve saved groups option and repeated full-text selection, enabled Beam 2 controls, Relative phase (rad), and effective groups A and A:beam-2. This is user-provided visual evidence, not an independently captured browser session. The displayed groups are distinct and therefore mutually incoherent; phase does not affect intensity between these groups. No screenshot file was added to Git. Narrow-width rendering and real completed-run summary appearance were not independently browser-qualified.

Before staging, all three implementation/test/documentation SHA-256 identities in `validation.json` and all three retained test-log hashes were verified unchanged. The retained result is **50 passed, one unchanged baseline failure**, with the identical saved-output lifecycle failure recorded against the accepted base. No simulations or new numerical comparisons were run for acceptance. Existing scientific parity retains its original scope. The local notebook's saved outputs are explicitly preserved.

The reviewed implementation changes only presentation. Effective groups come from canonical BeamStack resolution; phase is Beam 2 minus Beam 1 at their respective reference locations. The completed summary and plot titles decode the immutable completed request, not draft controls. Scientific request construction, normalization, solver settings, diagnostics, plot arrays/coordinates/scales and export serialization remain unchanged.

Explicit commit inventory (five files):

- `src/lcprop/notebook/interactive_lc.py`
- `tests/test_interactive_lc_multibeam_presentation.py`
- `docs/notebook_package.md`
- `results/Research/lcprop-interactive-beamplane-stage-1b-presentation-v1/REPORT.md`
- `results/Research/lcprop-interactive-beamplane-stage-1b-presentation-v1/validation.json`

Only this report is updated during acceptance, to record the user's visual review and commit scope. No local notebooks, scientific evidence arrays, screenshots, temporary outputs, unrelated PR modifications, dependencies or schemas belong to the commit. The commit message is `fix(notebook): improve multibeam coherence presentation`. No push or installation is authorized or performed.
