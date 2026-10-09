# Interactive Colab Stage 1C — presentation refinement

## Scope and outcome

Presentation implemented on the existing uncommitted Stage 1B candidate. No engine,
request schema, numerical settings, solver or scientific-array changes. No simulation,
installation, cluster/GPU operation, commit or push. Baseline HEAD:
`8c2e8b1f4bc4477787c2e686c38d4636180f3d75`.

Changed candidate files (exact hashes in `candidate-inventory.json`):

- `examples/lc_static_interactive.py`
- `notebooks/lc_static_interactive_cpu.ipynb`
- `notebooks/lc_static_interactive_cpu.md`
- `tests/test_lc_static_interactive.py`
- `tests/test_lc_static_interactive_presentation.py` (new)

The notebook delegates presentation and execution to the existing example helper;
its pinned helper hash is updated. No new public API or framework.

## Presentation

The launch figure is 3.75 inches square and displayed at 375 CSS pixels, including
axes and colorbar. XY uses equal physical aspect. Images shrink to available width.
Path fields are 300 px; buttons 140–170 px; progress 240 px. Controls wrap. Configure
can collapse and closes after a successful run, leaving Results space. Editing or
restoring controls does not run the scientific solver.

Results use a 9 by 6.2 inch figure capped at 900 CSS pixels: intensity XZ/YZ above
reorientation XZ/YZ, with z horizontal. Each row shares a normalization and colorbar.
Intensity is nonnegative; reorientation uses a symmetric signed range. Colorbar
labels use mathtext and ticks about three significant figures. Sampled coordinates
are shown to two decimals, while metadata retains full precision. A compact width
plot is below. See `fields.png`, rendered from the retained qualified snapshot.

## Corresponding bias and scientific identity

The Stage 1B completed snapshot stores `theta_bias` (128 by 128), alongside the
100 by 128 by 128 director and intensity volumes. Its bias array matches the retained
qualified Stage 4 reference hash exactly. The existing execution path obtains it
from `result.theta_bias` for the executed request, rather than from current controls.
The plotting path requires the baseline, correct shape and finite values, and rejects
missing data. It computes only the selected slices:

- XZ: `theta[:, :, iy] - theta_bias[:, iy][None, :]`
- YZ: `theta[:, ix, :] - theta_bias[ix, :][None, :]`

No bias solve, optical replay or full-volume subtraction is performed. Slice indices
come from the completed request and retained coordinates. For the preset they are
ix=29, iy=63, x=-20.21484375 µm, y=-0.390625 µm.

All nine retained scientific arrays were independently rehashed and match the Stage
1B/Stage 4 reference identities exactly (`retained-parity.json`). The retained
request equals canonical `build_request()`. The exported request decodes identically
and exported NPZ bytes are unchanged. This verifies presentation/export preservation;
new solver parity runs were not necessary and were not performed. Stage 1B's retained
canonical parity and 100-slice convergence evidence remain the numerical authority.

## Snapshot and export validation

Both inline display and export call `plot_completed` on the immutable completed-run
snapshot. Synthetic regression tests change position and voltage after completion,
restore controls and simulate a failed run; the old snapshot remains owned unchanged.
The exported PNG equals a separately rendered PNG from that same snapshot at the
same export DPI, and its arrays remain exact. Full-precision selection metadata is
exported separately. Retained historical summary metadata is preserved as recorded;
new visualization metadata specifies reorientation and the retained bias baseline.

## Tests and checks

**12 passed, 5 deselected** (`tests.txt`). The deselected existing tests execute
scientific workflows; their retained Stage 1B results are unchanged. Selected tests
cover canonical request/launch construction, headless dependencies, voltage request
validation, coordinates and unavailable products, snapshot ownership, exact plotted
slice values, shared norms, widget widths, square PNG shape, and export identity.
An initial new-test failure used `pytest.fail` to simulate a solver error, which
bypasses ordinary Exception handling; the test now injects ValueError for that case.
No Product correction was required.

Command (existing disposable environment, no installation):

```sh
MPLCONFIGDIR="$PWD/.codex-work/stage1c/mpl" PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
 /tmp/lc-interactive-stage1-env/bin/python -B -m pytest -q -p no:cacheprovider \
 --import-mode=importlib --basetemp=.codex-work/stage1c/pytest \
 tests/test_lc_static_interactive_presentation.py tests/test_lc_static_interactive.py \
 -k 'not scientific_parity_and_progress and not snapshot_immutable_and_export and not ui_stale_order_failure_and_repeated_run and not completed_voltage_and_slices_are_snapshot_owned and not voltage_fresh_bias_initialization'
```

Python 3.10 grammar checks passed for helper/tests and notebook Python cells (the
existing `%matplotlib inline` IPython magic excluded from Python parsing). Candidate
whitespace and `git diff --check` passed. All 848 previously tracked file hashes,
including both unrelated modifications, remain unchanged. Index remains empty.
Prior evidence and user inputs were read only. Temporary exports are under
`.codex-work/stage1c/`; existing unrelated untracked work was not modified.

## Remaining limitations

Matplotlib output was visually inspected locally; responsive widget properties were
tested, not a live hosted browser. Colab widget rendering, zoom, and viewport height
still need hands-on acceptance. Narrow/short windows may require vertical scrolling;
no fixed browser dimensions or guaranteed single-viewport fit. Result snapshots
retain existing detached decoding behavior; this is not a memory-architecture change.
Synchronous execution and existing exploratory-configuration restrictions remain.
No new physical qualification or stationary-soliton claim.

**STAGE 1C PRESENTATION VALIDATED LOCALLY — READY FOR HOSTED GUI REVIEW.**

## Follow-up: selectable wall-time budget and truthful timeout reporting

The same five-file candidate now offers 120/180/300-second elapsed wall-time choices
in Run, default 120. The choice is not part of `controls`, scientific requests or
experiment serialization, and does not invalidate completed ownership. It is disabled
while running. `_run` forwards it explicitly to the existing `execute` helper.
The helper validates finite positive budgets no greater than 300 seconds before work.
The notebook helper checksum and usage documentation have been updated.

At each completed-slice callback, one elapsed measurement is taken from the helper's
monotonic wall clock and sent in the progress event before checking `elapsed > limit`.
Equality is permitted. Timeout includes elapsed seconds, last completed slice/total,
configured seconds and the exact statement:
**Time budget exhausted; convergence not assessed.**
The previously completed snapshot and displayed output are preserved; export still
uses that snapshot. No partial timed-out run is promoted to a completed result.

This remains a cooperative between-slice limit, not a hard process deadline or CPU
utilization quota. Initialization and individual slices may exceed the limit before
a callback; launch-preview validation precedes timing, and post-run product/export
work is not interrupted by this callback check. Scientific convergence, tolerances,
iteration limits and algorithm selection are unchanged. Completed slice counts are
not claims of convergence. The hosted 2 mW run has not been repeated or qualified.

Validation: **21 passed, 5 deselected**, same focused command above with basetemp
`.codex-work/stage1c-budget/pytest`; new log `budget-tests.txt`. New synthetic-clock
tests cover equality and first exceeded boundary at all three budgets, progress-before-
exception ordering, invalid budgets (301, infinity, NaN, zero, negative), and propagation
of the selection. Widget tests verify unchanged canonical request identity, no execution
on budget change, preserved previous displayed output, snapshot and exported NPZ bytes
after timeout. No sleeps or scientific executions. Existing simulation-dependent tests
remain deselected; no new engine parity claim is based on mocks. Retained nine-array
hashes were rechecked unchanged. Grammar, whitespace, empty index and all 848 tracked
file identities were rechecked. Updated `candidate-inventory.json` binds this revision.

Hosted Colab rendering of the new control and a longer exploratory run still require
manual acceptance. No guarantee of 2 mW convergence or completion within 300 seconds.
