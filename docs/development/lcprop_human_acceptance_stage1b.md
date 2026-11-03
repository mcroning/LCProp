# Stage 1B — Results and numerical transparency

## Scope and baseline

Baseline: `d0e48c5cfef0fdc1f13234269c3456074120d3f0`, following Stage 1A
`048075ac5e87f0dead3a5391008fb40b13f50970` and its separately committed
portable-workspace release-contract correction. The baseline full Product suite
was green: 1,706 passed, 76 skipped.

This bounded milestone implements HA-06, the remaining execution-presentation
portion of HA-07, and HA-12 from the
[canonical punch list](lcprop_human_acceptance_punch_list_2026-09.md).
The [user guide](../user/user_guide.md#results-and-numerical-transparency)
describes the new presentation.

## Changes

- **HA-06:** material-owned presentation reads retained completed outer history
  and actual request tolerances, without recomputing scientific state. Generic
  read-only Samples / Tables exposes existing diagnostic rows, including member
  status, budget/termination, and exact strict-gate values/tolerances/pass-fail.
  A returned nonconverged member remains distinct from execution failure and
  physical nonexistence. Transverse eigenpair optical qualification is separate
  from final polishing. Applicable curves show tolerance lines; overlap has no
  additive offset; widths use scientific x/y labels. Continuation explanation
  preserves seeding from previously converged members and unchanged strict gates.
- **HA-07:** shared execution summaries show selected target/resource/backend,
  precision and retrieval policy. NumPy on a GPU resource warns without changing
  the request. PR Run Planning separates configured execution from unmodified
  comparison estimates, and target/retrieval changes invalidate stale estimates.
- **HA-12:** shared session-only display scale state provides Auto, manual limits,
  and temporal locking. Linked same-source slices share fixed/locked limits;
  unrelated fields remain independent. Auto retains existing per-view policies.
  Finite fallback ranges handle flat/nonfinite images. The selected field and
  Results tab survive refreshes. Failed result rendering clears table content
  along with Stage 1A's other ownership safeguards.

## Scientific non-change and qualifications

No solver, equation, convergence predicate/tolerance/budget, mixing, continuation
policy, backend implementation, estimator calibration, result array, scientific
persistence schema, codec version, or retention policy is changed. Presentation
adds scalar diagnostic records and view state; it does not copy scientific
volumes into a new product. The GUI existence request remains a default
`SolitonRequest` inside `ParameterSweepRequest`; its settings are tested against
the actual builder and request type.

Unknown/missing qualification evidence is explicitly unavailable. An outer
history row describes completed outer work, not a subsequent final polish.
Auto intentionally retains its existing transverse/longitudinal differences;
manual/locked linked scales provide direct cross-view comparison. Display
settings are not saved in scientific persistence. No new live/movie capability
or export framework is introduced.

Remote continuation remains **DEFERRED / NOT AUTHORIZED**. Stage 1A Local-only
Continue, active fanning prompts, LaunchPlane, Research, and the authoritative
Product workspace are outside implementation scope. The active Stage 1B prompt
is retained only in the isolated worktree, outside the candidate manifest.

## Local validation

Development + Local Validation is complete; pre-commit review is pending.
Nothing is staged, committed, or integrated by this milestone.

Tests use the established Python 3.12.13 `lcprop` environment, isolated
`PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`, `QT_QPA_PLATFORM=offscreen`, and
`-p no:cacheprovider`. Matplotlib cache and validation logs reside outside the
Product repositories. CuPy is absent; existing GPU skips remain effective.
The full-suite command is `python -m pytest -q -p no:cacheprovider tests`.
Only the isolated tracked Product test boundary is collected. Two historical
untracked test modules in the authoritative workspace are preserved but not
copied into this worktree, so collection differs from the earlier checkout log.

- Initial focused Stage 1B run: **20 passed**.
- Final Stage 1B coverage: **26 focused cases**, all included in the final
  affected batch covering shared GUI, products, sweep/soliton, planning,
  remote-GUI mocks, documentation, window geometry, and TD-from-static views:
  **184 passed** (17.06 seconds).
- Final complete Product suite: **1723 passed, 77 skipped, 0 failed**
  (521.08s (0:08:41)).
- Python 3.12 in-memory syntax compilation for all changed/new Python files,
  changed-document local paths/anchors, candidate whitespace, and
  `git diff --check`: passed.

An early affected run exposed the old width-label assertion; only its expected
presentation wording and legend labels were updated. The first full run found
new scale-control vertical overflow in both existing 1200×760 layout cases,
plus a later Qt draw-callback error. Compact spacing inside the new controls
resolved the layout regression without changing the existing geometry tests.
The corrected intermediate full run passed (1,721 passed, 77 skipped), and the
final full run above also covers the subsequent pending-edit guard and legacy
sweep gate exposure. No failure is accepted as a baseline exception.

The final controls preserve user-edited pending limits through progress refreshes;
Apply remains explicit. Full finite volumes are not copied just to obtain display
bounds. Existing generic sweeps retaining results without member records still
expose their available qualification gates.

No requested Stage 1B item is blocked on a scientific change. No new scientific
or architecture extension is authorized. The qualifications above remain
explicit; remote continuation is untouched and deferred.
