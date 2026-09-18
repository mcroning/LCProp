# Post-Stage-1 P1 residual polish

## Preflight and scope

Baseline: `dda75de9b8f90ed89b3639e4b9730681039a2fd5`, including Stage 1A–1C
and the committed [residual-action specification](lcprop_human_acceptance_post_stage1_residuals_2026-09.md).
Development branch: `feature/ha-p1-residual-polish`, in a disposable isolated
worktree. The authoritative Product branch is `feature/pr-second-order-static`.
The governing-equation/model change is **N/A**: this is shared presentation,
bounded LC/PR GUI text/widgets, regression tests, and minimal documentation.

Pre-development checks verified clean tracked Product/index at the baseline,
LaunchPlane at `add5f6bdbfcceaea7a8f2fd132e4ebf70b6eb5a8`, and clean Research main
at `b947fac51722eafa971e756a2cc2053be065294d`, equal to its local origin/main.
The Product historical inventory contains 4,332 untracked/ignored paths with
fingerprint `a7b3f02a2043fb7def986e43529bcef06cade8d2a114c6e838d7daa829f32b25`.
No older inventory is restored. Active operational prompts, logs, caches,
historical artifacts, and other repositories are outside the candidate.

Validation uses the established Python 3.12 `lcprop` environment, the isolated
`src` on PYTHONPATH, offscreen Qt, disabled bytecode and pytest cache, and an
external Matplotlib cache. Evidence consists of local software regressions and
source review, not scientific commissioning or new physical qualification.
Focused tests precede affected GUI/Product tests and the full tracked `tests/`
suite. No cluster, scheduler, CUDA, SSH, or remote execution is authorized.

## Residual dispositions

| P1 / canonical residual | Human observation and source finding | Correction and changed ownership | Coverage / remaining limitation |
| --- | --- | --- | --- |
| P1-01 / B1 | LC experiment description exposed `LocalRunner`; the sentence also fails to express selected-target capabilities. | LC `experiment_panel.py` now asks the user to choose an experiment and inspect request/execution target. Internal runner names are unchanged. | Focused text check; adjacent scientific/advanced text is retained. No global terminology replacement. |
| P1-02 / B2 | Numeric request sequences obscured ordinary result labels; IDs implement essential ownership. | Shared `workspace.py` shows lifecycle labels, retains IDs internally and in Request, and preserves previous-result and partial-failure handling. | New label/ownership tests plus unchanged transition logic and Stage 1A regressions; old assertions change only expected wording. |
| P1-03 / B3 | Existing timestamped boundaries lacked salience. Source used ordinary plain lines, visually similar to progress. | The same Workspace boundary method renders escaped, bold timestamp/operation text with a separator. | History order, one event per boundary, bold presentation and selected Console tab tested. No second boundary or logging system. Native perceptual acceptance remains a human check. |
| P1-04 / B4 | Raw floats and fixed decimal tails were widespread; tables and tolerance/limit labels expose those values directly. | Shared `number_format.py` distinguishes coordinate/tolerance/ordinary/overlap and exact text. Tables use exact tooltips; curves keep exact plot data. `numeric_widgets.py` uses concise resting text and the original decimal editor on focus, through shared helpers and PR material/timestep controls. | Decimal/scientific/coordinate/tolerance/overlap tests; unedited value round trips, focus, scientific entry and request equality. Stored precision, configured decimals, arrays and serialization are unchanged. |
| P1-05 / B5 | Longitudinal coordinates and selector shared one crowded horizontal row. | `longitudinal_pane.py` separates selector from coordinate/guides, formats coordinates, and uses compact spacing. | Portable row-separation checks and the existing default-window linked-slice/plot-boundary regressions. Native macOS clipping was not re-enacted. |
| P1-06 / B6, M1 | Limits were cramped; Auto/manual/locking were hard to distinguish. | Shared `display_scale.py` uses wrapping controls, explicit mode labels and Min/Max entries with exact tooltips. Unedited compact limits apply their exact retained value; explicit edits are parsed normally. | Existing scaling/array-isolation tests and new exact-range/edit tests. Auto remains the default: preferences are keyed by field identity across live/final views; automatic lifecycle-specific preference changes would need more design. |
| P1-07 / B7 | An absent longitudinal product during progress was described as unsupported. | Workspace supplies its existing current-state context to the longitudinal view; absent LC propagation products say not available yet during progress and not retained for a completed result. Stationary soliton workflows explicitly describe transverse-only results. Explicit capability/policy messages take precedence. | Progress/completed/explicit-unsupported and previous-result tests. No developing longitudinal data are generated. |
| P1-08 / B8 | Material steps in segment was opaque. | PR evolution label is Material time steps this run; tooltip distinguishes additional Continue steps. | Label/request-value tests. Nt, dt, integrator, request schema and Continue semantics unchanged. |
| P1-09 / B9 | LC TD showed an empty Samples/Tables view. The generic table consumes diagnostic rows; ordinary LC TD does not supply a retained table there. | `table_pane.py` explains absence and hides empty selector/grid while keeping the selected Results subtab. | Empty→populated→empty and partial-failure regressions. No new histories, metrics or artificial rows; those remain P2/P3. |
| P1-10 / B10 | The embedded profile choice was clipped. LaunchPlane owns the label and profile model. | Product Beam host supplies complete current-choice and item tooltips, preserving existing explanatory tooltips. | Full profile text, original angle guidance and bounded minimum-size checks; Stage 1C Beam/layout tests. No LaunchPlane modification. “Legacy” is retained: no independently qualified equivalent supports renaming it here. |
| P1-11 / C1–C2 | Exact gates existed but users had to discover tables and map members to powers. | Shared `convergence_summary.py` reads existing member/table/gate products; Workspace provides a primary member selector with power, solver flag and failed gates; Convergence details opens execution, termination, budget and exact failed values with the same result-ownership label. | Failed, unavailable, equality-at-strict-tolerance, execution-failure, post-polish divergence, selected-member and render-failure cases. It never recomputes qualification or invents a quality score. |
| P1-12 / C4 | Completed execution could be mistaken for convergence. | Shared operation status explicitly says Execution status; the member explanation separately labels solver convergence. | Summary/ownership regressions preserve the distinction. No second execution-status system. |
| P1-13 / H1–H2 | Corrected Help was still dense. | `help.py` provides headings, steps and bullets for Results, Slurm, and scattering; details remain in the User Guide. Guide changes cover new visible labels, summaries and exact-value access. | Help structure and scientific-semantic tests; guide links/anchors checked. Installed-package Slurm remains possible with valid deployment configuration. |
| P1-14 / G1 | A native chooser could select a foreign-material experiment. Source already rejects material mismatch before applying state and has actionable warnings. | **No code change.** Existing LC/PR load and open-action tests cover state/checkpoint preservation and application-specific guidance. | Run existing experiment GUI/persistence tests; no duplicate identity system or schema migration. Chooser selection alone is not a loading defect. |

## Boundaries and scientific non-change

No governing equation changed. No numerical solver/integrator changed. No
scientific default changed, including LC TD Nt=2 and PR scattering strength.
No schema/codec/checkpoint/retention contract changed. No optical propagation,
edge treatment, convergence predicate/tolerance/budget, mixing, backend numerical
implementation, resource-estimator calibration, scientific result array, or
LaunchPlane launch mathematics changed. Compact editors preserve the existing
configured decimal precision and return unchanged stored values when merely
reinterpreting resting display text.

PR canonical scattering equivalence remains on scientific hold and was not
investigated by this milestone. Help describes current controls and the hold;
it does not claim legacy/Photonics equivalence or authorize a default change.

Remote continuation remains **DEFERRED / NOT AUTHORIZED**. No remote-continuation
work occurred; Stage 1A Local-only Continue behavior is preserved by regression
coverage. No P2 live-results/far-field/multichannel capability was implemented.
Existing PR far-field coverage is not presented as a new P1 feature. Developing
longitudinal products, new histories, per-channel metrics, pump/signal roles,
scattering equivalence, integrator diagnostics, Gaussian unification, trajectory
overlays and vector LC eigensolitons remain separately scoped.

## Validation and development self-review

- Focused P1/layout/Stage 1C batch: **47 passed in 4.81s**, including 26 new P1 cases.
- Affected GUI/Product batch: **589 passed in 47.94s**. Coverage includes Stage 1A
  ownership/subtab/partial-failure, Stage 1B qualification/tolerance/overlap/
  execution/scaling, Stage 1C Beam/profile/layout, and existing cross-material
  open/rejection tests.
- Additional real offscreen focus/keyboard check: repeated focus cycles retained
  the exact spin-box value without value-change signals; deliberate entry of a
  shorter decimal was accepted as an edit. No scientific calculation involved.
- Complete tracked Product suite (`tests/`): **1,766 passed, 77 skipped in
  490.64s**. Zero failures; no accepted baseline exceptions.
- Python 3.12 syntax, Markdown links/anchors, whitespace (including new files)
  and `git diff --check`: passed. The final 19-file manifest is recorded outside
  the repository; all protected commits and the 4,332-path historical inventory
  retain their pre-development identities.

Commands use `python -m pytest -p no:cacheprovider` with `PYTHONPATH=src`,
`PYTHONDONTWRITEBYTECODE=1`, `QT_QPA_PLATFORM=offscreen` and an external
`MPLCONFIGDIR`. The focused batch is `tests/test_residual_polish_p1.py`,
`tests/test_results_fields_layout.py`, `tests/test_stage1c_new_user.py`, and
`tests/test_workspace.py`.
The affected batch additionally covers transparency, material GUI/Results,
experiment GUI/persistence, Beam, profiles, remote-dispatch mocks, Help and
Product documentation. Full-suite discovery is restricted to `tests/`.

Initial development validation caught a longitudinal layout-height regression
and a Help-format assertion. A subsequent mixed Qt batch crashed after failed
assertions; bounded focused reruns isolate corrections. These were development
failures, not accepted baseline exceptions. Self-review also found that an
expanded inline convergence explanation crowded the soliton image. The first
full-suite attempt was deliberately interrupted (577 passed, 1 skipped at that
point) before correcting that issue. A compact primary selector and owned
modeless details window now preserve the Fields area; the added regression
checks their combined layout and ownership lifecycle. A second preliminary
full-suite run was stopped to correct stationary-soliton progress wording;
that workflow must not imply future longitudinal products. The final focused
and affected batches include both corrections. These interrupted attempts
are not counted as complete-suite validation. The next complete run found one
new presentation regression (1 failed, 1,765 passed, 77 skipped): standalone
Workspace previews acquired a synthetic Request 0 prefix. Request numbering is
now shown only after a request has begun; focused coverage explicitly checks
both cases. This is corrected here, not accepted as a baseline exception. An affected-suite
rerun then exposed premature Qt widget deletion in the new scale-control row
(1 failed, 588 passed). Installing the wrapping layout before adding temporary
label/input groups gives Qt immediate ownership. The layout regression now
forces garbage collection and checks each control parent and height calculation.
The final focused, affected and complete runs above include both fixes and
pass without failures.

Development self-review confirms that all 14 P1 items have an implementation
or an explicit source-qualified disposition. Stage 1A ownership/subtab and
partial-render safeguards, Stage 1B qualification/scaling/backend presentation,
and Stage 1C responsive/profile/Beam behavior pass their affected regressions.
No new scientific or architectural capability was introduced. Native visual
acceptance and the explicitly deferred P2/P3 capabilities remain outside this
local software-validation result.

No staging, commit, integration, push, prompt archival or worktree removal is
part of this milestone. The exact final candidate manifest and validation logs
are retained outside the Product tree for the next strictly read-only review.
