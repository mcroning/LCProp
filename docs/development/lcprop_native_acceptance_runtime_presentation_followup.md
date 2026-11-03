# Native acceptance follow-up: stable runtime presentation

## Preflight and boundary

Baseline: `bd1c195bceec0f870f018765068f3cc845a25482` on authoritative branch
`feature/pr-second-order-static`. Development uses isolated branch
`feature/native-runtime-presentation`. The authoritative worktree is unchanged.
This follows the [post-Stage-1 residuals](lcprop_human_acceptance_post_stage1_residuals_2026-09.md)
and [P1 polish](lcprop_human_acceptance_residual_polish_p1.md).

Equations/model change: **N/A**. The candidate is restricted to shared GUI
presentation, two application headers, existing scale-contract tests, focused
regressions, minimal user guidance, and this record. Existing CPU regression
tests provide software-validation evidence only; no new scientific calculation,
commissioning, scheduler, SSH, or CUDA execution is performed.

The expected historical inventory was 4,332 paths. Before this task it already
contained **4,335 paths**, fingerprint
`6f3a025fdadf35ea89819966da875bd4f97fb227c84a1902e412607821a01d64`.
The actual inventory (paths, modes, sizes, modification times and content hashes)
is retained externally for comparison. Nothing is restored, deleted or normalized.
LaunchPlane remains at `add5f6bdbfcceaea7a8f2fd132e4ebf70b6eb5a8`; Research
remains at `b947fac51722eafa971e756a2cc2053be065294d`.

## Corrections and evidence

| Item | Human observation and source finding | Implementation | Acceptance evidence |
| --- | --- | --- | --- |
| Console separators | Native PR showed rules after routine messages. Workspace inserted an HTML rule/header, then used QTextEdit.append for subsequent messages, allowing block formatting to carry forward. | Shared Workspace appends explicit plain-text blocks: timestamped bold boundaries have modest spacing; ordinary logs have normal weight and zero block margins. No HTML rule can propagate; literal angle brackets and multiline messages are retained. Selection and the reader's scroll position are preserved; a reader at the end continues to follow new output. | Focused test inspects each QTextDocument block, timestamps, weights, margins, message order, absence of rules, preserved text selection, and selected Console tab. |
| Application identity | PR identified itself as LCProp PR, while LC used only LCProp. | LC title and header now say LCProp LC, alongside the existing Execution control. PR identity remains LCProp PR. | Both application titles and header labels checked. No package, entry-point, request or API rename. |
| Display-scale linkage | Shared Workspace supplied one DisplayScales object to both transverse and longitudinal panes, so source-related keys coupled all three cuts. | Workspace now owns separate transverse and longitudinal scale stores. The existing longitudinal pane still applies one scale to both x-z/y-z views; both groups reset locks for a fresh run. | Bidirectional manual-limit independence, independent locks across updates, fresh-run reset, linked x-z/y-z limits and array equality. Stage 1B assertions requiring the superseded cross-pane linkage are updated; unrelated Auto, invalid-entry, quantity isolation and scientific serialization checks remain. |
| Runtime layout stability | LC/PR runner labels, PR checkpoint/status text, remote availability, and Results operation/ownership labels wrapped or changed preferred size as text grew. | Shared RuntimeStatusLabel paints a single elided plain-text line with text-independent width/height hints. Full text remains in text(), tooltip and accessibility name. Applicable status labels use it without changing their state transitions. | At 1200- and 1450-pixel widths, representative preparation/progress/checkpoint/Continue/stop/completion/failure/Slurm text leaves label hints and designated control/tab geometries unchanged. Existing 1200×760 plot-fit regressions also pass. |

Additional jitter sources found: Run/Stop caption widths change during execution,
Stop appears/disappears, and the Results material-time/propagation indicator is
shown or hidden. Header buttons reserve their largest lifecycle caption; Stop
retains its space while hidden. The indicator uses the same single-line policy
and retains its layout space. These affect presentation only, not button
availability, dispatch, checkpoint compatibility, or progress delivery.

Runner messages reserve a bounded 160-pixel preferred width; longer details are
elided with full tooltip/accessibility text. PR status reserves 280 pixels and
remote availability 300. These hints remain constant across text changes but
allow the existing responsive layout to wrap controls when window width changes.
Fonts and plot minimum sizes are unchanged. User-driven workflow/configuration
changes may still change applicable controls; runtime text alone must not do so.

Initial focused validation caught two existing plot-fit regressions caused by
an unnecessarily wide reserved runner region adding a header row at 1200 pixels.
The final bounded runner width fixes that cause. A trial Workspace-margin change
was reverted. Those development failures are not baseline exceptions.

## Validation and self-review

Environment: established Python 3.12, isolated `src` on PYTHONPATH, offscreen Qt,
bytecode and pytest cache disabled, external Matplotlib cache. All commands use
`python -m pytest -p no:cacheprovider`; full discovery is restricted to `tests/`.

- Focused: **80 passed in 5.82s**. Modules: native-runtime presentation, P1 polish,
  Stage 1B, Results Fields layout, and Stage 1C.
- Affected: **597 passed in 47.72s**.
- Complete Product suite (`tests/`): **1,774 passed, 77 skipped in 478.58s**;
  zero failures and no accepted baseline exceptions.
- Python 3.12 in-memory syntax compilation, documentation links/anchors,
  whitespace including new files, and `git diff --check`: passed.
- Candidate scope: exactly ten files; index empty. All protected repository
  identities and the actual 4,335-path historical inventory remain unchanged.

The affected batch includes Stage 1A ownership/subtab/partial-failure,
Stage 1B exact gates/overlap/numeric/scaling, P1 numeric/convergence presentation,
Stage 1C profile/Beam/layout, LC/PR workflow GUI, mocked remote dispatch,
experiment persistence, Help and Product documentation. Final logs and the exact
candidate manifest are retained outside Product. Native macOS subjective visual
acceptance remains a separate human check; offscreen geometry is not a claim of
native visual qualification.

Development self-review confirms every requested correction has focused
coverage; the final full suite includes the corrected layout and Console
selection behavior. Only the explicitly superseded transverse/longitudinal
linkage assertions changed in existing tests. No public positional API,
scientific units, qualification criteria, persistence or transport contract
changed. Native subjective acceptance remains pending human evaluation.
Authoritative numerical evidence/schema gates are N/A: this is software
validation, not scientific commissioning. No unrelated hunks, prompts, logs,
caches or historical artifacts belong to the candidate.

## Scientific non-change and exclusions

No scientific arrays, equations, solvers, integrators, defaults, backend behavior,
request schemas, codecs, checkpoints, retention rules or persistence state change.
Scale preferences remain session-only. The user guide explicitly describes the
new transverse/longitudinal independence and continued x-z/y-z linkage.

No PR live-results, developing longitudinal products, far-field expansion,
multichannel diagnostics, N-beam/image-screen design, generic gain or Image
Amplification semantics are implemented. LaunchPlane is unchanged. PR canonical
scattering equivalence remains **SCIENTIFIC HOLD / UNESTABLISHED**; no scattering
or fanning investigation occurs. Remote continuation remains **DEFERRED / NOT
AUTHORIZED** and is untouched.

No stage, commit, integration, push, worktree cleanup or prompt archival occurs
in Development + Local Validation. The next gate is strictly read-only
pre-commit review of the completed candidate and retained validation evidence.
