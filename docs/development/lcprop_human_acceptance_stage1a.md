# Human acceptance Stage 1A — request, run, and result transparency

## Scope and baseline

This bounded Development + Local Validation candidate starts from LCProp
`36049c38e2b5d8757997e7e185f5550c7061c69b`, following the
[canonical September punch list](lcprop_human_acceptance_punch_list_2026-09.md).
It implements HA-01 through HA-05, the request-preview portion of HA-07, and
local source preflight from HA-08. It is prepared in an isolated worktree on
`feature/ha-stage1a-transparency`; it does not integrate or publish changes.

Equations/model changes: **none**. Scientific evidence generation: **N/A**.
Validation here is software regression testing, not new scientific evidence.
No cluster, scheduler, CUDA, or remote execution is authorized or used.

## Implementation and acceptance mapping

| Item | Change and validation boundary |
| --- | --- |
| HA-01 | PR input labels distinguish general beams / two-beam coupling from specialized image amplification. Screen, role, and symmetric x-z carrier validation is retained. Regressions construct, plan, and persist screen-free asymmetric general requests across all four GUI workflows, and reject invalid specialized requests. |
| HA-02 | Shared Results banners retain displayed-request identity separately from the requested/preview configuration. Starting an attempt marks existing fields and curves as Previous run before validation. Existing LC live products identify current accepted/progress states; returned results and stopped states retain their owner. Full original field names remain in Diagnostics; view labels omit “Authoritative”. |
| HA-03 | Ordinary refresh never selects another Results subtab. Regression coverage includes all five subtabs and verifies scientific array identity. |
| HA-04 | Shared operation text acknowledges validation/preparation, starting, running, stopping, and outcomes. Timestamped Console boundaries identify Run, Continue, experiment/checkpoint actions, and remote dispatch. Errors expose their cause above Results and retain full tracebacks. Existing workers, cancellation, and retention semantics are preserved. |
| HA-05 | Inspect Request uses existing request builders and summaries, without dispatching a worker. LC transverse refinement checks its exactly-one-channel capability before execution, without restricting ordinary propagation or fixed-point solitons. Existing float64 settings are unchanged. |
| HA-07, bounded | Preview reports target, requested backend, precision, and actual selected cluster/resource. Runtime backend resolution remains explicitly unresolved when device validation is needed; LC's NumPy execution is identified. Resource comparison estimates are not presented as the configured plan. No backend selection semantics change. |
| HA-08, bounded | The standard Slurm runner offers a local-only automatic-source preflight using the existing source resolver. GUI preview and execution invoke it before staging/submission. Non-checkout installations receive clean-checkout or existing pre-staged-route guidance. Explicit pre-staged routes remain supported. |

The shared implementation is in [Workspace](../../src/lcprop/gui/workspace.py)
and [request transparency](../../src/lcprop/gui/request_transparency.py), with
workflow-specific request/capability checks in the LC and PR windows.
[User guidance](../user/user_guide.md#inspecting-requests-and-identifying-displayed-results)
explains the controls and ownership states.

## Qualifications and non-change review

- Request capture commits pending beam-editor values and updates optical context.
  Existing specialized image validation may prepare raster/launch inputs. Preview
  performs no propagation, relaxation, workflow dispatch, remote staging, or
  device probe. It is a request snapshot, not a guarantee of runtime success.
- “Completed result” identifies a returned execution product, not numerical
  convergence. Current accepted/progress states appear only for existing payloads;
  no new PR live products or retained histories are introduced.
- Reduced PR material derivatives remain x-only, with y as a batch axis. General
  beams do not imply rotational equivalence of arbitrary crossing orientations.
- Source preflight checks local automatic-deployment eligibility only. Remote
  connectivity, runtime packages, and devices remain execution-time concerns.
  Custom/injected runners retain their source contracts; a runner without the
  optional preflight method is not represented as having passed that check.
- Public request constructors, positional APIs, scientific defaults, equations,
  convergence gates, persistence schemas, transport codecs, retention policy,
  and scientific product keys/data remain unchanged. The Results API gains only
  an optional keyword for ownership state; existing positional calls remain valid.
- No Stage 1B/1C or later-stage work, scientific metric design, scattering
  extension, profile management, LaunchPlane change, Research change, or prompt
  archival is included. There is no new private Research dependency.

## Local validation

Authoritative local interpreter: established `lcprop` environment, Python
3.12.13. Tests use the isolated candidate through `PYTHONPATH=src`,
`PYTHONDONTWRITEBYTECODE=1`, and `QT_QPA_PLATFORM=offscreen`. CuPy is absent;
GPU-dependent tests use their existing skips.

Focused commands:

```sh
python -m pytest -q tests/test_human_acceptance_transparency.py
python -m pytest -q tests/test_workspace.py tests/test_gui_*.py tests/test_pr_gui_*.py tests/test_experiment_gui.py tests/test_remote_gui_execution.py tests/test_source_deployment.py
python -m pytest -q tests/test_pr_gui_lifecycle.py
python -m pytest -q
```

- New Stage 1A regressions: **21 passed** (16.84 seconds).
- Affected GUI, persistence, remote-GUI, workspace, and source-deployment batch:
  **273 passed, 1 failed** (112.88 seconds). The failure was the existing
  subprocess lifecycle test's 20-second timeout for `td_close`; no functional
  assertion failed. An isolated rerun of all five lifecycle cases passed:
  **5 passed** (79.14 seconds). The timeout is reported rather than hidden.
- Final full Product suite: **1 failed, 1686 passed, 77 skipped** in
  484.37 seconds. All Stage 1A and lifecycle regressions passed in this run.
  The sole failure is the unchanged
  `test_release_installation_and_attribution_contracts_are_explicit` at
  `tests/test_product_documentation.py:77`: it asserts that
  `src/lcprop/core/LCProp.code-workspace` does not exist. Required baseline
  parent `0ba38d4e175f51f4eaa40ecc4d6712f8f7eea854` is explicitly titled
  “Restore portable LCProp workspace” and tracks that file. The failing test,
  workspace file, README, packaging metadata, and notices are byte-unchanged
  from the required baseline. This is a pre-existing baseline/test mismatch;
  neither removing the portable workspace nor changing this unrelated release
  test belongs to Stage 1A.
- Candidate syntax compilation, whitespace scan, local Markdown file links and
  fenced-code checks, and `git diff --check`: passed.

The earlier focused pass also exposed a result-display observer that accepts
only the existing positional argument. The final window call sites retain
that signature; ownership status is updated separately after the result is
presented. This preserves existing observer compatibility.

Self-review confirms the candidate is limited to GUI presentation/validation,
an additive local source-preflight method, focused regressions, and current
documentation. Product, LaunchPlane, and Research contents remain at their
protected identities, including pre-existing untracked/ignored files. The
milestone does not stage, commit, push, integrate, or archive any prompt.
Independent read-only pre-commit review remains the next gate.

## Pre-commit review remediation

The first independent review returned the candidate to Development for exactly
two blockers. The remediation retains the original Stage 1A scope and baseline.

1. **Result ownership across populated-to-empty transitions and render failures.**
   A PR static product has no curves; the old TD plot could remain visible under
   the new completed-request banner. Field and curve browsers now hide views
   with no applicable current product. Shared Results suppress painting during
   updates and publish ownership only after all panes accept the new product.
   If any update raises, fields, longitudinal views, curves, diagnostics, and
   movie access become unavailable, and the banner reports no displayed result.
   Request and timestamped Console history remain available. A later successful
   update restores the views. The selected Results subtab and scientific arrays
   are preserved. Regressions cover populated-to-empty transitions on all five
   tabs, failure after each of the three pane updates and diagnostic formatting,
   navigation after failure, and recovery.
2. **Local-only PR checkpoint continuation.**
   Continue now performs scientific/checkpoint validation without selected-target
   Slurm validation or source preflight. Its summary and worker dispatch messages
   use an explicit Local runner; the button tooltip explains this behavior.
   The ordinary Run target and backend selection are unchanged. No remote
   continuation is introduced. A dispatch-level regression selects Slurm with
   a non-checkout source, verifies that automatic deployment would reject that
   source, and then exercises the real worker and continuation adapter through
   the Local scientific-call boundary. The scientific function is replaced by
   a sentinel exception; no scientific calculation is needed for this check.
   Slurm validation/deployment calls are forbidden for Continue, and the test
   separately verifies that ordinary Run still invokes source preflight.

Relative to the rejected ten-file candidate, the two additional source files are
[CurvePane](../../src/lcprop/gui/views/curve_pane.py) and
[ImagePane](../../src/lcprop/gui/views/image_pane.py). The shared workspace and
request-summary helper, PR window, focused tests, user guide, and this record
are updated. All other original candidate files are unchanged.

Remediation validation uses the same Python 3.12.13 environment and
`PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`, `QT_QPA_PLATFORM=offscreen` settings.

```sh
python -m pytest -q tests/test_human_acceptance_transparency.py
python -m pytest -q tests/test_human_acceptance_transparency.py tests/test_workspace.py tests/test_gui_*.py tests/test_pr_gui_*.py tests/test_experiment_gui.py tests/test_remote_gui_execution.py tests/test_source_deployment.py
python -m pytest -q
```

- Focused regressions: **31 passed** in 18.49 seconds.
- Combined focused/affected batch: **287 passed** in 94.02 seconds, with no
  lifecycle timeout or other failure.
- Complete Product suite: **1 failed, 1696 passed, 77 skipped** in 494.34
  seconds. The only failure is the unchanged baseline workspace-documentation
  assertion at `tests/test_product_documentation.py:77`. All Stage 1A, remediation,
  affected-functionality, and lifecycle regressions passed; no other failure is
  accepted or present.
- In-memory Python syntax compilation, candidate whitespace checks, Markdown
  fences/local file links, and `git diff --check`: passed.

The focused standalone run preceded a docstring-formatting correction and the
movement of ownership-state preparation inside the guarded render block. The
combined affected run includes those final source changes and all 31 focused
regressions. The complete run validates the same source.

## Review disposition

Both review blockers are closed by the bounded source corrections and passing
regressions. Development and Local Validation are complete under the explicitly
authorized single-baseline-failure exception. Independent strictly read-only
pre-commit review remains required; this is not commit approval. No stage beyond
1A has begun.

The documentation assertion at `tests/test_product_documentation.py:77` is an
independently verified **pre-existing baseline inconsistency**, not a Stage 1A
regression. Baseline parent `0ba38d4e175f51f4eaa40ecc4d6712f8f7eea854` intentionally
restored `src/lcprop/core/LCProp.code-workspace`; the test still requires absence.
Both files remain byte-identical to baseline and outside this candidate. The
stale test belongs to a separately bounded release-contract maintenance change
after Stage 1A. Only that established baseline failure is allowed; no new or
other failure is accepted.

Scientific equations, arrays, result generation, defaults, convergence criteria,
request/persistence/transport schemas, backend-selection semantics, and retention
remain unchanged. No CUDA, remote execution, Slurm submission, or new scientific
evidence run is performed. Product, LaunchPane, and Research remain protected;
no prompt is archived and nothing is staged, committed, integrated, or pushed.
