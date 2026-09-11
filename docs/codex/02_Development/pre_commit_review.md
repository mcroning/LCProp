# Pre-Commit Review

**Status:** Reviewed
**Version:** 1.3
**Last reviewed:** 2026-09-11
**Prerequisites:** Completed Implementation, Local Validation, and Development Self-Review
**Usual next prompt:** N/A (next authorization: Approve commit)

---

## Purpose

Perform an independent technical review of a completed local
implementation before any commit is authorized.

The objective is to determine whether the implementation is technically
sound, remains within its approved architecture and scope, has adequate
validation evidence, and is ready for a bounded commit-approval stage.

This is a review-only prompt. It is not an implementation, architecture,
commissioning, or commit-authorization prompt.

---

## Required Inputs

- Expected Git SHA: `{{EXPECTED_GIT_SHA}}`
- Target branch: `{{TARGET_BRANCH}}`
- Approved implementation objective: `{{IMPLEMENTATION_OBJECTIVE}}`
- Approved implementation scope: `{{IMPLEMENTATION_SCOPE}}`
- Files and changes to review: `{{REVIEW_SCOPE}}`
- Acceptance criteria: `{{ACCEPTANCE_CRITERIA}}`
- Validation commands: `{{VALIDATION_COMMANDS}}`
- Expected validation evidence: `{{VALIDATION_EVIDENCE}}`
- Applicable commissioning evidence or `N/A`: `{{COMMISSIONING_EVIDENCE_OR_NA}}`
- Proposed commit scope: `{{PROPOSED_COMMIT_SCOPE}}`
- Proposed commit message: `{{PROPOSED_COMMIT_MESSAGE}}`

Every required field must contain an approved value, a conspicuous
`{{PLACEHOLDER}}`, or `N/A`. Blank required fields are unresolved.

**Stop before acting if any required placeholder remains unresolved.**

---

## Background

Current repository state:

Relevant branch:

Current Git SHA:

Approved architecture or implementation request:

Previous implementation and validation report:

Known limitations:

Expected follow-on work:

---

## References (optional)

The canonical lifecycle, authoritative-evidence, schema, self-review, and hard-
stop rules are defined in `docs/codex/README.md` under **Canonical Lifecycle
and Evidence Gates**. Review their application; do not duplicate them here.

List relevant architecture documents, implementation requests, design
decisions, issue reports, previous commits, validation records,
scientific references, and documentation.

---

## Authorization Matrix

| Action                          | Authorization  |
| ------------------------------- | -------------- |
| Local read-only inspection      | Yes            |
| Local repository changes        | No             |
| Local artifact writes/retrieval | No             |
| Local test execution            | No             |
| Commit                          | No             |
| Push                            | No             |
| Remote read-only commands       | No             |
| Remote interactive file changes | No             |
| Scheduler-created outputs       | No             |
| Environment changes             | No             |
| Job submission                  | No             |
| Automatic retry                 | Not applicable |
| Larger follow-on run            | No             |

Local read-only inspection includes repository inspection of the working
tree, staged state, history, diffs, status, and existing validation or
commissioning artifacts. It does not authorize a new validation cycle.

Authorization applies only to the scope and exact inputs declared in
this prompt. Anything not explicitly authorized is prohibited.

---

## Scope

### In Scope

Review the complete implementation and its supporting evidence,
including:

- every modified, added, deleted, renamed, and staged file;
- architectural compliance with the approved design;
- compliance with the approved implementation scope;
- public and internal API compatibility;
- separation of production changes from test, documentation, validation,
  and harness changes;
- numerical formulations, units, signs, axes, precision, and backend
  behavior where applicable;
- convergence and acceptance criteria;
- diagnostic completeness and accuracy;
- reported validation evidence and reproducible commands;
- remote commissioning evidence where applicable;
- numerical-equivalence evidence where applicable;
- provenance and checksums where applicable;
- compliance with the canonical authoritative-evidence manifest and schema
  gates;
- explicit separation of physical-model, discretization, resolution,
  normalization, and approximation-regime effects in scientific comparisons;
- completion of every canonical Development self-review item;
- regression coverage, controls, and important missing cases;
- source and user documentation updates;
- exact repository status, including unrelated and untracked files;
- generated artifacts and results that must not be committed;
- the exact proposed commit contents and exclusions;
- proposed staging commands and commit message.

Inspect the actual implementation and evidence. Do not rely only on an
implementation summary.

### Out of Scope

Do **not**:

- modify source, test, documentation, configuration, or generated files;
- implement fixes or recommendations;
- redesign the approved architecture;
- expand the approved feature scope;
- commit or push;
- access remote systems or the cluster;
- prepare or submit scheduler jobs;
- begin commissioning or research work.

---

## Acceptance Criteria

The review is complete only if it determines, with concrete evidence:

### Architecture and Scope

- whether the implementation follows the approved architectural
  boundaries;
- whether all changes are necessary and within scope;
- whether unrelated behavior or modules changed;
- whether new complexity, coupling, or duplicated logic is justified.

### Implementation

- whether APIs and established behavior remain compatible unless an
  approved change says otherwise;
- whether numerical algorithms and physical mappings are implemented
  correctly;
- whether convergence requires the declared residuals, tolerances, and
  independent checks rather than weak proxy criteria;
- whether diagnostics and provenance accurately describe actual
  execution;
- whether error handling, termination paths, and backend behavior are
  explicit and safe.

### Validation

- whether the reported commands were actually run and their results are
  accurately stated;
- whether focused tests, regression tests, controls, and relevant suites
  adequately cover the risks;
- whether applicable commissioning and numerical-equivalence evidence is
  complete and bound to the reviewed source identity;
- whether quantitative tolerances are declared and justified;
- whether skipped or unavailable validation is identified without
  overstating support;
- whether documentation matches the implemented behavior and known
  limitations.

Routine omissions covered by Development self-review are readiness defects,
not work to repair during this review. Formal review independently confirms the
completed gate and returns the milestone to Development if an item is missing.

### Commit Readiness

- whether the proposed commit contains every intended file and excludes
  unrelated files;
- whether generated artifacts, results, private data, and pre-existing
  dirty-tree content are excluded;
- whether the repository status and base Git SHA are recorded exactly;
- whether the proposed staging commands select only the reviewed files;
- whether the proposed commit message accurately describes the bounded
  milestone;
- whether the next lifecycle stage should be commit approval, further
  local work, or another review.

---

## Workflow

1. Resolve every required input and predecessor lifecycle gate.
2. Confirm the current branch, baseline Git SHA, and complete repository status.
3. Inspect the complete working-tree and staged diff, including untracked
   files in the declared review scope.
4. Compare the implementation with the approved architecture, objective,
   scope, and acceptance criteria.
5. Review APIs, numerical behavior, convergence logic, diagnostics,
   documentation, and failure paths.
6. Audit tests and validation evidence against the implementation risks.
7. Review the declared local validation and applicable commissioning
   evidence without rerunning tests or remote work.
8. Separate findings into blocking defects, nonblocking recommendations,
   and future work.
9. Identify the exact proposed commit contents and commit message.
10. Apply the task-specific result classification.
11. Produce the required report.
12. Stop at the approval gate.

If the review reveals a recurring defect class not covered adequately by the
reusable standards, record a post-milestone standards update recommendation.
Do not amend the standards during this read-only review.

Do not correct defects during this stage. A required correction returns
the work to Development and Local Validation under separate authorization.

---

## Validation

Inspect at minimum:

```text
git status --short --branch

git diff --check
```

Inspect both unstaged and staged changes and account for untracked files.
Review the recorded output from `{{VALIDATION_COMMANDS}}` and any
applicable commissioning evidence. Do not rerun tests, simulations, or
cluster work under this prompt.

Record:

- every review command actually executed;
- the validation and commissioning commands represented by the existing
  evidence;
- pass, fail, and skip counts;
- relevant warnings;
- numerical measurements and their tolerances;
- unvalidated platforms, backends, or configurations;
- any discrepancy between reported and reproduced evidence.

Missing, stale, or failed validation is evidence for the review. It does
not authorize a fix, test change, tolerance change, or rerun.

---

## Cluster Instructions

Cluster work is not required or authorized.

Do not access remote hosts, alter a remote checkout, inspect a cluster
environment, prepare a scheduler script, or submit a job.

---

## Failure Policy

Read-only diagnosis and additional bounded local inspection are
authorized when review evidence is incomplete or contradictory.

Do not modify files, environments, tests, tolerances, or implementation
parameters. Do not retry a failing validation after any change. Preserve
the observed evidence and classify the result as Not Ready or
Inconclusive, as appropriate.

Stop and report if the review cannot be completed because required inputs,
files, dependencies, or validation evidence are unavailable.

---

## Result Classification

This prompt intentionally replaces the generic `TEMPLATE.md` result
vocabulary with the following task-specific review classification. Use
exactly one classification.

### Ready for Commit

The implementation satisfies its approved architecture, scope, and
acceptance criteria; required validation and Development self-review pass; the
canonical evidence gates are satisfied where applicable; and no blocking
defect remains.

### Not Ready

One or more blocking implementation, validation, scope, compatibility,
documentation, or commit-content defects must be corrected before commit
approval.

### Blocked

The review cannot proceed because a required input, candidate artifact,
repository state, dependency, or validation record is unavailable.

### Inconclusive

Available evidence is insufficient to determine commit readiness.

---

## Deliverables

Produce:

- architectural assessment;
- implementation assessment;
- validation assessment;
- blocking issues;
- nonblocking recommendations;
- future work, kept separate from current-scope findings;
- exact repository status;
- exact proposed commit contents;
- exact proposed exclusions and staging commands;
- proposed commit message;
- recommendation for the next lifecycle stage.

No modified implementation or generated repository artifacts are
deliverables of this review-only stage.

---

## Stop Conditions

Apply the canonical hard stops. An unresolved predecessor gate, baseline drift,
ambiguous candidate boundary, or missing evidence units/provenance prevents a
`Ready for Commit` result.

Stop after producing the pre-commit review report.

Do **not**:

- modify files;
- commit;
- push;
- prepare Slurm jobs;
- access the cluster or any remote system;
- begin commissioning, research, or follow-on implementation.

---

## Required Report

Summarize:

### Classification

- exactly one task-specific review classification;
- concise justification.

### Architectural Assessment

- compliance with the approved architecture;
- boundaries, dependencies, and scope observations.

### Implementation Assessment

- API compatibility;
- numerical correctness;
- convergence and diagnostics;
- error handling and backend behavior;
- documentation accuracy.

### Validation Assessment

- commands actually executed;
- reviewed results, warnings, and skips;
- commissioning and numerical-equivalence evidence where applicable;
- regression coverage;
- remaining unvalidated claims.

### Findings

- blocking defects;
- nonblocking recommendations;
- future work.

### Repository and Commit Scope

- branch and exact Git SHA;
- exact `git status --short --branch` output;
- complete intended commit file list;
- unrelated or excluded files;
- exact proposed staging commands;
- proposed commit message.

### Recommendation

- commit approval;
- return to Development and Local Validation;
- or additional read-only evidence required.

Clearly distinguish verified facts, recorded measurements,
interpretation, and recommendations.

---

## Approval Context

Resolve the next lifecycle authorization in the final review report.

```text
Next approval phrase: Approve commit or N/A
Authorized target: {{EXACT_REVIEWED_COMMIT_SCOPE_OR_NA}}
Immutable identifier: {{REVIEWED_BASE_GIT_SHA_OR_NA}}
Action authorized on approval: Commit only the reviewed files with the reviewed message, or N/A
Stopping point: After reporting the commit SHA and repository status, or N/A
```

Use `Approve commit` only for a `Ready for Commit` result with an exact
file manifest, staging commands, commit message, and unchanged repository
context. Otherwise every field must be `N/A`. Approval is single-use and
does not authorize modification, push, or any adjacent lifecycle stage.

---

## Approval Gate

When the review classifies an exact file manifest and commit message as
`Ready for Commit`, the canonical next authorization is
**Approve commit**.

Stop here and wait for explicit authorization before:

- modifying any file;
- implementing a correction;
- committing;
- pushing;
- accessing a remote system;
- preparing or submitting a scheduler job;
- beginning the next lifecycle stage.

---

### Prompt Notes

This prompt provides the independent technical gate between local
implementation and commit approval.

Blocking defects prevent commit readiness. Nonblocking recommendations
may be reported without changing a `Ready for Commit` classification.
Future work is outside the approved commit scope and must not be used to
delay an otherwise ready implementation.

Any required source, test, documentation, or harness correction returns
the candidate to **Implementation and Local Validation** under separate
authorization. Do not fix it during Pre-Commit Review.
