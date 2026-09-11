# Implementation and Local Validation

**Status:** Reviewed
**Version:** 1.3
**Last reviewed:** 2026-09-11
**Prerequisites:** Approved architecture or implementation request
**Usual next prompt:** Development Self-Review, then Pre-Commit Review

---

## Purpose

Implement a bounded feature, bug fix, refactoring, or documentation
change within the local repository and validate the result before any
commit or remote execution.

The objective is to produce the smallest correct implementation while
preserving existing validated behavior.

---

## Required Inputs

- Expected Git SHA: `{{EXPECTED_GIT_SHA}}`
- Target branch: `{{TARGET_BRANCH}}`
- Implementation objective: `{{IMPLEMENTATION_OBJECTIVE}}`
- Intended candidate boundary: `{{INTENDED_CANDIDATE_BOUNDARY}}`
- Equations/model being changed or `N/A`: `{{EQUATIONS_MODEL_CHANGED_OR_NA}}`
- Expected scientific non-change areas: `{{SCIENTIFIC_NON_CHANGE_AREAS}}`
- Acceptance criteria: `{{ACCEPTANCE_CRITERIA}}`
- Validation commands: `{{VALIDATION_COMMANDS}}`
- Evidence/provenance strategy or `N/A`: `{{EVIDENCE_PROVENANCE_STRATEGY_OR_NA}}`
- Explicit exclusions: `{{EXPLICIT_EXCLUSIONS}}`
- Cluster/GPU access authorization: `{{CLUSTER_GPU_AUTHORIZATION}}`

Stop before acting if any required placeholder remains unresolved.

---

## Background

Current repository state:

Relevant branch:

Current Git SHA:

Related architecture documents:

Previous implementation:

Known limitations:

---

## Preflight

Before editing, resolve the preflight inputs above and record the actual
baseline SHA, branch, candidate boundary, non-change areas, tests, provenance
strategy, exclusions, and cluster/GPU authorization. Identify the equations or
model being changed, or record `N/A`. Confirm that HEAD matches the expected
baseline and that the boundary is unambiguous. If authoritative evidence is
planned, identify the clean immutable production checkout and any separately
checksummed harness before execution.

---

## References (optional)

The canonical lifecycle, authoritative-evidence, schema, self-review, and hard-
stop rules are defined in `docs/codex/README.md` under **Canonical Lifecycle
and Evidence Gates**. This prompt applies those shared rules.

List relevant:

- architecture documents;
- issue reports;
- design reviews;
- previous commits;
- validation reports;
- scientific notes.

---

## Authorization Matrix

| Action                          | Authorization          |
| ------------------------------- | ---------------------- |
| Local read-only inspection      | Yes                    |
| Local repository changes        | Yes, declared scope    |
| Local artifact writes/retrieval | Yes, declared scope    |
| Local test execution            | Yes                    |
| Commit                          | No                     |
| Push                            | No                     |
| Remote read-only commands       | No                     |
| Remote interactive file changes | No                     |
| Scheduler-created outputs       | No                     |
| Environment changes             | No                     |
| Job submission                  | No                     |
| Automatic retry                 | Not applicable         |
| Larger follow-on run            | No                     |

---

## Scope

### In Scope

Perform only the requested implementation.

Typical work includes:

- feature implementation;
- bug fixes;
- localized refactoring;
- documentation updates;
- test additions or updates;
- validation script updates.

### Out of Scope

Do **not**:

- redesign unrelated modules;
- broaden the implementation scope;
- modify public APIs unless requested;
- introduce unrelated refactoring;
- modify validated numerical algorithms without explicit approval.

---

## Acceptance Criteria

The task is complete only if:

### Implementation

- requested behavior is implemented;
- implementation is minimal;
- code is readable and maintainable;
- documentation is updated where appropriate.

### Validation

- requested validation commands succeed;
- no known regressions are introduced;
- modified functionality behaves as expected.

### Development Self-Review

- every applicable canonical Development self-review item is checked;
- every acceptance criterion has an executable test or identified retained
  evidence;
- the final candidate manifest and exclusions are exact;
- authoritative evidence, when produced, satisfies the canonical clean-source
  and manifest gates.

If validation cannot be completed, explain exactly why.

---

## Workflow

1. Record the compact preflight required by the canonical lifecycle gate.
2. Confirm the baseline and candidate boundary before editing.
3. Inspect the existing implementation and identify the smallest change.
4. Implement the change.
5. Run the requested local validation.
6. Complete the canonical Development self-review checklist.
7. Report the exact review boundary, exclusions, and retained evidence.
8. Stop for independent Pre-Commit Review.

Do not commit or push unless explicitly authorized.

---

## Validation

Run focused validation first. Then run the explicitly requested relevant
or full suite, and any broader validation warranted by the risk of the
change. Do not omit requested validation merely because focused tests
pass.

Examples:

```text
pytest tests/test_x.py

python scripts/checks/check_x.py

ruff check

mypy
```

Record:

- commands executed;
- pass/fail status;
- relevant numerical results;
- warnings.

If broader testing is appropriate, recommend it separately.

---

## Failure Policy

Ordinary in-scope development may use bounded
diagnose–fix–rerun cycles. When implementation or validation fails:

- diagnose the failure within the declared scope;
- make a directly supported in-scope correction;
- rerun the focused validation;
- preserve relevant evidence and commands;
- do not introduce speculative fixes or unrelated refactoring.

Stop and report when the failure persists, requires a scope expansion,
requires an architectural or public-API decision, requires an
environment change, or otherwise needs new authorization.

---

## Result Classification

### Passed

Implementation satisfies the requested objective, local validation passes, and
the Development self-review gate is complete.

### Failed

Implementation or validation demonstrates that the requested objective
was not achieved.

### Blocked

Implementation cannot proceed because of missing information,
dependencies, or external prerequisites.

### Inconclusive

Implementation completed but additional validation is required before
confidence is justified.

---

## Deliverables

Produce:

- modified source files;
- modified tests;
- modified documentation;
- validation results;
- concise implementation summary;
- recommended next step.

---

## Stop Conditions

Apply every canonical hard stop. In particular, stop before authoritative
evidence generation if production source is dirty, and stop on baseline drift,
an ambiguous candidate boundary, missing required units/provenance, unexpected
production changes in an analysis-only milestone, or an unresolved predecessor
gate.

Stop after:

- implementation;
- local validation;
- Development self-review;
- summary.

Do **not**:

- commit;
- push;
- begin follow-on work;
- perform remote execution;
- start cluster jobs.

After a `Passed` result, complete Pre-Commit Review and commit before beginning
a new milestone or non-exploratory scientific tangent, as required by the
canonical lifecycle gate.

---

## Required Report

Summarize:

### Preflight

- baseline SHA and branch;
- intended candidate boundary and exclusions;
- equations/model being changed, or `N/A`;
- scientific non-change areas;
- acceptance tests and evidence strategy;
- cluster/GPU authorization.

### Implementation

- files modified;
- major code changes;
- design decisions.

### Validation

- commands executed;
- tests run;
- numerical checks;
- warnings.

### Development Self-Review

- result of every applicable canonical self-review item;
- exact final candidate manifest and exclusions;
- authoritative-evidence provenance and schema checks, or `N/A`;
- preservation of unrelated dirty-tree content.

### Repository Status

- Git SHA;
- branch;
- modified files;
- untracked files.

### Recommendations

- ready for review;
- additional testing recommended;
- suggested next prompt.

Clearly distinguish:

- completed work;
- assumptions;
- unresolved issues;
- recommendations.

---

## Approval Gate

When the report identifies the completed implementation and exact review
scope, the canonical next authorization is **Approve review**.

Wait for explicit approval before:

- committing;
- pushing;
- broadening implementation scope;
- changing public APIs;
- beginning cluster execution;
- beginning research calculations.

---

### Prompt Notes

This prompt is intended for day-to-day development.

Prefer the smallest correct implementation.

Preserve existing validated behavior whenever practical.

Treat local validation as an integral part of implementation rather
than a separate activity.

When in doubt, stop with a clear explanation rather than extending the
scope of the requested work.
