# Implementation and Local Validation

**Status:** Reviewed
**Version:** 1.1
**Last reviewed:** 2026-08-05
**Prerequisites:** Approved architecture or implementation request
**Usual next prompt:** Pre-Commit Review (planned)

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
- Acceptance criteria: `{{ACCEPTANCE_CRITERIA}}`
- Validation commands: `{{VALIDATION_COMMANDS}}`

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

## References (optional)

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

If validation cannot be completed, explain exactly why.

---

## Workflow

1. Inspect the existing implementation.
2. Understand the requested change.
3. Identify the smallest correct modification.
4. Implement the change.
5. Run the requested local validation.
6. Review modified files.
7. Summarize results.
8. Stop.

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

Implementation satisfies the requested objective and local validation
passes.

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

Stop after:

- implementation;
- local validation;
- summary.

Do **not**:

- commit;
- push;
- begin follow-on work;
- perform remote execution;
- start cluster jobs.

---

## Required Report

Summarize:

### Implementation

- files modified;
- major code changes;
- design decisions.

### Validation

- commands executed;
- tests run;
- numerical checks;
- warnings.

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
