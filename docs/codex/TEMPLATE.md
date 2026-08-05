# <Prompt Title>

## Purpose

Briefly describe the objective of this task.

State the scientific or engineering motivation in one or two
sentences.

---

## Background

Current repository state.

Relevant branch:

Current Git SHA:

Previous milestone(s):

Relevant reports:

Relevant documentation:

Any previous validation or known limitations.

---

## References (optional)

List any architecture documents, reports, papers, previous prompts,
or Git commits that provide context for this task.

---

## Scope

### In Scope

List exactly what Codex is expected to modify.

For example:

- source files
- documentation
- tests
- cluster scripts

### Out of Scope

Explicitly list what must not change.

Examples:

- architecture
- unrelated modules
- public APIs
- existing validated workflows

unless required to fix a demonstrated bug.

---

## Acceptance Criteria

The task is complete only if:

- all requested functionality is implemented;
- requested tests pass;
- numerical behavior satisfies the stated tolerances;
- documentation is updated where appropriate;
- no unrelated regressions are introduced.

Include quantitative tolerances whenever practical.

---

## Workflow

Perform work in the following order.

1. Inspect the current implementation.
2. Implement the requested changes.
3. Run the requested validation.
4. Summarize the results.
5. Prepare (but do not perform) the next deployment step,
   unless explicitly authorized.

---

## Validation

Specify exactly which tests should be executed.

Examples:

```text
pytest ...

scripts/checks/...

GPU smoke test

Image amplification pilot
```

Validation should be objective and reproducible whenever practical.

Record numerical results whenever applicable.

---

## Cluster Instructions (optional)

If cluster work is required:

- update the cluster checkout;
- verify the repository is at the expected Git SHA before execution;
- preserve the exact Git SHA;
- use the approved Python environment;
- use scheduler-provided scratch storage (`$TMPDIR`) whenever
  available;
- copy only compact scientific products back to persistent
  storage;
- record provenance.

If cluster work is not required, state so explicitly.

---

## Deliverables

Specify exactly what should be produced.

Examples:

- modified source files;
- documentation;
- updated prompts or documentation;
- figures;
- metrics;
- provenance;
- Slurm script;
- report.

---

## Stop Conditions

Codex must stop after:

- implementation;
- validation;
- commit preparation;
- Slurm script preparation;
- pilot completion;

or another explicitly specified milestone.

Do **not** continue beyond this point without approval.

---

## Required Report

Summarize:

- files modified;
- validation performed;
- numerical results;
- Git SHA;
- repository status;
- remaining issues;
- recommended next step.

Include any warnings or unresolved questions.

---

## Approval Gate

Stop here and wait for explicit approval before:

- committing;
- pushing;
- submitting cluster jobs;
- running large parameter sweeps;
- changing architecture;
- beginning the next implementation or research stage.

---

### Prompt Notes

This template is intentionally conservative.

Favor reproducibility over speed.

Favor explicit validation over assumptions.

Favor stopping for approval rather than continuing into the
next stage automatically.