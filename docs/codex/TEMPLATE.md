# <Prompt Title>

<!--
Prompt metadata should be updated whenever this procedure changes
substantially.
-->

**Status:** Draft

**Version:** 1.1

**Last reviewed:** `{{LAST_REVIEWED_DATE}}`

**Prerequisites:** `{{PREREQUISITES_OR_NA}}`

**Usual next prompt:** `{{USUAL_NEXT_PROMPT_OR_NA}}`

---

## Purpose

Briefly describe the objective of this task.

State the scientific or engineering motivation in one or two
sentences.

---

## Required Inputs

Resolve every required input before execution.

Use conspicuous placeholders while preparing the prompt.

- Expected Git SHA: `{{EXPECTED_GIT_SHA}}`
- Target branch: `{{TARGET_BRANCH}}`
- Target host or cluster: `{{TARGET_HOST}}`
- Python environment or executable: `{{PYTHON_ENV}}`
- Exact test, script, or payload: `{{PAYLOAD}}`
- Numerical tolerances: `{{TOLERANCES}}`
- Input checksums: `{{INPUT_CHECKSUMS}}`
- Persistent output location: `{{OUTPUT_LOCATION}}`

Add or remove fields as appropriate for the task.

Every required field must contain an approved value, a conspicuous
`{{PLACEHOLDER}}`, or `N/A`. Blank required fields are unresolved.

**Stop before acting if any required placeholder remains unresolved.**

---

## Background

Current repository state:

Relevant branch:

Current Git SHA:

Previous milestone(s):

Relevant reports:

Relevant documentation:

Previous validation:

Known limitations:

---

## References (optional)

List any architecture documents, reports, papers, previous prompts,
Git commits, run records, or external references that provide context
for this task.

---

## Authorization Matrix

State exactly what this prompt authorizes.

| Action                          | Authorization                         |
| ------------------------------- | ------------------------------------- |
| Local read-only inspection      | Yes / No                              |
| Local repository changes        | Yes / No                              |
| Local artifact writes/retrieval | Yes / No / Approved locations only   |
| Local test execution            | Yes / No                              |
| Commit                          | Yes / No                              |
| Push                            | Yes / No                              |
| Remote read-only commands       | Yes / No                              |
| Remote interactive file changes | Yes / No                              |
| Scheduler-created outputs       | Yes / No / Manifest locations only   |
| Environment changes             | Yes / No                              |
| Job submission                  | No / Exactly N                        |
| Automatic retry                 | Yes / No / Not applicable            |
| Larger follow-on run            | Yes / No                              |

Authorization applies only to the scope and exact inputs declared in
this prompt.

Anything not explicitly authorized is prohibited.

---

## Scope

### In Scope

List exactly what Codex is expected to inspect, modify, execute, or
produce.

Examples:

- source files;
- documentation;
- tests;
- validation scripts;
- cluster scripts;
- scientific products.

### Out of Scope

Explicitly list what must not change.

Examples:

- architecture;
- unrelated modules;
- public APIs;
- validated workflows;
- physical models;
- numerical tolerances;
- cluster environments;

unless required to fix a demonstrated bug and separately authorized.

---

## Acceptance Criteria

The task is complete only if all predeclared criteria are satisfied.

Possible criteria include:

- requested functionality is implemented;
- required tests pass;
- numerical behavior satisfies stated tolerances;
- exact backend identity is verified;
- required artifacts are produced;
- provenance is complete;
- no unrelated regressions are introduced;
- repository and output state match expectations.

Include quantitative tolerances whenever practical.

For research work, distinguish operational acceptance from scientific
assessment.

### Operational Acceptance

Examples:

- correct Git SHA;
- correct execution environment;
- successful process exit;
- finite outputs;
- required files preserved;
- numerical conservation bounds satisfied.

### Scientific Assessment

Examples:

- agreement with an analytic prediction;
- expected qualitative behavior;
- predeclared gain, fidelity, or error thresholds;
- comparison with a control;
- approach toward a steady or converged result.

A run may pass operationally while its scientific assessment remains
inconclusive.

---

## Workflow

Perform work in the required order.

1. Inspect the current state.
2. Verify all required inputs.
3. Confirm authorization.
4. Perform the authorized implementation or preparation.
5. Run the requested validation.
6. Collect evidence.
7. Classify the result.
8. Produce the required report.
9. Stop at the approval gate.

Do not perform the next deployment, execution, or research stage unless
it is explicitly authorized.

---

## Validation

Specify exactly which checks must be executed.

Examples:

```text
python -m pytest ...

python scripts/checks/...

CPU/GPU comparison

Scientific control case
```

Validation should be objective and reproducible whenever practical.

Record numerical values, tolerances, commands, versions, and outputs
where applicable.

---

## Cluster Instructions (optional)

If cluster work is required:

- verify the expected Git SHA before execution;
- verify repository cleanliness;
- use only the approved host and scheduler configuration;
- use the approved Python environment;
- use scheduler-assigned compute resources;
- do not SSH manually from a login node to a compute node;
- use scheduler-provided scratch storage (`$TMPDIR`) whenever
  available;
- keep large transient products on scratch;
- copy only compact scientific products to persistent storage;
- separate stdout and stderr;
- record provenance and checksums;
- prohibit silent backend fallback;
- submit only the explicitly authorized number of jobs;
- do not retry automatically unless explicitly authorized.

Do not update a remote checkout, install software, alter environments,
or submit jobs unless those actions are explicitly authorized in this
prompt.

If cluster work is not required, state so explicitly.

---

## Launch Manifest (required for job submission)

Before submitting a job, resolve and verify this immutable manifest.

```text
SSH endpoint: {{SSH_ENDPOINT}}
Scheduler script: {{SCHEDULER_SCRIPT}}
Scheduler-script SHA-256: {{SCHEDULER_SCRIPT_SHA256}}
Expected Git SHA: {{EXPECTED_GIT_SHA}}
Input checksums: {{INPUT_CHECKSUMS_OR_NA}}
Python executable: {{PYTHON_EXECUTABLE}}
CUDA module: {{CUDA_MODULE_OR_NA}}
Account: {{SCHEDULER_ACCOUNT}}
Partition: {{SCHEDULER_PARTITION}}
QOS: {{SCHEDULER_QOS_OR_NA}}
Nodes: {{NODES}}
Tasks: {{TASKS}}
CPUs per task: {{CPUS_PER_TASK}}
Memory: {{MEMORY}}
Wall time: {{WALL_TIME}}
GPU request: {{GPU_REQUEST_OR_NA}}
Payload command: {{PAYLOAD_COMMAND}}
Scratch directory: {{SCRATCH_DIRECTORY_OR_POLICY}}
Persistent output directory: {{PERSISTENT_OUTPUT_DIRECTORY}}
Standard output: {{STDOUT_PATH}}
Standard error: {{STDERR_PATH}}
Local retrieval directory: {{LOCAL_RETRIEVAL_DIRECTORY}}
Authorized submissions: {{AUTHORIZED_SUBMISSION_COUNT}}
Automatic retries: {{AUTOMATIC_RETRY_POLICY}}
```

Every field must contain an approved value or `N/A`; blank fields are
unresolved. Stop immediately if the prepared job differs from the
approved manifest.

---

## Failure Policy

Specify the permitted response to failure.

- Is failure terminal for this prompt?
- Is read-only diagnosis authorized?
- Are file changes authorized?
- Are environment changes authorized?
- Is one corrected resubmission authorized?
- Is automatic retry prohibited?
- What evidence must be collected before stopping?

Unless explicitly stated otherwise:

- do not retry automatically;
- do not alter parameters;
- do not change environments;
- collect the failure evidence;
- stop and report.

---

## Result Classification

Use exactly one primary classification unless the task explicitly
declares a more suitable task-specific vocabulary. Research prompts
should report operational status and scientific assessment separately.

### Passed

Every predeclared acceptance criterion was satisfied.

### Failed

At least one predeclared acceptance criterion was violated.

### Inconclusive

Execution completed, but available evidence does not decide the
scientific or engineering question.

### Blocked

Execution could not start or complete because of an unresolved external
prerequisite or environment condition.

For research tasks, report operational status and scientific assessment
separately when appropriate.

---

## Deliverables

Specify exactly what should be produced.

Examples:

- modified source files;
- documentation;
- updated prompts;
- tests;
- scheduler scripts;
- figures;
- metrics;
- provenance;
- checksums;
- retrieval commands;
- scientific or engineering report.

Large data products intentionally excluded from Git should have their
checksums and archive locations recorded.

---

## Stop Conditions

State the exact milestone at which Codex must stop.

Examples:

- after read-only discovery;
- after implementation and local validation;
- before commit;
- after commit but before push;
- after job preparation but before submission;
- after one submitted job completes;
- after retrieval and reporting;
- before a larger follow-on calculation.

Do **not** continue beyond the declared stop condition without explicit
approval.

---

## Required Report

Summarize:

- result classification;
- operational status;
- scientific assessment, if applicable;
- files modified;
- commands actually executed;
- validation performed;
- numerical results and tolerances;
- Git SHA;
- repository status;
- environment and resource details;
- produced artifacts;
- output and archive locations;
- warnings;
- unresolved issues;
- recommended next step.

Clearly distinguish:

- verified facts;
- measured results;
- interpretation;
- assumptions;
- recommendations.

---

## Approval Gate

Stop here and wait for explicit approval before any unauthorized action,
including:

- committing;
- pushing;
- modifying remote files;
- changing environments;
- submitting or resubmitting jobs;
- running parameter sweeps;
- changing architecture;
- beginning the next implementation or research stage.

---

### Prompt Notes

This template is intentionally conservative.

Favor reproducibility over speed.

Favor explicit evidence over assumptions.

Favor immutable launch specifications over conversational intent.

Favor stopping for approval rather than continuing automatically.

Standalone execution prompts should remain understandable and safe even
when they also reference shared operational policy.
