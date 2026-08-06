# CPU Smoke Test

**Status:** Validated
**Version:** 1.2
**Last reviewed:** 2026-08-05
**Prerequisites:** Cluster Checkout Preparation (planned), Scheduler Job Preparation (planned), or equivalent explicitly approved preparation
**Usual next prompt:** GPU Smoke Test

---

## Purpose

Execute a minimal CPU-only LCProp validation on the target cluster.

The objective is to verify that a specific Git revision executes
correctly in the intended runtime environment before any GPU
commissioning or research-scale calculations begin.

The smoke test should be intentionally small, deterministic, and fast.

---

## Required Inputs

- Expected Git SHA: `{{EXPECTED_GIT_SHA}}`
- Target cluster: `{{TARGET_CLUSTER}}`
- SSH endpoint: `{{SSH_ENDPOINT}}`
- Python executable: `{{PYTHON}}`
- Scheduler script: `{{SBATCH_SCRIPT}}`
- Scheduler-script SHA-256: `{{SBATCH_SHA256}}`
- Test payload: `{{PAYLOAD}}`
- Persistent output directory: `{{OUTPUT_DIR}}`
- Local retrieval directory: `{{LOCAL_RETRIEVAL_DIR}}`

Stop before acting if any required placeholder remains unresolved.

---

## Background

Current repository state:

Relevant branch:

Current Git SHA:

Prepared scheduler script:

Previous commissioning reports:

Known constraints:

---

## References (optional)

List previous commissioning reports, onboarding reports,
validation reports, or related documentation.

---

## Authorization Matrix

| Action                          | Authorization          |
| ------------------------------- | ---------------------- |
| Local read-only inspection      | Yes                    |
| Local repository changes        | No                     |
| Local artifact writes/retrieval | Yes, manifest location |
| Local test execution            | No                     |
| Commit                          | No                     |
| Push                            | No                     |
| Remote read-only commands       | Yes                    |
| Remote interactive file changes | No                     |
| Scheduler-created outputs       | Yes, manifest locations |
| Environment changes             | No                     |
| Job submission                  | Exactly one            |
| Automatic retry                 | No                     |
| Larger follow-on run            | No                     |

---

## Scope

### In Scope

Perform:

- repository verification;
- Git SHA verification;
- environment verification;
- one approved CPU smoke-test submission;
- provenance recording;
- retrieval;
- reporting.

### Out of Scope

Do **not**:

- modify source code;
- perform GPU execution;
- perform parameter sweeps;
- modify cluster environments;
- install software.

---

## Acceptance Criteria

### Operational Acceptance

The smoke test passes operationally only if:

- expected Git SHA verified;
- repository clean;
- expected Python environment used;
- requested backend is NumPy;
- reported backend is NumPy;
- smoke test passes;
- stdout and stderr preserved;
- provenance recorded;
- exit code is zero.

### Scientific Assessment

Not applicable.

The CPU smoke test is an operational commissioning exercise.

---

## Workflow

1. Verify repository.
2. Verify Git SHA.
3. Verify Python environment.
4. Verify authorization.
5. Verify launch manifest.
6. Submit exactly one approved job.
7. Monitor completion.
8. Retrieve results.
9. Produce commissioning report.
10. Stop.

Do not submit additional jobs without explicit approval.

---

## Validation

Verify and record:

- Git SHA;
- repository cleanliness;
- Python version;
- NumPy version;
- SciPy version;
- LCProp version;
- scheduler allocation;
- execution node;
- runtime;
- exit code;
- requested backend;
- reported backend;
- stdout;
- stderr.

---

## Launch Manifest

Before submission verify:

```text
SSH endpoint: {{SSH_ENDPOINT}}
Scheduler script: {{SBATCH_SCRIPT}}
Scheduler-script SHA-256: {{SBATCH_SHA256}}
Expected Git SHA: {{EXPECTED_GIT_SHA}}
Input checksums: {{INPUT_CHECKSUMS_OR_NA}}
Python executable: {{PYTHON}}
CUDA module: N/A
Account: {{SLURM_ACCOUNT}}
Partition: {{SLURM_PARTITION}}
QOS: {{SLURM_QOS_OR_NA}}
Nodes: {{NODES}}
Tasks: {{TASKS}}
CPUs per task: {{CPUS_PER_TASK}}
Memory: {{MEMORY}}
Wall time: {{WALL_TIME}}
GPU request: N/A
Payload command: {{PAYLOAD}}
Scratch directory: {{SCRATCH_DIRECTORY_OR_POLICY}}
Persistent output directory: {{OUTPUT_DIR}}
Standard output: {{STDOUT_PATH}}
Standard error: {{STDERR_PATH}}
Local retrieval directory: {{LOCAL_RETRIEVAL_DIR}}
Authorized submissions: 1
Automatic retries: prohibited
```

Every field must contain an approved value or `N/A`; blank fields are
unresolved. Stop immediately if any item differs from the approved
manifest.

---

## Failure Policy

If execution fails:

- do not retry automatically;
- do not modify parameters;
- do not modify environments;
- collect stdout;
- collect stderr;
- collect scheduler status;
- collect exit code;
- stop and report.

---

## Result Classification

### Passed

All operational acceptance criteria satisfied.

### Failed

At least one operational criterion failed.

### Blocked

Execution could not begin because of an external prerequisite.

### Inconclusive

Unexpected situation requiring user review.

---

## Deliverables

Produce:

- verified scheduler script identity and checksum;
- provenance;
- stdout;
- stderr;
- exit code;
- retrieval commands;
- commissioning report.

---

## Stop Conditions

Stop after:

- successful retrieval;
- commissioning report.

Do not continue to GPU commissioning.

---

## Required Report

Summarize:

- result classification;
- Git SHA;
- repository status;
- scheduler configuration;
- execution node;
- runtime;
- backend requested;
- backend reported;
- exit status;
- validation results;
- output locations;
- warnings;
- recommended next step.

---

## Approval Gate

Wait for explicit approval before:

- GPU commissioning;
- additional submissions;
- environment changes;
- research calculations.

---

### Prompt Notes

The CPU smoke test is a commissioning exercise rather than a numerical
benchmark.

Its purpose is to establish confidence that a specific LCProp revision
executes correctly in a reproducible CPU environment before GPU
validation begins.
