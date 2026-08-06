# GPU Smoke Test

**Status:** Validated
**Version:** 1.2
**Last reviewed:** 2026-08-05
**Prerequisites:** CPU Smoke Test
**Usual next prompt:** PR Image Amplification Pilot

---

## Purpose

Execute a minimal GPU-accelerated LCProp validation on the target
cluster.

The objective is to demonstrate that a specific Git revision executes
correctly using the requested CuPy backend on a scheduler-assigned GPU
without fallback to NumPy.

The smoke test is a commissioning exercise intended to verify GPU
correctness and reproducibility rather than performance.

---

## Required Inputs

- Expected Git SHA: `{{EXPECTED_GIT_SHA}}`
- Target cluster: `{{TARGET_CLUSTER}}`
- SSH endpoint: `{{SSH_ENDPOINT}}`
- Python executable: `{{PYTHON}}`
- CUDA module: `{{CUDA_MODULE}}`
- Scheduler script: `{{SBATCH_SCRIPT}}`
- Scheduler-script SHA-256: `{{SBATCH_SHA256}}`
- Test payload: `{{PAYLOAD}}`
- CPU reference artifact: `{{CPU_REFERENCE_ARTIFACT}}`
- CPU reference artifact SHA-256: `{{CPU_REFERENCE_SHA256}}`
- CPU reference Git SHA: `{{CPU_REFERENCE_GIT_SHA}}`
- CPU/GPU request and initial-condition identity record: `{{CPU_GPU_IDENTITY_RECORD}}`
- Commissioning tolerances: `{{COMMISSIONING_TOLERANCES}}`
- Persistent output directory: `{{OUTPUT_DIR}}`
- Local retrieval directory: `{{LOCAL_RETRIEVAL_DIR}}`

Stop before acting if any required placeholder remains unresolved.

---

## Background

Current repository state:

Relevant branch:

Current Git SHA:

Prepared scheduler script:

Previous CPU commissioning report:

Known constraints:

---

## References (optional)

List previous commissioning reports, GPU validation reports,
CUDA notes, onboarding reports, or related documentation.

---

## Authorization Matrix

| Action                          | Authorization           |
| ------------------------------- | ----------------------- |
| Local read-only inspection      | Yes                     |
| Local repository changes        | No                      |
| Local artifact writes/retrieval | Yes, manifest location  |
| Local test execution            | No                      |
| Commit                          | No                      |
| Push                            | No                      |
| Remote read-only commands       | Yes                     |
| Remote interactive file changes | No                      |
| Scheduler-created outputs       | Yes, manifest locations |
| Environment changes             | No                      |
| Job submission                  | Exactly one             |
| Automatic retry                 | No                      |
| Larger follow-on run            | No                      |

---

## Scope

### In Scope

Perform:

- repository verification;
- Git SHA verification;
- CUDA verification;
- GPU verification;
- CuPy verification;
- one approved GPU smoke test;
- CPU/GPU comparison;
- provenance recording;
- retrieval;
- reporting.

### Out of Scope

Do **not**:

- modify production code;
- benchmark hardware;
- perform research calculations;
- perform parameter sweeps;
- modify environments;
- install software.

---

## Acceptance Criteria

### Operational Acceptance

The commissioning passes operationally only if:

- expected Git SHA verified;
- repository clean;
- requested backend is CuPy;
- reported backend is CuPy;
- no NumPy fallback occurs;
- scheduler-assigned GPU is used;
- CUDA initialization succeeds;
- provenance complete;
- stdout and stderr preserved;
- exit code is zero.

### Scientific Assessment

CPU and GPU results should agree within the documented commissioning
tolerances.

Before comparison, verify that the CPU reference artifact and GPU run
use the declared Git SHA and identical requests, initial conditions,
precision, integrator, grid, material-step count, optical-step count,
and other numerical controls.

Agreement should include:

- optical field;
- PR field;
- relative power drift.

This commissioning exercise establishes correctness rather than
performance.

---

## Workflow

1. Verify repository.
2. Verify Git SHA.
3. Verify CUDA environment.
4. Verify CPU-reference identity.
5. Verify the launch manifest and requested GPU resources.
6. Submit exactly one approved job.
7. Confirm the scheduler-assigned GPU inside the job.
8. Monitor completion.
9. Retrieve results.
10. Compare CPU and GPU outputs.
11. Produce commissioning report.
12. Stop.

Do not submit additional jobs without explicit approval.

---

## Validation

Verify and record:

- Git SHA;
- repository cleanliness;
- Python version;
- NumPy version;
- SciPy version;
- CuPy version;
- CUDA module;
- CUDA runtime;
- CUDA driver;
- GPU model;
- GPU device ID;
- requested backend;
- reported backend;
- CPU reference artifact path and checksum;
- CPU reference Git SHA;
- request and initial-condition identity;
- precision, integrator, grid, and step-count identity;
- CPU/GPU numerical agreement;
- relative power drift;
- cold CuPy initialization time;
- synchronized GPU execution time;
- stdout;
- stderr.

Do not interpret timing as a performance benchmark.

---

## Launch Manifest

Before submission verify:

```text
SSH endpoint: {{SSH_ENDPOINT}}
Scheduler script: {{SBATCH_SCRIPT}}
Scheduler-script SHA-256: {{SBATCH_SHA256}}
Expected Git SHA: {{EXPECTED_GIT_SHA}}
Input checksums: CPU reference {{CPU_REFERENCE_SHA256}}
Python executable: {{PYTHON}}
CUDA module: {{CUDA_MODULE}}
Account: {{SLURM_ACCOUNT}}
Partition: {{SLURM_PARTITION}}
QOS: {{SLURM_QOS_OR_NA}}
Nodes: {{NODES}}
Tasks: {{TASKS}}
CPUs per task: {{CPUS_PER_TASK}}
Memory: {{MEMORY}}
Wall time: {{WALL_TIME}}
GPU request: {{GPU_REQUEST}}
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
- collect scheduler diagnostics;
- collect exit code;
- collect CUDA diagnostics;
- stop and report.

Environment-only remediation requires separate approval.

---

## Result Classification

### Passed

All operational criteria satisfied and CPU/GPU agreement within the
declared commissioning tolerances.

### Failed

One or more commissioning criteria violated.

### Blocked

GPU execution could not begin because of an external prerequisite.

### Inconclusive

Execution completed but numerical agreement requires review.

---

## Deliverables

Produce:

- verified scheduler script identity and checksum;
- provenance;
- metrics;
- stdout;
- stderr;
- exit code;
- retrieval commands;
- GPU commissioning report.

---

## Stop Conditions

Stop after:

- successful retrieval;
- commissioning report.

Do not begin research calculations.

---

## Required Report

Summarize:

- result classification;
- operational status;
- scientific assessment;
- Git SHA;
- repository status;
- scheduler configuration;
- execution node;
- GPU model;
- CUDA environment;
- backend requested;
- backend reported;
- CPU/GPU agreement;
- relative power drift;
- cold initialization time;
- synchronized execution time;
- validation results;
- output locations;
- warnings;
- recommended next step.

Clearly distinguish:

- measured values;
- interpretation;
- recommendations.

---

## Approval Gate

Wait for explicit approval before:

- research calculations;
- parameter studies;
- environment changes;
- additional submissions.

---

### Prompt Notes

The GPU smoke test is a commissioning exercise rather than a
performance benchmark.

Its purpose is to establish confidence that GPU execution reproduces
CPU behavior within the documented commissioning tolerances while
executing entirely on the requested CuPy backend.

Performance measurements are recorded for provenance only and should
not be interpreted as throughput benchmarks.
