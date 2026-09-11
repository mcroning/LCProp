# GPU Validation Benchmark

**Status:** Reviewed

**Version:** 1.1

**Last reviewed:** 2026-09-11

**Prerequisites:** GPU Smoke Test completed for the target environment, and the locally validated candidate committed at an exact Git SHA

**Usual next prompt:** Pre-Commit Review or Research prompt, as appropriate

---

## Purpose

Execute a bounded GPU validation and performance benchmark for a
specific, committed and locally validated LCProp implementation change on an
already commissioned GPU environment.

The objective is to determine whether the candidate preserves the
declared numerical behavior relative to an approved baseline and
produces a measurable performance or memory improvement under
controlled conditions.

This prompt is for post-development commissioning. It does not authorize
implementation, initial GPU commissioning, or open-ended research.

---

## Required Inputs

- Expected production Git SHA: `{{EXPECTED_GIT_SHA}}`
- Baseline committed Git SHA: `{{BASELINE_GIT_SHA}}`
- Candidate committed Git SHA: `{{CANDIDATE_GIT_SHA}}`
- Target cluster: `{{TARGET_CLUSTER}}`
- SSH endpoint: `{{SSH_ENDPOINT}}`
- Python executable: `{{PYTHON}}`
- CUDA environment: `{{CUDA_ENVIRONMENT}}`
- GPU model or approved equivalence rule: `{{GPU_MODEL_OR_EQUIVALENCE_RULE}}`
- Benchmark scheduler script: `{{BENCHMARK_SBATCH_SCRIPT}}`
- Benchmark script SHA-256: `{{BENCHMARK_SBATCH_SHA256}}`
- Validation scheduler script or `N/A`: `{{VALIDATION_SBATCH_SCRIPT_OR_NA}}`
- Validation script SHA-256 or `N/A`: `{{VALIDATION_SBATCH_SHA256_OR_NA}}`
- Benchmark payload/configuration: `{{BENCHMARK_PAYLOAD}}`
- Validation payload: `{{VALIDATION_PAYLOAD}}`
- Numerical acceptance criterion: `{{NUMERICAL_ACCEPTANCE_CRITERION}}`
- Performance metrics: `{{PERFORMANCE_METRICS}}`
- Persistent output directory: `{{OUTPUT_DIR}}`
- Local retrieval directory: `{{LOCAL_RETRIEVAL_DIR}}`
- Authorized submission count: `{{AUTHORIZED_SUBMISSION_COUNT}}`

Every required field must contain an approved value, a conspicuous
`{{PLACEHOLDER}}`, or `N/A`. Blank required fields are unresolved.

**Stop before acting if any required placeholder remains unresolved.**

---

## Background

Current repository state:

Relevant branch:

Current Git SHA:

Locally validated candidate:

Approved baseline:

Previous CPU/GPU commissioning reports:

Known constraints:

---

## References (optional)

Authoritative benchmark evidence must satisfy `docs/codex/README.md`,
**Canonical Lifecycle and Evidence Gates**. This prompt adds GPU- and
scheduler-specific requirements without relaxing the shared clean-source,
manifest, or schema gates.

List the local-development report, GPU smoke report, previous benchmark
report, numerical validation specification, or other authoritative
documentation.

---

## Authorization Matrix

| Action                          | Authorization                              |
| ------------------------------- | ------------------------------------------ |
| Local read-only inspection      | Yes                                        |
| Local repository changes        | No                                         |
| Local artifact writes/retrieval | Yes, manifest locations                    |
| Local test execution            | No                                         |
| Commit                          | No                                         |
| Push                            | No                                         |
| Remote read-only commands       | Yes                                        |
| Remote interactive file changes | No                                         |
| Scheduler-created outputs       | Yes, manifest locations                    |
| Environment changes             | No                                         |
| Job submission                  | Exactly `{{AUTHORIZED_SUBMISSION_COUNT}}`  |
| Automatic retry                 | No                                         |
| Larger follow-on run            | No                                         |

Authorization applies only to the scope and exact resolved inputs in
this prompt. Anything not explicitly authorized is prohibited.

---

## Scope

### In Scope

Perform:

- immutable source and provenance verification;
- scheduler-script checksum verification;
- target GPU and environment verification;
- bounded baseline/candidate benchmark execution;
- declared numerical-equivalence validation;
- runtime and memory measurement;
- artifact retrieval and checksum verification;
- performance and commissioning reporting.

### Out of Scope

Do **not**:

- modify production source code;
- modify numerical algorithms;
- change tolerances after seeing results;
- modify cluster environments;
- install software;
- tune the candidate interactively on the cluster;
- automatically retry failed jobs;
- expand grid sizes or parameter ranges beyond the approved manifest;
- begin research-scale calculations;
- commit or push.

---

## Acceptance Criteria

### Preconditions

Before submission verify that:

1. the target GPU environment has passed GPU Smoke Test or equivalent
   commissioning;
2. the candidate has passed its required local validation;
3. the baseline and candidate are exact committed Git SHAs;
4. every production checkout is immutable, clean, and reports empty
   `git status --porcelain` before execution;
5. benchmark and validation scripts have approved checksums;
6. numerical acceptance criteria were declared before remote execution;
7. performance metrics were declared before remote execution;
8. all launch-manifest fields are resolved, including the canonical evidence
   manifest requirements.

If any precondition fails, stop and report `Blocked`.

### Operational Acceptance

Remote commissioning passes operationally only if:

- the expected committed production SHA and empty source status are verified;
- the approved Python, CUDA, and CuPy environment is used;
- the requested and reported backend are both CuPy;
- the requested GPU model is allocated, unless the manifest explicitly
  permits an equivalent model;
- no CPU fallback occurs;
- all approved jobs complete with zero exit status;
- stdout and stderr are preserved;
- provenance and checksums are recorded;
- artifacts are retrieved and checksum-verified.

### Numerical Acceptance

The candidate passes numerically only if it satisfies the predeclared:

`{{NUMERICAL_ACCEPTANCE_CRITERION}}`

The criterion may require bit-for-bit identity, exact iteration-path
identity, a repository-defined tolerance, or explicitly approved
norm/error thresholds. Do not weaken it after observing results.

If the candidate is faster but fails numerical acceptance, classify the
candidate as failed.

Compare every quantity required by the validation specification. As
applicable, include:

- final complex optical fields;
- selected intermediate fields;
- intensities and material fields;
- residuals and convergence histories;
- Newton/PCG iteration counts;
- replay diagnostics and power drift;
- requested scientific products.

Classify each comparison as bit-for-bit identical, within the declared
tolerance, or changed. Report any changed trajectory even if the final
result passes a looser declared tolerance.

### Performance Assessment

Measure the predeclared:

`{{PERFORMANCE_METRICS}}`

Where applicable report:

- total wall time;
- targeted component time;
- speedup or percentage change;
- iteration counts;
- FFT/kernel counts;
- host/device transfer time;
- synchronization count;
- peak GPU memory;
- persistent-cache or workspace memory;
- run-to-run variability when repeated runs were explicitly authorized.

A performance improvement is credible only if it exceeds measurement
noise or is otherwise supported by the approved benchmark design. Do
not retain complexity solely for a sub-noise timing difference.

The baseline and candidate must use the same approved physical
parameters, grid, longitudinal sampling, dtype, backend, GPU model,
convergence tolerances, requested products, random realization, and
scheduler resources. If the optimization is conditional, include an
appropriate control only when it is explicitly specified in the
manifest.

Do not silently change workload characteristics to improve the
candidate result.

---

## Workflow

1. Verify the local committed source SHA, clean status, and preparation artifacts.
2. Verify benchmark and validation script checksums.
3. Verify the remote committed source SHA and any archive/transport identity.
4. Verify Python, CUDA, CuPy, and GPU environment.
5. Verify the complete launch manifest.
6. Submit only the explicitly authorized bounded jobs.
7. Monitor scheduler state without modifying the jobs.
8. Preserve scheduler accounting, stdout, stderr, and exit codes.
9. Retrieve artifacts and verify their checksums.
10. Compare baseline and candidate numerical outputs.
11. Calculate the approved performance metrics.
12. Produce the commissioning benchmark report.
13. Stop.

Do not perform an additional optimization cycle under this prompt.

---

## Validation

Verify and record:

- exact baseline and candidate identities;
- benchmark and validation script checksums;
- request, input, precision, grid, solver, and random-realization
  identity;
- Python, CUDA, CuPy, and actual allocated GPU identity;
- requested and resolved backend;
- numerical comparison values and predeclared tolerances;
- baseline and candidate timings;
- synchronized component timings where applicable;
- iteration, FFT, kernel, and transfer counts where applicable;
- peak GPU memory;
- scheduler accounting and exit status;
- retrieved artifact checksums.
- post-execution evidence manifest compliance with the canonical shared gate.

Do not interpret GPU Smoke Test timing as baseline benchmark evidence.

---

## Cluster Instructions

- Use only the approved SSH endpoint and scheduler configuration.
- Use scheduler-assigned GPU resources; do not SSH manually from a
  login node to a compute node.
- Verify the expected committed source SHA and empty `git status --porcelain`
  before execution.
- Verify the actual GPU and CuPy backend inside the allocation.
- Use scheduler-provided scratch storage when available.
- Keep large transient products on scratch and preserve only the
  declared evidence and compact outputs.
- Separate stdout and stderr.
- Prohibit silent backend fallback.
- Submit exactly the authorized number of jobs.
- Do not update a remote checkout, install software, alter an
  environment, or retry unless separately authorized.

---

## Launch Manifest

Before submission verify:

```text
SSH endpoint: {{SSH_ENDPOINT}}
Expected production Git SHA: {{EXPECTED_GIT_SHA}}
Baseline committed Git SHA: {{BASELINE_GIT_SHA}}
Candidate committed Git SHA: {{CANDIDATE_GIT_SHA}}

Python executable: {{PYTHON}}
CUDA environment: {{CUDA_ENVIRONMENT}}
CuPy version: {{CUPY_VERSION_OR_VERIFY_AT_RUNTIME}}
GPU model or equivalence rule: {{GPU_MODEL_OR_EQUIVALENCE_RULE}}

Benchmark scheduler script: {{BENCHMARK_SBATCH_SCRIPT}}
Benchmark script SHA-256: {{BENCHMARK_SBATCH_SHA256}}
Validation scheduler script: {{VALIDATION_SBATCH_SCRIPT_OR_NA}}
Validation script SHA-256: {{VALIDATION_SBATCH_SHA256_OR_NA}}

Account: {{SLURM_ACCOUNT}}
Partition: {{SLURM_PARTITION}}
QOS: {{SLURM_QOS_OR_NA}}
Nodes: {{NODES}}
Tasks: {{TASKS}}
CPUs per task: {{CPUS_PER_TASK}}
Memory: {{MEMORY}}
Wall time: {{WALL_TIME}}
GPU request: {{GPU_REQUEST}}

Benchmark payload: {{BENCHMARK_PAYLOAD}}
Validation payload: {{VALIDATION_PAYLOAD}}
Numerical acceptance criterion: {{NUMERICAL_ACCEPTANCE_CRITERION}}
Performance metrics: {{PERFORMANCE_METRICS}}

Input checksums: {{INPUT_CHECKSUMS_OR_NA}}
Scratch directory: {{SCRATCH_DIRECTORY_OR_POLICY}}
Persistent output directory: {{OUTPUT_DIR}}
Standard output: {{STDOUT_PATH}}
Standard error: {{STDERR_PATH}}
Local retrieval directory: {{LOCAL_RETRIEVAL_DIR}}

Authorized submissions: {{AUTHORIZED_SUBMISSION_COUNT}}
Automatic retries: prohibited
```

Every field must contain an approved value or `N/A`. Blank fields are
unresolved. Stop immediately if the actual launch configuration differs
from the approved manifest.

---

## Failure Policy

If a job or validation step fails:

- do not retry automatically;
- do not modify source or scheduler scripts;
- do not change parameters, numerical tolerances, or environments;
- collect stdout, stderr, scheduler status/accounting, and exit code;
- retrieve any valid partial artifacts;
- identify whether the failure is production-, environment-, scheduler-,
  or harness-related;
- stop and report.

A harness-only defect does not authorize an automatic retry. Prepare a
corrected harness only under a separately authorized development or
preparation step, then return to a new approval gate before resubmission.

---

## Result Classification

Use exactly one classification.

### Passed — Numerically Equivalent and Faster

All operational and numerical criteria pass, and the candidate has a
credible measured performance improvement.

### Passed — Numerically Equivalent, No Measurable Speedup

Operational and numerical criteria pass, but the performance difference
is consistent with noise or too small to justify the change.

### Failed — Numerical Regression

Operational execution succeeds, but the candidate violates the declared
numerical acceptance criterion.

### Failed — Operational

One or more operational acceptance criteria fail.

### Blocked

Execution cannot begin because an external prerequisite or unresolved
manifest field prevents safe submission.

### Inconclusive

Unexpected conditions prevent a defensible numerical or performance
conclusion.

This task-specific vocabulary intentionally replaces the generic
`Passed` / `Failed` result vocabulary in `TEMPLATE.md`.

---

## Deliverables

Produce:

- verified source identities;
- scheduler-script identities and SHA-256 checksums;
- complete launch manifest;
- scheduler job IDs and accounting;
- stdout and stderr;
- numerical comparison results;
- timing, performance, and GPU-memory results;
- retrieved artifact checksums and local retrieval paths;
- commissioning benchmark report;
- recommended next prompt.

---

## Stop Conditions

Stop after:

- all authorized jobs have completed or failed;
- artifacts have been retrieved as far as possible;
- checksums have been verified;
- numerical and performance assessment has been completed;
- the report has been written.

Do **not**:

- retry;
- optimize or modify source;
- commit or push;
- begin a larger calculation;
- begin research execution.

---

## Required Report

Summarize:

- result classification;
- branch and exact production SHA;
- baseline and candidate committed SHAs;
- target GPU and node;
- scheduler configuration, job IDs, and exit status;
- numerical-equivalence and iteration/trajectory verdicts;
- baseline and candidate timings;
- speedup or percentage change;
- peak GPU memory;
- relevant component-level attribution;
- validation results;
- artifact locations and checksums;
- warnings or harness limitations;
- whether the candidate should be retained;
- recommended next step.

State explicitly when an observed speedup applies only to a conditional
workload such as scattering-enabled calculations. Clearly distinguish
verified facts, measured results, interpretation, assumptions, and
recommendations.

---

## Approval Context

Resolve the execution authorization before submission.

```text
Next approval phrase: Approve submission
Authorized target: {{EXACT_BENCHMARK_AND_VALIDATION_JOBS}}
Immutable identifier: {{SOURCE_AND_LAUNCH_MANIFEST_IDENTIFIERS}}
Action authorized on approval: Submit exactly {{AUTHORIZED_SUBMISSION_COUNT}} bounded GPU validation benchmark job(s)
Stopping point: After completion, retrieval, numerical/performance assessment, and reporting
```

Every field must contain a concrete value or `N/A`. The approval phrase
is invalid while any field is blank, ambiguous, or unresolved. Approval
is single-use and applies only to this exact context.

---

## Approval Gate

Once every required input, launch-manifest field, and approval-context
field is resolved, the canonical execution authorization is
**Approve submission**.

Wait for separate explicit approval before:

- any retry or additional submission;
- any larger benchmark or research-scale run;
- any environment change or source modification;
- commit or push.

---

### Prompt Notes

This is a commissioning prompt for an already commissioned GPU
environment.

It is intentionally distinct from:

- **GPU Smoke Test**, which establishes that a revision executes
  correctly on the GPU environment and is not a performance benchmark;
- **Implementation and Local Validation**, which authorizes bounded
  source changes and local tests but prohibits remote execution;
- **Research prompts**, which investigate scientific questions rather
  than whether a developed implementation change is numerically
  equivalent and faster.

Use this prompt for bounded before/after GPU validation of a locally
developed optimization or other performance-sensitive implementation
change.
