# PR Image Amplification Pilot

**Status:** Validated
**Version:** 1.2
**Last reviewed:** 2026-08-05
**Prerequisites:** GPU Smoke Test
**Usual next prompt:** Material-Time Approach-to-Steady-State Study (planned)

---

## Purpose

Execute a reproducible GPU-accelerated photorefractive image
amplification pilot using the validated LCProp execution path.

The objective is to validate the complete photorefractive imaging
workflow under realistic conditions while producing a reproducible
scientific record suitable for future convergence studies.

The pilot is intended to validate methodology rather than produce
final paper-quality numerical results.

---

## Required Inputs

- Expected Git SHA: `{{EXPECTED_GIT_SHA}}`
- Target cluster: `{{TARGET_CLUSTER}}`
- SSH endpoint: `{{SSH_ENDPOINT}}`
- Python executable: `{{PYTHON}}`
- CUDA module: `{{CUDA_MODULE}}`
- Scheduler script: `{{SBATCH_SCRIPT}}`
- Scheduler-script SHA-256: `{{SBATCH_SHA256}}`
- Input image checksum: `{{IMAGE_SHA256}}`
- Pilot parameter set: `{{PARAMETERS}}`
- Payload command: `{{PAYLOAD}}`
- Operational tolerances: `{{OPERATIONAL_TOLERANCES}}`
- Scientific controls and reference expectations: `{{SCIENTIFIC_CONTROLS}}`
- Persistent output directory: `{{OUTPUT_DIR}}`
- Local retrieval directory: `{{LOCAL_RETRIEVAL_DIR}}`

Stop before acting if any required placeholder remains unresolved.

---

## Background

Current repository state:

Relevant branch:

Current Git SHA:

Prepared scheduler script:

Previous GPU commissioning report:

Previous image-amplification reports:

Known constraints:

---

## References (optional)

List architecture documents, commissioning reports,
research reports, papers, previous pilot studies,
validation reports, or related documentation.

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
- GPU environment verification;
- one approved image-amplification pilot;
- image reconstruction;
- quantitative analysis;
- provenance recording;
- retrieval;
- scientific reporting.

### Out of Scope

Do **not**:

- modify production source code;
- perform parameter sweeps;
- begin convergence studies;
- optimize parameters;
- benchmark hardware.

---

## Acceptance Criteria

### Operational Acceptance

The pilot passes operationally only if:

- expected Git SHA verified;
- repository clean;
- requested backend is CuPy;
- reported backend is CuPy;
- no fallback occurs;
- finite numerical outputs are produced;
- reconstruction completes successfully;
- provenance is complete;
- stdout and stderr preserved;
- exit code is zero.

### Scientific Assessment

Evaluate and report:

- measured signal gain;
- reconstructed-image correlation;
- normalized RMSE;
- zero-response reconstruction;
- relative optical-power drift;
- comparison with analytic or control predictions;
- qualitative image fidelity.

Scientific conclusions should be reported separately from operational
success.

A pilot may pass operationally while the scientific outcome remains
inconclusive.

---

## Workflow

1. Verify repository.
2. Verify Git SHA.
3. Verify GPU environment.
4. Verify launch manifest.
5. Submit exactly one approved pilot.
6. Monitor completion.
7. Retrieve results.
8. Analyze scientific metrics.
9. Classify operational status.
10. Assess scientific outcome.
11. Produce research report.
12. Stop.

Do not begin additional calculations without explicit approval.

---

## Validation

Verify and record:

- Git SHA;
- repository cleanliness;
- Python version;
- CuPy version;
- CUDA environment;
- GPU model;
- requested backend;
- reported backend;
- signal gain;
- reconstructed-image correlation;
- normalized RMSE;
- zero-response reconstruction;
- relative optical-power drift;
- execution times;
- memory usage;
- stdout;
- stderr.

Scientific interpretation should be based on measured evidence rather
than expectations.

---

## Launch Manifest

Before submission verify:

```text
SSH endpoint: {{SSH_ENDPOINT}}
Scheduler script: {{SBATCH_SCRIPT}}
Scheduler-script SHA-256: {{SBATCH_SHA256}}
Expected Git SHA: {{EXPECTED_GIT_SHA}}
Input image checksum: {{IMAGE_SHA256}}
Python executable: {{PYTHON}}
CUDA module: {{CUDA_MODULE}}
Parameter set: {{PARAMETERS}}
Operational tolerances: {{OPERATIONAL_TOLERANCES}}
Scientific controls: {{SCIENTIFIC_CONTROLS}}
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
- preserve partial scientific products when practical;
- stop and report.

---

## Result Classification

Report operational status and scientific assessment independently.

### Operational Status

#### Passed

Every operational acceptance criterion was satisfied and the required
scientific measurements were recorded.

#### Failed

Operational acceptance criteria not satisfied.

#### Blocked

Execution could not begin because of an external prerequisite.

### Scientific Assessment

#### Supported

The predeclared controls and reference expectations were satisfied.

#### Not Supported

One or more predeclared scientific expectations were contradicted by
the measured evidence.

#### Inconclusive

Operational execution succeeded, but available measurements do not yet
support or contradict the scientific expectation.

---

## Deliverables

Produce:

- reconstructed images;
- selected scientific figures;
- metrics.json;
- provenance.txt;
- stdout;
- stderr;
- exit code;
- retrieval commands;
- research report in the required response.

Creating or modifying a repository research report requires separate
authorization for local repository changes.

Large data products intentionally excluded from Git should have their
checksums and archive locations recorded.

---

## Stop Conditions

Stop after:

- successful retrieval;
- scientific analysis;
- research report.

Do **not** begin convergence studies or production calculations
without explicit approval.

---

## Required Report

Summarize:

### Operational Status

- result classification;
- Git SHA;
- repository status;
- scheduler configuration;
- execution node;
- GPU model;
- CUDA environment;
- backend requested;
- backend reported;
- execution times;
- memory usage.

### Scientific Assessment

- measured signal gain;
- reconstructed-image correlation;
- normalized RMSE;
- power drift;
- comparison with analytic expectations;
- qualitative observations;
- limitations.

### Recommendations

- recommended next experiment;
- suggested convergence studies;
- remaining uncertainties.

Clearly distinguish:

- measured quantities;
- interpretation;
- limitations;
- recommendations.

---

## Approval Gate

Wait for explicit approval before:

- convergence studies;
- parameter sweeps;
- larger images;
- higher spatial resolution;
- longer material evolution;
- production calculations.

---

### Prompt Notes

This prompt is intended for research validation rather than production
simulation.

Interpret measured results within the physical assumptions of the
experiment.

Operational success does not imply scientific validation.

Favor reproducibility, objective evidence, and careful scientific
interpretation over aggressive exploration.
