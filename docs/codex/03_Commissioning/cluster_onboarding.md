# Cluster Onboarding

**Status:** Validated
**Version:** 1.2
**Last reviewed:** 2026-08-05
**Prerequisites:** None
**Usual next prompt:** Cluster Checkout Preparation (planned)

---

## Purpose

Prepare a new or existing HPC cluster for reproducible LCProp
execution without modifying the cluster until explicit approval is
granted.

The objective is to establish a documented execution environment,
identify all prerequisites for execution, and prepare a recommended
configuration for later commissioning.

This prompt performs discovery only.

---

## Required Inputs

- Target cluster: `{{TARGET_CLUSTER}}`
- SSH endpoint: `{{SSH_ENDPOINT}}`
- Repository name: `{{REPOSITORY}}`
- Expected Git SHA or `N/A`: `{{EXPECTED_GIT_SHA_OR_NA}}`

Stop before acting if any required placeholder remains unresolved.

---

## Background

Current repository state:

Relevant branch:

Current Git SHA:

Known cluster information:

Previous commissioning reports:

Known constraints:

---

## References (optional)

List any cluster documentation, onboarding guides, scheduler
documentation, previous commissioning reports, or environment notes.

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
| Remote read-only commands       | Yes            |
| Remote interactive file changes | No             |
| Scheduler-created outputs       | No             |
| Environment changes             | No             |
| Job submission                  | No             |
| Automatic retry                 | Not applicable |
| Larger follow-on run            | No             |

This prompt authorizes discovery only.

---

## Scope

### In Scope

Determine:

- scheduler type;
- account and QoS;
- available partitions;
- available GPU types;
- Python environments;
- CUDA modules;
- CuPy availability;
- LCProp installation status;
- project storage;
- scheduler scratch storage;
- recommended repository location;
- recommended run directories.

### Out of Scope

Do **not**:

- submit jobs;
- prepare Slurm scripts;
- install software;
- modify environments;
- create repositories;
- change scheduler configuration.

---

## Acceptance Criteria

The onboarding is complete only if:

- scheduler information is documented;
- available software is identified;
- storage locations are documented;
- recommended repository location is identified;
- recommended execution environment is identified;
- remaining unknowns are explicitly documented.

Unknown information should never be inferred.

---

## Workflow

1. Perform read-only discovery.
2. Verify scheduler configuration.
3. Verify software environments.
4. Verify storage locations.
5. Verify available compute resources.
6. Recommend repository location.
7. Recommend execution environment.
8. Produce onboarding report.
9. Stop.

---

## Validation

Record:

- scheduler type;
- account;
- QoS;
- partitions;
- available GPUs;
- Python versions;
- CUDA modules;
- CuPy availability;
- LCProp availability;
- storage paths;
- scratch configuration.

---

## Deliverables

Produce:

- cluster summary;
- environment summary;
- storage summary;
- recommended repository location;
- recommended run directory;
- recommended Python environment;
- recommended CUDA module;
- identified unknowns.

Do not prepare scheduler scripts.

---

## Result Classification

### Passed

Cluster environment successfully characterized.

### Blocked

Required cluster information could not be obtained.

### Inconclusive

Additional information is required before commissioning can continue.

---

## Stop Conditions

Stop after producing the onboarding report.

Do **not**:

- prepare jobs;
- create repositories;
- install software;
- submit jobs.

---

## Required Report

Summarize:

- scheduler;
- accounts;
- QoS;
- partitions;
- GPU availability;
- software environment;
- storage configuration;
- recommended execution configuration;
- remaining unknowns;
- recommended next prompt.

Clearly distinguish:

- verified facts;
- assumptions;
- unknowns.

---

## Approval Gate

Wait for explicit approval before:

- repository checkout;
- scheduler script preparation;
- environment modification;
- software installation;
- job submission.

---

### Prompt Notes

Cluster onboarding is intentionally read-only.

Its purpose is to characterize the execution environment and recommend
a configuration for later commissioning.

Repository checkout, job preparation, and execution belong to later
prompts.
