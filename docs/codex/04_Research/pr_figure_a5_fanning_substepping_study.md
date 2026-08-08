# PR Figure A5 Fanning-Ring and Optical-Substepping Study

<!--
Prompt metadata should be updated whenever this procedure changes
substantially.
-->

**Status:** Draft

**Version:** 1.0

**Last reviewed:** 2026-08-08

**Prerequisites:** Resolved Figure A5 request, controlled volume-noise realization, validated paper-scale pilot, and approved immutable launch manifest

**Usual next prompt:** Figure A5 Scientific Review and Permanent Research Record (planned)

---

## Purpose

Prepare and execute a reproducible three-case photorefractive fanning study
based on Photonics (2025), Figure A5. The study tests whether 50 µm material
updates combined with 25 optical substeps reproduce a true 2 µm material
calculation while reducing the number of PR material solves.

This is a research-validation procedure. It must not be used to alter the
physical model, tune parameters for visual agreement, or claim exact Figure A5
reproduction when the original request is not fully recoverable.

---

## Required Inputs

Resolve every required input before execution.

- Expected Git SHA: `{{EXPECTED_GIT_SHA}}`
- Target branch: `{{TARGET_BRANCH}}`
- Target cluster: `{{TARGET_CLUSTER}}`
- SSH endpoint: `{{SSH_ENDPOINT}}`
- Python executable: `{{PYTHON_EXECUTABLE}}`
- CUDA module: `{{CUDA_MODULE}}`
- Scheduler scripts and SHA-256 values: `{{SCHEDULER_SCRIPTS_AND_SHA256}}`
- Research payload and SHA-256: `{{PAYLOAD_AND_SHA256}}`
- Paper PDF SHA-256: `72307a380993c257be5e13d6c269670f203be8d26311755a05ffdc434b683f1d`
- Trusted PRProp3D SHA-256: `0adb5c824459ebaa31adbf9cad541c9ef634ac63f2b83bf42e8c3ee5a62ed28f`
- Resolved Figure A5 parameter table: `{{RESOLVED_PARAMETER_TABLE}}`
- Deterministic scattering specification and checksum: `{{SCATTERING_SPECIFICATION_AND_SHA256}}`
- Case A manifest: `{{CASE_A_MANIFEST}}`
- Case B manifest: `{{CASE_B_MANIFEST}}`
- Case C manifest: `{{CASE_C_MANIFEST}}`
- Numerical and scientific tolerances: `{{TOLERANCES}}`
- Persistent output root: `{{PERSISTENT_OUTPUT_ROOT}}`
- Local retrieval root: `{{LOCAL_RETRIEVAL_ROOT}}`

Every required field must contain an approved value, a conspicuous
`{{PLACEHOLDER}}`, or `N/A`. Blank fields are unresolved.

**Stop before acting if any required placeholder remains unresolved.**

---

## Background

Current repository state: `{{REPOSITORY_STATUS}}`

Relevant branch: `{{TARGET_BRANCH}}`

Current Git SHA: `{{EXPECTED_GIT_SHA}}`

Previous milestones: PR production static workflow, PR streaming workflow,
PR GPU commissioning, and corrected Figure 6 reproduction

Relevant reports: `{{RELEVANT_REPORTS}}`

Previous validation: LCProp longitudinal-step artifact-ring,
optical-substepping, and nominal-`dz` validation scripts

Known limitations:

- Figure A5 does not have a parameter JSON in the paper's official
  supplementary archive.
- The paper caption does not state every material, launch, noise, or
  apodization parameter.
- LCProp's legacy volume-noise screen is generated and applied once per
  nominal material slice. The Figure A5 study must instead use the reviewed
  partition-independent canonical-phase-slab representation.
- The production workflow supports optical substeps; the two explicitly named
  legacy Lie reference modes do not use `optical_substeps`.

---

## References

- Photonics 2025, 12, 113, Figure A5 and Appendix C.1, Equations (A12)-(A13)
- `reference/prprop/photonics-12-00113-v3/photonics-12-00113-v3.pdf`
- `reference/prprop/prprop3d.py`
- Official supplementary parameter files for Figures 9, 11, and 14b
- `src/lcprop/pr/static_streaming.py`
- `src/lcprop/pr/workflow.py`
- `src/lcprop/pr/optical_response.py`
- `src/lcprop/optics/splitstep.py`
- `scripts/checks/validation_03_prprop_artifact_rings.py`
- `scripts/checks/validation_04_optical_substepping.py`
- `scripts/checks/validation_06_nominal_dz_convergence.py`

---

## Authorization Matrix

| Action                          | Authorization                         |
| ------------------------------- | ------------------------------------- |
| Local read-only inspection      | Yes                                   |
| Local repository changes        | No                                    |
| Local artifact writes/retrieval | Yes, manifest locations only          |
| Local test execution            | Yes, declared preflight only          |
| Commit                          | No                                    |
| Push                            | No                                    |
| Remote read-only commands       | Yes                                   |
| Remote interactive file changes | No                                    |
| Scheduler-created outputs       | Yes, manifest locations only          |
| Environment changes             | No                                    |
| Job submission                  | Exactly 3, one approved case each      |
| Automatic retry                 | No                                    |
| Larger follow-on run            | No                                    |

Authorization applies only after every required input and launch-manifest
field has been resolved and separately approved. Anything not explicitly
authorized is prohibited.

---

## Scope

### In Scope

Execute the same production nonlinear Strang workflow for:

| Case | Material spacing | Optical substeps | Optical increment | Material slices | Accepted optical substeps |
|---|---:|---:|---:|---:|---:|
| A | 50 µm | 1 | 50 µm | 80 | 80 |
| B | 50 µm | 25 | 2 µm | 80 | 2,000 |
| C | 2 µm | 1 | 2 µm | 2,000 | 2,000 |

Use identical resolved physical inputs, launch field, precision, backend,
apodization, and controlled scattering realization. Record the actual number
of material solves, coupled optical replays, FFT hops, and accepted optical
substeps; do not infer executed work from nominal counts alone.

Produce:

- input and output near fields on matched physical axes;
- linear and logarithmic normalized far fields on matched direction-cosine
  axes;
- central spectral cuts and an azimuthal or elliptical radial profile;
- predicted Equation (A12)-(A13) artifact ellipses on Case A when the internal
  launch angle and refractive index are resolved;
- fanning efficiency using a checksummed, predeclared carrier-exclusion mask;
- total power, centroid, and RMS beam widths;
- phase-aligned complex-field and intensity comparisons of B against C;
- synchronized total, optical, material, diagnostic, and I/O timings;
- GPU-memory high-water mark, scratch use, and archived-product size.

### Out of Scope

Do **not**:

- alter production source code or PR physics;
- add a new noise model during this research execution;
- tune gain, noise, crop, mask, color limits, or launch angle after seeing the
  output;
- compare production Case B with legacy Case A/C as though only `dz` differed;
- retain a complete `E(z,x,y)` volume;
- submit unapproved pilots, retries, parameter sweeps, or larger runs;
- commit, push, or merge.

---

## Architectural Verification

Before launch, verify against the expected Git SHA and record source-line
evidence for all of the following:

1. The production workflow solves one PR material state per nominal material
   slice and recomputes that state only during the slice-level coupled solve.
2. `half_step_response_from_E()` uses `dz_material / Nsub` and returns the
   response for half of one optical substep.
3. `advance_prepared_response()` performs response-half, linear-hop,
   response-half for each substep.
4. One frozen candidate `E` is reused for every optical substep in a trial
   optical pass.
5. PR-driving intensity is measured before and after the complete nominal
   slice and averaged; it is not recomputed after each optical substep.
6. Every accepted coupled material update triggers a complete optical trial
   through the nominal slice, so executed optical work may exceed the accepted
   `Nz * Nsub` count.
7. Volume noise is applied after the complete nominal slice and is not divided
   among optical substeps.
8. Case B therefore means one accepted PR state per 50 µm nominal slice and
   25 Strang optical hops through that frozen state. It does **not** mean 25 PR
   solves or 25 midpoint-source refreshes.

Stop if any item differs. Update the preparation record and obtain new approval
rather than silently changing the study interpretation.

---

## Controlled Scattering Requirement

The three cases must represent one common physical scattering field, not merely
three pseudorandom sequences initialized from the same integer.

Before launch, provide a deterministic specification that defines the
scattering field independently of the material partition and demonstrate on a
small grid that:

- the Case A and B 50 µm screen for each interval equals the declared
  aggregation of the corresponding 25 Case C 2 µm contributions;
- integrated scattering variance over 4 mm is identical within the declared
  tolerance;
- the same predeclared realization and normalization are used by all cases;
- the screen-application locations are documented explicitly.

The legacy `SeedSequence(base_seed, z_index)` policy alone does not satisfy
this requirement. Use `PRCanonicalScatteringSpec` only after its development
milestone passes pre-commit review, and preserve its complete configuration
and seed checksums in every case manifest.

---

## Parameter-Provenance Gate

Prepare a table containing every request field and exactly one provenance
class:

- `Figure A5 caption`;
- `paper equation or methods text`;
- `official Figure A5 supplementary input`;
- `trusted Figure A5 PRProp3D run record`;
- `approved surrogate from another validated fanning run`;
- `derived without free parameters`.

An approved surrogate must never be labeled as a recovered Figure A5 value.
The report must distinguish:

1. direct reproduction of the published A/C calculation;
2. a controlled LCProp production A/B/C comparison;
3. qualitative comparison with the published panels.

If gain-length product, launch angle, beam count/ratio, noise strength and
correlation, noise realization, material constants, applied field, or Tukey
parameter remain unresolved, stop before job preparation.

---

## Longitudinal Ring Prediction

For harmonic order `j`, calculate the paper's spatial-frequency semiaxes

```text
f_jx = cos(theta_in) * sqrt(n*j/(lambda*dz) - (j/dz)^2)
f_jy =                 sqrt(n*j/(lambda*dz) - (j/dz)^2)
```

in cycles/µm. Convert to the plotted convention explicitly:

```text
k_jx = 2*pi*f_jx                 [rad/µm]
k_jy = 2*pi*f_jy                 [rad/µm]
s_jx = lambda*f_jx               [direction cosine]
s_jy = lambda*f_jy               [direction cosine]
```

Include only real, propagating, and discretely represented ellipses. Record the
FFT-bin uncertainty, array-axis convention, and any clipping by the displayed
field of view. Do not fit ring positions to the measured image.

---

## Acceptance Criteria

### Operational Acceptance

Each case passes operationally only if:

- its immutable manifest and checksums match;
- the exact clean Git SHA is used;
- scheduler-assigned CuPy execution occurs with no fallback;
- all requested material slices and coupled solves complete;
- final fields and diagnostics are finite;
- deterministic replay satisfies its predeclared tolerance;
- total-power accounting closes after explicitly accounting for any Tukey
  loss;
- stdout, stderr, exit code, metrics, and provenance are preserved;
- timing counters reconcile with total runtime;
- no complete material volume is retained or archived.

### Scientific Assessment

Predeclare quantitative tolerances before submission. At minimum report:

- Case A ring locations minus Equation (A12)-(A13) predictions in FFT bins;
- Case B/C phase-aligned field relative L2 error;
- Case B/C normalized near- and far-intensity errors;
- Case B/C log-far-field error over a predeclared intensity floor;
- Case B/C radial-profile error;
- fanning-efficiency absolute difference;
- power, centroid, and RMS-width differences;
- ring-contrast suppression A to B and A to C;
- `runtime_C / runtime_B`, with optical, material, and I/O contributions.

Do not define success as visual similarity alone. Classify the acceleration
hypothesis as supported only if Case B satisfies every predeclared B/C
scientific tolerance and has a measured runtime advantage.

---

## Workflow

1. Verify references, input checksums, Git SHA, and repository cleanliness.
2. Resolve the complete parameter-provenance table.
3. Re-audit optical-substep and volume-noise semantics at the expected SHA.
4. Validate the common scattering realization on a small local case.
5. Freeze all masks, axes, normalizations, tolerances, and color limits.
6. Run a reduced paper-aspect-ratio GPU pilot under separate approval if no
   prior equivalent pilot exists.
7. Re-estimate memory, runtime, scratch, and permanent storage from that pilot.
8. Prepare and checksum one scheduler script and manifest per case.
9. Stop for explicit approval of the three immutable manifests.
10. After approval, submit exactly Cases A, B, and C once each.
11. Monitor only those job IDs; do not retry automatically.
12. Retrieve compact products and analyze all cases with one checksummed
    postprocessing payload.
13. Report operational status separately for each case.
14. Assess Figure A5 reproduction and the substepping hypothesis separately.
15. Produce a permanent research record and stop.

---

## Validation

Before paper-scale submission, run and preserve:

```text
python -m pytest -q tests/test_splitstep.py tests/test_pr_static_streaming.py

python scripts/checks/validation_03_prprop_artifact_rings.py

python scripts/checks/validation_04_optical_substepping.py
```

Add a bounded research preflight, without production-code changes, that checks:

- A/B/C request fields differ only in `dz_um`, `Nz`, `optical_substeps`, and
  the partition-specific representation of the approved common scattering
  realization;
- Case B executes 25 diffraction kernels and 50 half-response applications per
  accepted nominal optical pass;
- Case B performs no material-source refresh inside those 25 substeps;
- phase-only propagation conserves power when apodization and noise are off;
- far-field normalization integrates to unity;
- the carrier-exclusion mask and Equation (A12)-(A13) overlay are unchanged by
  case selection;
- deterministic replay reproduces the declared fields and diagnostics.

Record commands, versions, durations, tolerances, and numerical results.

---

## Cluster Instructions

- Verify the expected Git SHA and clean detached checkout before execution.
- Use only the approved SSH endpoint, scheduler configuration, Python
  executable, and CUDA module.
- Use scheduler-assigned GPU resources; do not SSH to a compute node.
- Use `$TMPDIR` for transient fields, FFT products, and full-resolution
  diagnostics.
- Archive compact metrics, provenance, selected figures, and checksums only.
- Separate stdout and stderr and record scheduler accounting.
- Record requested/reported backend, CUDA/GPU identity, synchronized timings,
  peak CuPy-pool allocation, scheduler MaxRSS, and output sizes.
- Do not update the checkout, install software, alter the environment, or
  submit a replacement job under this prompt.

---

## Launch Manifest

Resolve and approve one copy of this manifest for each case:

```text
Case: {{CASE_ID}}
SSH endpoint: {{SSH_ENDPOINT}}
Scheduler script: {{SCHEDULER_SCRIPT}}
Scheduler-script SHA-256: {{SCHEDULER_SCRIPT_SHA256}}
Expected Git SHA: {{EXPECTED_GIT_SHA}}
Payload and SHA-256: {{PAYLOAD_AND_SHA256}}
Reference/input checksums: {{INPUT_CHECKSUMS}}
Scattering specification and SHA-256: {{SCATTERING_SPECIFICATION_AND_SHA256}}
Python executable: {{PYTHON_EXECUTABLE}}
CUDA module: {{CUDA_MODULE}}
Material spacing: {{MATERIAL_DZ_UM}}
Optical substeps: {{OPTICAL_SUBSTEPS}}
Material slices: {{MATERIAL_SLICES}}
Accepted optical substeps: {{ACCEPTED_OPTICAL_SUBSTEPS}}
Complete resolved parameter set: {{PARAMETER_SET}}
Scientific tolerances: {{SCIENTIFIC_TOLERANCES}}
Account: {{SCHEDULER_ACCOUNT}}
Partition: {{SCHEDULER_PARTITION}}
QOS: {{SCHEDULER_QOS_OR_NA}}
Nodes: {{NODES}}
Tasks: {{TASKS}}
CPUs per task: {{CPUS_PER_TASK}}
Memory: {{MEMORY}}
Wall time: {{WALL_TIME}}
GPU request: {{GPU_REQUEST}}
Payload command: {{PAYLOAD_COMMAND}}
Scratch directory: {{SCRATCH_DIRECTORY_OR_POLICY}}
Persistent output directory: {{PERSISTENT_OUTPUT_DIRECTORY}}
Standard output: {{STDOUT_PATH}}
Standard error: {{STDERR_PATH}}
Local retrieval directory: {{LOCAL_RETRIEVAL_DIRECTORY}}
Authorized submissions: 1
Automatic retries: prohibited
```

Every field must contain an approved value or `N/A`; blank fields and
placeholders are unresolved. Stop immediately if a prepared job differs from
its approved manifest.

---

## Failure Policy

Failure is terminal for the affected case under this prompt. Read-only
diagnosis and artifact retrieval are authorized; source changes, environment
changes, parameter changes, and resubmission are not.

Collect scheduler state, exit code, stdout, stderr, metrics, provenance,
partial-product inventory, and resource accounting. Do not automatically retry
or launch the remaining cases if the failure undermines their common setup.

---

## Result Classification

Report operational status and scientific assessment separately.

### Operational Status

- **Passed** — every operational criterion was satisfied.
- **Failed** — at least one operational criterion was violated.
- **Blocked** — execution could not begin because a prerequisite remained
  unresolved.

### Scientific Assessment

- **Supported** — B agrees with C within every predeclared tolerance and is
  measurably faster.
- **Not supported** — execution is valid but B violates at least one declared
  B/C tolerance or has no measured speed advantage.
- **Inconclusive** — valid evidence does not decide the hypothesis.

Figure A5 reproduction must receive a separate qualitative and quantitative
assessment; it is not implied by support for the acceleration hypothesis.

---

## Deliverables

- immutable manifests and checksums for Cases A-C;
- complete parameter-provenance table;
- architectural audit with source-line evidence;
- common-scattering validation evidence;
- near-field, far-field, log-far-field, spectral-cut, and radial figures;
- ring-prediction table and Case A overlay;
- fanning, power, centroid, width, field-error, and ring-contrast metrics;
- synchronized performance and resource metrics;
- stdout, stderr, exit codes, scheduler accounting, and environment provenance;
- compact machine-readable metrics and selected figures;
- checksums and archive locations for intentionally omitted large products;
- permanent research report.

---

## Stop Conditions

During preparation, stop after the immutable manifests are presented for
approval. During approved execution, stop after the three authorized jobs,
retrieval, comparison, and permanent report. Do not begin solver changes,
parameter tuning, retries, or a larger follow-on study.

---

## Required Report

Report:

- operational status for A, B, and C;
- separate assessments of published Figure A5 reproduction and the
  substepping-acceleration hypothesis;
- exact parameter values and provenance;
- exact substepping and scattering semantics;
- all commands executed and jobs submitted;
- Git SHA, environment, GPU, scheduler, and checksums;
- material-solve and optical-substep counts;
- all predeclared scientific metrics and tolerances;
- measured timing breakdown, speedup, memory, and storage;
- archived and omitted artifacts;
- warnings, limitations, and unresolved questions;
- recommended next scientific step.

Clearly distinguish verified facts, measured results, interpretation,
approved surrogate parameters, and recommendations.

---

## Approval Context

`Approve submission` is valid only after all three case manifests are concrete,
checksummed, mutually consistent, and approved. The phrase authorizes exactly
one submission of each case and no automatic retries.

---

## Approval Gate

Stop before submission and present the complete A/B/C manifests, resource
estimates, parameter provenance, scattering validation, and preflight results.
Await explicit approval.
