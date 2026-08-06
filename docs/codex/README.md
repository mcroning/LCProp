# LCProp Codex Prompt Library

## Purpose

This directory contains the standard prompt library used to conduct
software engineering, numerical validation, cluster execution, and
computational research within LCProp.

The prompts encode established operating procedures. They do not
replace scientific judgment. Reuse and refine validated prompts rather
than recreating routine workflows from scratch.

## Scope

The library defines operational workflows rather than project-specific
scientific conclusions. Each prompt should remain understandable and
safe when used on its own, even when it references shared policy or a
previous workflow stage.

## Philosophy

LCProp separates planning and scientific review from implementation and
execution. Planning and review are performed independently from implementation and execution.

The normal lifecycle is:

```text
Architecture
→ Development and local validation
→ Review and commit approval
→ Cluster commissioning
→ Research pilot
→ Scientific review
→ Scaling or convergence study
→ Permanent research record
```

Every significant computational result should be reproducible from an
exact Git revision, an identified execution environment, an immutable
launch manifest, and preserved evidence.

## Directory Organization

```text
docs/codex/
    README.md
    TEMPLATE.md
    01_Architecture/
    02_Development/
    03_Commissioning/
    04_Research/
```

Do not add empty categories in anticipation of future work. Add a new
category when the library contains a prompt whose primary purpose does
not fit the existing lifecycle.

## Prompt Status

Every specialized prompt records a status, version, review date,
prerequisite, and usual next stage.

- **Draft** — incomplete or not yet internally reviewed.
- **Reviewed** — internally reviewed but not yet exercised.
- **Validated** — successfully used for its intended workflow.
- **Superseded** — retained for historical reference after replacement.
- **Deprecated** — should no longer be used.

`TEMPLATE.md` remains Draft because it is an authoring template rather
than an executable procedure. A specialized prompt should use a real
review date. Increment its version when its operational procedure
changes materially.

## Prompt Catalog

| Prompt                                                  | Status    | Purpose                                                       | Prerequisite                                    | Authorized action                    | Next stage                                             |
| ------------------------------------------------------- | --------- | ------------------------------------------------------------- | ----------------------------------------------- | ------------------------------------ | ------------------------------------------------------ |
| `TEMPLATE.md`                                           | Draft     | Author a new operational prompt                               | N/A                                             | None until specialized               | N/A                                                    |
| `01_Architecture/architecture_review.md`                | Validated | Review a proposed architecture without implementation         | Design proposal                                 | Local read-only review               | Implementation and Local Validation                    |
| `02_Development/implementation_and_local_validation.md` | Reviewed  | Implement a bounded local change and validate it              | Approved architecture or implementation request | Local repository changes and tests   | Pre-Commit Review (planned)                            |
| `03_Commissioning/cluster_onboarding.md`                | Validated | Characterize a cluster without modifying it                   | Approved cluster access                         | Local and remote read-only discovery | Cluster Checkout Preparation (planned)                 |
| `03_Commissioning/cpu_smoke.md`                         | Validated | Execute one approved CPU commissioning job                    | Approved checkout and scheduler-job preparation | Exactly one CPU submission           | GPU Smoke Test                                         |
| `03_Commissioning/gpu_smoke.md`                         | Validated | Verify CuPy GPU execution against an identified CPU reference | Passed CPU smoke test                           | Exactly one GPU submission           | PR Image Amplification Pilot                           |
| `04_Research/pr_image_amplification_pilot.md`           | Validated | Execute one reproducible PR image-amplification pilot         | Passed GPU smoke test                           | Exactly one GPU research submission  | Material-Time Approach-to-Steady-State Study (planned) |

Prompts marked `(planned)` are intended workflow stages that are not yet
present in the library. Until they exist, an equivalent explicitly
approved preparation or review may satisfy the prerequisite.

## Required Inputs and Placeholders

Every executable prompt declares its required inputs before background
or workflow instructions.

Each required field must contain exactly one of:

- an approved concrete value;
- a conspicuous `{{PLACEHOLDER}}` while the prompt is being prepared;
- `N/A` when the field is intentionally not applicable.

Blank required fields are unresolved. An execution prompt must stop
before acting if a required placeholder or blank required field remains.
This rule applies to launch manifests as well as required-input lists.

## Authorization Vocabulary

Authorization matrices use these labels consistently:

- Local read-only inspection
- Local repository changes
- Local artifact writes/retrieval
- Local test execution
- Commit
- Push
- Remote read-only commands
- Remote interactive file changes
- Scheduler-created outputs
- Environment changes
- Job submission
- Automatic retry
- Larger follow-on run

`Remote interactive file changes` means changes made directly through a
login session. `Scheduler-created outputs` are files created by the one
approved job and are permitted only at manifest-declared locations.
Similarly, local artifact retrieval does not authorize changes to the
local Git repository.

Authorization applies only to the declared scope and exact resolved
inputs. Anything not explicitly authorized is prohibited.

## Prompt Structure

Each prompt should contain:

1. metadata;
2. purpose;
3. required inputs;
4. background and references;
5. authorization matrix;
6. scope;
7. acceptance criteria;
8. workflow;
9. validation;
10. launch manifest when submitting a job;
11. failure policy;
12. result classification;
13. deliverables;
14. stop conditions;
15. required report;
16. approval gate.

Specialized prompts may omit sections that are genuinely inapplicable.
Task-specific result terminology is allowed when it is declared
explicitly, as in an architecture review.

## Approval Gates

Approval gates separate planning, implementation, validation, commit,
remote preparation, execution, and scientific interpretation.

Typical approval points include:

- architecture ready for implementation;
- implementation ready for review;
- commit ready;
- remote checkout ready;
- scheduler job ready;
- pilot completed and retrieved;
- scaling or convergence study recommended.

For scheduler work, approval should identify an immutable launch
manifest containing the script path and checksum, Git SHA, input
checksums, environment, resources, payload, output locations, authorized
submission count, and retry policy.

## Operational Status and Scientific Assessment

Execution and research outcomes are not interchangeable.

- **Operational status** states whether the approved procedure ran in
  the approved environment and produced the required evidence.
- **Scientific assessment** states what the measurements support.

A research run may pass operationally while its scientific assessment
is inconclusive. Commissioning prompts may combine operational criteria
with a defined numerical comparison when that comparison is itself the
commissioning objective.

## Reproducibility

Research and commissioning calculations should record, when applicable:

- exact Git SHA and branch;
- repository cleanliness;
- Python executable and package versions;
- CUDA module, runtime, and driver;
- backend requested and reported;
- scheduler job ID, node, and allocated resources;
- GPU model and device identity;
- synchronized timing and memory measurements;
- numerical tolerances and measured differences;
- script and input checksums;
- stdout, stderr, exit code, metrics, and provenance;
- persistent and retrieved output locations.

Large transient datasets should remain outside Git. Commit compact
metrics, provenance, selected figures, reports, and carefully selected
checkpoints. Record checksums and archive locations for omitted products.

## Scratch Policy

Use scheduler-provided scratch storage, normally `$TMPDIR`, for transient
products whenever available. Persistent project storage should retain
only evidence and compact scientific products needed for reproduction or
review.

## Documentation Roles

LCProp documentation has four complementary roles:

### Architecture

Long-lived software design and architectural decisions.

### Development

Engineering history, implementation milestones, and validation records.

### Research

Scientific experiments, measurements, figures, and conclusions.

### Operations

Cluster procedures, commissioning guides, prompt templates, and workflow
documentation.

Raw conversations normally remain outside Git. Curated decisions,
reports, and milestone records belong in the repository.

## Design Principles

Prompts should:

- be explicit rather than implicit;
- define success objectively;
- declare authorization and stopping points;
- preserve validated behavior unless change is intentional;
- separate measured evidence from interpretation;
- distinguish operational success from scientific conclusions;
- permit bounded iteration without permitting scope creep;
- favor reproducibility over convenience.

## Long-Term Goal

The library should evolve into a reusable scientific-software operations
manual for LCProp. Future materials, numerical methods, optical models,
and computational workflows should specialize established procedures
rather than invent incompatible operating practices.
