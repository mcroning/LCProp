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

LCProp separates planning and scientific review from implementation and execution to encourage independent review.

After any required architecture decision, the normal change lifecycle is:

```text
Preflight
→ Development + Local Validation
→ Development Self-Review
→ Pre-Commit Review
→ Commit
```

Commissioning and Research follow a committed milestone and may add further
approval gates for preparation, execution, scientific review, scaling, and
permanent research records. Once local validation passes, finish self-review,
pre-commit review, and commit before starting a new milestone or scientific
tangent. An explicitly exploratory tangent may precede commit only when it does
not create authoritative evidence from dirty production source.

Every significant computational result should be reproducible from an
exact Git revision, an identified execution environment, an immutable
launch manifest, and preserved evidence.

## Canonical Lifecycle and Evidence Gates

This section is the shared source of truth for Development, Pre-Commit Review,
Commissioning, and Research prompts. Those procedures should reference these
gates and add task-specific details rather than restating them.

### Preflight

Before implementation, record the baseline SHA, branch, intended candidate
boundary, equations or model being changed, expected scientific non-change
areas, acceptance tests, evidence/provenance strategy, explicit exclusions,
and whether cluster or GPU access is authorized. Stop if the baseline has
drifted or the candidate boundary is ambiguous.

### Authoritative numerical evidence

Any numerical result that may be documented, committed, scientifically
compared, commissioned, or consumed downstream must execute against an exact
committed production SHA in a clean immutable checkout or worktree with empty
`git status --porcelain`. Dirty imported production source is a mandatory
pre-execution hard stop, not a warning. An uncommitted analysis or sweep
harness is permitted only as a separately supplied, SHA-256-checksummed input.

Its evidence manifest must record at minimum:

- production source SHA, clean flag, and exact `status --porcelain` result;
- harness or script SHA-256;
- canonical scientific-data checksum;
- checksums for committed CSV, JSON, figure, and other evidence artifacts;
- backend and precision when scientifically relevant.

Evidence fields encode units or normalization whenever ambiguity is possible,
using names such as `_normalized`, `_mW`, `_um`, and `_rad_per_um`. Missing
required provenance or ambiguous evidence schema blocks Development.

### Development self-review

Before reporting `Passed`, Development verifies the exact candidate manifest,
absence of mixed unrelated hunks, public positional and API compatibility,
persistence/transport backward compatibility where touched, evidence units and
normalization, mode-specific `Experimental` versus `Validated` presentation,
all required scientific qualifications, and executable-test or retained-
evidence support for every acceptance criterion. It also verifies syntax
compilation, `git diff --check`, exact exclusions, and preservation of unrelated
dirty-tree content. A new model axis requires status/warning presentation tests
when it affects user-visible validation status, and a specific experimental
mode must not inherit a broader workflow-level `Validated` label.

For scientific model comparisons, distinguish physical-model,
discretization, resolution, normalization, and approximation-regime effects.
Claims that depend on convergence, asymptotic scaling, or a special
qualification require quantitative evidence rather than prose alone.

### Hard stops and feedback

Stop on dirty production source before authoritative evidence generation, an
unresolved predecessor lifecycle gate, unexpected production-source changes in
an analysis-only milestone, an ambiguous candidate boundary, missing required
units or provenance, or source-baseline drift.

When pre-commit review identifies a recurring defect class, update the
appropriate reusable standard after the milestone so Development catches it
earlier. Formal pre-commit review independently confirms readiness; it should
not be the first routine check for the self-review items above.

Historical prompt instances remain immutable records. Re-executing an instance
uses the current standards: an older instance that relies on an uncommitted
production overlay must be re-instantiated with committed clean source before
its output can be treated as authoritative.

## Directory Organization

```text
docs/codex/
    README.md
    TEMPLATE.md
    01_Architecture/       # tracked reusable standards
    02_Development/        # tracked reusable standards
    03_Commissioning/      # tracked reusable standards
    04_Research/           # tracked standards; currently category README only
    instances/             # local-only execution/history prompts
        01_Architecture/
        02_Development/
        03_Commissioning/
        04_Research/
```

The four numbered top-level directories contain tracked, curated, and
versioned reusable standard prompts: general procedures intended to be
instantiated repeatedly. Task-specific prompts belong under the matching
`instances/` category. Instances are local-only execution/history state,
are ignored by Git, and may accumulate without cluttering the published
standard library. An instance records one concrete application of a
standard procedure, such as a named LC/PR milestone, exact file manifest,
commit SHA, scheduler job, benchmark, or physics configuration.

Every instance filename begins with its mandatory lifecycle category:
`01` for Architecture, `02` for Development, `03` for Commissioning, or
`04` for Research. A second numeric prefix is optional and task-local.
Sequenced tasks begin at `01` and increment only within that task; an
unrelated task may reuse the same category/sequence pair. Standalone
tasks may omit a sequence. Complete filenames, rather than category and
sequence pairs, must be unique. Historical instances are not renumbered
merely to impose a global chronology. New files should be classified by
their actual procedure, not merely by their title or original location.

Standard prompts may deliberately point to an instance as an example or
next workflow only when the reference is labeled as instance-specific.
Do not treat an instance as a reusable standard merely because it was
successful once.

The four established lifecycle categories are retained even when one has
no reusable prompt. Do not add further empty categories in anticipation
of future work; add a new category only when a reusable prompt does not
fit the existing lifecycle.

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
| `02_Development/implementation_and_local_validation.md` | Reviewed  | Implement a bounded local change and validate it              | Approved architecture or implementation request | Local repository changes and tests   | Development Self-Review, then Pre-Commit Review        |
| `02_Development/pre_commit_review.md`                   | Reviewed  | Independently review a completed local implementation         | Completed implementation, local validation, and Development Self-Review | Local read-only inspection only | `Approve commit`                                       |
| `03_Commissioning/cluster_onboarding.md`                | Validated | Characterize a cluster without modifying it                   | Approved cluster access                         | Local and remote read-only discovery | Cluster Checkout Preparation (planned)                 |
| `03_Commissioning/cpu_smoke.md`                         | Validated | Execute one approved CPU commissioning job                    | Approved checkout and scheduler-job preparation | Exactly one CPU submission           | GPU Smoke Test                                         |
| `03_Commissioning/gpu_smoke.md`                         | Validated | Verify CuPy GPU execution against an identified CPU reference | Passed CPU smoke test                           | Exactly one GPU submission           | GPU Validation Benchmark or approved Research instance |
| `03_Commissioning/gpu_validation_benchmark.md`          | Reviewed  | Validate numerical equivalence and GPU performance before/after a bounded implementation change | Passed GPU smoke test and an exact committed, locally validated candidate | Approved bounded GPU submissions | Pre-Commit Review or Research prompt                   |

GPU Validation Benchmark is a conditional post-development
commissioning path for performance-sensitive changes on an already
commissioned GPU. It does not replace GPU Smoke Test and is not a
required prelude to scientific research.

The library currently has no reusable prompt in `04_Research/`.
Research procedures may remain preserved locally as task-specific
instances until a genuinely general research protocol is extracted;
those local instances are intentionally absent from the published Git
repository.

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
16. approval context;
17. approval gate.

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

## Approval Vocabulary

Procedures define how work is performed. Canonical approval phrases
authorize one transition to the next resolved lifecycle state.

| Phrase                        | Required prior state                         | Authorization                                                  | Stopping point                                      |
| ----------------------------- | -------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------- |
| `Approve implementation`      | Approved bounded design or implementation request | Perform the declared local implementation and validation       | After the implementation report                     |
| `Approve review`              | Completed local implementation with resolved review scope | Perform the declared pre-commit review                         | After the review report                              |
| `Approve commit`              | Ready-to-commit review with exact files and message | Commit only the reviewed files with the reviewed message       | After reporting the commit SHA and repository status |
| `Approve push`                | Reviewed commit at HEAD with an unambiguous upstream | Push only that reviewed commit without force                   | After verifying the remote SHA and repository status |
| `Approve cluster preparation` | Approved remote target and expected Git SHA  | Perform only the declared checkout and scheduler preparation   | Before job submission                                |
| `Approve submission`          | Fully resolved immutable launch manifest     | Submit exactly the authorized job count with no automatic retry | At the stopping point declared by the execution prompt |

Each approval is single-use and applies only to the most recently
prepared unambiguous target in the current lifecycle. It does not resolve
placeholders, authorize an adjacent stage, or permit scope expansion.
Codex must stop if repository, manifest, remote, or scheduler state has
drifted from the approved context. Ambiguous approvals require
clarification.

Generic words such as `approved`, `proceed`, or `continue` are not
canonical shorthand unless their exact target and authorized action are
otherwise unambiguous.

`Approve push` does not authorize pulling, rebasing, force-pushing,
changing branches, reconciling divergence, including additional commits,
or beginning remote execution. `Approve submission` is invalid until the
launch manifest contains every required concrete value or `N/A`.

Retrieval remains part of the currently validated submission procedures.
Add `Approve retrieval` only if retrieval later becomes a genuinely
separate lifecycle gate.

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

Authoritative calculations first satisfy **Canonical Lifecycle and Evidence
Gates**. Research and commissioning manifests additionally record, when
applicable:

- Python executable and package versions;
- CUDA module, runtime, and driver;
- backend requested and reported;
- scheduler job ID, node, and allocated resources;
- GPU model and device identity;
- synchronized timing and memory measurements;
- numerical tolerances and measured differences;
- input checksums not already covered by the canonical evidence manifest;
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

Reusable engineering and validation procedures for bounded implementation.

### Research

Reusable procedures for scientific experiments, measurements, and review.

### Operations

Cluster procedures, commissioning guides, prompt templates, and workflow
documentation.

Raw conversations, one-off prompts, private experiment notebooks, unpublished
references, and scheduler output remain outside the public package. Curated
architecture decisions and reusable operating procedures belong here. Public
scientific validation records are retained only when they document supported
package behavior and do not depend on private paths.

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
