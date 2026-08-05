# LCProp Codex Prompt Library

## Purpose

This directory contains the standard prompt library used to conduct
software engineering, numerical validation, cluster execution, and
computational research within LCProp.

These prompts are not intended to replace scientific judgment.
Instead, they encode the project's established workflow so that
repetitive engineering tasks are performed consistently and
reproducibly.

Whenever possible, prompts should be reused and refined rather than
rewritten from scratch.

---

## Scope

This directory contains reusable prompts for Codex-assisted
development, validation, cluster execution, and computational
research. The prompts define operational workflows rather than
project-specific scientific conclusions.

## Philosophy

ChatGPT serves as the architectural and scientific planning partner.
Codex serves as the implementation and execution partner.

Both operate under explicit user supervision and approval gates.

LCProp development follows a staged workflow.

Each stage has explicit acceptance criteria and an approval gate before
moving to the next stage.

Typical progression:

Architecture
→ Local implementation
→ Local validation
→ Commit
→ Cluster checkout
→ Pilot calculation
→ Review
→ Scaling
→ Research report

The goal is to make every significant computational result
reproducible from an exact Git revision and documented execution
environment.

Reproducibility is treated as a primary project deliverable rather than an afterthought.

---

## Prompt Structure

Every prompt should follow the same overall organization.

### Purpose

What is being accomplished?

### Background

Current repository state.

Relevant Git SHA.

Previous milestones.

Scientific motivation.

### Scope

Exactly what Codex may modify.

Just as important, what Codex must *not* modify.

### Acceptance Criteria

Objective conditions required for success.

Examples include:

- tests passing
- numerical tolerances
- validation metrics
- documented provenance
- successful cluster execution

### Workflow

The required sequence of implementation and validation steps.

Examples:

- implement
- validate
- commit
- push
- update cluster checkout
- prepare Slurm job
- stop for approval

### Stop Conditions

Explicitly define where Codex must stop.

Most prompts intentionally terminate before:

- committing
- pushing
- cluster submission
- large parameter sweeps

unless explicitly authorized.

### Required Report

Every prompt concludes with a concise report including:

- files modified
- tests executed
- Git SHA
- validation results
- remaining issues
- recommended next step

---

## Approval Gates

LCProp intentionally uses approval gates.

Typical approval points include:

- implementation complete
- validation complete
- commit ready
- cluster ready
- job prepared
- pilot completed
- scaling recommendation

This minimizes accidental changes while allowing Codex to perform
substantial autonomous work between checkpoints.

Approval gates exist to separate planning, implementation, validation, and scientific interpretation, ensuring that significant decisions remain under explicit user control.

---

## Reproducibility

Research calculations should whenever practical record:

- Git SHA
- branch
- repository cleanliness
- Python environment
- CUDA module
- CuPy version
- Slurm job ID
- node
- GPU model
- execution timings
- memory usage
- numerical tolerances
- provenance
- output checksums

Large transient datasets should remain outside Git.

Only compact scientific products should be committed.

Reproducibility is treated as a primary deliverable, not merely a
post-processing activity.

---

## Scratch Policy

Transient computational products should be written to scheduler-provided
scratch storage (typically `$TMPDIR`) whenever available.

Persistent project storage should contain only:

- provenance
- metrics
- selected figures
- selected checkpoints
- reports
- compact scientific products

This keeps long-term storage manageable while preserving reproducibility.

---

## Documentation

LCProp documentation is organized into three complementary categories.

### Architecture

Long-lived software design.

### Development

Engineering history and implementation milestones.

### Research

Scientific experiments and results.

Raw Codex transcripts should normally remain outside the Git repository.
Curated summaries, reports, and milestones belong in the repository.

### Operations

Cluster procedures, commissioning guides, prompt templates, and
workflow documentation.

---

## Design Principles

Prompts should:

- be explicit rather than implicit;
- define success objectively;
- specify stopping points;
- minimize unnecessary architectural changes;
- preserve backwards compatibility unless intentionally changed;
- separate implementation from scientific interpretation.

---

## Long-Term Goal

The prompt library should evolve into a reusable operational manual for
LCProp.

Future material systems, numerical methods, optical models,
and computational workflows should require prompt specialization rather than entirely new operating procedures.