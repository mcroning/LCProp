# Architecture Review

**Status:** Validated
**Version:** 1.1
**Last reviewed:** 2026-08-05
**Prerequisites:** Proposed architectural change
**Usual next prompt:** Implementation and Local Validation

---

## Purpose

Perform an independent architectural review of a proposed LCProp
design or refactoring before implementation begins.

The objective is to identify opportunities, risks, unnecessary
complexity, hidden coupling, and simpler alternatives while preserving
the project's long-term maintainability.

---

## Required Inputs

- Proposed design document: `{{DESIGN_DOCUMENT}}`
- Expected Git SHA: `{{EXPECTED_GIT_SHA}}`
- Relevant branch: `{{TARGET_BRANCH}}`
- Scope of proposed change: `{{CHANGE_SCOPE}}`

**Stop before acting if any required placeholder remains unresolved.**

---

## Background

Current repository state:

Relevant branch:

Current Git SHA:

Relevant architecture documents:

Previous reviews:

Scientific motivation:

Known constraints:

---

## References (optional)

List architecture documents, design notes, implementation plans,
papers, previous reviews, dependency diagrams, or relevant commits.

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
| Remote read-only commands       | No             |
| Remote interactive file changes | No             |
| Scheduler-created outputs       | No             |
| Environment changes             | No             |
| Job submission                  | No             |
| Automatic retry                 | Not applicable |
| Larger follow-on run            | No             |

The purpose of this prompt is architectural review only.

---

## Scope

### In Scope

Evaluate the proposed architecture with respect to:

- separation of responsibilities;
- module boundaries;
- package organization;
- public APIs;
- dependency structure;
- maintainability;
- extensibility;
- backward compatibility;
- interaction with existing workflows;
- numerical implications;
- documentation implications.

Suggest simpler alternatives where appropriate.

### Out of Scope

Do **not**:

- implement code;
- modify source files;
- redesign unrelated systems;
- introduce abstractions without clear justification.

---

## Acceptance Criteria

The review should:

- identify architectural strengths;
- identify architectural weaknesses;
- distinguish architecture from implementation;
- discuss tradeoffs objectively;
- identify unnecessary complexity;
- recommend improvements where justified;
- prioritize recommendations by expected impact.

Every significant recommendation should reference concrete repository
locations whenever practical (modules, packages, classes, or functions).

---

## Workflow

1. Review the proposal.
2. Compare with the current architecture.
3. Identify strengths.
4. Identify concerns.
5. Consider alternative designs.
6. Evaluate tradeoffs.
7. Recommend a preferred direction.
8. Produce the review report.

Do not modify repository files unless explicitly instructed.

---

## Validation

Evaluate whether the proposed design:

- preserves validated workflows;
- reduces complexity;
- improves maintainability;
- supports future extensions;
- minimizes coupling;
- preserves scientific correctness.

Where appropriate, compare multiple candidate architectures.

---

## Result Classification

This review uses the following task-specific classification vocabulary.

### Accepted

The proposed architecture is suitable for implementation.

### Accepted with Recommendations

The architecture is fundamentally sound but would benefit from
identified improvements.

### Revision Required

Significant architectural concerns should be addressed before
implementation.

### Inconclusive

Additional information is required before a recommendation can be made.

---

## Deliverables

Produce:

- architectural assessment;
- prioritized recommendations;
- identified risks;
- suggested simplifications;
- alternative designs (if appropriate);
- recommended implementation strategy.

Simple dependency sketches or module diagrams are encouraged when they
clarify the discussion.

---

## Stop Conditions

Stop after completing the architectural review.

Do **not** implement recommendations.

Do **not** modify source code.

---

## Required Report

Summarize:

- overall assessment;
- architectural strengths;
- architectural concerns;
- required changes;
- recommended improvements;
- optional future ideas;
- implementation readiness.

Clearly distinguish:

- required changes;
- recommended improvements;
- optional future work.

Whenever practical, cite the affected files, packages, classes, or
functions supporting each recommendation.

---

## Approval Gate

Wait for explicit approval before:

- implementing recommendations;
- reorganizing packages;
- changing public APIs;
- beginning implementation.

---

### Prompt Notes

This prompt exists to obtain an independent architectural assessment
before implementation begins.

Recommendations should favor:

- simplicity;
- clear separation of responsibilities;
- long-term maintainability;
- preservation of validated scientific behavior;
- minimal unnecessary abstraction.
