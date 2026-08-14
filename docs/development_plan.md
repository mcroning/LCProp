# LCProp Development Plan

This document is forward-looking. Current implementation facts belong in
[`STATUS.md`](STATUS.md), and architectural decisions belong in
[`architecture/LCProp_Target_Architecture.md`](architecture/LCProp_Target_Architecture.md).

The peer-material ownership migration is complete. New work should preserve
that baseline rather than reopen package boundaries as part of unrelated
physics or numerical changes.

## Public-package priorities

1. **Stabilize public documentation and package metadata.**
   Keep installation, examples, application entry points, architecture links,
   and status synchronized with the repository.

2. **Define release validation.**
   Establish the supported Python/backend matrix, distinguish CPU tests from
   explicitly commissioned CUDA tests, and record exact commands and results
   for release candidates.

3. **Expand small reproducible examples.**
   Add bounded LC and PR examples that exercise canonical imports and complete
   request/workflow/result paths without requiring research-scale resources.

4. **Document compatibility policy.**
   Identify historical import surfaces, define their support window, and keep
   canonical ownership objectively testable through module provenance and
   object identity.

5. **Improve packaging completeness.**
   Review project metadata, licensing, supported entry points, and distribution
   contents before the first public release. Do not infer or add legal metadata
   without an explicit project decision.

6. **Maintain reusable operational procedures.**
   Keep `docs/codex/` limited to reusable architecture, development,
   commissioning, and research procedures. Store one-off prompts and private
   experiment histories outside the public package.

## Numerical and scientific work

Numerical or physical extensions should be proposed and validated as separate
bounded milestones. In particular:

- keep production LC and PR equations material-owned;
- keep the PR transverse reference isolated until a reviewed physics decision
  promotes any part of it;
- justify new shared abstractions with at least two concrete consumers;
- preserve prepared-response semantics and backend-native execution;
- separate small automated regression tests from research-scale studies.

Potential future optical models, material models, vector propagation,
nonlocal responses, and external plugin discovery remain design decisions, not
implicit roadmap commitments.

## Milestone discipline

Each implementation milestone should:

- state its approved scope and non-goals;
- preserve the canonical dependency direction;
- add focused regression coverage;
- run the complete relevant material suite;
- distinguish local CPU validation from GPU commissioning;
- pass an independent pre-commit review;
- leave durable public documentation only when it benefits package users or
  future contributors.
