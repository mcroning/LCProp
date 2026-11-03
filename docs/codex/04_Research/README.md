# Research Prompt Standards

This directory is reserved for reusable Research standard prompts.
No reusable Research standard has yet been adopted. Task-specific
Research instances belong locally under
`docs/codex/instances/04_Research/`; this absence is intentional.

Every Research instance that may produce documented, committed, compared, or
downstream-consumed numerical results must apply `docs/codex/README.md`,
**Canonical Lifecycle and Evidence Gates**. In particular, authoritative work
must execute from an exact committed SHA in a clean immutable production tree,
with the required checksummed evidence manifest. Dirty imported production
source is a pre-execution hard stop.

An instance may explicitly classify work as exploratory before execution. Such
work may not be promoted to authoritative evidence, scientific comparison, or
downstream input unless it is regenerated under the canonical provenance gate.
