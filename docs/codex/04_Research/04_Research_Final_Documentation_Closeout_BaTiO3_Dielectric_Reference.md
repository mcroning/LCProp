# Codex Research Prompt

## Final Documentation Closeout — BaTiO₃ Dielectric-Anisotropy Reference Milestone

**Status:** 04_Research

### Objective

Perform the final **documentation-only** closeout of the completed BaTiO₃ crystal-axis / dielectric-anisotropy reference milestone before the separate notation cleanup (`n → P`), final validation, commit, and push.

**Do not change any physics, numerical implementation, APIs, tests, solver behavior, production workflows, GUI, or GPU paths.**

## Scope

This task exists only to:

- complete the scientific record;
- recover omitted validation metrics;
- improve physical clarity;
- document remaining approximations;
- record the architectural status of the reference implementation.

It does **not** introduce new physics.

## Authoritative sources

- `src/lcprop/pr/transverse_reference.py`
- `tests/test_pr_transverse_reference.py`
- `scripts/checks/pr_transverse_reference.py`
- `docs/architecture/pr_transverse_reference_model.md`
- `docs/research/pr_batio3_dielectric_anisotropy_extension_2026-08-10.md`
- Full Transverse Hopping-Model Recovery report (context only).

## Required work

1. Document the full rotated dielectric tensor, including the nonzero εxz term, and clearly explain the transverse-slice approximation.
2. Recover the missing maximum |Ex| and |Ey| metrics from the existing comparison output (or rerun only the existing small comparison if necessary). State whether maxima are `max(E)` or `max(abs(E))`.
3. Document the exact definitions of the reported relative field differences by inspecting the implementation.
4. Use conservative scientific interpretation. Do **not** claim the historical ky narrowing has been reproduced.
5. Distinguish the implemented rotated crystal dielectric tensor from the more general photorefractive effective dielectric response. State explicitly what is **not** implemented.
6. Preserve `h_y = 1` as the isotropic validation baseline.
7. Add a short architectural-status section explaining that this implementation is the validated PR reference model intended to remain isolated from production workflows while future architecture may relocate PR to a peer material channel.
8. **Do not perform the planned notation cleanup (`n → P`) in this task.**

## Validation

Run only:

```bash
python -m pytest -q tests/test_pr_transverse_reference.py
```

and

```bash
git diff --check
```

No large calculations.

## Completion report

Report:

- files modified;
- recovered maximum-field values;
- exact definitions of the relative-difference metrics;
- where the εxz discussion was added;
- where the architectural-status note was added;
- focused test results;
- `git diff --check` result;
- confirmation that implementation behavior is unchanged;
- confirmation that the notation cleanup was intentionally deferred.

Do not commit.
