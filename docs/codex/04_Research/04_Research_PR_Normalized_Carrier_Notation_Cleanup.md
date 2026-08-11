# Codex Task: Semantic Carrier-Notation Cleanup Across the 2D PR Reference Record

## Objective

Perform a **notation-only** cleanup of the full-transverse
photorefractive reference work.

Replace the normalized mobile-carrier variable

    n

with

    P = p/p_d

everywhere that **n specifically denotes the normalized mobile-carrier
density** in the recovered 2D hopping model and its
dielectric-anisotropy extension.

Update:

1.  Full Transverse Hopping-Model Recovery report.
2.  BaTiO3 Dielectric-Anisotropy / Crystal-Axis report.
3.  Supporting reference implementation, tests, check script, and
    architecture documentation where this normalized-carrier variable
    appears.

This is strictly a notation and documentation correction.

Do **not** change physics, numerics, algorithms, validation results,
solver behavior, production workflows, or the A7 reduced model.

## Scientific motivation

The Photonics paper uses p for mobile-carrier density and p_d for the
uniform dark/background mobile-carrier density. It does not define n =
p/p_d.

Adopt throughout the archival record:

    P = p/p_d

This is a notation correction only.

## Required edits

-   Replace only occurrences where n represents the normalized carrier
    density.
-   Do not perform a blind global search-and-replace.
-   At the first appearance in each report define P = p/p_d and explain
    that P=1 is the uniform neutral reference state.

## Critical exclusion

Do **not** change any occurrence where n denotes refractive index.

Examples include:

-   n_o
-   n_e
-   n_eff
-   Δn
-   n_NL

This must be a semantic edit.

## Files

Inspect at minimum:

-   src/lcprop/pr/transverse_reference.py
-   tests/test_pr_transverse_reference.py
-   scripts/checks/pr_transverse_reference.py
-   docs/architecture/pr_transverse_reference_model.md
-   docs/development/full_transverse_hopping_model_recovery_vscode.md
-   docs/research/pr_batio3_dielectric_anisotropy_extension_2026-08-10.md

If the transverse-hopping report has moved, locate its canonical
replacement.

## Reports

Update both archival reports consistently.

Preserve all equations, derivations, numerical values, validation
tables, figures, appendices, and conclusions.

## Code

Rename local variables representing normalized carrier density where
practical.

Prefer P or a descriptive name such as carrier_ratio.

Do not introduce API incompatibilities merely to mirror mathematical
notation.

## Tables

Change human-readable labels such as:

-   RMS n-1

to

-   RMS P-1

without changing numerical values.

If changing a machine-readable metric key would break compatibility,
retain the key and update only displayed documentation.

## Validation

Run:

    python -m pytest -q tests/test_pr_transverse_reference.py

Run the repository PR validation suite.

Run the saved 3720 µm snapshot validation if still separate.

Run:

    git diff --check

## Semantic audit

Inspect every remaining standalone n in touched files and classify it
as:

-   refractive index (preserve),
-   unrelated symbol (preserve),
-   normalized carrier density (must become P).

## Non-goals

Do not modify:

-   PR physics
-   dielectric model
-   crystal-axis definitions
-   transport
-   numerical methods
-   production A7
-   architecture
-   material hierarchy

Do not commit or push.

## Completion report

Report:

-   files modified;
-   adopted definition P = p/p_d;
-   compatibility decisions;
-   preserved refractive-index occurrences;
-   normalized-carrier occurrences changed;
-   focused test result;
-   full PR validation result;
-   snapshot validation result;
-   git diff --check result;
-   confirmation that numerical outputs and production A7 are unchanged;
-   confirmation that no commit or push was performed.
