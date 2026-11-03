# Biased Full-Transverse Linearized PR Reference Solver

**Result:** Passed scoped validation; complete suite has two unrelated dirty-tree UI failures

**Date:** 2026-09-09

**Branch:** `feature/pr-second-order-static`

**Baseline HEAD:** `6af52b09326add91f2449214b400711247ee0faa`

## Scope

This milestone implements one isolated NumPy, frozen-intensity material
reference for the accepted periodic bulk, biased, current-carrying
full-transverse PR profile. It does not register a workflow or connect the
solver to optics, GUI, persistence, transport, Image Amplification, soliton
machinery, GPU execution, Slurm, or the cluster.

The current Product-level scientific contract is
`docs/science/pr_model_contracts.md`. Historical derivations, the superseded
blocker analysis, and paper-specific comparisons are retained in Research and
are not Product documentation dependencies.

## Implementation

The reference lives in:

```text
src/lcprop/pr/transverse/linearized_reference.py
```

It is intentionally imported directly and is not exported as a production
operation. Its API consists of:

```text
PRBiasedLinearizedReferenceSpec
PRBiasedLinearizedFourierSymbol
PRBiasedLinearizedReferenceResult
biased_linearized_fourier_symbol(shape, *, spec)
solve_pr_biased_linearized_reference(intensity, *, spec)
total_fields_from_perturbation(result)
```

The model identifier is:

```text
pr_full_transverse_periodic_biased_linearized_reference_v1
```

The result records the separate nonlinear-profile identifier and the
`fixed_harmonic_mean_field` ensemble.

### Input contract

The input is a two- or three-dimensional NumPy float64 array whose final axes
are `(Nx, Ny)`. It is the complete normalized physical transport intensity and
must be finite and strictly positive. The immutable spec requires finite:

- positive reference intensity `I0`;
- normalized applied field of either sign;
- positive normalized grid spacings;
- positive `m_y` and `h_y`.

The solver defines `delta_I = intensity - I0`. A nonzero mean is allowed and
reported. The mean is explicitly removed before the material FFT because it
does not create a perturbation potential at fixed reference state. It instead
changes first-order mean current by:

```text
delta <J_x> = -E_app * <delta_I>
delta <J_y> = 0
```

### Fourier response

For every resolved nonzero mode the implementation uses exactly:

```text
a_M = k_x^2 + m_y * k_y^2
a_H = k_x^2 + h_y * k_y^2

D = I0 * (a_M * (1 + a_H) + i * E_app * k_x * a_H)

delta_psi_hat = (
    -(a_M + i * E_app * k_x) / D
) * delta_I_hat
```

The fields are reconstructed without double-counting the mean applied field:

```text
delta_E_x_hat = -i * k_x * delta_psi_hat
delta_E_y_hat = -i * k_y * delta_psi_hat
delta_P_hat   = a_H * delta_psi_hat
```

The optional total-field helper returns `E_app + delta_E_x` and
`delta_E_y` separately from the perturbation result.

### Repository spectral convention and zero modes

The implementation reuses
`lcprop.pr.transverse.transport.spectral_wavevectors()`. This is important:
on even grids, the canonical first-derivative convention sets each Nyquist
symbol to zero. The discrete joint-null set can therefore contain four modes,
not only the continuum constant mode.

The response kernel is set to zero on every joint derivative-null mode. The
constant mode is the potential gauge; the extra even-grid modes are explicitly
identified as production derivative-null modes. No zero denominator is ever
divided. Odd grids have only the constant joint-null mode.

### Structural operation count

Each solve performs:

- one forward 2-D FFT of zero-mean `delta_I`;
- four inverse 2-D FFTs for `delta_psi`, `delta_E_x`, `delta_E_y`, and
  `delta_P`;
- pointwise construction/application of one analytic response kernel.

The counts are recorded as `forward_fft_count=1` and
`inverse_fft_count=4`. No timing-based performance claim is made.

## Scientific validation

Focused tests are in:

```text
tests/test_pr_biased_linearized_reference.py
```

They cover zero-bias x and y modes, positive- and negative-bias x modes, an
oblique biased mode, potential/field/carrier amplitude and phase, gauge,
bias-reversal symmetry, unbiased reduction, the qualified reduced limit,
operator nonsingularity, zero-mode mean current, batched planes, input
validation, provenance, operation counts, and the nonlinear Taylor remainder.

### Bias reversal

For the same discrete wavevector:

```text
K(-E_app, k_x, k_y) = conjugate(K(E_app, k_x, k_y))
```

The test verifies exact kernel and denominator conjugation. A real cosine
mode correspondingly reverses the bias-induced sine/phase component.

### Unbiased limit

At `E_app=0`, precision-level comparison verifies:

```text
K(0, k_x, k_y) = -1 / (I0 * (1 + a_H))
```

on every resolved mode. The mobility symbol cancels exactly at the algebraic
level, recovering the previously derived unbiased screened-Poisson response.

### Reduced x-only qualification

A y-independent mode verifies:

```text
delta_E_hat = (
    (i*k - E_app) / (I0 * (1 + k^2 + i*E_app*k))
) * delta_I_hat
```

The test explicitly sets `I_b = I0`, so the canonical A7 equilibrium field
`E_app * I_b / I0` equals the transverse fixed mean field. No unconditional
A7 equivalence is claimed.

### Nonlinear biased Taylor check

The nonlinear oracle is the existing low-level biased
`lcprop.pr.transverse.transport.potential_rhs()`, not the incompatible
pointwise-zero-flux static solver. For an oblique mode with `I0=1.3`,
`E_app=0.7`, `m_y=1.4`, and `h_y=2.1`:

| epsilon | RMS nonlinear potential-rate residual |
|---:|---:|
| 1.000e-3 | 1.892889583868e-7 |
| 5.000e-4 | 4.732223955813e-8 |
| 2.500e-4 | 1.183055990849e-8 |
| 1.250e-4 | 2.957639973484e-9 |

Observed consecutive convergence orders were:

```text
2.000000001
1.999999998
2.000000002
```

This is the required second-order discarded remainder.

### Nonsingularity

Representative odd and even spectral grids were tested at negative, zero, and
positive bias with anisotropic `m_y=0.7`, `h_y=3.1`. Every resolved mode had:

```text
real(D) > 0
abs(D) > 0
```

For the sampled even `(32, 34)` grid, 1084 modes were resolved, four were
joint derivative-null modes, and the minimum resolved real denominator was
`0.5425138404647717` for every tested bias. Bias changes only the imaginary
part of this lower-bound check.

## Local validation record

Focused solver tests:

```text
python -m pytest tests/test_pr_biased_linearized_reference.py -q
26 passed
```

Relevant low-level and PR non-regression tests:

```text
python -m pytest \
  tests/test_pr_transverse_timedependent_transport.py \
  tests/test_pr_transverse_reference.py \
  tests/test_pr_transverse_static.py \
  tests/test_pr_static.py -q
69 passed, 3 skipped
```

The exact project interpreter was
`/Users/mcroning/miniforge3/envs/lcprop/bin/python` (Python 3.12). Broader
suite, compilation, and diff checks follow.

Syntax and whitespace checks:

```text
python -m py_compile \
  src/lcprop/pr/transverse/linearized_reference.py \
  tests/test_pr_biased_linearized_reference.py
passed

git diff --check
passed
```

The three new untracked deliverables were also scanned directly for UTF-8
decoding, control characters, and trailing whitespace; all passed. Ruff is not
installed in the project environment (`No module named ruff`), and no
environment change was authorized.

An initial unscoped `pytest -q` invocation was invalid for this workspace: it
recursively collected archived source/test copies under untracked `results/`
directories and stopped with 213 duplicate-module import mismatches. No test
body failed in that attempt and no archived artifact was removed or modified.

The correctly bounded complete repository suite was then run as:

```text
python -m pytest tests -q
1291 passed, 58 skipped, 2 failed in 524.61 s
```

The two failures were:

```text
tests/test_image_pane.py::test_image_pane_lists_2d_fields
tests/test_image_pane.py::test_timedependent_image_pane_lists_only_initial_and_final_fields
```

Both assertions expect exactly four image-selector entries, while the current
dirty tree supplies two additional far-field entries. The new reference module
is not imported by `ImagePane`, static LC execution, or TD LC execution, and
the focused/relevant suites pass. These are therefore recorded as unrelated
pre-existing dirty-tree UI expectation failures and were not changed within
this milestone.

## Deferred work

The milestone deliberately defers:

- a nonlinear periodic current-carrying static reference solver;
- direct nonlinear steady-state field comparison beyond the authoritative
  Taylor residual gate;
- optical source refresh and self-consistency;
- production workflow or public operation registration;
- CuPy and GPU validation;
- persistence, transport, GUI, Image Amplification, and soliton integration;
- finite-electrode physics.

## Scientific non-change verdict

Existing nonlinear transverse equations and both canonical static/TD workflows
are unchanged. The new code is a separately named frozen-intensity reference
module. It does not alter defaults, bias validation, pointwise-zero-flux
semantics, optical propagation, or any production result contract.

## Next standard prompt

After this implementation receives review authorization, instantiate:

```text
docs/codex/02_Development/pre_commit_review.md
```

for exactly the new reference module, focused tests, and this development
record. Do not include unrelated dirty-tree files.
