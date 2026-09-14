# Reduced-static positive-carrier candidate-gate evaluation

## Decision

**KEEP the strict positive-carrier candidate gate for a later bounded production
implementation.** The evidence supports it as a defensive physical-admissibility
gate, not as an improved nonlinear root finder.

For the reduced model, Gauss's law reconstructs normalized carrier density as

\[
n=1+\partial_{x'}E.
\]

Variant B therefore rejects a Newton candidate exactly when
\(\min n\leq0\), then continues the existing backtracking schedule. It adds no
margin, clipping, projection, flooring, regularization, maximum-residual rule, or
tolerance change. The production solver remains unchanged at this milestone.

## Authoritative evidence

The finalized harness was supplied separately to an isolated clean checkout at
exact production commit `72a4d77cb42cd3d7ee3a3d7fe143f6b5f4a10f7c`.
`git status --porcelain=v1 --untracked-files=all` was empty before and after the
run, and imports were verified to resolve to that checkout. Provenance is:

- harness SHA-256:
  `2e36c4d35f9e7b1d6b242ba5430a40a81d90a40922d197fa3127af45c6f7e9f3`;
- separately supplied 3720 um snapshot SHA-256:
  `e2dc8c9aede6096e8c01745053c895c5f2e2635f67af8cb18c3d52bcca69b067`;
- committed prior maximum-residual evaluation input SHA-256:
  `cf4df65bab6e4dcd4e5c2907fbc62c5f9432e2b6add7dc5bbb21473ebf44ea80`;
- compact result SHA-256:
  `3064bc49a507bd0c642c4e5a2e50e88d148ad61bcd340996c6ebb5744b6e368b`;
- canonical scientific-data SHA-256:
  `2821e75e5c3259beb3a0fd5ad932de50fbbcffaa7e99235cccbe14bd9ce8e199`.

The canonical checksum is the sorted compact JSON encoding after recursively
omitting only `wall_time_s`. The adjacent manifest records backend, precision,
environment, source status, external inputs, repository-file hashes, and artifact
checksum closure.

## A/B method

The harness reuses the seven fixtures committed by the preceding solver-policy
study. Variant A reproduces production `hopping_rhs`, Jacobian construction,
cyclic Newton direction, RMS Armijo rule, convergence tolerances, and backtracking
schedule. Its field, iteration records, status, and convergence flag are checked
against the actual production solver. Variant B runs the same loop and differs
only by requiring strict positive candidate carrier density before the unchanged
production acceptance rule can accept a step.

For every candidate, retained evidence includes its step scale, residual RMS and
maximum, carrier minimum, production acceptance decision, gate decision, and
whether a smaller admissible step is subsequently accepted. Every accepted
state's carrier minimum is retained.

## Ordinary-fixture non-interference

| Fixture | A / B status | A / B Newton iterations | Carrier rejections | Final \(n_{\min}\) | Field difference |
|---|---|---:|---:|---:|---|
| weak periodic, \(m=0.01\) | converged / converged | 3 / 3 | 0 | 0.996521 | bitwise zero |
| moderate periodic, \(m=0.4\) | converged / converged | 4 / 4 | 0 | 0.881233 | bitwise zero |
| high periodic, \(m=0.95\) | converged / converged | 4 / 4 | 0 | 0.697668 | bitwise zero |
| low-frequency high-contrast periodic | converged / converged | 3 / 3 | 0 | 0.991706 | bitwise zero |
| production two-beam frozen optical source | converged / converged | 3 / 3 | 0 | 0.979641 | bitwise zero |

The gate never activates on an ordinary fixture. Accepted steps, RMS and maximum
residual histories, fields, means, and carrier histories are identical. Relative
L2 and maximum absolute field differences are exactly zero. No tested valid
physical convergence is blocked or perturbed.

## Difficult cases

| Fixture | Policy | Status | Newton iterations | Line-search rejects | Carrier-gate rejects | Final residual RMS / max | Final \(n_{\min}\) |
|---|---|---|---:|---:|---:|---:|---:|
| deterministic stress | production | failed | 7 | 48 | 0 | 0.0465583 / 1.34016 | -0.547956 |
| deterministic stress | positive-carrier gate | failed | 15 | 158 | 158 | 0.0827934 / 0.451324 | **1.78066e-9** |
| historical 3720 um | production | failed | 7 | 76 | 0 | 0.000761582 / 0.700066 | -0.115913 |
| historical 3720 um | positive-carrier gate | failed | 14 | 162 | 162 | 0.000527633 / 0.155143 | **2.64639e-8** |

The carrier-gated solver does not find a physical root in either case. It first
uses smaller admissible steps, then approaches the boundary \(n_{\min}=0\) and
fails cleanly when all 21 attempted step scales through
`9.5367431640625e-7` remain nonpositive. Every retained accepted state remains
strictly physical. The stress case has 156 gate rejections that the RMS Armijo
rule alone would accept; the historical case has 149.

For the stress fixture, the first production-acceptable full step would lower RMS
from 1.4944277501 to 0.5712171490 but gives \(n_{\min}=-0.9080547795\).
Backtracking to 0.5 is admissible and accepted. At termination, the smallest
candidate has \(n_{\min}=-5.06605\times10^{-9}\), so no admissible step remains.

For the historical fixture, the first production-acceptable full step lowers RMS
from 0.00175160848 to 0.000773540120 but gives
\(n_{\min}=-0.114571131\). Backtracking to 0.5 is admissible and accepted. At
termination, the smallest candidate still has
\(n_{\min}=-3.51040\times10^{-8}\).

Terminal fields differ from production by relative L2 values 0.847381 for the
stress case and 0.0742525 for the historical case. These are stalled iterates,
not alternate converged physical solutions, and must not be interpreted as a
physical solution comparison. No independently accepted physical root is known
for either fixture.

## Direct comparison with the rejected maximum-residual guard

| Fixture | Policy | Status reported | Iterations / rejects | Final \(n_{\min}\) | Interpretation |
|---|---|---|---:|---:|---|
| stress | production | failed | 7 / 48 | -0.547956 | visible failure after accepting nonphysical states |
| stress | maximum-residual guard | **converged** | 7 / 5 | -0.00695994 | false physical success |
| stress | positive-carrier gate | failed | 15 / 158 | **1.78066e-9** | honest physical-boundary failure |
| historical | production | failed | 7 / 76 | -0.115913 | visible failure after accepting nonphysical states |
| historical | maximum-residual guard | **converged** | 33 / 124 | -0.0169404 | false physical success |
| historical | positive-carrier gate | failed | 14 / 162 | **2.64639e-8** | honest physical-boundary failure |

The positive-carrier gate solves the validity-reporting pathology exposed by the
previous experiment: it cannot report convergence after accepting a negative
carrier candidate. It does not solve the underlying difficult nonlinear problem;
both fixtures stagnate at the physical boundary. That distinction is why the
recommendation is KEEP as a defensive acceptance invariant, not a claim of
improved convergence robustness.

## Scope

No reduced nonlinear equation, residual, Jacobian/Newton algebra, tolerance,
optical propagation, full-transverse model, linearized model, or TD code changed.
In particular, `src/lcprop/pr/static.py` remains byte-identical to the baseline.
The 62 MiB historical snapshot remains excluded from the permanent boundary.
