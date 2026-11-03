# Reduced-static maximum-residual guard evaluation

## Decision

**REVERT the proposed `candidate_max <= residual_max` line-search guard.**

The guard is numerically inert on every ordinary scientific fixture tested. It
does redirect two difficult trajectories to algebraically converged roots, but
both roots have negative normalized carrier density

\[
n = 1 + \partial_{x'}E,
\]

and therefore are not physically admissible reduced-model solutions. The guard
would turn a visible solver failure into apparent success without establishing a
physical state. This does not meet the scientific bar for a production change.

The production hunk was consequently removed. The established RMS Armijo rule,
residual definition, Jacobian/Newton algebra, tolerances, and equations remain
unchanged from baseline `7c936425ac1f77fbfd2634b8daae21a37466ecbd`.

## Authoritative clean-source regeneration

The seven-fixture evaluation was regenerated from an isolated clean checkout at
exact production commit `7c936425ac1f77fbfd2634b8daae21a37466ecbd`.
`git status --porcelain=v1 --untracked-files=all` was empty before and after the
run. The finalized harness was supplied separately with SHA-256
`da6ac0f2717fd7768b2df94d307a3cceeab71181058b75794e5529875ce21cc7`.

The retained 3720 um input was also supplied separately. It is the fixed-source
failure snapshot from
`outputs/pr_cw_diagnostic_2266990/products/Cw/first_pass_incremental/`
`failure_snapshot.npz`, has size 64,521,555 bytes, and has SHA-256
`e2dc8c9aede6096e8c01745053c895c5f2e2635f67af8cb18c3d52bcca69b067`.
It is an ignored historical input, not part of the permanent candidate boundary.

The compact retained result is
`results/pr_reduced_static_max_residual_guard_evaluation_2026-09-14/`
`evaluation.json`, with SHA-256
`cf4df65bab6e4dcd4e5c2907fbc62c5f9432e2b6add7dc5bbb21473ebf44ea80`.
The canonical scientific-data checksum is
`c2b7929be1a8b0628bff89489abed75b565f7ff0c5769c593536d699b01dfd9e`;
it is the sorted compact JSON encoding after recursively omitting only
`wall_time_s`. The adjacent manifest records the source, input, environment,
schema, and artifact checksum closure.

All scientific findings and previously recorded values from the exploratory run
reproduced exactly, including statuses, iteration/backtrack counts, accepted
steps, final residuals, carrier minima, field SHA-256 values, and the cited
rejection events. The exploratory run did not retain a checksum-closed full JSON
artifact, so this is not a claim of bitwise comparison for every unreported trial.
Only noncanonical single-run wall times changed. Across the first six fixtures the
clean-minus-exploratory timing changes ranged from -0.00119733 s to +0.000107416 s.
For the 1024-square snapshot, A changed by -0.503809 s and B by -0.863904 s. These
timing variations do not alter the solver-method conclusion.

## A/B method

The local harness runs the same production `hopping_rhs`, fixed-intensity
Jacobian, cyclic Newton direction, termination tolerances, and backtracking
schedule in two branches:

- A: the established RMS Armijo acceptance rule;
- B: the same rule plus nonincrease of the maximum absolute residual.

The predicate is harness-only and is not a public solver option. With the guard
temporarily present, branch B reproduced the production field and iteration
records bitwise. After its removal, branch A reproduces the production result
bitwise. Every attempted line-search step retains RMS, maximum norm, scalar merit,
required Armijo RMS, acceptance decisions, and step size.

## Results

The table reports Newton iterations and rejected line-search trials. Timings are
single local measurements and are included only to exclude an obvious cost
regression; they are not benchmarks.

| Fixture | A status; iterations / rejects | B status; iterations / rejects | A / B wall time (s) | Guard activations | Minimum final carrier A / B | Field comparison |
|---|---:|---:|---:|---:|---:|---|
| weak periodic, \(m=0.01\) | converged; 3 / 0 | converged; 3 / 0 | 0.01515 / 0.01452 | 0 | 0.996521 / 0.996521 | bitwise equal |
| moderate periodic, \(m=0.4\) | converged; 4 / 0 | converged; 4 / 0 | 0.01908 / 0.01867 | 0 | 0.881233 / 0.881233 | bitwise equal |
| high periodic, \(m=0.95\) | converged; 4 / 0 | converged; 4 / 0 | 0.01883 / 0.01868 | 0 | 0.697668 / 0.697668 | bitwise equal |
| low-frequency high-contrast periodic | converged; 3 / 0 | converged; 3 / 0 | 0.01400 / 0.01402 | 0 | 0.991706 / 0.991706 | bitwise equal |
| production two-beam frozen optical source | converged; 3 / 0 | converged; 3 / 0 | 0.00298 / 0.00297 | 0 | 0.979641 / 0.979641 | bitwise equal |
| bounded deterministic stress | failed; 7 / 48 | converged; 7 / 5 | 0.00496 / 0.00304 | 3 | -0.547956 / **-0.006960** | not comparable as physical roots |
| retained Cw snapshot at 3720 µm | failed; 7 / 76 | converged; 33 / 124 | 2.221 / 5.224 | 124 | -0.115913 / **-0.016940** | not comparable as physical roots |

For the five cases in which both methods converge, the complete trajectories and
final fields are bitwise identical: relative L2 and maximum absolute field
differences are exactly zero. Their mean-field constraints and carrier minima are
also identical. The guard neither slows nor changes those solutions because it
never activates.

The stress and retained Cw states both begin with positive carrier minima
(0.145834 and 0.234378 respectively). Branch B reaches the requested residual
tolerances, but crosses into negative carrier density. Algebraic residual
convergence is therefore not evidence of physical robustness in these cases.

## Rejected-step analysis

The first stress step illustrates the intended numerical effect. The original
rule accepts a full step whose RMS falls from 1.4944277501 to 0.5712171490 and
whose scalar merit falls from 1.1166571501 to 0.1631445157, while its maximum
residual rises from 7.5212248340 to 17.2968677478. Branch B rejects it and later
converges algebraically after steps
`0.5, 0.0625, 1, 1, 1, 1, 1`.

For the historical snapshot, the first full step lowers RMS from
0.00175160847943 to 0.000773540120111 and merit from
1.53406613260e-6 to 2.99182158711e-7, but raises the maximum residual from
0.0597436071690 to 0.746999567280. Branch B ultimately uses 33 Newton iterations
and 124 rejected trials, compared with branch A's failure after seven iterations
and 76 rejected trials. Its last guard-specific rejection at iteration 29 again
reduces RMS (0.000170287741400 to 0.0000279193304058) while increasing the maximum
(0.0230730800546 to 0.0284587452533).

Thus temporary maximum-norm increases are accepted by the established merit rule
and precede failure in the two triggering cases; no tested case shows that such an
increase is useful for eventual convergence. Conversely, monotone maximum norm
alone is not a physical-state safeguard: it still permits negative carrier
density. A future robustness change would need an explicitly justified physical
admissibility policy rather than treating maximum-residual monotonicity as a
surrogate.

## Prior harmonic evidence

The clean-source regeneration of the one-period nonlinear harmonic study already
provided an independent inert-case check. Its clean and provisional summary CSVs
have the same SHA-256,
`9a91a09cc5cdf0c9cd9474b1b643d6c6ffba6016113a9220a7733f2feee2a10c`, and all
reported metrics and fitted exponents are bitwise identical. The provisional run
had imported the dirty guarded solver, so the guard did not affect that fixture or
its scientific conclusions.

## Scope and reproducibility

This is a bounded authoritative Development solver-method comparison, not a new
production sweep. The executable harness is
`scripts/checks/pr_reduced_static_max_residual_guard_evaluation.py`; it regenerates
all comparisons from the baseline production equation and the test-only
acceptance predicate. No optical propagation, carrier physics,
reduced/full-transverse operator, linearized model, TD code, or solver tolerance
was changed.
