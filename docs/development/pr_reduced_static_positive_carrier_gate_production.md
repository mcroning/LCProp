# Reduced-static positive-carrier production gate

## Decision and scope

The reduced x-only fully nonlinear static Newton solver now treats strict
positive normalized carrier density as a physical invariant of every accepted
line-search candidate. This adopts the policy evaluated in commit
`45e59078cdc112c27f998dce46a87f7b87580e39` without adding a user-facing option
or changing result schemas.

For the reduced model, the existing centered periodic derivative reconstructs

\[
n_{\mathrm{candidate}} = 1 + \partial_{x'}E_{\mathrm{candidate}}.
\]

The production acceptance order is:

1. construct the Newton candidate and reject it if its field is non-finite;
2. compute the canonical centered-difference derivative and require exactly
   \(\min n_{\mathrm{candidate}}>0\);
3. only for an admissible candidate, evaluate the existing residual and apply
   the unchanged RMS/Armijo criterion.

Rejected candidates continue the existing halving schedule. Exhausting that
schedule retains the last accepted state and returns the established
`line_search_failed` non-convergence result. The solver does not report
convergence merely because the retained state is physical.

## Scientific interpretation

This is a defensive physical-admissibility safeguard, not a convergence
accelerator. The gate prevents a nonpositive-carrier candidate from becoming an
authoritative Newton state. It does not make the nonlinear Newton solver find
roots that it could not otherwise find.

The earlier maximum-residual proposal was rejected because it could converge to
negative-carrier roots. Production acceptance is now physical admissibility plus
the existing RMS/Armijo merit rule. It does not require a nonincreasing maximum
residual.

No carrier clipping, projection, flooring, regularization, epsilon, or
dtype-dependent positivity margin is applied. The criterion is strict
\(n_{\min}>0\), exactly as evaluated.

## Validation against the committed evaluation

The five ordinary fixtures never activate the gate. Their final fields,
convergence status, accepted steps, iteration records, and RMS and maximum
residual histories remain bitwise identical to the pre-gate baseline.

The two difficult fixtures reproduce the committed carrier-gated endpoints:

| Fixture | Status | Newton iterations | Final accepted \(n_{\min}\) |
|---|---|---:|---:|
| deterministic stress | `line_search_failed` | 15 | `1.7806636165573764e-9` |
| historical 3720 um snapshot | `line_search_failed` | 14 | `2.6463893965100738e-8` |

Neither case finds a new physical root. Both fail honestly after no remaining
backtracked candidate satisfies physical admissibility and the existing merit
criterion, while the last accepted state remains strictly physical. The large
historical snapshot is supplied separately for local validation and remains
outside Git.

## Scientific non-change and limitations

The reduced hopping equation, carrier definition, derivative discretization,
Newton direction and Jacobian, residuals, tolerances, iteration and backtracking
limits, optical propagation, coherence/source physics, linearized models,
full-transverse models, TD solvers, carrier-power diagnostics, and Image
Amplification mathematics are unchanged.

The validated cases show non-interference on five ordinary fixtures and honest
failure on two known pathologies. They do not establish improved global Newton
convergence or prove that a physical root exists for either difficult fixture.
