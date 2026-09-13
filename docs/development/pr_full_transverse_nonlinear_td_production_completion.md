# Full-Transverse Nonlinear PR TD Production Completion

## Scope and baseline

This bounded Development and Local Validation milestone completed the existing
full-transverse, fully nonlinear, time-dependent photorefractive production
path.  The required baseline `d21a95e816d6750bab2e7b5e9f18cfc1ece6e5cf`
was superseded by compatible descendant
`24a8b069131022e921d081210ec86845d70d69fd`.  A read-only comparison found one
intervening commit containing only reduced-material research documentation,
evidence, an analysis harness, and its tests; it changed no code or interface
relevant to this milestone.

This record is ordinary local-development evidence.  It makes no authoritative
benchmark or GPU-commissioning claim and therefore did not require a retained
clean-source numerical evidence package.

## Recovered production implementation

The production path already existed under the single
`pr_transverse_timedependent` workflow identity.  A
`PRTransverseRunRequest` defaults to the fully nonlinear response, so old
payloads and positional callers remain nonlinear without specifying a new
field.  The normal registered operation, request/result codecs, Full/Fast
transport, product conversion, local runner, and GUI all use this same path.

The authoritative state is the zero-mean electrostatic potential
`psi(x,y,z,tau)`.  The unchanged equation is

```text
-D_H psi_tau = div_perp M[
    grad_perp(P I) - P I (E_app e_x - grad_perp psi)
]
P = 1 - D_H psi,       tau = t/t0.
```

`E_x`, `E_y`, carrier density, and `E_active` are reconstructed from `psi` by
the existing spectral full-transverse machinery; they are not independently
evolved.  The production default remains first-order spectral IMEX Euler.
The transparent explicit-Euler reference remains available and unchanged.
Periodic spectral derivatives, the zero Fourier gauge, normalization, and all
constitutive definitions are unchanged.

For each requested material interval, production performs one complete optical
pass through the last accepted material volume, constructs the intensity source
for every longitudinal plane, freezes that source for the material update, and
only then accepts the complete candidate `psi`.  The next optical pass sees that
accepted state.  A final complete replay through the last accepted state forms
the authoritative output optical field and source.  This cadence is shared with
the linearized TD path; it was not changed for nonlinear evolution.

## Gap closed

The solver, coupling, UI exposure, persistence, and products did not need a new
workflow.  The missing completion work was nonlinear-specific production
guarding and observability:

- the initial and each complete candidate nonlinear state are now required to
  reconstruct finite fields and strictly positive carrier density before the
  candidate becomes authoritative;
- rejected nonphysical candidates are never accepted;
- nonlinear results report the actual final frozen-source TD RHS RMS and
  maximum, minimum carrier density, physical-state validity, nonlinear material
  update count, complete optical-pass count, exact source cadence, and memory
  policy;
- the diagnostic correctly identifies the fixed-step integrator and does not
  invent Newton, PCG, or linearized-modal iteration fields.

The commissioned linearized path retains its exact reference solver, modal
equations, source-cadence value, response-call semantics, status, and
linearized-only diagnostics.  The new physical acceptance gate is deliberately
nonlinear-only.

## Boundary and bias

The nonlinear Profile v1 electrical boundary remains the existing unbiased
zero-flux profile.  It rejects both nonzero `material.applied_field` and the
periodic biased-current boundary owned by the linearized model.  Consequently,
positive/negative nonlinear bias evolution is not claimed here.  Adding such a
boundary would change scientific scope and requires a separate derivation and
milestone.  The already commissioned linearized periodic biased-current path is
unchanged.

## Scientific validation

All quantities below use repository-normalized material coordinates and time.
They are deterministic local test fixtures, not performance benchmarks.

### Static equilibrium

For a frozen positive `8 x 8` resolved two-dimensional source, `400` nonlinear
IMEX steps of `Delta tau = 0.05` approached the existing authoritative
full-transverse nonlinear static solver:

| Quantity | Result |
| --- | ---: |
| relative L2 difference in `psi` | `2.1415146789332578e-7` |
| maximum absolute `E_x` difference | `7.305766741572128e-9` |
| maximum absolute `E_y` difference | `5.756180808044742e-9` |
| maximum absolute `E_active` difference | `7.305766741572128e-9` |
| final nonlinear TD RHS maximum | `2.2935060556676906e-16` |

The static solver converged under its own authoritative zero-flux residual.
The TD RHS is reported separately because the repository already documents the
continuum-static versus production-derivative discretization distinction.

### Weak and strong comparisons

The independent `potential_rhs()` Taylor oracle continues to establish an
`O(epsilon^2)` nonlinear-minus-linearized RHS remainder.  A weak production
comparison also remains small.  That end-to-end comparison contains a
first-order IMEX versus exact-modal temporal-truncation floor and is therefore
not used by itself to fit the perturbative exponent.  It preserves identical
frozen source and optical output in the controlled zero-gain fixture.

At modulation `0.6`, ten steps of `Delta tau = 0.05` produced a nonlinear versus
linearized potential relative L2 difference of `0.1434279017881015`, while the
nonlinear minimum carrier density remained `0.6368188272195432`.  This is a
bounded dispatch sanity check: it demonstrates genuine nonlinear evolution but
does not establish a universal ordering of gain or harmonics.

### Timestep behavior

At fixed final `tau = 0.5`, solutions using `Delta tau = 0.02`, `0.01`, and
`0.005` gave a successive-difference ratio of `1.9597346260821544`, consistent
with the documented first-order global accuracy.  The existing low-level tests
also show that IMEX damps resolved high modes beyond the explicit-Euler
stability limit.  Timestep choice is part of the request and has no hidden GUI
default dependency.

## Cancellation, storage, and interfaces

Cancellation checkpoints exist before material work, during every optical
longitudinal traversal, and after a complete material candidate.  A new test
cancels from inside the nonlinear update after candidate computation: the
candidate is discarded, `completed_steps` remains zero, and the final replay
returns the initial accepted `psi` and optical field.  Existing tests cover
pre-work cancellation, cancellation after an accepted state, and repeated Stop
behavior.  The final replay always uses the last accepted state.

NumPy `float64/complex128` and `float32/complex64` paths pass.  The established
conditional CuPy seam is retained, but no CUDA execution or commissioning is
claimed.  Backend working storage consists of accepted material and frozen
source volumes with transverse FFT batches; only initial/final longitudinal
material volumes are retained, not a material-time history.

Full transport retains the canonical initial/final material volumes and optical
endpoints.  Fast transport retains optical endpoints, compact diagnostics, and
provenance while omitted material fields reconstruct as unavailable rather
than synthesized.  Existing persistence, transport, product, and old-payload
default-nonlinear tests remain passing.  No schema change was required.

The GUI already exposes Full transverse + Fully nonlinear + Time Dependent as
Validated and constructs the normal production request.  Linearized TD remains
Experimental.  The existing local run-cost guard already accounts
conservatively for transverse grid size, longitudinal planes, TD steps, optical
passes, backend, and precision; it was not weakened.  A practical local smoke
case is `8 x 8 x 1`, one to two material steps, and one optical substep.  Large
three-dimensional, long-time, or high-resolution cases should use the guarded
Slurm/H200 path after a separate commissioning decision.

## Validation and scientific non-change

Focused completion tests: `9 passed`.  The expanded affected suite, including
nonlinear static, commissioned linearized TD, persistence/transport/products,
GUI, and run-cost regressions, passed `176 passed, 2 skipped`.  The complete PR
suite passed `691 passed, 74 skipped`; skips include conditional GPU cases.
Syntax compilation and `git diff --check` also passed.

The candidate boundary is exactly:

- `src/lcprop/pr/transverse/workflow.py`
- `tests/test_pr_transverse_nonlinear_timedependent_completion.py`
- `docs/development/pr_full_transverse_nonlinear_td_production_completion.md`

There is no change to the nonlinear governing equation or integrator algebra,
static nonlinear solver, linearized static/TD references or production physics,
optical propagation, source/coherence physics, persistence or transport
schemas, GUI or run-cost behavior, reduced models, carrier diagnostics, Image
Amplification mathematics, Slurm, or soliton code.  Unrelated dirty-tree work
is excluded and preserved.

H200/CuPy commissioning, nonlinear biased-boundary development, soliton work,
and performance optimization are explicitly deferred.
