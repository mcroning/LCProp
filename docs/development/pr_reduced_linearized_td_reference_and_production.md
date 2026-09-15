# Reduced x-only linearized PR time dependence

## Scope and status

This milestone completes the production cell **time dependent × reduced
x-only drift/diffusion × linearized response**.  Its physical response is
linearized about an explicit positive `I0`; its software-evidence status is
**Locally validated**.  CuPy support is implemented through the normal backend
abstraction but is not GPU-commissioned here.

The authoritative transient state remains the normalized reduced-model field
`E(z,x,y,tau)`.  The leading `z` dimension is a batch of independently updated
planes during each frozen-source material interval; it is not a material
history.  The existing seven production cells, optical propagation, source
construction, and nonlinear equations and integrators are unchanged.

## Recovered production equation

The committed reduced nonlinear TD implementation evolves

```text
dE/dtau = E_app I_b - (E I - D1 I)(1 + D1 E) + I D2 E .
```

The transverse coordinate is `x' = k_D x`; `E` and material time `tau` retain
the established reduced-model normalization.  `I` is optical intensity divided
by the sum of individual channel peak intensities after coherent-group source
formation, plus dark and configured uniform background intensity.
`I_b` is the explicit normalized background intensity used only in the bias
source.  `E_app` is the configured reduced-model applied field, `tau` is the
repository's normalized material time, and `D1`, `D2` are periodic centered
differences along x.  Existing nonlinear production uses explicit Euler or
the semi-implicit trapezoidal predictor/corrector.

For a uniform reference intensity `I0 > 0`, the uniform equilibrium is

```text
E0 = E_app I_b / I0 .
```

Writing `I = I0 + delta_I` and `E = E0 + delta_E` and retaining first-order
terms gives

```text
d(delta_E)/dtau =
    -E0 delta_I - I0 delta_E + D1 delta_I
    - E0 I0 D1 delta_E + I0 D2 delta_E .
```

This derivation comes directly from the reduced production RHS rather than
from an x-only restriction of the full-transverse potential equation.

## Discrete modal solution

For FFT convention `f_hat[j] = sum_n f[n] exp(-2 pi i j n/N)`, with inverse
including `1/N`, let `theta_j = 2 pi j/N`.  The exact centered-difference
symbols are

```text
k1 = sin(theta_j)/dx'
k2_squared = 4 sin(theta_j/2)^2/dx'^2
D1 -> i k1
D2 -> -k2_squared .
```

Every frozen-source mode therefore obeys

```text
d(delta_E_hat)/dtau = -Lambda delta_E_hat + S delta_I_hat
Lambda = I0 (1 + k2_squared + i E0 k1)
S = -E0 + i k1 .
```

The exact update is valid and is the canonical integrator for this cell:

```text
delta_E_hat(tau+dt) = delta_E_hat_inf
    + (delta_E_hat(tau)-delta_E_hat_inf) exp(-Lambda dt)
delta_E_hat_inf = (S/Lambda) delta_I_hat .
```

No epsilon or modal regularization is used.  Since
`Re(Lambda) = I0(1+k2_squared) > 0`, the uniform reference is linearly stable
for every mode.  Bias leaves the decay rate's real part unchanged and changes
its imaginary drift; reversing bias conjugates `Lambda`, changes
`S` to `-conj(S)`, and gives the corresponding reflected/sign-reversed real
field symmetry for real cosine forcing.

The constant mode is finite: `Lambda=I0`, `S=-E0`.  On even grids the Nyquist
mode is a `D1` null but not a `D2` null: `k1=0` and
`k2_squared=4/dx'^2`.  It is therefore evolved directly.  In the unbiased
case its source coefficient is zero; arbitrary initial Nyquist content still
decays.  Odd grids have no Nyquist bin.

## Scientific gates

`S/Lambda` is algebraically identical to the committed reduced linearized
static response kernel.  In the resolved float64 fixture, initializing the TD
solver at that static solution changed `E` by at most
`5.55e-17`; the independently reconstructed equilibrium was bitwise equal and
the residual maximum was `7.80e-15`.

The nonlinear `hopping_rhs()` supplied an independent Taylor oracle.  For
epsilon `0.02, 0.01, 0.005, 0.0025`, the remainder norms were respectively
`4.75917e-5`, `1.18980e-5`, `2.97450e-6`, and `7.43625e-7`, giving fitted order
`1.99999777`.  This verifies an `O(epsilon^2)` remainder without reusing the
linearized modal implementation.

Tests cover resolved multimode evolution, arbitrary/zero/equilibrium initial
states, positive/negative/zero bias, bias reversal, stability, constant and
Nyquist modes on odd/even grids, independent batched planes, NumPy float64 and
float32 dtype preservation, and a conditional CuPy seam.

## Production contract

`PRRunRequest.material_response` is appended after every historical positional
field, so old positional construction is unchanged and defaults to nonlinear.
Linearized TD requires `material_response.model="linearized"`, an explicit
positive `reference_intensity`, and `solver.integrator="exact_modal"`.
Conversely, nonlinear TD rejects `exact_modal`; its Euler and semi-implicit
behavior remain the defaults and are regression protected.

For each accepted material interval production performs:

1. one complete optical z pass through the last accepted `E`;
2. formation of the PR-driving intensity stack;
3. one exact modal update while that source is frozen;
4. acceptance of the complete new `E` state;
5. the next interval's optical pass (or the final optical replay).

Thus no hidden cadence was introduced.  Diagnostics report `Exact modal`, the
model identity, explicit `I0`, final linearized RHS norms, and locally validated
software evidence.  Run-cost metadata treats material work as a fixed count of
one-dimensional FFT operations rather than nonlinear iteration.

Experiment schema 5, reduced-TD transport codec 3, and checkpoint schema 4
persist the response axis.  Immediately preceding and older supported payloads
without it migrate to nonlinear.  Checkpoints retain authoritative `E`, `A0`,
material time, accepted-step count, request/coupling provenance, and initial
state.  No source volume is needed across an accepted boundary because the
established cadence recomputes it from `A0` and accepted `E`; split continuation
matches an uninterrupted run exactly in the controlled test.

Full retention preserves material volumes and the checkpoint.  Fast retention
keeps optical endpoints, compact diagnostics/provenance, and the standard x-z
and y-z optical intensity cuts while marking material volumes and checkpoint
unavailable; omitted histories are not synthesized.  Both policies convert
through the standard product path.

Cancellation before evolution or during an optical/material substage discards
the incomplete candidate.  Cancellation after an accepted interval retains
the accepted `E`, completed count, material time, and available optical/source
observation.  Continuation resumes from that boundary.

## Limitations and exclusions

`I0` is a modeling input and is never inferred from an optical field.  The
linearization is a frozen-source, small-perturbation material model; it does not
change nonlinear PR physics or claim nonlinear validity.  CuPy execution is
conditional pending later hardware commissioning.  This milestone adds no
eight-mode GUI controls and makes no change to full-transverse models, static
models, optical propagation, IA mathematics, fanning, or soliton code.
