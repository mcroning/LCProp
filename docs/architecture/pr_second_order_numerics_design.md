# Photorefractive Second-Order Numerics: Semi-Discrete Equation and Candidate IMEX Split

**Date:** 2026-08-05

**Branch:** `feature/pr-second-order-static`

**Status:** Production integrator and strict fixed-intensity reference solver validated

## Purpose

This document derives the photorefractive material equation actually integrated by LCProp and evaluates candidate second-order semi-implicit splits before implementation. It is the second step of the numerical-foundation upgrade that precedes image-amplification validation.

The derivation is intentionally PR-owned. It does not change the material composition architecture, the optical propagation seam, LC workflows, or shared material-state abstractions.

## Authoritative Sources

The physics and sign conventions are taken from:

- Equation (4) of `docs/photonics-12-00113-v3.pdf`;
- the full `dEdt_f()` implementation in `reference/prprop/prprop3d.py`;
- the validated implementation in `src/lcprop/pr/evolution.py`;
- the intensity mapping in `src/lcprop/pr/source.py`;
- the frozen-state optical pass in `src/lcprop/pr/workflow.py`.

The paper and trusted implementation define the continuum model. The LCProp implementation defines the present spatial discretization, transverse boundary treatment, intensity sampling, and optical/material stepping order that the new time integrator must preserve.

## Continuum Equation and Normalization

In normalized variables, the PR hopping model is

\[
\frac{\partial E}{\partial t}
= E_{\mathrm{app}} I_b
- (E I-I_x)(1+E_x)
+ I E_{xx}.
\]

Here:

- \(E\) is the normalized physical space-charge field;
- \(x_n=k_0x\) is normalized transverse distance;
- \(t\) is normalized material time;
- \(I=I_{\mathrm{optical}}/I_{\mathrm{peak}}+I_b\) is the complete PR-driving intensity;
- \(I_b\) is LCProp's total uniform background, equal to dark intensity plus optional uniform optical background;
- \(E_{\mathrm{app}}\) is the normalized applied field;
- subscripts \(x\) denote derivatives with respect to normalized \(x_n\).

The positive \(I E_{xx}\) term is diffusive because the second-derivative eigenvalue is negative for every nonzero Fourier mode. The trusted PRProp3D expression

```text
E_app*I_b - ((E*I - I_x)*(1 + E_x) - E_xx*I)
```

has the same sign after expansion.

The paper writes the applied-field source using the equivalent dark intensity. LCProp's already validated convention passes the total uniform background `dark_intensity + uniform_background_intensity` as \(I_b\) both in the driving intensity and in the applied-field source. Changing that convention is outside this numerical task.

## Current Spatial Discretization

The PR state has shape

\[
(N_z,N_x,N_y).
\]

Material derivatives act only along array axis `-2`, the \(x\) axis. There are no material derivatives in \(y\) or \(z\). The numerical material boundary is periodic in \(x\); each \((z,y)\) line is differentiated independently.

Let

\[
h=k_0\,\Delta x_{\mathrm{physical}}
\]

be the normalized grid spacing. For a periodic grid index \(j\), the implemented centered differences are

\[
(D_1u)_j=\frac{u_{j+1}-u_{j-1}}{2h},
\]

\[
(D_2u)_j=\frac{u_{j+1}-2u_j+u_{j-1}}{h^2}.
\]

Their Fourier symbols are

\[
D_1\rightarrow i\frac{\sin(kh)}{h},
\qquad
D_2\rightarrow -\frac{4\sin^2(kh/2)}{h^2}.
\]

These operators, not spectral derivatives, define the discrete equation to be integrated. A new material-time method must not silently substitute the PRProp3D spectral discretization.

## Authoritative Discrete Residual

For elementwise multiplication denoted by \(\odot\), the current discrete residual is

\[
R_h(E;I)
=E_{\mathrm{app}}I_b
-(E\odot I-D_1I)\odot(1+D_1E)
+I\odot D_2E.
\]

Expanding without changing the discrete operations gives

\[
R_h(E;I)
=E_{\mathrm{app}}I_b
+D_1I
-I\odot E
-I\odot E\odot D_1E
+(D_1I)\odot D_1E
+I\odot D_2E.
\]

The existing `hopping_rhs()` function is already the authoritative implementation of \(R_h\). It should remain the reference used by Euler, residual diagnostics, and static convergence checks. An implicit method may assemble part of the expanded residual as a linear system, but it must not introduce a second independently maintained physical equation.

## The Actual Coupled Semi-Discrete System

For prescribed intensity, the material equation would be

\[
\dot E=R_h(E;I).
\]

That is useful for unit tests and the future fixed-intensity static solver, but it is not the complete time-dependent LCProp workflow.

Define \(\mathcal{P}(E)\) as the deterministic frozen-state optical mapping. Starting from the entrance field \(A_0\), LCProp propagates through every longitudinal slice using the current accepted \(E\) and `advance_prepared_response()`. At slice \(m\), it records

\[
I_m(E)=\frac{1}{2}\left(I_m^-+I_m^+\right),
\]

where \(I_m^-\) and \(I_m^+\) are the normalized PR-driving intensities before and after the slice's optical advancement. Both include \(I_b\). The resulting stack is

\[
\mathcal{P}(E)=\left(I_0(E),\ldots,I_{N_z-1}(E)\right).
\]

Because the field advances sequentially in \(z\), \(\mathcal{P}\) is nonlocal in the longitudinal direction: changing an upstream response can change downstream intensities. All material slices are nevertheless updated synchronously after the complete optical pass.

The actual method-of-lines system is therefore

\[
\boxed{\dot E=F_h(E)=R_h\!\left(E;\mathcal{P}(E)\right)}.
\]

A method that is second order only for frozen prescribed \(I\), while evaluating \(\mathcal{P}(E)\) only to first order, is not second order for the actual workflow.

## Legacy and Current Stepping Orders

PRProp3D uses spectral derivatives and explicit Euler. During each material-time iteration it marches in \(z\), updates the local space-charge field immediately from the current optical amplitude, applies a full PR response, and then continues to the next optical slice. Its material update and optical march are interleaved.

LCProp deliberately uses a different and already validated order:

1. hold the complete accepted \(E_n\) fixed;
2. propagate \(A_0\) through all slices with optical Strang splitting;
3. construct the complete \(I_n=\mathcal{P}(E_n)\) stack;
4. update every material slice synchronously.

The numerical upgrade applies to step 4 and to the time-level evaluation of \(\mathcal{P}\). It must not restore the legacy Lie optical ordering.

## Candidate Operator Splits

Write

\[
F_h(E)=L_{I(E)}E+N(E;I(E)),
\qquad I(E)=\mathcal{P}(E).
\]

Two splits are mathematically consistent with the complete residual.

### Split A: diffusion-only implicit operator

Define

\[
L_I E=I\odot D_2E,
\]

and

\[
N(E;I)=E_{\mathrm{app}}I_b
-(E\odot I-D_1I)\odot(1+D_1E).
\]

Advantages:

- it treats the only \(O(h^{-2})\) term implicitly;
- the remaining explicit restriction is not dominated by grid-scale diffusion;
- the implicit matrix has a particularly robust cyclic tridiagonal form;
- it follows the recommendation already recorded in the PR implementation notes;
- it is straightforward to compare with explicit Euler using the same residual.

Limitations:

- reaction, drift, and nonlinear gradient terms remain explicit;
- a nonlinear or intensity-dependent stability restriction remains;
- larger stable timesteps are not automatically accurate timesteps.

This is the recommended first split to prototype.

### Split B: expanded linear terms implicit

The expanded residual permits

\[
L_IE
=I\odot D_2E
-I\odot E
+(D_1I)\odot D_1E,
\]

\[
N(E;I)
=E_{\mathrm{app}}I_b+D_1I-I\odot E\odot D_1E.
\]

Advantages:

- the uniform-state reaction is implicit;
- all terms linear in \(E\) for prescribed \(I\) are represented in the linear solve;
- only the quadratic material drift remains nonlinear and explicit.

Limitations:

- the added first-derivative term makes the cyclic system nonsymmetric;
- strong intensity gradients can weaken diagonal dominance;
- it adds implementation and backend complexity before Split A has shown a deficiency;
- it does not eliminate the state dependence through \(I(E)\).

Split B should be considered only if Split A passes correctness tests but retains a practically important non-diffusive stability restriction.

## Implicit Linear System

For Split A, a solve of

\[
(\mathbf{1}-\alpha L_I)u=b
\]

has row coefficients

\[
\left(1+\frac{2\alpha I_j}{h^2}\right)u_j
-\frac{\alpha I_j}{h^2}u_{j-1}
-\frac{\alpha I_j}{h^2}u_{j+1}
=b_j,
\]

with periodic corner entries. For \(I_j>0\) and \(\alpha>0\), each row is strictly diagonally dominant by one. The solve consists of independent cyclic tridiagonal systems along \(x\), batched over \((z,y)\).

This structure is preferable to a spectral solve because \(I\) varies spatially. Multiplication by \(I(x)) makes \(I D_2\) a variable-coefficient operator that is not diagonal in Fourier space.

The next implementation step must establish a PR-owned batched solver that:

- preserves periodic corner coupling;
- supports `float32` and `float64`;
- has defined NumPy and CuPy behavior;
- verifies the linear residual directly;
- detects singular, nonfinite, or poorly conditioned results;
- does not modify the LC algorithms package.

## Candidate Time Integrators

### Candidate 1: one-step linearly implicit trapezoidal predictor-corrector

Let

\[
I_n=\mathcal{P}(E_n),\qquad
L_n=L_{I_n},\qquad
N_n=N(E_n;I_n).
\]

First compute an IMEX-Euler predictor:

\[
(\mathbf{1}-\Delta t L_n)E^p
=E_n+\Delta t N_n.
\]

Then evaluate the optical mapping at the predicted state:

\[
I_p=\mathcal{P}(E^p),\qquad
L_p=L_{I_p},\qquad
N_p=N(E^p;I_p).
\]

The corrected state is defined by

\[
(\mathbf{1}-\tfrac{1}{2}\Delta t L_p)E_{n+1}
=E_n+\frac{\Delta t}{2}
\left(L_nE_n+N_n+N_p\right).
\]

For a constant implicit operator and no explicit term, the corrector reduces exactly to Crank–Nicolson. For no implicit operator, it reduces to explicit Heun. The first-order predictor approximates the endpoint state with \(O(\Delta t^2)\) error; its coefficient and residual evaluations therefore preserve the corrector's \(O(\Delta t^3)\) local truncation error for smooth solutions.

Properties:

- one-step method with no multistep startup or checkpoint history;
- evaluates the self-consistent optical mapping at both the accepted and predicted states;
- two implicit solves per material step;
- normally two optical passes per material step, although an accepted-state intensity may be cached between steps;
- expected global second-order accuracy, subject to direct measurement on both prescribed-intensity and coupled workflows.

This is the recommended correctness baseline.

### Candidate 2: variable-operator CNAB2

For the diffusion split, define an extrapolated midpoint intensity

\[
I_{n+1/2}^{\mathrm{AB}}
=\frac{3}{2}I_n-\frac{1}{2}I_{n-1},
\qquad
\widehat L_{n+1/2}=L_{I_{n+1/2}^{\mathrm{AB}}}.
\]

A candidate variable-operator CNAB2 step is

\[
(\mathbf{1}-\tfrac{1}{2}\Delta t\widehat L_{n+1/2})E_{n+1}
=(\mathbf{1}+\tfrac{1}{2}\Delta t\widehat L_{n+1/2})E_n
+\Delta t\left(\frac{3}{2}N_n-\frac{1}{2}N_{n-1}\right).
\]

This uses one new optical pass per step after startup and should be second order for smooth solutions when the extrapolated midpoint operator remains valid. It also introduces material risks that the one-step method avoids:

- extrapolated intensity can become locally negative during a rapid transient, destroying the favorable diffusion matrix properties;
- continuation requires previous accepted intensity/residual history or a deterministic restart transition;
- a separate second-order startup method remains necessary;
- cancellation and disk checkpoints must occur only after history and state are jointly accepted;
- reproducing an uninterrupted trajectory requires preserving or exactly reconstructing history.

CNAB2 is therefore retained as a performance candidate, not selected as the initial implementation merely by preference.

## Recommendation

Prototype Split A with the one-step linearly implicit trapezoidal predictor-corrector first.

This combination addresses the dominant diffusion stiffness while treating the actual state-dependent optical mapping to second-order consistency. It avoids a checkpoint schema expansion solely for multistep history and provides a clean reference against which a later CNAB2 optimization can be judged.

CNAB2 should replace it only if all of the following are demonstrated:

1. the second optical pass is a material performance bottleneck;
2. midpoint intensity extrapolation remains positive or is handled without changing the equation;
3. coupled-workflow temporal order remains second order;
4. checkpoint history and restart behavior are exact and versioned;
5. the optimized path agrees with the one-step reference as \(\Delta t\) is refined.

## Stability Semantics

The current `conservative_timestep_limit()` is an explicit-Euler guard. It must remain associated with Euler and must not be presented as the stability limit of a semi-implicit method.

For uniform \(I_0\) and equilibrium \(\bar E\), Split A has the linearized mode decomposition

\[
\lambda_L(k)=-I_0 k_{2,h}^2,
\]

\[
\lambda_N(k)=-I_0-iI_0\bar E k_{1,h},
\]

where \(k_{1,h}\) and \(k_{2,h}^2\) are the centered-difference modified wavenumbers. The diffusion contribution is handled by the implicit stability function, but the explicit reaction and drift remain restricted by the explicit part of the method.

For constant scalar mode eigenvalues, define

\[
z_L=\Delta t\lambda_L,
\qquad
z_N=\Delta t\lambda_N.
\]

The predictor amplification is

\[
P=\frac{1+z_N}{1-z_L},
\]

and the proposed corrector has amplification

\[
G(z_L,z_N)
=\frac{1+\tfrac{1}{2}(z_L+z_N)+\tfrac{1}{2}z_NP}
{1-\tfrac{1}{2}z_L}.
\]

For \(z_N=0\), this is the Crank–Nicolson factor

\[
G=\frac{1+z_L/2}{1-z_L/2}.
\]

For \(z_L=0\), it is the Heun factor

\[
G=1+z_N+\frac{z_N^2}{2}.
\]

The implementation step must use this factor to test the selected method over all representable modes. That linearized rule will remain a local uniform-state guide, not a nonlinear global guarantee. Accuracy recommendations and nonfinite-state guards must be reported separately from absolute stability.

No acceptance criterion should require a timestep above the Euler limit unless the derived semi-implicit stability region and observed convergence support it.

## Continuation and Persistence Implications

The recommended one-step method requires only the latest accepted physical state \(E_n\). The current checkpoint's `E_current`, `E_initial`, and `A0` remain sufficient. Adding an integrator identifier to `PRSolverOptions` would still affect request serialization and compatibility, but no numerical history array would be required.

If CNAB2 is later selected, its previous intensity and explicit residual—or enough previous physical state to reproduce them—become accepted integrator state. They must be represented explicitly in the PR-owned checkpoint contract. Existing schema-version-1 checkpoints must decode as Euler checkpoints, and any transition into a multistep method must use a documented deterministic second-order startup.

The programmatic default must not silently reinterpret an old request or checkpoint as a different integrator.

## Static-Solver Implications

For prescribed \(I\), the future strict static problem is

\[
R_h(E;I)=0.
\]

It separates into independent periodic nonlinear systems along \(x\) for every \((z,y)\). The static solver should evaluate convergence with the same `hopping_rhs()` residual. A small CPU reference solver based on damped Newton or Newton–Krylov would provide a more independent validation target than driving the new transient method to equilibrium twice.

Design and implementation of that solver belong to a later step. It is not necessary to change the semi-discrete equation or the operator split derived here.

## Required Evidence Before Selecting the Production Method

The prototype phase must provide:

1. cyclic linear-solve residuals for constant and spatially varying positive intensity;
2. NumPy `float32` and `float64` results;
3. CuPy compatibility where available, or an explicit reviewed backend limitation;
4. reduction to Crank–Nicolson for pure constant-coefficient diffusion;
5. reduction to Heun for a zero implicit operator;
6. measured second-order convergence for prescribed intensity;
7. measured second-order convergence for the complete \(E_t=R_h(E;\mathcal{P}(E))\) workflow;
8. agreement with explicit Euler as \(\Delta t\rightarrow0\);
9. deterministic uninterrupted and checkpoint-resumed trajectories;
10. power conservation for every optical pass.

## Prototype Evidence

The bounded prototype implements Split A and the one-step predictor-corrector without connecting it to the production workflow.

The cyclic variable-coefficient solve was compared both with its direct discrete residual and with dense `numpy.linalg.solve()` reference systems. Maximum `float64` residuals were below \(2\times10^{-14}\); `float32` residuals were below \(3\times10^{-6}\). Constant-intensity Fourier modes reproduce the centered-difference Crank–Nicolson amplification factor.

Temporal convergence was measured against independent SciPy DOP853 integrations of the authoritative semi-discrete residual:

| Intensity mapping | Errors at successive refinements | Pairwise orders |
|---|---|---|
| Prescribed intensity | \(5.7813\times10^{-4}\), \(1.4604\times10^{-4}\), \(3.6773\times10^{-5}\) | 1.9850, 1.9897 |
| State-dependent local proxy | \(5.7955\times10^{-4}\), \(1.4643\times10^{-4}\), \(3.6875\times10^{-5}\) | 1.9847, 1.9895 |
| Actual frozen-\(E\) optical mapping | \(5.5996\times10^{-5}\), \(1.3938\times10^{-5}\), \(3.4792\times10^{-6}\) | 2.0063, 2.0022 |

For the same actual optical-mapping case, explicit Euler produced errors \(3.6289\times10^{-3}\), \(1.7617\times10^{-3}\), and \(8.6856\times10^{-4}\), with measured orders 1.0425 and 1.0203. The new method is therefore second order for the coupled equation, not merely for prescribed intensity.

Every prototype material step invokes the optical mapping exactly twice. Both the accepted-state and predictor-state optical passes conserve total optical power to the existing `float64` tolerance.

An illustrative NumPy CPU timing used a `(Nz, Nx, Ny) = (20, 128, 64)` state. One frozen-state optical mapping took 8.16 ms on average; one complete prototype material step took 37.18 ms, a per-step ratio of 4.56. The extra cost includes a second optical pass and two cyclic linear solves. This is not a portable performance benchmark, but it establishes that the method is substantially more expensive per step than Euler.

The accuracy comparison changes the interpretation of that overhead. On the coupled test, the two-step second-order result was already more than an order of magnitude more accurate than the eight-step Euler result. For accuracy-controlled work, the higher per-step cost need not imply higher total cost.

NumPy `float32` and `float64` paths are validated. The CuPy test is implemented but was skipped because no usable GPU device was available in the test environment. Production selection therefore requires retaining that explicit validation limitation.

## Production Integration

`PRSolverOptions.integrator` now carries an explicit material-integrator identity. The supported values are:

- `euler`;
- `semi_implicit_trapezoidal`.

The dataclass default remains `euler` so existing programmatic requests retain their original numerical meaning. The standalone PR GUI defaults explicitly to `semi_implicit_trapezoidal` and exposes both choices. Loading a checkpoint restores the saved selection; changing it is an incompatible continuation edit.

`run_pr_timedependent()` dispatches at the accepted material-step boundary. Euler continues to perform one frozen-state optical mapping followed by `euler_step()`. The semi-implicit path passes the same `_optical_pass()` mapping into `semi_implicit_trapezoidal_step()`, which evaluates accepted and predictor states. Progress, cancellation, and final products remain on accepted complete material steps. There is no GUI-only physics path.

The previous `conservative_timestep_limit()` remains the Euler guard. The semi-implicit guard evaluates the derived scalar amplification factor over every representable centered-difference mode, locates the first absolute-stability boundary, and applies the established factor-of-eight conservative margin. This removes the grid-scale diffusion restriction while retaining explicit reaction and drift restrictions. `validate_timestep()` dispatches by integrator and reports the selected integrator in failures and result diagnostics.

The one-step method requires no new numerical history arrays. PR checkpoint schema version 2 records the integrator in the request. Authentic schema-version-1 checkpoints contain no integrator field and load deterministically as Euler. Version 1 rejects a new integrator field, and version 2 rejects a missing field, preventing ambiguous reinterpretation.

Uninterrupted, in-memory resumed, and disk-resumed semi-implicit runs produce exactly equal accepted physical states in the focused tests. The PR GUI preserves this equivalence through load, control hydration, compatibility validation, and continuation.

## Strict Fixed-Intensity Reference Solver

The strict material-static problem is

\[
R_h(E;I)=0
\]

for prescribed \(I\). `solve_pr_static_intensity()` implements a PR-owned CPU `float64` damped-Newton reference solver. It uses `hopping_rhs()` as the authoritative residual and an exact analytic cyclic Jacobian.

At x index \(j\), define

\[
Q_j=E_jI_j-(D_1I)_j,
\qquad
G_j=1+(D_1E)_j.
\]

The nonzero Jacobian entries are

\[
\frac{\partial R_j}{\partial E_{j-1}}
=\frac{I_j}{h^2}+\frac{Q_j}{2h},
\]

\[
\frac{\partial R_j}{\partial E_j}
=-I_jG_j-\frac{2I_j}{h^2},
\]

\[
\frac{\partial R_j}{\partial E_{j+1}}
=\frac{I_j}{h^2}-\frac{Q_j}{2h}.
\]

The periodic systems are independent for every \((z,y)\) line. The reference implementation solves their dense cyclic matrices independently and applies one global Armijo backtracking factor. This favors independence and transparency over production-scale throughput.

Convergence requires both residual RMS and residual maximum tolerances. Neither a small Newton direction nor exhaustion of the line search is reported as convergence. Results include status, Newton iteration count, final residual metrics, and an accepted-iteration history. A supplied initial \(E\) supports continuation-friendly validation.

The analytic Jacobian directional derivative agrees with a centered finite-difference derivative to relative error below \(2\times10^{-9}\). Long-time semi-implicit comparisons used \(\Delta t=0.1\), 200 steps, and therefore final normalized time 20:

| Prescribed intensity | Newton iterations | Static residual RMS / max | Normalized transient-static \(L^2\) | Maximum state difference | Transient residual RMS / max |
|---|---:|---:|---:|---:|---:|
| Uniform | 1 | \(2.23\times10^{-16}\) / \(4.61\times10^{-16}\) | \(4.02\times10^{-11}\) | \(2.01\times10^{-12}\) | \(2.41\times10^{-12}\) / \(2.41\times10^{-12}\) |
| Spatially modulated | 3 | \(2.34\times10^{-13}\) / \(4.88\times10^{-13}\) | \(3.17\times10^{-10}\) | \(3.19\times10^{-11}\) | \(2.90\times10^{-11}\) / \(3.42\times10^{-11}\) |

The static solver is deliberately not a universal self-consistent propagation-static framework and is not exposed in the GUI. Its role is to supply an independent root for numerical validation and image-amplification readiness checks.

## Design Decision

Split A with the one-step linearly implicit trapezoidal predictor-corrector is selected and integrated as the PR production method.

The selection is based on measured second-order accuracy for the actual self-consistent optical mapping, exact one-step continuation state, strong variable-diffusion solve residuals, and a large accuracy improvement over Euler. The measured per-step overhead is accepted for the first production implementation because it purchases a verified coupled second-order method and avoids multistep checkpoint history.

Retain CNAB2 as a future performance optimization only. It should be reconsidered after the production one-step method supplies a trusted reference and representative image-amplification profiles reveal whether material-step cost is limiting.

The strict fixed-intensity static reference and transient-versus-static acceptance tests are now implemented. The final numerical gate before substantive image-amplification validation is also complete. A bounded coherent broad-pump/weak-signal case uses opposite grid-commensurate tilts on a 64 × 16 × 10 grid. Its normalized timestep of 0.05 is 3.57 times the conservative explicit-Euler limit of 0.0140056022409 and remains below the semi-implicit limit of 0.238095238095.

After 250 material steps, the final coupled residual was `6.22445031137e-7` RMS and `6.78252578409e-6` maximum. An independent zero-start Newton solve with the final optical source prescribed converged and differed from the transient physical E state by `9.66995080703e-6` relative L2 and `8.36204123975e-6` maximum absolute error. Normalized optical power drift was `-2.33146835171e-15`. Matched-field output signal power changed by `-0.910564848713` relative to the zero-response replay, confirming nontrivial signal/pump interaction without treating that diagnostic as a physical image-amplification benchmark.

This bounded readiness case uses the existing PR request, workflow, optical propagation seam, strict static solver, and matched-field diagnostics. It introduces no workflow or architecture layer. The numerical foundation is therefore ready for a separately specified substantive image-amplification benchmark. CNAB2, adaptive control, self-consistent static propagation, and broader GUI changes remain deferred.

---

End of numerical design record.
