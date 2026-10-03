# Unified 1D material core — M2

M2 adds an explicitly called material solver to the committed M1 contracts.
It does not register a workflow or import runtime research code. Existing
Static/TD workflows, optics, GUI, persistence and resource estimates are untouched.
The namespace `lcprop.pr.unified` remains documentation-only on import.

## API and ownership

Import `solve_static_material` from `lcprop.pr.unified.static`. Supply an M1
`PRTransportIntensity` with an already resident, correctly typed NumPy/CuPy array,
and an explicit `PRElectricalClosureSpec` as `closure=`. Return value is
`(PRUnifiedMaterialState, PRMaterialDiagnostics)`.

The spatial spec permits pure `(Nx,)` or x-active/y-batch `(Nx,Ny)`. Genuine 2D
is rejected. Each column is solved separately, in deterministic column order;
no reduction, damping decision, convergence gate or harmonic field is shared
between columns. Closure targets are applied independently to each column.

The input is borrowed, must remain alive and unmodified during the call, and is
never changed. One column is copied into solver-owned workspace. Returned q,
psi and b are detached solver-owned arrays; their M1 record borrows those buffers
and does not make them immutable. No partial plane is returned on failure.
`MaterialConvergenceError` identifies the failed column and preserves the cause.
There is no initialization override, fallback, E-polish or intensity homotopy.

## Exact material and spatial conventions

State is q, periodic psi, harmonic b. Carrier is exp(q), oriented right-face
field is `b - (psi[i+1]-psi[i])/h`, and the Poisson operator is minus the
conservative divergence of that forward gradient. Constant potential is the
only gauge nullspace; the even-grid alternating mode is not a null mode.

The oriented face drop is `v=psi[i+1]-psi[i]-b*h`. With w=I*exp(q), flux is
`[B(v)*w[i]-B(-v)*w[i+1]]/h`, B(v)=v/expm1(v), with the frozen research
small/large-argument evaluation branches and derivative. This is the fitted
hopping flux, not the old expanded centered-E discrete equation. All occurrences
of I use the supplied dark-inclusive total intensity. No background is removed,
re-added, floored or substituted for the reference normalization.

Supported closures are unbiased zero flux, fixed mean field, prescribed mean
current and A7 current. A7 retains E_app and I_b and imposes J_ext=E_app*I_b;
the illuminated internal b is solved. A7 I_b must exactly match the declared
dark-plus-uniform transport background metadata. Mixed/open-transverse closure is rejected.
Potential gauge removal never sets the physical harmonic field.

## Frozen solver and numerical gates

Arithmetic is ported from the scientific Stage-E1.1 manifest
`e6e0e4b1bd25f6a85fd42ebca29721ac5d3e8aceb01d9c427c1fdc8abbed0cf2`,
certified in research job 4827999. The source chain is Stage-A 1D formulation,
Stage-B independent physics/discretization qualification, and the Stage-C/D/E
shared implementation. This is not native certification of the Product port.

Unbiased solves use the normalized log-carrier relation and potential Newton.
Other closures solve the constrained q/psi/b equations, all Gauss rows, all but
one dependent divergence row, gauge and electrical closure. Final diagnostics
check **every** divergence row. Bounds remain 60 Newton updates, 30 Armijo trials
per update, coefficient 1e-4. Trial carrier must be finite and strictly positive.
No clipping, E-space rescue, continuation or changes to research gates occur.

Float64 physical RMS/max limits are 1e-10/1e-9, neutrality/gauge 1e-11.
Float32 uses the frozen stencil-cancellation scales Gs, Js and eps32:

- Gauss: Gs=1+max(n)+4 max(abs(psi))/h²;
- flux balance: Js=1+4 max(abs(nI))/h²+2 max(abs(J))/h;
- closure scale: 1;
- RMS: max(1e-10,8 eps32 scale);
- maximum: max(1e-9,32 eps32 scale);
- neutrality/gauge: max(1e-11,8 eps32);
- zero-flux magnitude: max(1e-9,32 eps32 Js).

For noninitial float32 states, final relative potential correction must also be
at most 2e-5 and q/b correction at most 2e-6, as in the certified reference.
Float32 state assembles coefficients/RHS in state precision, promotes the sparse
linear system to float64, solves there, and casts the correction once back to
float32. Bernoulli evaluation is float64 then rounded to state precision. The
small closure initialization solve is also float64. Float64 remains float64.
M1 precision identities are unchanged.

## Diagnostics and backend boundary

Returned M1 diagnostics are named scalar observations, with column-qualified
Gauss, divergence, closure, neutrality, gauge, carrier extrema, harmonic field,
mean current, finite status, iteration history and actual physical limits.
`column_k.converged=1` reports the solver's completed physical gates; M1 itself
still only validates metadata. Backend and precision are carried in the state,
with corresponding scalar indicators in diagnostics. A failure raises rather
than manufacturing a successful diagnostics record.

Existing Product backend resolution is used with an explicit backend, never
`auto`. CuPy requires native CSR QR; no CPU scientific fallback exists. The only
explicit execution-time device-to-host boundary transfers at most 32 already
reduced float64 status scalars (256 bytes). No field/intensity array is exported.
The input device context is used for allocations. GPU qualification tests remain
pending when CuPy/CUDA is unavailable; no native performance claim is made.

## Allocation inventory

For Nx by Ny reduced columns, storage is one borrowed intensity plane plus
owned q/psi output planes and Ny harmonic components. One column at a time owns
an intensity copy, accepted q/psi/b, trial q/psi/b, carrier/flux/residual arrays,
Newton correction and sparse Jacobian/factorization workspace. Float64 promotion
holds a sparse matrix/RHS and linear correction alongside state-precision arrays.
The zero-flux matrix is Nx square sparse; constrained closure is (2Nx+1) square
sparse. No dense plane-sized Jacobian, longitudinal optical/material volume,
replay or optical endpoint exists. Sparse factorization fill is solver-dependent;
this inventory is not a measured native memory bound. Host diagnostics store
bounded scalar history (at most 61 states per column), not scientific arrays.

## Validation boundary

Tests bind the frozen scientific manifest, benchmark matrix, exact dark-inclusive
inputs and output archives by SHA-256. All Request-15 columns of cells 51/52/54/56
are checked in both precisions, plus uniform, weak, .95/.99/.999, A7, fixed-field
and prescribed-current fixtures. Stage-B difficult-column roots are independently
bound. Missing evidence fails; it is neither regenerated nor silently skipped.

The independent test oracle evaluates the integrated edge flux in long double,
with explicit periodic indexing; it does not call Product Bernoulli, residual or
operators. It checks Gauss, every flux balance, closure, gauge, neutrality and
strict positive carrier using the frozen gates. Root comparisons use frozen
quantity-specific Stage-E parity tolerances. No comparison demands equivalence
to obsolete centered-E coarse-grid roots.

Batch tests compare roots and full scalar damping histories with separate solves,
including mixed difficulties, replicated columns and permutations. Additional
checks cover analytic Jacobians, constant-only nullspace, conservation, sign,
mixed linear precision, transfer bounds, input ownership and failure atomicity.
M1 namespace-isolation tests remain unchanged. No existing dispatch imports this
solver. Only focused M1/M2 tests are in scope.
