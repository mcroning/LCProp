# Unified material core — bounded 2D extension (M3)

M3 extends the M2 callable `solve_static_material` under the unchanged M1
contracts. No workflow, optical projection, GUI, codec, persistence, checkpoint
or TD integration exists. This document extends the historical M2 1D-only scope;
M2's arithmetic, precision policy and independent-column semantics remain intact.

## One formulation, two execution geometries

For active x with optional y batch, the original one-column path allocates and
solves an Nx problem separately for each column. `solve_column` explicitly rejects
multi-axis geometry. There are no y faces or cross-column reductions. The same
solver state, damping history, residual limits and scalar diagnostics are retained.

For active x,y, the connected-plane wrapper constructs the explicit electrical
matrices and calls `solve_domain` once on the connected grid. It never delegates
a y-invariant plane to independent columns. That shared function is the unchanged
M2 Newton arithmetic, with only its dimension check generalized. Operators iterate
over the active axes; there is one hopping-flux formula and one analytic Jacobian,
not a new transverse material equation. Dimensions/batching come from M1 rather
than being inferred from array rank.

Canonical state is q, periodic psi and harmonic vector b. Positive carrier is
exp(q); right-face E_j=b_j-(psi[next_j]-psi)/h_j. Poisson is minus divergence of
forward gradient, with only the constant potential nullspace. Face flux is
[B(v_j)nI-B(-v_j)(nI)[next_j]]/h_j, where v_j includes the affine -b_j h_j even
across a periodic wrap. Divergence includes both oriented x/y faces in 2D. Total
transport intensity includes the supplied dark/background contribution unchanged.
No spectral transverse or old centered-E solver is imported.

## Electrical experiments

- Unbiased zero flux: b=(0,0), detailed-balance potential solve.
- Fixed mean field: prescribe both harmonic components.
- Prescribed mean current: solve both harmonic components for the two targets.
- Fixed-x/open-y: prescribe b_x and mean(J_y)=0; solve b_y, which need not vanish.

These use the frozen Stage-D U b+V mean(J)=target mapping. A7 remains a 1D
reservoir-current specialization; matching prescribed-current 2D can reduce to it,
but a 2D A7 spec is not introduced. Potential gauge and harmonic closure remain
distinct. M1 identities and constraints are unchanged.

## Precision, physical gates and failure

Both geometries use the same M2 backend helper, float64 sensitive linear solve,
float64 Bernoulli evaluation and state-precision cast-back. The helper is unchanged.
No CPU scientific fallback, iterative solver or custom CUDA kernel is introduced.
The original 60-update/30-trial Armijo bounds and Stage-E physical/correction gates
are unchanged. Finite positive carrier, full Gauss/divergence, closure, neutrality
and gauge gates precede return. No partial connected state is returned on failure.
Inputs are borrowed during execution, never mutated; outputs own detached buffers.

Connected diagnostics retain scalar iteration evidence, independent residual
components and limits, carrier extrema, and explicit harmonic_x/y and
mean_current_x/y values. There is no conflation of zero y field and zero mean y
current. The M1 diagnostics record does not itself certify physical acceptance;
the solver reports its completed gates. Reduced diagnostics retain M2 names.

## Direct-solver bound and allocations

Connected solves reject more than **12,288 nodes** before backend/workspace
construction. This matches the largest Stage-C replicated 4096-by-3 reference and
is a conservative scope boundary, not a performance guarantee for every shape or
closure. Large-2D solver feasibility requires a separate review. All supported
active axes have at least two samples; 1D is never a degenerate Ny=1 2D solve.

Reduced batch storage remains output q/psi planes, one harmonic component per
column, borrowed I, and one active Nx column's accepted/trial/linear workspace.
The bounded shared routine adds no y faces or NxNy sparse matrix to this path.

Connected storage holds one borrowed I plane, one owned I copy, accepted/trial
q/psi and two b components, temporary carrier/face/residual arrays for both axes,
and sparse 5-point Poisson/Newton blocks. Zero-flux matrix order is N=NxNy;
constrained matrix order is 2N+2. State-precision assembly and promoted float64
linear workspace coexist. Sparse factorization fill depends on geometry/closure.
Returned state reuses owned accepted arrays without another full-plane copy.
History contains at most 61 sets of reduced scalars. Neither path has a z volume,
optical/replay volume or persistent derived E/J arrays.

## Qualification

Exact local-operator reduction is checked for replicated states: carrier,
x-difference, x-field, x-flux, divergence, Poisson and Gauss. Appropriate replicated
states have exactly zero y fields/fluxes. Independently solved state reduction uses
Stage-C float64 rtol=atol=1e-8 and the frozen Stage-E float32 quantity-specific
policy; global normalization and factorization order need not be bitwise identical.
Uniform, weak, strong .999 and the full 4096-sample Request-15 difficult column are
included. Prescribed current is compared with the same 1D A7 current target.

Frozen Stage-C/D archives and every genuine-2D Stage-E case are hash-bound. The
independent 2D oracle uses extended-precision integrated edge flux and indexed
neighbors; it imports no Product operators/residuals. It checks Gauss, divergence,
closure, gauge, neutrality, harmonic means, finite positive carrier and zero-flux
magnitude when applicable. Native CuPy checks are optional only when CUDA is
unavailable; the Product port is not natively commissioned by this milestone.

A predeclared same-fixture 4096x64 benchmark compares the committed M2 path with
M3, including exact q/psi/b and diagnostics hashes. Structural tests prohibit
routing reduced columns to the connected wrapper. Timings detect gross regression
with normal noise qualification, not a sub-percent performance claim.

## Connected direct guard after scalable integration

The 12,288-node guard remains mandatory for the connected **direct/reference**
solver described here. It is not a universal full-x-y restriction: explicit
scalable requests have their own backend/precision qualification envelope.
Reduced independent-column behavior and the direct/reference equations remain
unchanged. Historical requests never acquire scalable identity implicitly.
