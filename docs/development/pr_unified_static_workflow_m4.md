# Unified headless Static optical/material integration (M4)

This is a direct, unregistered callable in `lcprop.pr.unified.workflow`.
It does not replace any existing workflow, GUI model, codec, or TD integrator.
M1–M3 material equations, closure definitions, precision policies and solvers
are unchanged. Native optical integration certification is still required.

## Identities and input boundary

`UnifiedStaticRequest` explicitly binds:

- workflow: `pr_static_unified_published_optical_first_v1`;
- arithmetic: `per_cell_full_linear_arriving_unified_material_full_phase_scattering_v1`;
- material: `pr_unified_nonlinear_hopping_v1`;
- spatial: `periodic_compatible_fitted_flux_v1`, with active/batch axes;
- named electrical closure and targets;
- M3 state/linear/Bernoulli precision policy;
- projection: `adjacent_face_arithmetic_to_optical_node_v1`.

The caller supplies an already prepared `(Nch,Nx,Ny)` complex launch in the
explicit NumPy/CuPy backend and exact execution dtype. There is no implicit
upload, dtype conversion, renormalization or launch builder at this milestone.
The launch is borrowed until a synchronous execution-safe copy is made; do not
mutate caller buffers concurrently with that acceptance. The workflow owns its
accepted launch snapshot and computes the existing sum-of-channel-peaks
scientific normalization from that snapshot. Coherence groups affect source
construction, independently of that normalization convention.

The spatial lengths must exactly match physical aperture times the configured
characteristic wavenumber. Reduced optical planes use active x and independent
batch y, with b shaped `(Ny,1)`. Full x-y uses connected `(Nx,Ny)` with b `(2,)`.
The connected 12,288-node direct/reference ceiling is checked before backend or
grid construction. Optical boundaries are periodic. Only integral numbers of
full requested cells are accepted, within a two-ULP binary64 endpoint envelope;
the final reported coordinate canonicalizes to L, without changing h.

Electrical closure is authoritative. The legacy PRMaterialSpec applied-field
scalar must be zero except for explicit A7, where it must equal the retained
reservoir field; A7 background must equal dark plus uniform background. Fixed
mean-field/current targets belong to the closure, not that legacy scalar.

## Projection and electro-optics

`electric_field_face` returns only the requested native oriented component,
using M3's neighbor/subtraction/division arithmetic. Stored x face i is at
`x_i+dx/2`, and likewise for y. `electric_field_optical_node` computes:

`(face + roll(face, +1, active_axis)) * state_dtype(0.5)`.

It does not substitute the algebraically equal centered-potential difference.
All projection operations remain in state precision on the state's device.
Outputs are detached; state buffers are borrowed and never mutated. Requesting
y from a reduced batch fails. Ordinary optics reconstructs x only; a requested
2D y product uses exactly the same arithmetic along y.

The symbol is `-i sin(k dx)/dx`, second-order at low frequency and zero for an
exact even-grid node Nyquist component. The cosine attenuation from averaging
faces is intentional collocation, not a replacement material operator. Never
use projected fields for M3 Gauss or conservative-flux acceptance. Harmonic b
is preserved, independently per batch column. Native and projected products
are explicitly named `electric_field_x_face` and
`electric_field_x_optical_node` (similarly y).

Existing `delta_n_from_E` is reused, followed by the full exponential from the
commissioned published driver. Only E_x drives optical response. The physical
phase is `exp(-2i * gain_length_product * h/L * E_x)`; production arithmetic
continues to form delta-n and multiply by the vacuum wavenumber. No material
half phase, extra n factor, new tensor projection, clipping or filter is added.

## Execution and transactionality

Each cell executes exactly:

`P_h -> arriving total I -> accepted M3 q/psi/b -> node E_x -> full M_h -> S`.

The material solver is called once per cell, with its internal nonlinear
iterations unchanged. Every named physical limit, finite/positive/converged
status returned by M3 must pass before projection. No coupled optical/material
iteration, midpoint averaging, backtracking, replay, or continuation exists.
Source intensity remains optical/reference + dark + uniform.

The optical field, material state and bookkeeping are candidates until the
entire cell passes. Fallible ledger/cut construction precedes the single
accepted-state replacement. Cancellation is checked before each cell and
again after full candidate construction/bookkeeping, before acceptance. An M3
solve is indivisible here. Cancellation during it can complete computation but
cannot accept that candidate. Failure returns the last accepted optical and
selected material state, cell count, reached z and compact stage/reason.
No rejected cell contributes cuts or coordinates. Product-generation failure
reports failure while preserving already accepted science and any original
cell failure reason.

Canonical V2 scattering is generated once per candidate cell and applied after
the full material phase. Preflight verifies contiguous half-open canonical
slab coverage. A cancelled/rejected candidate may have computed its phase but
does not become an accepted scientific cell; there is no retry or RNG advance
policy hidden in this interface.

## Minimal headless outputs and memory

The accepted endpoint is always returned. Selection optionally retains the
launch, raw complex x-z/y-z cuts, last q/psi/b, derived last carrier, last native
face fields, last optical-node fields, and an exact endpoint far-field spectrum
through `direction_cosine_spectrum`. These are backend runtime products, not a
display-normalization or persistence schema. Native face currents, previews
and codecs are not introduced just to satisfy a legacy API.

Optical boundary coordinates are `(0,h,...,mh)`; material right endpoints are
`(h,...,mh)`. Transverse face staggering never shifts the material z coordinate.
Launch-only cancellation has one boundary and no material sample. Far-field
output carries the actual accepted `far_field_z_um`, including partial results.
Cuts use the optical samples nearest zero x/y and record those indices/axes.

Default storage is transverse accepted/candidate optical and q/psi/b state,
M3 sparse solve workspace, and scalar per-column/per-cell evidence. Optional
cuts add `O(Ncells*Nch*(Nx+Ny))`, not longitudinal optical/material volumes.
Optional launch/endpoint material products add bounded transverse arrays.
Projection computes one active component at a time with face/roll/output
buffers; ordinary optics never computes y. No host plane transfer occurs.
Backend scalar qualification and M3's existing compact status transfers remain
permitted. The direct solver's sparse factorization memory is not estimated by
the transverse-array count. No native peak is claimed.

## Validation boundary

Focused tests bind the frozen design's Fourier tolerances (float32 absolute
4e-7, float64 1e-14 on its fixtures), smooth field/phase refinement order
1.85–2.15, exact replicated 1D/2D projection/phase equality, native face
arithmetic, state precision, ownership and one-component allocation behavior.
Workflow tests use independent explicit marches and the unchanged commissioned
cell primitive with an identical supplied response to isolate optical order
from different material equations. Fault injection covers each cell stage and
bookkeeping; cancellation and retention cannot change accepted science.

M1–M3 regressions, including frozen material references, remain required.
CuPy-only projection/workflow parity tests may skip locally for unavailable
CUDA; native commissioning must execute them and transfer guards later.

## S4-M2 explicit headless material solver

`UnifiedStaticRequest.solver` resolves omitted metadata to the historical reduced
independent-column or connected direct solver, according to active transport
dimensionality. It never selects a solver from grid size, backend or a planner.
Explicit `pr_unified_connected_scalable_v1` routes only the material operation to
the committed PCG/GMRES core; published optical ordering and cold initialization
are unchanged. State32 scalable execution requires
`state32_carrier64_coeff64_linear64_bernoulli64_v3`; state64 retains its existing
identity. Reduced/direct requests retain their existing precision policies.

The direct 12,288-node bound remains. Scalable metadata validation separately
limits ordinary NumPy grids to 96×96, CuPy state32 to 256×256 and CuPy state64 to
512×512, with the explicit unbiased 384×32 bridge. These are qualification
bounds, not memory estimates or convergence promises. The 128² CuPy routing test
is metadata-only and does not certify native Product workflow execution.

Scalable failure results retain the original numerical exception in the in-memory
`failure['material_exception']`, including its chained cause, last valid solver
iterate and trace. This is not an accepted optical/material workflow state.
Accepted state, products, coordinates and observers never advance on a failed
material call. V3 quantitative carrier/current products retain float64.

The S4-M2 codec bridge preserves V1 reduced/direct request meaning and vocabulary.
Scalable requests/results explicitly reject serialization until S4-M3 implements
complete versioned solver/precision persistence. No new schema, GUI selector,
transport registration or resource planning is introduced here.

Workflow overlap comparisons use the frozen root budgets (state32 2e-4/2e-5,
state64 1e-8/1e-8). Float32 face-current comparisons alone retain the prior
cancellation-aware absolute allowance
`max(2e-5,16*eps32*Imax*n_reference_max/min(h))`, with rtol 2e-4.
This does not relax physical residual, root, phase or optical-product gates.

### Local support-policy qualification update

The original ordinary NumPy 96×96 support boundary is superseded for float64
by qualification through 512×512. This changes only metadata eligibility and
planning; see [local qualification](pr_local_large_grid_qualification.md).
Scientific algorithms, state32/H200 bounds and the direct/reference guard
remain unchanged.
