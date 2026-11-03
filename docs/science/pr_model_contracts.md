# Photorefractive Model Contracts

This document is the canonical Product-level scientific contract for LCProp's
reduced and full-transverse PR model choices. It records current equations,
normalization, electrical profiles, approximation boundaries, and parameter
ownership without reproducing historical development narratives.

## Shared intensity and optical projection

The complete dimensionless transport intensity is

```text
I = I_optical / I_peak_reference + I_b
I_b = I_dark + I_uniform
```

`I_peak_reference` is the sum of individual incident-channel peak intensities.
Fields in each coherence group are summed before forming intensity; group
intensities are then added. Coherent cross terms enter `I_optical` but not the
normalization reference. Dark and uniform backgrounds are already present in
the transport intensity and are not added again by a material operator.

The scalar optical response uses the declared active space-charge component.
The current full-transverse v1 projection is `E_active = E_x`; `E_y` remains a
material field and diagnostic rather than an extra scalar optical term.

## Canonical PR optical diffraction

Canonical PR uses full scalar angular-spectrum diffraction. For vacuum
wavelength lambda, reference index n, and unshifted transverse FFT frequencies
fx, fy in cycles/um:

```text
k = 2*pi*n/lambda
q = 1 - (lambda/n)**2 * (fx**2 + fy**2)
H(d) = exp(i*k*d*sqrt(q)) for q >= 0; otherwise 0
```

The longitudinal carrier phase is retained. Grazing q=0 is retained without
an epsilon; nonpropagating modes are discarded, also for negative or zero
distance. Forward/backward composition is identity only on retained spectral
support. It cannot recover cutoff, aperture or absorption losses.

The [shared primitive](../../src/lcprop/optics/splitstep.py) constructs the
cutoff and phase in backend float64, without host transfer, then returns the
requested complex64/complex128 optical dtype. Cutoff refers to the supplied
frequency grid, including its rounding. Near-cutoff phase is ill-conditioned;
CPU local tests do not constitute GPU commissioning.

The shared dispersion is used by the propagation paths, including optical
substeps and replay where configured. It does not identify the material-update
ordering: full-cell hops and frozen-material half-screen propagation are
separate consumers. See the static workflow table below for exact identities
and sequences; do not infer source cadence or scattering placement from the
diffraction kernel alone.
This remains scalar homogeneous diffraction plus local material phase screens,
not a full-vector or exact inhomogeneous Helmholtz solver.

Launch phase gradients retain their rad/um meaning. Narrow-packet trajectories
use Kx/Kz, Ky/Kz; Gaussian focus/radius construction remains the existing
paraxial entrance-field prescription, without an exact scalar-focus claim.

This pre-release migration does not preserve accidental ordinary Fresnel
behavior through a selector, checkpoint version, or saved-data migration.
Historical Image Amplification analysis keeps its paraxial post-processing
contract for now; it is not a scalar inverse. LC and its existing helpers are
not migrated. See the [development record](../development/lcprop_pr_full_angular_spectrum_migration.md).

## Reduced x-only nonlinear model

The reduced state is the normalized space-charge field `E`. Material
derivatives are periodic centered differences in x; y is an independent batch
axis. The canonical nonlinear equation is

```text
E_t = E_app I_b - (E I - I_x)(1 + E_x) + I E_xx.
```

Static execution solves the same residual set to zero. Candidate Newton states
must be finite and have strictly positive normalized carrier density
`n = 1 + E_x`; physical admissibility is checked before the unchanged
RMS/Armijo acceptance rule. The gate prevents acceptance of nonphysical states
but does not guarantee that a physical root can be found.

## Reduced Static field-linear response: paper Eq. (5)

The public `field_linear_local_intensity` response solves

```text
E + E_app D_x E - D_xx E = (E_app I_b + D_x I) / I.
```

I is the complete local transport intensity defined above. It must be finite
and strictly positive; there is no intensity floor and no fixed reference I0.
At zero applied field this is `(1-D_xx)E = D_x I/I`. The direct NumPy/CuPy
solve forms the local quotient before Fourier inversion. Its denominator is
`1 + k2_squared + i E_app k1`, using the trusted centered difference symbols.
At bias, the coefficient is E_app, not an inferred equilibrium E_app I_b/I0.
The matching residual is the left side minus the complete local quotient.

This implements Cronin-Golomb, Photonics 12, 113 (2025), Eq. (5), with production
centered discretization. It is linear in the material field for prescribed I,
but not the complete field-only Taylor expansion about arbitrary nonuniform I.
For zero bias, the full nonlinear equation additionally retains `(I_x/I)E_x`
and `-E E_x` on the right. The first of these is itself linear in E at fixed I.
Do not describe local-I forcing as the full nonlinear hopping equation.

The material equation does not identify the optical/material update sequence.
Use the explicit workflow and arithmetic identities below when interpreting a
saved request or extending a solver.

### Static workflow ordering

Here `P(d)` is shared angular-spectrum diffraction over distance `d`, `T(I)` is
a material solve for the stated intensity, `M(d,E)` is the corresponding local
material phase over `d`, and `S(h)` is optional canonical scattering for that
physical cell. These are implementation sequences, not numerical-order or new
scientific-validation guarantees. Optical substeps and half response screens
inside a frozen-material propagator do not imply a material update at every
substep.

| Variant and identity | Implemented sequence and source |
|---|---|
| Published optical-first reduced Static: `pr_static_published_optical_first_v1`; arithmetic `per_cell_full_linear_arriving_material_full_phase_scattering_v1` | For a cell of length `h`: `P(h) -> arriving I -> T(I) -> M(h,E) -> S(h)`. The material callback selects Local-I direct response or reduced nonlinear hopping. Material and scattering placement is at the cell's right endpoint; there is no final optical half hop. See [run_published_static / solve_arriving_material](../../src/lcprop/pr/published_static.py) and [step_published_cell](../../src/lcprop/pr/published_static_step.py). |
| Unified nonlinear Static: `pr_static_unified_published_optical_first_v1`; arithmetic `per_cell_full_linear_arriving_unified_material_full_phase_scattering_v1` | Also `P(h) -> arriving I -> unified material equilibrium -> M(h,E_active) -> S(h)`. It uses explicit electrical closure, spatial discretization, solver and projection contracts, rather than the published reduced material callback. See [execute_unified](../../src/lcprop/pr/unified/integration.py), [run_unified_static](../../src/lcprop/pr/unified/workflow.py) and [closure specifications](../../src/lcprop/pr/unified/specs.py). Do not substitute the reduced A7 equation for every unified closure or assume identical normalization. |
| Retained Local-I plane workflow: `pr_static_local_intensity_planes_v1`; arithmetic `per_cell_half_linear_full_local_phase_v1` | `P(h/2) -> center-plane I -> direct Local-I solve -> M(h,E) -> S(h) -> P(h/2)`. This is the distinct retained center-plane implementation, not the fresh GUI Local-I selection. See [run_local_intensity_planes](../../src/lcprop/pr/local_plane_workflow.py) and [step_local_intensity_cell](../../src/lcprop/pr/local_plane_step.py). |
| Coupled midpoint Static: `pr_static` | Trial material corrections repropagate the entrance field and refresh the source `0.5*(I_before+I_after)`; accepted states are checked by an independent optical replay. For frozen `E`, each of `Nsub` optical substeps uses material half-screen, `P(h/Nsub)`, material half-screen; scattering is applied after the slice. See [run_pr_static](../../src/lcprop/pr/static_workflow.py), [advance_pr_slice_with_midpoint_source](../../src/lcprop/pr/workflow.py), and [advance_prepared_response](../../src/lcprop/optics/splitstep.py). This differs from `P(h/2) -> T -> P(h/2)`. |
| Coupled transverse Static: `pr_transverse_static` | Material/optical fixed-point iterations use transverse material state and the same frozen-material midpoint optical primitive. See [run_pr_transverse_static / optical passes](../../src/lcprop/pr/transverse/static_workflow.py). It is distinct from the fresh nonlinear unified selection. |
| Streaming modes | `production_nonlinear_strang` uses coupled midpoint optical propagation and a second independent solve from the original launch for replay. `full_nonlinear_lie_reference` and `legacy_linearized_spectral_lie` retain reference-loop full-hop/arriving-source/material-phase ordering with optional windows and their own material operators. These reference modes are not fresh GUI defaults. See [static_streaming](../../src/lcprop/pr/static_streaming.py). |

The [GUI selector](../../src/lcprop/pr/gui/evolution_panel.py) routes fresh
nonlinear reduced/transverse Static to the unified workflow unless legacy or
image-amplification mode applies. Reduced Static Local-I selects the published
optical-first workflow. [Request construction](../../src/lcprop/pr/gui/request_adapter.py)
preserves their distinct request types. A historical half-hop description must
not be applied to those fresh GUI selections. The separate
[transverse marching reference](../../src/lcprop/pr/transverse/marching_static.py)
also uses frozen-material half response screens and optical substeps; it is not
the unified optical-first implementation.

Old `pr_static` requests retain their midpoint identity and remain headlessly
executable under that identity. The GUI rejects old midpoint Local-I requests
rather than converting them implicitly. This documentation reconciliation does
not certify native execution, convergence order, or scientific validity of any
new combination; use matching workflow-specific evidence.

## Uniform-reference tangent: research reduced operator and transverse model

Writing `E=E_bar+e`, `I=I0+i`, `E_bar=E_app I_b/I0`, joint first-order expansion
of the full reduced hopping equation gives

```text
e_t = -I0 (1 + E_bar d_x - d_xx) e + (d_x - E_bar) i.
```

This requires small field/intensity perturbations with controlled derivatives.
For unbiased static response, expanding `I_x/I` gives `i_x/I0 + O(i*i_x)`.
Thus freezing the intensity denominator is an additional approximation, not
synonymous with being linear in E. This tangent remains in research reference
modules and tests, not the reduced production GUI or execution workflows.
Saved reduced `linearized` requests are rejected, never reinterpreted as Eq. (5).

Reduced local-intensity TD is unsupported. A static closure does not uniquely
specify its transient. A possible extension `E_t=E_app I_b-I E+I_x-I E_app E_x+I E_xx`
has variable coefficients; Fourier modes couple for nonuniform I. It is neither
the existing exact-modal uniform-reference update nor the strict field-only
Taylor equation (which retains I_x E_x). Deriving and validating a particular
TD approximation and variable-coefficient integrator is separate follow-up.
Reduced production TD therefore exposes nonlinear hopping only.

## Full-transverse nonlinear Profile v1

The authoritative plane-local state is a periodic zero-mean potential
`psi(x,y,z,tau)`. With positive tensor coefficients,

```text
E = -grad(psi)
P = 1 - D_H psi
J = M [grad(P I) - P I E]
-D_H psi_t = div(J)
<psi> = 0, <P> = 1.
```

The named production v1 profile is isotropic (`m_y=h_y=1`), periodic in x and
y, uses `E_active=E_x`, and is **unbiased**. The static production closure is
pointwise zero flux. For positive `P` and `I`,

```text
P_eq = (exp(-psi) / I) / <exp(-psi) / I>.
```

The full-transverse nonlinear TD path uses the continuity equation above. A
nonzero harmonic mean field is deliberately rejected because Profile v1 does
not define a finite-electrode boundary model and a periodic biased steady state
can carry current rather than satisfy pointwise zero flux.

## Periodic biased current-carrying linearized profile

The linearized full-transverse profile is a distinct, explicitly named
fixed-mean-field periodic bulk model. It writes

```text
E = E_app e_x - grad(psi)
P = 1 - D_H psi
<psi> = 0, <P> = 1
J = M [grad(P I) - P I E].
```

The applied field is a harmonic component outside the periodic solved
potential. A uniform state has `psi=0`, `P=1`, and mean current
`<J_x> = -I0 E_app`. Static response means `div(J)=0`, not `J=0` pointwise.
At zero bias and positive state, this closure reduces to the existing
zero-flux equilibrium.

For each resolved Fourier mode about a uniform complete intensity `I0`, define

```text
a_M = k_x^2 + m_y k_y^2
a_H = k_x^2 + h_y k_y^2
D = I0 [a_M (1 + a_H) + i E_app k_x a_H].
```

Then

```text
delta_psi_hat = -(a_M + i E_app k_x) / D * delta_I_hat.
```

The constant gauge and every joint derivative-null mode in the repository's
even-grid spectral convention are set to zero and never divided. Fields and
carrier perturbation follow from

```text
delta_E_x_hat = -i k_x delta_psi_hat
delta_E_y_hat = -i k_y delta_psi_hat
delta_P_hat   = a_H delta_psi_hat.
```

Bias reversal conjugates the response symbol. The named production profile
uses `m_y=h_y=1` and `E_active=E_x`. Linearized static uses this analytic
frozen-source response inside its optical/material fixed-point iterations;
linearized TD integrates the exact modal transient for each frozen source.

## Parameter ownership

- `PRMaterialSpec.applied_field` belongs to reduced x-only A7 physics.
- The full-transverse linearized mean field belongs to the transverse
  electrical boundary profile as `applied_field_x`.
- Full-transverse nonlinear Profile v1 requires both meanings to be zero.
- Full-transverse tangent requests require an explicit finite positive `I0`; it is not
  inferred from beam peaks, dark intensity, background, or spatial mean.
- Optical boundary absorption is material-neutral and does not change the
  material electrical profile.

## Approximation and validation language

“Fully nonlinear” and “linearized” identify material physics. “Production”
means the request is an intended supported model choice. Local validation,
backend coverage, composite-experiment validation, and H200 commissioning are
separate evidence and must be stated independently.

Uniform-reference tangent linearization requires small field/intensity perturbations and resolved
derivatives; small amplitude alone is insufficient at unresolved or extreme
spatial frequency. Reduced-vs-full comparisons can contain physical model,
discretization, resolution, normalization, and approximation-regime effects.
No one discrepancy should be labeled solely as transverse transport without
separating those contributions.

## Implementation sources

The executable contracts are owned by:

- `src/lcprop/pr/evolution.py` and `src/lcprop/pr/static.py`;
- `src/lcprop/pr/reduced_linearized.py` and
  `src/lcprop/pr/reduced_linearized_timedependent.py`;
- `src/lcprop/pr/transverse/transport.py` and
  `src/lcprop/pr/transverse/static.py`;
- `src/lcprop/pr/transverse/linearized_reference.py` and
  `src/lcprop/pr/transverse/linearized_timedependent_reference.py`;
- the corresponding PR-owned workflow and request modules.

Historical derivations, rejected closures, commissioning narratives, and
paper-specific comparisons are Research records, not runtime dependencies of
this Product contract.
