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

## Reduced uniform-reference linearization

Let `E = E_bar + e`, `I = I0 + i`, and
`E_bar = E_app I_b / I0`. To first order,

```text
e_t = -I0 (1 + E_bar d_x - d_xx) e + (d_x - E_bar) i.
```

The static response sets the left time derivative to zero. The production
operator uses the repository's actual discrete derivative symbols; reduced
centered-difference results need not match a continuum-spectral full-
transverse reduction at finite resolution. The exact-modal reduced TD update
evolves the derived discrete equation directly, including its even-grid
Nyquist behavior.

This is a tangent material model about a declared uniform reference. The
optical source may still be refreshed self-consistently, so “linearized
material” does not mean globally linear optical propagation.

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
- Linearized requests require an explicit finite positive `I0`; it is not
  inferred from beam peaks, dark intensity, background, or spatial mean.
- Optical boundary absorption is material-neutral and does not change the
  material electrical profile.

## Approximation and validation language

“Fully nonlinear” and “linearized” identify material physics. “Production”
means the request is an intended supported model choice. Local validation,
backend coverage, composite-experiment validation, and H200 commissioning are
separate evidence and must be stated independently.

Linearization requires small field/intensity perturbations and resolved
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
