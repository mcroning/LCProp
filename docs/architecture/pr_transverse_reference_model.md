# Transverse Photorefractive Hopping Reference Model

## Purpose and scope

LCProp's production photorefractive material model remains the reduced-x A7
equation

\[
E_t=E_{\rm app}I_d-(EI-I_x)(1+E_x)+IE_{xx}.
\]

The implementation in `src/lcprop/pr/transverse_reference.py` is a separate,
small NumPy reference for the full transverse continuum structure recovered
from hopping Equations A1--A5. It is not a production workflow, is not GPU
optimized, and does not change the production A7 implementation.

## Governing equations

The normalized carrier ratio is \(P=p/p_d\), where \(P=1\) is the uniform
neutral reference state. The transverse carrier equation
and electrostatic closure are

\[
P_\tau=\nabla_\perp\cdot\left[
\mathbf M\left(\nabla_\perp(PI)-PI\mathbf e\right)\right],
\]

\[
P-1=\nabla_\perp\cdot(\mathbf H\mathbf e),
\qquad \nabla_\perp\times\mathbf e=0.
\]

The reference uses a periodic zero-mean potential,

\[
\mathbf e=(E_{\rm app},0)-\nabla_\perp\psi,
\]

so that

\[
P=1-\nabla_\perp\cdot(\mathbf H\nabla_\perp\psi).
\]

The evolved potential satisfies

\[
-\nabla_\perp\cdot(\mathbf H\nabla_\perp\psi_\tau)
=\nabla_\perp\cdot\left[
\mathbf M\left(\nabla_\perp(PI)
-PI\left((E_{\rm app},0)-\nabla_\perp\psi\right)\right)
\right].
\]

Only constant diagonal tensors are represented:

\[
\mathbf M=\operatorname{diag}(1,m_y),
\qquad
\mathbf H=\operatorname{diag}(1,h_y).
\]

The defaults (m_y=h_y=1) define an isotropic transverse reference. They are
assumptions, not measured or tensor-calibrated BaTiO3 coefficients. No values
for the (D_y/D_x) or \(\varepsilon_y/\varepsilon_x\) ratios used in the
historical k-y-narrowing calculation have been recovered.

## Crystal-axis-controlled dielectric reference

The isotropic default remains \(h_y=1\). An additive constructor,
`TransverseReferenceOptions.from_barium_titanate_c_axis`, provides a bounded
BaTiO3 dielectric reference whose physical input is
`c_axis_xz_angle_deg`. The angle convention is

\[
\hat{\mathbf c}=(\cos\gamma,0,\sin\gamma),
\]

so \(\gamma=0^\circ\) places the tetragonal c-axis along simulation x and
\(\gamma=90^\circ\) places it along propagation z.

The crystal is treated as uniaxial with crystal-frame dielectric tensor

\[
\boldsymbol{\varepsilon}_{\mathrm{crystal}}
=\operatorname{diag}(\varepsilon_a,\varepsilon_a,\varepsilon_c).
\]

The implementation uses approximate room-temperature clamped reference
values

\[
\varepsilon_a=\varepsilon_{11}=2200,
\qquad
\varepsilon_c=\varepsilon_{33}=56.
\]

They are exposed as inputs and constants rather than treated as universal
BaTiO3 values. Rotation into simulation coordinates is equivalently

\[
\boldsymbol{\varepsilon}
=\varepsilon_a\mathbf I
+(\varepsilon_c-\varepsilon_a)\hat{\mathbf c}\hat{\mathbf c}^{T}.
\]

Because the material solver represents only the transverse x-y plane, the
required block is diagonal:

\[
\varepsilon_{xx}
=\varepsilon_a\sin^2\gamma+\varepsilon_c\cos^2\gamma,
\qquad
\varepsilon_{yy}=\varepsilon_a,
\qquad
\varepsilon_{xy}=0.
\]

Normalizing by \(\varepsilon_{xx}\) preserves \(H_{xx}=1\) and gives

\[
h_y(\gamma)=\frac{\varepsilon_{yy}}{\varepsilon_{xx}}
=\frac{\varepsilon_a}
{\varepsilon_a\sin^2\gamma+\varepsilon_c\cos^2\gamma}.
\]

The analytic orientation limits for the reference values are:

| c-axis angle | \(\varepsilon_{xx}\) | \(\varepsilon_{yy}\) | \(h_y\) |
| --- | ---: | ---: | ---: |
| \(0^\circ\), c along x | 56 | 2200 | 39.2857142857 |
| \(45^\circ\) | 1128 | 2200 | 1.95035460993 |
| \(90^\circ\), c along z | 2200 | 2200 | 1 |

`RotatedUniaxialDielectric` exposes the angle, crystal-frame constants, both
derived transverse components, and \(h_y\). Direct low-level construction of
`TransverseReferenceOptions` with an explicit `h_y` remains available for
isotropic validation and controlled numerical tests.

This is a physically motivated rotated crystal-frame electrostatic model, not
the complete angle-dependent photorefractive effective dielectric response.
The constants are clamped values; electromechanical contributions and the
grating-wavevector-dependent
\(\varepsilon_{\mathrm{eff}}^{PR}(\mathbf k)\) are absent. Transport remains
isotropic unless an existing low-level `m_y` test override is supplied. The
optical projection remains \(E_{\mathrm{active}}=E_x\); although the same
crystal orientation will eventually enter a full electro-optic tensor
contraction, that contraction is outside this reference extension.

## State and integration

The authoritative evolved state is (psi). This choice enforces a curl-free
field by construction, fixes the potential gauge explicitly, and derives the
carrier density and both field components from one state:

\[
E_x=E_{\rm app}-\psi_x,
\qquad E_y=-\psi_y.
\]

Time advancement is explicit Euler. It is intentionally transparent and is
appropriate only for small validation grids with a timestep selected below the
resolved diffusion stability limit. There is no adaptive stepping or claim of
production robustness.

## Boundaries and FFT conventions

Both material axes are periodic. The potential has zero spatial mean and the
uniform applied field is the separately prescribed harmonic component. Total
carrier number is conserved by the periodic divergence.

Derivatives use NumPy's unshifted `fft2`/`ifft2` ordering and angular
wavenumbers `2*pi*fftfreq(N, d=spacing)`. For even grids, the unpaired Nyquist
coefficient of each real first derivative is set to zero. Potential modes that
are null under the resulting discrete gradient-divergence operator are also
fixed to zero. This makes gradient, divergence, electrostatic inversion, curl,
and Gauss diagnostics one internally consistent discrete operator.

## Optical response

Transport produces (E_x) and (E_y), but the optical response remains a
scalar projection. The reference interface defaults to

\[
E_{\rm active}=E_x.
\]

This is an explicit geometry assumption. It is not implied by the transport
equations and is not a full BaTiO3 Pockels-tensor calculation. The existing
production `gain_length_product` interpretation is unchanged.

## Reduction to A7

For fields and intensity independent of y, (m_y=h_y=1), and (E_y=0),

\[
P=1+\partial_x E_x.
\]

The carrier equation integrates once to

\[
E_t=(PI)_x-PIE+C(t).
\]

Substitution of \(P=1+E_x\), followed by the paper's boundary choice
(C=E_{\rm app}I_d), gives A7 exactly.

The regression test makes two comparisons. First, it evaluates the transverse
reference and A7 with the same spectral derivatives on a band-limited,
y-uniform problem; these agree to roundoff. Second, it compares the spectral
reference with production A7, whose authoritative derivatives are centered
finite differences. That difference converges at second order as x resolution
is refined and is therefore a discretization difference, not a physics
difference.

## Validation and limitations

`tests/test_pr_transverse_reference.py` contains the four original physics
gates: y-independent A7 reduction, carrier conservation, curl/Gauss closure,
and a localized two-dimensional forcing case with nonzero (E_y). It also
checks the three analytic crystal-axis orientations and a localized isotropic
versus rotated-dielectric comparison. The compact visual check is
`scripts/checks/pr_transverse_reference.py`.

The reviewed local validation produced:

| Gate | Result |
| --- | --- |
| Spectral y-independent reduction | relative error (6.0\times10^{-15}) to (5.6\times10^{-13}) for (N_x=32\ldots256) |
| Production centered-A7 comparison | relative error (2.71\times10^{-2}, 6.86\times10^{-3}, 1.72\times10^{-3}, 4.30\times10^{-4}), demonstrating second-order refinement |
| Carrier conservation | maximum relative drift (1.18\times10^{-16}) |
| Curl closure | RMS (6.03\times10^{-18}), maximum (3.21\times10^{-17}) |
| Gauss closure | RMS (6.05\times10^{-17}), maximum (1.26\times10^{-16}) |
| Localized transverse response | RMS (E_y=7.26\times10^{-3}); RMS (P-1=8.02\times10^{-3}) |
| Difference from independent A7 y columns | relative (E_x) difference (1.85\times10^{-2}) after normalized time 0.1 |

For the same localized 64 by 56 intensity profile evolved to normalized time
0.2, the isotropic and 45-degree rotated-dielectric results were:

| Quantity | Isotropic \(h_y=1\) | 45-degree BaTiO3 reference |
| --- | ---: | ---: |
| RMS \(E_x\) | 0.00578489380087 | 0.00454832392545 |
| Maximum \(|E_x|\) | 0.0428524752621 | 0.0321660948938 |
| RMS \(E_y\) | 0.00991267699839 | 0.00564664643982 |
| Maximum \(|E_y|\) | 0.0738908728987 | 0.0428523989164 |
| RMS \(P-1\) | 0.00950760300109 | 0.00974244097343 |

The relative field differences were 0.279862964085 for \(E_x\) and
0.438651958420 for \(E_y\). The plotting script uses shared color limits for
corresponding isotropic and anisotropic fields.

The A7 reduction validation uses zero applied field so the periodic fixed-mean
field convention does not introduce a different integration constant from the
paper's finite-boundary convention.

This reference does not claim to reproduce the historical k-y narrowing. It
does not include calibrated BaTiO3 transport or dielectric anisotropy,
crystallographic orientation machinery, a full electro-optic tensor, optical
propagation, a production static solve, or a CuPy backend.
