# Full Transverse Hopping-Model Recovery

## Executive conclusion

The multidimensional hopping model can be recovered from Equations A1–A5, but it is not a local scalar equation obtained by adding $y$-derivative terms to A7.

The scientifically consistent transverse model consists of:

- a two-dimensional carrier-continuity equation;
- electrostatic closure through a scalar potential or curl-free vector field;
- a scalar optical response obtained by projecting that vector field onto the electro-optically active BaTiO₃ component.

The present A7 equation is recovered exactly when all y dependence vanishes. In more than one dimension, the integration of Gauss’s law that produces local scalar A7 is no longer available. A globally coupled 2D electrostatic solve is required.

No earlier full-transverse implementation was found in the LCProp branches, PRProp3D history, notebooks, scripts, documentation, or available unpublished prompts. Therefore, the structure of the correct model is recovered, but the particular transverse transport and dielectric coefficients used in the earlier $k_y$-narrowing research remain unknown.

No files were modified.

## 1. Starting equations

The paper’s hopping equation is

$$
\frac{dW_n}{dt}
=
-\sum_m D_{mn}
\left[
W_nI_n\exp\left(\frac{\beta\phi_{nm}}{2}\right)
-
W_mI_m\exp\left(\frac{\beta\phi_{mn}}{2}\right)
\right],
\tag{A1}
$$

where:

- $W_n$ is mobile-carrier occupation at site $n$;
- $I_n$ includes optical and equivalent dark/background intensity;
- $D_{mn}$ is the hopping coefficient;
- $\beta=q/(k_BT)$;
- $\phi_{nm}=\phi_n-\phi_m$.

For $|\beta\phi_{nm}|\ll1$, the exponentials are expanded to first order, giving A2. The published derivation then restricts the neighbors to $n\pm1$ along x and obtains

$$
\frac{\partial p}{\partial t}
=
Ds^2
\frac{\partial}{\partial x}
\left[
\beta pI\frac{\partial\phi}{\partial x}
+
\frac{\partial(pI)}{\partial x}
\right].
\tag{A5}
$$

The original hopping paper is Feinberg et al., “Photorefractive effects and light-induced charge migration in barium titanate”. The modern derivation appears in Appendix B of the Photonics PRProp3D paper.

## 2. Two-dimensional continuum limit

Let a site be located at transverse position $\mathbf r=(x,y)$, with symmetric nearest-neighbor displacement vectors $\pm\mathbf a_\alpha$. Define

$$
F(\mathbf r)=p(\mathbf r)I(\mathbf r).
$$

Expanding $F(\mathbf r\pm\mathbf a_\alpha)$ and $\phi(\mathbf r\pm\mathbf a_\alpha)$ through second order, the first-order terms cancel between opposite neighbors. The surviving contribution from direction $\alpha$ is

$$
D_\alpha
(\mathbf a_\alpha\cdot\nabla)
\left[
(\mathbf a_\alpha\cdot\nabla)(pI)
+
\beta pI(\mathbf a_\alpha\cdot\nabla)\phi
\right].
$$

Summing over independent positive neighbor directions gives

$$
\boxed{
\frac{\partial p}{\partial t}
=
\nabla_\perp\cdot
\left[
\mathbf K_\perp
\left(
\nabla_\perp(pI)
+
\beta pI\nabla_\perp\phi
\right)
\right]
}
\tag{1}
$$

with the hopping/transport tensor

$$
\mathbf K_\perp
=
\sum_{\alpha>0}
D_\alpha\,\mathbf a_\alpha\mathbf a_\alpha^{T}.
\tag{2}
$$

Since

$$
\mathbf E_\perp=-\nabla_\perp\phi,
$$

Equation (1) is equivalently

$$
\boxed{
\frac{\partial p}{\partial t}
=
\nabla_\perp\cdot
\left[
\mathbf K_\perp
\left(
\nabla_\perp(pI)-\beta pI\mathbf E_\perp
\right)
\right].
}
\tag{3}
$$

The associated carrier flux is

$$
\boxed{
\mathbf j_p
=
\mathbf K_\perp
\left[
\beta pI\mathbf E_\perp-\nabla_\perp(pI)
\right],
\qquad
p_t+\nabla_\perp\cdot\mathbf j_p=0.
}
\tag{4}
$$

Thus:

- $\beta pI\mathbf E_\perp$ is drift;
- $-\nabla(pI)$ is diffusion;
- variations of $I$ drive carriers even when $p$ is initially uniform;
- transverse hopping is governed by the tensor $\mathbf K_\perp$, not automatically by an isotropic Laplacian.

For a rectangular nearest-neighbor lattice aligned with x and y,

$$
\mathbf K_\perp
=
\begin{pmatrix}
D_xs_x^2&0\\
0&D_ys_y^2
\end{pmatrix}.
$$

The isotropic square-lattice specialization is $D_x=D_y$ and $s_x=s_y$.

## 3. Required electrostatic closure

The hopping equation evolves charge density, not an independently arbitrary electric field. Electrostatics requires

$$
\boxed{
\mathbf E_\perp=-\nabla_\perp\phi,
\qquad
\nabla_\perp\times\mathbf E_\perp=0,
}
\tag{5}
$$

and, within the paper’s transverse-slice approximation,

$$
\boxed{
\nabla_\perp\cdot
\left(\boldsymbol{\varepsilon}_\perp\mathbf E_\perp\right)
=
q(p-p_d).
}
\tag{6}
$$

Equivalently,

$$
\boxed{
-\nabla_\perp\cdot
\left(\boldsymbol{\varepsilon}_\perp\nabla_\perp\phi\right)
=
q(p-p_d).
}
\tag{7}
$$

This is the essential difference from the current model. In one dimension, Gauss’s law can be integrated locally. In two dimensions, the charge distribution determines a global potential.

For tetragonal BaTiO₃, the effective dielectric response depends on crystal and grating orientation; it is not generally justified to replace $\boldsymbol{\varepsilon}_\perp$ with a scalar without declaring that approximation. This angular dependence is documented experimentally by Zgonik, Nakagawa, and Günter in JOSA B 12, 1416–1421.

## 4. Normalized full transverse system

Let the normalized mobile-carrier ratio be

$$
P=\frac{p}{p_d},\qquad
i=\frac{I}{I_0},
$$

where $P=1$ is the uniform neutral reference state,

and use the paper’s normalized transverse coordinate. Define normalized tensors

$$
\mathbf M=\frac{\mathbf K_\perp}{K_x},
\qquad
\mathbf H=\frac{\boldsymbol{\varepsilon}_\perp}{\varepsilon_x}.
$$

Let $\mathbf e$ be normalized electric field. The recovered system is

$$
\boxed{
\frac{\partial P}{\partial\tau}
=
\nabla_\perp\cdot
\left[
\mathbf M
\left(
\nabla_\perp(Pi)-Pi\mathbf e
\right)
\right],
}
\tag{8}
$$

$$
\boxed{
P-1=\nabla_\perp\cdot(\mathbf H\mathbf e),
\qquad
\nabla_\perp\times\mathbf e=0.
}
\tag{9}
$$

Writing the field as

$$
\mathbf e=\mathbf e_h-\nabla_\perp\psi,
$$

where $\mathbf e_h$ is the imposed harmonic/applied-field component, gives

$$
\boxed{
P=1-\nabla_\perp\cdot
(\mathbf H\nabla_\perp\psi).
}
\tag{10}
$$

The time-dependent potential equation is consequently

$$
\boxed{
-\nabla_\perp\cdot
(\mathbf H\nabla_\perp\psi_\tau)
=
\nabla_\perp\cdot
\left[
\mathbf M
\left(
\nabla_\perp(Pi)-Pi(\mathbf e_h-\nabla_\perp\psi)
\right)
\right].
}
\tag{11}
$$

Equations (8)–(11), not an $E_{xx}+E_{yy}$ substitution, are the recovered full-transverse hopping model.

For constant isotropic tensors and periodic boundaries, the equivalent vector-field form is

$$
\mathbf e_\tau
=
\mathcal P_{\mathrm{curl\text{-}free}}
\left[
\nabla_\perp(Pi)-Pi\mathbf e
\right],
\tag{12}
$$

where $\mathcal P_{\mathrm{curl\text{-}free}}$ is the longitudinal Helmholtz projection. This projection is nonlocal in two dimensions.

## 5. Explicit reduction to production A7

Set:

$$
\partial_y=0,\qquad
M_{xx}=H_{xx}=1,\qquad
e_y=0,
$$

and denote $e_x=E$. Gauss’s law becomes

$$
P=1+E_x.
$$

Equation (8) becomes

$$
\frac{\partial P}{\partial t}
=
\frac{\partial}{\partial x}
\left[
\frac{\partial(PI)}{\partial x}-PIE
\right].
$$

Since $P_t=(E_t)_x$,

$$
E_t
=
(PI)_x-PIE+C(t).
$$

Substituting $P=1+E_x$,

$$
E_t
=
\left[I(1+E_x)\right]_x
-EI(1+E_x)+C(t),
$$

or

$$
E_t
=
I_x(1+E_x)+IE_{xx}
-EI(1+E_x)+C(t).
$$

Therefore,

$$
\boxed{
E_t
=
-(EI-I_x)(1+E_x)+IE_{xx}+C(t).
}
\tag{13}
$$

Under the paper’s boundary assumption—optical excess intensity vanishes, $I=I_d$, derivatives vanish, and the boundary field is $E_{\rm app}$—the integration constant is

$$
C(t)=E_{\rm app}I_d.
$$

Hence

$$
\boxed{
E_t
=
E_{\rm app}I_d
-(EI-I_x)(1+E_x)
+IE_{xx},
}
$$

which is exactly production Equation A7 and [hopping_rhs() (line 186)](/Users/mcroning/LCProp/src/lcprop/pr/evolution.py:186).

## 6. BaTiO₃ electro-optic response

Two-dimensional charge transport does not require vector optical propagation.

The electro-optic response is a contraction of the BaTiO₃ Pockels tensor with the space-charge field and the optical polarization. For fixed crystal orientation and polarization, it reduces to a scalar effective index change:

$$
\Delta n_{\rm eff}
=
-\frac12 n_{\rm eff}^3 r_{\rm eff}E_{\rm active},
$$

where

$$
E_{\rm active}=\mathbf g\cdot\mathbf E
$$

is the geometry-dependent active field projection.

Tetragonal BaTiO₃ has a particularly large $r_{42}=r_{51}$ coefficient. Its magnitude and relation to transverse dielectric response are documented in the original electro-optic measurements: JOSA 55, 828.

Published anisotropic photorefractive models demonstrate precisely this separation:

- a 2D electrostatic potential satisfies an elliptic equation containing both x and y derivatives;
- the scalar optical index response is proportional to one field component, commonly $\partial_x\phi$.

For example,

$$
\nabla_\perp^2\phi
+
\nabla_\perp\phi\cdot\nabla_\perp\ln(1+I)
=
\partial_x\ln(1+I),
$$

while the scalar optical response is

$$
n_{\rm NL}\propto\partial_x\phi.
$$

See the model equations in “Two-dimensional self-trapped nonlinear photonic lattices”.

Therefore:

- $E_y$ participates in transport and electrostatic closure;
- $E_y$ need not be applied directly to LCProp’s scalar phase screen;
- the phase screen may continue to use the appropriately projected $E_{\rm active}$;
- identifying $E_{\rm active}=E_x$ must be an explicit geometry choice, not an automatic consequence of notation.

LCProp’s current gain_length_product folds $r_{\rm eff}$, orientation, and interaction strength into a calibrated scalar response. The current request objects do not contain enough crystallographic orientation and polarization information to reconstruct the full tensor contraction independently.

## 7. Boundary conditions

Two distinct conventions must not be conflated.

The paper’s A7 derivation uses a physical far-boundary condition:

- optical excess intensity vanishes;
- $I\rightarrow I_d$;
- the field approaches $E_{\rm app}$;
- derivatives vanish.

That condition produces $C=E_{\rm app}I_d$.

A periodic transverse implementation instead naturally uses:

- periodic carrier density and periodic space-charge potential;
- fixed mean potential, usually $\langle\psi\rangle=0$;
- prescribed harmonic/mean applied field;
- conserved total carrier number.

For zero applied field, these conventions agree cleanly for localized beams in a sufficiently large aperture. With nonzero applied field, the distinction must be tested because a fixed-mean periodic field does not automatically reproduce the paper’s boundary-derived integration constant.

## 8. Recovered historical implementation

No implementation of the full transverse hopping model was found in:

- every LCProp branch and accessible Git revision;
- current PR code and research scripts;
- documentation and development records;
- attached text archives;
- the full public PRProp3D Git history;
- PRProp3D’s main notebooks;
- earlier fanning notebooks;
- the basic-birefringence notebook.

All recovered PRProp3D variants compute $I_x$, $E_x$, and $E_{xx}$ using FFTs along x only. The basic-birefringence work altered optical propagation but retained the same x-only material equation.

Thus the reported earlier $k_y$-narrowing calculation is not preserved in the sources currently available. In particular, I could not recover:

- $D_y/D_x$;
- the transverse dielectric ratio;
- its material boundary conditions;
- whether it used a carrier or potential state;
- the exact optically active field projection.

## 9. Solver consequences

The independent cyclic x-line solves are valid only for A7, where every y column is materially independent.

They are not valid for Equations (8)–(11), because:

- y transport couples neighboring y samples;
- Poisson/electrostatic closure couples the entire x–y plane;
- eliminating the potential produces a nonlocal Helmholtz projection;
- the static Jacobian is a genuinely 2D operator.

The z-slice organization can remain: longitudinal material derivatives are still omitted, so each accepted optical slice has its own transverse material problem. But each such problem becomes a coupled 2D solve.

For constant coefficients and periodic boundaries:

- Poisson inversion can use a 2D FFT;
- the operation is naturally NumPy/CuPy-compatible;
- cost is $O(N_xN_y\log(N_xN_y))$ per electrostatic solve;
- a static Newton solve should use matrix-free Newton–Krylov or another 2D nonlinear method;
- the existing RMS Armijo plus nonincreasing maximum-residual rule remains applicable to the global residual;
- the cyclic solver remains useful only as a reduced-model solver or possible preconditioner.

Compared with the present $O(N_xN_y)$ batched cyclic solve, the full system adds FFTs and Krylov iterations. The practical slowdown cannot be responsibly estimated before a small reference implementation establishes iteration counts.

## 10. Smallest scientifically correct production path

I recommend the following eventual implementation boundary:

- Preserve A7 as the existing explicit reduced_x model.
- Add an opt-in transverse_potential model.
- Treat carrier density plus potential—or potential with derived carrier density—as the authoritative material state.
- Represent transport and dielectric anisotropy explicitly with constant 2×2 transverse tensors.
- Use periodic x–y electrostatics initially, with a prescribed mean applied field and zero-mean space-charge potential.
- Compute the optical screen from an explicit active-field projection, initially preserving the existing calibrated x-directed response.
- Use a small CPU 2D reference solve before designing a production GPU solver.
- Require exact y-independent reduction to A7.

One remaining physics decision is required before calling this specifically a calibrated BaTiO₃ model: authoritative values or an explicit declared assumption for

$$
D_y/D_x
\quad\text{and}\quad
\varepsilon_y/\varepsilon_x.
$$

Setting both to one would be the exact isotropic 2D continuation of the paper’s simplest lattice assumptions, but it would be an explicitly isotropic transverse-hopping reference—not a fully tensor-calibrated BaTiO₃ model.

The equations are now sufficiently recovered to design a reference implementation, but the historical coefficient set that reportedly produced $k_y$ narrowing has not been recovered.

---

# Appendix A — Verbatim Codex Completion Report

Implemented the minimal 2D photorefractive reference solver as a fully isolated CPU/NumPy model. Production PR physics, workflows, solvers, APIs, and GPU paths were not changed.

## Files added

- [transverse_reference.py](/Users/mcroning/LCProp/src/lcprop/pr/transverse_reference.py)
- [test_pr_transverse_reference.py](/Users/mcroning/LCProp/tests/test_pr_transverse_reference.py)
- [pr_transverse_reference.py](/Users/mcroning/LCProp/scripts/checks/pr_transverse_reference.py)
- [pr_transverse_reference_model.md](/Users/mcroning/LCProp/docs/architecture/pr_transverse_reference_model.md)

## Reference model

The authoritative state is a periodic, zero-mean electrostatic potential $\psi(x,y)$:

$$
\mathbf e=(E_{\rm app},0)-\nabla\psi
$$

$$
P-1=\nabla\cdot(\mathbf H\mathbf e)
      =-\nabla\cdot(\mathbf H\nabla\psi)
$$

$$
-\nabla\cdot(\mathbf H\nabla\psi_\tau)
=
\nabla\cdot\left[
\mathbf M\left(\nabla(PI)-PI\mathbf e\right)
\right]
$$

with diagonal transport tensors

$$
\mathbf M=\operatorname{diag}(1,m_y),\qquad
\mathbf H=\operatorname{diag}(1,h_y).
$$

The implementation uses:

- periodic spectral derivatives in both transverse axes;
- FFT elliptic inversion;
- explicit Euler time integration;
- a fixed zero-mean potential gauge;
- consistent handling of even-grid Nyquist null modes;
- $m_y=h_y=1$ as an explicitly isotropic reference assumption;
- $E_{\rm active}=E_x$ as the documented scalar optical-response assumption.

This is a reference transport model, not a calibrated multidimensional BaTiO₃ production model.

## Validation gates

| Gate                                             | Result                                      |
| ------------------------------------------------ | ------------------------------------------- |
| 1D spectral reduction relative error             | $6.0\times10^{-15}$ to $5.5\times10^{-13}$  |
| Agreement with production centered-difference A7 | Second-order convergence under refinement   |
| Carrier conservation relative drift              | $1.18\times10^{-16}$                        |
| Curl RMS / maximum                               | $6.03\times10^{-18}$ / $3.21\times10^{-17}$ |
| Gauss-law RMS / maximum                          | $6.05\times10^{-17}$ / $1.26\times10^{-16}$ |
| Nontrivial transverse $E_y$, RMS / maximum       | $7.26\times10^{-3}$ / $4.52\times10^{-2}$   |
| Carrier perturbation RMS / maximum               | $8.02\times10^{-3}$ / $7.03\times10^{-2}$   |
| $E_x$ difference from independent 1D lines       | $1.85\times10^{-2}$                         |

The last result confirms that the localized 2D case produces genuine transverse coupling rather than merely evaluating independent x-directed solves.

For y-independent fields, the model reduces to the existing scalar equation after applying the paper’s longitudinal boundary-current condition. The documentation also explains why a periodic fixed-mean-field calculation with nonzero applied field is not automatically identical to that boundary convention.

## Tests

Focused reference tests:

7 passed in 0.39s

Complete PR suite:

190 passed, 9 skipped

This includes the external saved 3720 µm failure-snapshot validation, run separately because of its large stored fixture.

The visualization/reference script also completed successfully and produced:

- `/private/tmp/lcprop-pr-transverse-reference/transverse_reference_fields.png`
- `/private/tmp/lcprop-pr-transverse-reference/transverse_reference_fields.npz`
- `/private/tmp/lcprop-pr-transverse-reference/metrics.json`

git diff --check passed.

## Deliberate limitations

The reference presently has:

- no calibrated BaTiO₃ anisotropy;
- no recovered historical transverse-hopping coefficients;
- no full electro-optic tensor or vector optical response;
- no nonperiodic material boundaries;
- no implicit or production-grade time integrator;
- no CuPy/GPU implementation;
- no optical-workflow or coupled-static integration.

No files were committed or pushed.
