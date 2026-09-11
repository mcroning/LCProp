# Full-Transverse Linearized PR Time-Dependent Reference

## Scope and status

This milestone adds an isolated, material-only reference for the periodic
biased full-transverse linearized time-dependent (TD) response. The optical
intensity perturbation is prescribed and constant in material time. There is
no optical propagation, optical self-consistency, workflow registration, GUI,
Slurm, or nonlinear-TD change.

The implementation is
`src/lcprop/pr/transverse/linearized_timedependent_reference.py`. It reuses the
accepted static response kernel from
`src/lcprop/pr/transverse/linearized_reference.py`; it does not duplicate or
modify that static operator.

## Recovered nonlinear equation and time normalization

The repository nonlinear transverse transport in
`src/lcprop/pr/transverse/transport.py` uses

\[
 P=1-D_H\psi,\qquad
 D_H=\nabla_\perp\!\cdot H\nabla_\perp,
 \qquad H=\operatorname{diag}(1,h_y),
\]

\[
 \mathbf E=E_{\rm app}\mathbf e_x-\nabla_\perp\psi,
 \qquad q=PI,
\]

\[
 \mathbf J=M[\nabla_\perp q-q\mathbf E],
 \qquad M=\operatorname{diag}(1,m_y),
\]

and

\[
 -D_H\psi_\tau=\nabla_\perp\!\cdot\mathbf J.
 \tag{1}
\]

The dynamical variable is the zero-mean periodic potential \(\psi\). The time
coordinate is exactly the existing dimensionless material time
\(\tau=t/t_0\), exposed as `time_normalized`; this reference introduces no new
time scale.

## Independent linearization

Linearize about the uniform current-carrying state

\[
 I=I_0+\delta I,\qquad \psi=\delta\psi,\qquad P=1+\delta P,
\]

where

\[
 \delta P=-D_H\delta\psi,
 \qquad
 \delta q=\delta I+I_0\delta P.
\]

To first order,

\[
 \delta\mathbf J
 =M[\nabla_\perp\delta q+I_0\nabla_\perp\delta\psi
      -E_{\rm app}\delta q\,\mathbf e_x],
\]

so the linearized TD equation is

\[
 -D_H\delta\psi_\tau
 =\nabla_\perp\!\cdot\delta\mathbf J.
 \tag{2}
\]

There is no independent generation or recombination source. The prescribed
\(\delta I\) enters through \(\delta q\), while the potential-dependent terms
provide decay and bias drift.

## Fourier convention and modal equation

For grid points \((x_j,y_l)\), the repository uses NumPy/CuPy `fft2`:

\[
 \widehat f_{pq}=\sum_{j,l}f_{jl}
 e^{-i(k_{x,p}x_j+k_{y,q}y_l)},
 \qquad
 f_{jl}=\frac{1}{N_xN_y}\sum_{p,q}\widehat f_{pq}
 e^{i(k_{x,p}x_j+k_{y,q}y_l)}.
\]

Define

\[
 a_M=k_x^2+m_yk_y^2,\qquad
 a_H=k_x^2+h_yk_y^2.
\]

For every resolved mode, \(a_H>0\), equation (2) becomes

\[
 a_H\frac{d\widehat{\delta\psi}}{d\tau}
 =-(a_M+iE_{\rm app}k_x)\widehat{\delta I}
 -I_0[a_M(1+a_H)+iE_{\rm app}k_xa_H]
  \widehat{\delta\psi}.
\]

Equivalently,

\[
 \frac{d\widehat{\delta\psi}}{d\tau}
 =-\Lambda(\mathbf k)\widehat{\delta\psi}
  +S(\mathbf k)\widehat{\delta I},
 \tag{3}
\]

with

\[
 \boxed{\Lambda
 =I_0\frac{a_M(1+a_H)+iE_{\rm app}k_xa_H}{a_H}}
 =I_0\left[\frac{a_M(1+a_H)}{a_H}+iE_{\rm app}k_x\right],
\]

\[
 \boxed{S=-\frac{a_M+iE_{\rm app}k_x}{a_H}}.
\]

Thus

\[
 \operatorname{Re}\Lambda
 =I_0\frac{a_M(1+a_H)}{a_H}>0
\]

on every resolved mode. Nonzero bias gives
\(\operatorname{Im}\Lambda=I_0E_{\rm app}k_x\): the transient has a decaying
envelope and a bias-dependent phase drift, but no instability.

## Exact transient and static-limit proof

For prescribed time-independent \(\delta I\), the exact solution of (3) is

\[
 \widehat{\delta\psi}(\tau)
 =\widehat{\delta\psi}_\infty
 +[\widehat{\delta\psi}(0)-\widehat{\delta\psi}_\infty]
 e^{-\Lambda\tau},
 \tag{4}
\]

where

\[
 \widehat{\delta\psi}_\infty
 =\frac{S}{\Lambda}\widehat{\delta I}
 =-\frac{a_M+iE_{\rm app}k_x}
 {I_0[a_M(1+a_H)+iE_{\rm app}k_xa_H]}
 \widehat{\delta I}.
 \tag{5}
\]

Equation (5) is exactly the accepted full-transverse static linearized
response. Since \(\operatorname{Re}\Lambda>0\), the exponential in (4)
vanishes as \(\tau\to\infty\), proving the required static limit for every
resolved mode. In code, equation (5) is obtained from the existing static
kernel rather than reimplemented.

## Bias reversal

At fixed Fourier coordinate,

\[
 \Lambda(-E_{\rm app})=\Lambda(E_{\rm app})^*,\qquad
 S(-E_{\rm app})=S(E_{\rm app})^*.
\]

The real decay rate is unchanged, while the phase/drift direction reverses.
For real prescribed and initial fields, conjugate Fourier symmetry is
preserved and the inverse transform remains real. Tests check exact array
conjugation of both modal symbols and the corresponding transient coefficient.

## Gauge and derivative-null modes

The reference imports the accepted static Fourier symbol. The constant mode
and every even-grid joint first-derivative null combination have \(a_H=0\).
The decay, source, equilibrium, initial, and evolved coefficients are fixed to
zero there: the implementation neither divides by nor evolves an undefined
mode. No epsilon regularization is used. Supplied initial conditions are
projected to this same zero-mean, derivative-resolved subspace.

All four odd/even grid-parity combinations are tested. Resolved results are
unchanged when constant or representable Nyquist-null content is added to the
source and initial potential.

## Initial conditions, batching, backend, and precision

The exact solver supports:

1. zero perturbation by default;
2. an arbitrary finite initial perturbation, projected to the accepted gauge;
3. initialization from the accepted static solution for equilibrium checks.

Inputs may have shape `(Nx, Ny)` or `(Nbatch, Nx, Ny)`. The leading dimension
contains independent material planes; it is not material-time history or
longitudinal evolution.

NumPy float64/complex128 is the default. NumPy float32/complex64 is supported
without promotion in returned fields or modal coefficients and is checked
against float64 with precision-appropriate tolerances. The same backend
abstraction supplies a CuPy-compatible path. Its test is conditional because
no local GPU execution or commissioning is part of this milestone.

## Local analytic validation

The focused tests cover:

- exact x, y, and oblique modal amplitude, phase, source, and decay;
- positive and negative bias and bias reversal;
- zero, arbitrary, and equilibrium initial conditions;
- independent batch semantics;
- odd/even gauge and Nyquist-null handling;
- a deterministic multimode long-time comparison with the accepted static
  reference at near-machine float64 precision;
- the \(\tau=0\) derivative against the independently evaluated linearized
  RHS;
- the nonlinear `potential_rhs()` Taylor oracle over four perturbation sizes,
  with second-order remainder;
- zero means, finiteness, curl-free electric perturbation, and Gauss closure;
- NumPy float32 dtype preservation and float64 agreement;
- conditional CuPy/NumPy agreement.

This is ordinary local candidate validation, not a retained authoritative
numerical-evidence package. No solver output artifact is committed.

## Limitations and deferred work

- The optical source is prescribed and time independent.
- There is no optical propagation, self-consistency, or time-varying source.
- The reference is not registered as a production workflow or public package
  API.
- CuPy compatibility is implemented but awaits actual GPU validation.
- No nonlinear TD equation, static response, current-carrying profile,
  propagation, GUI, persistence, transport codec, Slurm, IA, or soliton code is
  changed.

Production integration, optical coupling, persistence/transport exposure, GUI
status presentation, and GPU commissioning require separate milestones and
reviews.
