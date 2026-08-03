# Minimal photorefractive vertical slice

## State, normalization, and boundaries

The material state is the normalized space-charge field `E` with shape
`(Nz, Nx, Ny)`. The hopping model is

```text
E_t = E_app I_b - (E I - I_x)(1 + E_x) + I E_xx
```

where `x_normalized = k0*x`, `I = I_optical/I_peak_reference + I_b`, and
`I_b` is dark intensity plus the optional uniform background. The positive
`I E_xx` term is the diffusive sign in the paper and trusted implementation.

The material derivatives are periodic in x and act on array axis `-2`.
There are no y derivatives in this scalar hopping equation, so each y column
has independent material evolution; the optical FFT propagation remains
periodic in both x and y. Numerical periodicity is not a claim that a finite
crystal is physically periodic. A finite-beam calculation should use an
aperture large enough that optical intensity and material perturbations are
negligible at the transverse edges. PRProp3D additionally apodized the optical
field to suppress FFT wraparound. This minimal slice deliberately does not.

Centered differences were selected for the first slice because they are
local, inexpensive on NumPy and CuPy, converge at second order, and leave a
simple path to implicit treatment of the stiff diffusion term. Their Fourier
symbols are

```text
k1_modified = sin(k*dx)/dx
k2_modified_squared = 4*sin(k*dx/2)^2/dx^2
```

Thus they attenuate resolved high-frequency derivatives relative to spectral
derivatives, especially the first derivative near Nyquist. Frequency-sweep
tests characterize that tradeoff explicitly. A PR-owned spectral reference
path can be added later if benchmark work needs it; LC optics does not need to
change.

## Explicit-Euler timestep guard

Paper Equation (15) chooses one quarter of the shortest linearized time
constant:

```text
dt_paper = 1 / (4*(1 + k_max^2))
```

The trusted PRProp3D source actually computes `1/(4*k_max^2)`, omitting the
reaction term. Both use a spectral wavenumber and effectively take the
normalized uniform optical intensity as one.

For the implemented centered differences, linearization about uniform `I0`
and `Ebar` gives

```text
mu(k) = -I0*(1 + k2_modified_squared)
        - 1j*I0*Ebar*k1_modified
```

and forward Euler is absolutely stable when

```text
dt <= 2*(-Re(mu))/abs(mu)^2.
```

The workflow guard evaluates this boundary over every representable mode with
`I0 = 1 + I_b` and `Ebar = E_app*I_b/I0`, then divides it by eight. For zero
drift this is `1/[4*I0*(1 + k2_modified_squared)]`: the paper's quarter-time-
constant margin, adapted to the implemented operator and total plane-wave
intensity. It is a conservative uniform-state linearization, not a proof of
stability for arbitrary nonlinear, spatially concentrated intensity.

## Optical and material stepping

The improved optical coupling remains a frozen-E Strang pass:

```text
half PR response -> linear hop -> half PR response
```

for every optical substep through `advance_prepared_response()`. After a full
optical pass, all z slices of E receive one synchronous material-time update.
This intentionally differs from PRProp3D's interleaved full-hop/full-response
Lie ordering.

Explicit Euler is retained only as a transparent reference integrator:

- Euler is simplest but inherits the high-frequency diffusion restriction.
- Heun/RK2 improves temporal accuracy but does not remove diffusion stiffness.
- Semi-implicit diffusion removes the dominant real high-frequency stiffness
  while retaining explicit nonlinear drift and source terms.
- Semi-implicit diffusion with Picard correction can improve coupling of the
  variable coefficient `I` and nonlinear `(E*I-I_x)*E_x` term.
- Adaptive control is valuable after a reliable higher-order or implicit step
  and an error estimator exist.

The recommended next integrator is semi-implicit diffusion with the remaining
terms explicit, followed by a Picard correction if benchmarks show it is
needed. It directly targets `I E_xx`, the stiffest high-frequency term, while
preserving a small PR-owned solver and backend-portable local operators.
