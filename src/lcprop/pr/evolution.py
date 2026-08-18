"""Normalized time-dependent photorefractive hopping-model evolution."""

from __future__ import annotations

import math
from typing import Any, Callable

from lcprop.pr.cyclic import solve_cyclic_tridiagonal_rows
from lcprop.pr.cyclic_gpu import solve_cyclic_tridiagonal_rows_gpu
from lcprop.pr.specs import (
    PRMaterialSpec,
    PR_EULER_INTEGRATOR,
    PR_SEMI_IMPLICIT_INTEGRATOR,
)


def centered_difference_symbols(
    k_normalized: float,
    *,
    dx_normalized: float,
) -> tuple[float, float]:
    """Return centered-difference ``(k1, k2_squared)`` mode symbols.

    For a Fourier mode ``exp(i*k*x)``, the implemented first derivative is
    ``i*k1`` and the implemented second derivative is ``-k2_squared``.
    """

    k = float(k_normalized)
    dx = float(dx_normalized)
    if not math.isfinite(k):
        raise ValueError("k_normalized must be finite")
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx_normalized must be finite and positive")
    phase = k * dx
    k1 = math.sin(phase) / dx
    k2_squared = 4.0 * math.sin(0.5 * phase) ** 2 / (dx * dx)
    return k1, k2_squared


def linearized_euler_mode_limit(
    k_normalized: float,
    *,
    dx_normalized: float,
    uniform_intensity: float,
    equilibrium_field: float = 0.0,
) -> float:
    """Return the exact Euler stability boundary for one discrete mode.

    Linearizing the implemented PDE about uniform ``I0`` and ``Ebar`` gives
    ``mu = -I0*(1 + k2_squared) - 1j*I0*Ebar*k1``. The returned
    step is the largest one satisfying ``|1 + dt*mu| <= 1``. It is a local,
    uniform-state result, not a nonlinear global guarantee.
    """

    intensity = float(uniform_intensity)
    field = float(equilibrium_field)
    if not math.isfinite(intensity) or intensity <= 0.0:
        raise ValueError("uniform_intensity must be finite and positive")
    if not math.isfinite(field):
        raise ValueError("equilibrium_field must be finite")
    k1, k2_squared = centered_difference_symbols(
        k_normalized,
        dx_normalized=dx_normalized,
    )
    decay = intensity * (1.0 + k2_squared)
    drift = intensity * field * k1
    return 2.0 * decay / (decay * decay + drift * drift)


def semi_implicit_amplification(
    dt_normalized: float,
    *,
    lambda_implicit: complex,
    lambda_explicit: complex,
) -> complex:
    """Return the prototype predictor/corrector scalar amplification."""

    dt = float(dt_normalized)
    if not math.isfinite(dt) or dt < 0.0:
        raise ValueError("dt_normalized must be finite and nonnegative")
    z_implicit = dt * complex(lambda_implicit)
    z_explicit = dt * complex(lambda_explicit)
    predictor = (1.0 + z_explicit) / (1.0 - z_implicit)
    return (
        1.0
        + 0.5 * (z_implicit + z_explicit)
        + 0.5 * z_explicit * predictor
    ) / (1.0 - 0.5 * z_implicit)


def linearized_semi_implicit_mode_limit(
    k_normalized: float,
    *,
    dx_normalized: float,
    uniform_intensity: float,
    equilibrium_field: float = 0.0,
) -> float:
    """Return the first scalar stability boundary for one split mode.

    Diffusion is assigned to the implicit operator.  Uniform-state reaction
    and drift remain explicit.  This is a local linearized result, not a
    nonlinear stability guarantee.
    """

    intensity = float(uniform_intensity)
    field = float(equilibrium_field)
    if not math.isfinite(intensity) or intensity <= 0.0:
        raise ValueError("uniform_intensity must be finite and positive")
    if not math.isfinite(field):
        raise ValueError("equilibrium_field must be finite")
    k1, k2_squared = centered_difference_symbols(
        k_normalized,
        dx_normalized=dx_normalized,
    )
    lambda_implicit = -intensity * k2_squared
    lambda_explicit = -intensity - 1j * intensity * field * k1

    scale = max(abs(lambda_implicit), abs(lambda_explicit), 1e-15)
    lower = 0.0
    upper = 1.0 / scale
    for _ in range(100):
        amplification = semi_implicit_amplification(
            upper,
            lambda_implicit=lambda_implicit,
            lambda_explicit=lambda_explicit,
        )
        if abs(amplification) > 1.0:
            break
        lower = upper
        upper *= 2.0
    else:
        return math.inf

    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        amplification = semi_implicit_amplification(
            midpoint,
            lambda_implicit=lambda_implicit,
            lambda_explicit=lambda_explicit,
        )
        if abs(amplification) <= 1.0:
            lower = midpoint
        else:
            upper = midpoint
    return lower


def paper_mode_timestep_rule(k_normalized: float) -> float:
    """Return the paper Equation (15) quarter-time-constant rule."""

    k = float(k_normalized)
    if not math.isfinite(k):
        raise ValueError("k_normalized must be finite")
    return 1.0 / (4.0 * (1.0 + k * k))


def legacy_mode_timestep_rule(k_normalized: float) -> float:
    """Return the rule coded in PRProp3D, including its k=0 singularity."""

    k = float(k_normalized)
    if not math.isfinite(k):
        raise ValueError("k_normalized must be finite")
    if k == 0.0:
        return math.inf
    return 1.0 / (4.0 * k * k)


def periodic_derivatives_x(field, *, dx_normalized: float, xp: Any):
    """Return centered first/second x derivatives with periodic boundaries.

    LCProp arrays use x on the penultimate axis for both ``(Nx, Ny)`` and
    ``(Nz, Nx, Ny)`` fields.
    """

    dx = float(dx_normalized)
    if not math.isfinite(dx) or dx <= 0.0:
        raise ValueError("dx_normalized must be finite and positive")
    if field.ndim < 2:
        raise ValueError("field must include x and y axes")
    forward = xp.roll(field, -1, axis=-2)
    backward = xp.roll(field, 1, axis=-2)
    first = (forward - backward) / (2.0 * dx)
    second = (forward - 2.0 * field + backward) / (dx * dx)
    return first, second


def hopping_rhs(
    E,
    intensity,
    *,
    applied_field: float,
    background_intensity: float,
    dx_normalized: float,
    xp: Any,
):
    """Return paper Equation (4)/(A7) in normalized variables.

    ``intensity`` already includes dark and optional uniform background:

    ``E_t = E_app*I_b - (E*I - I_x)*(1 + E_x) + I*E_xx``.

    ``background_intensity`` is the normalized ``I_b`` that enters the
    applied-field source term; it is explicit rather than inferred from a
    spatial intensity minimum.
    """

    if E.shape != intensity.shape:
        raise ValueError("E and intensity must have identical shapes")
    if E.ndim not in (2, 3):
        raise ValueError("E and intensity must have shape (Nx, Ny) or (Nz, Nx, Ny)")
    background = float(background_intensity)
    if background < 0.0:
        raise ValueError("background_intensity must be nonnegative")

    E_x, E_xx = periodic_derivatives_x(
        E,
        dx_normalized=dx_normalized,
        xp=xp,
    )
    I_x, _ = periodic_derivatives_x(
        intensity,
        dx_normalized=dx_normalized,
        xp=xp,
    )
    return (
        float(applied_field) * background
        - (E * intensity - I_x) * (1.0 + E_x)
        + E_xx * intensity
    )


def diffusion_implicit_split(
    E,
    intensity,
    *,
    applied_field: float,
    background_intensity: float,
    dx_normalized: float,
    xp: Any,
):
    """Return ``(I*D2(E), R(E; I) - I*D2(E))``.

    The two returned arrays sum to the authoritative ``hopping_rhs`` exactly
    up to floating-point subtraction.  Only the grid-stiff diffusion term is
    selected for implicit treatment; this function does not define a second
    physical residual.
    """

    residual = hopping_rhs(
        E,
        intensity,
        applied_field=applied_field,
        background_intensity=background_intensity,
        dx_normalized=dx_normalized,
        xp=xp,
    )
    _, E_xx = periodic_derivatives_x(
        E,
        dx_normalized=dx_normalized,
        xp=xp,
    )
    implicit = intensity * E_xx
    return implicit, residual - implicit


def _array_has_true(value, *, xp: Any) -> bool:
    reduced = xp.any(value)
    return bool(reduced.item() if hasattr(reduced, "item") else reduced)


def _validate_diffusion_solve_inputs(
    rhs,
    intensity,
    *,
    alpha: float,
    dx_normalized: float,
    xp: Any,
):
    if rhs.shape != intensity.shape:
        raise ValueError("rhs and intensity must have identical shapes")
    if rhs.ndim not in (2, 3):
        raise ValueError(
            "rhs and intensity must have shape (Nx, Ny) or (Nz, Nx, Ny)"
        )
    if int(rhs.shape[-2]) < 3:
        raise ValueError("periodic diffusion solve requires Nx >= 3")
    if getattr(rhs.dtype, "kind", None) != "f":
        raise TypeError("rhs must have a real floating-point dtype")
    if getattr(intensity.dtype, "kind", None) != "f":
        raise TypeError("intensity must have a real floating-point dtype")
    resolved_alpha = float(alpha)
    if not math.isfinite(resolved_alpha) or resolved_alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    resolved_dx = float(dx_normalized)
    if not math.isfinite(resolved_dx) or resolved_dx <= 0.0:
        raise ValueError("dx_normalized must be finite and positive")
    if _array_has_true(~xp.isfinite(rhs), xp=xp):
        raise ValueError("rhs must contain only finite values")
    if _array_has_true(~xp.isfinite(intensity), xp=xp):
        raise ValueError("intensity must contain only finite values")
    if _array_has_true(intensity < 0.0, xp=xp):
        raise ValueError("intensity must be nonnegative")
    return resolved_alpha, resolved_dx


def solve_periodic_variable_diffusion(
    rhs,
    intensity,
    *,
    alpha: float,
    dx_normalized: float,
    xp: Any,
):
    """Solve ``(1 - alpha*I*D2) u = rhs`` along periodic x.

    ``rhs`` and ``intensity`` have shape ``(Nx, Ny)`` or
    ``(Nz, Nx, Ny)``.  The centered second derivative ``D2`` acts on axis
    ``-2``.  The solve is cyclic tridiagonal along x and batched over every
    z/y line.  Spatially varying nonnegative intensity is retained exactly in
    the discrete row coefficients.
    """

    resolved_alpha, resolved_dx = _validate_diffusion_solve_inputs(
        rhs,
        intensity,
        alpha=alpha,
        dx_normalized=dx_normalized,
        xp=xp,
    )
    if resolved_alpha == 0.0:
        return rhs.copy()

    q = (
        xp.asarray(resolved_alpha, dtype=rhs.dtype)
        * intensity.astype(rhs.dtype, copy=False)
        / xp.asarray(resolved_dx * resolved_dx, dtype=rhs.dtype)
    )
    lower = -q
    upper = -q
    diagonal = 1.0 + 2.0 * q
    if getattr(xp, "__name__", "") == "cupy":
        return solve_cyclic_tridiagonal_rows_gpu(
            lower,
            diagonal,
            upper,
            rhs,
        )
    return solve_cyclic_tridiagonal_rows(
        lower, diagonal, upper, rhs, xp=xp
    )


def semi_implicit_trapezoidal_step(
    E,
    intensity_from_state: Callable[[Any], Any],
    *,
    dt_normalized: float,
    applied_field: float,
    background_intensity: float,
    dx_normalized: float,
    xp: Any,
    cancellation_check: Callable[[], None] | None = None,
):
    """Advance one production second-order PR material step.

    The stiff ``I*D2(E)`` term is treated by a linearly implicit
    predictor/corrector.  ``intensity_from_state`` is evaluated at both the
    accepted state and the predicted state, so it may represent either a
    prescribed intensity or the complete self-consistent optical mapping.

    This primitive is selected by the production PR workflow when the
    ``semi_implicit_trapezoidal`` integrator is requested.  The optional
    ``cancellation_check`` is called only between discardable substages; an
    exception from it prevents this function from returning a candidate
    state and therefore cannot accept a partial material step.
    """

    dt = float(dt_normalized)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_normalized must be finite and positive")
    if not callable(intensity_from_state):
        raise TypeError("intensity_from_state must be callable")

    intensity_n = xp.asarray(intensity_from_state(E), dtype=E.dtype)
    if cancellation_check is not None:
        cancellation_check()
    implicit_n, explicit_n = diffusion_implicit_split(
        E,
        intensity_n,
        applied_field=applied_field,
        background_intensity=background_intensity,
        dx_normalized=dx_normalized,
        xp=xp,
    )
    predictor = solve_periodic_variable_diffusion(
        E + dt * explicit_n,
        intensity_n,
        alpha=dt,
        dx_normalized=dx_normalized,
        xp=xp,
    )
    if cancellation_check is not None:
        cancellation_check()

    intensity_predictor = xp.asarray(
        intensity_from_state(predictor),
        dtype=E.dtype,
    )
    if cancellation_check is not None:
        cancellation_check()
    _, explicit_predictor = diffusion_implicit_split(
        predictor,
        intensity_predictor,
        applied_field=applied_field,
        background_intensity=background_intensity,
        dx_normalized=dx_normalized,
        xp=xp,
    )
    corrected_rhs = E + 0.5 * dt * (
        implicit_n + explicit_n + explicit_predictor
    )
    corrected = solve_periodic_variable_diffusion(
        corrected_rhs,
        intensity_predictor,
        alpha=0.5 * dt,
        dx_normalized=dx_normalized,
        xp=xp,
    )
    if cancellation_check is not None:
        cancellation_check()
    return corrected


def euler_step(
    E,
    intensity,
    *,
    dt_normalized: float,
    applied_field: float,
    background_intensity: float,
    dx_normalized: float,
    xp: Any,
):
    """Advance the physical normalized space-charge state by explicit Euler."""

    dt = float(dt_normalized)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_normalized must be finite and positive")
    return E + dt * hopping_rhs(
        E,
        intensity,
        applied_field=applied_field,
        background_intensity=background_intensity,
        dx_normalized=dx_normalized,
        xp=xp,
    )


def _spectral_kmax_normalized(grid, material: PRMaterialSpec) -> float:
    """Return the Nyquist wavenumber used by paper Equation (15)."""

    material.validate()
    return math.pi / (
        material.characteristic_wavenumber_per_um * float(grid.dx_um)
    )


def paper_conservative_timestep_limit(grid, material: PRMaterialSpec) -> float:
    """Return paper Equation (15), without discretization corrections."""

    return paper_mode_timestep_rule(_spectral_kmax_normalized(grid, material))


def legacy_conservative_timestep_limit(grid, material: PRMaterialSpec) -> float:
    """Return the exact conservative-step expression coded by PRProp3D."""

    material.validate()
    positive_frequency_index = int(grid.Nx) // 2 - 1
    if positive_frequency_index <= 0:
        return math.inf
    frequency_per_um = positive_frequency_index / (
        float(grid.Nx) * float(grid.dx_um)
    )
    kmax_normalized = (
        2.0
        * math.pi
        * frequency_per_um
        / material.characteristic_wavenumber_per_um
    )
    return legacy_mode_timestep_rule(kmax_normalized)


def conservative_timestep_limit(grid, material: PRMaterialSpec) -> float:
    """Return the new solver's conservative linearized Euler guard.

    Every representable centered-difference mode is included. The exact
    Euler absolute-stability boundary is divided by eight, matching the
    paper's choice of one quarter of the decay time constant (Euler itself is
    stable up to twice that time constant). The linearization uses the
    plane-wave total intensity ``1 + I_b`` and its uniform equilibrium field.
    """

    material.validate()
    Nx = int(grid.Nx)
    dx_normalized = (
        material.characteristic_wavenumber_per_um * float(grid.dx_um)
    )
    uniform_intensity = 1.0 + material.background_intensity
    equilibrium_field = (
        float(material.applied_field)
        * material.background_intensity
        / uniform_intensity
    )
    limits = []
    for index in range(Nx):
        signed_index = index if index <= Nx // 2 else index - Nx
        k_normalized = 2.0 * math.pi * signed_index / (Nx * dx_normalized)
        limits.append(
            linearized_euler_mode_limit(
                k_normalized,
                dx_normalized=dx_normalized,
                uniform_intensity=uniform_intensity,
                equilibrium_field=equilibrium_field,
            )
            / 8.0
        )
    limit = min(limits)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("could not determine a positive PR timestep limit")
    return limit


def semi_implicit_conservative_timestep_limit(
    grid,
    material: PRMaterialSpec,
) -> float:
    """Return the quarter-time-style guard for the selected IMEX method."""

    material.validate()
    Nx = int(grid.Nx)
    dx_normalized = (
        material.characteristic_wavenumber_per_um * float(grid.dx_um)
    )
    uniform_intensity = 1.0 + material.background_intensity
    equilibrium_field = (
        float(material.applied_field)
        * material.background_intensity
        / uniform_intensity
    )
    limits = []
    for index in range(Nx):
        signed_index = index if index <= Nx // 2 else index - Nx
        k_normalized = 2.0 * math.pi * signed_index / (Nx * dx_normalized)
        boundary = linearized_semi_implicit_mode_limit(
            k_normalized,
            dx_normalized=dx_normalized,
            uniform_intensity=uniform_intensity,
            equilibrium_field=equilibrium_field,
        )
        limits.append(boundary / 8.0)
    limit = min(limits)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError(
            "could not determine a positive semi-implicit PR timestep limit"
        )
    return limit


def validate_timestep(
    dt_normalized: float,
    grid,
    material: PRMaterialSpec,
    *,
    integrator: str = PR_EULER_INTEGRATOR,
) -> float:
    """Validate and return the selected integrator's conservative guard."""

    dt = float(dt_normalized)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_normalized must be finite and positive")
    if integrator == PR_EULER_INTEGRATOR:
        limit = conservative_timestep_limit(grid, material)
    elif integrator == PR_SEMI_IMPLICIT_INTEGRATOR:
        limit = semi_implicit_conservative_timestep_limit(grid, material)
    else:
        raise ValueError(f"unknown PR integrator: {integrator}")
    if dt > limit:
        raise ValueError(
            f"dt_normalized={dt:g} exceeds conservative PR limit "
            f"{limit:g} for integrator {integrator}"
        )
    return limit


__all__ = [
    "centered_difference_symbols",
    "diffusion_implicit_split",
    "linearized_semi_implicit_mode_limit",
    "linearized_euler_mode_limit",
    "paper_mode_timestep_rule",
    "legacy_mode_timestep_rule",
    "periodic_derivatives_x",
    "semi_implicit_amplification",
    "semi_implicit_conservative_timestep_limit",
    "semi_implicit_trapezoidal_step",
    "solve_periodic_variable_diffusion",
    "hopping_rhs",
    "euler_step",
    "paper_conservative_timestep_limit",
    "legacy_conservative_timestep_limit",
    "conservative_timestep_limit",
    "validate_timestep",
]
