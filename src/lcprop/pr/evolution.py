"""Normalized time-dependent photorefractive hopping-model evolution."""

from __future__ import annotations

import math
from typing import Any

from lcprop.pr.specs import PRMaterialSpec


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


def validate_timestep(dt_normalized: float, grid, material: PRMaterialSpec) -> float:
    """Validate and return the implemented conservative normalized-time guard."""

    dt = float(dt_normalized)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_normalized must be finite and positive")
    limit = conservative_timestep_limit(grid, material)
    if dt > limit:
        raise ValueError(
            f"dt_normalized={dt:g} exceeds conservative PR limit {limit:g}"
        )
    return limit


__all__ = [
    "centered_difference_symbols",
    "linearized_euler_mode_limit",
    "paper_mode_timestep_rule",
    "legacy_mode_timestep_rule",
    "periodic_derivatives_x",
    "hopping_rhs",
    "euler_step",
    "paper_conservative_timestep_limit",
    "legacy_conservative_timestep_limit",
    "conservative_timestep_limit",
    "validate_timestep",
]
