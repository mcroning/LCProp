"""Multichannel split-step optical propagation kernels.

This module is a numerical black box. It consumes channel-stack fields,
theta slices, kernels, and optical coefficients. It knows nothing about beam
setup, LC-cell experiments, GUI state, files, continuation, or diagnostics.

Field convention
----------------
Optical fields are always channel stacks:

    A.shape == (Nch, Nx, Ny)

A single beam is represented by ``Nch=1``. There is no scalar-field special
case inside this module.

The material-neutral split-step ordering is symmetric for one optical substep:

    for substep:
        apply a prepared half-step optical response
        apply linear Fourier hop
        apply the prepared half-step optical response

The prepared response is a multiplicative complex screen. It may be shared by
all channels with shape ``(Nx, Ny)`` or supplied per channel with shape
``(Nch, Nx, Ny)``. The caller decides how responses, kernels, substeps,
wavelengths, and reference indices are constructed.

The theta-based functions remain LC compatibility wrappers around this
material-neutral advancement surface.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lcprop.core.beams import normalize_coherence_groups

Array = Any


def _xp_from(*arrays: Array, xp: Any | None = None):
    """Return NumPy/CuPy-like array module, preferring explicit ``xp``."""
    if xp is not None:
        return xp
    for a in arrays:
        if type(a).__module__.split(".")[0] == "cupy":
            import cupy as cp  # type: ignore

            return cp
    return np


def as_channel_stack(A: Array, *, xp: Any | None = None) -> Array:
    """Return ``A`` as a channel stack.

    This helper is intended for adapters/tests. Core algorithms should already
    pass channel stacks.
    """

    xp = _xp_from(A, xp=xp)
    if A.ndim == 3:
        return A
    if A.ndim == 2:
        return A[None, :, :]
    raise ValueError("A must have shape (Nch, Nx, Ny) or (Nx, Ny)")


def channel_intensities(A: Array, *, xp: Any | None = None) -> Array:
    """Return per-channel intensities ``|A_c|^2``."""

    xp = _xp_from(A, xp=xp)
    if A.ndim != 3:
        raise ValueError("A must have shape (Nch, Nx, Ny)")
    return xp.abs(A) ** 2


def total_intensity(
    A: Array,
    *,
    coherent: bool = False,
    coherence_groups: tuple[str, ...] | list[str] | None = None,
    xp: Any | None = None,
) -> Array:
    """Return total intensity from a channel stack.

    Parameters
    ----------
    coherent
        If False, return ``sum_c |A_c|^2``. If True, return
        ``|sum_c A_c|^2``. Retained for backward compatibility when
        ``coherence_groups`` is not supplied.
    coherence_groups
        Per-channel group names. Channels in each group are summed as fields;
        the resulting group intensities are then summed.
    """

    xp = _xp_from(A, xp=xp)
    if A.ndim != 3:
        raise ValueError("A must have shape (Nch, Nx, Ny)")

    groups = _coherence_group_indices(
        A.shape[0],
        coherent=coherent,
        coherence_groups=coherence_groups,
    )
    intensity = xp.zeros_like(xp.abs(A[0]) ** 2)
    for _, group_field in _coherent_group_fields(A, groups, xp=xp):
        intensity = intensity + xp.abs(group_field) ** 2
    return intensity


def weighted_theta_intensity(
    A: Array,
    weights: Array | None = None,
    *,
    coherent: bool = False,
    coherence_groups: tuple[str, ...] | list[str] | None = None,
    xp: Any | None = None,
) -> Array:
    """Return effective plain intensity for theta algorithms.

    The theta algorithms expect plain intensity and multiply by ``bi``
    internally. ``weights`` are relative optical-coupling multipliers, such as
    ``bi_c / bi_ref``; they weight intensity, not field amplitude. Coherently
    interfering channels must therefore have equal weights within their group.
    """

    xp = _xp_from(A, weights, xp=xp)

    if weights is None:
        return total_intensity(
            A,
            coherent=coherent,
            coherence_groups=coherence_groups,
            xp=xp,
        )

    weights = xp.asarray(weights)
    if weights.ndim != 1 or weights.shape[0] != A.shape[0]:
        raise ValueError("weights must have shape (Nch,)")

    groups = _coherence_group_indices(
        A.shape[0],
        coherent=coherent,
        coherence_groups=coherence_groups,
    )
    intensity = xp.zeros_like(xp.abs(A[0]) ** 2)
    for group_name, group_field in _coherent_group_fields(A, groups, xp=xp):
        indices = groups[group_name]
        group_weights = weights[list(indices)]
        equal = xp.all(group_weights == group_weights[0])
        if not bool(equal.item() if hasattr(equal, "item") else equal):
            raise ValueError(
                f"theta_weights must be equal within coherent group {group_name!r}; "
                f"got {group_weights.tolist()}"
            )
        intensity = intensity + group_weights[0] * xp.abs(group_field) ** 2
    return intensity


def _coherence_group_indices(
    n_channels: int,
    *,
    coherent: bool,
    coherence_groups: tuple[str, ...] | list[str] | None,
) -> dict[str, tuple[int, ...]]:
    """Map normalized coherence names to their channel indices."""

    names = normalize_coherence_groups(
        n_channels,
        coherent=coherent,
        coherence_groups=coherence_groups,
    )

    grouped: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        grouped.setdefault(name, []).append(index)
    return {name: tuple(indices) for name, indices in grouped.items()}


def _coherent_group_fields(A: Array, groups: dict[str, tuple[int, ...]], *, xp: Any):
    """Yield each group name and its coherently summed field."""

    for group_name, indices in groups.items():
        yield group_name, xp.sum(A[list(indices)], axis=0)


def neff_from_theta(theta: Array, *, ne: float, no: float, xp: Any | None = None) -> Array:
    """Extraordinary-ray effective index from director angle theta.

    Compatibility wrapper around lcprop.core.derived.compute_neff.
    """

    from lcprop.lc.optical_response import neff_from_theta as lc_neff_from_theta

    xp = _xp_from(theta, xp=xp)
    return lc_neff_from_theta(theta, ne=ne, no=no, xp=xp)


def linear_kernel(
    fxy2: Array,
    *,
    dz: float,
    wavelength: float,
    n_ref: float,
    xp: Any | None = None,
) -> Array:
    """Return paraxial Fourier-space linear propagator.

    ``fxy2`` has shape ``(Nx, Ny)`` and stores ``fx^2 + fy^2`` in micron units.
    """

    xp = _xp_from(fxy2, xp=xp)
    return xp.exp(-1j * np.pi * float(dz) * float(wavelength) * fxy2 / float(n_ref))


def hop_linear(A: Array, kernel: Array, *, xp: Any | None = None) -> Array:
    """Apply one linear Fourier hop to every channel."""

    xp = _xp_from(A, kernel, xp=xp)
    if A.ndim != 3:
        raise ValueError("A must have shape (Nch, Nx, Ny)")
    return xp.fft.ifft2(
        xp.fft.fft2(A, axes=(-2, -1)) * kernel[None, :, :],
        axes=(-2, -1),
    )


def hop_linear_inplace(A: Array, kernel: Array, *, xp: Any | None = None) -> Array:
    """Apply one linear Fourier hop in place and return ``A``."""

    A[...] = hop_linear(A, kernel, xp=xp)
    return A


def nonlinear_phase(
    theta: Array,
    *,
    dz: float,
    wavelength: float,
    n_ref: float,
    ne: float,
    no: float,
    xp: Any | None = None,
) -> Array:
    """Compatibility wrapper for the LC optical-response phase screen."""

    from lcprop.lc.optical_response import phase_screen_from_theta

    xp = _xp_from(theta, xp=xp)
    return phase_screen_from_theta(
        theta,
        dz=dz,
        wavelength=wavelength,
        n_ref=n_ref,
        ne=ne,
        no=no,
        xp=xp,
    )


def apply_response_screen_inplace(
    A: Array,
    response_screen: Array,
    *,
    xp: Any | None = None,
) -> Array:
    """Apply a prepared shared or per-channel optical response in place.

    ``response_screen`` is a multiplicative complex screen for the optical
    field. A two-dimensional screen is shared by all channels. A
    three-dimensional screen must match the complete channel-stack shape.
    """

    xp = _xp_from(A, response_screen, xp=xp)
    if A.ndim != 3:
        raise ValueError("A must have shape (Nch, Nx, Ny)")

    response_screen = xp.asarray(response_screen)
    if response_screen.ndim == 2:
        if response_screen.shape != A.shape[1:]:
            raise ValueError(
                "shared response_screen must have shape (Nx, Ny) matching A"
            )
        multiplier = response_screen[None, :, :]
    elif response_screen.ndim == 3:
        if response_screen.shape != A.shape:
            raise ValueError(
                "per-channel response_screen must have shape (Nch, Nx, Ny) "
                "matching A"
            )
        multiplier = response_screen
    else:
        raise ValueError(
            "response_screen must have shape (Nx, Ny) or (Nch, Nx, Ny)"
        )

    A[...] = A * multiplier
    return A


def advance_prepared_response(
    A: Array,
    *,
    kernel: Array,
    half_step_response: Array,
    Nsub: int = 1,
    xp: Any | None = None,
) -> Array:
    """Advance a channel stack using a prepared material response.

    ``kernel`` is the full linear hop for one optical substep.
    ``half_step_response`` is the multiplicative material-response screen for
    one half substep. It may have shape ``(Nx, Ny)`` for a response shared by
    all channels or ``(Nch, Nx, Ny)`` for per-channel responses.

    The field is updated in place using symmetric Strang ordering and returned.
    """

    xp = _xp_from(A, kernel, half_step_response, xp=xp)
    if A.ndim != 3:
        raise ValueError("A must have shape (Nch, Nx, Ny)")
    if kernel.ndim != 2 or kernel.shape != A.shape[1:]:
        raise ValueError("kernel must have shape (Nx, Ny) matching A")
    if int(Nsub) < 1:
        raise ValueError("Nsub must be >= 1")

    # Validate the response before mutating A, including when Nsub is one.
    response = xp.asarray(half_step_response)
    if response.ndim == 2:
        if response.shape != A.shape[1:]:
            raise ValueError(
                "shared response_screen must have shape (Nx, Ny) matching A"
            )
    elif response.ndim == 3:
        if response.shape != A.shape:
            raise ValueError(
                "per-channel response_screen must have shape (Nch, Nx, Ny) "
                "matching A"
            )
    else:
        raise ValueError(
            "response_screen must have shape (Nx, Ny) or (Nch, Nx, Ny)"
        )

    for _ in range(int(Nsub)):
        apply_response_screen_inplace(A, response, xp=xp)
        hop_linear_inplace(A, kernel, xp=xp)
        apply_response_screen_inplace(A, response, xp=xp)

    return A


def apply_nonlinear_phase_inplace(A: Array, phase: Array, *, xp: Any | None = None) -> Array:
    """Compatibility wrapper applying a shared or per-channel phase screen."""

    return apply_response_screen_inplace(A, phase, xp=xp)


def advance_slice(
    A: Array,
    theta: Array,
    *,
    kernel: Array,
    dz: float,
    wavelength: float,
    n_ref: float,
    ne: float,
    no: float,
    Nsub: int = 1,
    xp: Any | None = None,
) -> Array:
    """Advance a channel stack through one z slice.

    The field ``A`` is updated in place and also returned.
    """

    xp = _xp_from(A, theta, kernel, xp=xp)
    if int(Nsub) < 1:
        raise ValueError("Nsub must be >= 1")

    dz_sub = float(dz) / int(Nsub)

    half_phase = nonlinear_phase(
        theta,
        dz=0.5 * dz_sub,
        wavelength=wavelength,
        n_ref=n_ref,
        ne=ne,
        no=no,
        xp=xp,
    )
    return advance_prepared_response(
        A,
        kernel=kernel,
        half_step_response=half_phase,
        Nsub=Nsub,
        xp=xp,
    )


def advance_slice_with_midintensity(
    A: Array,
    theta: Array,
    *,
    kernel: Array,
    dz: float,
    wavelength: float,
    n_ref: float,
    ne: float,
    no: float,
    Nsub: int = 1,
    coherent: bool = False,
    coherence_groups: tuple[str, ...] | list[str] | None = None,
    theta_weights: Array | None = None,
    xp: Any | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Advance one z slice and return midpoint intensity.

    Returns
    -------
    A
        Updated channel stack.
    I_before
        Effective plain theta intensity before propagation.
    I_after
        Effective plain theta intensity after propagation.
    I_mid
        ``0.5 * (I_before + I_after)``.
    """

    xp = _xp_from(A, theta, kernel, theta_weights, xp=xp)

    I_before = weighted_theta_intensity(
        A,
        theta_weights,
        coherent=coherent,
        coherence_groups=coherence_groups,
        xp=xp,
    )

    advance_slice(
        A,
        theta,
        kernel=kernel,
        dz=dz,
        wavelength=wavelength,
        n_ref=n_ref,
        ne=ne,
        no=no,
        Nsub=Nsub,
        xp=xp,
    )

    I_after = weighted_theta_intensity(
        A,
        theta_weights,
        coherent=coherent,
        coherence_groups=coherence_groups,
        xp=xp,
    )

    I_mid = 0.5 * (I_before + I_after)
    return A, I_before, I_after, I_mid


__all__ = [
    "as_channel_stack",
    "channel_intensities",
    "total_intensity",
    "weighted_theta_intensity",
    "neff_from_theta",
    "linear_kernel",
    "hop_linear",
    "hop_linear_inplace",
    "nonlinear_phase",
    "apply_response_screen_inplace",
    "advance_prepared_response",
    "apply_nonlinear_phase_inplace",
    "advance_slice",
    "advance_slice_with_midintensity",
]
