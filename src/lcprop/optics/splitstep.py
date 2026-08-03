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

The split-step ordering is symmetric for one optical substep:

    for substep:
        apply half the nonlinear LC phase from theta
        apply linear Fourier hop
        apply half the nonlinear LC phase from theta

The caller decides how kernels, substeps, wavelengths, and reference indices
are constructed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lcprop.core.beams import normalize_coherence_groups
from lcprop.core.derived import compute_neff

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

    xp = _xp_from(theta, xp=xp)
    return compute_neff(theta, ne=ne, no=no, xp=xp)


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
    """Return LC nonlinear phase factor for one propagation step."""

    xp = _xp_from(theta, xp=xp)
    k0 = 2.0 * np.pi / float(wavelength)
    dn = neff_from_theta(theta, ne=ne, no=no, xp=xp) - float(n_ref)
    return xp.exp(1j * k0 * float(dz) * dn)


def apply_nonlinear_phase_inplace(A: Array, phase: Array, *, xp: Any | None = None) -> Array:
    """Multiply all channels by a common transverse phase."""

    xp = _xp_from(A, phase, xp=xp)
    if A.ndim != 3:
        raise ValueError("A must have shape (Nch, Nx, Ny)")
    A[...] = A * phase[None, :, :]
    return A


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
    for _ in range(int(Nsub)):
        apply_nonlinear_phase_inplace(A, half_phase, xp=xp)
        hop_linear_inplace(A, kernel, xp=xp)
        apply_nonlinear_phase_inplace(A, half_phase, xp=xp)

    return A


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
    "apply_nonlinear_phase_inplace",
    "advance_slice",
    "advance_slice_with_midintensity",
]
