"""Liquid-crystal optical propagation compatibility wrappers.

Shared optics advances fields through prepared multiplicative responses. This
module owns the LC-specific conversion from director angle to optical response
and preserves the established theta-based advancement API for LC callers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lcprop.lc.optical_response import (
    neff_from_theta as _neff_from_theta,
    phase_screen_from_theta,
)
from lcprop.optics import splitstep

Array = Any


def _xp_from(*arrays: Array, xp: Any | None = None):
    if xp is not None:
        return xp
    for array in arrays:
        if type(array).__module__.split(".")[0] == "cupy":
            import cupy as cp  # type: ignore

            return cp
    return np


def neff_from_theta(
    theta: Array,
    *,
    ne: float,
    no: float,
    xp: Any | None = None,
) -> Array:
    """Return the LC extraordinary-ray effective index."""

    xp = _xp_from(theta, xp=xp)
    return _neff_from_theta(theta, ne=ne, no=no, xp=xp)


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
    """Return the LC optical-response phase screen for one distance."""

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
    """Advance a channel stack through one LC director slice."""

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
    return splitstep.advance_prepared_response(
        A,
        kernel=kernel,
        half_step_response=half_phase,
        Nsub=Nsub,
        xp=xp,
    )


__all__ = ["advance_slice", "neff_from_theta", "nonlinear_phase"]
