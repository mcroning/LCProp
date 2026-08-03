"""Liquid-crystal material state to optical-response mappings."""

from __future__ import annotations

from typing import Any

import numpy as np

from lcprop.core.derived import compute_neff


def neff_from_theta(theta, *, ne: float, no: float, xp: Any | None = None):
    """Return the extraordinary-ray effective index for an LC director field."""

    if xp is None:
        xp = np
    return compute_neff(theta, ne=ne, no=no, xp=xp)


def phase_screen_from_theta(
    theta,
    *,
    dz: float,
    wavelength: float,
    n_ref: float,
    ne: float,
    no: float,
    xp: Any | None = None,
):
    """Return the LC multiplicative optical screen over distance ``dz``."""

    if xp is None:
        xp = np
    k0 = 2.0 * np.pi / float(wavelength)
    dn = neff_from_theta(theta, ne=ne, no=no, xp=xp) - float(n_ref)
    return xp.exp(1j * k0 * float(dz) * dn)


__all__ = ["neff_from_theta", "phase_screen_from_theta"]
