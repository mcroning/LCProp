"""Photorefractive material-state to scalar optical-response mappings."""

from __future__ import annotations

import math
from typing import Any


def delta_n_from_E(
    E,
    *,
    gain_length_product: float,
    interaction_length_um: float,
    wavelength_um: float,
):
    """Return the PR index change used by trusted PRProp3D ``dn_calc``."""

    length = float(interaction_length_um)
    wavelength = float(wavelength_um)
    if length <= 0.0 or wavelength <= 0.0:
        raise ValueError("interaction length and wavelength must be positive")
    k_vacuum = 2.0 * math.pi / wavelength
    return -E * (2.0 * float(gain_length_product)) / (length * k_vacuum)


def half_step_response_from_E(
    E,
    *,
    dz_substep_um: float,
    wavelength_um: float,
    interaction_length_um: float,
    gain_length_product: float,
    xp: Any,
):
    """Return the complex PR screen for half of one optical substep."""

    dz_substep = float(dz_substep_um)
    wavelength = float(wavelength_um)
    if dz_substep <= 0.0:
        raise ValueError("dz_substep_um must be positive")
    delta_n = delta_n_from_E(
        E,
        gain_length_product=gain_length_product,
        interaction_length_um=interaction_length_um,
        wavelength_um=wavelength,
    )
    k_vacuum = 2.0 * math.pi / wavelength
    return xp.exp(0.5j * k_vacuum * dz_substep * delta_n)


__all__ = ["delta_n_from_E", "half_step_response_from_E"]
