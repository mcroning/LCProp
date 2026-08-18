"""Permanent diagnostics for the frozen full-transverse profile."""

from __future__ import annotations

import numpy as np

from lcprop.pr.transverse.transport import (
    PRTransverseState,
    spectral_derivatives,
    spectral_wavevectors,
)


def electrostatic_residuals(
    state: PRTransverseState,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return discrete curl and Gauss residual volumes."""

    kx, ky = spectral_wavevectors(
        state.psi.shape[-2:],
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
    )
    E_y_x, _ = spectral_derivatives(state.E_y, kx=kx, ky=ky)
    _, E_x_y = spectral_derivatives(state.E_x, kx=kx, ky=ky)
    E_x_x, _ = spectral_derivatives(state.E_x, kx=kx, ky=ky)
    _, E_y_y = spectral_derivatives(state.E_y, kx=kx, ky=ky)
    curl = E_y_x - E_x_y
    gauss = (
        state.carrier_density
        - 1.0
        - E_x_x
        - float(h_y) * E_y_y
    )
    return curl, gauss


def state_diagnostics(
    state: PRTransverseState,
    *,
    dx_normalized: float,
    dy_normalized: float,
    h_y: float = 1.0,
) -> dict[str, object]:
    curl, gauss = electrostatic_residuals(
        state,
        dx_normalized=dx_normalized,
        dy_normalized=dy_normalized,
        h_y=h_y,
    )
    carrier_integrals = (
        np.sum(state.carrier_density, axis=(-2, -1))
        * float(dx_normalized)
        * float(dy_normalized)
    )

    def metrics(value):
        array = np.asarray(value, dtype=np.float64)
        return float(np.sqrt(np.mean(array * array))), float(np.max(np.abs(array)))

    curl_rms, curl_max = metrics(curl)
    gauss_rms, gauss_max = metrics(gauss)
    ex_rms, ex_max = metrics(state.E_x)
    ey_rms, ey_max = metrics(state.E_y)
    carrier_rms, carrier_max = metrics(state.carrier_density - 1.0)
    return {
        "carrier_integrals_per_z": np.asarray(carrier_integrals).copy(),
        "curl_rms": curl_rms,
        "curl_max": curl_max,
        "gauss_rms": gauss_rms,
        "gauss_max": gauss_max,
        "E_x_rms": ex_rms,
        "E_x_max_abs": ex_max,
        "E_y_rms": ey_rms,
        "E_y_max_abs": ey_max,
        "P_minus_one_rms": carrier_rms,
        "P_minus_one_max_abs": carrier_max,
        "potential_mean_max_abs": float(
            np.max(np.abs(np.mean(state.psi, axis=(-2, -1))))
        ),
        "finite_material_state": bool(
            all(
                np.all(np.isfinite(value))
                for value in (
                    state.psi,
                    state.carrier_density,
                    state.E_x,
                    state.E_y,
                )
            )
        ),
    }


__all__ = ["electrostatic_residuals", "state_diagnostics"]
