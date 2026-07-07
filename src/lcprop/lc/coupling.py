"""LC/optical coupling coefficients."""

from __future__ import annotations

from scipy.constants import c as C0

from lcprop.core.context import GridSpec, LCMaterial
from lcprop.core.beams import BeamStack


def compute_bi_from_power(
    P_mW: float,
    *,
    d_um: float = 75.0,
    K: float = 7e-12,
    ne: float = 1.7,
    no: float = 1.5,
) -> float:
    d_m = float(d_um) * 1e-6
    P_W = float(P_mW) * 1e-3
    na2 = float(ne) ** 2 - float(no) ** 2
    return float(na2 * d_m**2 * 1e12 * P_W / (8.0 * C0 * float(K)))


def resolved_bi(grid: GridSpec, material: LCMaterial, beams: BeamStack) -> float:
    P_mW = sum(ch.power_mW for ch in beams.channels)

    return compute_bi_from_power(
        P_mW,
        d_um=grid.x_aperture_um,
        K=material.K,
        ne=material.ne,
        no=material.no,
    )
