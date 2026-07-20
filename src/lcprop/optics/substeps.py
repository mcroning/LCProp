"""Shared optical-substep selection and diffraction-kernel construction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

from lcprop.optics.splitstep import linear_kernel


@dataclass(frozen=True)
class OpticalSubstepPlan:
    """Resolved common substep policy for one nominal longitudinal interval."""

    Nsub: int
    dz_sub_um: float
    phi_est_rad: float
    cap_reached: bool
    shortest_wavelength_um: float

    def diagnostics(self) -> dict[str, int | float | bool]:
        return {
            "optical_Nsub": int(self.Nsub),
            "optical_dz_sub_um": float(self.dz_sub_um),
            "optical_phi_est_rad": float(self.phi_est_rad),
            "optical_substep_cap_reached": bool(self.cap_reached),
        }


def resolve_optical_substeps(
    *,
    dz_um: float,
    wavelengths_um: Iterable[float],
    enabled: bool,
    dn_max_est: float,
    max_phase_per_substep_rad: float,
    max_substeps: int,
) -> OpticalSubstepPlan:
    """Resolve a common Nsub using the shortest active channel wavelength."""

    dz_um = float(dz_um)
    wavelengths = tuple(float(value) for value in wavelengths_um)
    dn_max_est = float(dn_max_est)
    max_phase = float(max_phase_per_substep_rad)
    max_substeps = int(max_substeps)

    if dz_um <= 0.0:
        raise ValueError("dz_um must be > 0")
    if not wavelengths or any(value <= 0.0 for value in wavelengths):
        raise ValueError("wavelengths_um must contain positive values")
    if dn_max_est < 0.0:
        raise ValueError("dn_max_est must be >= 0")
    if max_phase <= 0.0:
        raise ValueError("max_phase_per_substep_rad must be > 0")
    if max_substeps < 1:
        raise ValueError("max_substeps must be >= 1")

    shortest_wavelength_um = min(wavelengths)
    phi_est_rad = (
        (2.0 * math.pi / shortest_wavelength_um) * dz_um * dn_max_est
    )
    requested_substeps = max(1, int(math.ceil(phi_est_rad / max_phase)))
    if enabled:
        Nsub = min(max_substeps, requested_substeps)
        cap_reached = requested_substeps >= max_substeps
    else:
        Nsub = 1
        cap_reached = False

    return OpticalSubstepPlan(
        Nsub=Nsub,
        dz_sub_um=dz_um / Nsub,
        phi_est_rad=phi_est_rad,
        cap_reached=cap_reached,
        shortest_wavelength_um=shortest_wavelength_um,
    )


def build_optical_substep_kernel(
    fxy2,
    *,
    dz_um: float,
    propagation_wavelength_um: float,
    active_wavelengths_um: Iterable[float],
    n_ref: float,
    enabled: bool,
    dn_max_est: float,
    max_phase_per_substep_rad: float,
    max_substeps: int,
    xp: Any | None = None,
):
    """Return a resolved plan and a matching dz/Nsub diffraction kernel."""

    plan = resolve_optical_substeps(
        dz_um=dz_um,
        wavelengths_um=active_wavelengths_um,
        enabled=enabled,
        dn_max_est=dn_max_est,
        max_phase_per_substep_rad=max_phase_per_substep_rad,
        max_substeps=max_substeps,
    )
    kernel = linear_kernel(
        fxy2,
        dz=plan.dz_sub_um,
        wavelength=propagation_wavelength_um,
        n_ref=n_ref,
        xp=xp,
    )
    return plan, kernel


__all__ = [
    "OpticalSubstepPlan",
    "build_optical_substep_kernel",
    "resolve_optical_substeps",
]
