"""Numerical grid specifications and builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lcprop.core.context import GridSpec

Array = Any


@dataclass(frozen=True)
class RuntimeGrid:
    """Backend arrays and derived spacings for algorithms/builders."""

    spec: GridSpec
    xp: Any
    real_dtype: Any

    Nx: int
    Ny: int
    Nz: int

    dx_um: float
    dy_um: float
    dz_um: float

    du: float
    dv: float

    x_um: Array
    y_um: Array
    fx_um: Array
    fy_um: Array
    fxy2_um: Array

    def summary(self) -> dict[str, float | int | str]:
        return {
            "Nx": self.Nx,
            "Ny": self.Ny,
            "Nz": self.Nz,
            "dx_um": float(self.dx_um),
            "dy_um": float(self.dy_um),
            "dz_um": float(self.dz_um),
            "du": float(self.du),
            "dv": float(self.dv),
            "x_aperture_um": float(self.spec.x_aperture_um),
            "y_aperture_um": float(self.spec.y_aperture_um),
            "z_length_um": float(self.spec.z_length_um),
            "real_dtype": str(np.dtype(self.real_dtype)),
        }


def round_nz(z_length_um: float, dz_um: float) -> int:
    """Return Nz from z length and dz, requiring at least one slice."""
    return max(1, int(round(float(z_length_um) / float(dz_um))))


def make_grid(
    spec: GridSpec,
    *,
    xp: Any = np,
    real_dtype: Any = np.float32,
) -> RuntimeGrid:
    """Build a runtime grid from GridSpec."""

    spec.validate()

    Nx = int(spec.Nx)
    Ny = int(spec.Ny)
    Nz = round_nz(spec.z_length_um, spec.dz_um)

    dx_um = float(spec.x_aperture_um) / Nx
    dy_um = float(spec.y_aperture_um) / Ny
    dz_um = float(spec.dz_um)

    x_um = (
        (xp.arange(Nx, dtype=real_dtype) - Nx / 2) * dx_um + 0.5 * dx_um
    ).astype(real_dtype, copy=False)

    y_um = (
        (xp.arange(Ny, dtype=real_dtype) - Ny / 2) * dy_um + 0.5 * dy_um
    ).astype(real_dtype, copy=False)

    fx_um = xp.fft.fftfreq(Nx, d=dx_um).astype(real_dtype, copy=False)
    fy_um = xp.fft.fftfreq(Ny, d=dy_um).astype(real_dtype, copy=False)
    fxy2_um = (fx_um[:, None] ** 2 + fy_um[None, :] ** 2).astype(
        real_dtype,
        copy=False,
    )

    # Trusted LC theta coordinates: u = 2x/d, with d = x aperture.
    du = 2.0 / (Nx - 1)
    dv = du * (dy_um / dx_um)

    return RuntimeGrid(
        spec=spec,
        xp=xp,
        real_dtype=real_dtype,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        dx_um=dx_um,
        dy_um=dy_um,
        dz_um=dz_um,
        du=du,
        dv=dv,
        x_um=x_um,
        y_um=y_um,
        fx_um=fx_um,
        fy_um=fy_um,
        fxy2_um=fxy2_um,
    )


def from_cell(
    *,
    Nx: int,
    Ny: int,
    dz_um: float,
    thickness_um: float,
    y_aperture_um: float,
    interaction_length_um: float,
) -> GridSpec:
    """Convenience constructor from LC-cell geometry."""
    return GridSpec(
        Nx=Nx,
        Ny=Ny,
        dz_um=dz_um,
        x_aperture_um=thickness_um,
        y_aperture_um=y_aperture_um,
        z_length_um=interaction_length_um,
    )
