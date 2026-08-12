"""Liquid-crystal coordinate and legacy time normalization.

The shared runtime grid describes the physical numerical laboratory.  This
module derives the nondimensional coordinates and spacings used specifically
by the LC director equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lcprop.core.context import GridSpec
from lcprop.core.grid import RuntimeGrid


@dataclass(frozen=True)
class LCSpatialNormalization:
    """LC director coordinates and trusted discrete solver spacings.

    ``u`` and ``v`` are the sampled coordinates ``2*x/d`` and ``2*y/d``,
    where ``d`` is the LC cell thickness (the physical x aperture).  ``du``
    and ``dv`` deliberately preserve LCProp's established theta-solver
    discretization: ``du = 2/(Nx - 1)`` and
    ``dv = du*(dy_um/dx_um)``.  Because the shared physical grid is
    cell-centered, these trusted solver spacings are not inferred by taking
    differences of the sampled coordinate arrays.
    """

    u: Any
    v: Any
    du: float
    dv: float

    def summary(self) -> dict[str, float]:
        """Return the LC-normalized spacings used by director algorithms."""

        return {"du": float(self.du), "dv": float(self.dv)}


def make_lc_spatial_normalization(grid: RuntimeGrid) -> LCSpatialNormalization:
    """Derive LC-normalized coordinates from a shared physical runtime grid."""

    half_thickness_um = max(
        1.0e-300,
        float(grid.spec.x_aperture_um) / 2.0,
    )
    du = 2.0 / (int(grid.Nx) - 1)
    dv = du * (float(grid.dy_um) / float(grid.dx_um))
    return LCSpatialNormalization(
        u=grid.x_um / half_thickness_um,
        v=grid.y_um / half_thickness_um,
        du=du,
        dv=dv,
    )


def lc_grid_summary(
    grid: RuntimeGrid,
    normalization: LCSpatialNormalization | None = None,
) -> dict[str, float | int | str]:
    """Return physical grid provenance augmented by LC solver spacings."""

    resolved = normalization or make_lc_spatial_normalization(grid)
    return {**grid.summary(), **resolved.summary()}


def from_cell(
    *,
    Nx: int,
    Ny: int,
    dz_um: float,
    thickness_um: float,
    y_aperture_um: float,
    interaction_length_um: float,
) -> GridSpec:
    """Construct a physical grid specification from LC-cell geometry."""

    return GridSpec(
        Nx=Nx,
        Ny=Ny,
        dz_um=dz_um,
        x_aperture_um=thickness_um,
        y_aperture_um=y_aperture_um,
        z_length_um=interaction_length_um,
    )


@dataclass(frozen=True)
class TimeSpec:
    """Legacy LC normalized material-time discretization choices.

    Active LC workflows use ``TimeDependentSolverOptions``.  This retained
    value object preserves the historical ``lcprop.core.context.TimeSpec``
    import while making its LC ownership explicit.
    """

    dt: float = 7.5e-4
    Nt: int = 20

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.Nt < 0:
            raise ValueError("Nt must be nonnegative")


__all__ = [
    "LCSpatialNormalization",
    "TimeSpec",
    "from_cell",
    "lc_grid_summary",
    "make_lc_spatial_normalization",
]
