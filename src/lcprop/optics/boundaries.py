"""Material-neutral transverse optical boundary policies."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal


TransverseBoundaryMode = Literal["periodic", "sponge", "tukey"]


@dataclass(frozen=True)
class TransverseBoundarySpec:
    """Amplitude boundary applied after each complete propagation increment.

    ``periodic`` is an exact no-op. ``sponge`` defines a continuous field-
    amplitude absorption rate over the outer ``width_fraction`` of each
    transverse half-aperture. ``tukey`` supplies a discrete separable
    square-root Tukey apodization window and is not interpreted as an
    absorption rate.
    """

    mode: TransverseBoundaryMode = "periodic"
    width_fraction: float = 0.15
    attenuation_per_um: float = 0.05
    profile_order: int = 2
    tukey_alpha: float = 0.1

    def validate(self) -> None:
        if self.mode not in ("periodic", "sponge", "tukey"):
            raise ValueError("boundary mode must be periodic, sponge, or tukey")
        if not math.isfinite(float(self.width_fraction)) or not 0.0 < float(
            self.width_fraction
        ) <= 1.0:
            raise ValueError("width_fraction must be finite and in (0, 1]")
        if not math.isfinite(float(self.attenuation_per_um)) or float(
            self.attenuation_per_um
        ) <= 0.0:
            raise ValueError("attenuation_per_um must be finite and positive")
        if int(self.profile_order) != self.profile_order or int(self.profile_order) < 1:
            raise ValueError("profile_order must be a positive integer")
        if not math.isfinite(float(self.tukey_alpha)) or not 0.0 <= float(
            self.tukey_alpha
        ) <= 1.0:
            raise ValueError("tukey_alpha must be finite and in [0, 1]")


def _edge_fraction(coordinate, *, aperture_um: float, width_fraction: float, xp: Any):
    half = 0.5 * float(aperture_um)
    width = float(width_fraction) * half
    start = half - width
    return xp.clip((xp.abs(coordinate) - start) / width, 0.0, 1.0)


def _tukey_axis(size: int, alpha: float, *, xp: Any):
    if alpha <= 0.0:
        return xp.ones(size)
    if alpha >= 1.0:
        index = xp.arange(size)
        return 0.5 * (1.0 - xp.cos(2.0 * math.pi * index / (size - 1)))
    x = xp.linspace(0.0, 1.0, size)
    result = xp.ones(size)
    left = x < alpha / 2.0
    right = x >= 1.0 - alpha / 2.0
    result[left] = 0.5 * (
        1.0 + xp.cos(math.pi * (2.0 * x[left] / alpha - 1.0))
    )
    result[right] = 0.5 * (
        1.0
        + xp.cos(math.pi * (2.0 * x[right] / alpha - 2.0 / alpha + 1.0))
    )
    return result


def transverse_boundary_mask(
    grid,
    spec: TransverseBoundarySpec,
    *,
    propagation_distance_um: float | None = None,
):
    """Return a backend-native amplitude mask, or ``None`` for periodic.

    A sponge mask is ``exp(-gamma(x,y) * abs(dz))`` and therefore requires the
    distance represented by one application. Its product over substeps is
    invariant to subdivision of a fixed total distance. A Tukey mask is a
    one-time apodization and does not accept distance semantics.
    """

    spec.validate()
    if spec.mode == "periodic":
        return None
    xp = grid.xp
    if spec.mode == "sponge":
        if propagation_distance_um is None or not math.isfinite(
            float(propagation_distance_um)
        ) or float(propagation_distance_um) == 0.0:
            raise ValueError(
                "sponge boundary requires a finite nonzero propagation distance"
            )
        sx = _edge_fraction(
            grid.x_um,
            aperture_um=grid.spec.x_aperture_um,
            width_fraction=spec.width_fraction,
            xp=xp,
        )
        sy = _edge_fraction(
            grid.y_um,
            aperture_um=grid.spec.y_aperture_um,
            width_fraction=spec.width_fraction,
            xp=xp,
        )
        order = int(spec.profile_order)
        rate = float(spec.attenuation_per_um) * (
            sx[:, None] ** order + sy[None, :] ** order
        )
        return xp.exp(-rate * abs(float(propagation_distance_um)))
    if propagation_distance_um is not None:
        raise ValueError("Tukey boundary is a discrete window, not a z-rate")
    wx = _tukey_axis(grid.Nx, float(spec.tukey_alpha), xp=xp)
    wy = _tukey_axis(grid.Ny, float(spec.tukey_alpha), xp=xp)
    return xp.sqrt(wx[:, None] * wy[None, :])


def apply_transverse_boundary_inplace(field, mask):
    """Apply one prepared shared transverse amplitude mask in place."""

    if mask is None:
        return field
    if field.ndim not in (2, 3) or mask.ndim != 2 or field.shape[-2:] != mask.shape:
        raise ValueError("boundary mask must match the field transverse shape")
    field[...] = field * mask
    return field


__all__ = [
    "TransverseBoundaryMode",
    "TransverseBoundarySpec",
    "apply_transverse_boundary_inplace",
    "transverse_boundary_mask",
]
