from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class GridSpec:
    """Human-facing transverse/z discretization choices."""

    Nx: int = 256
    Ny: int = 256
    dz_um: float = 5.0
    x_aperture_um: float = 75.0
    y_aperture_um: float = 100.0
    z_length_um: float = 3000.0

    def validate(self) -> None:
        if self.Nx <= 1 or self.Ny <= 1:
            raise ValueError("Nx and Ny must be > 1")
        if self.dz_um <= 0.0:
            raise ValueError("dz_um must be positive")
        if self.x_aperture_um <= 0.0:
            raise ValueError("x_aperture_um must be positive")
        if self.y_aperture_um <= 0.0:
            raise ValueError("y_aperture_um must be positive")
        if self.z_length_um <= 0.0:
            raise ValueError("z_length_um must be positive")


@dataclass(frozen=True)
class TimeSpec:
    """Physical/pseudo-time discretization choices."""

    dt: float = 7.5e-4
    Nt: int = 20

    def validate(self) -> None:
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.Nt < 0:
            raise ValueError("Nt must be nonnegative")


_LC_COMPATIBILITY_EXPORTS = {"BiasSpec", "LCContext", "LCMaterial"}


def __getattr__(name: str):
    if name not in _LC_COMPATIBILITY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("lcprop.lc.specs"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LC_COMPATIBILITY_EXPORTS)


__all__ = [
    "BiasSpec",
    "GridSpec",
    "LCContext",
    "LCMaterial",
    "TimeSpec",
]
