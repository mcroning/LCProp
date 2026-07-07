from dataclasses import dataclass
from typing import Optional
import math


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


@dataclass(frozen=True)
class LCMaterial:
    name: str = "generic"
    ne: float = 1.70
    no: float = 1.50
    K: float = 7.0e-12
    delta_epsilon: float = 13.0

    def validate(self) -> None:
        if self.ne <= 0.0 or self.no <= 0.0:
            raise ValueError("ne and no must be positive")
        if self.K <= 0.0:
            raise ValueError("K must be positive")
        if self.delta_epsilon <= 0.0:
            raise ValueError("delta_epsilon must be positive")


@dataclass(frozen=True)
class BiasSpec:
    V_bias: float = 0.9144
    theta_bc: float = 0.0
    theta_min: float = 0.0
    theta_max: float = math.pi / 2
    theta_center: Optional[float] = None
    b_override: Optional[float] = None

    def validate(self) -> None:
        if self.V_bias < 0.0:
            raise ValueError("V_bias must be nonnegative")
        if self.b_override is not None and self.b_override < 0.0:
            raise ValueError("b_override must be nonnegative")
        if self.theta_min > self.theta_max:
            raise ValueError("theta_min must be <= theta_max")
        if not (self.theta_min <= self.theta_bc <= self.theta_max):
            raise ValueError("theta_bc must lie within [theta_min, theta_max]")


@dataclass
class LCContext:
    """
    Runtime numerical context built from a RunRequest.

    LCContext should contain derived grid arrays, backend arrays,
    propagation plans, and derived physical parameters. It is not part
    of the immutable experiment request.
    """

    grid: GridSpec
    material: LCMaterial
    bias: BiasSpec
