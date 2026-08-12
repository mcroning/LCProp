"""Canonical liquid-crystal material specifications."""

from dataclasses import dataclass
import math
from typing import Optional

from lcprop.core.context import GridSpec


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
    """Runtime numerical context built from an LC request."""

    grid: GridSpec
    material: LCMaterial
    bias: BiasSpec


__all__ = ["BiasSpec", "LCContext", "LCMaterial"]
