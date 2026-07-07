from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CoherenceMode = Literal["incoherent", "coherent"]


@dataclass(frozen=True)
class BeamChannel:
    """One optical input channel."""

    name: str = "beam"

    wavelength_um: float = 0.633
    power_mW: float = 1.0

    waist_x_um: float = 3.0
    waist_y_um: float = 3.0

    x0_um: float = 0.0
    y0_um: float = 0.0

    tilt_x_rad_per_um: float = 0.0
    tilt_y_rad_per_um: float = 0.0

    phase_rad: float = 0.0

    theta_weight: float = 1.0

    def validate(self) -> None:
        if self.wavelength_um <= 0.0:
            raise ValueError("wavelength_um must be positive")
        if self.power_mW < 0.0:
            raise ValueError("power_mW must be nonnegative")
        if self.waist_x_um <= 0.0 or self.waist_y_um <= 0.0:
            raise ValueError("waists must be positive")
        if self.theta_weight < 0.0:
            raise ValueError("theta_weight must be nonnegative")


@dataclass(frozen=True)
class BeamStack:
    """Collection of optical input channels."""

    channels: tuple[BeamChannel, ...] = field(
        default_factory=lambda: (BeamChannel(),)
    )

    coherence: CoherenceMode = "incoherent"

    def validate(self) -> None:
        if len(self.channels) < 1:
            raise ValueError("at least one beam channel is required")

        if self.coherence not in ("coherent", "incoherent"):
            raise ValueError("invalid coherence mode")

        for ch in self.channels:
            ch.validate()

    @property
    def Nch(self) -> int:
        return len(self.channels)