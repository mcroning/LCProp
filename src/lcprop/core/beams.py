from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

CoherenceMode = Literal["incoherent", "coherent"]
BeamProfile = Literal["collimated_gaussian"]

# Reserved internal migration sentinel; user-defined coherence groups must not
# use this value.
LEGACY_COHERENCE_GROUP = "__lcprop_legacy__"


def normalize_coherence_groups(
    n_channels: int,
    *,
    coherent: bool,
    coherence_groups: tuple[str, ...] | list[str] | None = None,
) -> tuple[str, ...]:
    """Normalize legacy coherence or validate explicit per-channel groups."""

    if n_channels < 1:
        raise ValueError("at least one beam channel is required")

    if coherence_groups is None:
        if coherent:
            return tuple("__lcprop_coherent__" for _ in range(n_channels))
        return tuple(f"__lcprop_channel_{index}__" for index in range(n_channels))

    if len(coherence_groups) != n_channels:
        raise ValueError("coherence_groups must have length Nch")
    groups = tuple(coherence_groups)
    if any(not isinstance(group, str) or not group.strip() for group in groups):
        raise ValueError("coherence_groups must contain non-empty strings")
    return groups


@dataclass(frozen=True, kw_only=True)
class BeamChannel:
    """External physical intent; wavevectors are derived, never input fields.

    Radii are external beam-normal 1/e field radii. Phase is referenced to
    the beam axis intersection with the laboratory entrance plane.
    """

    name: str = "beam"
    wavelength_um: float = 0.633
    power_mW: float = 1.0
    n_ext: float = 1.0
    theta_ext_rad: float = 0.0
    phi_rad: float = 0.0
    w1_um: float = 3.0
    w2_um: float = 3.0
    psi_rad: float = 0.0
    x0_um: float = 0.0
    y0_um: float = 0.0
    phase_rad: float = 0.0
    coherence_group: str = LEGACY_COHERENCE_GROUP
    profile: BeamProfile = "collimated_gaussian"

    def __post_init__(self) -> None:
        # Match the public intent model's azimuth/roll canonicalization.
        # Construction remains separate from validation for request builders.
        for name in ("wavelength_um", "power_mW", "n_ext", "theta_ext_rad",
                     "phi_rad", "w1_um", "w2_um", "psi_rad", "x0_um", "y0_um", "phase_rad"):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "phi_rad", self.phi_rad % math.tau)
        object.__setattr__(self, "psi_rad", self.psi_rad % math.tau)

    @property
    def tilt_x_rad_per_um(self) -> float:
        """Derived conserved tangential kx, retained as a read-only consumer API."""
        return 2*math.pi*self.n_ext/self.wavelength_um*math.sin(self.theta_ext_rad)*math.cos(self.phi_rad)

    @property
    def tilt_y_rad_per_um(self) -> float:
        """Derived conserved tangential ky."""
        return 2*math.pi*self.n_ext/self.wavelength_um*math.sin(self.theta_ext_rad)*math.sin(self.phi_rad)

    def validate(self) -> None:
        for name in ("wavelength_um", "power_mW", "n_ext", "theta_ext_rad",
                     "phi_rad", "w1_um", "w2_um", "psi_rad", "x0_um", "y0_um", "phase_rad"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.wavelength_um <= 0 or self.n_ext <= 0:
            raise ValueError("wavelength_um and n_ext must be positive")
        if not math.isfinite(2*math.pi*self.n_ext/self.wavelength_um):
            raise ValueError("external wavevector must be finite")
        if self.power_mW < 0:
            raise ValueError("power_mW must be nonnegative")
        if self.w1_um <= 0 or self.w2_um <= 0:
            raise ValueError("external beam-normal radii must be positive")
        if not 0 <= self.theta_ext_rad < math.pi/2:
            raise ValueError("theta_ext_rad must be in [0, pi/2); grazing launch is unsupported")
        if not isinstance(self.coherence_group, str) or not self.coherence_group.strip():
            raise ValueError("coherence_group must be non-empty")
        if self.profile != "collimated_gaussian":
            raise ValueError("only collimated physical Gaussian launch is supported; focused launch is deferred")


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

        # Resolve once during validation so partially migrated stacks fail at
        # the model boundary instead of silently changing optical semantics.
        self.coherence_groups

    @property
    def Nch(self) -> int:
        return len(self.channels)

    @property
    def coherence_groups(self) -> tuple[str, ...]:
        """Return normalized per-channel coherence groups.

        Channels that retain the legacy default are migrated from the
        stack-wide mode: a coherent stack gets one shared group, while an
        incoherent stack gets one distinct group per channel. Once any channel
        has an explicit group, all channels must have explicit groups.
        """

        groups = tuple(ch.coherence_group for ch in self.channels)
        legacy = tuple(group == LEGACY_COHERENCE_GROUP for group in groups)

        if all(legacy):
            return normalize_coherence_groups(
                len(groups),
                coherent=self.coherence == "coherent",
            )

        if any(legacy):
            raise ValueError(
                "coherence_group migration is incomplete: either leave all channels "
                "at the legacy default or assign every channel an explicit group"
            )

        return normalize_coherence_groups(
            len(groups),
            coherent=self.coherence == "coherent",
            coherence_groups=groups,
        )


__all__ = [
    "BeamChannel",
    "BeamProfile",
    "BeamStack",
    "CoherenceMode",
    "normalize_coherence_groups",
]
