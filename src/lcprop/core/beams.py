from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CoherenceMode = Literal["incoherent", "coherent"]

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


@dataclass(frozen=True)
class BeamChannel:
    """One optical input channel.

    ``tilt_x_rad_per_um`` and ``tilt_y_rad_per_um`` are transverse phase
    gradients, not geometric angles.  The launch phase is
    ``tilt_x_rad_per_um * x + tilt_y_rad_per_um * y + phase_rad``.
    """

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
    coherence_group: str = LEGACY_COHERENCE_GROUP

    def validate(self) -> None:
        if self.wavelength_um <= 0.0:
            raise ValueError("wavelength_um must be positive")
        if self.power_mW < 0.0:
            raise ValueError("power_mW must be nonnegative")
        if self.waist_x_um <= 0.0 or self.waist_y_um <= 0.0:
            raise ValueError("waists must be positive")
        if self.theta_weight < 0.0:
            raise ValueError("theta_weight must be nonnegative")
        if not isinstance(self.coherence_group, str) or not self.coherence_group.strip():
            raise ValueError("coherence_group must be non-empty")


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
    "BeamStack",
    "CoherenceMode",
    "normalize_coherence_groups",
]
