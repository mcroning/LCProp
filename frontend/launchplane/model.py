"""Immutable external physical beam intent; no Qt or material resolver.

Angles are radians, lengths micrometres, power mW. This breaking model does
not expose the old laboratory-waist/wavevector launch interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math


def _finite(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass(frozen=True, kw_only=True)
class BeamDefinition:
    """External collimated beam, before interface and laboratory screens.

    w1/w2 are external beam-normal 1/e field radii. Phase is referenced at
    (x_um, y_um, z=0), the axis/entrance intersection. Power is through the
    external beam-normal plane, not an unweighted laboratory-plane integral.
    phi is direction azimuth; psi is independent profile roll. At theta=0,
    phi is retained as editor intent but has no directional significance.
    """
    name: str = "Beam"
    wavelength_um: float = 0.633
    power_mW: float = 1.0
    n_ext: float = 1.0
    theta_ext_rad: float = 0.0
    phi_rad: float = 0.0
    w1_um: float = 3.0
    w2_um: float = 3.0
    psi_rad: float = 0.0
    x_um: float = 0.0
    y_um: float = 0.0
    phase_rad: float = 0.0
    coherence_group: str = "laser_A"
    enabled: bool = True
    profile: str = "collimated_gaussian"

    def __post_init__(self) -> None:
        for name in (
            "wavelength_um", "power_mW", "n_ext", "theta_ext_rad", "phi_rad",
            "w1_um", "w2_um", "psi_rad", "x_um", "y_um", "phase_rad",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name=name))
        object.__setattr__(self, "phi_rad", self.phi_rad % math.tau)
        object.__setattr__(self, "psi_rad", self.psi_rad % math.tau)
        self.validate()

    def validate(self) -> None:
        for name in ("name", "coherence_group"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        for name in ("wavelength_um", "n_ext", "w1_um", "w2_um"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.power_mW < 0:
            raise ValueError("power_mW must be non-negative")
        if not -math.pi / 2 < self.theta_ext_rad < math.pi / 2:
            raise ValueError("theta_ext_rad must satisfy -pi/2 < theta_ext_rad < pi/2")
        if self.profile != "collimated_gaussian":
            raise ValueError("Only collimated_gaussian physical intent is supported; focused launch is deferred")
        if not all(math.isfinite(k) for k in self.external_wavevector_rad_per_um):
            raise ValueError("External wavevector exceeds finite numeric range")

    @property
    def external_direction(self) -> tuple[float, float, float]:
        s = math.sin(self.theta_ext_rad)
        return s * math.cos(self.phi_rad), s * math.sin(self.phi_rad), math.cos(self.theta_ext_rad)

    @property
    def external_wavevector_rad_per_um(self) -> tuple[float, float, float]:
        k = math.tau * self.n_ext / self.wavelength_um
        return tuple(k * v for v in self.external_direction)

    @property
    def transverse_wavevector_rad_per_um(self) -> tuple[float, float]:
        return self.external_wavevector_rad_per_um[:2]


@dataclass(frozen=True)
class BeamStackDefinition:
    """Ordered physical intents; ordering and names do not provide stable IDs."""
    beams: tuple[BeamDefinition, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "beams", tuple(self.beams))
        self.validate()

    def validate(self) -> None:
        names = set()
        for beam in self.beams:
            if not isinstance(beam, BeamDefinition):
                raise ValueError("beams must contain physical BeamDefinition objects")
            beam.validate()
            if beam.name in names:
                raise ValueError(f"Duplicate beam name '{beam.name}'.")
            names.add(beam.name)

    @property
    def total_power_mW(self) -> float:
        """Sum of enabled requested incident powers, not coherent group flux."""
        return sum(b.power_mW for b in self.beams if b.enabled)

    @property
    def coherence_groups(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(b.coherence_group for b in self.beams))


@dataclass(frozen=True)
class LaunchPlaneDefinition:
    """Laboratory face extent; x horizontal/right, y vertical/up."""
    x_aperture_um: float = 75.0
    y_aperture_um: float = 100.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for name in ("x_aperture_um", "y_aperture_um"):
            if _finite(getattr(self, name), name=name) <= 0:
                raise ValueError(f"{name} must be positive")


__all__ = ["BeamDefinition", "BeamStackDefinition", "LaunchPlaneDefinition"]
