"""Convert LaunchPane beam models into LCProp beam models.

LaunchPane is an optional, independent package. It is imported only when a
conversion function is called so the rest of LCProp remains usable without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lcprop.core.beams import BeamChannel, BeamStack

if TYPE_CHECKING:
    from launchplane.model import BeamDefinition, BeamStackDefinition


_MISSING_LAUNCHPANE_MESSAGE = (
    "the separate 'launchplane' package is required to use the LaunchPane adapter"
)


def _launchplane_types():
    try:
        from launchplane.model import BeamDefinition, BeamStackDefinition
    except ImportError as exc:
        raise ImportError(_MISSING_LAUNCHPANE_MESSAGE) from exc
    return BeamDefinition, BeamStackDefinition


def _validated_theta_weight(theta_weight: float) -> float:
    value = float(theta_weight)
    if value <= 0.0:
        raise ValueError("theta_weight must be positive")
    return value


def _beam_definition_to_channel(
    beam: BeamDefinition,
    *,
    theta_weight: float,
) -> BeamChannel:
    return BeamChannel(
        name=beam.name,
        wavelength_um=beam.wavelength_um,
        power_mW=beam.power_mW,
        x0_um=beam.x_um,
        y0_um=beam.y_um,
        waist_x_um=beam.waist_x_um,
        waist_y_um=beam.waist_y_um,
        tilt_x_rad_per_um=beam.tilt_x_rad_per_um,
        tilt_y_rad_per_um=beam.tilt_y_rad_per_um,
        phase_rad=beam.phase_rad,
        coherence_group=beam.coherence_group,
        theta_weight=theta_weight,
    )


def beam_definition_to_channel(
    beam: BeamDefinition,
    *,
    theta_weight: float = 1.0,
) -> BeamChannel:
    """Convert one LaunchPane beam definition to an LCProp channel."""

    BeamDefinition, _ = _launchplane_types()
    if not isinstance(beam, BeamDefinition):
        raise TypeError("beam must be a launchplane.model.BeamDefinition")

    beam.validate()
    weight = _validated_theta_weight(theta_weight)
    channel = _beam_definition_to_channel(beam, theta_weight=weight)
    channel.validate()
    return channel


def beam_stack_definition_to_lcprop(
    stack: BeamStackDefinition,
    *,
    theta_weight: float = 1.0,
) -> BeamStack:
    """Convert the enabled beams in a LaunchPane stack to an LCProp stack."""

    _, BeamStackDefinition = _launchplane_types()
    if not isinstance(stack, BeamStackDefinition):
        raise TypeError("stack must be a launchplane.model.BeamStackDefinition")

    # Validate the complete LaunchPane model before disabled beams are filtered.
    stack.validate()
    weight = _validated_theta_weight(theta_weight)
    enabled_beams = tuple(beam for beam in stack.beams if beam.enabled)
    if not enabled_beams:
        raise ValueError("LaunchPane beam stack contains no enabled beams")

    converted = BeamStack(
        channels=tuple(
            _beam_definition_to_channel(beam, theta_weight=weight)
            for beam in enabled_beams
        )
    )
    converted.validate()
    return converted


__all__ = [
    "beam_definition_to_channel",
    "beam_stack_definition_to_lcprop",
]
