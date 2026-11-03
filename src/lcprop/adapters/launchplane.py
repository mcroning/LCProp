"""Convert LaunchPane beam models into LCProp beam models.

LaunchPane is an optional, independent package. It is imported only when a
conversion function is called so the rest of LCProp remains usable without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.adapters.legacy_beams import discard_legacy_unit_theta_weight

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
    from launchplane.serialization import SCHEMA_VERSION
    if SCHEMA_VERSION != 4:
        raise ImportError("physical LCProp launch requires LaunchPlane schema 4; update both packages together")
    return BeamDefinition, BeamStackDefinition


def _beam_definition_to_channel(beam: BeamDefinition) -> BeamChannel:
    return BeamChannel(
        name=beam.name, wavelength_um=beam.wavelength_um, power_mW=beam.power_mW,
        n_ext=beam.n_ext, theta_ext_rad=beam.theta_ext_rad, phi_rad=beam.phi_rad,
        w1_um=beam.w1_um, w2_um=beam.w2_um, psi_rad=beam.psi_rad,
        x0_um=beam.x_um, y0_um=beam.y_um, phase_rad=beam.phase_rad,
        coherence_group=beam.coherence_group, profile=beam.profile,
    )


def beam_definition_to_channel(
    beam: BeamDefinition,
    **legacy_options,
) -> BeamChannel:
    """Convert one LaunchPane beam definition to an LCProp channel."""

    BeamDefinition, _ = _launchplane_types()
    if not isinstance(beam, BeamDefinition):
        raise TypeError("beam must be a launchplane.model.BeamDefinition")

    beam.validate()
    discard_legacy_unit_theta_weight(legacy_options)
    channel = _beam_definition_to_channel(beam)
    channel.validate()
    return channel


def beam_stack_definition_to_lcprop(
    stack: BeamStackDefinition,
    **legacy_options,
) -> BeamStack:
    """Convert the enabled beams in a LaunchPane stack to an LCProp stack."""

    _, BeamStackDefinition = _launchplane_types()
    if not isinstance(stack, BeamStackDefinition):
        raise TypeError("stack must be a launchplane.model.BeamStackDefinition")

    # Validate the complete LaunchPane model before disabled beams are filtered.
    stack.validate()
    discard_legacy_unit_theta_weight(legacy_options)
    enabled_beams = tuple(beam for beam in stack.beams if beam.enabled)
    if not enabled_beams:
        raise ValueError("LaunchPane beam stack contains no enabled beams")

    converted = BeamStack(
        channels=tuple(
            _beam_definition_to_channel(beam)
            for beam in enabled_beams
        )
    )
    converted.validate()
    return converted


def beam_stack_to_launchplane(stack: BeamStack):
    """Convert an LCProp beam stack to enabled LaunchPane definitions.

    All optical channel fields and explicit coherence groups are preserved.
    """

    BeamDefinition, BeamStackDefinition = _launchplane_types()
    if not isinstance(stack, BeamStack):
        raise TypeError("stack must be an lcprop.core.beams.BeamStack")
    stack.validate()
    definition = BeamStackDefinition(
        beams=tuple(
            BeamDefinition(
                name=channel.name,
                wavelength_um=channel.wavelength_um,
                power_mW=channel.power_mW,
                x_um=channel.x0_um,
                y_um=channel.y0_um,
                n_ext=channel.n_ext,
                theta_ext_rad=channel.theta_ext_rad,
                phi_rad=channel.phi_rad,
                w1_um=channel.w1_um,
                w2_um=channel.w2_um,
                psi_rad=channel.psi_rad,
                phase_rad=channel.phase_rad,
                coherence_group=group,
                enabled=True,
                profile=channel.profile,
            )
            for channel, group in zip(
                stack.channels,
                stack.coherence_groups,
            )
        )
    )
    definition.validate()
    return definition


__all__ = [
    "beam_definition_to_channel",
    "beam_stack_definition_to_lcprop",
    "beam_stack_to_launchplane",
]
