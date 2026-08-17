from __future__ import annotations

from launchplane.model import BeamDefinition, BeamStackDefinition

from lcprop.gui.panels.beam_panel import BeamPanel


def pr_default_beam_stack_definition() -> BeamStackDefinition:
    """Return the finite Gaussian launch used by the first PR GUI."""

    return BeamStackDefinition(
        beams=(
            BeamDefinition.from_launch_angles(
                name="PR beam",
                wavelength_um=0.633,
                power_mW=1.0,
                x_um=0.0,
                y_um=0.0,
                waist_x_um=20.0,
                waist_y_um=20.0,
                angle_x_rad=0.0,
                angle_y_rad=0.0,
                phase_rad=0.0,
                coherence_group="pr-laser",
                enabled=True,
            ),
        )
    )


def make_pr_beam_panel(
    *,
    x_aperture_um: float,
    y_aperture_um: float,
) -> BeamPanel:
    panel = BeamPanel(
        x_aperture_um=x_aperture_um,
        y_aperture_um=y_aperture_um,
    )
    panel.set_beam_stack_definition(pr_default_beam_stack_definition())
    return panel


__all__ = ["make_pr_beam_panel", "pr_default_beam_stack_definition"]
