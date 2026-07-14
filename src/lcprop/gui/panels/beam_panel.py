from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from lcprop.adapters.launchplane import beam_stack_definition_to_lcprop
from lcprop.core.beams import BeamStack

try:
    from launchplane.launchpane import LaunchPlaneWidget
    from launchplane.model import (
        BeamDefinition,
        BeamStackDefinition,
        LaunchPlaneDefinition,
    )
except ImportError as exc:
    raise ImportError(
        "The LCProp Beam tab requires the separate 'launchplane' package. "
        "Install it before starting the GUI (for local development, use "
        "'python -m pip install -e /path/to/LaunchPane')."
    ) from exc


class BeamPanel(QWidget):
    """LCProp Beam tab backed by the independent LaunchPane widget."""

    def __init__(
        self,
        *,
        x_aperture_um: float = 75.0,
        y_aperture_um: float = 100.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        launch_plane = LaunchPlaneDefinition(
            x_aperture_um=x_aperture_um,
            y_aperture_um=y_aperture_um,
        )
        default_stack = BeamStackDefinition(
            beams=(
                BeamDefinition(
                    name="beam",
                    wavelength_um=0.633,
                    power_mW=1.0,
                    x_um=0.0,
                    y_um=0.0,
                    waist_x_um=3.0,
                    waist_y_um=3.0,
                    tilt_x_rad_per_um=0.0,
                    tilt_y_rad_per_um=0.0,
                    phase_rad=0.0,
                    coherence_group="laser_A",
                    enabled=True,
                ),
            )
        )

        self.launch_plane_widget = LaunchPlaneWidget(
            launch_plane=launch_plane,
            parent=self,
        )
        self.launch_plane_widget.set_beam_stack(default_stack, selected_index=0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.launch_plane_widget)
        self.setMinimumSize(1000, 650)

    @property
    def beam_stack_definition(self) -> BeamStackDefinition:
        """Return the Beam tab's authoritative LaunchPane beam state."""

        return self.launch_plane_widget.beam_stack

    @property
    def launch_plane_definition(self) -> LaunchPlaneDefinition:
        """Return the aperture currently displayed by LaunchPane."""

        return self.launch_plane_widget.scene.definition

    def set_beam_stack_definition(self, stack: BeamStackDefinition) -> None:
        """Replace the Beam tab state for replay and tests."""

        selected_index = 0 if stack.beams else None
        self.launch_plane_widget.set_beam_stack(stack, selected_index=selected_index)

    def set_aperture(
        self,
        x_aperture_um: float,
        y_aperture_um: float,
    ) -> None:
        """Update only the LaunchPane aperture, preserving physical beam data."""

        definition = LaunchPlaneDefinition(
            x_aperture_um=x_aperture_um,
            y_aperture_um=y_aperture_um,
        )
        stack = self.launch_plane_widget.beam_stack
        selected_row = self.launch_plane_widget.object_list.currentRow()
        selected_index = selected_row if selected_row >= 0 else None

        self.launch_plane_widget.scene.set_definition(definition)

        # LaunchPane's inspector ranges are created from its initial aperture.
        # Include out-of-aperture beams when updating those ranges so reducing
        # the aperture never clamps or rewrites their physical coordinates.
        scene = self.launch_plane_widget.scene
        x_values = [beam.x_um for beam in stack.beams]
        y_values = [beam.y_um for beam in stack.beams]
        self.launch_plane_widget.x_spin.setRange(
            min([scene.x_min, *x_values]),
            max([scene.x_max, *x_values]),
        )
        self.launch_plane_widget.y_spin.setRange(
            min([scene.y_min, *y_values]),
            max([scene.y_max, *y_values]),
        )
        self.launch_plane_widget.set_beam_stack(
            stack,
            selected_index=selected_index,
        )
        self.launch_plane_widget.view.fit_aperture()

    def beams(self) -> BeamStack:
        """Return enabled LaunchPane beams adapted to LCProp channels."""

        return beam_stack_definition_to_lcprop(
            self.launch_plane_widget.beam_stack
        )
