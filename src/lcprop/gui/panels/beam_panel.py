from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QAbstractSpinBox, QSplitter, QVBoxLayout, QWidget

from lcprop.adapters.launchplane import beam_stack_definition_to_lcprop
from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.gui.panels.input_screen_editor import InputScreenEditor
from lcprop.optics.screens import ChannelLaunchElements

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
        preview_Nx: int = 128,
        preview_Ny: int = 128,
        input_screens_enabled: bool = False,
        input_screens_disabled_reason: str = (
            "Input screens are unavailable because this host does not yet "
            "consume shared launch-element plans."
        ),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._preview_Nx = int(preview_Nx)
        self._preview_Ny = int(preview_Ny)

        launch_plane = LaunchPlaneDefinition(
            x_aperture_um=x_aperture_um,
            y_aperture_um=y_aperture_um,
        )
        default_stack = BeamStackDefinition(
            beams=(
                BeamDefinition.from_launch_angles(
                    name="beam",
                    wavelength_um=0.633,
                    power_mW=1.0,
                    x_um=0.0,
                    y_um=0.0,
                    waist_x_um=3.0,
                    waist_y_um=3.0,
                    angle_x_rad=0.0,
                    angle_y_rad=0.0,
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

        self.input_screen_editor = InputScreenEditor(
            beam_definitions=lambda: self.beam_stack_definition,
            beams=self.beams,
            runtime_grid=self._screen_preview_grid,
            enabled=input_screens_enabled,
            disabled_reason=input_screens_disabled_reason,
            parent=self,
        )
        self.input_screen_editor.setMinimumWidth(360)
        self.launch_plane_widget.beamStackChanged.connect(
            self.input_screen_editor.sync_beams
        )

        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.addWidget(self.launch_plane_widget)
        self.splitter.addWidget(self.input_screen_editor)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)
        self.setMinimumSize(1000, 650)
        self._initial_aperture_fit_queued = False
        self._initial_aperture_fit_done = False

    def showEvent(self, event) -> None:
        """Fit once after Qt has assigned the embedded view its real size."""

        super().showEvent(event)
        if not self._initial_aperture_fit_done and not self._initial_aperture_fit_queued:
            self._initial_aperture_fit_queued = True
            QTimer.singleShot(0, self._fit_initial_aperture)

    def _fit_initial_aperture(self) -> None:
        self._initial_aperture_fit_queued = False
        if self._initial_aperture_fit_done or not self.isVisible():
            return
        self.launch_plane_widget.view.fit_aperture()
        self._initial_aperture_fit_done = True

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

        stack.validate()
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
        selected_index = 0 if stack.beams else None
        self.launch_plane_widget.set_beam_stack(stack, selected_index=selected_index)
        self.input_screen_editor.sync_beams()

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
        self.input_screen_editor.refresh_preview()

    def set_preview_grid(self, Nx: int, Ny: int) -> None:
        """Set preview sampling without changing source images or beam geometry."""

        if int(Nx) < 2 or int(Ny) < 2:
            raise ValueError("preview Nx and Ny must be at least two")
        self._preview_Nx = int(Nx)
        self._preview_Ny = int(Ny)
        self.input_screen_editor.refresh_preview()

    def _screen_preview_grid(self):
        definition = self.launch_plane_definition
        return make_grid(
            GridSpec(
                Nx=self._preview_Nx,
                Ny=self._preview_Ny,
                x_aperture_um=float(definition.x_aperture_um),
                y_aperture_um=float(definition.y_aperture_um),
                z_length_um=1.0,
                dz_um=1.0,
            ),
            real_dtype=np.float64,
        )

    def beams(self) -> BeamStack:
        """Return enabled LaunchPane beams adapted to LCProp channels."""

        # LaunchPane disables keyboard tracking on its numerical editors.  A
        # value typed into the active editor therefore may not yet have
        # reached its immutable BeamDefinition (notably when a platform does
        # not move keyboard focus to the Run button).  Commit every pending
        # numerical edit before taking the request snapshot.
        # LaunchPane retains both the angle and transverse-wavevector editors
        # and hides the inactive pair.  Interpreting an inactive editor would
        # switch the beam back to that editor's input mode, so only commit the
        # controls belonging to the currently displayed inspector state.
        editors = [
            editor
            for editor in self.launch_plane_widget.findChildren(QAbstractSpinBox)
            if not editor.isHidden()
        ]
        pending_text = [(editor, editor.lineEdit().text()) for editor in editors]
        for editor, text in pending_text:
            # Committing one field makes LaunchPane refresh the whole
            # inspector, so restore each captured text immediately before it
            # is interpreted or a preceding commit can erase it.
            editor.lineEdit().setText(text)
            editor.interpretText()

        return beam_stack_definition_to_lcprop(
            self.launch_plane_widget.beam_stack
        )

    def launch_elements(self) -> tuple[ChannelLaunchElements, ...]:
        """Return the declarative screen plan for canonical enabled channels."""

        return self.input_screen_editor.launch_elements()

    def launch_configuration(
        self,
    ) -> tuple[BeamStack, tuple[ChannelLaunchElements, ...]]:
        """Return canonical beams together with their ordered launch elements."""

        beams = self.beams()
        return beams, self.input_screen_editor.launch_elements()
