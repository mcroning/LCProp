from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lcprop.adapters.launchplane import beam_stack_definition_to_lcprop
from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.gui.panels.input_screen_editor import InputScreenEditor
from lcprop.optics.launch_configuration import LaunchConfiguration
from lcprop.optics.boundaries import TransverseBoundarySpec
from lcprop.optics.launch import OpticalLaunchContext, build_launch
from lcprop.optics.physical_launch import resolve_beam_geometry
from lcprop.optics.sampling import qualify_launch_sampling
from lcprop.optics.screens import ChannelLaunchElements
from lcprop.optics.physical_launch import external_direction_for_exit

try:
    from launchplane.launchpane import LaunchPlaneWidget
    from launchplane.model import (
        BeamDefinition,
        BeamStackDefinition,
        LaunchPlaneDefinition,
    )
    from launchplane.serialization import SCHEMA_VERSION as LAUNCHPANE_SCHEMA_VERSION
    if (LAUNCHPANE_SCHEMA_VERSION != 4
            or not getattr(LaunchPlaneWidget, "supports_resolved_internal_rays", False)
            or not hasattr(LaunchPlaneWidget, "set_inverse_resolver")
            or not hasattr(LaunchPlaneWidget, "commit_pending_edits")):
        raise ImportError("LaunchPlane schema 4 with host-resolved ray preview, inverse editing and pending-edit synchronization support is required")
except ImportError as exc:
    raise ImportError(
        "The LCProp Beam tab requires coordinated LaunchPlane schema 4 with "
        "host-resolved ray preview, inverse editing and pending-edit synchronization support. Install the reviewed LCProp/LaunchPlane "
        "pair together before starting the GUI; obsolete packages are incompatible."
    ) from exc


class BeamPanel(QWidget):
    """LCProp Beam tab backed by the independent LaunchPlane widget."""

    beamStackChanged = Signal(object)

    def _direction_for_exit(self, beam, x_um: float, y_um: float):
        if not self._has_optical_context:
            raise ValueError("material-aware optical context is unavailable")
        channel = beam_stack_definition_to_lcprop(
            BeamStackDefinition(beams=(beam,))).channels[0]
        return external_direction_for_exit(channel, self._preview_n_ref,
                                           self._interaction_length_um, x_um, y_um)

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
        self._preview_n_ref = 1.0
        self._interaction_length_um = 1.0
        self._propagation_sign = 1
        self._has_optical_context = False

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
                    w1_um=3.0,
                    w2_um=3.0,
                    theta_ext_rad=0.0,
                    phi_rad=0.0,
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
        # Presentation adaptation only; LaunchPlane remains the intent owner.
        widget = self.launch_plane_widget
        for number, editor in enumerate((widget.w1_spin, widget.w2_spin), 1):
            label = f"Principal radius {number}"
            editor.setAccessibleName(label)
            for form in widget.findChildren(QFormLayout):
                field_label = form.labelForField(editor)
                if field_label is not None:
                    field_label.setText(label)
            editor.setToolTip(
                "Radius along a beam-normal principal axis.\n"
                "1/e field radius = 1/e² intensity radius.\n"
                "At startup: normal incidence, zero roll.\n"
                "Principal radius 1 aligns with x; radius 2 with y.\n"
                "Away from startup, these need not be laboratory x/y axes.")
        widget.psi_spin.setToolTip(
            "Rotates analytic profiles about the beam direction.\n"
            "Profile Roll does not rotate an applied image;\n"
            "their orientation stays fixed in image/lab-plane coordinates.")
        widget.enabled_checkbox.setToolTip(
            "Enabled includes this channel in the optical stack.\nDisabling retains "
            "its GUI settings but excludes the channel.\nTo extinguish an existing "
            "channel during compatible TD continuation, retain it and set power to "
            "zero; the workflow's continuation compatibility rules still apply.")
        # QFormLayout labels are separate hover targets from their editors.
        for form in widget.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                label_item = form.itemAt(row, QFormLayout.LabelRole)
                field_item = form.itemAt(row, QFormLayout.FieldRole)
                if label_item and field_item and label_item.widget() and field_item.widget():
                    label_item.widget().setToolTip(field_item.widget().toolTip())
        self.launch_plane_widget.set_inverse_resolver(self._direction_for_exit)
        self.launch_plane_widget.set_beam_stack(default_stack, selected_index=0)
        # Keep embedded choices accessible even when native selectors elide text.
        for selector in self.launch_plane_widget.findChildren(QComboBox):
            guidance = selector.toolTip()
            def update_tooltip(text, widget=selector, explanation=guidance):
                widget.setToolTip(text + ("\n" + explanation if explanation else ""))
                for form in self.launch_plane_widget.findChildren(QFormLayout):
                    label = form.labelForField(widget)
                    if label is not None:
                        label.setToolTip(widget.toolTip())
            selector.currentTextChanged.connect(update_tooltip)
            update_tooltip(selector.currentText())
            for index in range(selector.count()):
                selector.setItemData(index, selector.itemText(index), Qt.ToolTipRole)

        self.input_screen_editor = InputScreenEditor(
            beam_definitions=lambda: self.beam_stack_definition,
            # Preview is a read of committed intent, not a request boundary.
            beams=lambda: beam_stack_definition_to_lcprop(self.beam_stack_definition),
            runtime_grid=self._screen_preview_grid,
            launch_context=self._preview_launch_context,
            enabled=input_screens_enabled,
            disabled_reason=input_screens_disabled_reason,
            parent=self,
        )
        self.input_screen_editor.setMinimumWidth(0)
        self.launch_plane_widget.beamStackChanged.connect(
            self._beam_stack_changed
        )

        self.splitter = QSplitter(Qt.Horizontal, self)
        self.splitter.addWidget(self.launch_plane_widget)

        self.splitter.setStretchFactor(0, 3)
        boundary_box = QGroupBox("Optical edge treatment", self)
        boundary_form = QFormLayout(boundary_box)
        self.boundary_mode = QComboBox(boundary_box)
        self.boundary_mode.addItem("None", "periodic")
        self.boundary_mode.addItem("Sponge", "sponge")
        self.boundary_mode.addItem("Tukey", "tukey")
        self.boundary_width = QDoubleSpinBox(boundary_box)
        self.boundary_width.setRange(0.001, 1.0)
        self.boundary_width.setDecimals(4)
        self.boundary_width.setValue(0.15)
        self.boundary_attenuation = QDoubleSpinBox(boundary_box)
        self.boundary_attenuation.setRange(1.0e-6, 1.0e3)
        self.boundary_attenuation.setDecimals(6)
        self.boundary_attenuation.setSuffix(" µm⁻¹")
        self.boundary_attenuation.setValue(0.05)
        self.boundary_order = QSpinBox(boundary_box)
        self.boundary_order.setRange(1, 16)
        self.boundary_order.setValue(2)
        self.boundary_tukey_alpha = QDoubleSpinBox(boundary_box)
        self.boundary_tukey_alpha.setRange(0.0, 1.0)
        self.boundary_tukey_alpha.setDecimals(4)
        self.boundary_tukey_alpha.setValue(0.1)
        boundary_form.addRow("Policy", self.boundary_mode)
        boundary_form.addRow("Sponge width / half-aperture", self.boundary_width)
        boundary_form.addRow("Sponge amplitude rate", self.boundary_attenuation)
        boundary_form.addRow("Sponge profile order", self.boundary_order)
        boundary_form.addRow("Tukey alpha", self.boundary_tukey_alpha)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.beam_tabs = QTabWidget(self)
        self.beam_tabs.addTab(self.splitter, "Beams")
        self.beam_tabs.addTab(boundary_box, "Optical edge treatment")
        if input_screens_enabled:
            # Image controls appear dynamically and can exceed the tab height.
            # Let Qt honor their size hints instead of compressing form rows.
            self.input_screen_scroll = QScrollArea(self)
            self.input_screen_scroll.setWidgetResizable(True)
            self.input_screen_scroll.setFrameShape(QScrollArea.NoFrame)
            self.input_screen_scroll.setWidget(self.input_screen_editor)
            self.beam_tabs.addTab(self.input_screen_scroll, "Input Screen")
        else:
            self.input_screen_editor.hide()
        layout.addWidget(self.beam_tabs)
        self._initial_aperture_fit_queued = False
        self._initial_aperture_fit_done = False
        for control in (
            self.boundary_mode,
            self.boundary_width,
            self.boundary_attenuation,
            self.boundary_order,
            self.boundary_tukey_alpha,
        ):
            signal = (
                control.currentIndexChanged
                if isinstance(control, QComboBox)
                else control.valueChanged
            )
            signal.connect(self._boundary_changed)
        self._boundary_changed()

    def bind_aperture_controls(self, grid_panel) -> None:
        """Expose the existing Grid aperture on Input face; no second grid state."""
        row = QHBoxLayout()
        for axis in ("x", "y"):
            canonical = getattr(grid_panel, f"{axis}_aperture_um")
            editor = QDoubleSpinBox(self)
            editor.setRange(canonical.minimum(), canonical.maximum())
            editor.setDecimals(canonical.decimals())
            editor.setSingleStep(canonical.singleStep())
            editor.setKeyboardTracking(canonical.keyboardTracking())
            editor.setSuffix(" µm")
            editor.setValue(canonical.value())
            editor.setToolTip("Full input-face width; shared with Grid and saved in the experiment.")
            setattr(self, f"{axis}_aperture_control", editor)
            row.addWidget(QLabel(f"{axis} full width", self))
            row.addWidget(editor)
            editor.valueChanged.connect(canonical.setValue)
            canonical.valueChanged.connect(editor.setValue)
        self.launch_plane_widget.canvas_panel.layout().insertLayout(1, row)

    def optical_boundary(self) -> TransverseBoundarySpec:
        return TransverseBoundarySpec(
            mode=str(self.boundary_mode.currentData()),
            width_fraction=self.boundary_width.value(),
            attenuation_per_um=self.boundary_attenuation.value(),
            profile_order=self.boundary_order.value(),
            tukey_alpha=self.boundary_tukey_alpha.value(),
        )

    def set_optical_boundary(self, spec: TransverseBoundarySpec) -> None:
        spec.validate()
        controls = (
            (self.boundary_mode, self.boundary_mode.findData(spec.mode)),
            (self.boundary_width, spec.width_fraction),
            (self.boundary_attenuation, spec.attenuation_per_um),
            (self.boundary_order, spec.profile_order),
            (self.boundary_tukey_alpha, spec.tukey_alpha),
        )
        for control, value in controls:
            control.blockSignals(True)
            if isinstance(control, QComboBox):
                control.setCurrentIndex(int(value))
            else:
                control.setValue(value)
            control.blockSignals(False)
        self._boundary_changed()

    def set_optical_context(
        self, *, n_ref: float, interaction_length_um: float, propagation_sign: int = 1
    ) -> None:
        self._has_optical_context = True
        self._preview_n_ref = float(n_ref)
        self._interaction_length_um = float(interaction_length_um)
        self._propagation_sign = int(propagation_sign)
        self._boundary_changed()
        self.input_screen_editor.refresh_preview()

    def _preview_launch_context(self, grid) -> OpticalLaunchContext:
        return OpticalLaunchContext(
            grid=grid,
            n_ref=self._preview_n_ref,
            interaction_length_um=self._interaction_length_um,
            propagation_sign=self._propagation_sign,
        )

    def _boundary_changed(self, _value=None) -> None:
        contours = []
        endpoints = [None] * len(self.beam_stack_definition.beams)
        try:
            if not self._has_optical_context:
                raise ValueError("material index and interaction length are not yet supplied")
            grid = self._screen_preview_grid()
            beams = beam_stack_definition_to_lcprop(self.beam_stack_definition)
            sampling = qualify_launch_sampling(beams, grid, self._preview_n_ref)
            launch = (None if sampling.errors else
                      build_launch(beams, grid, complex_dtype=np.complex128,
                                   context=self._preview_launch_context(grid)))
            geometries = tuple(resolve_beam_geometry(b, self._preview_n_ref) for b in beams.channels)
            descriptions = []
            angles = np.linspace(0, 2*np.pi, 65)
            circle = np.stack((np.cos(angles), np.sin(angles)))
            enabled_indices = [i for i, b in enumerate(self.beam_stack_definition.beams) if b.enabled]
            for index, (beam, geometry) in enumerate(zip(beams.channels, geometries)):
                eigenvalues, axes = np.linalg.eigh(geometry.interface_quadratic)
                points = (axes @ (circle/np.sqrt(eigenvalues)[:, None])).T
                points += [beam.x0_um, beam.y0_um]
                contours.append(points.tolist())
                endpoints[enabled_indices[index]] = (
                    beam.x0_um + self._interaction_length_um * geometry.kx / geometry.kz_internal,
                    beam.y0_um + self._interaction_length_um * geometry.ky / geometry.kz_internal,
                )
                capture_text = ("capture estimate unavailable on preview sampling grid" if launch is None
                                else f"captured entrance flux estimate {launch.power_metadata['capture_fraction'][index]:.2%}")
                descriptions.append(
                    f"{beam.name}: internal angle {np.degrees(geometry.theta_internal):.3g}°, "
                    f"{capture_text}"
                )
            description = (
                "Host-resolved 1/e field footprints (dashed); rays: internal crystal exit. " + "; ".join(descriptions)
                + ". Scalar ideal interface; spectral/sampling qualification applies."
            )
        except (ValueError, TypeError) as exc:
            contours = []
            endpoints = [None] * len(self.beam_stack_definition.beams)
            description = f"Resolved preview unavailable: {exc}"
        self.launch_plane_widget.set_resolved_preview(
            contours, description=description, ray_endpoints=endpoints)

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
        """Return the Beam tab's authoritative LaunchPlane beam state."""

        return self.launch_plane_widget.beam_stack

    @property
    def launch_plane_definition(self) -> LaunchPlaneDefinition:
        """Return the aperture currently displayed by LaunchPlane."""

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
        self._beam_stack_changed(stack)

    def _beam_stack_changed(self, stack: BeamStackDefinition) -> None:
        """Relay every canonical interactive or programmatic stack change."""

        self.input_screen_editor.sync_beams()
        self._boundary_changed()
        self.beamStackChanged.emit(stack)

    def set_aperture(
        self,
        x_aperture_um: float,
        y_aperture_um: float,
    ) -> None:
        """Update only the LaunchPlane aperture, preserving physical beam data."""

        definition = LaunchPlaneDefinition(
            x_aperture_um=x_aperture_um,
            y_aperture_um=y_aperture_um,
        )
        stack = self.launch_plane_widget.beam_stack
        selected_row = self.launch_plane_widget.object_list.currentRow()
        selected_index = selected_row if selected_row >= 0 else None

        self.launch_plane_widget.scene.set_definition(definition)

        # LaunchPlane's inspector ranges are created from its initial aperture.
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
        # Replacing the scene invalidates derived contours; redeliver the
        # current host geometry after the final stack/scene replacement.
        self._boundary_changed()
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
        """Return enabled LaunchPlane beams adapted to LCProp channels."""

        # Run, Save and both material request builders share this boundary.
        # The editor validates and commits once; its synchronous stack signal
        # redelivers the host preview before this request snapshot is returned.
        self.launch_plane_widget.commit_pending_edits()

        return beam_stack_definition_to_lcprop(
            self.launch_plane_widget.beam_stack
        )

    def launch_elements(self) -> tuple[ChannelLaunchElements, ...]:
        """Return the declarative screen plan for canonical enabled channels."""

        return self.input_screen_editor.launch_elements()

    def clear_launch_elements(self) -> None:
        """Clear all transient declarative input-screen assignments."""

        self.input_screen_editor.clear_launch_elements()

    def launch_configuration(
        self,
    ) -> LaunchConfiguration:
        """Return canonical beams together with their ordered launch elements."""

        beams = self.beams()
        return LaunchConfiguration(
            beams=beams,
            channel_elements=self.input_screen_editor.launch_elements(),
        )
