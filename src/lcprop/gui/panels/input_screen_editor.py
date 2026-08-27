"""Material-neutral BeamPanel editor for shared launch-plane input screens."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from lcprop.core.beams import BeamStack
from lcprop.core.grid import RuntimeGrid
from lcprop.gui.image_sources import (
    RASTER_PREVIEW_SIZE,
    decode_user_raster,
    supported_user_image_formats,
)
from lcprop.optics.launch import build_launch
from lcprop.optics.screens import (
    ChannelLaunchElements,
    EVEN_SQUARE_NEAREST_TRANSPARENT_V1,
    IntensityRasterScreen,
    RasterSource,
    ScreenPlacement,
    prepare_intensity_raster_screen,
)


NO_SCREEN = "none"
INTENSITY_IMAGE_SCREEN = "intensity_image"
_TRANSFORMED_PREVIEW_SIZE = QSize(320, 220)


@dataclass(frozen=True)
class _ScreenBinding:
    beam_definition: object
    screen: IntensityRasterScreen


def _array_pixmap(values, *, xy_axes: bool, size: QSize) -> QPixmap:
    pixels = np.asarray(values, dtype=np.float64)
    if pixels.ndim != 2 or not np.all(np.isfinite(pixels)):
        raise ValueError("preview values must be a finite two-dimensional array")
    if xy_axes:
        pixels = pixels.T
    minimum = float(np.min(pixels))
    maximum = float(np.max(pixels))
    if maximum > minimum:
        scaled = (pixels - minimum) / (maximum - minimum)
    elif maximum > 0.0:
        scaled = np.ones_like(pixels)
    else:
        scaled = np.zeros_like(pixels)
    grayscale = np.ascontiguousarray(np.rint(255.0 * scaled).astype(np.uint8))
    height, width = grayscale.shape
    image = QImage(
        grayscale.data,
        width,
        height,
        grayscale.strides[0],
        QImage.Format.Format_Grayscale8,
    ).copy()
    return QPixmap.fromImage(image).scaled(
        size,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )


class InputScreenEditor(QGroupBox):
    """Edit declarative B1 screens without invoking material propagation."""

    configurationChanged = Signal()

    def __init__(
        self,
        *,
        beam_definitions: Callable[[], object],
        beams: Callable[[], BeamStack],
        runtime_grid: Callable[[], RuntimeGrid],
        enabled: bool,
        disabled_reason: str,
        standard_sources: tuple[RasterSource, ...] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("Input Screen", parent)
        self._beam_definitions = beam_definitions
        self._beams = beams
        self._runtime_grid = runtime_grid
        self._editor_enabled = bool(enabled)
        self._disabled_reason = str(disabled_reason)
        self._standard_sources = tuple(standard_sources)
        self._bindings: list[_ScreenBinding] = []
        self._enabled_definitions: tuple[object, ...] = ()
        self._source: RasterSource | None = None
        self._loading_controls = False
        self._refreshing_preview = False

        layout = QVBoxLayout(self)
        self.availability = QLabel()
        self.availability.setWordWrap(True)
        layout.addWidget(self.availability)

        form = QFormLayout()
        self.channel = QComboBox()
        self.screen_type = QComboBox()
        self.screen_type.addItem("None", NO_SCREEN)
        self.screen_type.addItem("Intensity image", INTENSITY_IMAGE_SCREEN)
        self.source_type = QComboBox()
        self.source_type.addItem("Standard image", "standard")
        self.source_type.addItem("User image", "user")
        self.standard_source = QComboBox()
        for index, source in enumerate(self._standard_sources):
            self.standard_source.addItem(source.display_name, index)
        self.standard_status = QLabel()
        self.standard_status.setWordWrap(True)
        self.standard_status.setMaximumHeight(42)
        if not self._standard_sources:
            self.standard_status.setText(
                "No approved packaged images are available; use User image."
            )
            self.source_type.setCurrentIndex(self.source_type.findData("user"))

        self.choose_file = QPushButton("Choose image…")
        self.selected_source = QLabel("No source selected")
        self.selected_source.setWordWrap(True)
        self.invert = QCheckBox("Invert intensity before padding")
        preview_grid = self._runtime_grid()
        self.width_um = self._length_control(
            min(12.0, 0.5 * float(preview_grid.spec.x_aperture_um))
        )
        self.height_um = self._length_control(
            min(12.0, 0.5 * float(preview_grid.spec.y_aperture_um))
        )
        self.center_x_um = self._position_control(0.0)
        self.center_y_um = self._position_control(0.0)
        self.center_on_beam = QPushButton("Center on beam")

        form.addRow("Channel", self.channel)
        form.addRow("Screen", self.screen_type)
        form.addRow("Source", self.source_type)
        form.addRow("Standard image", self.standard_source)
        form.addRow("Standard status", self.standard_status)
        form.addRow("User image", self.choose_file)
        form.addRow("Selected source", self.selected_source)
        form.addRow("Width (µm)", self.width_um)
        form.addRow("Height (µm)", self.height_um)
        form.addRow("Center x (µm)", self.center_x_um)
        form.addRow("Center y (µm)", self.center_y_um)
        form.addRow("", self.center_on_beam)
        form.addRow("", self.invert)
        layout.addLayout(form)

        self.preview_mode = QComboBox()
        self.preview_mode.addItem("Transmission", "transmission")
        self.preview_mode.addItem("Post-screen beam", "post_screen")
        self.transmission_preview = QLabel("Screen transmission")
        self.transmission_preview.setAlignment(Qt.AlignCenter)
        self.transmission_preview.setMinimumSize(RASTER_PREVIEW_SIZE)
        self.transmission_preview.setMaximumHeight(RASTER_PREVIEW_SIZE.height())
        self.transmission_preview.setStyleSheet("border: 1px solid #888;")
        self.transformed_preview = QLabel("Post-screen channel intensity")
        self.transformed_preview.setAlignment(Qt.AlignCenter)
        self.transformed_preview.setMinimumSize(_TRANSFORMED_PREVIEW_SIZE)
        self.transformed_preview.setMaximumHeight(
            _TRANSFORMED_PREVIEW_SIZE.height()
        )
        self.transformed_preview.setStyleSheet("border: 1px solid #888;")
        self.preview_stack = QStackedWidget()
        self.preview_stack.addWidget(self.transmission_preview)
        self.preview_stack.addWidget(self.transformed_preview)
        preview_row = QHBoxLayout()
        preview_row.addWidget(QLabel("Preview"))
        preview_row.addWidget(self.preview_mode)
        layout.addLayout(preview_row)
        layout.addWidget(self.preview_stack)

        power_form = QFormLayout()
        self.incident_power = QLabel("—")
        self.transmitted_power = QLabel("—")
        self.throughput = QLabel("—")
        power_form.addRow("Incident power (mW)", self.incident_power)
        power_form.addRow("Transmitted power (mW)", self.transmitted_power)
        power_form.addRow("Throughput", self.throughput)
        layout.addLayout(power_form)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self._editable_controls = (
            self.channel,
            self.screen_type,
            self.source_type,
            self.standard_source,
            self.choose_file,
            self.width_um,
            self.height_um,
            self.center_x_um,
            self.center_y_um,
            self.center_on_beam,
            self.invert,
        )
        self.channel.currentIndexChanged.connect(self._selected_channel_changed)
        self.screen_type.currentIndexChanged.connect(self._screen_type_changed)
        self.source_type.currentIndexChanged.connect(self._source_type_changed)
        self.standard_source.currentIndexChanged.connect(
            self._standard_source_changed
        )
        self.choose_file.clicked.connect(self._choose_user_image)
        self.center_on_beam.clicked.connect(self._center_on_selected_beam)
        self.preview_mode.currentIndexChanged.connect(self._preview_mode_changed)
        for control in (
            self.width_um,
            self.height_um,
            self.center_x_um,
            self.center_y_um,
        ):
            control.valueChanged.connect(self._screen_controls_changed)
        self.invert.toggled.connect(self._screen_controls_changed)
        self._screen_detail_widgets = (
            self.source_type,
            self.standard_source,
            self.standard_status,
            self.choose_file,
            self.selected_source,
            self.width_um,
            self.height_um,
            self.center_x_um,
            self.center_y_um,
            self.center_on_beam,
            self.invert,
            self.preview_mode,
            self.preview_stack,
            self.incident_power,
            self.transmitted_power,
            self.throughput,
        )
        self._screen_detail_labels = tuple(
            label
            for widget in self._screen_detail_widgets
            if (label := form.labelForField(widget)) is not None
        ) + tuple(
            label
            for widget in (
                self.incident_power,
                self.transmitted_power,
                self.throughput,
            )
            if (label := power_form.labelForField(widget)) is not None
        )
        self._standard_source_label = form.labelForField(self.standard_source)
        self._standard_status_label = form.labelForField(self.standard_status)
        self._choose_file_label = form.labelForField(self.choose_file)
        self.sync_beams()
        self._update_enabled_state()
        if self._editor_enabled:
            self.refresh_preview()

    @staticmethod
    def _length_control(value: float) -> QDoubleSpinBox:
        control = QDoubleSpinBox()
        control.setRange(1e-6, 1e7)
        control.setDecimals(6)
        control.setValue(value)
        return control

    @staticmethod
    def _position_control(value: float) -> QDoubleSpinBox:
        control = QDoubleSpinBox()
        control.setRange(-1e7, 1e7)
        control.setDecimals(6)
        control.setValue(value)
        return control

    @property
    def editor_enabled(self) -> bool:
        return self._editor_enabled

    @property
    def disabled_reason(self) -> str:
        return self._disabled_reason

    def _current_enabled_definitions(self) -> tuple[object, ...]:
        stack = self._beam_definitions()
        return tuple(beam for beam in stack.beams if beam.enabled)

    @staticmethod
    def _matching_index(target: object, candidates: tuple[object, ...]) -> int | None:
        identity = [
            index for index, candidate in enumerate(candidates) if candidate is target
        ]
        if len(identity) == 1:
            return identity[0]
        equal = [
            index for index, candidate in enumerate(candidates) if candidate == target
        ]
        return equal[0] if len(equal) == 1 else None

    @classmethod
    def _reconciled_definitions(
        cls,
        previous: tuple[object, ...],
        current: tuple[object, ...],
    ) -> dict[int, int]:
        """Map unchanged definitions and one unambiguous immutable edit."""

        mapping: dict[int, int] = {}
        unmatched_current = set(range(len(current)))
        for previous_index, definition in enumerate(previous):
            candidates = tuple(current[index] for index in sorted(unmatched_current))
            relative = cls._matching_index(definition, candidates)
            if relative is None:
                continue
            current_index = sorted(unmatched_current)[relative]
            mapping[previous_index] = current_index
            unmatched_current.remove(current_index)

        unmatched_previous = [
            index for index in range(len(previous)) if index not in mapping
        ]
        if (
            len(previous) == len(current)
            and len(unmatched_previous) == 1
            and len(unmatched_current) == 1
        ):
            mapping[unmatched_previous[0]] = next(iter(unmatched_current))
        return mapping

    def sync_beams(self, *_signal_args, refresh: bool = True) -> None:
        """Reconcile screen bindings after add/remove/reorder/disable changes."""

        definitions = self._current_enabled_definitions()
        mapping = self._reconciled_definitions(self._enabled_definitions, definitions)
        selected = self.channel.currentData()
        selected_old_index = (
            selected
            if isinstance(selected, int)
            and 0 <= selected < len(self._enabled_definitions)
            else None
        )

        reconciled: list[_ScreenBinding] = []
        dropped = False
        for binding in self._bindings:
            old_index = self._matching_index(
                binding.beam_definition,
                self._enabled_definitions,
            )
            if old_index is None or old_index not in mapping:
                dropped = True
                continue
            reconciled.append(
                _ScreenBinding(definitions[mapping[old_index]], binding.screen)
            )
        self._bindings = reconciled
        self._enabled_definitions = definitions

        self.channel.blockSignals(True)
        self.channel.clear()
        for index, definition in enumerate(definitions):
            self.channel.addItem(f"{index + 1}: {definition.name}", index)
        selected_index = (
            mapping.get(selected_old_index) if selected_old_index is not None else None
        )
        self.channel.setCurrentIndex(0 if selected_index is None else selected_index)
        self.channel.blockSignals(False)
        self._load_selected_binding(refresh=refresh)
        if dropped:
            self.status.setText(
                "A stale screen assignment was removed after the beam list changed."
            )

    def _selected_definition(self) -> object | None:
        index = self.channel.currentData()
        if not isinstance(index, int) or not (
            0 <= index < len(self._enabled_definitions)
        ):
            return None
        return self._enabled_definitions[index]

    def _binding_for_selected(self) -> _ScreenBinding | None:
        target = self._selected_definition()
        for binding in self._bindings:
            if binding.beam_definition is target:
                return binding
        return None

    def _replace_selected_binding(self, screen: IntensityRasterScreen | None) -> None:
        target = self._selected_definition()
        if target is None:
            raise ValueError("no enabled channel is selected")
        self._bindings = [
            binding
            for binding in self._bindings
            if binding.beam_definition is not target
        ]
        if screen is not None:
            self._bindings.append(_ScreenBinding(target, screen))

    def _load_selected_binding(self, *, refresh: bool = True) -> None:
        self._loading_controls = True
        binding = self._binding_for_selected()
        if binding is None:
            self.screen_type.setCurrentIndex(self.screen_type.findData(NO_SCREEN))
            self._source = None
            self.selected_source.setText("No source selected")
        else:
            screen = binding.screen
            self.screen_type.setCurrentIndex(
                self.screen_type.findData(INTENSITY_IMAGE_SCREEN)
            )
            self._source = screen.source
            self.selected_source.setText(screen.source.basename)
            self.width_um.setValue(screen.placement.width_um)
            self.height_um.setValue(screen.placement.height_um)
            self.center_x_um.setValue(screen.placement.center_x_um)
            self.center_y_um.setValue(screen.placement.center_y_um)
            self.invert.setChecked(screen.invert)
        self._loading_controls = False
        self._update_enabled_state()
        if self._editor_enabled and refresh:
            self.refresh_preview()

    def _screen_from_controls(self) -> IntensityRasterScreen:
        if self._source is None:
            raise ValueError("choose an image source before attaching the screen")
        screen = IntensityRasterScreen(
            source=self._source,
            placement=ScreenPlacement(
                center_x_um=self.center_x_um.value(),
                center_y_um=self.center_y_um.value(),
                width_um=self.width_um.value(),
                height_um=self.height_um.value(),
                resampling="nearest",
                outside_intensity_transmission=1.0,
                boundary_policy="reject",
            ),
            invert=self.invert.isChecked(),
            preprocessing_policy=EVEN_SQUARE_NEAREST_TRANSPARENT_V1,
        )
        screen.validate()
        return screen

    def _commit_controls(self) -> None:
        if self._loading_controls or not self._editor_enabled:
            return
        if self.screen_type.currentData() == NO_SCREEN:
            self._replace_selected_binding(None)
            return
        self._replace_selected_binding(self._screen_from_controls())

    def _selected_channel_changed(self, *_args) -> None:
        if not self._loading_controls:
            self._load_selected_binding()

    def _screen_controls_changed(self, *_args) -> None:
        if self._loading_controls:
            return
        self._update_enabled_state()
        try:
            self._commit_controls()
            self.status.clear()
        except ValueError as exc:
            self.status.setText(str(exc))
        self.refresh_preview()
        self.configurationChanged.emit()

    def _screen_type_changed(self, *_args) -> None:
        if (
            not self._loading_controls
            and self.screen_type.currentData() == INTENSITY_IMAGE_SCREEN
            and self._binding_for_selected() is None
        ):
            self._set_center_from_selected_beam()
        self._screen_controls_changed()

    def _set_center_from_selected_beam(self) -> None:
        definition = self._selected_definition()
        if definition is None:
            raise ValueError("no enabled channel is selected")
        self.center_x_um.blockSignals(True)
        self.center_y_um.blockSignals(True)
        try:
            self.center_x_um.setValue(float(definition.x_um))
            self.center_y_um.setValue(float(definition.y_um))
        finally:
            self.center_x_um.blockSignals(False)
            self.center_y_um.blockSignals(False)

    def _center_on_selected_beam(self) -> None:
        try:
            self._set_center_from_selected_beam()
            self._screen_controls_changed()
        except ValueError as exc:
            self.status.setText(str(exc))

    def _preview_mode_changed(self, index: int) -> None:
        self.preview_stack.setCurrentIndex(max(0, int(index)))

    def _source_type_changed(self, *_args) -> None:
        self._update_enabled_state()
        if self.source_type.currentData() == "standard":
            self._standard_source_changed()

    def _standard_source_changed(self, *_args) -> None:
        if self._loading_controls or self.source_type.currentData() != "standard":
            return
        index = self.standard_source.currentData()
        if isinstance(index, int) and 0 <= index < len(self._standard_sources):
            self.set_source(self._standard_sources[index])

    def _update_enabled_state(self) -> None:
        image_mode = (
            self._editor_enabled
            and self.screen_type.currentData() == INTENSITY_IMAGE_SCREEN
        )
        for widget in (*self._screen_detail_widgets, *self._screen_detail_labels):
            widget.setVisible(image_mode)
        if not self._editor_enabled:
            self.availability.setText(self._disabled_reason)
            for control in self._editable_controls:
                control.setEnabled(False)
            return
        self.availability.setText(
            "Screens are declarative launch elements; source pixels never "
            "select the grid."
        )
        self.channel.setEnabled(bool(self._enabled_definitions))
        self.screen_type.setEnabled(bool(self._enabled_definitions))
        standard = self.source_type.currentData() == "standard"
        for widget in (self.standard_source, self.standard_status):
            widget.setVisible(image_mode and standard)
        for label in (
            self._standard_source_label,
            self._standard_status_label,
        ):
            if label is not None:
                label.setVisible(image_mode and standard)
        self.choose_file.setVisible(image_mode and not standard)
        if self._choose_file_label is not None:
            self._choose_file_label.setVisible(image_mode and not standard)
        self.source_type.setEnabled(image_mode)
        self.standard_source.setEnabled(
            image_mode and standard and bool(self._standard_sources)
        )
        self.choose_file.setEnabled(image_mode and not standard)
        for control in (
            self.width_um,
            self.height_um,
            self.center_x_um,
            self.center_y_um,
            self.invert,
        ):
            control.setEnabled(image_mode)
        self.center_on_beam.setEnabled(
            image_mode and self._selected_definition() is not None
        )
        self.preview_mode.setEnabled(image_mode)

    def _choose_user_image(self) -> None:
        formats = " ".join(f"*.{value}" for value in supported_user_image_formats())
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Choose input-screen image",
            "",
            f"Raster images ({formats});;All files (*)",
        )
        if path:
            self.load_user_image(path)

    def load_user_image(self, path: str | Path) -> RasterSource:
        decoded = decode_user_raster(path)
        self.set_source(decoded.source)
        return decoded.source

    def set_source(self, source: RasterSource) -> None:
        if not isinstance(source, RasterSource):
            raise TypeError("source must be a RasterSource")
        self._source = source
        self.selected_source.setText(source.basename)
        if self.screen_type.currentData() != INTENSITY_IMAGE_SCREEN:
            self.screen_type.setCurrentIndex(
                self.screen_type.findData(INTENSITY_IMAGE_SCREEN)
            )
        self._screen_controls_changed()

    def clear_launch_elements(self) -> None:
        """Remove every transient channel assignment from the editor."""

        if not self._editor_enabled:
            raise ValueError(self._disabled_reason)
        self._bindings.clear()
        self._load_selected_binding()
        self.configurationChanged.emit()

    def launch_elements(self) -> tuple[ChannelLaunchElements, ...]:
        """Return ordered screen assignments for canonical enabled channels."""

        if not self._editor_enabled:
            raise ValueError(self._disabled_reason)
        current_definitions = self._current_enabled_definitions()
        definitions_are_current = len(current_definitions) == len(
            self._enabled_definitions
        ) and all(
            current is previous
            for current, previous in zip(
                current_definitions,
                self._enabled_definitions,
                strict=True,
            )
        )
        if not definitions_are_current:
            self.sync_beams(refresh=False)
        self._commit_controls()
        definitions = self._current_enabled_definitions()
        grid = self._runtime_grid()
        assignments = []
        for binding in self._bindings:
            index = self._matching_index(binding.beam_definition, definitions)
            if index is None:
                raise ValueError("stale input-screen channel assignment")
            prepare_intensity_raster_screen(binding.screen, grid)
            assignments.append(
                ChannelLaunchElements(index, elements=(binding.screen,))
            )
        return tuple(sorted(assignments, key=lambda value: value.channel_index))

    def refresh_preview(self) -> None:
        if not self._editor_enabled or self._refreshing_preview:
            return
        self._refreshing_preview = True
        try:
            definitions = self._current_enabled_definitions()
            if not definitions:
                raise ValueError("no enabled channel is available")
            assignments = self.launch_elements()
            beams = self._beams()
            grid = self._runtime_grid()
            launch = build_launch(
                beams,
                grid,
                complex_dtype=np.complex128,
                launch_elements=assignments,
            )
            index = self.channel.currentData()
            if not isinstance(index, int) or not 0 <= index < len(beams.channels):
                raise ValueError("no enabled channel is selected")
            binding = next(
                (
                    assignment
                    for assignment in assignments
                    if assignment.channel_index == index
                ),
                None,
            )
            transmission = (
                np.ones((grid.Nx, grid.Ny), dtype=np.float64)
                if binding is None
                else prepare_intensity_raster_screen(binding.elements[0], grid)
            )
            self.transmission_preview.setPixmap(
                _array_pixmap(
                    transmission,
                    xy_axes=True,
                    size=RASTER_PREVIEW_SIZE,
                )
            )
            self.transformed_preview.setPixmap(
                _array_pixmap(
                    np.abs(np.asarray(launch.A0[index])) ** 2,
                    xy_axes=True,
                    size=_TRANSFORMED_PREVIEW_SIZE,
                )
            )
            incident = float(np.asarray(launch.physical_powers_mW)[index])
            transmitted = float(
                np.asarray(launch.post_element_physical_powers_mW)[index]
            )
            throughput = float(
                np.asarray(launch.channel_throughput_fractions)[index]
            )
            self.incident_power.setText(f"{incident:.9g}")
            self.transmitted_power.setText(f"{transmitted:.9g}")
            self.throughput.setText(f"{throughput:.9g}")
            self.status.clear()
        except Exception as exc:
            self.incident_power.setText("—")
            self.transmitted_power.setText("—")
            self.throughput.setText("—")
            self.status.setText(str(exc))
        finally:
            self._refreshing_preview = False


__all__ = [
    "INTENSITY_IMAGE_SCREEN",
    "InputScreenEditor",
    "NO_SCREEN",
]
