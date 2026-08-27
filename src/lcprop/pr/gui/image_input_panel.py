"""PR GUI controls for Gaussian and image-amplification launch modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from lcprop.gui.image_sources import (
    decode_user_raster,
    supported_user_image_formats as shared_supported_user_image_formats,
)
from lcprop.pr.image_amplification import (
    PRBeamPanelImageAmplificationRunRequest,
    PRImageAmplificationExperimentRequest,
)
from lcprop.pr.image_sources import PRImageSource


PR_GAUSSIAN_INPUT_MODE = "gaussian_beams"
PR_IMAGE_AMPLIFICATION_INPUT_MODE = "image_amplification"


@dataclass(frozen=True)
class DecodedPRImage:
    source: PRImageSource
    preview: QImage


def supported_user_image_formats() -> tuple[str, ...]:
    return shared_supported_user_image_formats()


def decode_user_image(path: str | Path) -> DecodedPRImage:
    """Compatibility wrapper around shared bounded raster decoding."""

    decoded = decode_user_raster(path, source_factory=PRImageSource)
    return DecodedPRImage(source=decoded.source, preview=decoded.preview)


class PRImageInputPanel(QWidget):
    """Input-mode selector and PR-owned pump/signal role controls."""

    modeChanged = Signal(str)
    configurationChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._beam_panel = None
        self._role_names: list[str | None] = [None, None]
        self._enabled_channel_names: list[str] = []

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.input_mode = QComboBox()
        self.input_mode.addItem("Gaussian beams", PR_GAUSSIAN_INPUT_MODE)
        self.input_mode.addItem(
            "Image amplification", PR_IMAGE_AMPLIFICATION_INPUT_MODE
        )
        self.pump_channel = QComboBox()
        self.signal_channel = QComboBox()
        self.role_note = QLabel(
            "Configure beams and the signal intensity screen on the shared "
            "Beam tab. Channel numbers use the enabled-channel ordering."
        )
        self.role_note.setWordWrap(True)

        form.addRow("Input mode", self.input_mode)
        form.addRow("Pump channel", self.pump_channel)
        form.addRow("Signal channel", self.signal_channel)
        layout.addLayout(form)
        layout.addWidget(self.role_note)
        layout.addStretch(1)
        self._image_widgets = tuple(
            widget
            for widget in (
                self.pump_channel,
                self.signal_channel,
                self.role_note,
            )
        )
        self.input_mode.currentIndexChanged.connect(self._mode_changed)
        self.pump_channel.currentIndexChanged.connect(
            lambda _index: self._role_changed(0)
        )
        self.signal_channel.currentIndexChanged.connect(
            lambda _index: self._role_changed(1)
        )
        self._mode_changed()

    def set_beam_panel(self, beam_panel) -> None:
        """Bind role selectors to the shared authoritative BeamPanel."""

        if self._beam_panel is not None:
            try:
                self._beam_panel.beamStackChanged.disconnect(self.sync_channels)
            except RuntimeError:
                pass
        self._beam_panel = beam_panel
        beam_panel.beamStackChanged.connect(self.sync_channels)
        self.sync_channels()

    def mode_id(self) -> str:
        return str(self.input_mode.currentData())

    def is_image_amplification(self) -> bool:
        return self.mode_id() == PR_IMAGE_AMPLIFICATION_INPUT_MODE

    def _mode_changed(self, *_args) -> None:
        enabled = self.is_image_amplification()
        for widget in self._image_widgets:
            widget.setVisible(enabled)
        self.modeChanged.emit(self.mode_id())
        self.configurationChanged.emit()

    def sync_channels(self, *_args) -> None:
        """Refresh roles by beam name, clearing roles that became stale."""

        if self._beam_panel is None:
            return
        previous_role_names = tuple(self._role_names)
        previous_enabled_names = tuple(self._enabled_channel_names)
        enabled = [
            beam for beam in self._beam_panel.beam_stack_definition.beams
            if beam.enabled
        ]
        names = [str(beam.name) for beam in enabled]
        self._enabled_channel_names = names

        mapping: dict[int, int] = {}
        unmatched_current = set(range(len(names)))
        for previous_index, previous_name in enumerate(previous_enabled_names):
            matches = [
                index
                for index in unmatched_current
                if names[index] == previous_name
            ]
            if len(matches) == 1:
                mapping[previous_index] = matches[0]
                unmatched_current.remove(matches[0])
        unmatched_previous = [
            index
            for index in range(len(previous_enabled_names))
            if index not in mapping
        ]
        if (
            len(previous_enabled_names) == len(names)
            and len(unmatched_previous) == 1
            and len(unmatched_current) == 1
        ):
            mapping[unmatched_previous[0]] = next(iter(unmatched_current))

        resolved_indices = []
        for previous_name in previous_role_names:
            old_matches = [
                index
                for index, name in enumerate(previous_enabled_names)
                if name == previous_name
            ]
            resolved_indices.append(
                mapping.get(old_matches[0]) if len(old_matches) == 1 else None
            )
        if (
            resolved_indices[0] is not None
            and resolved_indices[0] == resolved_indices[1]
        ):
            resolved_indices[1] = None
        if len(names) == 2:
            used = {index for index in resolved_indices if index is not None}
            for role, resolved in enumerate(resolved_indices):
                if resolved is not None:
                    continue
                available = [index for index in range(2) if index not in used]
                if available:
                    resolved_indices[role] = available[0]
                    used.add(available[0])
        for role, selector in enumerate((self.pump_channel, self.signal_channel)):
            selector.blockSignals(True)
            selector.clear()
            for index, name in enumerate(names):
                selector.addItem(f"{index} — {name}", index)
            resolved = resolved_indices[role]
            if resolved is not None:
                selector.setCurrentIndex(resolved)
                self._role_names[role] = names[resolved]
            else:
                selector.setCurrentIndex(-1)
                self._role_names[role] = None
            selector.blockSignals(False)
        self.configurationChanged.emit()

    def _role_changed(self, role: int) -> None:
        selector = (self.pump_channel, self.signal_channel)[role]
        index = selector.currentData()
        self._role_names[role] = (
            self._enabled_channel_names[index]
            if isinstance(index, int)
            and 0 <= index < len(self._enabled_channel_names)
            else None
        )
        self.configurationChanged.emit()

    def set_role_indices(self, pump_index: int, signal_index: int) -> None:
        """Restore pump/signal roles in canonical enabled-channel ordering."""

        self.sync_channels()
        count = len(self._enabled_channel_names)
        for name, value in (
            ("pump_channel_index", pump_index),
            ("signal_channel_index", signal_index),
        ):
            if type(value) is not int or not 0 <= value < count:
                raise ValueError(f"{name} does not identify an enabled channel")
        if pump_index == signal_index:
            raise ValueError("pump and signal channels must be distinct")
        self.pump_channel.setCurrentIndex(pump_index)
        self.signal_channel.setCurrentIndex(signal_index)

    def build_request(self, *, grid, material, solver, backend, launch_configuration):
        if not self.is_image_amplification():
            raise ValueError("image-amplification input mode is not selected")
        self.sync_channels()
        if self.pump_channel.currentData() is None:
            raise ValueError("select a valid enabled pump channel")
        if self.signal_channel.currentData() is None:
            raise ValueError("select a valid enabled signal channel")
        request = PRBeamPanelImageAmplificationRunRequest(
            grid=grid,
            material=material,
            solver=solver,
            backend=backend,
            launch_configuration=launch_configuration,
            pump_channel_index=int(self.pump_channel.currentData()),
            signal_channel_index=int(self.signal_channel.currentData()),
        )
        request.validate()
        return request

    def build_experiment_request(
        self,
        *,
        base_workflow_id: str,
        base_request,
        launch_configuration,
    ) -> PRImageAmplificationExperimentRequest:
        """Compose image roles over one canonical ordinary PR request."""

        if not self.is_image_amplification():
            raise ValueError("image-amplification input mode is not selected")
        self.sync_channels()
        if self.pump_channel.currentData() is None:
            raise ValueError("select a valid enabled pump channel")
        if self.signal_channel.currentData() is None:
            raise ValueError("select a valid enabled signal channel")
        request = PRImageAmplificationExperimentRequest(
            base_workflow_id=base_workflow_id,
            base_request=base_request,
            launch_configuration=launch_configuration,
            pump_channel_index=int(self.pump_channel.currentData()),
            signal_channel_index=int(self.signal_channel.currentData()),
        )
        request.validate()
        return request


__all__ = [
    "DecodedPRImage",
    "PR_GAUSSIAN_INPUT_MODE",
    "PR_IMAGE_AMPLIFICATION_INPUT_MODE",
    "PRImageInputPanel",
    "decode_user_image",
    "supported_user_image_formats",
]
