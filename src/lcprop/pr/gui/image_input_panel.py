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
        self._roles_initialized = False
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
                self._beam_panel.launch_plane_widget.beamStackChanged.disconnect(
                    self.sync_channels
                )
            except RuntimeError:
                pass
        self._beam_panel = beam_panel
        beam_panel.launch_plane_widget.beamStackChanged.connect(self.sync_channels)
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
        previous_names = tuple(self._role_names)
        enabled = [
            beam for beam in self._beam_panel.beam_stack_definition.beams
            if beam.enabled
        ]
        names = [str(beam.name) for beam in enabled]
        self._enabled_channel_names = names
        resolved_matches = [
            [index for index, name in enumerate(names) if name == previous]
            for previous in previous_names
        ]
        default_pair = len(names) == 2 and not self._roles_initialized
        for role, selector in enumerate((self.pump_channel, self.signal_channel)):
            selector.blockSignals(True)
            selector.clear()
            for index, name in enumerate(names):
                selector.addItem(f"{index} — {name}", index)
            previous = previous_names[role]
            matches = resolved_matches[role]
            if previous is not None and len(matches) == 1:
                selector.setCurrentIndex(matches[0])
                self._role_names[role] = previous
            elif default_pair:
                selector.setCurrentIndex(role)
                self._role_names[role] = names[role]
            else:
                selector.setCurrentIndex(-1)
                self._role_names[role] = None
            selector.blockSignals(False)
        if default_pair:
            self._roles_initialized = True
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


__all__ = [
    "DecodedPRImage",
    "PR_GAUSSIAN_INPUT_MODE",
    "PR_IMAGE_AMPLIFICATION_INPUT_MODE",
    "PRImageInputPanel",
    "decode_user_image",
    "supported_user_image_formats",
]
