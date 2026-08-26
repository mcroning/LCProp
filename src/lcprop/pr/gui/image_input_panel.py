"""PR GUI controls for Gaussian and image-amplification launch modes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QImage, QImageReader, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from lcprop.pr.image_amplification import (
    PRImageAmplificationRunRequest,
    PRImageLaunchSpec,
)
from lcprop.pr.image_sources import PRImageSource, standard_image_catalog


PR_GAUSSIAN_INPUT_MODE = "gaussian_beams"
PR_IMAGE_AMPLIFICATION_INPUT_MODE = "image_amplification"
_MAX_SOURCE_PIXELS = 100_000_000
_MAX_SOURCE_BYTES = 512 * 1024 * 1024
_PREVIEW_SIZE = QSize(320, 220)
_RASTER_FORMATS = {
    "bmp",
    "jpeg",
    "jpg",
    "pgm",
    "png",
    "ppm",
    "tif",
    "tiff",
    "webp",
}


@dataclass(frozen=True)
class DecodedPRImage:
    source: PRImageSource
    preview: QImage


def supported_user_image_formats() -> tuple[str, ...]:
    available = {
        bytes(value).decode("ascii").lower()
        for value in QImageReader.supportedImageFormats()
    }
    return tuple(sorted(available & _RASTER_FORMATS))


def _rgba_bytes(image: QImage) -> np.ndarray:
    converted = image.convertToFormat(QImage.Format.Format_RGBA8888)
    height = converted.height()
    width = converted.width()
    stride = converted.bytesPerLine()
    raw = np.frombuffer(
        converted.constBits(),
        dtype=np.uint8,
        count=converted.sizeInBytes(),
    ).reshape(height, stride)
    return raw[:, : 4 * width].reshape(height, width, 4).copy()


def _pillow_compatible_grayscale(rgba: np.ndarray) -> np.ndarray:
    """Return Pillow ``convert('L')``-compatible BT.601 integer luminance."""

    rgb = np.asarray(rgba, dtype=np.uint32)[..., :3]
    gray = (
        19595 * rgb[..., 0]
        + 38470 * rgb[..., 1]
        + 7471 * rgb[..., 2]
        + 32768
    ) >> 16
    return gray.astype(np.uint8)


def decode_user_image(path: str | Path) -> DecodedPRImage:
    """Decode one bounded raster source without retaining its absolute path."""

    source_path = Path(path)
    try:
        size_bytes = source_path.stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot read image file: {exc}") from exc
    if size_bytes < 1 or size_bytes > _MAX_SOURCE_BYTES:
        raise ValueError("image file size is outside the supported bounded range")
    digest = hashlib.sha256()
    try:
        with source_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot read image file: {exc}") from exc
    reader = QImageReader(str(source_path))
    reader.setAutoTransform(True)
    encoded_format = bytes(reader.format()).decode("ascii", errors="replace").lower()
    if encoded_format not in supported_user_image_formats():
        raise ValueError(
            "unsupported image format; supported raster formats are "
            + ", ".join(supported_user_image_formats())
        )
    size = reader.size()
    if not size.isValid():
        raise ValueError("image dimensions could not be read")
    width, height = size.width(), size.height()
    if width < 1 or height < 1 or width * height > _MAX_SOURCE_PIXELS:
        raise ValueError(
            "decoded image dimensions exceed the 100-megapixel safety limit"
        )
    image = reader.read()
    if image.isNull():
        raise ValueError(f"could not decode image: {reader.errorString()}")
    width, height = image.width(), image.height()
    if width < 1 or height < 1 or width * height > _MAX_SOURCE_PIXELS:
        raise ValueError(
            "decoded image dimensions exceed the 100-megapixel safety limit"
        )
    grayscale = _pillow_compatible_grayscale(_rgba_bytes(image))

    preview_reader = QImageReader(str(source_path))
    preview_reader.setAutoTransform(True)
    preview_reader.setScaledSize(size.scaled(_PREVIEW_SIZE, Qt.KeepAspectRatio))
    preview = preview_reader.read()
    if preview.isNull():
        preview = image.scaled(
            _PREVIEW_SIZE,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    source = PRImageSource(
        source_kind="user",
        display_name=source_path.name,
        basename=source_path.name,
        sha256=digest.hexdigest(),
        width=width,
        height=height,
        encoded_format=encoded_format,
        decoded_mode="L",
        grayscale=grayscale,
    )
    return DecodedPRImage(source=source, preview=preview)


class PRImageInputPanel(QWidget):
    """Input-mode selector and bounded image-amplification launch controls."""

    modeChanged = Signal(str)
    configurationChanged = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source: PRImageSource | None = None
        self._catalog = standard_image_catalog()

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.input_mode = QComboBox()
        self.input_mode.addItem("Gaussian beams", PR_GAUSSIAN_INPUT_MODE)
        self.input_mode.addItem(
            "Image amplification", PR_IMAGE_AMPLIFICATION_INPUT_MODE
        )
        self.source_type = QComboBox()
        self.source_type.addItem("Standard image", "standard")
        self.source_type.addItem("User image", "user")
        self.standard_image = QComboBox()
        for asset in self._catalog:
            self.standard_image.addItem(asset.display_name, asset.asset_id)
        self.standard_status = QLabel()
        if not self._catalog:
            self.standard_image.setEnabled(False)
            self.standard_status.setText(
                "No packaged standard images: provenance approval is pending."
            )
            self.source_type.setCurrentIndex(self.source_type.findData("user"))
        self.choose_file = QPushButton("Choose image…")
        self.choose_file.clicked.connect(self._choose_user_image)
        self.selected_file = QLabel("No user image selected")
        self.preview = QLabel("Image preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(_PREVIEW_SIZE)
        self.preview.setMaximumHeight(_PREVIEW_SIZE.height())
        self.preview.setStyleSheet("border: 1px solid #888;")
        self.invert_image = QCheckBox("Invert intensity before padding")
        self.image_size_um = QDoubleSpinBox()
        self.image_size_um.setRange(1e-6, 1e7)
        self.image_size_um.setDecimals(6)
        self.image_size_um.setValue(12.0)
        self.pump_incident_power_mW = QDoubleSpinBox()
        self.pump_incident_power_mW.setRange(1e-12, 1e12)
        self.pump_incident_power_mW.setDecimals(9)
        self.pump_incident_power_mW.setValue(1.0)
        self.signal_incident_power_mW = QDoubleSpinBox()
        self.signal_incident_power_mW.setRange(1e-12, 1e12)
        self.signal_incident_power_mW.setDecimals(12)
        self.signal_incident_power_mW.setValue(1e-3)
        self.wavelength_um = QDoubleSpinBox()
        self.wavelength_um.setRange(1e-6, 1e4)
        self.wavelength_um.setDecimals(9)
        self.wavelength_um.setValue(0.633)
        self.beam_waist_x_um = QDoubleSpinBox()
        self.beam_waist_x_um.setRange(1e-6, 1e7)
        self.beam_waist_x_um.setDecimals(6)
        self.beam_waist_x_um.setValue(12.0)
        self.beam_waist_y_um = QDoubleSpinBox()
        self.beam_waist_y_um.setRange(1e-6, 1e7)
        self.beam_waist_y_um.setDecimals(6)
        self.beam_waist_y_um.setValue(12.0)
        self.positive_mode_index = QSpinBox()
        self.positive_mode_index.setRange(1, 1_000_000)
        self.positive_mode_index.setValue(2)

        form.addRow("Input mode", self.input_mode)
        form.addRow("Image source", self.source_type)
        form.addRow("Standard image", self.standard_image)
        form.addRow("Standard-image status", self.standard_status)
        form.addRow("User image", self.choose_file)
        form.addRow("Selected source", self.selected_file)
        form.addRow("Preview", self.preview)
        form.addRow("", self.invert_image)
        form.addRow("Image physical size (µm)", self.image_size_um)
        form.addRow("Pump incident power (mW)", self.pump_incident_power_mW)
        form.addRow("Signal incident power (mW)", self.signal_incident_power_mW)
        form.addRow("Wavelength (µm)", self.wavelength_um)
        form.addRow("Beam waist x (µm)", self.beam_waist_x_um)
        form.addRow("Beam waist y (µm)", self.beam_waist_y_um)
        form.addRow("Positive Fourier mode", self.positive_mode_index)
        layout.addLayout(form)
        self.grid_independence_note = QLabel(
            "Source pixels are resampled onto the explicitly selected Grid; "
            "they never set Nx or Ny."
        )
        self.grid_independence_note.setWordWrap(True)
        layout.addWidget(self.grid_independence_note)
        layout.addStretch(1)
        self._image_widgets = tuple(
            widget
            for widget in (
                self.source_type,
                self.standard_image,
                self.standard_status,
                self.choose_file,
                self.selected_file,
                self.preview,
                self.invert_image,
                self.image_size_um,
                self.pump_incident_power_mW,
                self.signal_incident_power_mW,
                self.wavelength_um,
                self.beam_waist_x_um,
                self.beam_waist_y_um,
                self.positive_mode_index,
                self.grid_independence_note,
            )
        )
        self.input_mode.currentIndexChanged.connect(self._mode_changed)
        self.source_type.currentIndexChanged.connect(self._source_type_changed)
        for widget in (
            self.image_size_um,
            self.pump_incident_power_mW,
            self.signal_incident_power_mW,
            self.wavelength_um,
            self.beam_waist_x_um,
            self.beam_waist_y_um,
            self.positive_mode_index,
        ):
            widget.valueChanged.connect(self.configurationChanged)
        self.invert_image.toggled.connect(self.configurationChanged)
        self._mode_changed()

    def mode_id(self) -> str:
        return str(self.input_mode.currentData())

    def is_image_amplification(self) -> bool:
        return self.mode_id() == PR_IMAGE_AMPLIFICATION_INPUT_MODE

    def _mode_changed(self, *_args) -> None:
        enabled = self.is_image_amplification()
        for widget in self._image_widgets:
            widget.setVisible(enabled)
        self._source_type_changed()
        self.modeChanged.emit(self.mode_id())
        self.configurationChanged.emit()

    def _source_type_changed(self, *_args) -> None:
        image_mode = self.is_image_amplification()
        standard = self.source_type.currentData() == "standard"
        self.standard_image.setEnabled(image_mode and standard and bool(self._catalog))
        self.choose_file.setEnabled(image_mode and not standard)
        self.configurationChanged.emit()

    def _choose_user_image(self) -> None:
        formats = " ".join(f"*.{value}" for value in supported_user_image_formats())
        path, _selected = QFileDialog.getOpenFileName(
            self,
            "Choose image-amplification source",
            "",
            f"Raster images ({formats});;All files (*)",
        )
        if path:
            self.load_user_image(path)

    def load_user_image(self, path: str | Path) -> PRImageSource:
        decoded = decode_user_image(path)
        self._source = decoded.source
        self.selected_file.setText(decoded.source.basename)
        pixmap = QPixmap.fromImage(decoded.preview)
        self.preview.setPixmap(
            pixmap.scaled(
                _PREVIEW_SIZE,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        self.configurationChanged.emit()
        return decoded.source

    @property
    def source(self) -> PRImageSource | None:
        return self._source

    def build_request(self, *, grid, material, solver, backend):
        if not self.is_image_amplification():
            raise ValueError("image-amplification input mode is not selected")
        if self.source_type.currentData() == "standard":
            raise ValueError(
                "no redistributable standard image is packaged; choose a user image"
            )
        if self._source is None:
            raise ValueError("choose a user image before running image amplification")
        request = PRImageAmplificationRunRequest(
            grid=grid,
            material=material,
            solver=solver,
            backend=backend,
            source=self._source,
            launch=PRImageLaunchSpec(
                wavelength_um=self.wavelength_um.value(),
                positive_mode_index=self.positive_mode_index.value(),
                beam_waist_x_um=self.beam_waist_x_um.value(),
                beam_waist_y_um=self.beam_waist_y_um.value(),
                image_physical_size_um=self.image_size_um.value(),
                pump_incident_power_mW=self.pump_incident_power_mW.value(),
                signal_incident_power_mW=self.signal_incident_power_mW.value(),
                invert_image=self.invert_image.isChecked(),
                require_full_footprint=True,
            ),
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
