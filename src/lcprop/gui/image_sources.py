"""Material-neutral bounded raster decoding for GUI source selectors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage, QImageReader

from lcprop.optics.screens import RasterSource


MAX_RASTER_SOURCE_PIXELS = 100_000_000
MAX_RASTER_SOURCE_BYTES = 512 * 1024 * 1024
RASTER_PREVIEW_SIZE = QSize(320, 220)
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
class DecodedRasterImage:
    """One validated shared raster source and its bounded GUI preview."""

    source: RasterSource
    preview: QImage


def supported_user_image_formats() -> tuple[str, ...]:
    """Return safe raster formats supported by the active Qt installation."""

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


def decode_user_raster(
    path: str | Path,
    *,
    source_factory: Callable[..., RasterSource] = RasterSource,
) -> DecodedRasterImage:
    """Decode a bounded raster without retaining its absolute source path."""

    source_path = Path(path)
    try:
        size_bytes = source_path.stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot read image file: {exc}") from exc
    if size_bytes < 1 or size_bytes > MAX_RASTER_SOURCE_BYTES:
        raise ValueError("image file size is outside the supported bounded range")

    digest = hashlib.sha256()
    encoded = bytearray()
    try:
        with source_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                encoded.extend(chunk)
    except OSError as exc:
        raise ValueError(f"cannot read image file: {exc}") from exc

    reader = QImageReader(str(source_path))
    reader.setAutoTransform(True)
    encoded_format = bytes(reader.format()).decode("ascii", errors="replace").lower()
    supported = supported_user_image_formats()
    if encoded_format not in supported:
        raise ValueError(
            "unsupported image format; supported raster formats are "
            + ", ".join(supported)
        )
    size = reader.size()
    if not size.isValid():
        raise ValueError("image dimensions could not be read")
    width, height = size.width(), size.height()
    if width < 1 or height < 1 or width * height > MAX_RASTER_SOURCE_PIXELS:
        raise ValueError(
            "decoded image dimensions exceed the 100-megapixel safety limit"
        )

    image = reader.read()
    if image.isNull():
        raise ValueError(f"could not decode image: {reader.errorString()}")
    width, height = image.width(), image.height()
    if width < 1 or height < 1 or width * height > MAX_RASTER_SOURCE_PIXELS:
        raise ValueError(
            "decoded image dimensions exceed the 100-megapixel safety limit"
        )
    grayscale = _pillow_compatible_grayscale(_rgba_bytes(image))

    preview_reader = QImageReader(str(source_path))
    preview_reader.setAutoTransform(True)
    preview_reader.setScaledSize(size.scaled(RASTER_PREVIEW_SIZE, Qt.KeepAspectRatio))
    preview = preview_reader.read()
    if preview.isNull():
        preview = image.scaled(
            RASTER_PREVIEW_SIZE,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
    source = source_factory(
        source_kind="user",
        display_name=source_path.name,
        basename=source_path.name,
        sha256=digest.hexdigest(),
        width=width,
        height=height,
        encoded_format=encoded_format,
        decoded_mode="L",
        grayscale=grayscale,
        encoded_bytes=bytes(encoded),
    )
    return DecodedRasterImage(source=source, preview=preview)


__all__ = [
    "DecodedRasterImage",
    "MAX_RASTER_SOURCE_BYTES",
    "MAX_RASTER_SOURCE_PIXELS",
    "RASTER_PREVIEW_SIZE",
    "decode_user_raster",
    "supported_user_image_formats",
]
