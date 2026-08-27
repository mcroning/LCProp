"""Material-neutral launch-plane raster screens and passive transforms.

Raster samples are source data, not simulation-grid data.  They are prepared
on an explicitly supplied :class:`~lcprop.core.grid.RuntimeGrid` only when the
launch is built.  Passive elements act after incident channel normalization and
never renormalize the transformed field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Literal

import numpy as np
from scipy.ndimage import zoom

from lcprop.core.backend import asnumpy
from lcprop.core.grid import RuntimeGrid


EVEN_SQUARE_NEAREST_TRANSPARENT_V1 = "even_square_nearest_transparent_v1"
RASTER_SOURCE_IDENTITY_V1 = "raster_source_identity_v1"

ScreenBoundaryPolicy = Literal["reject", "clip"]
ScreenResampling = Literal["nearest"]


@dataclass(frozen=True, eq=False)
class RasterSource:
    """Immutable decoded raster samples plus portable source provenance.

    ``grayscale`` uses conventional image indexing ``(row_y, column_x)``. Source
    dimensions do not define a numerical grid, physical aperture, or any
    longitudinal discretization.
    """

    source_kind: str
    display_name: str
    basename: str
    sha256: str
    width: int
    height: int
    encoded_format: str
    decoded_mode: str
    grayscale: Any
    preprocessing_policy: str = RASTER_SOURCE_IDENTITY_V1
    asset_id: str | None = None
    encoded_bytes: bytes | None = None

    def __post_init__(self) -> None:
        if self.source_kind not in ("standard", "user", "array"):
            raise ValueError("source_kind must be standard, user, or array")
        for name in (
            "display_name",
            "basename",
            "sha256",
            "decoded_mode",
            "preprocessing_policy",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("sha256 must be lowercase hexadecimal")
        pixels = np.asarray(self.grayscale)
        if pixels.ndim != 2 or pixels.dtype.kind not in "fiu":
            raise TypeError("grayscale must be a real two-dimensional array")
        if pixels.shape != (int(self.height), int(self.width)):
            raise ValueError("grayscale shape must match source width and height")
        if not np.all(np.isfinite(pixels)) or np.any(pixels < 0):
            raise ValueError("grayscale must be finite and nonnegative")
        encoded = self.encoded_bytes
        if encoded is not None and not isinstance(encoded, bytes):
            raise TypeError("encoded_bytes must be bytes or None")
        if self.source_kind == "user" and encoded is not None:
            if hashlib.sha256(encoded).hexdigest() != self.sha256:
                raise ValueError("encoded_bytes do not match source sha256")
        owned = pixels.copy()
        owned.flags.writeable = False
        object.__setattr__(self, "grayscale", owned)

    def __eq__(self, other) -> bool:
        if not isinstance(other, RasterSource):
            return NotImplemented
        scalar_names = (
            "source_kind",
            "display_name",
            "basename",
            "sha256",
            "width",
            "height",
            "encoded_format",
            "decoded_mode",
            "preprocessing_policy",
            "asset_id",
            "encoded_bytes",
        )
        return all(
            getattr(self, name) == getattr(other, name) for name in scalar_names
        ) and np.array_equal(self.grayscale, other.grayscale)

    @classmethod
    def from_array(cls, image_intensity, *, display_name: str = "array input"):
        """Create a deterministic source from a finite nonnegative 2-D array."""

        supplied = np.asarray(image_intensity)
        if supplied.ndim != 2 or supplied.dtype.kind not in "fiu":
            raise TypeError("image_intensity must be a real two-dimensional array")
        values = supplied.astype(np.float64, copy=False)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("image_intensity must be finite and nonnegative")
        maximum = float(np.max(values))
        if not math.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("image_intensity must contain a positive value")
        pixels = (values / maximum).astype(np.float64, copy=True)
        identity = (
            f"shape={pixels.shape};dtype=float64;".encode("ascii")
            + pixels.tobytes(order="C")
        )
        return cls(
            source_kind="array",
            display_name=display_name,
            basename="<array>",
            sha256=hashlib.sha256(identity).hexdigest(),
            width=pixels.shape[1],
            height=pixels.shape[0],
            encoded_format="array",
            decoded_mode="L",
            grayscale=pixels,
        )


@dataclass(frozen=True)
class ScreenPlacement:
    """Physical placement and sampling policy for one launch-plane raster."""

    center_x_um: float = 0.0
    center_y_um: float = 0.0
    width_um: float = 1.0
    height_um: float = 1.0
    resampling: ScreenResampling = "nearest"
    outside_intensity_transmission: float = 1.0
    boundary_policy: ScreenBoundaryPolicy = "reject"

    def validate(self) -> None:
        for name in ("center_x_um", "center_y_um"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        for name in ("width_um", "height_um"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.resampling != "nearest":
            raise ValueError("resampling must be 'nearest'")
        outside = float(self.outside_intensity_transmission)
        if not math.isfinite(outside) or not 0.0 <= outside <= 1.0:
            raise ValueError(
                "outside_intensity_transmission must lie between zero and one"
            )
        if self.boundary_policy not in ("reject", "clip"):
            raise ValueError("boundary_policy must be 'reject' or 'clip'")


@dataclass(frozen=True)
class IntensityRasterScreen:
    """Interpret one source raster as passive intensity transmission ``T``."""

    source: RasterSource
    placement: ScreenPlacement
    invert: bool = False
    preprocessing_policy: str = EVEN_SQUARE_NEAREST_TRANSPARENT_V1

    def validate(self) -> None:
        if not isinstance(self.source, RasterSource):
            raise TypeError("source must be a RasterSource")
        self.placement.validate()
        if self.preprocessing_policy != EVEN_SQUARE_NEAREST_TRANSPARENT_V1:
            raise ValueError("unsupported intensity-raster preprocessing policy")


@dataclass(frozen=True)
class ChannelLaunchElements:
    """Ordered launch-plane elements assigned to one canonical channel index."""

    channel_index: int
    elements: tuple[IntensityRasterScreen, ...] = field(default_factory=tuple)

    def validate(self, *, n_channels: int) -> None:
        if type(self.channel_index) is not int:
            raise TypeError("channel_index must be an integer")
        if not 0 <= self.channel_index < int(n_channels):
            raise ValueError(
                "channel_index must identify a canonical enabled beam channel"
            )
        if not isinstance(self.elements, tuple):
            raise TypeError("elements must be an ordered tuple")
        for element in self.elements:
            if not isinstance(element, IntensityRasterScreen):
                raise TypeError("B1 channel elements must be IntensityRasterScreen")
            element.validate()


def validate_channel_launch_elements(
    assignments: tuple[ChannelLaunchElements, ...],
    *,
    n_channels: int,
) -> None:
    """Validate canonical channel indices and deterministic assignment order."""

    if not isinstance(assignments, tuple):
        raise TypeError("launch_elements must be a tuple")
    seen: set[int] = set()
    for assignment in assignments:
        if not isinstance(assignment, ChannelLaunchElements):
            raise TypeError("launch_elements must contain ChannelLaunchElements")
        assignment.validate(n_channels=n_channels)
        if assignment.channel_index in seen:
            raise ValueError("each channel may have only one ordered element assignment")
        seen.add(assignment.channel_index)


def _normalized_intensity_raster(samples) -> np.ndarray:
    supplied = np.asarray(samples)
    if supplied.ndim != 2:
        raise ValueError("intensity raster must be a two-dimensional array")
    if supplied.dtype.kind not in "fiu":
        raise TypeError("intensity raster must have a real numeric dtype")
    image = supplied.astype(np.float64, copy=True)
    if not np.all(np.isfinite(image)):
        raise ValueError("intensity raster must contain only finite values")
    if np.any(image < 0.0):
        raise ValueError("intensity raster must be nonnegative")
    maximum = float(np.max(image))
    if maximum <= 0.0:
        raise ValueError("intensity raster must contain a positive value")
    return image / maximum


def prepare_intensity_raster_transmission(
    samples,
    grid: RuntimeGrid,
    *,
    placement: ScreenPlacement,
    invert: bool = False,
    preprocessing_policy: str = EVEN_SQUARE_NEAREST_TRANSPARENT_V1,
) -> np.ndarray:
    """Prepare a passive intensity raster on an explicitly selected grid.

    The version-1 policy preserves the established image-amplification order:
    max normalization, optional inversion, even-square padding, image-to-``xy``
    rotation, nearest-neighbor resampling, and explicit exterior transmission.
    """

    placement.validate()
    if preprocessing_policy != EVEN_SQUARE_NEAREST_TRANSPARENT_V1:
        raise ValueError("unsupported intensity-raster preprocessing policy")
    image = _normalized_intensity_raster(samples)
    if bool(invert):
        image = 1.0 - image

    side = max(image.shape)
    if side % 2:
        side += 1
    outside = float(placement.outside_intensity_transmission)
    square = np.full((side, side), outside, dtype=np.float64)
    y_offset = (side - image.shape[0]) // 2
    x_offset = (side - image.shape[1]) // 2
    square[
        y_offset : y_offset + image.shape[0],
        x_offset : x_offset + image.shape[1],
    ] = image
    image_xy = np.rot90(square)

    target_nx = max(2, int(round(float(placement.width_um) / float(grid.dx_um))))
    target_ny = max(2, int(round(float(placement.height_um) / float(grid.dy_um))))
    resized = zoom(
        image_xy,
        (target_nx / image_xy.shape[0], target_ny / image_xy.shape[1]),
        order=0,
        mode="nearest",
        prefilter=False,
    )
    resized = resized[:target_nx, :target_ny]

    x_values = np.asarray(asnumpy(grid.x_um))
    y_values = np.asarray(asnumpy(grid.y_um))
    center_x = int(np.argmin(np.abs(x_values - float(placement.center_x_um))))
    center_y = int(np.argmin(np.abs(y_values - float(placement.center_y_um))))
    destination = np.full((grid.Nx, grid.Ny), outside, dtype=np.float64)
    x_start = center_x - resized.shape[0] // 2
    y_start = center_y - resized.shape[1] // 2
    x_stop = x_start + resized.shape[0]
    y_stop = y_start + resized.shape[1]
    if placement.boundary_policy == "reject" and (
        x_start < 0
        or y_start < 0
        or x_stop > grid.Nx
        or y_stop > grid.Ny
    ):
        raise ValueError(
            "screen footprint extends outside the simulation aperture; "
            "increase the aperture, reduce screen size, or adjust placement"
        )
    destination_x0 = max(0, x_start)
    destination_y0 = max(0, y_start)
    destination_x1 = min(grid.Nx, x_stop)
    destination_y1 = min(grid.Ny, y_stop)
    if destination_x0 >= destination_x1 or destination_y0 >= destination_y1:
        raise ValueError("screen lies outside the transverse grid")
    source_x0 = destination_x0 - x_start
    source_y0 = destination_y0 - y_start
    source_x1 = source_x0 + destination_x1 - destination_x0
    source_y1 = source_y0 + destination_y1 - destination_y0
    destination[
        destination_x0:destination_x1,
        destination_y0:destination_y1,
    ] = resized[source_x0:source_x1, source_y0:source_y1]
    return destination


def prepare_intensity_raster_screen(
    screen: IntensityRasterScreen,
    grid: RuntimeGrid,
) -> np.ndarray:
    """Return the runtime-grid intensity transmission for ``screen``."""

    if not isinstance(screen, IntensityRasterScreen):
        raise TypeError("screen must be an IntensityRasterScreen")
    screen.validate()
    return prepare_intensity_raster_transmission(
        screen.source.grayscale,
        grid,
        placement=screen.placement,
        invert=screen.invert,
        preprocessing_policy=screen.preprocessing_policy,
    )


def intensity_transmission_to_field_transmittance(
    intensity_transmission,
) -> np.ndarray:
    """Map passive intensity transmission ``T`` to field multiplier ``sqrt(T)``."""

    transmission = np.asarray(intensity_transmission)
    if transmission.ndim != 2 or transmission.dtype.kind not in "fiu":
        raise TypeError("intensity transmission must be a real two-dimensional array")
    values = transmission.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise ValueError("intensity transmission must be finite")
    tolerance = 32.0 * np.finfo(values.dtype).eps
    if np.any(values < -tolerance) or np.any(values > 1.0 + tolerance):
        raise ValueError("passive intensity transmission must lie between zero and one")
    return np.sqrt(np.clip(values, 0.0, 1.0))


def apply_passive_field_transmittance(
    field,
    field_transmittance,
    *,
    xp=None,
):
    """Apply ``A_out=t*A_in`` with passive ``|t|<=1`` and no renormalization."""

    namespace = np if xp is None else xp
    supplied_field = namespace.asarray(field)
    transmittance = namespace.asarray(field_transmittance)
    if supplied_field.ndim != 2 or transmittance.shape != supplied_field.shape:
        raise ValueError("field and field transmittance must have the same 2-D shape")
    if not bool(np.asarray(asnumpy(namespace.all(namespace.isfinite(transmittance))))):
        raise ValueError("field transmittance must be finite")
    precision_dtype = (
        np.dtype(transmittance.real.dtype)
        if transmittance.dtype.kind in "fc"
        else np.dtype(np.float64)
    )
    tolerance = 64.0 * np.finfo(precision_dtype).eps
    passive = namespace.all(namespace.abs(transmittance) <= 1.0 + tolerance)
    if not bool(np.asarray(asnumpy(passive))):
        raise ValueError("passive field transmittance magnitude must not exceed one")
    return supplied_field * transmittance


def apply_channel_launch_elements(
    A0,
    grid: RuntimeGrid,
    assignments: tuple[ChannelLaunchElements, ...],
):
    """Apply ordered per-channel raster elements to a copied launch stack."""

    validate_channel_launch_elements(assignments, n_channels=int(A0.shape[0]))
    if not assignments:
        return A0
    xp = grid.xp
    transformed = A0.copy()
    for assignment in assignments:
        index = assignment.channel_index
        for element in assignment.elements:
            intensity = prepare_intensity_raster_screen(element, grid)
            field_transmittance = intensity_transmission_to_field_transmittance(
                intensity
            )
            transformed[index] = apply_passive_field_transmittance(
                transformed[index],
                xp.asarray(field_transmittance, dtype=grid.real_dtype),
                xp=xp,
            )
    return transformed


__all__ = [
    "ChannelLaunchElements",
    "EVEN_SQUARE_NEAREST_TRANSPARENT_V1",
    "IntensityRasterScreen",
    "RASTER_SOURCE_IDENTITY_V1",
    "RasterSource",
    "ScreenBoundaryPolicy",
    "ScreenPlacement",
    "ScreenResampling",
    "apply_channel_launch_elements",
    "apply_passive_field_transmittance",
    "intensity_transmission_to_field_transmittance",
    "prepare_intensity_raster_screen",
    "prepare_intensity_raster_transmission",
    "validate_channel_launch_elements",
]
