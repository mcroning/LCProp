"""Portable PR-owned encoding for declarative optical launch elements."""

from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
from typing import Any, Mapping

import numpy as np

from lcprop.optics.screens import (
    ChannelLaunchElements,
    IntensityRasterScreen,
    RasterSource,
    ScreenPlacement,
    validate_channel_launch_elements,
)
from lcprop.pr.image_sources import (
    PR_IMAGE_PREPROCESSING_POLICY_V1,
    PRImageSource,
)


class PortableLaunchPayloadError(ValueError):
    """Raised when portable launch metadata is malformed or inconsistent."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PortableLaunchPayloadError(f"{name} must be a JSON object")
    return value


def _exact_keys(
    value: Mapping[str, Any], *, required: set[str], name: str
) -> None:
    missing = required - set(value)
    extra = set(value) - required
    if missing or extra:
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected " + ", ".join(sorted(extra)))
        raise PortableLaunchPayloadError(f"{name} has " + "; ".join(details))


def _strict_base64(value: Any, *, name: str) -> bytes:
    if not isinstance(value, str):
        raise PortableLaunchPayloadError(f"{name} must be a base64 string")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise PortableLaunchPayloadError(f"{name} is not valid base64") from exc


def encode_raster_source(source: RasterSource) -> dict[str, Any]:
    """Encode one immutable raster source without authoritative file paths."""

    if not isinstance(source, RasterSource):
        raise PortableLaunchPayloadError("image source must be a RasterSource")
    pixels = np.asarray(source.grayscale)
    if pixels.dtype.kind not in "fiu" or pixels.ndim != 2:
        raise PortableLaunchPayloadError(
            "image source pixels must be a real 2-D array"
        )
    raw = pixels.tobytes(order="C")
    encoded_bytes = source.encoded_bytes
    if source.source_kind == "user" and encoded_bytes is None:
        raise PortableLaunchPayloadError(
            "user image source lacks embedded encoded bytes; reload the source "
            "from its original file before saving or transporting it"
        )
    return {
        "source_kind": source.source_kind,
        "display_name": source.display_name,
        "basename": source.basename,
        "sha256": source.sha256,
        "width": source.width,
        "height": source.height,
        "encoded_format": source.encoded_format,
        "decoded_mode": source.decoded_mode,
        "preprocessing_policy": source.preprocessing_policy,
        "asset_id": source.asset_id,
        "alpha_policy": "discarded_not_an_optical_mask",
        "grayscale_dtype": pixels.dtype.str,
        "grayscale_shape": list(pixels.shape),
        "grayscale_sha256": hashlib.sha256(raw).hexdigest(),
        "grayscale_base64": base64.b64encode(raw).decode("ascii"),
        "encoded_bytes_base64": (
            None
            if encoded_bytes is None
            else base64.b64encode(encoded_bytes).decode("ascii")
        ),
    }


def decode_raster_source(value: Any) -> RasterSource:
    """Decode and checksum-verify one portable raster source."""

    payload = _mapping(value, name="image source")
    required = {
        "source_kind", "display_name", "basename", "sha256", "width",
        "height", "encoded_format", "decoded_mode", "preprocessing_policy",
        "asset_id", "alpha_policy", "grayscale_dtype", "grayscale_shape",
        "grayscale_sha256", "grayscale_base64", "encoded_bytes_base64",
    }
    _exact_keys(payload, required=required, name="image source")
    if payload["alpha_policy"] != "discarded_not_an_optical_mask":
        raise PortableLaunchPayloadError("unsupported image alpha policy")
    try:
        dtype = np.dtype(payload["grayscale_dtype"])
    except (TypeError, ValueError) as exc:
        raise PortableLaunchPayloadError("invalid image grayscale dtype") from exc
    shape = payload["grayscale_shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(item) is not int or item < 1 for item in shape)
    ):
        raise PortableLaunchPayloadError(
            "image grayscale_shape must contain two positive integers"
        )
    if dtype.kind not in "fiu":
        raise PortableLaunchPayloadError(
            "image grayscale dtype must be real numeric"
        )
    raw = _strict_base64(payload["grayscale_base64"], name="grayscale_base64")
    if hashlib.sha256(raw).hexdigest() != payload["grayscale_sha256"]:
        raise PortableLaunchPayloadError("image grayscale checksum mismatch")
    expected_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(raw) != expected_size:
        raise PortableLaunchPayloadError(
            "image grayscale byte length is inconsistent"
        )
    pixels = np.frombuffer(raw, dtype=dtype).reshape(tuple(shape)).copy()
    encoded_value = payload["encoded_bytes_base64"]
    encoded = (
        None
        if encoded_value is None
        else _strict_base64(encoded_value, name="encoded_bytes_base64")
    )
    if payload["source_kind"] == "user":
        if encoded is None:
            raise PortableLaunchPayloadError("user image is missing embedded bytes")
        if hashlib.sha256(encoded).hexdigest() != payload["sha256"]:
            raise PortableLaunchPayloadError(
                "embedded user-image checksum mismatch"
            )
    source_type = (
        PRImageSource
        if payload["preprocessing_policy"] == PR_IMAGE_PREPROCESSING_POLICY_V1
        else RasterSource
    )
    try:
        return source_type(
            source_kind=payload["source_kind"],
            display_name=payload["display_name"],
            basename=payload["basename"],
            sha256=payload["sha256"],
            width=payload["width"],
            height=payload["height"],
            encoded_format=payload["encoded_format"],
            decoded_mode=payload["decoded_mode"],
            grayscale=pixels,
            preprocessing_policy=payload["preprocessing_policy"],
            asset_id=payload["asset_id"],
            encoded_bytes=encoded,
        )
    except (TypeError, ValueError) as exc:
        raise PortableLaunchPayloadError(f"invalid image source: {exc}") from exc


def encode_launch_elements(
    assignments: tuple[ChannelLaunchElements, ...],
) -> list[dict[str, Any]]:
    """Encode ordered channel-associated launch elements."""

    encoded = []
    for assignment in assignments:
        encoded.append({
            "channel_index": assignment.channel_index,
            "elements": [
                {
                    "element_type": "intensity_raster",
                    "interpretation": "intensity_transmission",
                    "source": encode_raster_source(element.source),
                    "placement": asdict(element.placement),
                    "invert": element.invert,
                    "preprocessing_policy": element.preprocessing_policy,
                }
                for element in assignment.elements
            ],
        })
    return encoded


def decode_launch_elements(
    value: Any, *, n_channels: int
) -> tuple[ChannelLaunchElements, ...]:
    """Decode and validate ordered channel-associated launch elements."""

    if not isinstance(value, list):
        raise PortableLaunchPayloadError("launch_elements must be a JSON array")
    assignments = []
    for assignment_index, item in enumerate(value):
        assignment_name = f"launch_elements[{assignment_index}]"
        assignment = _mapping(item, name=assignment_name)
        _exact_keys(
            assignment,
            required={"channel_index", "elements"},
            name=assignment_name,
        )
        element_items = assignment["elements"]
        if not isinstance(element_items, list):
            raise PortableLaunchPayloadError(
                f"{assignment_name}.elements must be an array"
            )
        elements = []
        for element_index, item in enumerate(element_items):
            name = f"{assignment_name}.elements[{element_index}]"
            element = _mapping(item, name=name)
            _exact_keys(
                element,
                required={
                    "element_type", "interpretation", "source", "placement",
                    "invert", "preprocessing_policy",
                },
                name=name,
            )
            if element["element_type"] != "intensity_raster":
                raise PortableLaunchPayloadError(
                    f"{name} has unsupported element_type"
                )
            if element["interpretation"] != "intensity_transmission":
                raise PortableLaunchPayloadError(
                    f"{name} has unsupported interpretation"
                )
            placement = _mapping(element["placement"], name=f"{name}.placement")
            expected_placement = set(ScreenPlacement.__dataclass_fields__)
            _exact_keys(
                placement,
                required=expected_placement,
                name=f"{name}.placement",
            )
            try:
                elements.append(IntensityRasterScreen(
                    source=decode_raster_source(element["source"]),
                    placement=ScreenPlacement(**placement),
                    invert=element["invert"],
                    preprocessing_policy=element["preprocessing_policy"],
                ))
            except (TypeError, ValueError) as exc:
                raise PortableLaunchPayloadError(
                    f"invalid {name}: {exc}"
                ) from exc
        try:
            assignments.append(ChannelLaunchElements(
                channel_index=assignment["channel_index"],
                elements=tuple(elements),
            ))
        except (TypeError, ValueError) as exc:
            raise PortableLaunchPayloadError(
                f"invalid {assignment_name}: {exc}"
            ) from exc
    result = tuple(assignments)
    try:
        validate_channel_launch_elements(result, n_channels=n_channels)
    except (TypeError, ValueError) as exc:
        raise PortableLaunchPayloadError(f"invalid launch_elements: {exc}") from exc
    return result


__all__ = [
    "PortableLaunchPayloadError",
    "decode_launch_elements",
    "decode_raster_source",
    "encode_launch_elements",
    "encode_raster_source",
]
