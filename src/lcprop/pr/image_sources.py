"""PR-owned raster source records and packaged standard-image catalog."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import resources
import json
import math
from typing import Any

import numpy as np


PR_IMAGE_PREPROCESSING_POLICY_V1 = "pr_intensity_transparency_v1"
_CATALOG_PACKAGE = "lcprop.pr.assets.images"
_CATALOG_NAME = "catalog.json"


@dataclass(frozen=True)
class PRStandardImageAsset:
    """One checksummed, redistributable package image."""

    asset_id: str
    display_name: str
    resource_filename: str
    sha256: str
    width: int
    height: int
    decoded_mode: str
    default_invert: bool

    def validate(self) -> None:
        for name in ("asset_id", "display_name", "resource_filename"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be a non-empty trimmed string")
        if "/" in self.resource_filename or "\\" in self.resource_filename:
            raise ValueError("standard-image resource_filename must be a basename")
        if len(self.sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.sha256
        ):
            raise ValueError("standard-image sha256 must be lowercase hexadecimal")
        if int(self.width) < 1 or int(self.height) < 1:
            raise ValueError("standard-image dimensions must be positive")
        if not isinstance(self.decoded_mode, str) or not self.decoded_mode:
            raise ValueError("decoded_mode must be non-empty")


@dataclass(frozen=True, eq=False)
class PRImageSource:
    """Immutable decoded grayscale source plus portable provenance.

    ``grayscale`` uses conventional image indexing ``(row_y, column_x)`` and
    contains real nonnegative intensity samples. User-file decoding stores
    compact unsigned 8-bit samples; the array compatibility path retains
    normalized floating-point precision. It is scientific input, not a
    numerical-grid or material-state array.
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
    preprocessing_policy: str = PR_IMAGE_PREPROCESSING_POLICY_V1
    asset_id: str | None = None

    def __post_init__(self) -> None:
        if self.source_kind not in ("standard", "user", "array"):
            raise ValueError("source_kind must be standard, user, or array")
        for name in ("display_name", "basename", "sha256", "decoded_mode"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if len(self.sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.sha256
        ):
            raise ValueError("sha256 must be lowercase hexadecimal")
        if self.preprocessing_policy != PR_IMAGE_PREPROCESSING_POLICY_V1:
            raise ValueError("unsupported PR image preprocessing policy")
        pixels = np.asarray(self.grayscale)
        if pixels.ndim != 2 or pixels.dtype.kind not in "fiu":
            raise TypeError("grayscale must be a real two-dimensional array")
        if pixels.shape != (int(self.height), int(self.width)):
            raise ValueError("grayscale shape must match source width and height")
        if not np.all(np.isfinite(pixels)) or np.any(pixels < 0):
            raise ValueError("grayscale must be finite and nonnegative")
        owned = pixels.copy()
        owned.flags.writeable = False
        object.__setattr__(self, "grayscale", owned)

    def __eq__(self, other) -> bool:
        if not isinstance(other, PRImageSource):
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
        )
        return all(
            getattr(self, name) == getattr(other, name) for name in scalar_names
        ) and np.array_equal(
            self.grayscale,
            other.grayscale,
        )

    @classmethod
    def from_array(cls, image_intensity, *, display_name: str = "array input"):
        """Create a deterministic compatibility source from a real 2-D array."""

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


def standard_image_catalog() -> tuple[PRStandardImageAsset, ...]:
    """Load and validate the installed standard-image catalog."""

    payload = json.loads(
        resources.files(_CATALOG_PACKAGE).joinpath(_CATALOG_NAME).read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "assets"}:
        raise ValueError("invalid PR standard-image catalog structure")
    if payload["schema_version"] != 1 or not isinstance(payload["assets"], list):
        raise ValueError("unsupported PR standard-image catalog")
    assets = tuple(PRStandardImageAsset(**item) for item in payload["assets"])
    identifiers = set()
    for asset in assets:
        asset.validate()
        if asset.asset_id in identifiers:
            raise ValueError(f"duplicate standard-image asset_id {asset.asset_id!r}")
        identifiers.add(asset.asset_id)
        data = resources.files(_CATALOG_PACKAGE).joinpath(
            asset.resource_filename
        ).read_bytes()
        if hashlib.sha256(data).hexdigest() != asset.sha256:
            raise ValueError(
                f"standard-image checksum mismatch for {asset.asset_id!r}"
            )
    return assets


__all__ = [
    "PR_IMAGE_PREPROCESSING_POLICY_V1",
    "PRImageSource",
    "PRStandardImageAsset",
    "standard_image_catalog",
]
