"""PR-owned raster source records and packaged standard-image catalog."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import resources
import json

from lcprop.optics.screens import RasterSource


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
class PRImageSource(RasterSource):
    """Compatibility name for a shared raster with the PR Stage-A policy."""

    preprocessing_policy: str = PR_IMAGE_PREPROCESSING_POLICY_V1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.preprocessing_policy != PR_IMAGE_PREPROCESSING_POLICY_V1:
            raise ValueError("unsupported PR image preprocessing policy")


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
