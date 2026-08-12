"""Compatibility ingestion for pre-Stage-5 beam channel mappings."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

from lcprop.core.beams import BeamChannel


_LEGACY_THETA_WEIGHT_ERROR = (
    "legacy non-unit theta_weight is unsupported after the material-source "
    "boundary migration; recreate the request without weighted LC sources"
)


def canonical_beam_channel_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return canonical optical channel values from a legacy mapping.

    Missing or unit-valued historical ``theta_weight`` metadata is discarded.
    A non-unit value is rejected explicitly because silently discarding it
    would change the historical LC source semantics.
    """

    canonical = dict(values)
    legacy_weight = canonical.pop("theta_weight", 1.0)
    if (
        isinstance(legacy_weight, bool)
        or not isinstance(legacy_weight, Real)
        or float(legacy_weight) != 1.0
    ):
        raise ValueError(_LEGACY_THETA_WEIGHT_ERROR)
    return canonical


def beam_channel_from_mapping(values: Mapping[str, Any]) -> BeamChannel:
    """Construct a canonical channel from current or legacy serialized data."""

    return BeamChannel(**canonical_beam_channel_values(values))


def discard_legacy_unit_theta_weight(options: dict[str, Any]) -> None:
    """Consume the removed LaunchPane keyword when it is exactly unit-valued."""

    if not options:
        return
    unexpected = set(options) - {"theta_weight"}
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise TypeError(f"unexpected compatibility option(s): {names}")
    canonical_beam_channel_values({"theta_weight": options["theta_weight"]})


__all__ = [
    "beam_channel_from_mapping",
    "canonical_beam_channel_values",
    "discard_legacy_unit_theta_weight",
]
