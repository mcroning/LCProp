from __future__ import annotations

import pytest

from lcprop.adapters.legacy_beams import beam_channel_from_mapping
from lcprop.core.beams import BeamChannel


def _channel_values() -> dict[str, object]:
    return {
        "name": "legacy",
        "power_mW": 2.0,
        "coherence_group": "laser",
    }


@pytest.mark.parametrize("legacy_weight", [None, 1, 1.0])
def test_missing_or_unit_legacy_theta_weight_is_canonicalized(legacy_weight):
    values = _channel_values()
    if legacy_weight is not None:
        values["theta_weight"] = legacy_weight

    channel = beam_channel_from_mapping(values)

    assert channel.name == "legacy"
    assert channel.power_mW == 2.0
    assert not hasattr(channel, "theta_weight")


@pytest.mark.parametrize(
    "legacy_weight",
    [0.0, -1.0, 0.5, 2.0, "1", "invalid", True, None],
)
def test_nonunit_legacy_theta_weight_is_rejected(legacy_weight):
    values = _channel_values()
    values["theta_weight"] = legacy_weight

    with pytest.raises(ValueError, match="legacy non-unit theta_weight"):
        beam_channel_from_mapping(values)


def test_removed_theta_weight_is_not_a_canonical_beam_keyword():
    with pytest.raises(TypeError, match="theta_weight"):
        BeamChannel(theta_weight=1.0)
