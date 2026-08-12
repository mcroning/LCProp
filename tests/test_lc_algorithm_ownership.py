"""Ownership and compatibility checks for relocated LC algorithms."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    (
        "theta_cn",
        "theta_cn_zcoupled",
        "theta_picard",
        "static_relax",
        "td_zmarch",
    ),
)
def test_legacy_algorithm_modules_export_canonical_lc_objects(module_name):
    canonical = importlib.import_module(f"lcprop.lc.{module_name}")
    legacy = importlib.import_module(f"lcprop.algorithms.{module_name}")

    assert legacy is canonical
    assert legacy.__all__ == canonical.__all__
    for name in canonical.__all__:
        canonical_value = getattr(canonical, name)
        assert getattr(legacy, name) is canonical_value
        value_module = getattr(canonical_value, "__module__", None)
        if value_module is not None and value_module != "typing":
            assert value_module == canonical.__name__


def test_shared_algorithm_kernels_remain_canonically_shared():
    for module_name in ("fft_y", "thomas", "thomas_fast"):
        module = importlib.import_module(f"lcprop.algorithms.{module_name}")
        for name in module.__all__:
            value_module = getattr(getattr(module, name), "__module__", None)
            if value_module is not None:
                assert value_module == module.__name__
