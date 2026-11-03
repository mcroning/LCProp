"""Stationary entry-point and real LC Console physical-power contract regressions."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from lcprop.core.beams import BeamStack
from lcprop.lc.requests import SolitonRequest
from lcprop.lc.workflows import soliton, soliton_trans
from lcprop.lc.gui.main_window import LCPropMainWindow
from lcprop.lc.operations import LC_STATIC_OPERATION
from lcprop.lc.products import _result_power_diagnostics
from lcprop.runners.local import LocalRunner
from tests.test_all_workflows import make_base_static_request


RULE = "stationary LC eigenmodes require normal launch; tilted physical launch is not qualified"


def _invoke(entry, request):
    if entry == "ordinary":
        return soliton.run_soliton(request)
    if entry == "transverse":
        return soliton_trans.run_soliton(request)
    return soliton_trans.polish_soliton(request, SimpleNamespace(A=None, theta=None))


@pytest.mark.parametrize("entry", ["ordinary", "transverse", "polish"])
@pytest.mark.parametrize("angle", [1e-12, 0.3])
def test_stationary_entry_rejects_before_runtime_or_normalization(monkeypatch, entry, angle):
    base = make_base_static_request()
    beam = replace(base.beams.channels[0], theta_ext_rad=angle)
    request = SolitonRequest(base=replace(base, beams=BeamStack(channels=(beam,))))

    def forbidden(*args, **kwargs):
        pytest.fail("unsupported stationary launch reached runtime/normalization")

    for module in (soliton, soliton_trans):
        monkeypatch.setattr(module, "build_runtime_components", forbidden)
        monkeypatch.setattr(module, "_prepare_initial_A", forbidden)
        monkeypatch.setattr(module, "_normalize_power", forbidden)
    with pytest.raises(ValueError) as error:
        _invoke(entry, request)
    assert str(error.value) == RULE


@pytest.mark.parametrize("entry", ["ordinary", "transverse", "polish"])
def test_normal_incidence_validation_preserves_request_and_runtime_dispatch(monkeypatch, entry):
    request = SolitonRequest(base=make_base_static_request())
    before = repr(request)
    calls = []

    class RuntimeReached(Exception):
        pass

    def reached(base, **kwargs):
        calls.append((base, kwargs))
        raise RuntimeReached

    for module in (soliton, soliton_trans):
        monkeypatch.setattr(module, "build_runtime_components", reached)
    with pytest.raises(RuntimeReached):
        _invoke(entry, request)
    assert calls == [(request.base, {"theta_dt": 7.5e-4, "mobility": 1.0})]
    assert repr(request) == before


@pytest.mark.parametrize("channel_count", [1, 2])
def test_real_lc_result_console_reports_lineage_current_without_mutation(channel_count):
    app = QApplication.instance() or QApplication([])
    base = make_base_static_request()
    beam = replace(base.beams.channels[0], w1_um=8., w2_um=8., coherence_group="overlap")
    request = replace(base, beams=BeamStack(channels=tuple(
        replace(beam, name=f"beam {index}") for index in range(channel_count))))
    delivered = LocalRunner(operations=(LC_STATIC_OPERATION,)).run_registered(
        "lc", "static", request)
    result = delivered.result
    values = (result.physical_power_initial_mW, result.physical_power_final_mW)
    arrays = (result.A_initial.copy(), result.A_final.copy())
    group_current = result.launch_summary["power_normalization"][
        "coherent_group_scalar_flux"]["overlap"]["scalar_axial_current_mW"]
    # Identical in-phase fields: cross terms multiply the lineage sum by N.
    assert group_current == pytest.approx(channel_count * values[0])
    window = LCPropMainWindow()
    try:
        window._display_runner_result(delivered)
        text = window.results_panel.workspace.console.toPlainText()
        diagnostics = _result_power_diagnostics(result)
        for stage, value in zip(("initial", "final"), values):
            key = f"scalar_lineage_axial_current_{stage}_mW"
            assert f"{key}: {value:.8g}" in text
            assert diagnostics[key] == value
            assert f"physical_power_{stage}_mW:" not in text
        assert diagnostics["power_qualification"] in text
        assert "excludes coherent cross terms" in text
        assert "not anisotropic/vector Poynting power" in text
        assert "coherent_group_total" not in text
        assert values == (result.physical_power_initial_mW, result.physical_power_final_mW)
        np.testing.assert_array_equal(result.A_initial, arrays[0])
        np.testing.assert_array_equal(result.A_final, arrays[1])
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
