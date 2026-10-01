"""Large-grid presentation tests; no transverse scientific arrays are allocated."""
from dataclasses import replace
import math
import os
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from lcprop.core.context import GridSpec
from lcprop.core.grid import round_nz
from lcprop.pr.gui.grid_panel import (
    PR_DEFAULT_GRID, PR_GUI_MAX_TRANSVERSE_SAMPLES, PRGridPanel,
)
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.runtime_estimator import estimate_pr_resources
from lcprop.pr.scattering import PRCanonicalScatteringSpec, PR_CANONICAL_SCATTERING_V2
from lcprop.pr.specs import PRRunRequest, PR_TIMEDEPENDENT_WORKFLOW
from lcprop.pr.static_workflow import PRStaticRunRequest, PR_STATIC_WORKFLOW
from lcprop.pr.static_transport_codec import (
    encode_pr_static_transport_request, decode_pr_static_transport_request,
)
from lcprop.pr.timedependent_transport_codec import (
    encode_pr_timedependent_transport_request, decode_pr_timedependent_transport_request,
)
from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def no_runtime_grid(monkeypatch):
    import lcprop.core.grid as grids
    import lcprop.pr.gui.request_adapter as adapter
    def forbidden(*args, **kwargs):
        pytest.fail("ordinary presentation attempted runtime grid/FFT allocation")
    monkeypatch.setattr(grids, "make_grid", forbidden)
    monkeypatch.setattr(adapter, "make_grid", forbidden, raising=False)
    original = np.fft.fftfreq
    def bounded_preview_only(n, *args, **kwargs):
        # LaunchPlane independently renders a fixed 128² visual preview. It must
        # never adopt the scientific request dimensions; do not fake that preview.
        assert n <= 128, "presentation allocated request-sized frequencies"
        return original(n, *args, **kwargs)
    monkeypatch.setattr(np.fft, "fftfreq", bounded_preview_only)


@pytest.mark.parametrize("shape", [(8192, 4096), (4096, 8192), (1_000_000, 8192), (2**31 - 1, 2)])
@pytest.mark.parametrize("workflow,model", [
    (PR_STATIC_WORKFLOW, "field_linear_local_intensity"),
    (PR_STATIC_WORKFLOW, "nonlinear"),
    (PR_TIMEDEPENDENT_WORKFLOW, "nonlinear"),
])
def test_large_grid_request_inspection_persistence_transport_and_planning(
    app, no_runtime_grid, tmp_path, shape, workflow, model,
):
    window = PRMainWindow()
    try:
        grid = replace(PR_DEFAULT_GRID, Nx=shape[0], Ny=shape[1],
                       x_aperture_um=1000.0, y_aperture_um=1000.0,
                       z_length_um=4000.0, dz_um=50.0)
        window.grid_panel.set_grid(grid)
        panel = window.evolution_panel
        panel.set_workflow_id(workflow)
        choices = tuple(panel.material_response.itemData(i) for i in range(panel.material_response.count()))
        assert choices == (("nonlinear", "field_linear_local_intensity")
                           if workflow == PR_STATIC_WORKFLOW else ("nonlinear",))
        panel.set_transverse_material_response(PRTransverseMaterialResponseSpec(
            model=model,
        ), applied_field_x=0.0)
        assert panel.reference_intensity.isHidden()
        scattering = PRCanonicalScatteringSpec(
            0.02, 0.4, 0, 1.0, PR_CANONICAL_SCATTERING_V2,
        )
        panel.set_scattering_spec(scattering)
        request = window.build_request()
        assert request.grid == grid
        assert request.scattering == scattering
        assert request.material_response.model == model
        assert isinstance(request, PRStaticRunRequest if workflow == PR_STATIC_WORKFLOW else PRRunRequest)
        assert f"Grid: {shape[0]} × {shape[1]}, Nz=80" in window.describe_request(request)

        path = tmp_path / "large-grid.json"
        window.save_experiment_to(path)
        window.grid_panel.set_grid(PR_DEFAULT_GRID)
        window.load_experiment_from(path)
        assert window.build_request() == request

        encode, decode = (
            (encode_pr_static_transport_request, decode_pr_static_transport_request)
            if workflow == PR_STATIC_WORKFLOW else
            (encode_pr_timedependent_transport_request, decode_pr_timedependent_transport_request)
        )
        encoded = encode(request)
        assert not encoded.payload.arrays
        assert decode(encoded.payload.metadata, encoded.payload.arrays) == request

        estimate = estimate_pr_resources(request)
        assert estimate.grid_shape == (*shape, 80)
        assert math.isfinite(estimate.peak_gpu_memory.high)
        assert estimate.peak_gpu_memory.high > 0
        assert estimate.full_result_size.high > 2**32  # byte estimates are not uint32
        assert math.isfinite(estimate.full_result_size.high)
        window.resource_estimator_panel.estimate_button.click()
        text = window.resource_estimator_panel.output.toPlainText()
        assert f"Grid: {shape[0]} × {shape[1]} × 80" in text
        assert "Estimated peak GPU memory:" in text
        assert window.run_status == "idle" and window.last_result is None
        from lcprop.pr.gui.request_adapter import validate_pr_gui_request
        if workflow == PR_TIMEDEPENDENT_WORKFLOW and model == "nonlinear" and shape[0] > 4096:
            preflight = validate_pr_gui_request(request)
            assert preflight.timestep_assessment == "pending"
            assert preflight.conservative_dt_limit is None
            assert "assessment pending" in text
            assert "assessment pending" in window.describe_request(request)
    finally:
        window.close()


def test_grid_defaults_and_direct_widget_entry(app):
    panel = PRGridPanel()
    assert panel.grid() == PR_DEFAULT_GRID
    assert (panel.Nx.minimum(), panel.Ny.minimum()) == (2, 2)
    assert (panel.Nx.maximum(), panel.Ny.maximum()) == (PR_GUI_MAX_TRANSVERSE_SAMPLES,) * 2
    panel.Nx.setValue(8192)
    panel.Ny.setValue(4096)
    assert (panel.grid().Nx, panel.grid().Ny) == (8192, 4096)
    for size in (2, 128, 4096):
        grid = replace(PR_DEFAULT_GRID, Nx=size, Ny=size)
        panel.set_grid(grid)
        assert panel.grid() == grid


@pytest.mark.parametrize("dimension", ["Nx", "Ny"])
@pytest.mark.parametrize("value", [0, 1, PR_GUI_MAX_TRANSVERSE_SAMPLES + 1])
def test_invalid_or_unrepresentable_grid_is_rejected_without_clamping(app, dimension, value):
    panel = PRGridPanel()
    before = panel.grid()
    grid = replace(before, **{dimension: value})
    with pytest.raises(ValueError):
        panel.set_grid(grid)
    assert panel.grid() == before


def test_td_presentation_never_scans_large_nonlinear_and_rejects_field_linear(app, no_runtime_grid, monkeypatch):
    import lcprop.pr.gui.request_adapter as adapter
    window = PRMainWindow()
    try:
        window.grid_panel.set_grid(replace(PR_DEFAULT_GRID, Nx=1_000_000, Ny=8192))
        def forbidden(*args, **kwargs):
            pytest.fail("presentation enumerated nonlinear modes")
        monkeypatch.setattr(adapter, "validate_timestep", forbidden)
        window.evolution_panel.set_workflow_id(PR_TIMEDEPENDENT_WORKFLOW)
        panel = window.evolution_panel
        assert panel.material_response.count() == 1
        assert panel.material_response.currentData() == "nonlinear"
        assert panel.reference_intensity.isHidden()
        request = window.build_request()
        assert adapter.validate_pr_gui_request(request).timestep_assessment == "pending"
        for model in ("linearized", "field_linear_local_intensity"):
            assert panel.material_response.findData(model) == -1
            response = PRTransverseMaterialResponseSpec(
                model=model, reference_intensity=1.0 if model == "linearized" else None,
            )
            with pytest.raises(ValueError, match="unsupported"):
                panel.set_transverse_material_response(response, applied_field_x=0.0)
            with pytest.raises(ValueError, match="Reduced TD supports nonlinear hopping only"):
                adapter.validate_pr_gui_request(replace(request, material_response=response))
            assert window.build_request() == request
    finally:
        window.close()


@pytest.mark.parametrize("shape", [(8192, 4096), (4096, 8192), (1_000_000, 8192), (2**31 - 1, 2)])
def test_large_grid_workflow_switch_preserves_model_boundaries(app, no_runtime_grid, shape):
    window = PRMainWindow()
    try:
        window.grid_panel.set_grid(replace(PR_DEFAULT_GRID, Nx=shape[0], Ny=shape[1]))
        panel = window.evolution_panel
        panel.set_workflow_id(PR_STATIC_WORKFLOW)
        panel.set_transverse_material_response(PRTransverseMaterialResponseSpec(
            model="field_linear_local_intensity"), applied_field_x=0.)
        assert window.build_request().material_response.model == "field_linear_local_intensity"
        assert panel.reference_intensity.isHidden()
        with pytest.raises(ValueError, match="retired"):
            panel.set_transverse_material_response(PRTransverseMaterialResponseSpec(
                model="linearized", reference_intensity=1.), applied_field_x=0.)
        panel.set_workflow_id(PR_TIMEDEPENDENT_WORKFLOW)
        assert panel.material_response.count() == 1
        assert window.build_request().material_response.model == "nonlinear"
        assert panel.reference_intensity.isHidden()
        for workflow in ("pr_transverse_static", "pr_transverse_timedependent"):
            panel.set_workflow_id(workflow)
            assert tuple(panel.material_response.itemData(i) for i in range(panel.material_response.count())) == ("nonlinear", "linearized")
            panel.set_transverse_material_response(PRTransverseMaterialResponseSpec(
                model="linearized", reference_intensity=1.), applied_field_x=0.)
            assert not panel.reference_intensity.isHidden()
            assert window.build_request().material_response.model == "linearized"
        panel.set_workflow_id(PR_STATIC_WORKFLOW)
        assert panel.material_response.findData("linearized") == -1
        assert panel.material_response.findData("field_linear_local_intensity") >= 0
        assert panel.reference_intensity.isHidden()
    finally:
        window.close()


def test_pending_presentation_does_not_waive_execution_timestep_guard(app, monkeypatch):
    import lcprop.pr.gui.request_adapter as adapter
    from lcprop.pr.workflow import run_pr_timedependent
    window = PRMainWindow()
    try:
        window.evolution_panel.set_workflow_id(PR_TIMEDEPENDENT_WORKFLOW)
        request = window.build_request()
        request = replace(request, grid=replace(request.grid, Nx=8, Ny=8, z_length_um=request.grid.dz_um),
                          solver=replace(request.solver, dt_normalized=1e6))
        monkeypatch.setattr(adapter, "PR_GUI_TIMESTEP_SCAN_MAX_MODES", 4)
        assert adapter.validate_pr_gui_request(request).timestep_assessment == "pending"
        with pytest.raises(ValueError, match="exceeds conservative PR limit"):
            run_pr_timedependent(request)
    finally:
        window.close()
