"""General PR configuration through native workers and optional image analysis."""
import math

from dataclasses import replace
import gc

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QThread
from PySide6.QtWidgets import QApplication
from launchplane.model import BeamDefinition, BeamStackDefinition

from lcprop.optics.launch import build_launch, OpticalLaunchContext
from lcprop.core.grid import make_grid
from lcprop.persistence import load_experiment
from lcprop.pr.checkpoint import validate_pr_continuation
from lcprop.pr.gui.image_input_panel import (
    PR_GAUSSIAN_INPUT_MODE, PR_IMAGE_AMPLIFICATION_INPUT_MODE,
    PR_IMAGE_ANALYSIS_INPUT_MODE,
)
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.image_sources import PRImageSource
from lcprop.pr.image_amplification import (
    PRImageAmplificationExperimentRequest, run_image_amplification_experiment,
)
from lcprop.pr.live_results import PRLiveSnapshot
from lcprop.pr.optional_image_analysis import (
    PRImageAnalysisSelection, add_optional_image_analysis,
)
from lcprop.pr.specs import PRRunRequest, PR_TIMEDEPENDENT_WORKFLOW
from tests.test_pr_gui_main_window import _wait_for
from tests.test_pr_gui_image_amplification import _configured_multi_algorithm_image_window
import lcprop.pr.optional_image_analysis as optional


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def windows(app):
    owned = []
    yield owned
    for w in owned:
        if w._background_running:
            w.stop_clicked()
            _wait_for(app, lambda: not w._background_running)
        w.close()
        w.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    gc.collect()


def mode(w, value):
    w.input_panel.input_mode.setCurrentIndex(w.input_panel.input_mode.findData(value))


def general_window(windows, n, screens=(), *, asymmetric=False, zero=False):
    w = PRMainWindow()
    windows.append(w)
    w.grid_panel.Nx.setValue(16)
    w.grid_panel.Ny.setValue(16)
    w.grid_panel.x_aperture_um.setValue(40)
    w.grid_panel.y_aperture_um.setValue(40)
    w.grid_panel.z_length_um.setValue(10)
    w.grid_panel.dz_um.setValue(5)
    w.evolution_panel.Nt.setValue(2)
    w.evolution_panel.dt_normalized.setValue(.001)
    w.evolution_panel.backend.setCurrentText("numpy")
    w.material_panel.dark_intensity.setValue(.4)
    w.material_panel.gain_length_product.setValue(.02)
    w.beam_panel.set_beam_stack_definition(BeamStackDefinition(beams=tuple(
        BeamDefinition(
            name=f'Beam {j}',
            power_mW=0.0 if zero and j == 0 else 1.0,
            coherence_group='group-a' if j % 2 == 0 else 'group-b',
            w1_um=10.0,
            w2_um=10.0,
            theta_ext_rad=math.asin(math.hypot(0.15 if j == 0 else -0.15 * j, 0.08 if asymmetric and j == 0 else 0.0) * 0.633 / (2 * math.pi)),
            phi_rad=math.atan2(0.08 if asymmetric and j == 0 else 0.0, 0.15 if j == 0 else -0.15 * j) % (2 * math.pi),
        ) for j in range(n)
    )))
    editor = w.beam_panel.input_screen_editor
    for j in screens:
        editor.channel.setCurrentIndex(j)
        raster = np.ones((4, 4))
        raster[1:3, 1:3] = 0.0
        editor.set_source(PRImageSource.from_array(raster))
        editor.width_um.setValue(12.)
        editor.height_um.setValue(12.)
    mode(w, PR_IMAGE_ANALYSIS_INPUT_MODE)
    return w


@pytest.mark.parametrize("n,screens,asymmetric,zero", [
    (1, (), False, False), (2, (), False, False),
    (3, (), False, False), (4, (), False, False),
    (1, (0,), False, False), (3, (2,), False, False),
    (4, (0, 2), False, False), (2, (0,), True, False),
    (2, (1,), False, True),
])
def test_real_gui_worker_dispatch_general_configs(
    app, windows, monkeypatch, n, screens, asymmetric, zero,
):
    w = general_window(windows, n, screens, asymmetric=asymmetric, zero=zero)
    expected = w.build_request()
    assert isinstance(expected, PRRunRequest)
    assert len(expected.beams.channels) == n
    assert tuple(e.channel_index for e in expected.launch_elements) == screens
    calls, live = [], []
    dispatch = w.local_runner.run_registered
    render = w.results_panel.workspace.set_run_data

    def observe_dispatch(material, workflow, request, **kwargs):
        calls.append((material, workflow, request, QThread.currentThread()))
        return dispatch(material, workflow, request, **kwargs)

    def observe_render(data, *, state=None):
        render(data, state=state)
        if state == "Current accepted state":
            ws = w.results_panel.workspace
            live.append((w.last_progress.latest_field_state,
                         ws.result_ownership.text(), len(data.fields),
                         ws.longitudinal_pane.field_selector.count()))

    monkeypatch.setattr(w.local_runner, "run_registered", observe_dispatch)
    monkeypatch.setattr(w.results_panel.workspace, "set_run_data", observe_render)
    w.run_clicked()
    _wait_for(app, lambda: not w._background_running)
    assert len(calls) == 1
    assert calls[0][:3] == ("pr", PR_TIMEDEPENDENT_WORKFLOW, expected)
    assert calls[0][3] != QThread.currentThread()
    assert w.run_status == "completed"
    assert w.last_runner_result.kind == PR_TIMEDEPENDENT_WORKFLOW
    assert w.last_result.status == "completed"
    analysis = w.last_runner_result.run_data.diagnostics["optional_image_analysis"].values
    assert analysis["status"] == "unavailable" and analysis["reason"]
    assert live and isinstance(live[0][0], PRLiveSnapshot)
    assert live[0][0].completed_steps == 1 < expected.solver.Nt
    assert live[0][1] == "Current accepted state" and live[0][2] and live[0][3]

    # The same ordinary request remains scientifically identical across modes.
    mode(w, PR_GAUSSIAN_INPUT_MODE)
    ordinary = w.build_request()
    assert ordinary == expected
    reference = dispatch("pr", PR_TIMEDEPENDENT_WORKFLOW, ordinary)
    for key in ("A_initial", "A_final", "E_final", "E_initial", "source_intensity_stack"):
        np.testing.assert_array_equal(getattr(w.last_result, key), getattr(reference.result, key))
    assert (w.last_runner_result.run_data.diagnostics["carrier_power"].values
            == reference.run_data.diagnostics["carrier_power"].values)
    validate_pr_continuation(ordinary, w.last_checkpoint)
    with pytest.raises(ValueError):
        validate_pr_continuation(
            replace(ordinary, material=replace(ordinary.material, gain_length_product=.5)),
            w.last_checkpoint,
        )

    grid = make_grid(ordinary.grid, real_dtype=np.float64)
    context = OpticalLaunchContext(grid, ordinary.material.refractive_index, ordinary.grid.z_length_um)
    prepared = build_launch(ordinary.beams, grid, context=context, complex_dtype=np.complex128,
                            launch_elements=ordinary.launch_elements)
    incident = build_launch(ordinary.beams, grid, context=context, complex_dtype=np.complex128)
    np.testing.assert_array_equal(w.last_result.A_initial, prepared.A0)
    if screens:
        assert np.sum(abs(prepared.A0)**2) < np.sum(abs(incident.A0)**2)
    for j in set(range(n)) - set(screens):
        np.testing.assert_array_equal(prepared.A0[j], incident.A0[j])


def valid_image_window(app, windows):
    w = _configured_multi_algorithm_image_window(app)
    windows.append(w)
    # Resolved, widely separated carriers with a transparent image allow an
    # exact legacy-analysis comparison without inventing separation criteria.
    beams = w.beam_panel.beam_stack_definition
    w.beam_panel.set_beam_stack_definition(replace(beams, beams=tuple(
        replace(b, profile='collimated_gaussian', w1_um=8., w2_um=8.,
                theta_ext_rad=math.asin(3*b.wavelength_um/(40*b.n_ext)),
                phi_rad=0. if j == 0 else math.pi)
        for j, b in enumerate(beams.beams)
    )))
    editor = w.beam_panel.input_screen_editor
    editor.channel.setCurrentIndex(1)
    editor.set_source(PRImageSource.from_array(np.ones((8, 8))))
    w.input_panel.set_role_indices(0, 1)
    return w


def test_optional_success_matches_historical_analysis_and_saved_contract(app, windows, tmp_path):
    w = valid_image_window(app, windows)
    historical = w.build_request()
    assert isinstance(historical, PRImageAmplificationExperimentRequest)
    old_path = tmp_path / "historical.json"
    w.save_experiment_to(old_path)
    old = run_image_amplification_experiment(w.local_runner, historical)
    assert old.result.analysis_status == "completed"
    mode(w, PR_IMAGE_ANALYSIS_INPUT_MODE)
    expected = w.build_request()
    w.run_clicked()
    _wait_for(app, lambda: not w._background_running)
    new = w.last_runner_result
    assert new.run_data.workflow == PR_TIMEDEPENDENT_WORKFLOW
    assert new.run_data.diagnostics["optional_image_analysis"].values["status"] == "completed"
    np.testing.assert_array_equal(new.result.A_final, old.result.run_result.A_final)
    assert (new.run_data.diagnostics["image_amplification_metrics"].values
            == old.run_data.diagnostics["image_amplification_metrics"].values)
    assert "signal_carrier_mask" in new.run_data.fields
    assert new.run_data.diagnostics["carrier_power"].values["carrier_gain"] == old.run_data.diagnostics["carrier_power"].values["carrier_gain"]

    new_path = tmp_path / "ordinary.json"
    w.save_experiment_to(new_path)
    assert load_experiment(new_path).request == expected
    w.load_experiment_from(new_path)
    assert w.input_panel.mode_id() == PR_GAUSSIAN_INPUT_MODE
    w.load_experiment_from(old_path)
    assert w.input_panel.mode_id() == PR_IMAGE_AMPLIFICATION_INPUT_MODE
    assert w.build_request() == historical


@pytest.mark.parametrize("fault", ["roles", "separation", "exception", "augmentation"])
def test_optional_failure_cannot_fail_propagation(app, windows, monkeypatch, fault):
    w = valid_image_window(app, windows)
    mode(w, PR_IMAGE_ANALYSIS_INPUT_MODE)
    if fault == "roles":
        w.input_panel.clear_roles.click()
    elif fault == "separation":
        beams = w.beam_panel.beam_stack_definition
        w.beam_panel.set_beam_stack_definition(replace(beams, beams=tuple(
            replace(b, theta_ext_rad=0., phi_rad=0.) for b in beams.beams
        )))
        w.beam_panel.input_screen_editor.channel.setCurrentIndex(1)
        w.beam_panel.input_screen_editor.set_source(PRImageSource.from_array(np.ones((8, 8))))
    else:
        def fail(*args, **kwargs):
            raise RuntimeError("injected optional analysis failure")
        if fault == "exception":
            monkeypatch.setattr(optional, "analyze_image_amplification_result", fail)
        else:
            import lcprop.pr.products as products
            monkeypatch.setattr(products, "augment_pr_image_amplification_run_data", fail)
    w.run_clicked()
    _wait_for(app, lambda: not w._background_running)
    assert w.run_status == "completed" and w.last_result.status == "completed"
    diagnostic = w.last_runner_result.run_data.diagnostics["optional_image_analysis"].values
    assert diagnostic["status"] == ("failed" if fault in ("exception", "augmentation") else "unavailable")
    assert diagnostic["reason"] and diagnostic["propagation_status"] == "completed"
    assert "signal_carrier_mask" not in w.last_runner_result.run_data.fields


def test_screen_removal_and_genuine_model_rejection(app, windows):
    w = valid_image_window(app, windows)
    mode(w, PR_IMAGE_ANALYSIS_INPUT_MODE)
    w.beam_panel.input_screen_editor.clear_launch_elements()
    request = w.build_request()
    assert not request.launch_elements
    w._validate_execution_request(request)
    beams = w.beam_panel.beam_stack_definition
    w.beam_panel.set_beam_stack_definition(replace(beams, beams=(
        beams.beams[0], replace(beams.beams[1], wavelength_um=.7),
    )))
    with pytest.raises(ValueError, match="shared wavelength"):
        w.build_request()


@pytest.mark.parametrize("workflow", [
    "pr_static", "pr_transverse_static", "pr_transverse_timedependent",
])
def test_other_ordinary_models_keep_dispatch_with_optional_analysis(
    app, windows, monkeypatch, workflow,
):
    w = general_window(windows, 3, (0, 2))
    w.evolution_panel.set_workflow_id(workflow)
    w._set_product_policy("full")  # This dispatch comparison requires complex endpoints.
    # Small linearized fixtures exercise the selected production dispatch, not
    # an assertion of validity for arbitrary finite-envelope illuminations.
    selector = w.evolution_panel.material_response
    selector.setCurrentIndex(selector.findData("linearized"))
    w.material_panel.gain_length_product.setValue(0.)
    request = w.build_request()
    calls = []
    dispatch = w.local_runner.run_registered

    def observe(material, actual_workflow, actual_request, **kwargs):
        calls.append((actual_workflow, actual_request))
        return dispatch(material, actual_workflow, actual_request, **kwargs)

    monkeypatch.setattr(w.local_runner, "run_registered", observe)
    w.run_clicked()
    _wait_for(app, lambda: not w._background_running)
    assert calls == [(workflow, request)]
    assert w.run_status == "completed"
    assert w.last_runner_result.kind == workflow
    assert w.last_runner_result.run_data.diagnostics["optional_image_analysis"].values["status"] == "unavailable"
    mode(w, PR_GAUSSIAN_INPUT_MODE)
    assert w.build_request() == request
    reference = dispatch("pr", workflow, request)
    np.testing.assert_array_equal(w.last_result.A_final, reference.result.A_final)


def test_post_run_cancellation_preserves_successful_base(app, windows):
    from lcprop.core.execution import CancellationToken
    w = valid_image_window(app, windows)
    mode(w, PR_IMAGE_ANALYSIS_INPUT_MODE)
    request = w.build_request()
    base = w.local_runner.run_registered("pr", PR_TIMEDEPENDENT_WORKFLOW, request)
    token = CancellationToken()
    token.cancel()
    result = add_optional_image_analysis(
        base, request, PRImageAnalysisSelection(0, 1), cancellation_token=token,
    )
    assert result.result is base.result
    assert result.result.status == "completed"
    assert result.run_data.diagnostics["optional_image_analysis"].values["status"] == "cancelled"
