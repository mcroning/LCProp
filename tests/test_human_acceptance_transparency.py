"""Stage 1A presentation/dispatch regressions; no scientific evidence runs."""
import math
from dataclasses import replace
import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from lcprop.gui.workspace import Workspace
from lcprop.gui.request_transparency import execution_summary
from lcprop.lc.gui.main_window import LCPropMainWindow
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.gui.image_input_panel import PR_IMAGE_AMPLIFICATION_INPUT_MODE
from lcprop.products.data_model import RunData, FieldCollection, FieldData
from lcprop.runners.slurm import SlurmRunner
from lcprop.runners.source_deployment import SourceDeploymentError, SourceDeploymentManager


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _data():
    return RunData("static", fields=FieldCollection([
        ("intensity", FieldData("intensity", "Final Authoritative Optical Intensity x-y",
                                np.ones((4, 4)), ("x", "y"), "intensity")),
    ]))


def _two_beams(window):
    from launchplane.model import BeamDefinition, BeamStackDefinition
    window.beam_panel.set_beam_stack_definition(BeamStackDefinition(beams=(
        BeamDefinition(
            name='pump',
            wavelength_um=0.633,
            power_mW=1,
            coherence_group='laser',
            w1_um=12,
            w2_um=12,
            theta_ext_rad=math.asin(math.hypot(0.01, 0.0) * 0.633 / (2 * math.pi)),
            phi_rad=math.atan2(0.0, 0.01) % (2 * math.pi),
        ),
        BeamDefinition(
            name='signal',
            wavelength_um=0.633,
            power_mW=0.2,
            coherence_group='laser',
            w1_um=12,
            w2_um=12,
            theta_ext_rad=math.asin(math.hypot(-0.02, 0.0) * 0.633 / (2 * math.pi)),
            phi_rad=math.atan2(0.0, -0.02) % (2 * math.pi),
        ),
    )))


@pytest.mark.parametrize("tab", range(5))
def test_refresh_preserves_every_subtab_and_scientific_records(app, tab):
    workspace = Workspace()
    data = _data()
    workspace.begin_request("Run")
    workspace.set_request_summary("first request")
    workspace.tabs.setCurrentIndex(tab)
    workspace.set_run_data(data, state="Current accepted state")
    workspace.set_run_data(data)
    assert workspace.tabs.currentIndex() == tab
    assert workspace.image_pane._run_data.fields['intensity'].data is data.fields['intensity'].data
    assert "Authoritative" not in workspace.image_pane.field_selector.currentText()
    assert "Authoritative" in workspace.diagnostics_view.toPlainText()
    workspace.operation_boundary("Save Experiment")
    assert "Save Experiment" in workspace.console.toPlainText()
    assert "Completed result" in workspace.result_ownership.text()
    np.testing.assert_array_equal(data.fields['intensity'].data, np.ones((4, 4)))
    workspace.close()


@pytest.mark.parametrize("window_type,slot", [(LCPropMainWindow, "run_static_clicked"),
                                              (PRMainWindow, "run_clicked")])
def test_rejected_request_retains_previous_result_and_request(app, monkeypatch, window_type, slot):
    window = window_type()
    if window_type is LCPropMainWindow:
        window.experiment_panel.experiment.setCurrentText("Static propagation")
    workspace = window.results_panel.workspace
    workspace.begin_request("Run")
    workspace.set_request_summary("successful request")
    data = _data()
    workspace.set_run_data(data)
    def invalid():
        raise ValueError("correct the beam configuration")
    monkeypatch.setattr(window, "build_request", invalid)
    getattr(window, slot)()
    assert "Previous result" in workspace.result_ownership.text()
    assert "successful request" in workspace.request_summary.toPlainText()
    assert "correct the beam configuration" in workspace.operation_status.text()
    assert "Traceback" in workspace.console.toPlainText()
    assert not window._background_running
    assert workspace.image_pane._run_data.fields['intensity'].data is data.fields['intensity'].data
    window.close()


@pytest.mark.parametrize("window_type", [LCPropMainWindow, PRMainWindow])
def test_inspection_never_starts_worker_or_runner(app, monkeypatch, window_type):
    window = window_type()
    def forbidden(*args, **kwargs):
        pytest.fail("preview attempted execution")
    monkeypatch.setattr(window.local_runner, "run_registered", forbidden)
    module = __import__(window_type.__module__, fromlist=['WorkflowWorker'])
    monkeypatch.setattr(module, "WorkflowWorker", forbidden)
    window.preview_request_clicked()
    workspace = window.results_panel.workspace
    assert "Pre-run inspection" in workspace.request_summary.toPlainText()
    assert "Precision: float64" in workspace.request_summary.toPlainText()
    assert "not executed" in workspace.operation_status.text()
    assert not window._background_running
    window.close()


def test_transverse_one_channel_gate_is_specific_and_before_worker(app, monkeypatch):
    window = LCPropMainWindow()
    _two_beams(window)
    window.experiment_panel.experiment.setCurrentText("Soliton")
    window.solver_panel.refine_transverse_checkbox.setChecked(True)
    def forbidden(*args, **kwargs):
        pytest.fail("incompatible refinement started worker")
    import lcprop.lc.gui.main_window as module
    monkeypatch.setattr(module, "WorkflowWorker", forbidden)
    window.run_static_clicked()
    assert "exactly one optical channel" in window.results_panel.workspace.operation_status.text()
    assert not window._background_running
    window._validate_execution_request(replace(window.build_soliton_request(), refine_transverse=False))
    window._validate_execution_request(window.build_request())
    window.close()


@pytest.mark.parametrize("workflow", ['pr_timedependent', 'pr_static',
                                      'pr_transverse_static', 'pr_transverse_timedependent'])
def test_general_asymmetric_screen_free_tbc_planning_and_persistence(app, tmp_path, workflow):
    from lcprop.persistence import load_experiment
    window = PRMainWindow()
    window.evolution_panel.set_workflow_id(workflow)
    _two_beams(window)
    request = window.build_request()
    window._validate_execution_request(request)
    assert len(request.beams.channels) == 2
    assert request.beams.channels[0].tilt_x_rad_per_um != -request.beams.channels[1].tilt_x_rad_per_um
    window._estimate_current_resources()
    assert "Model:" in window.resource_estimator_panel.output.toPlainText()
    # The real planning formatter is invoked, not a stubbed estimate.
    from lcprop.pr.runtime_estimator import estimate_pr_resources
    assert estimate_pr_resources(window._request_for_local_cost(request)) is not None
    path = tmp_path / 'tbc.json'
    window.save_experiment_to(path)
    assert load_experiment(path).request == request
    restored = PRMainWindow()
    restored.load_experiment_from(path)
    assert restored.build_request() == request
    restored.close()
    window.input_panel.input_mode.setCurrentIndex(
        window.input_panel.input_mode.findData(PR_IMAGE_AMPLIFICATION_INPUT_MODE)
    )
    with pytest.raises(ValueError):
        window.build_request()  # Specialized setup still needs its image screen.
    window.close()


@pytest.mark.parametrize("backend", ['numpy', 'cupy', 'auto'])
def test_remote_summary_never_infers_gpu_execution_from_slurm(app, backend):
    window = PRMainWindow()
    request = window.build_request()
    request = replace(request, backend=replace(request.backend, backend=backend))
    runner = SimpleNamespace(config=SimpleNamespace(cluster_profile='test cluster',
                                                    default_resource_profile='test resource'))
    window.runner = window.slurm_runner = window._explicit_slurm_runner = runner
    window.remote_execution_controls.runner_kwargs = lambda: {}
    text = execution_summary(window, request)
    assert f"Requested backend: {backend}" in text
    assert "Resolved backend: unresolved" in text
    assert "Cluster: test cluster" in text
    assert "Resource: test resource" in text
    assert request.backend.backend == backend
    window.close()


def test_source_preflight_is_local_and_actionable(tmp_path):
    class NoRemote:
        def ssh(self, *args):
            pytest.fail("preflight contacted remote")
        def upload(self, *args):
            pytest.fail("preflight uploaded files")
    runner = object.__new__(SlurmRunner)
    runner.config = SimpleNamespace(remote_source_path=None)
    runner._source_deployment_manager = SourceDeploymentManager(
        host='example.invalid', source_root='/source', local_source=tmp_path,
        transport=NoRemote(),
    )
    with pytest.raises(SourceDeploymentError, match='source_not_git_checkout') as error:
        runner.preflight_source()
    assert 'clean LCProp Git checkout' in str(error.value)
    assert 'LCPROP_SLURM_SOURCE_PATH' in str(error.value)
    runner.config.remote_source_path = '/pre-staged'
    runner.preflight_source()


def test_specialized_preview_keeps_screen_geometry_validation(app, tmp_path):
    from PySide6.QtGui import QImage
    from tests.test_experiment_gui import _configure_image_amplification_window
    image_path = tmp_path / 'signal.png'
    image = QImage(4, 4, QImage.Format.Format_RGB32)
    image.fill(0xFFFFFFFF)
    assert image.save(str(image_path))
    window = PRMainWindow()
    _configure_image_amplification_window(window, image_path, 'pr_timedependent')
    window.preview_request_clicked()
    assert 'not executed' in window.results_panel.workspace.operation_status.text()
    stack = window.beam_panel.beam_stack_definition
    window.beam_panel.set_beam_stack_definition(replace(
        stack, beams=(stack.beams[0], replace(
            stack.beams[1],
            theta_ext_rad=math.asin(math.hypot(-0.1, stack.beams[1].transverse_wavevector_rad_per_um[1]) * stack.beams[1].wavelength_um / (2 * math.pi * stack.beams[1].n_ext)),
            phi_rad=math.atan2(stack.beams[1].transverse_wavevector_rad_per_um[1], -0.1) % (2 * math.pi),
        ))
    ))
    window.preview_request_clicked()
    assert 'symmetric' in window.results_panel.workspace.operation_status.text()
    assert not window._background_running
    window.close()


def test_gui_source_failure_precedes_worker_and_remote_execution(app, tmp_path, monkeypatch):
    window = LCPropMainWindow()
    window.experiment_panel.experiment.setCurrentText('Static propagation')
    runner = object.__new__(SlurmRunner)
    runner.config = SimpleNamespace(remote_source_path=None)
    runner._source_deployment_manager = SourceDeploymentManager(
        host='example.invalid', source_root='/source', local_source=tmp_path,
    )
    window.runner = window.slurm_runner = window._explicit_slurm_runner = runner
    import lcprop.lc.gui.main_window as module
    monkeypatch.setattr(module, 'WorkflowWorker', lambda *a, **k: pytest.fail('started worker'))
    monkeypatch.setattr(runner._source_deployment_manager, 'resolve_or_stage',
                        lambda: pytest.fail('attempted source deployment'))
    window.run_static_clicked()
    assert 'clean LCProp Git checkout' in window.results_panel.workspace.operation_status.text()
    assert not window._background_running
    window.close()


def test_failure_with_current_progress_and_stop_keep_true_owner(app):
    from lcprop.gui.request_transparency import report_failure
    window = LCPropMainWindow()
    workspace = window.results_panel.workspace
    workspace.begin_request('Run')
    workspace.set_request_summary('request one')
    workspace.set_run_data(_data())
    workspace.set_td_time_indicator('Final z: 10')
    workspace.begin_request('Continue')
    workspace.set_request_summary('request two')
    workspace.set_td_time_indicator('new progress time')
    assert 'Previous result' in workspace.result_ownership.text()
    workspace.set_run_data(_data(), state='Current accepted state')
    report_failure(window, 'RuntimeError: synthetic worker failure')
    assert workspace.result_ownership.text() == 'State at failure'
    workspace.finish_attempt('State at stop/cancellation')
    assert 'Request 2' in workspace.request_summary.toPlainText()
    assert 'request two' in workspace.request_summary.toPlainText()
    window.close()


@pytest.mark.parametrize("tab", range(5))
def test_populated_to_empty_products_cannot_relabel_old_views(app, tab):
    from lcprop.products.data_model import CurveCollection, CurveData
    workspace = Workspace()
    td = replace(_data(), workflow="pr_timedependent", curves=CurveCollection([
        ("power", CurveData("power", "TD power", np.arange(3), np.ones(3), "t", "P")),
    ]))
    workspace.begin_request("Run")
    workspace.set_run_data(td)
    assert not workspace.curve_pane.curve_view.isHidden()
    workspace.tabs.setCurrentIndex(tab)
    workspace.begin_request("Run")
    # PR static products have fields but no curves.
    workspace.set_run_data(replace(_data(), workflow="pr_static"))
    assert workspace.curve_pane.curve_selector.count() == 0
    assert workspace.curve_pane.curve_view.isHidden()
    assert not workspace.image_pane.image_view.isHidden()
    assert workspace.result_ownership.text() == "Completed result"
    workspace.begin_request("Run")
    workspace.set_run_data(RunData("empty"))
    assert workspace.image_pane.field_selector.count() == 0
    assert workspace.image_pane.image_view.isHidden()
    assert workspace.curve_pane.curve_view.isHidden()
    assert "TD power" not in workspace.diagnostics_view.toPlainText()
    assert "intensity" not in workspace.diagnostics_view.toPlainText()
    assert workspace.tabs.currentIndex() == tab
    workspace.begin_request("Run")
    workspace.set_run_data(td)
    assert not workspace.curve_pane.curve_view.isHidden()
    assert not workspace.image_pane.image_view.isHidden()
    assert workspace.tabs.currentIndex() == tab
    workspace.close()


@pytest.mark.parametrize("failure_at", ["image_pane", "longitudinal_pane", "curve_pane", "diagnostics"])
def test_partial_render_failure_has_no_displayed_owner_and_recovers(app, monkeypatch, failure_at):
    workspace = Workspace()
    workspace.begin_request("Run")
    workspace.set_request_summary("old request")
    workspace.set_run_data(_data())
    workspace.begin_request("Run")
    workspace.set_request_summary("new request")
    workspace.tabs.setCurrentIndex(2)
    target = workspace if failure_at == "diagnostics" else getattr(workspace, failure_at)
    method = "_format_diagnostics" if failure_at == "diagnostics" else "set_run_data"
    original = getattr(target, method)
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        raise ValueError("render failure after updating")
    monkeypatch.setattr(target, method, fail)
    with pytest.raises(ValueError, match="render failure"):
        workspace.set_run_data(_data())
    workspace.finish_attempt("State at failure")
    assert workspace._displayed_attempt is None
    assert workspace.result_ownership.text() == "No displayed result — result update failed"
    assert workspace.image_pane.isHidden()
    assert workspace.longitudinal_pane.isHidden()
    assert workspace.curve_pane.curve_view.isHidden()
    assert workspace.curve_pane.curve_selector.isHidden()
    assert not workspace.diagnostics_view.toPlainText()
    assert workspace.open_td_preview.isHidden()
    assert workspace.tabs.currentIndex() == 2
    assert workspace.updatesEnabled()
    for tab in range(5):
        workspace.tabs.setCurrentIndex(tab)
        assert workspace.image_pane.isHidden()
        assert workspace.curve_pane.curve_view.isHidden()
    monkeypatch.setattr(target, method, original)
    workspace.set_run_data(_data())
    assert workspace.result_ownership.text() == "Completed result"
    assert not workspace.image_pane.isHidden()
    assert not workspace.image_pane.image_view.isHidden()
    assert workspace.tabs.currentIndex() == 4
    workspace.close()


def test_pr_continuation_dispatch_is_local_with_slurm_and_noncheckout_source(app, tmp_path, monkeypatch):
    import lcprop.pr.gui.main_window as module
    from tests.test_pr_gui_main_window import _wait_for
    window = PRMainWindow()
    runner = object.__new__(SlurmRunner)
    runner.config = SimpleNamespace(remote_source_path=None)
    runner._source_deployment_manager = SourceDeploymentManager(
        host="example.invalid", source_root="/source", local_source=tmp_path,
    )
    # Establish that this installed-package-style source cannot auto-deploy.
    with pytest.raises(SourceDeploymentError, match="source_not_git_checkout"):
        runner.preflight_source()
    window.slurm_runner = window._explicit_slurm_runner = runner
    window.execution_target_selector.setCurrentIndex(
        window.execution_target_selector.findData("slurm")
    )
    assert window.runner is runner
    def forbidden(*args, **kwargs):
        pytest.fail("local continuation attempted Slurm validation or deployment")
    monkeypatch.setattr(runner, "preflight_source", forbidden)
    monkeypatch.setattr(runner, "run_registered", forbidden)
    monkeypatch.setattr(window.remote_execution_controls, "validate_backend", forbidden)
    checkpoint = SimpleNamespace(completed_steps=3, time_normalized=0.003)
    window.last_checkpoint = checkpoint
    monkeypatch.setattr(module, "validate_pr_continuation", lambda request, supplied: None)
    dispatched = []
    def local_continuation(request, supplied, steps, **kwargs):
        dispatched.append((request, supplied, steps))
        # Stop at the scientific boundary; exercise the real worker/adapter.
        raise RuntimeError("local dispatch reached; no scientific calculation")
    monkeypatch.setattr(module, "continue_pr_timedependent", local_continuation)
    window.continue_clicked()
    _wait_for(app, lambda: not window._background_running)
    assert len(dispatched) == 1
    assert dispatched[0][1] is checkpoint
    workspace = window.results_panel.workspace
    summary = workspace.request_summary.toPlainText()
    assert "Execution target: Local" in summary
    assert "Continuation is Local-only" in summary
    assert "Execution target: Slurm" not in summary
    console = workspace.console.toPlainText()
    assert "Continuing locally" in console
    assert "Slurm submission / remote execution" not in console
    assert "local dispatch reached" in console
    assert window.runner is runner  # Ordinary Run keeps its selected target.
    # The ordinary Run validation still invokes the automatic-source gate.
    calls = []
    monkeypatch.setattr(window.remote_execution_controls, "validate_backend", lambda *a: None)
    monkeypatch.setattr(window, "_slurm_supports_workflow", lambda *a: True)
    monkeypatch.setattr(runner, "preflight_source", lambda: calls.append("source"))
    window._validate_execution_request(window.build_request())
    assert calls == ["source"]
    window.close()
