"""Bounded fresh-request integration; no remote execution or continuation."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lcprop-mpl-local-plane-integration')

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from tests.test_pr_published_static import request
from lcprop.pr.published_static_integration import (
    PUBLISHED_STATIC_OPERATION, PUBLISHED_STATIC_TRANSPORT_CODEC, execute_published_static,
    encode_published_static_request, decode_published_static_request, selection_for_policy,
)
from lcprop.pr.published_static import PublishedStaticRequest, PR_PUBLISHED_STATIC_WORKFLOW, PR_PUBLISHED_STATIC_ARITHMETIC
from lcprop.pr.gui.request_adapter import apply_pr_request, validate_pr_gui_request_representable
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.runtime_estimator import estimate_pr_resources, format_pr_resource_estimate
from lcprop.pr.static_workflow import PRStaticRunRequest, PR_STATIC_WORKFLOW
from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec
from lcprop.persistence import EXPERIMENT_CODECS, save_experiment, load_experiment
from lcprop.runners.local import LocalRunner
from lcprop.transport.defaults import default_transport_registry, default_transport_operations
from lcprop.transport.io import write_request_package, read_request_package, write_result_package, read_result_package


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app, monkeypatch):
    import lcprop.pr.gui.main_window as module
    def report(*args): pytest.fail(str(args))
    monkeypatch.setattr(module, 'report_failure', report)
    w = PRMainWindow()
    apply_pr_request(replace(request(), initial_A=None), material_panel=w.material_panel,
        beam_panel=w.beam_panel, grid_panel=w.grid_panel, evolution_panel=w.evolution_panel)
    w._update_product_controls()
    yield w
    w.close()


def test_gui_fresh_identity_controls_and_inspection(window):
    r = window.build_request()
    assert isinstance(r, PublishedStaticRequest)
    assert window._workflow_id_for_request(r) == PR_PUBLISHED_STATIC_WORKFLOW
    assert window.evolution_panel.max_coupled_passes.isHidden()
    assert window.evolution_panel.optical_substeps.isHidden()
    text = window.describe_request(r)
    assert PR_PUBLISHED_STATIC_WORKFLOW in text and PR_PUBLISHED_STATIC_ARITHMETIC in text
    assert 'Maximum coupled passes' not in text
    assert 'accepted cells' in text
    assert window._analysis_actions['far_field_intensity'].isEnabled()
    assert not window._analysis_actions['complex_input'].isEnabled()


def test_gui_local_dispatch_status_and_experiment_roundtrip(window, tmp_path):
    window._set_product_policy('analysis:far_field_intensity')
    r = window.build_request()
    path = window.save_experiment_to(tmp_path/'fresh.lcprop.json')
    loaded = load_experiment(path, expected_material_id='pr')
    assert loaded.workflow_id == PR_PUBLISHED_STATIC_WORKFLOW
    assert loaded.request == r
    window.load_experiment_from(path)
    assert window.build_request() == r
    progress = []
    out = window._run_registered(r, progress_callback=progress.append)
    assert out.kind == PR_PUBLISHED_STATIC_WORKFLOW
    assert out.result.status == 'completed'
    assert out.result.run.scientific.far_field is not None
    assert all(p.workflow == PR_PUBLISHED_STATIC_WORKFLOW for p in progress)
    assert progress[-1].diagnostics['material_residual_rms'] is not None
    window._on_progress(progress[-1])
    assert 'Cells' in window.status_label.text()
    window._on_finished(out)
    assert window.run_status == 'completed'
    assert window.last_checkpoint is None
    assert 'far_field_intensity' in out.run_data.fields


@pytest.mark.parametrize('policy', ['fast', 'full', 'analysis:far_field_intensity', 'analysis:complex_output,far_field_intensity'])
def test_registered_dispatch_and_disk_request_result_roundtrip(tmp_path, policy):
    r = replace(request(), initial_A=None)
    registry = default_transport_registry()
    assert registry.codec('pr', PR_PUBLISHED_STATIC_WORKFLOW) is PUBLISHED_STATIC_TRANSPORT_CODEC
    runner = LocalRunner(operations=default_transport_operations())
    out = runner.run_registered('pr', PR_PUBLISHED_STATIC_WORKFLOW, r, _result_policy=policy)
    assert out.result.status == 'completed'
    write_request_package(tmp_path, registry=registry, material_id='pr', workflow_id=PR_PUBLISHED_STATIC_WORKFLOW,
        request=r, run_id='local-plane-test', execution_target='local', result_policy=policy)
    decoded = read_request_package(tmp_path, registry=registry)
    assert decoded.request == r
    write_result_package(tmp_path, codec=decoded.codec, result=out.result, request_envelope=decoded.envelope)
    restored = read_result_package(tmp_path, registry=registry)
    assert restored.envelope.converged is None
    a, b = out.result.run.products, restored.result.run.products
    assert a.boundary_z_um.tobytes() == b.boundary_z_um.tobytes()
    assert a.material_z_um.tobytes() == b.material_z_um.tobytes()
    assert a.preview.tobytes() == b.preview.tobytes()
    for name in a.cuts:
        assert a.cuts[name][0].tobytes() == b.cuts[name][0].tobytes()
    if 'far_field_intensity' in policy or policy == 'full':
        assert out.result.run.scientific.far_field.intensity.tobytes() == restored.result.run.scientific.far_field.intensity.tobytes()
    if policy == 'full':
        assert out.run_data.fields['published_static_material_xz'].coordinates['z'].size == 2
        assert out.run_data.fields['published_static_intensity_xz'].coordinates['z'].size == 3


def test_old_midpoint_identity_is_not_migrated(tmp_path):
    r = replace(request(), initial_A=None)
    old = PRStaticRunRequest(grid=r.grid, beams=r.beams, backend=r.backend,
                            material=r.material, material_response=r.material_response)
    path = save_experiment(old, tmp_path/'old.json', material_id='pr', workflow_id=PR_STATIC_WORKFLOW)
    restored = load_experiment(path, expected_material_id='pr')
    assert type(restored.request) is PRStaticRunRequest
    assert restored.workflow_id == PR_STATIC_WORKFLOW
    with pytest.raises(ValueError, match='midpoint/symmetric'):
        validate_pr_gui_request_representable(restored.request)
    payload = encode_published_static_request(r)
    for key, value in [('workflow_id', PR_STATIC_WORKFLOW), ('arithmetic_id', 'midpoint')]:
        with pytest.raises(ValueError, match='incompatible'):
            decode_published_static_request(dict(payload, **{key:value}))


def test_other_gui_models_keep_dispatch(window):
    panel = window.evolution_panel
    panel.material_response.setCurrentIndex(panel.material_response.findData('nonlinear'))
    r = window.build_request()
    assert type(r) is PublishedStaticRequest
    assert panel.workflow_id() == PR_PUBLISHED_STATIC_WORKFLOW
    assert panel.max_coupled_passes.isHidden()
    assert not panel.material_iterations.isHidden()
    from lcprop.pr.specs import PR_TIMEDEPENDENT_WORKFLOW, PRRunRequest
    panel.set_workflow_id(PR_TIMEDEPENDENT_WORKFLOW)
    assert isinstance(window.build_request(), PRRunRequest)
    from lcprop.pr.transverse.static_workflow import PR_TRANSVERSE_STATIC_WORKFLOW, PRTransverseStaticRunRequest
    panel.set_workflow_id(PR_TRANSVERSE_STATIC_WORKFLOW)
    panel.material_response.setCurrentIndex(panel.material_response.findData('linearized'))
    assert isinstance(window.build_request(), PRTransverseStaticRunRequest)


def test_retention_independent_science_and_analysis_scope():
    r = request()
    results = [execute_published_static(r, result_policy=p) for p in ('full','analysis:complex_output')]
    assert results[0].run.scientific.boundary_field.tobytes() == results[1].run.scientific.boundary_field.tobytes()
    assert results[0].run.scientific.ledger == results[1].run.scientific.ledger
    with pytest.raises(ValueError, match='supports exact'):
        selection_for_policy('analysis:complex_input')
    with pytest.raises(ValueError, match='runtime arrays'):
        encode_published_static_request(r)
    nonlinear = execute_published_static(replace(r,material_response=PRTransverseMaterialResponseSpec(model='nonlinear')))
    assert nonlinear.status == 'completed'


def test_streaming_estimator_no_longitudinal_gpu_volumes():
    r = request(n=2)
    short = estimate_pr_resources(r)
    long = estimate_pr_resources(replace(r, grid=replace(r.grid, z_length_um=200.)))
    assert long.peak_gpu_memory == short.peak_gpu_memory
    assert long.fast_result_size.high > short.fast_result_size.high
    assert short.local_runtime is short.h200_runtime is None
    assert short.optical_passes == 2
    assert 'Uncalibrated' in format_pr_resource_estimate(short)
    assert 'continuation' in ' '.join(short.qualifications)


def test_normal_gui_run_button_uses_registered_worker(window, app):
    from tests.test_pr_gui_main_window import _wait_for
    window._set_product_policy('analysis:far_field_intensity')
    calls = []
    real = window.runner.run_registered
    def observed(material, workflow, req, **kwargs):
        calls.append((material, workflow, type(req)))
        return real(material, workflow, req, **kwargs)
    window.runner.run_registered = observed
    window.run_button.click()
    _wait_for(app, lambda: not window._background_running)
    assert calls == [('pr', PR_PUBLISHED_STATIC_WORKFLOW, PublishedStaticRequest)]
    assert window.run_status == 'completed'
    assert window.last_result.run.scientific.completed_cells == 2
    assert window.last_result.run.scientific.far_field is not None
    assert not window.continue_button.isEnabled()
    assert not window.save_checkpoint_button.isEnabled()


def test_failure_preserves_truthful_gui_status(window, monkeypatch):
    import lcprop.pr.published_static as workflow
    def fail(*a, **kw): raise ValueError('injected material failure')
    monkeypatch.setattr(workflow, 'solve_arriving_material', fail)
    out = window._run_registered(window.build_request())
    assert out.result.status == 'failed'
    assert out.result.run.scientific.completed_cells == 0
    assert out.result.run.products.boundary_z_um.tolist() == [0.]
    window._on_finished(out)
    assert window.run_status == 'failed'
    assert window.status_label.text() == 'Failed'
    encoded = PUBLISHED_STATIC_TRANSPORT_CODEC.encode_result(out.result)
    assert encoded.converged is None and encoded.scientific_status == 'failed'


def test_selected_package_executor_is_local_and_registered(tmp_path):
    from lcprop.transport.executor import execute_run_directory
    registry = default_transport_registry()
    r = replace(request(), initial_A=None)
    write_request_package(tmp_path, registry=registry, material_id='pr', workflow_id=PR_PUBLISHED_STATIC_WORKFLOW,
        request=r, run_id='executor-unit-test', execution_target='local', result_policy='analysis:far_field_intensity')
    assert execute_run_directory(tmp_path) == 0
    result = read_result_package(tmp_path, registry=registry).result
    assert result.status == 'completed' and result.run.scientific.far_field is not None


def test_fresh_gui_scattering_requires_v2(window):
    from lcprop.pr.scattering import PR_CANONICAL_SCATTERING_V1, PR_CANONICAL_SCATTERING_V2
    panel = window.evolution_panel
    assert panel.scattering_algorithm.currentData() == PR_CANONICAL_SCATTERING_V2
    assert not panel.scattering_algorithm.model().item(panel.scattering_algorithm.findData(PR_CANONICAL_SCATTERING_V1)).isEnabled()
    panel.scattering_enabled.setChecked(True)
    assert window.build_request().scattering.algorithm_version == PR_CANONICAL_SCATTERING_V2


@pytest.mark.parametrize('model',['field_linear_local_intensity','nonlinear'])
def test_explicit_material_selection_survives_gui_load(window,model):
    r=replace(window.build_request(),material_response=PRTransverseMaterialResponseSpec(model=model))
    apply_pr_request(r,material_panel=window.material_panel,beam_panel=window.beam_panel,
        grid_panel=window.grid_panel,evolution_panel=window.evolution_panel)
    assert window.build_request()==r
    text=window.describe_request(r)
    assert model in text and PR_PUBLISHED_STATIC_WORKFLOW in text
    assert window.evolution_panel.material_iterations.isHidden() == (model!='nonlinear')
