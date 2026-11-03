"""Bounded fresh-request integration; no remote execution or continuation."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lcprop-mpl-local-plane-integration')

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from tests.test_pr_local_plane_workflow import request
from lcprop.pr.local_plane_integration import (
    LOCAL_PLANE_OPERATION, LOCAL_PLANE_TRANSPORT_CODEC, execute_local_plane,
    encode_local_plane_request, decode_local_plane_request, selection_for_policy,
)
from lcprop.pr.local_plane_workflow import LocalPlaneRunRequest, PR_LOCAL_PLANE_WORKFLOW, PR_LOCAL_PLANE_ARITHMETIC
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








@pytest.mark.parametrize('policy', ['fast', 'full', 'analysis:far_field_intensity', 'analysis:complex_output,far_field_intensity'])
def test_registered_dispatch_and_disk_request_result_roundtrip(tmp_path, policy):
    r = replace(request(), initial_A=None)
    registry = default_transport_registry()
    assert registry.codec('pr', PR_LOCAL_PLANE_WORKFLOW) is LOCAL_PLANE_TRANSPORT_CODEC
    runner = LocalRunner(operations=default_transport_operations())
    out = runner.run_registered('pr', PR_LOCAL_PLANE_WORKFLOW, r, _result_policy=policy)
    assert out.result.status == 'completed'
    write_request_package(tmp_path, registry=registry, material_id='pr', workflow_id=PR_LOCAL_PLANE_WORKFLOW,
        request=r, run_id='local-plane-test', execution_target='local', result_policy=policy)
    decoded = read_request_package(tmp_path, registry=registry)
    assert decoded.request == r
    write_result_package(tmp_path, codec=decoded.codec, result=out.result, request_envelope=decoded.envelope)
    restored = read_result_package(tmp_path, registry=registry)
    assert restored.envelope.converged is None
    a, b = out.result.run.products, restored.result.run.products
    assert a.boundary_z_um.tobytes() == b.boundary_z_um.tobytes()
    assert a.center_z_um.tobytes() == b.center_z_um.tobytes()
    assert a.preview.tobytes() == b.preview.tobytes()
    for name in a.cuts:
        assert a.cuts[name][0].tobytes() == b.cuts[name][0].tobytes()
    if 'far_field_intensity' in policy or policy == 'full':
        assert out.result.run.scientific.far_field.intensity.tobytes() == restored.result.run.scientific.far_field.intensity.tobytes()
    if policy == 'full':
        assert out.run_data.fields['local_plane_material_xz'].coordinates['z'].size == 2
        assert out.run_data.fields['local_plane_intensity_xz'].coordinates['z'].size == 3


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
    payload = encode_local_plane_request(r)
    for key, value in [('workflow_id', PR_STATIC_WORKFLOW), ('arithmetic_id', 'midpoint')]:
        with pytest.raises(ValueError, match='incompatible'):
            decode_local_plane_request(dict(payload, **{key:value}))




def test_retention_independent_science_and_analysis_scope():
    r = request()
    results = [execute_local_plane(r, result_policy=p) for p in ('full','analysis:complex_output')]
    assert results[0].run.scientific.boundary_field.tobytes() == results[1].run.scientific.boundary_field.tobytes()
    assert results[0].run.scientific.ledger == results[1].run.scientific.ledger
    with pytest.raises(ValueError, match='supports exact'):
        selection_for_policy('analysis:complex_input')
    with pytest.raises(ValueError, match='runtime arrays'):
        encode_local_plane_request(r)
    with pytest.raises(ValueError, match='requires Local-intensity'):
        execute_local_plane(replace(r,material_response=PRTransverseMaterialResponseSpec(model='nonlinear')))


def test_streaming_estimator_no_longitudinal_gpu_volumes():
    r = request(n=2)
    short = estimate_pr_resources(r)
    long = estimate_pr_resources(replace(r, grid=replace(r.grid, z_length_um=200.)))
    assert long.peak_gpu_memory == short.peak_gpu_memory
    assert long.fast_result_size.high > short.fast_result_size.high
    assert short.local_runtime is short.h200_runtime is None
    assert short.optical_passes == 4
    assert 'Uncalibrated' in format_pr_resource_estimate(short)
    assert 'continuation' in ' '.join(short.qualifications)






def test_selected_package_executor_is_local_and_registered(tmp_path):
    from lcprop.transport.executor import execute_run_directory
    registry = default_transport_registry()
    r = replace(request(), initial_A=None)
    write_request_package(tmp_path, registry=registry, material_id='pr', workflow_id=PR_LOCAL_PLANE_WORKFLOW,
        request=r, run_id='executor-unit-test', execution_target='local', result_policy='analysis:far_field_intensity')
    assert execute_run_directory(tmp_path) == 0
    result = read_result_package(tmp_path, registry=registry).result
    assert result.status == 'completed' and result.run.scientific.far_field is not None




def test_symmetric_identity_rejects_silent_gui_migration():
    with pytest.raises(ValueError, match="midpoint/symmetric"):
        validate_pr_gui_request_representable(replace(request(), initial_A=None))
