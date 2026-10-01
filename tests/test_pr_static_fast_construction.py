"""Retention changes construction, never the accepted Static evolution."""
from dataclasses import replace
import weakref

import numpy as np
import pytest

import lcprop.pr.static_workflow as workflow
from lcprop.core.execution import CancellationToken
from lcprop.pr.operations import PR_STATIC_OPERATION
from lcprop.pr.scattering import PRCanonicalScatteringSpec
from lcprop.pr.static_transport_codec import (
    encode_pr_static_transport_result, decode_pr_static_transport_result,
)
from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec
from lcprop.runners.local import LocalRunner
from lcprop.runners.base import WorkflowOperation
from lcprop.transport.executor import execute_run_directory
from lcprop.transport.io import write_request_package, read_result_package
from lcprop.transport.defaults import default_transport_registry
from lcprop.transport.envelopes import TransportCodecError
from tests.test_pr_static_workflow import _static_request


def request_for(linearized, scattering, precision='float64'):
    request = _static_request(Nz=3)
    return replace(request,
        backend=replace(request.backend, precision=precision),
        material_response=PRTransverseMaterialResponseSpec(
            model='field_linear_local_intensity' if linearized else 'nonlinear'),
        scattering=PRCanonicalScatteringSpec(
            epsilon=.002, transverse_correlation_um=.4, realization_seed=19,
            canonical_dz_um=1., algorithm_version='canonical_phase_slabs_v2_cross_backend',
        ) if scattering else None)


def compare_encoded(a, b):
    assert a.payload.metadata == b.payload.metadata
    assert a.payload.arrays.keys() == b.payload.arrays.keys()
    for key in a.payload.arrays:
        np.testing.assert_array_equal(a.payload.arrays[key], b.payload.arrays[key])


@pytest.mark.parametrize('linearized', [False, True])
@pytest.mark.parametrize('scattering', [False, True])
@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_fast_is_exactly_full_then_pruned(linearized, scattering, precision):
    request = request_for(linearized, scattering, precision)
    full = workflow.run_pr_static(request)
    fast = workflow.run_pr_static(request, result_policy='fast')
    for name in workflow.PR_STATIC_FAST_OMITTED_FIELDS:
        assert getattr(fast, name) is None
        assert getattr(full, name) is not None
    assert full.retention_summary['policy'] == 'full'
    old = encode_pr_static_transport_result(full, 'fast')
    new = encode_pr_static_transport_result(fast, 'fast')
    assert 'result__A_initial' not in new.payload.arrays
    assert 'result__A_final' not in new.payload.arrays
    recovered = decode_pr_static_transport_result(old.payload.metadata, old.payload.arrays)
    old_products = PR_STATIC_OPERATION.to_run_data(recovered)
    for name in ('input_intensity', 'output_intensity', 'far_field_intensity'):
        expected = old_products.fields[name].data.astype(np.float32)
        np.testing.assert_array_equal(fast.selected_products['previews'][name]['data'], expected)
    for name in ('longitudinal_intensity_xz', 'longitudinal_intensity_yz', 'intensity_preview'):
        np.testing.assert_array_equal(getattr(fast, name), getattr(recovered, name))
    assert fast.replay_diagnostics == full.replay_diagnostics
    assert fast.slice_summaries == full.slice_summaries


@pytest.mark.parametrize('stop_after', [0, 1, 2])
def test_fast_cancellation_matches_post_pruned_completed_length(stop_after):
    request = request_for(True, True)
    def run(policy):
        token = CancellationToken()
        if stop_after == 0:
            token.cancel()
        def progress(value):
            if value.message == 'PR static z slice completed' and value.completed_units == stop_after:
                token.cancel()
        return workflow.run_pr_static(request, result_policy=policy,
            cancellation_token=token, progress_callback=progress)
    full, fast = run('full'), run('fast')
    assert fast.completed_slices == stop_after
    old = encode_pr_static_transport_result(full, 'fast')
    recovered = decode_pr_static_transport_result(old.payload.metadata, old.payload.arrays)
    for name in ('longitudinal_intensity_xz', 'longitudinal_intensity_yz', 'intensity_preview'):
        np.testing.assert_array_equal(getattr(fast, name), getattr(recovered, name))
    assert fast.replay_diagnostics == full.replay_diagnostics


@pytest.mark.parametrize('linearized', [False, True])
@pytest.mark.parametrize('scattering', [False, True])
def test_fast_never_exports_omitted_volumes_and_releases_them(monkeypatch, linearized, scattering):
    request = request_for(linearized, scattering)
    shape = (3, request.grid.Nx, request.grid.Ny)
    volumes = []
    backend_factory = workflow.get_backend
    class Arrays:
        __name__ = 'numpy'
        __version__ = np.__version__
        def __getattr__(self, name):
            original = getattr(np, name)
            if name not in ('zeros', 'empty', 'empty_like'):
                return original
            def allocate(*args, **kwargs):
                a = original(*args, **kwargs)
                if a.shape == shape:
                    volumes.append(weakref.ref(a))
                return a
            return allocate
    monkeypatch.setattr(workflow, 'get_backend', lambda spec: replace(backend_factory(spec), xp=Arrays()))
    original_host = workflow._owned_host_result
    transfers = []
    def host(value):
        assert value.shape != shape
        # At endpoint conversion, only the replay source volume is still live.
        assert sum(ref() is not None for ref in volumes) == 1
        return original_host(value)
    def transfer(value):
        assert value.shape != shape
        assert value.ndim <= 2  # cuts or one source plane
        transfers.append(value.shape)
        return np.asarray(value)
    monkeypatch.setattr(workflow, '_owned_host_result', host)
    monkeypatch.setattr(workflow, 'asnumpy', transfer)
    result = workflow.run_pr_static(request, result_policy='fast')
    assert len(volumes) == 6
    assert all(ref() is None for ref in volumes)
    assert len(transfers) >= 2 + result.completed_slices
    assert result.A_initial is None and result.A_final is None
    assert result.E_final is None


def test_fast_cannot_be_mislabeled_full():
    fast = workflow.run_pr_static(request_for(True, False), result_policy='fast')
    with pytest.raises(TransportCodecError, match='different policy'):
        encode_pr_static_transport_result(fast, 'full')


def test_executor_passes_envelope_policy_before_construction(tmp_path):
    registry = default_transport_registry()
    request = request_for(True, True)
    seen = []
    def run(request, **kwargs):
        assert kwargs['result_policy'] == 'fast'
        result = workflow.run_pr_static(request, **kwargs)
        assert result.E_final is None
        seen.append(result)
        return result
    operation = replace(PR_STATIC_OPERATION, run=run)
    write_request_package(tmp_path, registry=registry, material_id='pr', workflow_id='pr_static',
        request=request, run_id='fast-construction', execution_target='local', result_policy='fast')
    assert execute_run_directory(tmp_path, registry=registry, operations=(operation,)) == 0
    received = read_result_package(tmp_path, registry=registry)
    assert len(seen) == 1
    compare_encoded(encode_pr_static_transport_result(seen[0], 'fast'),
                    encode_pr_static_transport_result(received.result, 'fast'))


def test_local_runner_policy_capability_is_opt_in():
    runner = LocalRunner((PR_STATIC_OPERATION,))
    result = runner.run_registered('pr', 'pr_static', request_for(True, False), _result_policy='fast')
    assert result.result.E_final is None
    assert result.run_data is not None
    untouched = WorkflowOperation('other', 'other', lambda request: request, lambda result: result)
    assert runner.run_operation(untouched, 3, _result_policy='fast').result == 3
    with pytest.raises(ValueError, match='conflicting'):
        runner.run_operation(PR_STATIC_OPERATION, request_for(True, False),
                             _result_policy='fast', result_policy='full')


@pytest.mark.parametrize('scattering', [False, True])
def test_cupy_fast_matches_post_pruned_full_when_available(scattering):
    cp = pytest.importorskip('cupy')
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip('no CUDA device')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CUDA unavailable')
    request = request_for(True, scattering)
    request = replace(request, backend=replace(request.backend, backend='cupy'))
    full = workflow.run_pr_static(request)
    fast = workflow.run_pr_static(request, result_policy='fast')
    old = encode_pr_static_transport_result(full, 'fast')
    recovered = decode_pr_static_transport_result(old.payload.metadata, old.payload.arrays)
    for name in ('longitudinal_intensity_xz', 'longitudinal_intensity_yz', 'intensity_preview'):
        np.testing.assert_array_equal(getattr(fast, name), getattr(recovered, name))
    assert fast.replay_diagnostics == full.replay_diagnostics


def test_invalid_retention_fails_before_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('backend resolved before invalid policy rejected')
    monkeypatch.setattr(workflow, 'get_backend', forbidden)
    with pytest.raises(ValueError):
        workflow.run_pr_static(request_for(True, False), result_policy='unknown')
