"""Metadata-only support boundaries: no large arrays, solves or device probes."""
from dataclasses import replace
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.gui.request_transparency import inspect_request
from lcprop.pr.unified import integration as a, local_planning as lp
from lcprop.pr.unified.specs import (
    UNBIASED, FIXED_FIELD, PRESCRIBED_CURRENT, OPEN_TRANSVERSE,
)
from lcprop.pr.unified.solver_specs import PRUnifiedSolverSpec, SCALABLE, ITERATIVE_POLICY
from tests.test_pr_unified_integration import fresh, app, window, apply

CLOSURES = (UNBIASED, FIXED_FIELD, PRESCRIBED_CURRENT, OPEN_TRANSVERSE)


def request(n, kind=UNBIASED, backend='numpy', precision='float64'):
    r = fresh(2, kind, precision)
    return replace(r, grid=replace(r.grid, Nx=n, Ny=n),
                   solver=PRUnifiedSolverSpec(SCALABLE, ITERATIVE_POLICY),
                   backend=BackendSpec(backend, precision, False))


@pytest.mark.parametrize('kind', CLOSURES)
@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
def test_evidence_specific_limits(kind, backend, monkeypatch):
    monkeypatch.setattr(a, 'get_backend', lambda *a: pytest.fail('device probe'))
    limit = 2048 if backend == 'cupy' or kind == UNBIASED else 1024
    for n in (512, 1024, limit):
        r = request(n, kind, backend)
        before = a.encode_fresh(r)
        a.validate_fresh(r)
        plan = a.resource_plan(r, 'minimal')
        assert f'qualified <={limit}' in plan['support_envelope']
        assert plan['active_dimensions'] == 2 and plan['independent_columns'] == 1
        assert a.encode_fresh(r) == before
    for shape in ((limit+1, 32), (limit+1, limit), (4096, 4096)):
        bad = replace(r, grid=replace(r.grid, Nx=shape[0], Ny=shape[1]))
        with pytest.raises(ValueError, match='outside scalable commissioned envelope'):
            a.resource_plan(bad)


@pytest.mark.parametrize('backend,limit', [('numpy', 96), ('cupy', 256)])
def test_state32_and_backend_restrictions(backend, limit):
    a.validate_fresh(request(limit, backend=backend, precision='float32'))
    with pytest.raises(ValueError, match='envelope'):
        a.validate_fresh(request(limit+1, backend=backend, precision='float32'))
    from lcprop.pr.unified.solver_specs import validate_execution
    core = a.core_request(request(16))
    with pytest.raises(ValueError, match='unknown scalable backend'):
        validate_execution(core.solver, core.spatial, core.closure, core.precision, 'auto')


def test_reduced_is_not_connected_qualification():
    r = fresh(1)
    r = replace(r, grid=replace(r.grid, Nx=4096, Ny=512))
    a.validate_fresh(r)
    plan = a.resource_plan(r, 'minimal')
    assert plan['active_dimensions'] == 1 and plan['independent_columns'] == 512
    assert plan['support_envelope'] == 'reduced independent columns'
    with pytest.raises(ValueError, match='genuine x-y'):
        a.validate_fresh(replace(r, solver=PRUnifiedSolverSpec(SCALABLE, ITERATIVE_POLICY)))


def test_expensive_support_and_memory_advice_are_distinct(monkeypatch):
    monkeypatch.setattr(lp, 'physical_memory_bytes', lambda: 128*1024**3)
    small = a.resource_plan(request(256), 'minimal')
    large = a.resource_plan(request(2048), 'minimal')
    assert small['local_assessment']['classification'] == 'comfortable'
    assert large['local_assessment']['classification'] == 'large/slow'
    assert 'Supported but expensive' in ' '.join(large['local_assessment']['warnings'])
    assert large['bytes']['material_fft_buffers'] == 64*small['bytes']['material_fft_buffers']
    assert large['measured_native_peak'] is None  # never replace formula with one measured fixture
    monkeypatch.setattr(lp, 'physical_memory_bytes', lambda: 1024**3)
    assert a.resource_plan(request(2048), 'minimal')['local_assessment']['classification'] == 'memory-risk'
    a.validate_fresh(request(2048))  # memory advice is not a scientific limit
    native = a.resource_plan(request(2048, backend='cupy'), 'minimal')
    assert 'local_assessment' not in native


def test_runtime_estimator_propagates_policy_and_longitudinal_cost(monkeypatch):
    from lcprop.pr.runtime_estimator import estimate_pr_resources
    monkeypatch.setattr(lp, 'physical_memory_bytes', lambda: 16*1024**3)
    r = request(2048, backend='cupy')
    estimate = estimate_pr_resources(r)
    assert 'H200/CuPy state64 qualified <=2048' in estimate.recommendation
    long = replace(r, grid=replace(r.grid, z_length_um=80*r.grid.dz_um))
    fast = a.resource_plan(long, 'fast')
    full = a.resource_plan(long, 'full')
    assert full['longitudinal_full_volume_bytes'] > fast['longitudinal_full_volume_bytes']
    assert sum(full['bytes'].values()) > sum(fast['bytes'].values())
    local = replace(long, backend=BackendSpec('numpy', 'float64', False))
    assert a.resource_plan(local, 'full')['local_assessment']['classification'] == 'memory-risk'
    with pytest.raises(ValueError, match='outside scalable commissioned envelope'):
        estimate_pr_resources(request(4096, backend='cupy'))


def test_inspect_supported_expensive_and_rejected(window, monkeypatch):
    monkeypatch.setattr(window, '_validate_execution_request', lambda r: a.validate_fresh(r))
    monkeypatch.setattr(lp, 'physical_memory_bytes', lambda: 128*1024**3)
    apply(window, request(1024))
    inspect_request(window, window.build_request)
    assert 'Request validated; not executed' in window.results_panel.workspace.operation_status.text()
    assert 'larger supported grids are expensive' in window.describe_request(window.build_request())
    messages = []
    monkeypatch.setattr(window.results_panel.workspace, 'append_console', messages.append)
    window.grid_panel.set_grid(replace(window.build_request().grid, Nx=4096, Ny=4096))
    inspect_request(window, window.build_request)
    assert len(messages) == 1
    assert 'Invalid request: outside scalable commissioned envelope' in messages[0]
    assert 'Traceback' not in messages[0]
    assert 'Failed' in window.results_panel.workspace.operation_status.text()


def test_unexpected_inspect_failure_keeps_diagnostics(window, monkeypatch):
    messages = []
    monkeypatch.setattr(window.results_panel.workspace, 'append_console', messages.append)
    def broken():
        raise RuntimeError('unexpected implementation failure')
    inspect_request(window, broken)
    assert 'Traceback' in messages[0] and 'unexpected implementation failure' in messages[0]
