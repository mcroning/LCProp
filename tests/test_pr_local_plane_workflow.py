"""Bounded independent orchestration, ownership and failure oracles."""
from dataclasses import replace
import hashlib
import math
import weakref

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec, get_backend
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.core.grid import make_grid
from lcprop.optics.boundaries import TransverseBoundarySpec
from lcprop.optics.farfield import direction_cosine_spectrum
from lcprop.optics.launch import build_launch, OpticalLaunchContext
from lcprop.pr import local_plane_workflow as flow
from lcprop.pr.local_plane_step import step_local_intensity_cell
from lcprop.pr.scattering import PRCanonicalScatteringSpec, PR_CANONICAL_SCATTERING_V2
from lcprop.pr.source import channel_peak_intensity_reference
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec


def request(n=2, h=2., precision='float64', scattering=False, supplied=True):
    A = np.broadcast_to(1+.2*np.exp(2j*np.pi*np.arange(16)[:, None]/16), (16, 8))[None].copy()
    A = A.astype(np.complex64 if precision == 'float32' else np.complex128)
    return flow.LocalPlaneRunRequest(
        GridSpec(Nx=16, Ny=8, x_aperture_um=16., y_aperture_um=16., dz_um=h, z_length_um=n*h),
        BeamStack(channels=(BeamChannel(wavelength_um=.633, w1_um=4., w2_um=4.),)),
        material=PRMaterialSpec(gain_length_product=.2, characteristic_wavenumber_per_um_override=1.),
        backend=BackendSpec('numpy', precision, False), initial_A=A if supplied else None,
        scattering=PRCanonicalScatteringSpec(.02, .4, 0, 1., PR_CANONICAL_SCATTERING_V2) if scattering else None)


def explicit_march(req):
    backend = get_backend(req.backend)
    grid = make_grid(req.grid, xp=backend.xp, real_dtype=backend.real_dtype)
    if req.initial_A is None:
        launch = build_launch(req.beams, grid, complex_dtype=backend.complex_dtype,
            context=OpticalLaunchContext(grid, req.material.refractive_index, req.grid.z_length_um))
        A = launch.A0.copy()
    else:
        A = req.initial_A.copy()
    initial = A.copy()
    reference = channel_peak_intensity_reference(A, xp=backend.xp)
    n = round(req.grid.z_length_um/req.grid.dz_um)
    for k in range(n):
        cell = step_local_intensity_cell(A, grid=grid, cell_index=k,
            z_start_um=k*req.grid.dz_um, dz_um=req.grid.dz_um,
            interaction_length_um=req.grid.z_length_um, wavelength_um=.633,
            material=req.material, peak_intensity_reference=reference, backend=req.backend,
            coherence_groups=req.beams.coherence_groups, scattering=req.scattering)
        A = cell.A_candidate
    return initial, A, reference


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('n', [1, 2, 5])
@pytest.mark.parametrize('scattering', [False, True])
def test_exact_explicit_primitive_oracle(precision, n, scattering):
    req = request(n=n, precision=precision, scattering=scattering)
    initial, expected, reference = explicit_march(req)
    out = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    assert out.status == 'completed', out.failure
    assert out.completed_cells == out.requested_cells == n
    assert out.reached_z_um == req.grid.z_length_um
    np.testing.assert_array_equal(out.boundary_field, expected)
    assert out.launch_identity['sha256'] == hashlib.sha256(initial.tobytes()).hexdigest()
    assert out.launch_identity['peak_intensity_reference'] == reference
    assert all(row['peak_intensity_reference'] == reference for row in out.ledger)
    spec = direction_cosine_spectrum(expected, dx_um=1., dy_um=2., wavelength_um=.633,
                                    refractive_index=2.4, coherence_groups=req.beams.coherence_groups)
    for name in ('intensity', 's_x', 's_y'):
        np.testing.assert_array_equal(getattr(out.far_field, name), getattr(spec, name))
    assert out.far_field.intensity.dtype == np.dtype(precision)
    assert out.far_field_z_um == req.grid.z_length_um
    assert out.last_material_plane_um == (n-.5)*req.grid.dz_um
    assert out.workflow_identity == 'pr_static_local_intensity_planes_v1'
    assert out.arithmetic_identity == 'per_cell_half_linear_full_local_phase_v1'
    assert not hasattr(out, 'converged') and not hasattr(out, 'replay_diagnostics')


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_authoritative_generated_launch(precision):
    req = request(precision=precision, supplied=False)
    initial, expected, reference = explicit_march(req)
    out = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    assert out.status == 'completed', out.failure
    np.testing.assert_array_equal(out.boundary_field, expected)
    assert out.launch_identity['sha256'] == hashlib.sha256(initial.tobytes()).hexdigest()
    assert out.launch_identity['peak_intensity_reference'] == reference


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_launch_snapshot_and_retention_do_not_change_arithmetic(monkeypatch, precision):
    req = request(precision=precision)
    original = req.initial_A.copy()
    fields = []
    real_step = flow.step_local_intensity_cell
    def observe(*a, **kw):
        out = real_step(*a, **kw)
        fields.append(out.A_candidate.copy())
        return out
    monkeypatch.setattr(flow, 'step_local_intensity_cell', observe)
    def mutate(progress):
        if progress.completed_cells == 0:
            req.initial_A[:] = 77
        if progress.last_cell is not None:
            progress.last_cell['material_weight_um'] = -1
    minimal = flow.run_local_intensity_planes(req, progress_callback=mutate)
    first_fields = fields[:]
    fields.clear()
    retained = flow.run_local_intensity_planes(replace(req, initial_A=original), retain_boundary_field=True)
    assert minimal.status == retained.status == 'completed'
    assert minimal.boundary_field is None
    for a, b in zip(first_fields, fields):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(minimal.far_field.intensity, retained.far_field.intensity)
    assert minimal.launch_identity == retained.launch_identity
    assert minimal.ledger == retained.ledger
    assert all(row['material_weight_um'] == 2. for row in minimal.ledger)


@pytest.mark.parametrize('when', [0, 1, 2])
def test_cancellation_at_accepted_boundaries(when):
    req = request(n=3)
    token = CancellationToken()
    if when == 0:
        token.cancel()
    def progress(p):
        if p.completed_cells == when:
            token.cancel()
    out = flow.run_local_intensity_planes(req, cancellation_token=token,
        retain_boundary_field=True, progress_callback=progress)
    assert out.status == 'cancelled'
    assert out.completed_cells == when and out.reached_z_um == 2.*when
    assert out.far_field is None and out.far_field_z_um is None
    assert len(out.ledger) == when
    if when == 0:
        np.testing.assert_array_equal(out.boundary_field, req.initial_A)
    else:
        # Keep original L/Gamma, even when constructing only the accepted prefix.
        real_step = step_local_intensity_cell
        backend = get_backend(req.backend)
        grid = make_grid(req.grid, xp=np, real_dtype=backend.real_dtype)
        A = req.initial_A.copy()
        ref = channel_peak_intensity_reference(A, xp=np)
        for k in range(when):
            A = real_step(A, grid=grid, cell_index=k, z_start_um=2.*k, dz_um=2.,
                interaction_length_um=6., wavelength_um=.633, material=req.material,
                peak_intensity_reference=ref, backend=req.backend).A_candidate
        np.testing.assert_array_equal(out.boundary_field, A)


@pytest.mark.parametrize('k', [0, 2])
def test_complete_candidate_discarded_when_cancelled(monkeypatch, k):
    req = request(n=3)
    token = CancellationToken()
    before = []
    real_step = flow.step_local_intensity_cell
    def cancel(A, **kw):
        if kw['cell_index'] == k:
            before.append(A.copy())
        cell = real_step(A, **kw)
        if kw['cell_index'] == k:
            token.cancel()
        return cell
    monkeypatch.setattr(flow, 'step_local_intensity_cell', cancel)
    out = flow.run_local_intensity_planes(req, cancellation_token=token, retain_boundary_field=True)
    assert out.status == 'cancelled' and out.completed_cells == k
    assert out.reached_z_um == 2.*k
    np.testing.assert_array_equal(out.boundary_field, before[0])
    assert out.far_field is None


@pytest.mark.parametrize('kind', ['source', 'residual', 'scattering', 'optical'])
def test_failure_preserves_previous_boundary(monkeypatch, kind):
    from lcprop.pr import local_plane_step as cell_module
    req = request(n=3, scattering=True)
    before, attempts = [], []
    original_step = flow.step_local_intensity_cell
    def bad(*a, **kw):
        raise ValueError({'source': 'local total transport intensity must be finite and strictly positive',
                          'residual': 'material residual exceeds cell validity limits',
                          'scattering': 'scattering phase must be finite',
                          'optical': 'injected optical failure'}[kind])
    name = {'source': 'pr_driving_intensity', 'residual': 'solve_pr_reduced_field_linear_intensity',
            'scattering': 'canonical_scattering_phase_increment', 'optical': 'hop_linear_inplace'}[kind]
    def wrapped(A, **kw):
        attempts.append(kw['cell_index'])
        if kw['cell_index'] == 1:
            before.append(A.copy())
            with monkeypatch.context() as m:
                m.setattr(cell_module, name, bad)
                return original_step(A, **kw)
        return original_step(A, **kw)
    monkeypatch.setattr(flow, 'step_local_intensity_cell', wrapped)
    out = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    assert out.status == 'failed' and out.completed_cells == 1
    assert out.reached_z_um == 2. and len(out.ledger) == 1
    assert out.failure['stage'] == 'cell_step' and out.failure['cell_index'] == 1
    assert out.failure['reason'] == out.reason
    assert out.failure['exception_type'] == 'ValueError'
    assert out.failure['kind'] == ('material_invalid' if kind in ('source', 'residual') else
                                  'scattering_failure' if kind == 'scattering' else 'exception')
    np.testing.assert_array_equal(out.boundary_field, before[0])
    assert attempts == [0, 1] and out.far_field is None


@pytest.mark.parametrize('stage', ['launch', 'final_far_field'])
def test_launch_and_product_failure_evidence(monkeypatch, stage):
    def fail(*a, **kw):
        raise RuntimeError('injected '+stage)
    monkeypatch.setattr(flow, 'build_launch' if stage == 'launch' else 'direction_cosine_spectrum', fail)
    req = request()
    out = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    assert out.status == 'failed' and out.failure['stage'] == stage
    assert out.far_field is None
    if stage == 'launch':
        assert out.completed_cells == 0 and out.boundary_field is None and out.launch_identity is None
    else:
        assert out.completed_cells == 2 and out.reached_z_um == 4.
        _, expected, _ = explicit_march(req)
        np.testing.assert_array_equal(out.boundary_field, expected)


@pytest.mark.parametrize('length,h,n', [(4000., 50., 80), (.3, .1, 3), (.15, .05, 3),
                                      (1., .2, 5), (3999.9, 1333.3, 3), (.375, .125, 3)])
def test_valid_domain(length, h, n):
    assert flow.validate_local_plane_domain(length, h) == n


@pytest.mark.parametrize('length,h', [(1., .3), (.31, .1), (.29, .1), (1., 2.),
                                     (math.inf, .1), (1., 0.), (0., 1.),
                                     (float(np.nextafter(np.nextafter(np.nextafter(np.nextafter(.3, np.inf), np.inf), np.inf), np.inf)), .1)])
def test_invalid_domain(length, h):
    with pytest.raises(ValueError):
        flow.validate_local_plane_domain(length, h)


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_fractional_full_workflow(precision):
    req = request(n=3, h=.1, precision=precision)
    req = replace(req, grid=replace(req.grid, z_length_um=.3), scattering=
                  PRCanonicalScatteringSpec(.02, .4, 0, .1, PR_CANONICAL_SCATTERING_V2))
    out = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    assert out.status == 'completed', out.failure
    assert out.completed_cells == 3 and out.reached_z_um == .3
    assert [(r['scattering']['slab_start'], r['scattering']['slab_stop_exclusive'])
            for r in out.ledger] == [(0, 1), (1, 2), (2, 3)]
    assert all(r['material_weight_um'] == .1 for r in out.ledger)


@pytest.mark.parametrize('kind', ['domain', 'model', 'boundary', 'v1', 'unaligned'])
def test_rejections_before_launch(monkeypatch, kind):
    req = request()
    if kind == 'domain':
        req = replace(req, grid=replace(req.grid, z_length_um=4.1))
    elif kind == 'model':
        req = replace(req, material_response=PRTransverseMaterialResponseSpec(model='nonlinear'))
    elif kind == 'boundary':
        req = replace(req, optical_boundary=TransverseBoundarySpec(mode='tukey'))
    else:
        req = replace(req, scattering=PRCanonicalScatteringSpec(.02, .4, 0, 3. if kind == 'unaligned' else 1.))
        if kind == 'unaligned':
            req = replace(req, scattering=replace(req.scattering, algorithm_version=PR_CANONICAL_SCATTERING_V2))
    monkeypatch.setattr(flow, 'build_launch', lambda *a, **k: pytest.fail('invalid request reached launch'))
    out = flow.run_local_intensity_planes(req)
    assert out.status == 'failed' and out.completed_cells == 0 and not out.ledger
    assert out.boundary_field is None and out.far_field is None


@pytest.mark.parametrize('precision,tol', [('float32', 3e-5), ('float64', 1e-12)])
def test_zero_gain_constant_mode_total_distance(precision, tol):
    req = request(n=3, h=.1, precision=precision)
    req.initial_A[:] = 1
    req = replace(req, material=replace(req.material, gain_length_product=0.),
                  grid=replace(req.grid, z_length_um=.3))
    out = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    assert out.status == 'completed', out.failure
    expected = np.exp(2j*np.pi*2.4/.633*.3)
    np.testing.assert_allclose(out.boundary_field, expected, atol=tol, rtol=tol)
    assert out.reached_z_um == .3


def test_bounded_ownership_and_compact_ledger(monkeypatch):
    req = request(n=30)
    refs, peaks = [], []
    real_step = flow.step_local_intensity_cell
    def watch(*a, **kw):
        result = real_step(*a, **kw)
        refs.extend(weakref.ref(x) for x in (result.A_candidate, result.E, result.source_intensity))
        peaks.append(sum(r() is not None for r in refs))
        return result
    monkeypatch.setattr(flow, 'step_local_intensity_cell', watch)
    for name in ('zeros', 'empty', 'ones', 'full'):
        original = getattr(np, name)
        def guard(shape, *a, _original=original, **kw):
            if isinstance(shape, tuple):
                assert shape != (30, 16, 8), 'longitudinal plane stack'
            return _original(shape, *a, **kw)
        monkeypatch.setattr(np, name, guard)
    out = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    assert out.status == 'completed', out.failure
    assert max(peaks) <= 4  # previous accepted optical field + current three outputs
    assert sum(r() is not None for r in refs) == 1
    def no_arrays(v):
        assert not isinstance(v, np.ndarray)
        if isinstance(v, dict):
            for x in v.values(): no_arrays(x)
        elif isinstance(v, (tuple, list)):
            for x in v: no_arrays(x)
    no_arrays(out.ledger)
    assert len(out.ledger) == 30
    snapshot = out.boundary_field.copy()
    flow.run_local_intensity_planes(req)
    np.testing.assert_array_equal(out.boundary_field, snapshot)


def test_hash_transfers_are_bounded(monkeypatch):
    seen = []
    def transfer(a):
        seen.append(a.nbytes)
        return np.asarray(a)
    monkeypatch.setattr(flow, 'asnumpy', transfer)
    array = np.ones((1, 512, 512), dtype=np.complex128)
    assert flow._launch_hash(array) == hashlib.sha256(array.tobytes()).hexdigest()
    assert max(seen) <= 1024**2 and sum(seen) == array.nbytes


def test_old_static_request_not_reinterpreted_and_old_workflow_isolated(monkeypatch):
    from lcprop.pr.static_workflow import PRStaticRunRequest, run_pr_static
    req = request(n=1)
    old = PRStaticRunRequest(grid=req.grid, beams=req.beams, material=replace(req.material, gain_length_product=0.),
                            backend=req.backend, initial_A=np.ones_like(req.initial_A), material_response=req.material_response)
    rejected = flow.run_local_intensity_planes(old)
    assert rejected.status == 'failed' and rejected.failure['stage'] == 'request_validation'
    monkeypatch.setattr(flow, 'run_local_intensity_planes', lambda *a, **k: pytest.fail('old Static routed to new workflow'))
    before = run_pr_static(old)
    monkeypatch.setattr(flow, 'step_local_intensity_cell', lambda *a, **k: pytest.fail('old Static called new step'))
    after = run_pr_static(old)
    assert before.converged and after.converged
    np.testing.assert_array_equal(before.A_final, after.A_final)
    np.testing.assert_array_equal(before.E_final, after.E_final)


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_native_workflow_when_available(precision):
    try:
        get_backend(BackendSpec('cupy', precision, False))
    except RuntimeError:
        pytest.skip('CuPy/CUDA unavailable: streaming workflow parity')
    req = request(precision=precision, scattering=True)
    cpu = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    gpu = flow.run_local_intensity_planes(replace(req, backend=BackendSpec('cupy', precision, False)), retain_boundary_field=True)
    assert cpu.status == gpu.status == 'completed', gpu.failure
    tol = 3e-5 if precision == 'float32' else 1e-11
    import cupy as cp
    np.testing.assert_allclose(cp.asnumpy(gpu.boundary_field), cpu.boundary_field, rtol=tol, atol=tol)
    np.testing.assert_allclose(cp.asnumpy(gpu.far_field.intensity), cpu.far_field.intensity, rtol=tol, atol=tol)


@pytest.mark.parametrize('precision,tol', [('float32', 3e-5), ('float64', 1e-12)])
def test_prescribed_constant_material_phase(precision, tol, monkeypatch):
    from types import SimpleNamespace
    from lcprop.pr import local_plane_step as cell_module
    req = request(n=5, h=.1, precision=precision)
    req.initial_A[:] = 1
    def constant(I, **kw):
        return SimpleNamespace(E=np.full_like(I, .37), residual=np.zeros_like(I))
    monkeypatch.setattr(cell_module, 'solve_pr_reduced_field_linear_intensity', constant)
    out = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    expected = np.exp(1j*(2*np.pi*2.4/.633*.5 - 2*.2*.37))
    assert out.status == 'completed', out.failure
    np.testing.assert_allclose(out.boundary_field, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize('bad', ['zero_source', 'nonfinite_material'])
def test_real_material_validity_gates_return_compact_failure(monkeypatch, bad):
    from types import SimpleNamespace
    from lcprop.pr import local_plane_step as cell_module
    req = request(n=1)
    if bad == 'zero_source':
        monkeypatch.setattr(cell_module, 'pr_driving_intensity', lambda A, **kw: np.zeros(A.shape[1:]))
    else:
        monkeypatch.setattr(cell_module, 'solve_pr_reduced_field_linear_intensity',
            lambda I, **kw: SimpleNamespace(E=np.full_like(I, np.nan), residual=np.zeros_like(I)))
    out = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    assert out.status == 'failed' and out.completed_cells == 0 and out.reached_z_um == 0.
    assert out.failure['kind'] == 'material_invalid'
    assert out.failure['stage'] == 'cell_step' and out.failure['cell_index'] == 0
    np.testing.assert_array_equal(out.boundary_field, req.initial_A)


def test_ledger_cannot_capture_arrays(monkeypatch):
    real_step = flow.step_local_intensity_cell
    def capture(*args, **kw):
        cell = real_step(*args, **kw)
        cell.diagnostics['bad_array'] = cell.source_intensity
        return cell
    monkeypatch.setattr(flow, 'step_local_intensity_cell', capture)
    req = request(n=1)
    out = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    assert out.status == 'failed' and not out.ledger and out.completed_cells == 0
    assert out.failure['stage'] == 'cell_diagnostics'
    np.testing.assert_array_equal(out.boundary_field, req.initial_A)


def test_cancellation_during_final_product_does_not_publish_spectrum(monkeypatch):
    token = CancellationToken()
    real_spectrum = flow.direction_cosine_spectrum
    def cancel(*args, **kw):
        spectrum = real_spectrum(*args, **kw)
        token.cancel()
        return spectrum
    monkeypatch.setattr(flow, 'direction_cosine_spectrum', cancel)
    out = flow.run_local_intensity_planes(request(), cancellation_token=token, retain_boundary_field=True)
    assert out.status == 'cancelled' and out.completed_cells == 2 and out.reached_z_um == 4.
    assert out.boundary_field is not None and out.far_field is None and out.far_field_z_um is None


def test_compact_records_accept_scalar_metadata_but_not_planes():
    req = request(scattering=True)
    req = replace(req, scattering=replace(req.scattering, epsilon=np.float32(.02), realization_seed=np.int64(0)))
    out = flow.run_local_intensity_planes(req)
    assert out.status == 'completed', out.failure
    spec = out.ledger[0]['scattering']['spec']
    assert type(spec['epsilon']) is float and type(spec['realization_seed']) is int
    with pytest.raises(TypeError):
        flow._compact({'plane': np.ones((2, 2))})


@pytest.mark.parametrize('fail_cell', [0, 1])
@pytest.mark.parametrize('append_first', [False, True])
def test_bookkeeping_failure_is_atomic(monkeypatch, fail_cell, append_first):
    req = request(n=3)
    primitive = flow.step_local_intensity_cell
    expected = req.initial_A.copy()
    calls = []
    def step(*args, **kwargs):
        nonlocal expected
        cell = primitive(*args, **kwargs)
        calls.append(kwargs['cell_index'])
        if kwargs['cell_index'] < fail_cell:
            expected = cell.A_candidate.copy()
        return cell
    def append(ledger, record):
        if record['cell_index'] == fail_cell:
            if append_first:
                ledger.append(record)
            raise MemoryError('injected bookkeeping failure after successful cell')
        ledger.append(record)
    monkeypatch.setattr(flow, 'step_local_intensity_cell', step)
    monkeypatch.setattr(flow, '_append_cell_record', append)
    out = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    assert out.status == 'failed'
    assert out.failure['stage'] == 'cell_bookkeeping'
    assert out.failure['cell_index'] == fail_cell
    assert 'injected bookkeeping failure' in out.reason
    assert out.completed_cells == len(out.ledger) == fail_cell
    assert out.reached_z_um == fail_cell * req.grid.dz_um
    assert out.boundary_field.tobytes() == expected.tobytes()
    assert calls == list(range(fail_cell + 1))
    assert out.far_field is None


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('scattering', [False, True])
def test_observation_exactness_coordinates_and_explicit_copy_lifetime(monkeypatch, precision, scattering):
    req = request(n=3, precision=precision, scattering=scattering)
    plain = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    primitive = flow.step_local_intensity_cell
    expected, copied, expired, refs = [], [], [], []
    def step(*args, **kwargs):
        cell = primitive(*args, **kwargs)
        expected.append(tuple(a.tobytes() for a in (cell.A_candidate, cell.source_intensity, cell.E)))
        return cell
    def observe(frame):
        k = len(copied)
        assert frame.cell_index == k
        assert frame.z_start_um == 2*k
        assert frame.z_end_um == 2*(k+1)
        assert frame.material_plane_um == 2*k+1
        arrays = (frame.boundary_field, frame.source_intensity, frame.material_field)
        copied.append(tuple(a.copy() for a in arrays))
        refs.extend(weakref.ref(a) for a in arrays)
        expired.append(frame)
        return False  # Never a scientific/acceptance decision.
    monkeypatch.setattr(flow, 'step_local_intensity_cell', step)
    watched = flow.run_local_intensity_planes(req, retain_boundary_field=True, observation_callback=observe)
    assert watched.status == plain.status == 'completed'
    assert watched.completed_cells == plain.completed_cells == 3
    assert watched.ledger == plain.ledger
    assert watched.launch_identity == plain.launch_identity
    assert watched.boundary_field.tobytes() == plain.boundary_field.tobytes()
    for name in ('intensity', 's_x', 's_y'):
        assert getattr(watched.far_field, name).tobytes() == getattr(plain.far_field, name).tobytes()
    assert [tuple(a.tobytes() for a in group) for group in copied] == expected
    assert all(ref() is None for ref in refs)
    for frame in expired:
        for name in ('boundary_field', 'source_intensity', 'material_field'):
            with pytest.raises(RuntimeError, match='expired'):
                getattr(frame, name)


@pytest.mark.parametrize('name', ['boundary_field', 'source_intensity', 'material_field'])
def test_observation_mutation_fails_closed_without_changing_science(monkeypatch, name):
    req = request(n=3)
    primitive = flow.step_local_intensity_cell
    candidates = []
    def step(*args, **kwargs):
        cell = primitive(*args, **kwargs)
        candidates.append(cell.A_candidate.copy())
        return cell
    def observe(frame):
        if frame.cell_index == 1:
            getattr(frame, name).reshape(-1).view(np.uint8)[0] ^= np.uint8(1)
    monkeypatch.setattr(flow, 'step_local_intensity_cell', step)
    out = flow.run_local_intensity_planes(req, retain_boundary_field=True, observation_callback=observe)
    assert out.status == 'failed'
    assert out.failure['stage'] == 'cell_observation'
    assert 'mutated' in out.reason
    assert out.completed_cells == len(out.ledger) == 1
    assert out.reached_z_um == 2.
    assert len(candidates) == 2
    assert out.boundary_field.tobytes() == candidates[0].tobytes()
    assert out.far_field is None


@pytest.mark.parametrize('action', ['raise', 'cancel'])
def test_observation_failure_or_cancellation_preserves_previous_boundary(action):
    req = request(n=3)
    token = CancellationToken()
    frames = []
    def observe(frame):
        frames.append(frame)
        if action == 'raise':
            raise RuntimeError('observation failed')
        token.cancel()
    out = flow.run_local_intensity_planes(req, retain_boundary_field=True,
        cancellation_token=token, observation_callback=observe)
    assert out.status == ('failed' if action == 'raise' else 'cancelled')
    assert out.completed_cells == len(out.ledger) == 0
    assert out.reached_z_um == 0.
    assert out.boundary_field.tobytes() == req.initial_A.tobytes()
    assert len(frames) == 1
    with pytest.raises(RuntimeError, match='expired'):
        frames[0].boundary_field


def test_no_hook_makes_no_observation_copies(monkeypatch):
    monkeypatch.setattr(flow, '_observe_cell', lambda *a: pytest.fail('unrequested observation'))
    out = flow.run_local_intensity_planes(request())
    assert out.status == 'completed', out.failure


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_native_observation_when_available(precision):
    try:
        backend = get_backend(BackendSpec('cupy', precision, False))
    except RuntimeError:
        pytest.skip('CuPy/CUDA unavailable: observation exactness and mutation guard')
    req = replace(request(precision=precision), backend=BackendSpec('cupy', precision, False))
    plain = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    copied = []
    watched = flow.run_local_intensity_planes(req, retain_boundary_field=True,
        observation_callback=lambda frame: copied.append(frame.source_intensity.copy()))
    assert plain.status == watched.status == 'completed'
    xp = backend.xp
    assert bool(xp.array_equal(plain.boundary_field.view(xp.uint8), watched.boundary_field.view(xp.uint8)))
    assert bool(xp.array_equal(plain.far_field.intensity.view(xp.uint8), watched.far_field.intensity.view(xp.uint8)))
    assert plain.ledger == watched.ledger
    assert len(copied) == req.grid.z_length_um / req.grid.dz_um
    def mutate(frame):
        frame.material_field[:] = 0
    failed = flow.run_local_intensity_planes(req, observation_callback=mutate)
    assert failed.status == 'failed' and failed.completed_cells == 0
    assert failed.failure['stage'] == 'cell_observation'
