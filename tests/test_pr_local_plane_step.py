"""Independent algebra, operation tracing and transactional cell-step tests."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from lcprop.core.backend import BackendSpec, get_backend
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.pr import local_plane_step as step
from lcprop.pr.scattering import PRCanonicalScatteringSpec, PR_CANONICAL_SCATTERING_V2
from lcprop.pr.specs import PRMaterialSpec


def case(precision='float64', backend='numpy', n=2, h=2.0):
    spec = BackendSpec(backend, precision, False)
    resolved = get_backend(spec)
    xp = resolved.xp
    grid = make_grid(GridSpec(Nx=16, Ny=6, x_aperture_um=8.,
                             y_aperture_um=6., z_length_um=n*h, dz_um=h),
                     xp=xp, real_dtype=resolved.real_dtype)
    x = xp.arange(16)[:, None]
    A = xp.broadcast_to(1 + .3*xp.exp(2j*xp.pi*x/16), (16, 6))[None].copy()
    A = A.astype(resolved.complex_dtype)
    options = dict(grid=grid, cell_index=0, z_start_um=0., dz_um=h,
                   interaction_length_um=n*h, wavelength_um=.633,
                   material=PRMaterialSpec(gain_length_product=.2,
                                          characteristic_wavenumber_per_um_override=1.),
                   peak_intensity_reference=2.0, backend=spec)
    return A, options


def oracle_response(monkeypatch, value):
    def solve(I, **kw):
        xp = get_backend(kw['backend']).xp
        return SimpleNamespace(E=xp.full_like(I, value), residual=xp.zeros_like(I))
    monkeypatch.setattr(step, 'solve_pr_reduced_field_linear_intensity', solve)


def noise_spec():
    return PRCanonicalScatteringSpec(.02, .4, 0, 1., PR_CANONICAL_SCATTERING_V2)


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('n', [1, 2, 5])
def test_order_source_and_cell_accounting(monkeypatch, precision, n):
    A, kw = case(precision, n=n)
    original = A.copy()
    events, distances, received_sources, expected_sources, screens = [], [], [], [], []
    original_hop, original_source = step.hop_linear_inplace, step.pr_driving_intensity
    original_solve, original_apply = step.solve_pr_reduced_field_linear_intensity, step.apply_response_screen_inplace
    original_kernel = step.scalar_angular_spectrum_kernel
    def kernel(*a, **k):
        distances.append(k['dz'])
        return original_kernel(*a, **k)
    def hop(A, *a, **k):
        events.append('hop')
        return original_hop(A, *a, **k)
    def source(A, **k):
        events.append('source')
        I = original_source(A, **k)
        expected_sources.append(I.copy())
        return I
    def solve(I, **k):
        events.append('solve')
        received_sources.append(I.copy())
        return original_solve(I, **k)
    def scatter(*a, **k):
        events.append('scatter_generate')
        screens.append((k['z_start_um'], k['dz_um']))
        return np.full((16, 6), .17, dtype=precision)
    def apply(*a, **k):
        events.append('apply')
        return original_apply(*a, **k)
    for name, fn in [('scalar_angular_spectrum_kernel', kernel), ('hop_linear_inplace', hop),
                     ('pr_driving_intensity', source), ('solve_pr_reduced_field_linear_intensity', solve),
                     ('canonical_scattering_phase_increment', scatter), ('apply_response_screen_inplace', apply)]:
        monkeypatch.setattr(step, name, fn)
    records = []
    for j in range(n):
        result = step.step_local_intensity_cell(A, **dict(kw, cell_index=j, z_start_um=j*2., scattering=noise_spec()))
        A = result.A_candidate
        records.append(result.diagnostics)
    assert events == ['hop', 'source', 'solve', 'apply', 'scatter_generate', 'apply', 'hop'] * n
    assert distances == [1.] * n  # kernel reused for each cell's two separate hops
    assert len([e for e in events if e == 'hop']) * distances[0] == n*2.
    assert sum(r['material_weight_um'] for r in records) == n*2.
    assert [r['material_plane_um'] for r in records] == [1.+2*j for j in range(n)]
    assert screens == [(2.*j, 2.) for j in range(n)]
    assert [(r['scattering']['slab_start'], r['scattering']['slab_stop_exclusive']) for r in records] == [(2*j, 2*j+2) for j in range(n)]
    assert records[-1]['z_end_um'] == n*2.
    for actual, expected in zip(received_sources, expected_sources):
        np.testing.assert_array_equal(actual, expected)
    initial_I = original_source(original, peak_intensity_reference=2., background_intensity=.01, xp=np)
    assert not np.allclose(initial_I, received_sources[0], rtol=1e-7, atol=1e-7)
    assert A.dtype == original.dtype


@pytest.mark.parametrize('precision,tol', [('float32', 4e-5), ('float64', 3e-11)])
@pytest.mark.parametrize('n', [1, 2, 7, 80])
def test_constant_material_full_phase_and_uniform_noise(monkeypatch, precision, tol, n):
    A, kw = case(precision, n=n, h=50.)
    A[:] = 1
    kw['material'] = replace(kw['material'], gain_length_product=10.)
    oracle_response(monkeypatch, .37)
    monkeypatch.setattr(step, 'canonical_scattering_phase_increment',
                        lambda *a, **k: np.full((16, 6), .13, dtype=precision))
    for j in range(n):
        A = step.step_local_intensity_cell(A, **dict(kw, cell_index=j, z_start_um=50.*j,
                                                    scattering=noise_spec())).A_candidate
    expected = np.exp(1j*(2*np.pi*2.4/.633*(n*50.) - 2*10*.37 + n*.13))
    np.testing.assert_allclose(A, expected, rtol=tol, atol=tol)


def test_a5_material_coefficient_independent():
    E = np.array([-.3, 0., .7])
    dn = step.delta_n_from_E(E, gain_length_product=10., interaction_length_um=4000., wavelength_um=.633)
    np.testing.assert_allclose((2*np.pi/.633)*50.*dn, -.25*E, atol=1e-16, rtol=1e-15)


@pytest.mark.parametrize('precision,tol', [('float32', 2e-5), ('float64', 1e-12)])
@pytest.mark.parametrize('mode', [0, 1, 8])
def test_zero_response_supported_modes_and_hard_cutoff(monkeypatch, precision, tol, mode):
    A, kw = case(precision, n=3)
    kw['wavelength_um'] = 3.0  # mode 8: f=1 > n/lambda=.8, thus cut off
    A[:] = np.exp(2j*np.pi*mode*np.arange(16)/16)[None, :, None]
    incoming = A.copy()
    oracle_response(monkeypatch, 0.)
    for j in range(3):
        A = step.step_local_intensity_cell(A, **dict(kw, cell_index=j, z_start_um=2.*j)).A_candidate
    q = 1 - (3./2.4)**2*(mode/8.)**2
    expected = incoming*np.exp(2j*np.pi*2.4/3.*6.*np.sqrt(q)) if q >= 0 else np.zeros_like(A)
    np.testing.assert_allclose(A, expected, atol=tol, rtol=tol)


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_actual_scattering_exact_manual_cell_and_nonmutation(precision):
    A, kw = case(precision)
    before = A.copy()
    xp = np
    kernel = step.scalar_angular_spectrum_kernel(kw['grid'].fxy2_um, dz=1., wavelength=.633,
                                                n_ref=2.4, complex_dtype=A.dtype, xp=xp)
    B = A.copy()
    step.hop_linear_inplace(B, kernel, xp=xp)
    I = step.pr_driving_intensity(B, peak_intensity_reference=2., background_intensity=.01, xp=xp)
    material = step.solve_pr_reduced_field_linear_intensity(I, spec=step.PRReducedFieldLinearSpec(0., .01, .5), backend=kw['backend'])
    dn = step.delta_n_from_E(material.E, gain_length_product=.2, interaction_length_um=4., wavelength_um=.633)
    B *= np.exp(1j*(2*np.pi/.633)*2.*dn)[None]
    phi = step.canonical_scattering_phase_increment(noise_spec(), z_start_um=0., dz_um=2., z_length_um=4.,
        Nx=16, Ny=6, x_aperture_um=8., y_aperture_um=6., real_dtype=np.dtype(precision), xp=xp)
    B *= np.exp(1j*phi)[None]
    step.hop_linear_inplace(B, kernel, xp=xp)
    result = step.step_local_intensity_cell(A, **kw, scattering=noise_spec())
    np.testing.assert_array_equal(result.A_candidate, B)
    np.testing.assert_array_equal(result.E, material.E)
    np.testing.assert_array_equal(result.source_intensity, I)
    np.testing.assert_array_equal(A, before)
    assert not np.shares_memory(A, result.A_candidate)
    assert not np.shares_memory(A, result.E)
    assert result.diagnostics['source_min'] > .01


@pytest.mark.parametrize('stage', ['first_hop', 'source', 'solve', 'residual', 'material_phase', 'material_apply', 'scattering', 'second_hop'])
def test_failures_never_mutate_accepted_input(monkeypatch, stage):
    A, kw = case()
    before = A.copy()
    def fail(*a, **k):
        raise RuntimeError('injected '+stage)
    if stage in ('first_hop', 'second_hop'):
        original = step.hop_linear_inplace
        calls = []
        def hop(A, *a, **k):
            calls.append(1)
            if len(calls) == (1 if stage == 'first_hop' else 2):
                A[:] = 99  # destructive failure affects only the scratch field
                fail()
            return original(A, *a, **k)
        monkeypatch.setattr(step, 'hop_linear_inplace', hop)
    elif stage == 'residual':
        def invalid(I, **kw):
            return SimpleNamespace(E=I.copy(), residual=np.full_like(I, 1.))
        monkeypatch.setattr(step, 'solve_pr_reduced_field_linear_intensity', invalid)
    else:
        name = {'source': 'pr_driving_intensity', 'solve': 'solve_pr_reduced_field_linear_intensity',
                'material_phase': 'delta_n_from_E', 'material_apply': 'apply_response_screen_inplace',
                'scattering': 'canonical_scattering_phase_increment'}[stage]
        monkeypatch.setattr(step, name, fail)
    with pytest.raises((RuntimeError, ValueError)):
        step.step_local_intensity_cell(A, **kw, scattering=noise_spec())
    np.testing.assert_array_equal(A, before)


@pytest.mark.parametrize('which', ['E', 'residual'])
def test_nonfinite_material_fails_closed(monkeypatch, which):
    A, kw = case()
    def invalid(I, **kw):
        arrays = dict(E=np.zeros_like(I), residual=np.zeros_like(I))
        arrays[which][0, 0] = np.nan
        return SimpleNamespace(**arrays)
    monkeypatch.setattr(step, 'solve_pr_reduced_field_linear_intensity', invalid)
    with pytest.raises(ValueError, match='finite'):
        step.step_local_intensity_cell(A, **kw)


@pytest.mark.parametrize('change', [dict(dz_um=0), dict(z_start_um=-1), dict(dz_um=5),
                                     dict(peak_intensity_reference=0), dict(residual_max_tolerance=-1)])
def test_invalid_metadata(change):
    A, kw = case()
    with pytest.raises(ValueError):
        step.step_local_intensity_cell(A, **dict(kw, **change))


def test_disabled_scattering_is_identity():
    A, kw = case()
    absent = step.step_local_intensity_cell(A, **kw)
    disabled = step.step_local_intensity_cell(A, **kw, scattering=replace(noise_spec(), epsilon=0.))
    np.testing.assert_array_equal(absent.A_candidate, disabled.A_candidate)
    assert absent.diagnostics['scattering'] is None


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_native_parity_residency_when_available(monkeypatch, precision):
    try:
        resolved = get_backend(BackendSpec('cupy', precision, False))
    except RuntimeError:
        pytest.skip('CuPy/CUDA unavailable: native cell parity and host-transfer guard')
    cp = resolved.xp
    A, kw = case(precision, backend='cupy')
    cpu, cpu_kw = case(precision)
    expected = step.step_local_intensity_cell(cpu, **cpu_kw, scattering=noise_spec())
    before = A.copy()
    def no_transfer(*a, **k):
        raise AssertionError('full-plane host transfer')
    with monkeypatch.context() as m:
        m.setattr(cp, 'asnumpy', no_transfer)
        actual = step.step_local_intensity_cell(A, **kw, scattering=noise_spec())
    assert isinstance(actual.A_candidate, cp.ndarray) and isinstance(actual.E, cp.ndarray)
    assert bool(cp.all(A == before).item())
    tol = 3e-5 if precision == 'float32' else 1e-11
    np.testing.assert_allclose(cp.asnumpy(actual.A_candidate), expected.A_candidate, atol=tol, rtol=tol)
    np.testing.assert_allclose(cp.asnumpy(actual.E), expected.E, atol=tol, rtol=tol)


def test_only_transverse_arrays_returned_and_no_host_conversion(monkeypatch):
    A, kw = case(n=100)
    allocations = []
    for name in ('empty', 'zeros', 'ones', 'full'):
        original = getattr(np, name)
        def record(shape, *args, _original=original, **kwargs):
            if isinstance(shape, tuple):
                allocations.append(shape)
                assert len(shape) <= 2, 'longitudinal allocation'
            return _original(shape, *args, **kwargs)
        monkeypatch.setattr(np, name, record)
    import lcprop.core.backend as backend_module
    monkeypatch.setattr(backend_module, 'asnumpy', lambda *a: pytest.fail('host transfer'))
    result = step.step_local_intensity_cell(A, **kw, scattering=noise_spec())
    assert result.A_candidate.shape == (1, 16, 6)
    assert result.E.shape == (16, 6)
    assert result.source_intensity.shape == (16, 6)
    assert all(not isinstance(v, np.ndarray) for v in result.diagnostics.values())
    assert allocations  # includes the canonical scattering workspace


@pytest.mark.parametrize('model', ['field_linear_local_intensity', 'nonlinear', 'td'])
def test_existing_workflows_do_not_dispatch_to_new_primitive(monkeypatch, model):
    from lcprop.core.beams import BeamChannel, BeamStack
    from lcprop.pr.static_workflow import PRStaticRunRequest, run_pr_static
    from lcprop.pr.workflow import run_pr_timedependent
    from lcprop.pr.specs import PRRunRequest, PRSolverOptions
    from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec
    monkeypatch.setattr(step, 'step_local_intensity_cell',
                        lambda *a, **k: pytest.fail('existing workflow dispatched to new primitive'))
    common = dict(grid=GridSpec(Nx=8, Ny=4, x_aperture_um=16., y_aperture_um=8.,
                               z_length_um=1., dz_um=1.),
                  beams=BeamStack(channels=(BeamChannel(wavelength_um=.633, w1_um=4., w2_um=4.),)),
                  material=PRMaterialSpec(gain_length_product=0.),
                  initial_A=np.ones((1, 8, 4), dtype=np.complex128),
                  backend=BackendSpec('numpy', 'float64', False))
    if model == 'td':
        result = run_pr_timedependent(PRRunRequest(**common, solver=PRSolverOptions(Nt=1)))
        assert result.completed_steps == 1
    else:
        result = run_pr_static(PRStaticRunRequest(**common,
            material_response=PRTransverseMaterialResponseSpec(model=model)))
        assert result.completed_slices == 1
        assert result.converged


def test_legacy_scattering_and_unaligned_cell_rejected():
    A, kw = case()
    with pytest.raises(ValueError, match='V2'):
        step.step_local_intensity_cell(A, **kw,
            scattering=replace(noise_spec(), algorithm_version='canonical_phase_slabs_v1'))
    with pytest.raises(ValueError, match='align'):
        step.step_local_intensity_cell(A, **dict(kw, dz_um=1.5), scattering=noise_spec())


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('h,length,quantum,n', [
    (.1, .3, .1, 3), (.05, .15, .05, 3), (.2, 1., .1, 5),
    (1333.3, 3999.9, .1, 3), (.125, .375, .125, 3),
])
def test_fractional_domain_accounting(monkeypatch, precision, h, length, quantum, n):
    from decimal import Decimal
    from lcprop.pr.scattering import canonical_slab_range
    A, kw = case(precision)
    kw.update(dz_um=h, interaction_length_um=length)
    scattering = replace(noise_spec(), epsilon=0., canonical_dz_um=quantum)
    kernel_distances, phase_weights, scattering_cells, records = [], [], [], []
    original_kernel = step.scalar_angular_spectrum_kernel
    original_dn = step.delta_n_from_E
    original_scatter = step.canonical_scattering_phase_increment
    def kernel(*args, **kwargs):
        kernel_distances.append(kwargs['dz'])
        return original_kernel(*args, **kwargs)
    def dn(E, **kwargs):
        phase_weights.append(kwargs['interaction_length_um'])
        return original_dn(E, **kwargs)
    def scatter(spec, **kwargs):
        scattering_cells.append((kwargs['z_start_um'], kwargs['dz_um'],
                                 canonical_slab_range(spec, z_start_um=kwargs['z_start_um'],
                                   dz_um=kwargs['dz_um'], z_length_um=kwargs['z_length_um'])))
        return original_scatter(spec, **kwargs)
    monkeypatch.setattr(step, 'scalar_angular_spectrum_kernel', kernel)
    monkeypatch.setattr(step, 'delta_n_from_E', dn)
    monkeypatch.setattr(step, 'canonical_scattering_phase_increment', scatter)
    for j in range(n):
        result = step.step_local_intensity_cell(A, **dict(kw, cell_index=j,
            z_start_um=j*h, scattering=scattering))
        A = result.A_candidate
        records.append(result.diagnostics)
    assert len(records) == n
    assert records[-1]['z_end_um'] == length
    assert all(r['material_weight_um'] == h for r in records)
    assert kernel_distances == [h/2]*n
    assert phase_weights == [length]*n
    # Decimal accounting is independent of binary summation; h was not shortened.
    assert sum(Decimal(str(r['material_weight_um'])) for r in records) == Decimal(str(length))
    assert sum(2*Decimal(str(d)) for d in kernel_distances) == Decimal(str(length))
    assert [(z, width) for z, width, _ in scattering_cells] == [(j*h, h) for j in range(n)]
    slab_count = int(Decimal(str(length))/Decimal(str(quantum)))
    assert scattering_cells[0][2].start == 0
    assert scattering_cells[-1][2].stop == slab_count
    assert all(left[2].stop == right[2].start for left, right in zip(scattering_cells, scattering_cells[1:]))
    for record, (_, _, slabs) in zip(records, scattering_cells):
        assert record['scattering']['slab_start'] == slabs.start
        assert record['scattering']['slab_stop_exclusive'] == slabs.stop
    if h == .1:
        assert records[-1]['computed_z_end_um'] == .2+.1
        assert records[-1]['endpoint_canonicalized'] is True
        assert records[-1]['material_weight_um'] != length-.2
    if h == .125:
        assert not any(r['endpoint_canonicalized'] for r in records)


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('kind', ['large', 'four_ulps'])
def test_true_endpoint_overshoot_fails_before_optics(monkeypatch, precision, kind):
    import math
    A, kw = case(precision)
    z, h, length = .2, .1, .3
    if kind == 'large':
        h = .11
    else:
        length = z+h
        for _ in range(4):
            length = math.nextafter(length, -math.inf)
        assert (z+h)-length > 2*max(math.ulp(v) for v in (z, h, z+h, length))
    monkeypatch.setattr(step, 'hop_linear_inplace', lambda *a, **k: pytest.fail('invalid domain propagated'))
    with pytest.raises(ValueError, match='original interaction domain'):
        step.step_local_intensity_cell(A, **dict(kw, z_start_um=z, dz_um=h, interaction_length_um=length))


def test_interior_endpoint_not_canonicalized():
    A, kw = case()
    result = step.step_local_intensity_cell(A, **dict(kw, z_start_um=.2, dz_um=.1, interaction_length_um=1.))
    assert result.diagnostics['z_end_um'] == .2+.1
    assert result.diagnostics['z_end_um'] != .3
    assert not result.diagnostics['endpoint_canonicalized']


@pytest.mark.parametrize('direction', [-np.inf, np.inf])
def test_one_ulp_boundary_discrepancy_canonicalizes_only_metadata(direction):
    A, kw = case()
    computed_end = .2+.1
    length = float(np.nextafter(computed_end, direction))
    result = step.step_local_intensity_cell(A, **dict(kw, z_start_um=.2, dz_um=.1, interaction_length_um=length))
    assert result.diagnostics['z_end_um'] == length
    assert result.diagnostics['material_weight_um'] == .1
    assert result.diagnostics['optical_half_step_um'] == .05


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('metadata_dtype', [np.float32, np.float64])
def test_coordinate_precision_is_host_binary64(precision, metadata_dtype):
    import math
    A, kw = case(precision)
    z, h, length = [metadata_dtype(v) for v in (.2, .1, .3)]
    result = step.step_local_intensity_cell(A, **dict(kw, z_start_um=z, dz_um=h, interaction_length_um=length))
    end = float(z)+float(h)
    bound = 2*max(math.ulp(v) for v in (float(z), float(h), end, float(length)))
    assert result.diagnostics['endpoint_roundoff_bound_um'] == bound
    if metadata_dtype == np.float32:
        # Widening does not undo lost input precision or grant float32 tolerance.
        assert end < float(length)-bound
        assert result.diagnostics['z_end_um'] == end
        assert not result.diagnostics['endpoint_canonicalized']
    else:
        assert result.diagnostics['z_end_um'] == float(length)


def test_float32_metadata_overshoot_not_hidden_by_field_precision():
    A, kw = case('float32')
    with pytest.raises(ValueError, match='original interaction domain'):
        step.step_local_intensity_cell(A, **dict(kw, z_start_um=np.float32(.4),
            dz_um=np.float32(.3), interaction_length_um=np.float32(.7)))


def test_roundoff_does_not_consume_neighboring_canonical_slab():
    import math
    A, kw = case()
    # A preceding slab boundary lies within two ULPs of L, but is a distinct
    # canonical address. Fail closed rather than silently claim the final slab.
    q = math.ulp(1.)
    with pytest.raises(ValueError, match='disagrees with scattering slab range'):
        step.step_local_intensity_cell(A, **dict(kw, z_start_um=1.-2*q, dz_um=q,
            interaction_length_um=1., scattering=replace(noise_spec(), canonical_dz_um=q)))
