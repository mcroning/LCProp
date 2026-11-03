"""Full stacks exist only in these independent, bounded reference oracles."""
from dataclasses import replace, fields
import io
import json
import weakref

import numpy as np
import pytest

from tests.test_pr_local_plane_workflow import request
from lcprop.core.backend import get_backend, BackendSpec
from lcprop.core.execution import CancellationToken
from lcprop.core.grid import make_grid
from lcprop.optics.farfield import direction_cosine_spectrum
from lcprop.pr import local_plane_workflow as flow
from lcprop.pr import local_plane_products as products
from lcprop.pr.local_plane_products import LocalPlaneProductSelection as Selection
from lcprop.pr.local_plane_products_codec import encode_local_plane_products, decode_local_plane_products
from lcprop.pr.local_plane_step import step_local_intensity_cell
from lcprop.pr.source import pr_driving_intensity, channel_peak_intensity_reference
from lcprop.pr.reduced_field_linear import PRReducedFieldLinearSpec, reduced_field_linear_residual
from lcprop.pr.longitudinal_cuts import extract_longitudinal_optical_intensity_cuts
from lcprop.pr.visualization import block_average_2d


def all_products():
    return Selection(**{f.name: True for f in fields(Selection)})


def reference(req):
    backend = get_backend(req.backend)
    grid = make_grid(req.grid, xp=np, real_dtype=backend.real_dtype)
    A = req.initial_A.copy()
    launch_reference = channel_peak_intensity_reference(A, xp=np)
    boundaries, sources, materials, residuals = [A.copy()], [], [], []
    for k in range(round(req.grid.z_length_um/req.grid.dz_um)):
        cell = step_local_intensity_cell(A, grid=grid, cell_index=k,
            z_start_um=k*req.grid.dz_um, dz_um=req.grid.dz_um,
            interaction_length_um=req.grid.z_length_um, wavelength_um=.633,
            material=req.material, peak_intensity_reference=launch_reference,
            backend=req.backend, scattering=req.scattering)
        A = cell.A_candidate
        boundaries.append(A.copy())
        sources.append(cell.source_intensity.copy())
        materials.append(cell.E.copy())
        residuals.append(reduced_field_linear_residual(cell.E, cell.source_intensity,
            spec=PRReducedFieldLinearSpec(req.material.applied_field, req.material.background_intensity,
                req.material.characteristic_wavenumber_per_um*grid.dx_um), xp=np))
    return grid, np.stack(boundaries), np.stack(sources), np.stack(materials), np.stack(residuals)


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('scattering', [False, True])
@pytest.mark.parametrize('accepted', [0, 1, 3])
def test_streaming_exact_full_stack_reference_and_codec(precision, scattering, accepted):
    req = request(n=3, precision=precision, scattering=scattering)
    grid, optical, source, material, residual = reference(req)
    token = CancellationToken()
    def progress(event):
        if accepted < 3 and event.completed_cells == accepted:
            token.cancel()
    result = products.run_local_plane_products(req, selection=all_products(),
        cancellation_token=token, progress_callback=progress)
    s, p = result.scientific, result.products
    assert s.status == ('completed' if accepted == 3 else 'cancelled'), s.failure
    assert s.completed_cells == accepted
    assert p.boundary_z_um.tolist() == [2*k for k in range(accepted+1)]
    assert p.center_z_um.tolist() == [2*k+1 for k in range(accepted)]
    ref = float(np.sum(np.max(np.abs(req.initial_A)**2, axis=(-2, -1))))
    assert p.presentation_reference == ref
    raw = np.stack([pr_driving_intensity(A, peak_intensity_reference=channel_peak_intensity_reference(req.initial_A, xp=np),
        background_intensity=req.material.background_intensity, xp=np) for A in optical[:accepted+1]])
    expected = extract_longitudinal_optical_intensity_cuts(raw, grid_summary=grid.summary(),
        peak_intensity_reference=ref, background_intensity=req.material.background_intensity)
    assert p.cuts['intensity'][0].tobytes() == expected.xz.tobytes()
    assert p.cuts['intensity'][1].tobytes() == expected.yz.tobytes()
    ix, iy = np.argmin(abs(p.x_um)), np.argmin(abs(p.y_um))
    for name, stack in [('optical', optical[:accepted+1]), ('source', source[:accepted]),
                         ('material', material[:accepted]), ('residual', residual[:accepted])]:
        assert p.cuts[name][0].tobytes() == stack[..., :, iy].tobytes()
        assert p.cuts[name][1].tobytes() == stack[..., ix, :].tobytes()
    maximum = min(96, int(np.sqrt(grid.Nx+grid.Ny)))
    preview = np.stack([block_average_2d((a.astype(np.float64)-req.material.background_intensity)*ref,
        max_x=maximum, max_y=maximum)[0].astype(np.float32) for a in raw])
    assert p.preview.tobytes() == preview.tobytes()
    assert s.boundary_field.tobytes() == optical[accepted].tobytes()
    if accepted == 3:
        expected_ff = direction_cosine_spectrum(optical[-1], dx_um=grid.dx_um, dy_um=grid.dy_um,
            wavelength_um=.633, refractive_index=req.material.refractive_index, xp=np)
        assert s.far_field.intensity.tobytes() == expected_ff.intensity.tobytes()
    else:
        assert s.far_field is None
    restored = decode_local_plane_products(encode_local_plane_products(result))
    for name in p.cuts:
        for a, b in zip(p.cuts[name], restored.products.cuts[name]):
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    for name in ('boundary_z_um', 'center_z_um', 'preview', 'preview_x_um', 'preview_y_um'):
        assert getattr(p, name).tobytes() == getattr(restored.products, name).tobytes()
    assert restored.scientific.boundary_field.tobytes() == s.boundary_field.tobytes()
    if s.far_field is not None:
        for name in ('intensity', 's_x', 's_y'):
            assert getattr(restored.scientific.far_field, name).tobytes() == getattr(s.far_field, name).tobytes()
    xe = np.linspace(0, grid.Nx, min(grid.Nx, maximum)+1, dtype=np.int64)
    ye = np.linspace(0, grid.Ny, min(grid.Ny, maximum)+1, dtype=np.int64)
    assert p.preview_x_um.tolist() == [float(np.mean(p.x_um[a:b])) for a, b in zip(xe[:-1], xe[1:])]
    assert p.preview_y_um.tolist() == [float(np.mean(p.y_um[a:b])) for a, b in zip(ye[:-1], ye[1:])]


@pytest.mark.parametrize('fault', ['bookkeeping', 'cancel', 'primitive', 'observation'])
def test_unaccepted_candidate_never_published(monkeypatch, fault):
    req = request(n=3)
    token = CancellationToken()
    original = flow._append_cell_record
    def append(ledger, record):
        if record['cell_index'] == 1 and fault == 'bookkeeping':
            ledger.append(record)
            raise MemoryError('after candidate observed')
        original(ledger, record)
    monkeypatch.setattr(flow, '_append_cell_record', append)
    observe = products._Collector.observe
    def hooked(self, frame):
        observe(self, frame)
        if frame.cell_index == 1:
            if fault == 'cancel': token.cancel()
            if fault == 'observation': raise RuntimeError('after product reduction')
    monkeypatch.setattr(products._Collector, 'observe', hooked)
    primitive = flow.step_local_intensity_cell
    def step(*a, **kw):
        if kw['cell_index'] == 1 and fault == 'primitive': raise ValueError('step failed')
        return primitive(*a, **kw)
    monkeypatch.setattr(flow, 'step_local_intensity_cell', step)
    out = products.run_local_plane_products(req, selection=all_products(), cancellation_token=token)
    assert out.scientific.status == ('cancelled' if fault == 'cancel' else 'failed')
    assert out.scientific.completed_cells == 1
    assert out.products.boundary_z_um.tolist() == [0., 2.]
    assert out.products.center_z_um.tolist() == [1.]
    assert out.products.preview.shape[0] == 2
    decode_local_plane_products(encode_local_plane_products(out))


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_selection_cannot_change_science(precision):
    req = request(n=3, precision=precision)
    bare = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    for selection in (all_products(), Selection(**{f.name: f.name == 'endpoint' for f in fields(Selection)})):
        out = products.run_local_plane_products(req, selection=selection)
        assert out.scientific.status == 'completed', out.scientific.failure
        assert out.scientific.boundary_field.tobytes() == bare.boundary_field.tobytes()
        assert out.scientific.ledger == bare.ledger
        assert out.scientific.launch_identity == bare.launch_identity


def test_bounded_reduced_storage_and_unrequested_work(monkeypatch):
    req = request(n=30)
    refs = []
    observe = products._Collector.observe
    def watched(self, frame):
        refs.append(weakref.ref(frame.boundary_field))
        observe(self, frame)
        assert sum(r() is not None for r in refs) == 1
        # Every retained/pending array is a line, or an explicitly reduced plane.
        records = self.records + [self.pending]
        for record in records:
            for pair in record['cuts'].values():
                for a in pair: assert a.size <= req.grid.Nx + req.grid.Ny
            if record['preview'] is not None:
                assert record['preview'].size <= req.grid.Nx + req.grid.Ny
    monkeypatch.setattr(products._Collector, 'observe', watched)
    out = products.run_local_plane_products(req, selection=all_products())
    assert out.scientific.status == 'completed', out.scientific.failure
    assert all(r() is None for r in refs)
    assert out.products.preview.shape[0] == 31
    def forbidden(*a, **kw): pytest.fail('unrequested product arithmetic')
    for name in ('presentation_peak_intensity_reference', 'pr_driving_intensity', 'reduced_field_linear_residual'):
        monkeypatch.setattr(products, name, forbidden)
    minimal = products.run_local_plane_products(req, selection=Selection(**{f.name: False for f in fields(Selection)}))
    assert minimal.scientific.status == 'completed', minimal.scientific.failure
    assert minimal.products.cuts == {} and minimal.products.preview is None


@pytest.mark.parametrize('fault', ['launch', 'launch_observer'])
def test_failed_launch_has_no_products(monkeypatch, fault):
    def fail(*a, **kw): raise ValueError('launch failure')
    monkeypatch.setattr(flow if fault == 'launch' else products._Collector,
                        'build_launch' if fault == 'launch' else 'observe', fail)
    out = products.run_local_plane_products(request())
    assert out.scientific.status == 'failed'
    assert out.products.boundary_z_um.size == out.products.center_z_um.size == 0
    assert out.products.preview is None
    decode_local_plane_products(encode_local_plane_products(out))


def test_codec_rejects_common_z_axis_and_corruption():
    out = products.run_local_plane_products(request(), selection=all_products())
    with pytest.raises(ValueError, match='count'):
        encode_local_plane_products(replace(out, products=replace(out.products,
            center_z_um=out.products.boundary_z_um.copy())))
    payload = encode_local_plane_products(out)
    with np.load(io.BytesIO(payload), allow_pickle=False) as source:
        arrays = {name: source[name] for name in source.files}
    arrays['intensity_xz'][0, 0] += 1
    target = io.BytesIO()
    np.savez(target, **arrays)
    with pytest.raises(ValueError, match='identity'):
        decode_local_plane_products(target.getvalue())


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_generated_launch_uses_accepted_bytes_not_reconstruction(monkeypatch, precision):
    req = request(precision=precision, supplied=False)
    observed = []
    original = flow._observe_arrays
    def spy(callback, originals, xp, index, start, end, center, kind):
        if kind == 'launch_boundary': observed.append(originals[0].copy())
        return original(callback, originals, xp, index, start, end, center, kind)
    monkeypatch.setattr(flow, '_observe_arrays', spy)
    builds = []
    launch = flow.build_launch
    def build(*args, **kwargs):
        builds.append(1)
        return launch(*args, **kwargs)
    monkeypatch.setattr(flow, 'build_launch', build)
    out = products.run_local_plane_products(req, selection=all_products())
    assert out.scientific.status == 'completed', out.scientific.failure
    assert len(builds) == len(observed) == 1
    A = observed[0]
    ref = float(np.sum(np.max(np.abs(A)**2, axis=(-2, -1))))
    assert out.products.presentation_reference == ref
    ix, iy = np.argmin(abs(out.products.x_um)), np.argmin(abs(out.products.y_um))
    assert out.products.cuts['optical'][0][0].tobytes() == A[:, :, iy].tobytes()
    assert out.products.cuts['optical'][1][0].tobytes() == A[:, ix, :].tobytes()


def test_fractional_terminal_metadata_and_center_coordinates():
    req = request(n=3, h=.1)
    req = replace(req, grid=replace(req.grid, z_length_um=.3))
    out = products.run_local_plane_products(req, selection=all_products())
    assert out.scientific.status == 'completed', out.scientific.failure
    assert out.products.boundary_z_um.tolist() == [0., .1, .2, .3]
    assert out.products.center_z_um.tolist() == [.05, .1+.05, .2+.05]
    restored = decode_local_plane_products(encode_local_plane_products(out))
    assert restored.products.boundary_z_um[-1] == .3


def test_failed_launch_all_selections_codec(monkeypatch):
    def fail(*args, **kw): raise ValueError('before launch')
    monkeypatch.setattr(flow, 'build_launch', fail)
    out = products.run_local_plane_products(request(), selection=all_products())
    restored = decode_local_plane_products(encode_local_plane_products(out))
    assert restored.products.cuts['optical'][0].shape == (0, 1, 16)


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_native_products_when_available(precision):
    try:
        get_backend(BackendSpec('cupy', precision, False))
    except RuntimeError:
        pytest.skip('CuPy/CUDA unavailable: local-plane products and codec')
    from lcprop.core.backend import asnumpy
    req = replace(request(precision=precision), backend=BackendSpec('cupy', precision, False))
    baseline = flow.run_local_intensity_planes(req, retain_boundary_field=True)
    out = products.run_local_plane_products(req, selection=all_products())
    assert out.scientific.status == 'completed', out.scientific.failure
    assert asnumpy(out.scientific.boundary_field).tobytes() == asnumpy(baseline.boundary_field).tobytes()
    restored = decode_local_plane_products(encode_local_plane_products(out))
    assert restored.scientific.boundary_field.tobytes() == asnumpy(baseline.boundary_field).tobytes()
    assert restored.products.preview.tobytes() == out.products.preview.tobytes()


def test_repeated_fractional_cells_preserve_observed_coordinates():
    # index*h and previous_start+h can differ by an ULP at interior boundaries.
    req = request(n=10, h=.1)
    out = products.run_local_plane_products(req)
    assert out.scientific.status == 'completed', out.scientific.failure
    assert out.products.boundary_z_um.tolist() == [0.] + [r['z_end_um'] for r in out.scientific.ledger]
    assert out.products.boundary_z_um[-1] == 1.
    assert out.scientific.ledger[6]['z_start_um'] != out.products.boundary_z_um[6]
    restored = decode_local_plane_products(encode_local_plane_products(out))
    assert restored.products.boundary_z_um.tobytes() == out.products.boundary_z_um.tobytes()
