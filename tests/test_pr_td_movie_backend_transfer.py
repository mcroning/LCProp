"""Movie reduction preserves presentation bytes without a full-plane transfer."""
from dataclasses import replace
import numpy as np
import pytest
from lcprop.pr import visualization as v, workflow as w
from tests.test_pr_td_published import case
from lcprop.pr.scattering import PRCanonicalScatteringSpec, PR_CANONICAL_SCATTERING_V2


class BackendStandIn:
    """Exercise backend arithmetic locally without pretending to certify CUDA."""
    def __getattr__(self, name):
        return getattr(np, name)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('shape', [(16, 6), (257, 259), (1153, 133), (16641, 7)])
def test_backend_movie_matches_original_reduceat_exactly(dtype, shape):
    source = np.random.default_rng(91).lognormal(0, 8, shape).astype(dtype)
    before = source.copy()
    expected = v.block_average_2d(source, max_x=128, max_y=128)[0].astype(np.float32)
    transfers = []
    def transfer(a):
        assert a.dtype == np.float32
        assert a.shape == (min(shape[0], 128), min(shape[1], 128))
        assert a.nbytes <= 65536
        transfers.append((a.shape, a.nbytes))
        return a.copy()
    actual = v.downsample_td_movie_frame(source, xp=BackendStandIn(), asnumpy=transfer)
    np.testing.assert_array_equal(actual, expected)
    assert actual.tobytes() == expected.tobytes()
    np.testing.assert_array_equal(source, before)
    assert len(transfers) == 1
    assert v.downsample_td_movie_frame(source).tobytes() == expected.tobytes()
    if shape[0] > 128:
        with pytest.raises(AssertionError):
            transfer(source)  # The pre-fix full-plane route is rejected.


@pytest.mark.parametrize('n', [1, 2, 8, 9, 17, 64, 129, 130, 257])
def test_reduceat_rounding_and_signed_zero(n):
    source = np.resize(np.array([1e16, 1., -1e16, 3., -0., 1e-20]), (n, 3))
    edges = np.array([0, n])
    actual = v._movie_reduceat(source, edges, axis=0, xp=np)
    expected = np.add.reduceat(source, edges[:-1], axis=0)
    assert actual.tobytes() == expected.tobytes()


def cupy():
    cp = pytest.importorskip('cupy')
    try:
        cp.zeros(1)
    except Exception as exc:
        pytest.skip(f'CuPy/CUDA unavailable: {exc}')
    return cp


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cupy_preview_transfer_is_bounded_exact_and_nonmutating(dtype):
    cp = cupy()
    host = np.random.default_rng(18).lognormal(0, 8, (1153, 259)).astype(dtype)
    source = cp.asarray(host)
    original = cp.asnumpy
    transfers = []
    def transfer(a):
        assert a.shape == (128, 128) and a.dtype == cp.float32
        assert a.nbytes == 65536
        transfers.append(a.nbytes)
        return original(a)
    with pytest.raises(AssertionError):
        transfer(source)
    actual = v.downsample_td_movie_frame(source, xp=cp, asnumpy=transfer)
    expected = v.downsample_td_movie_frame(host)
    assert actual.tobytes() == expected.tobytes()
    assert original(source).tobytes() == host.tobytes()
    assert transfers == [65536]


@pytest.mark.parametrize('scatter', [False, True])
@pytest.mark.parametrize('backend', ['numpy', 'cupy'])
@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_callback_preserves_stages_and_movie_reference(monkeypatch, backend, precision, scatter):
    xp = np if backend == 'numpy' else cupy()
    host = np.asarray if backend == 'numpy' else xp.asnumpy
    req, _, A, E, _ = case()
    req = replace(req, backend=replace(req.backend, backend=backend, precision=precision),
                  initial_A=A.astype('complex64' if precision == 'float32' else 'complex128'),
                  initial_E=E.astype(precision), solver=replace(req.solver, Nt=2),
                  scattering=(PRCanonicalScatteringSpec(epsilon=.02, transverse_correlation_um=1.,
                      realization_seed=17, canonical_dz_um=1.,
                      algorithm_version=PR_CANONICAL_SCATTERING_V2) if scatter else None))
    original_temporal = w.semi_implicit_trapezoidal_step
    original_movie = w.downsample_td_movie_frame
    runs = []
    frames = []
    phases = [[], []]
    original_phase = w._canonical_scattering_phase_for_slice
    def phase(*args, **kwargs):
        result = original_phase(*args, **kwargs)
        phases[len(runs) - 1].append((kwargs["z_index"], host(result).tobytes()))
        return result
    monkeypatch.setattr(w, "_canonical_scattering_phase_for_slice", phase)
    def temporal(e, source, **kw):
        record = []
        def observed(state):
            before = host(state).copy()
            intensity = source(state)
            assert host(state).tobytes() == before.tobytes()
            record.append((before.tobytes(), host(intensity).tobytes()))
            return intensity
        result = original_temporal(e, observed, **kw)
        assert len(record) == 2
        runs[-1].append((record, host(result).tobytes()))
        return result
    def movie(intensity, **kw):
        before = host(intensity).copy()  # Test-only oracle transfer.
        transfers = []
        def bounded(a):
            assert a.dtype == xp.float32 and max(a.shape) <= 128
            assert a.nbytes <= 65536
            transfers.append(a.nbytes)
            return host(a)
        actual = original_movie(intensity, xp=kw['xp'], asnumpy=bounded)
        assert actual.tobytes() == original_movie(before).tobytes()
        assert host(intensity).tobytes() == before.tobytes()
        if backend == 'cupy':
            assert transfers == [16 * 6 * 4]
        frames.append(actual)
        return actual
    monkeypatch.setattr(w, 'semi_implicit_trapezoidal_step', temporal)
    monkeypatch.setattr(w, 'downsample_td_movie_frame', movie)
    runs.append([])
    plain = w.run_pr_timedependent(req)
    runs.append([])
    progress = []
    observed = w.run_pr_timedependent(req, progress_callback=progress.append)
    assert runs[0] == runs[1]
    assert frames and progress
    if scatter:
        assert phases[0] and phases[1]
        expected = dict(phases[0])
        assert all(expected[i] == value for run in phases for i, value in run)
    for name in ('E_final', 'A_final', 'source_intensity_stack'):
        assert getattr(plain, name).tobytes() == getattr(observed, name).tobytes()
    assert plain.completed_steps == observed.completed_steps == 2
