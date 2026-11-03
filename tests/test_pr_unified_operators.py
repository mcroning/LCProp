"""Independent operator/Jacobian identities and bounded backend qualification."""
import ast
from pathlib import Path
import numpy as np
import pytest
from lcprop.pr.unified._backend import MaterialBackend
from lcprop.pr.unified._newton import system, residual
from lcprop.pr.unified.operators import Geometry, bernoulli, flux
from lcprop.pr.unified.static import solve_static_material
from tests.test_pr_unified_static import transport, zero


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_incidence_conservation_nullspace_and_flux(precision):
    B = MaterialBackend('numpy', precision)
    g = Geometry((16,), (4.,), B)
    rng = np.random.default_rng(71)
    p = rng.normal(size=16).astype(precision)
    f = rng.normal(size=16).astype(precision)
    h = .25
    expected_grad = np.array([(p[(i+1)%16]-p[i])/h for i in range(16)], dtype=precision)
    np.testing.assert_array_equal(g.gradient(p)[0], expected_grad)
    np.testing.assert_allclose(np.sum(g.gradient(p)[0]*f), -np.sum(p*g.divergence((f,))),
                               rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(g.sparse_poisson()@p, g.poisson(p), rtol=2e-6, atol=2e-6)
    eigenvalues = np.linalg.eigvalsh(g.sparse_poisson().toarray().astype(float))
    assert np.count_nonzero(abs(eigenvalues) < 1e-10) == 1
    nyquist = np.array([(-1.)**i for i in range(16)], dtype=precision)
    np.testing.assert_array_equal(g.poisson(nyquist), 4/h**2*nyquist)
    z = np.zeros(16, dtype=precision)
    I = (1.2+.2*np.cos(np.arange(16))).astype(precision)
    np.testing.assert_array_equal(flux(g, I, z, z, np.zeros(1, dtype=precision))[0],
                                  -(np.roll(I, -1)-I)/h)
    J = flux(g, np.ones(16, dtype=precision), z, z, np.array([.3], dtype=precision))[0]
    np.testing.assert_allclose(J, .3, rtol=2e-6, atol=2e-7)
    with pytest.raises(ValueError, match='one or two active'):
        Geometry((8, 2, 2), (4., 2., 2.), B)


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_bernoulli_independent_reference(precision):
    v = np.array([-1000., -51., -1., -.001, -1e-8, 0., 1e-8, .001, 1., 51., 1000.], dtype=precision)
    value, derivative = bernoulli(np, v)
    x = v.astype(np.longdouble)
    expected = np.ones_like(x)
    nonzero = x != 0
    # Some platforms have binary64 long double: overflow at +1000 yields
    # the correct representable zero limit, not a changed acceptance tolerance.
    with np.errstate(over='ignore'):
        expected[nonzero] = x[nonzero]/np.expm1(x[nonzero])
    tol = dict(rtol=2e-6, atol=2e-7) if precision == 'float32' else dict(rtol=3e-13, atol=3e-14)
    np.testing.assert_allclose(value, expected, **tol)
    assert derivative[5] == -.5 and value[5] == 1
    assert value.dtype == derivative.dtype == v.dtype
    assert np.isfinite(derivative).all() and (derivative <= 0).all()


@pytest.mark.parametrize('fixed', [False, True])
def test_analytic_jacobian_against_independent_direction(fixed):
    B = MaterialBackend('numpy', 'float64')
    g = Geometry((12,), (5.,), B)
    rng = np.random.default_rng(13)
    I = 1+.2*rng.random(12)
    q, p = .01*rng.normal(size=(2, 12))
    b = np.array([.3])
    closure = (np.array([[float(fixed)]]), np.array([[float(not fixed)]]), np.array([.1]))
    r, matrix = system(g, I, q, p, b, closure)
    v = rng.normal(size=r.size)
    def shifted(t):
        return residual(g, I, q+t*v[:12], p+t*v[12:24], b+t*v[24:], closure)
    np.testing.assert_allclose(matrix@v, (shifted(2e-6)-shifted(-2e-6))/4e-6, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_sensitive_linear_solve_is_double(monkeypatch, precision):
    B = MaterialBackend('numpy', precision)
    seen = []
    original = B.linalg.spsolve
    def wrapped(a, b):
        seen.append((a.dtype, b.dtype))
        return original(a, b)
    monkeypatch.setattr(B.linalg, 'spsolve', wrapped)
    x = B.direct(B.sparse.eye(4, dtype=precision), np.ones(4, dtype=precision))
    assert seen == [(np.dtype('float64'), np.dtype('float64'))]
    assert x.dtype == np.dtype(precision)
    with pytest.raises(ValueError):
        B.status([np.ones((4, 4))])
    with pytest.raises(ValueError):
        B.status([np.float64(1)]*33)


def test_no_workflow_imports_or_longitudinal_allocation(monkeypatch):
    root = Path(__file__).resolve().parents[1]/'src/lcprop'
    for path in root.rglob('*.py'):
        # M6 adds only explicit adapter/registration imports, never legacy science.
        adapters={'transport/defaults.py','pr/experiment_codec.py','pr/runtime_estimator.py',
                  'pr/gui/evolution_panel.py','pr/gui/unified_controls.py','pr/gui/request_adapter.py',
                  'pr/gui/main_window.py','pr/gui/run_cost.py'}
        if 'unified' in path.parts or path.relative_to(root).as_posix() in adapters:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert 'unified' not in (node.module or '').split('.'), path
            elif isinstance(node, ast.Import):
                assert not any('unified' in alias.name.split('.') for alias in node.names), path
    I = np.ones((12, 3))
    seen = []
    original = np.empty
    def bounded(shape, *args, **kwargs):
        shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
        seen.append(shape_tuple)
        assert len(shape_tuple) <= 2
        return original(shape, *args, **kwargs)
    monkeypatch.setattr(np, 'empty', bounded)
    state, _ = solve_static_material(transport(I), closure=zero())
    assert state.q.shape == state.psi.shape == (12, 3) and state.b.shape == (3, 1)
    assert seen


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_cupy_native_no_plane_transfer(precision, monkeypatch):
    cp = pytest.importorskip('cupy', reason='CuPy/CUDA unavailable locally')
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            pytest.skip('CuPy/CUDA unavailable locally')
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip('CuPy/CUDA unavailable locally')
    I = (1+.4*np.cos(np.arange(64)*2*np.pi/64)).astype(precision)
    cpu, _ = solve_static_material(transport(I, precision=precision), closure=zero())
    device = cp.asarray(I)
    before = device.copy()
    original = cp.asnumpy
    transfers = []
    def guarded(a, *args, **kwargs):
        assert a.ndim == 1 and a.size <= 32 and a.dtype == cp.float64 and a.nbytes <= 256
        transfers.append(a.nbytes)
        return original(a, *args, **kwargs)
    # Also intercept direct .get()/item calls so bypassing asnumpy is detected.
    import sys
    previous = sys.getprofile()
    def profile(frame, event, arg):
        if event == 'c_call' and getattr(arg, '__name__', '') in ('get', 'item', 'tolist'):
            obj = getattr(arg, '__self__', None)
            if isinstance(obj, cp.ndarray):
                assert obj.ndim <= 1 and obj.size <= 32 and obj.nbytes <= 256
        if previous is not None:
            previous(frame, event, arg)
    with monkeypatch.context() as m:
        m.setattr(cp, 'asnumpy', guarded)
        sys.setprofile(profile)
        try:
            gpu, _ = solve_static_material(transport(device, backend='cupy', precision=precision), closure=zero())
        finally:
            sys.setprofile(previous)
    assert transfers and max(transfers) <= 256
    np.testing.assert_array_equal(cp.asnumpy(device), cp.asnumpy(before))
    for name in ('q', 'psi', 'b'):
        np.testing.assert_allclose(cp.asnumpy(getattr(gpu, name)), getattr(cpu, name),
            rtol=2e-4 if precision == 'float32' else 1e-8,
            atol=2e-5 if precision == 'float32' else 1e-8)
