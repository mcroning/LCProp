"""M2: fixed-plane solves only, independent physical oracle and frozen evidence."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from lcprop.pr.unified.specs import (
    PRUnifiedSpatialSpec, PRMaterialPrecisionSpec, PRElectricalClosureSpec,
    UNBIASED, FIXED_FIELD, PRESCRIBED_CURRENT, MIXED_PRECISION,
)
from lcprop.pr.unified.state import PRTransportIntensity
from lcprop.pr.unified.static import solve_static_material, MaterialConvergenceError
from lcprop.pr.unified._backend import MaterialBackend
from lcprop.pr.unified.operators import Geometry, field, flux
from tests._pr_unified_oracle import evaluate

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT/'results/Research/pr-unified-potential-carrier-stage-e1-mixed-v1/candidate'
SEAL = 'e6e0e4b1bd25f6a85fd42ebca29721ac5d3e8aceb01d9c427c1fdc8abbed0cf2'
CASES = ['uniform-unbiased', 'weak-unbiased', 'sinusoid-0.95-512',
         'sinusoid-0.99-512', 'sinusoid-0.999-512', 'uniform-A7',
         'request15-cell51', 'request15-cell52', 'request15-cell54', 'request15-cell56',
         'reduction-1d-fixed', 'reduction-1d-current']


def checked(relative):
    manifest = FROZEN/'scientific-manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == SEAL
    entry = json.loads(manifest.read_text())['files'][relative]
    path = FROZEN/relative
    data = path.read_bytes()
    assert len(data) == entry['bytes']
    assert hashlib.sha256(data).hexdigest() == entry['sha256']
    return path


def transport(I, length=6., backend='numpy', precision='float64'):
    s = PRUnifiedSpatialSpec((I.shape[0],), (length,),
        batch_axes=('y',) if I.ndim == 2 else (), batch_shape=(I.shape[1],) if I.ndim == 2 else ())
    p = PRMaterialPrecisionSpec() if precision == 'float64' else PRMaterialPrecisionSpec(
        MIXED_PRECISION, 'float32', output_dtype='float32')
    return PRTransportIntensity(I, s, p, backend, 1., .01, 0.)


def zero():
    return PRElectricalClosureSpec(UNBIASED, 1, (0.,))


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('name', CASES)
def test_frozen_research(name, precision, record_property):
    matrix = json.loads(checked('benchmark-matrix.json').read_text())
    case = next(c for c in matrix['cases'] if c['name'] == name)
    with np.load(checked(case['input'])) as data:
        I = data['I'].astype(precision)
    kind = case['closure']['kind']
    closure = zero() if kind == 'zero' else PRElectricalClosureSpec(
        FIXED_FIELD if kind == 'fixed' else PRESCRIBED_CURRENT, 1, tuple(case['closure']['target']))
    if name == 'uniform-A7':
        closure = PRElectricalClosureSpec.a7(.4, .2)
    before = I.tobytes()
    request = transport(I, case['lengths'][0], precision=precision)
    if name == 'uniform-A7':
        request = replace(request, dark_intensity=.2)
    state, diagnostics = solve_static_material(request, closure=closure)
    assert I.tobytes() == before
    assert not np.shares_memory(state.q, I) and not np.shares_memory(state.psi, I)
    reference = np.load(checked(f'references/{name}-{precision}.npz'))
    observations = dict(diagnostics.observations)
    count = I.shape[1] if I.ndim == 2 else 1
    B = MaterialBackend('numpy', precision)
    g = Geometry((I.shape[0],), tuple(case['lengths']), B)
    for y in range(count):
        q, p, b = (state.q[:, y], state.psi[:, y], state.b[y]) if I.ndim == 2 else (state.q, state.psi, state.b)
        inp = I[:, y] if I.ndim == 2 else I
        evaluate(inp, q, p, b[0], case['lengths'][0], kind in ('zero', 'fixed'),
                 closure.target[0], kind == 'zero', precision)
        values = dict(psi=p, n=np.exp(q), b=b, field=np.stack(field(g, p, b)), flux=np.stack(flux(g, inp, q, p, b)))
        for key, value in values.items():
            tolerance = dict(matrix['parity'][precision])
            if key == 'flux' and precision == 'float32':
                tolerance['atol'] = max(tolerance['atol'], 16*np.finfo(np.float32).eps*
                    float(I.max())*float(reference['n'].max())/g.spacing[0])
            np.testing.assert_allclose(value, reference[key][y], **tolerance)
        assert observations[f'column_{y}.converged'] == 1
        assert observations[f'column_{y}.iterations'] <= 60
    record_property('case', name)
    record_property('precision', precision)
    record_property('columns', count)
    record_property('maximum_iterations', max(observations[f'column_{y}.iterations'] for y in range(count)))
    record_property('minimum_carrier', min(observations[f'column_{y}.carrier_min'] for y in range(count)))
    record_property('maximum_gauss_residual', max(observations[f'column_{y}.gauss_max'] for y in range(count)))
    record_property('maximum_flux_residual', max(observations[f'column_{y}.flux_divergence_max'] for y in range(count)))
    for key in ('q', 'psi', 'b'):
        record_property(key+'_sha256', hashlib.sha256(getattr(state, key).tobytes()).hexdigest())
    reference.close()


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('closure', [zero(), PRElectricalClosureSpec(FIXED_FIELD, 1, (.2,)),
                                   PRElectricalClosureSpec(PRESCRIBED_CURRENT, 1, (.04,))])
def test_batch_independence(precision, closure):
    x = np.arange(32)*2*np.pi/32
    I = np.stack([np.ones(32), 1+.01*np.cos(x), 1+.95*np.cos(x), 1+.01*np.cos(x)], axis=1).astype(precision)
    batch, info = solve_static_material(transport(I, precision=precision), closure=closure)
    d = dict(info.observations)
    for y in range(4):
        single, diag = solve_static_material(transport(I[:, y], precision=precision), closure=closure)
        for name in ('q', 'psi'):
            np.testing.assert_array_equal(getattr(batch, name)[:, y], getattr(single, name))
        np.testing.assert_array_equal(batch.b[y], single.b)
        expected = {k.replace('column_0.', '', 1): v for k, v in diag.observations if k.startswith('column_0.')}
        actual = {k.replace(f'column_{y}.', '', 1): v for k, v in info.observations if k.startswith(f'column_{y}.')}
        assert actual == expected  # Complete damping/iteration history, not just root.
    assert len({d[f'column_{y}.iterations'] for y in range(4)}) > 1
    permutation = [2, 0, 3, 1]
    other, _ = solve_static_material(transport(I[:, permutation], precision=precision), closure=closure)
    np.testing.assert_array_equal(other.q, batch.q[:, permutation])
    np.testing.assert_array_equal(other.psi, batch.psi[:, permutation])
    np.testing.assert_array_equal(other.b, batch.b[permutation])


def test_invalid_inputs_and_no_partial_result(monkeypatch):
    from lcprop.pr.unified import static
    I = np.ones((8, 2))
    original = static.solve_column
    calls = []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('injected failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(static, 'solve_column', fail_second)
    with pytest.raises(MaterialConvergenceError, match='column 1.*injected failure'):
        solve_static_material(transport(I), closure=zero())
    np.testing.assert_array_equal(I, np.ones((8, 2)))
    assert len(calls) == 2
    monkeypatch.setattr(static, 'solve_column', original)
    for value in (0., -1., np.nan, np.inf):
        with pytest.raises(MaterialConvergenceError):
            solve_static_material(transport(np.full(8, value)), closure=zero())
    full = replace(transport(I), spatial=PRUnifiedSpatialSpec((8, 2), (6., 2.), ('x', 'y')))
    with pytest.raises(ValueError, match='one-dimensional'):
        solve_static_material(full, closure=PRElectricalClosureSpec(UNBIASED, 2, (0., 0.)))


@pytest.mark.parametrize('cell,column', [(51, 31), (52, 31), (54, 30), (56, 31)])
def test_stage_b_independent_frozen_root(cell, column):
    base = ROOT/'results/Research/pr-unified-potential-carrier-stage-b-v1'
    manifest = base/'evidence-manifest.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == '5f7158ad365e756f405fd389a22f31bb931d0ca4516f1064a4225269ef17bb97'
    filename = f'cell{cell}-y{column}-4096-state.npz'
    path = base/filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == json.loads(manifest.read_text())['files'][filename]['sha256']
    matrix = json.loads(checked('benchmark-matrix.json').read_text())
    case = next(c for c in matrix['cases'] if c['name'] == f'request15-cell{cell}')
    with np.load(checked(case['input'])) as archive:
        I = archive['I'][:, column]
    state, _ = solve_static_material(transport(I, case['lengths'][0]), closure=zero())
    with np.load(path) as reference:
        np.testing.assert_allclose(state.psi, reference['psi'], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose(np.exp(state.q), np.exp(reference['q']), rtol=1e-8, atol=1e-8)


def test_oracle_detects_physical_faults():
    I = 1+.2*np.cos(np.arange(16)*2*np.pi/16)
    state, _ = solve_static_material(transport(I), closure=zero())
    for q, p, b, target in [(state.q+.1, state.psi, 0., 0.),
                            (state.q, state.psi+.1, 0., 0.),
                            (state.q, state.psi, .1, 0.),
                            (state.q, state.psi, 0., .1)]:
        with pytest.raises(AssertionError):
            evaluate(I, q, p, b, 6., True, target, True, 'float64')


def test_a7_uniform_reservoir_is_not_internal_field():
    I = np.full(16, .8+.01+.13)
    closure = PRElectricalClosureSpec.a7(-.7, .01+.13)
    intensity = replace(transport(I), dark_intensity=.01, uniform_background=.13)
    state, diagnostics = solve_static_material(intensity, closure=closure)
    np.testing.assert_allclose(state.b, -.7*(.01+.13)/I[0], rtol=1e-12, atol=1e-12)
    assert state.b[0] != closure.reservoir_field
    np.testing.assert_allclose(np.exp(state.q), 1., rtol=1e-12, atol=1e-12)
    observations = dict(diagnostics.observations)
    for name, limit in diagnostics.limits:
        assert observations[name] <= limit


def test_a7_background_provenance_mismatch_fails():
    with pytest.raises(ValueError, match='background provenance'):
        solve_static_material(transport(np.ones(8)), closure=PRElectricalClosureSpec.a7(.4, .2))
    with pytest.raises(TypeError, match='ClosureSpec'):
        solve_static_material(transport(np.ones(8)), closure=None)
