"""Hash-bound S3/v3 material-only overlap; no workflow or research imports."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from lcprop.pr.unified.specs import (
    PRMaterialPrecisionSpec, PRUnifiedSpatialSpec, PRElectricalClosureSpec,
    POSITIVE_PRECISION, MIXED_PRECISION, UNBIASED, FIXED_FIELD,
    PRESCRIBED_CURRENT, OPEN_TRANSVERSE,
)
from lcprop.pr.unified.state import PRTransportIntensity
from lcprop.pr.unified.solver_specs import PRUnifiedSolverSpec, SCALABLE, ITERATIVE_POLICY
from lcprop.pr.unified._backend import MaterialBackend
from lcprop.pr.unified import _scalable as core, _newton, operators
from lcprop.pr.unified._positive import Geometry, ARITHMETIC, carrier
from lcprop.pr.unified.static import solve_static_material
from tests._pr_unified_oracle import evaluate_plane
from tests._pr_unified_v3_oracle import evaluate as wide_oracle

DATA = Path(__file__).parent/'data/pr_unified_scalable_v3'
MANIFEST = json.loads((DATA/'manifest.json').read_text())
SOLVER = PRUnifiedSolverSpec(SCALABLE, ITERATIVE_POLICY)


def frozen(name):
    path = DATA/'frozen.npz.fixture'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == MANIFEST['files']['frozen.npz.fixture']
    with np.load(path) as z:
        result = {k.split('__', 1)[1]: z[k] for k in z.files if k.startswith(name+'__')}
    for key, array in result.items():
        assert hashlib.sha256(array.tobytes()).hexdigest() == MANIFEST['array_hashes'][name+'__'+key]
    return result


def closure(kind):
    identity, target = {'zero': (UNBIASED, (0., 0.)),
        'fixed': (FIXED_FIELD, (.15, -.08)), 'current': (PRESCRIBED_CURRENT, (.08, -.03)),
        'open': (OPEN_TRANSVERSE, (.15, 0.))}[kind]
    return PRElectricalClosureSpec(identity, 2, target)


def electrical(B, kind):
    U, V = {'zero': (np.eye(2), np.zeros((2, 2))),
        'fixed': (np.eye(2), np.zeros((2, 2))),
        'current': (np.zeros((2, 2)), np.eye(2)),
        'open': (np.diag([1., 0.]), np.diag([0., 1.]))}[kind]
    return tuple(B.array(a) for a in (U, V, closure(kind).target))


def request(I, lengths, backend='numpy'):
    precision = PRMaterialPrecisionSpec() if I.dtype.name == 'float64' else PRMaterialPrecisionSpec(
        POSITIVE_PRECISION, 'float32', output_dtype='float32')
    return PRTransportIntensity(I, PRUnifiedSpatialSpec(I.shape, lengths, ('x', 'y')),
        precision, backend, 1., .01, 0.)


def metrics(a, b):
    return float(np.max(abs(a-b)))


@pytest.mark.parametrize('case', MANIFEST['roots'], ids=lambda c: c['id'])
def test_frozen_and_independent_direct_overlap(case, record_property):
    f = frozen(case['id']); I = f['I']; dt = case['precision']; ck = case['closure']
    c = closure(ck); req = request(I, tuple(case['lengths'])); before = I.tobytes()
    state, trace = core.solve_material(req, closure=c, solver=SOLVER)
    assert I.tobytes() == before
    assert trace[-1]['passed']
    tol = dict(rtol=2e-4, atol=2e-5) if dt == 'float32' else dict(rtol=1e-8, atol=1e-8)
    for k in ('q', 'psi', 'b'):
        np.testing.assert_allclose(getattr(state, k), f[k], **tol)
        # Same-backend promoted algorithm must preserve retained state64 bits.
        if dt == 'float64': np.testing.assert_array_equal(getattr(state, k), f[k])
        assert getattr(state, k).dtype == np.dtype(dt)
        record_property('frozen_'+k+'_max', metrics(getattr(state,k), f[k]))
    direct_precision = PRMaterialPrecisionSpec() if dt == 'float64' else PRMaterialPrecisionSpec(
        MIXED_PRECISION, 'float32', output_dtype='float32')
    direct, diagnostics = solve_static_material(replace(req, precision=direct_precision), closure=c)
    fixed = (True, True) if ck in ('zero', 'fixed') else ((True, False) if ck == 'open' else (False, False))
    for s in (state, direct):
        evaluate_plane(I, s.q, s.psi, s.b, case['lengths'], fixed, c.target, ck == 'zero', dt)
    g = Geometry(I.shape, tuple(case['lengths']), MaterialBackend('numpy', dt))
    oldg = operators.Geometry(g.shape, g.lengths, g.backend)
    a = dict(q=state.q, psi=state.psi, b=state.b, carrier=carrier(np,state.q),
        current=np.stack(ARITHMETIC.flux(g,I,state.q,state.psi,state.b)),
        E=np.stack(operators.field(g,state.psi,state.b)))
    b = dict(q=direct.q, psi=direct.psi, b=direct.b, carrier=np.exp(direct.q),
        current=np.stack(operators.flux(oldg,I,direct.q,direct.psi,direct.b)),
        E=np.stack(operators.field(oldg,direct.psi,direct.b)))
    a['optical_E'] = np.stack([(e+np.roll(e,1,axis=j))*e.dtype.type(.5) for j,e in enumerate(a['E'])])
    b['optical_E'] = np.stack([(e+np.roll(e,1,axis=j))*e.dtype.type(.5) for j,e in enumerate(b['E'])])
    for key in a:
        t = dict(tol)
        if key == 'current' and dt == 'float32':
            t['atol'] = max(t['atol'], 16*np.finfo(np.float32).eps*I.max()*b['carrier'].max()/min(g.spacing))
        np.testing.assert_allclose(a[key],b[key],**t)
        record_property('direct_'+key+'_max',metrics(a[key],b[key]))
    record_property('newton_updates',len(trace)-1)
    record_property('max_inner',max((len(t.get('linear',[])) for t in trace),default=0))
    record_property('carrier_min',float(a['carrier'].min()))
    record_property('physical_values',trace[-1]['values'])
    assert all(dict(diagnostics.observations)[key] <= limit for key,limit in diagnostics.limits)


@pytest.mark.parametrize('case', MANIFEST['primitives'], ids=lambda c:c['id'])
def test_matrix_free_primitive_oracles(case):
    r = frozen(case['id']); B=MaterialBackend('numpy',case['precision'])
    g=Geometry(tuple(case['shape']),tuple(case['lengths']),B);e=electrical(B,case['closure'])
    I,q,p,b,v=[r[k] for k in ('I','q','psi','b','v')]
    n=carrier(np,q);L=core.Laplacian(g)
    got=dict(Jv=core.WideJacobian(g,I,q,p,b,e)(v),
        A0=L(v[:g.size].reshape(g.shape),(B.dtype(L.center)+n).astype(float)).ravel(),
        Schur=core.preconditioner(g,I,q,e,False)(v),FFT=core.preconditioner(g,I,q,e,True)(v[:g.size]))
    tol=dict(rtol=2e-6,atol=2e-7) if case['precision']=='float32' else dict(rtol=3e-13,atol=3e-14)
    for key,value in got.items():np.testing.assert_allclose(value,r[key],**tol)
    if case['precision']=='float32':
        independent=wide_oracle(I,q,p,b,g.lengths,*e,v)
        for key,value in got.items():np.testing.assert_allclose(value,independent[key],**tol)
        faces=ARITHMETIC.flux(g,I,q,p,b)
        for key,value in dict(carrier=n,flux=np.stack(faces),divergence=g.divergence(faces),
                Gauss=g.poisson(p)-n+1).items():
            assert value.dtype==np.float64
            np.testing.assert_allclose(value,independent[key],**tol)
    else:
        _,matrix=_newton.system(g,I,q,p,b,e)
        np.testing.assert_allclose(got['Jv'],matrix@v,**tol)


def test_no_direct_assembly_and_no_scientific_input_mutation(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('direct spatial assembly or factorization')
    monkeypatch.setattr(MaterialBackend,'direct',forbidden)
    monkeypatch.setattr(operators.Geometry,'sparse_poisson',forbidden)
    monkeypatch.setattr(operators.Geometry,'forward_pair_matrix',forbidden)
    monkeypatch.setattr(_newton,'system',forbidden)
    for kind in ('zero','fixed','current','open'):
        I=(1.1+.1*np.cos(np.arange(8)[:,None])+np.zeros((8,4))).astype('float32')
        before=I.tobytes();s,t=core.solve_material(request(I,(6.,3.)),closure=closure(kind),solver=SOLVER)
        assert t[-1]['passed'] and I.tobytes()==before and not np.shares_memory(s.q,I)


def test_failure_retains_last_iterate_and_trace(monkeypatch):
    def fail(*args,**kwargs):raise core.Failure('injected Krylov failure')
    monkeypatch.setattr(core,'pcg',fail)
    I=(1.1+.1*np.cos(np.arange(8)[:,None])+np.zeros((8,4))).astype('float32')
    before=I.tobytes()
    with pytest.raises(core.Failure,match='injected') as exc:
        core.solve_material(request(I,(6.,3.)),closure=closure('zero'),solver=SOLVER)
    assert exc.value.trace and exc.value.state['psi'].shape==I.shape
    assert I.tobytes()==before and np.all(carrier(np,exc.value.state['q'])>0)


@pytest.mark.parametrize('dt',['float32','float64'])
def test_native_residency_when_available(dt,monkeypatch):
    cp=pytest.importorskip('cupy',reason='CuPy/CUDA unavailable')
    try:cp.zeros(1)
    except Exception:pytest.skip('CuPy/CUDA unavailable')
    I=cp.asarray((1.1+.1*np.cos(np.arange(8)[:,None])+np.zeros((8,4))).astype(dt))
    original=cp.asnumpy;transfers=[]
    def guard(a,*args,**kwargs):
        assert a.ndim<=1 and a.size<=32
        transfers.append(a.nbytes)
        return original(a,*args,**kwargs)
    monkeypatch.setattr(cp,'asnumpy',guard)
    s,t=core.solve_material(request(I,(6.,3.),'cupy'),closure=closure('current'),solver=SOLVER)
    assert t[-1]['passed'] and isinstance(s.q,cp.ndarray)
    assert max(transfers)<=256

    # Explicit completed-output exports for test instrumentation, outside the
    # guarded application call; never part of material execution.
    cpu, _ = core.solve_material(request(original(I), (6., 3.)), closure=closure('current'), solver=SOLVER)
    tol = dict(rtol=2e-4, atol=2e-5) if dt == 'float32' else dict(rtol=1e-8, atol=1e-8)
    for key in ('q', 'psi', 'b'):
        np.testing.assert_allclose(original(getattr(s, key)), getattr(cpu, key), **tol)


@pytest.fixture
def failure_fixture(monkeypatch):
    """Small nonuniform plane; no direct spatial fallback is permitted."""
    I=(1.1+.1*np.cos(np.arange(8)[:,None])+np.zeros((8,4))).astype('float32')
    req=request(I,(6.,3.));before=I.tobytes()
    def forbid(*args,**kwargs):raise AssertionError('direct fallback')
    monkeypatch.setattr(MaterialBackend,'direct',forbid)
    yield req
    assert I.tobytes()==before


def assert_failure_metadata(exc, req, kind):
    assert exc.solver is SOLVER
    assert exc.closure==closure(kind)
    assert exc.precision is req.precision
    assert exc.backend=='numpy'
    if exc.state is not None:
        assert not np.shares_memory(exc.state['I'],req.values)
        assert exc.state['q'].dtype==np.float32
        assert np.all(carrier(np,exc.state['q'])>0)


def test_domain_initialization_preserves_cause_without_fabricated_iterate(monkeypatch, failure_fixture):
    from lcprop.pr.unified._positive import CarrierDomainError
    I=np.full((8,4),1e10,np.float32)
    I[0,0]=np.nextafter(np.float32(0),np.float32(1))
    req=replace(failure_fixture,values=I,dark_intensity=0.)
    before=I.tobytes()
    def forbidden(*args,**kwargs):raise AssertionError('work after initialization domain failure')
    monkeypatch.setattr(core,'pcg',forbidden);monkeypatch.setattr(core,'gmres',forbidden)
    with pytest.raises(core.Failure) as caught:
        core.solve_material(req,closure=closure('zero'),solver=SOLVER)
    e=caught.value
    assert isinstance(e.__cause__,CarrierDomainError)
    assert str(e)==str(e.__cause__)
    assert e.stage=='carrier_domain' and e.operation_stage=='diagnosis'
    assert e.state is None and e.state_iteration is None
    assert e.trace==[] and e.last_diagnostics is None and e.inner_iterations==0
    assert I.tobytes()==before
    assert_failure_metadata(e,req,'zero')


@pytest.mark.parametrize('kind,linear,operation',[
    ('zero','pcg','dot'), ('current','gmres','lstsq')])
def test_backend_linalg_failure_through_real_krylov(monkeypatch,failure_fixture,kind,linear,operation):
    original_error=np.linalg.LinAlgError('injected backend linear failure')
    calls=[];old_linear=getattr(core,linear);captured={};old_diagnose=core.diagnose
    def capture(g,I,q,p,b,*args,**kwargs):
        result=old_diagnose(g,I,q,p,b,*args,**kwargs)
        captured.update(q=q.copy(),psi=p.copy(),b=b.copy())
        return result
    monkeypatch.setattr(core,'diagnose',capture)
    def fail(*args,**kwargs):
        calls.append(operation)
        assert len(calls)==1
        raise original_error
    def injected(*args,**kwargs):
        # Enter the actual PCG/GMRES loop. Fail after real history exists.
        original=np.dot if operation=='dot' else np.linalg.lstsq
        count=0
        def after_progress(*a,**k):
            nonlocal count
            count+=1
            threshold=4 if operation=='dot' else 2
            if count==threshold:return fail(*a,**k)
            return original(*a,**k)
        with monkeypatch.context() as patch:
            if operation=='dot':patch.setattr(np,'dot',after_progress)
            else:patch.setattr(np.linalg,'lstsq',after_progress)
            return old_linear(*args,**kwargs)
    monkeypatch.setattr(core,linear,injected)
    with pytest.raises(core.Failure) as caught:
        core.solve_material(failure_fixture,closure=closure(kind),solver=SOLVER)
    e=caught.value
    assert e.__cause__ is original_error and str(e)==str(original_error)
    assert e.stage==linear and e.linear_solver==linear
    assert e.iteration==len(e.trace)-1 and e.state_iteration==e.iteration
    assert e.inner_iterations>=1 and e.inner_iterations==len(e.trace[-1]['linear'])
    assert 'true_relative' in e.trace[-1]['linear'][-1]
    if linear=='gmres':assert 'restart' in e.trace[-1]['linear'][-1]
    assert e.last_diagnostics is not None
    assert calls==[operation]
    for key in captured:np.testing.assert_array_equal(e.state[key],captured[key])
    assert_failure_metadata(e,failure_fixture,kind)


@pytest.mark.parametrize('when', ['setup','application'])
def test_preconditioner_linalg_stage(monkeypatch,failure_fixture,when):
    error=np.linalg.LinAlgError('preconditioner failure');old=core.preconditioner;calls=[]
    def fail(*args,**kwargs):
        calls.append(when);raise error
    def setup(*args,**kwargs):
        if when=='setup':return fail()
        old(*args,**kwargs)
        return fail
    monkeypatch.setattr(core,'preconditioner',setup)
    with pytest.raises(core.Failure) as caught:
        core.solve_material(failure_fixture,closure=closure('current'),solver=SOLVER)
    e=caught.value
    assert e.__cause__ is error and e.stage=='preconditioner_'+when
    assert e.state is not None and e.trace and calls==[when]
    assert_failure_metadata(e,failure_fixture,'current')


def test_trial_domain_failure_keeps_previous_iterate(monkeypatch,failure_fixture):
    from lcprop.pr.unified._positive import CarrierDomainError
    old=core.residual;captured={};error=CarrierDomainError('injected trial domain error')
    calls=0
    def residual(g,I,q,p,b,electrical):
        nonlocal calls
        calls+=1
        if calls==1:
            captured.update(q=q.copy(),psi=p.copy(),b=b.copy())
            return old(g,I,q,p,b,electrical)
        assert calls==2
        assert not np.array_equal(p,captured['psi'])
        raise error
    monkeypatch.setattr(core,'residual',residual)
    with pytest.raises(core.Failure) as caught:
        core.solve_material(failure_fixture,closure=closure('current'),solver=SOLVER)
    e=caught.value
    assert e.__cause__ is error and e.stage=='carrier_domain' and e.operation_stage=='globalization'
    assert e.state_iteration==0 and calls==2
    for key in captured:np.testing.assert_array_equal(e.state[key],captured[key])
    assert e.trace[-1]['linear'] and len(e.trace[-1]['trials'])==1
    assert 'alpha' not in e.trace[-1]  # No trial was accepted.
    assert_failure_metadata(e,failure_fixture,'current')


def test_later_diagnosis_failure_keeps_newest_accepted_solver_iterate(monkeypatch,failure_fixture):
    from lcprop.pr.unified._positive import CarrierDomainError
    old=core.diagnose;error=CarrierDomainError('injected later diagnosis error');calls=0;captured={}
    def diagnosis(g,I,q,p,b,*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==1:return old(g,I,q,p,b,*args,**kwargs)
        assert calls==2
        captured.update(q=q.copy(),psi=p.copy(),b=b.copy())
        raise error
    monkeypatch.setattr(core,'diagnose',diagnosis)
    with pytest.raises(core.Failure) as caught:
        core.solve_material(failure_fixture,closure=closure('current'),solver=SOLVER)
    e=caught.value
    assert e.__cause__ is error and e.stage=='carrier_domain' and e.operation_stage=='diagnosis'
    assert e.state_iteration==1 and e.iteration==1 and e.diagnostics_iteration==0
    assert 'alpha' in e.trace[0] and len(e.trace)==1
    for key in captured:np.testing.assert_array_equal(e.state[key],captured[key])
    assert_failure_metadata(e,failure_fixture,'current')


@pytest.mark.parametrize('error',[AssertionError('bug'),TypeError('bug'),AttributeError('bug'),IndexError('bug'),RuntimeError('unexpected API defect')])
def test_programming_exceptions_are_not_numerical_failures(monkeypatch,failure_fixture,error):
    def fail(*args,**kwargs):raise error
    monkeypatch.setattr(core,'preconditioner',fail)
    with pytest.raises(type(error)) as caught:
        core.solve_material(failure_fixture,closure=closure('current'),solver=SOLVER)
    assert caught.value is error and not isinstance(caught.value,core.Failure)


@pytest.mark.parametrize('seam',['electrical_metadata','harmonic_initialization'])
def test_initialization_linalg_failure_without_iterate(monkeypatch,failure_fixture,seam):
    error=np.linalg.LinAlgError('initialization failure');calls=[]
    def fail(*args,**kwargs):
        calls.append(seam);raise error
    monkeypatch.setattr(MaterialBackend,'array' if seam=='electrical_metadata' else 'direct_dense',fail)
    with pytest.raises(core.Failure) as caught:
        core.solve_material(failure_fixture,closure=closure('current'),solver=SOLVER)
    e=caught.value
    assert e.__cause__ is error and e.stage=='initialization'
    assert e.state is None and e.trace==[] and e.last_diagnostics is None
    assert e.iteration is None and calls==[seam]
    assert_failure_metadata(e,failure_fixture,'current')


@pytest.mark.parametrize('error',[FloatingPointError('arithmetic fault'),OverflowError('arithmetic overflow')])
def test_coefficient_numerical_failure_keeps_state(monkeypatch,failure_fixture,error):
    def fail(*args,**kwargs):raise error
    monkeypatch.setattr(core,'WideJacobian',fail)
    with pytest.raises(core.Failure) as caught:
        core.solve_material(failure_fixture,closure=closure('current'),solver=SOLVER)
    e=caught.value
    assert e.__cause__ is error and e.stage=='coefficient_construction'
    assert e.state is not None and e.trace and e.inner_iterations==0
    assert_failure_metadata(e,failure_fixture,'current')
