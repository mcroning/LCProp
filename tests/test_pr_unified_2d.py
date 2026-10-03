"""M3 bounded connected planes; frozen oracles and explicit dimensional reduction."""
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest
from lcprop.pr.unified.specs import PRUnifiedSpatialSpec, PRElectricalClosureSpec, UNBIASED, FIXED_FIELD, PRESCRIBED_CURRENT, OPEN_TRANSVERSE
from lcprop.pr.unified.static import solve_static_material
from lcprop.pr.unified._backend import MaterialBackend
from lcprop.pr.unified.operators import Geometry, carrier, field, flux
from tests.test_pr_unified_static import transport, checked, ROOT, zero
from tests._pr_unified_oracle import evaluate_plane

CASES = [f'{pattern}-{closure}' for pattern in ('separable','asymmetric','localized') for closure in ('zero','fixed','current','open')]+['reduction-2d-fixed','reduction-2d-current']


def plane(I, lengths=(2*np.pi,2*np.pi), precision='float64', backend='numpy'):
    return replace(transport(I,precision=precision,backend=backend), spatial=PRUnifiedSpatialSpec(I.shape,lengths,('x','y')))


def closure_for(kind,target=None):
    if kind=='zero':return PRElectricalClosureSpec(UNBIASED,2,(0.,0.))
    if kind=='open':return PRElectricalClosureSpec(OPEN_TRANSVERSE,2,(float(target),0.))
    return PRElectricalClosureSpec(FIXED_FIELD if kind=='fixed' else PRESCRIBED_CURRENT,2,tuple(target))


def oracle(I,s,lengths,precision):
    ident=s.closure.identity
    fixed=(True,True) if ident in (UNBIASED,FIXED_FIELD) else ((True,False) if ident==OPEN_TRANSVERSE else (False,False))
    return evaluate_plane(I,s.q,s.psi,s.b,lengths,fixed,s.closure.target,ident==UNBIASED,precision)


@pytest.mark.parametrize('precision',['float32','float64'])
@pytest.mark.parametrize('name',CASES)
def test_frozen_e_connected(name,precision,record_property):
    matrix=json.loads(checked('benchmark-matrix.json').read_text())
    case=next(c for c in matrix['cases'] if c['name']==name)
    with np.load(checked(case['input'])) as data:I=data['I'].astype(precision)
    c=closure_for(case['closure']['kind'],case['closure'].get('target'))
    before=I.tobytes()
    state,diagnostics=solve_static_material(plane(I,tuple(case['lengths']),precision),closure=c)
    assert I.tobytes()==before and not np.shares_memory(I,state.q)
    oracle(I,state,case['lengths'],precision)
    g=Geometry(I.shape,tuple(case['lengths']),MaterialBackend('numpy',precision))
    arrays=dict(psi=state.psi,n=np.exp(state.q),b=state.b,field=np.stack(field(g,state.psi,state.b)),flux=np.stack(flux(g,I,state.q,state.psi,state.b)))
    with np.load(checked(f'references/{name}-{precision}.npz')) as ref:
        for key,value in arrays.items():
            tol=dict(matrix['parity'][precision])
            if key=='flux' and precision=='float32':
                tol['atol']=max(tol['atol'],16*np.finfo(np.float32).eps*float(I.max())*float(ref['n'].max())/min(g.spacing))
            np.testing.assert_allclose(value,ref[key][0],**tol)
    d=dict(diagnostics.observations)
    for key,limit in diagnostics.limits:assert d[key]<=limit
    if name=='asymmetric-open':
        assert abs(state.b[1])>1e-5
        assert abs(d['mean_current_y'])<(1e-5 if precision=='float32' else 1e-9)
    record_property('case',name);record_property('precision',precision)
    record_property('iterations',d['iterations']);record_property('harmonic_y',d['harmonic_y'])
    for key in ('q','psi','b'):record_property(key+'_sha256',hashlib.sha256(getattr(state,key).tobytes()).hexdigest())


@pytest.mark.parametrize('precision',['float32','float64'])
def test_bitwise_local_operator_reduction(precision):
    B=MaterialBackend('numpy',precision)
    g=Geometry((64,),(6.,),B);G=Geometry((64,3),(6.,4.),B)
    x=np.arange(64)*2*np.pi/64
    I=(1+.3*np.cos(x)).astype(precision);q=(.1*np.sin(x)).astype(precision);p=(.2*np.cos(x)).astype(precision)
    b=np.array([.2],dtype=precision);bb=np.array([.2,0.],dtype=precision)
    rep=lambda a:np.repeat(a[:,None],3,axis=1)
    for a,c in [(carrier(np,q),carrier(np,rep(q))), (g.neighbor(p,0)-p,G.neighbor(rep(p),0)-rep(p)),
                (g.gradient(p)[0],G.gradient(rep(p))[0]),(g.poisson(p),G.poisson(rep(p))),
                (g.poisson(p)-carrier(np,q)+1,G.poisson(rep(p))-carrier(np,rep(q))+1),
                (field(g,p,b)[0],field(G,rep(p),bb)[0]),(flux(g,I,q,p,b)[0],flux(G,rep(I),rep(q),rep(p),bb)[0]),
                (g.divergence(flux(g,I,q,p,b)),G.divergence(flux(G,rep(I),rep(q),rep(p),bb)))]:
        np.testing.assert_array_equal(rep(a),c)
    np.testing.assert_array_equal(field(G,rep(p),bb)[1],0.)
    np.testing.assert_array_equal(flux(G,rep(I),rep(q),rep(p),bb)[1],0.)


@pytest.mark.parametrize('name,precision',[(n,p) for n in ('uniform-unbiased','weak-unbiased','sinusoid-0.999-512') for p in ('float32','float64')]+[('request15-cell51','float64')])
def test_solved_reduction(name,precision,record_property):
    matrix=json.loads(checked('benchmark-matrix.json').read_text())
    case=next(c for c in matrix['cases'] if c['name']==name)
    with np.load(checked(case['input'])) as data:I=data['I'].astype(precision)
    if I.ndim==2:I=I[:,31]
    length=case['lengths'][0];II=np.repeat(I[:,None],3,axis=1)
    a,_=solve_static_material(transport(I,length,precision=precision),closure=zero())
    b,_=solve_static_material(plane(II,(length,4.),precision),closure=closure_for('zero'))
    oracle(II,b,(length,4.),precision)
    tol=matrix['parity'][precision]
    rep=lambda v:np.repeat(v[:,None],3,axis=1)
    for x,y in [(a.q,b.q),(np.exp(a.q),np.exp(b.q)),(a.psi-a.psi.mean(),b.psi-b.psi.mean())]:
        np.testing.assert_allclose(rep(x),y,**tol)
    B=MaterialBackend('numpy',precision);g=Geometry(I.shape,(length,),B);G=Geometry(II.shape,(length,4.),B)
    E,J=field(g,a.psi,a.b),flux(g,I,a.q,a.psi,a.b)
    EE,JJ=field(G,b.psi,b.b),flux(G,II,b.q,b.psi,b.b)
    np.testing.assert_allclose(rep(E[0]),EE[0],**tol)
    flux_tol=dict(tol)
    if precision=='float32':flux_tol['atol']=max(tol['atol'],16*np.finfo(np.float32).eps*I.max()*np.exp(a.q).max()/min(G.spacing))
    np.testing.assert_allclose(rep(J[0]),JJ[0],**flux_tol)
    np.testing.assert_allclose(EE[1],0.,**tol);np.testing.assert_allclose(JJ[1],0.,**flux_tol)
    np.testing.assert_allclose(b.b,[a.b[0],0.],**tol)
    gauss_difference=rep(g.poisson(a.psi)-np.exp(a.q)+1)-(G.poisson(b.psi)-np.exp(b.q)+1)
    balance_difference=rep(g.divergence(J))-G.divergence(JJ)
    if precision=='float64':
        np.testing.assert_allclose(gauss_difference,0.,rtol=1e-8,atol=1e-8)
        np.testing.assert_allclose(balance_difference,0.,rtol=1e-8,atol=1e-8)
    record_property('gauss_max_difference',float(np.max(abs(gauss_difference))))
    record_property('balance_max_difference',float(np.max(abs(balance_difference))))
    record_property('potential_max_difference',float(np.max(abs(rep(a.psi)-b.psi))))
    record_property('carrier_max_difference',float(np.max(abs(rep(np.exp(a.q))-np.exp(b.q)))))


def test_current_reduces_to_a7():
    I=1+.3*np.cos(np.arange(64)*2*np.pi/64)
    request=replace(transport(I,2*np.pi),dark_intensity=.2)
    a7=PRElectricalClosureSpec.a7(.4,.2)
    a,_=solve_static_material(request,closure=a7)
    II=np.repeat(I[:,None],3,axis=1)
    b,_=solve_static_material(plane(II),closure=closure_for('current',(a7.target[0],0.)))
    for name in ('q','psi'):
        np.testing.assert_allclose(np.repeat(getattr(a,name)[:,None],3,axis=1),getattr(b,name),rtol=1e-8,atol=1e-8)
    np.testing.assert_allclose(b.b,[a.b[0],0.],rtol=1e-8,atol=1e-8)
    oracle(II,b,(2*np.pi,2*np.pi),'float64')


def stage_file(stage,filename):
    root=ROOT/f'results/Research/pr-unified-potential-carrier-stage-{stage}-v1'
    seals={'c':'384569804ced218271a944d4e87ae3615523b506db2cf9052219dcd7673768f1',
           'd':'4a360d25e30d8f45ef15f90fdb1094dcb9c8fa209302598e349f64b668028774'}
    m=root/'evidence-manifest.json'
    assert hashlib.sha256(m.read_bytes()).hexdigest()==seals[stage]
    p=root/filename
    assert hashlib.sha256(p.read_bytes()).hexdigest()==json.loads(m.read_text())['files'][filename]['sha256']
    return p


@pytest.mark.parametrize('pattern',['separable','asymmetric','localized'])
@pytest.mark.parametrize('stage,kind',[('c','zero'),('d','zero'),('d','fixed'),('d','current'),('d','open')])
def test_original_stage_c_d(pattern,stage,kind):
    filename=f'{pattern}-16.npz' if stage=='c' else f'{pattern}-16-{kind}.npz'
    with np.load(stage_file(stage,filename)) as ref:
        I=ref['I'];c=closure_for(kind,.3 if kind=='open' else ([.3,0.] if kind=='fixed' else [.06,0.]))
        s,_=solve_static_material(plane(I),closure=c)
        oracle(I,s,(2*np.pi,2*np.pi),'float64')
        for key in ('q','psi'):
            np.testing.assert_allclose(getattr(s,key),ref[key],rtol=1e-8,atol=1e-8)
        if 'b' in ref:np.testing.assert_allclose(s.b,ref['b'],rtol=1e-8,atol=1e-8)


def test_open_current_is_not_zero_by():
    with np.load(stage_file('d','asymmetric-16-open.npz')) as ref:I=ref['I']
    fixed,fd=solve_static_material(plane(I),closure=closure_for('fixed',(.3,0.)))
    opened,od=solve_static_material(plane(I),closure=closure_for('open',.3))
    assert abs(fixed.b[1])<1e-10 and abs(dict(fd.observations)['mean_current_y'])>1e-5
    assert abs(opened.b[1])>1e-5 and abs(dict(od.observations)['mean_current_y'])<1e-9


def test_1d_allocations_and_driver_remain_independent(monkeypatch):
    import lcprop.pr.unified.static as static
    from lcprop.pr.unified import _newton
    original=_newton.solve_domain
    shapes=[]
    def observed(g,*args,**kwargs):
        shapes.append(g.shape)
        assert len(g.shape)==1 and g.size==64
        return original(g,*args,**kwargs)
    def prohibited(*args,**kwargs):raise AssertionError('1D routed to connected plane')
    monkeypatch.setattr(_newton,'solve_domain',observed)
    monkeypatch.setattr(static,'_solve_connected_plane',prohibited)
    I=np.stack([np.ones(64),1+.5*np.cos(np.arange(64)*2*np.pi/64)],axis=1)
    state,_=solve_static_material(transport(I),closure=zero())
    assert shapes==[(64,),(64,)] and state.b.shape==(2,1)


def test_connected_bound_before_backend_and_no_partial_state(monkeypatch):
    import lcprop.pr.unified.static as static
    I=np.ones((129,96))
    def prohibited(*args,**kwargs):raise AssertionError('backend allocated')
    with monkeypatch.context() as m:
        m.setattr(static,'MaterialBackend',prohibited)
        with pytest.raises(ValueError,match='bounded direct-solver'):
            solve_static_material(plane(I),closure=closure_for('zero'))
    I=np.ones((8,8));before=I.tobytes()
    def failed(*args,**kwargs):raise RuntimeError('injected Newton failure')
    monkeypatch.setattr(static,'solve_domain',failed)
    with pytest.raises(RuntimeError,match='connected material plane.*injected'):
        solve_static_material(plane(I),closure=closure_for('zero'))
    assert I.tobytes()==before


def test_2d_oracle_detects_carrier_gauge_and_y_closure_faults():
    x,y=np.meshgrid(np.arange(8),np.arange(8),indexing='ij');I=1+.2*np.cos(x+y)
    state,_=solve_static_material(plane(I),closure=closure_for('open',.3))
    for bad in [replace(state,q=state.q+.1),replace(state,psi=state.psi+.1),
                replace(state,b=state.b+np.array([0.,.1]))]:
        with pytest.raises(AssertionError):oracle(I,bad,(2*np.pi,2*np.pi),'float64')


@pytest.mark.parametrize('kind',['zero','fixed','current','open'])
@pytest.mark.parametrize('precision',['float32','float64'])
def test_cupy_connected_parity_and_transfers(kind,precision,monkeypatch):
    cp=pytest.importorskip('cupy',reason='CuPy/CUDA unavailable locally')
    try:
        if cp.cuda.runtime.getDeviceCount()<1:pytest.skip('CuPy/CUDA unavailable locally')
    except cp.cuda.runtime.CUDARuntimeError:pytest.skip('CuPy/CUDA unavailable locally')
    x,y=np.meshgrid(np.arange(8)*2*np.pi/8,np.arange(8)*2*np.pi/8,indexing='ij')
    I=(1+.2*np.cos(x+y)+.1*np.sin(y)).astype(precision)
    c=closure_for(kind,.3 if kind=='open' else ([.3,0.] if kind=='fixed' else [.06,0.]))
    cpu,_=solve_static_material(plane(I,precision=precision),closure=c)
    device=cp.asarray(I);original=cp.asnumpy;transfers=[]
    def guarded(a,*args,**kwargs):
        assert a.ndim<=1 and a.size<=32 and a.nbytes<=256
        transfers.append(a.nbytes)
        return original(a,*args,**kwargs)
    import sys
    previous=sys.getprofile()
    def profile(frame,event,arg):
        if event=='c_call' and getattr(arg,'__name__','') in ('get','item','tolist'):
            obj=getattr(arg,'__self__',None)
            if isinstance(obj,cp.ndarray):assert obj.ndim<=1 and obj.size<=32 and obj.nbytes<=256
        if previous is not None:previous(frame,event,arg)
    with monkeypatch.context() as m:
        m.setattr(cp,'asnumpy',guarded);sys.setprofile(profile)
        try:gpu,_=solve_static_material(plane(device,precision=precision,backend='cupy'),closure=c)
        finally:sys.setprofile(previous)
    assert transfers and max(transfers)<=256
    np.testing.assert_array_equal(cp.asnumpy(device),I)
    tol=dict(rtol=2e-4,atol=2e-5) if precision=='float32' else dict(rtol=1e-8,atol=1e-8)
    for key in ('q','psi','b'):np.testing.assert_allclose(cp.asnumpy(getattr(gpu,key)),getattr(cpu,key),**tol)
    host=replace(gpu,q=cp.asnumpy(gpu.q),psi=cp.asnumpy(gpu.psi),b=cp.asnumpy(gpu.b),backend='numpy')
    oracle(I,host,(2*np.pi,2*np.pi),precision)


@pytest.mark.parametrize('kind',['fixed','open'])
def test_connected_jacobian_and_constant_only_nullspace(kind):
    from lcprop.pr.unified._newton import system,residual
    B=MaterialBackend('numpy','float64');g=Geometry((8,6),(5.,4.),B)
    eigenvalues=np.linalg.eigvalsh(g.sparse_poisson().toarray())
    assert np.count_nonzero(abs(eigenvalues)<1e-10)==1
    rng=np.random.default_rng(17)
    I=1+.1*rng.random(g.shape);q=.01*rng.normal(size=g.shape);p=.01*rng.normal(size=g.shape);b=np.array([.3,0.])
    U=np.eye(2) if kind=='fixed' else np.diag([1.,0.])
    V=np.zeros((2,2)) if kind=='fixed' else np.diag([0.,1.])
    electrical=(U,V,np.array([.3,0.]))
    r,matrix=system(g,I,q,p,b,electrical);v=rng.normal(size=r.size);N=g.size
    def shifted(t):return residual(g,I,q+t*v[:N].reshape(g.shape),p+t*v[N:2*N].reshape(g.shape),b+t*v[-2:],electrical)
    np.testing.assert_allclose(matrix@v,(shifted(2e-6)-shifted(-2e-6))/4e-6,rtol=2e-8,atol=2e-8)
