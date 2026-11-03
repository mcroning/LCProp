"""Post-acceptance observer contract; bounded scientific fixtures only."""
from dataclasses import replace
import numpy as np
import pytest
from lcprop.core.execution import CancellationToken
from lcprop.pr.unified import workflow as w, products as p, progress as pg
from tests.test_pr_unified_workflow import request
from tests.test_pr_unified_integration import science_equal, exact

SELECT=p.UnifiedSelection(boundary_intensity=True,intensity_cuts=True,far_field=True)


def equal(a,b):
    science_equal(a.scientific,b.scientific)
    for key in a.arrays:exact(a.arrays[key],b.arrays[key])
    for key in a.coordinates:exact(a.coordinates[key],b.coordinates[key])


@pytest.mark.parametrize('precision',['float32','float64'])
@pytest.mark.parametrize('dimension',[1,2])
def test_observer_science_and_detached_payload(precision,dimension):
    r=request(2,precision=precision,dimension=dimension,scatter=True)
    baseline=p.run_unified_products(r,selection=SELECT)
    noop=p.run_unified_products(r,selection=SELECT,observer=lambda e:None)
    events=[]
    def observe(e):
        events.append((e.completed_cells,e.z_um,e.preview.copy()))
        assert e.preview.dtype==np.float32 and max(e.preview.shape)<=128
        e.preview[:]=0;e.x_um[:]=0;e.diagnostics.clear()
    recorded=p.run_unified_products(r,selection=SELECT,observer=observe)
    equal(baseline,noop);equal(baseline,recorded)
    assert [(n,z) for n,z,_ in events]==[(1,2.),(2,4.)]


def test_exact_order_and_no_extra_scientific_calls(monkeypatch,record_property):
    import json
    events=[]
    for name,label in [('hop_linear_inplace','P'),('pr_driving_intensity','I'),
        ('solve_static_material','material'),('electric_field_optical_node','projection'),
        ('apply_response_screen_inplace','phase'),('canonical_scattering_phase_increment','scattering')]:
        original=getattr(w,name)
        def wrapped(*args,_old=original,_label=label,**kwargs):
            events.append(_label);return _old(*args,**kwargs)
        monkeypatch.setattr(w,name,wrapped)
    original_notify=pg.notify_accepted
    def verify_acceptance(observer,field,grid,groups,record,completed,total,xp):
        import inspect
        # Observe the actual transaction locals; do not infer acceptance from callback order.
        caller=inspect.currentframe().f_back.f_locals
        assert caller['accepted'] is field
        assert caller['completed']==completed and caller['reached']==record['z_end_um']
        assert caller['ledger'][-1]==record
        events.append('accepted:'+str(completed))
        return original_notify(observer,field,grid,groups,record,completed,total,xp)
    monkeypatch.setattr(pg,'notify_accepted',verify_acceptance)
    def observe(e):
        events.append('observer')
    out=p.run_unified_products(request(2,scatter=True),selection=SELECT,observer=observe)
    assert out.scientific.completed_cells==2
    cell=['P','I','material','projection','phase','scattering','phase']
    assert events==cell+['accepted:1','observer']+cell+['accepted:2','observer']
    record_property('accepted_event_trace',json.dumps(events))


@pytest.mark.parametrize('site',['hop_linear_inplace','solve_static_material','electric_field_optical_node',
    'canonical_scattering_phase_increment','_prepare_acceptance','_collect'])
def test_preaccept_failure_no_event_for_candidate(monkeypatch,site):
    original=getattr(w,site);calls=0;events=[]
    # _collect includes the launch observation.
    fail_at=3 if site=='_collect' else 2
    def fail(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==fail_at:raise RuntimeError('injected candidate failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(w,site,fail)
    out=p.run_unified_products(request(3,scatter=True),selection=SELECT,observer=events.append)
    s=out.scientific
    assert s.status=='failed' and s.completed_cells==1 and s.reached_z_um==2
    assert len(events)==1 and events[0].completed_cells==1
    assert len(s.ledger)==1 and out.arrays['intensity_cut_x'].shape[0]==2
    assert out.next_address['cell_index']==1


@pytest.mark.parametrize('when',['before','candidate','after'])
def test_cancellation(monkeypatch,when):
    token=CancellationToken();events=[]
    if when=='before':token.cancel()
    if when=='candidate':
        original=w._prepare_acceptance
        def cancel(*a,**k):
            value=original(*a,**k);token.cancel();return value
        monkeypatch.setattr(w,'_prepare_acceptance',cancel)
    def observer(e):events.append(e);token.cancel()
    out=p.run_unified_products(request(3,scatter=True),selection=SELECT,cancellation_token=token,observer=observer)
    n=1 if when=='after' else 0
    assert out.scientific.status=='cancelled'
    assert out.scientific.completed_cells==n and len(events)==n
    assert out.scientific.reached_z_um==2*n
    assert out.arrays['intensity_cut_x'].shape[0]==n+1


def test_observer_error_retains_accepted_state():
    r=request(3,scatter=True);token=CancellationToken()
    good=p.run_unified_products(r,selection=SELECT,cancellation_token=token,observer=lambda e:token.cancel())
    def fail(e):raise RuntimeError('observer display failed')
    bad=p.run_unified_products(r,selection=SELECT,observer=fail)
    assert bad.scientific.status=='failed'
    assert bad.scientific.failure['stage']=='post_acceptance_observer'
    assert bad.scientific.failure['reason']=='observer display failed'
    assert bad.scientific.completed_cells==1 and bad.scientific.reached_z_um==2
    science_equal(replace(bad.scientific,status='cancelled'),good.scientific)
    for key in good.arrays:exact(good.arrays[key],bad.arrays[key])


@pytest.mark.parametrize('backend',['numpy','cupy'])
@pytest.mark.parametrize('dtype',['float32','float64'])
def test_bounded_backend_preview(backend,dtype,monkeypatch,record_property):
    import json
    from types import SimpleNamespace
    from lcprop.pr.visualization import downsample_td_movie_frame
    if backend=='cupy':
        try:
            import cupy as xp
            xp.zeros(1)
        except Exception:pytest.skip('CuPy/CUDA unavailable')
    else:xp=np
    host=np.arange(257*259,dtype=np.float64).reshape(1,257,259)/10000
    A=xp.asarray(host,dtype='complex64' if dtype=='float32' else 'complex128')
    before=A.copy();transfers=[]
    if backend=='cupy':
        original=xp.asnumpy
        def guarded(a,*args,**kwargs):
            transfers.append((a.shape,a.nbytes))
            assert a.size<=128*128 and a.nbytes<=65536
            return original(a,*args,**kwargs)
        monkeypatch.setattr(xp,'asnumpy',guarded)
    events=[]
    pg.notify_accepted(events.append,A,SimpleNamespace(x_um=xp.arange(257),y_um=xp.arange(259)),('channel',),
        {'z_end_um':2.,'cell_index':0,'observations':{}},1,2,xp)
    assert bool(xp.array_equal(A,before))
    e=events[0];assert e.preview.shape==(128,128) and e.preview.nbytes==65536
    if backend=='numpy':
        from lcprop.optics.splitstep import total_intensity
        exact(e.preview,downsample_td_movie_frame(total_intensity(A,coherence_groups=('channel',),xp=np)))
    record_property('preview_transfers',json.dumps(transfers))


@pytest.mark.parametrize('precision',['float32','float64'])
def test_cuts_are_exact_accepted_plane_slices_and_selection_independent(monkeypatch,precision):
    from lcprop.optics.splitstep import total_intensity
    r=request(2,precision=precision,scatter=True);planes=[]
    original=w._collect
    def capture(collector,previous,A,state,I,grid,record,identities,xp):
        planes.append(total_intensity(A,coherence_groups=identities['coherence_groups'],xp=xp).copy())
        return original(collector,previous,A,state,I,grid,record,identities,xp)
    monkeypatch.setattr(w,'_collect',capture)
    selected=p.run_unified_products(r,selection=SELECT)
    minimal=p.run_unified_products(r,selection=p.UnifiedSelection(far_field=True))
    science_equal(selected.scientific,minimal.scientific)
    exact(selected.arrays['far_field_intensity'],minimal.arrays['far_field_intensity'])
    x=selected.coordinates['x_um'];y=selected.coordinates['y_um']
    stack=np.stack(planes)
    exact(selected.arrays['intensity_cut_x'],stack[:,:,np.argmin(abs(y))])
    exact(selected.arrays['intensity_cut_y'],stack[:,np.argmin(abs(x)),:])
    assert selected.coordinates['boundary_z_um'].tolist()==[0.,2.,4.]
    assert selected.coordinates['material_z_um'].tolist()==[2.,4.]


@pytest.mark.parametrize('precision',['float32','float64'])
def test_native_observer_identity_when_available(precision):
    try:
        import cupy as cp
        cp.zeros(1)
    except Exception:pytest.skip('CuPy/CUDA unavailable')
    r=request(2,precision=precision,scatter=True)
    r=replace(r,backend='cupy',initial_A=cp.asarray(r.initial_A))
    no=p.run_unified_products(r,selection=SELECT);events=[]
    yes=p.run_unified_products(r,selection=SELECT,observer=events.append)
    assert yes.scientific.status==no.scientific.status=='completed'
    assert yes.scientific.ledger==no.scientific.ledger and len(events)==2
    for left,right in [(yes.scientific.boundary_field,no.scientific.boundary_field)]+[
            (getattr(yes.scientific.material_state,k),getattr(no.scientific.material_state,k)) for k in ('q','psi','b')]+[
            (yes.arrays[k],no.arrays[k]) for k in no.arrays]:
        assert bool(cp.array_equal(left.view(cp.uint8),right.view(cp.uint8)))
