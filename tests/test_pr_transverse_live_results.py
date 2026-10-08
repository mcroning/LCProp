"""Bounded accepted-pass observation, using deterministic local fixtures only."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from tests.test_pr_transverse_continuation import request
from lcprop.pr.transverse import workflow as w
from lcprop.pr.transverse.live_preview import AcceptedPassObserver, LatestPreview
from lcprop.pr.transverse.continuation import checkpoint_from_result,continue_transverse_td
from lcprop.core.execution import CancellationToken


def same(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    else:assert a==b


def observer(interval=0.,clock=lambda:0.):
    box=LatestPreview('run')
    return AcceptedPassObserver(box,segment_id='segment',interval_seconds=interval,clock=clock),box


@pytest.mark.parametrize('scattering',[False,True])
def test_scientific_parity_and_pass_identity(monkeypatch,scattering):
    r=request(3)
    if scattering:
        from lcprop.pr.scattering import PRCanonicalScatteringSpec
        r=replace(r,scattering=PRCanonicalScatteringSpec(epsilon=.01,
            transverse_correlation_um=.4,realization_seed=7,canonical_dz_um=1.))
    events=[];original=w._optical_pass
    def march(*args,**kw):
        result=original(*args,**kw);events.append((args[1].copy(),result[1].copy()));return result
    monkeypatch.setattr(w,'_optical_pass',march)
    a=w.run_pr_transverse_timedependent(r);baseline=list(events);events.clear()
    obs,box=observer();frames=[];pub=box.publish
    def publish(f):frames.append(f);pub(f)
    box.publish=publish
    b=w.run_pr_transverse_timedependent(r,optical_preview=obs)
    for name in ('psi_final','A_final','source_intensity_stack','psi_initial','A_initial',
                 'longitudinal_intensity_xz','longitudinal_intensity_yz'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    same(a.diagnostics,b.diagnostics)
    same(a.td_scalar_history,b.td_scalar_history)
    same(a.resolved_profile,b.resolved_profile)
    assert (a.status,a.power_initial,a.power_final)==(b.status,b.power_initial,b.power_final)
    assert a.completed_steps==b.completed_steps and a.time_normalized==b.time_normalized
    assert checkpoint_from_result(r,a).identity==checkpoint_from_result(r,b).identity
    assert len(events)==len(baseline)==r.solver.Nt+1
    assert [f.segment_step for f in frames]==[0,1,2,3]
    assert [f.final_replay for f in frames]==[False]*3+[True]
    for f,(state,source),(old_state,old_source) in zip(frames,events,baseline):
        np.testing.assert_array_equal(state,old_state);np.testing.assert_array_equal(source,old_source)
        ref=b.resolved_profile['source_normalization'];bg=ref['dark_fraction']+ref['uniform_fraction']
        scale=ref['reference_irradiance_W_cm2']/ref['optical_scale_W_cm2']
        np.testing.assert_array_equal(f.intensity_xz,(source[:,:,(source.shape[2]-1)//2]-bg)*scale)
        np.testing.assert_array_equal(f.z_um,(np.arange(source.shape[0])+1)*r.grid.dz_um)
        assert f.cumulative_time==f.segment_step*r.solver.dt_normalized
        assert not f.intensity_xz.flags.writeable
    assert box.take() is frames[-1] and box.take() is None


def test_continuation_ledger_and_segment():
    r=request(3,dt=.1);first=w.run_pr_transverse_timedependent(r);cp=checkpoint_from_result(r,first)
    next_r=replace(r,solver=replace(r.solver,Nt=2))
    obs,box=observer();frames=[];box.publish=frames.append
    out=continue_transverse_td(next_r,cp,optical_preview=obs)
    assert frames[0].observed_step==cp.completed_steps
    assert frames[0].cumulative_time==cp.time_normalized
    assert frames[-1].cumulative_time==out.resolved_profile["continuation_segment"]["cumulative_time"]
    assert frames[-1].observed_step==out.resolved_profile["continuation_segment"]["cumulative_completed_steps"]
    assert all(f.segment_id=='segment' for f in frames)


def test_partial_and_candidate_failure(monkeypatch):
    r=request(3);obs,box=observer();frames=[];box.publish=frames.append
    original=w._optical_pass
    def broken(*a,**k):
        original(*a,**k)
        raise RuntimeError('partial pass before return')
    monkeypatch.setattr(w,'_optical_pass',broken)
    with pytest.raises(RuntimeError):w.run_pr_transverse_timedependent(r,optical_preview=obs)
    assert frames==[]
    monkeypatch.setattr(w,'_optical_pass',original)
    def bad(*a,**k):return np.full_like(a[0],np.nan)
    monkeypatch.setattr(w,'imex_euler_step',bad)
    with pytest.raises(ValueError,match="finite"):w.run_pr_transverse_timedependent(r,optical_preview=obs)
    assert [f.segment_step for f in frames]==[0]


def test_cancellation_one_pass_lag(monkeypatch):
    r=request(3);token=CancellationToken();obs,box=observer();frames=[];scalars=[]
    box.publish=frames.append
    def progress(p):
        scalars.append(p);token.cancel()
        assert frames[-1].segment_step==p.completed_units-1
    result=w.run_pr_transverse_timedependent(r,optical_preview=obs,progress_callback=progress,cancellation_token=token)
    assert result.status=='cancelled' and result.completed_steps==1
    assert [f.segment_step for f in frames]==[0,1] and frames[-1].final_replay


def test_throttle_bounds_mailbox_and_errors():
    now=[0.];obs,box=observer(.5,lambda:now[0])
    class Source:
        shape=(1000,4096,2048)
        def __array__(self,*args):raise AssertionError('full volume conversion')
        def __getitem__(self,key):
            iz,ix,iy=key
            assert iz.size<=256 and ix.size<=128 and iy==1023
            return np.ones((iz.size,ix.size))
    args=dict(grid=SimpleNamespace(dx_um=.5,dy_um=.25,dz_um=2.),reference=2.,background=.1,dt=.01)
    source=Source()
    obs.observe(source,step=0,pass_id=1,**args);first=box.take()
    assert first.intensity_xz.nbytes==262144
    obs.observe(source,step=1,pass_id=2,**args);assert box.take() is None
    now[0]=.5;obs.observe(source,step=2,pass_id=3,**args)
    obs.observe(source,step=3,pass_id=4,final=True,**args)
    last=box.take();assert last.pass_id==4
    box.publish(first);box.publish(replace(last,run_id='old',pass_id=100));assert box.take() is None
    obs.observe(source,step=3,pass_id=4,final=True,**args);assert box.take() is None
    box.close();box.publish(replace(last,pass_id=5));assert box.take() is None
    obs.observe(source,step=4,pass_id=6,final=True,**dict(args,reference=float('nan')))
    assert 'nonfinite' in box.error


def test_zero_optical_scale():
    obs,box=observer()
    obs.observe(np.ones((2,3,4)),grid=SimpleNamespace(dx_um=1.,dy_um=2.,dz_um=3.),
        reference=SimpleNamespace(optical_scale_W_cm2=0),background=1.,step=0,dt=.1,pass_id=1)
    assert np.count_nonzero(box.take().intensity_xz)==0


def test_cancel_inside_incomplete_pass_retains_only_final_replay(monkeypatch):
    r=request(3);token=CancellationToken();obs,box=observer();frames=[];box.publish=frames.append
    original=w.advance_pr_published_frozen_slice
    def stop(*a,**kw):
        result=original(*a,**kw);token.cancel();return result
    monkeypatch.setattr(w,'advance_pr_published_frozen_slice',stop)
    out=w.run_pr_transverse_timedependent(r,cancellation_token=token,optical_preview=obs)
    assert out.completed_steps==0
    assert len(frames)==1 and frames[0].final_replay and frames[0].segment_step==0


def test_observation_failure_does_not_change_science():
    r=request(2);a=w.run_pr_transverse_timedependent(r);obs,box=observer(clock=lambda:1/0)
    b=w.run_pr_transverse_timedependent(r,optical_preview=obs)
    assert 'ZeroDivisionError' in box.error
    assert box.take() is None
    np.testing.assert_array_equal(a.psi_final,b.psi_final)
    np.testing.assert_array_equal(a.A_final,b.A_final)
    assert checkpoint_from_result(r,a).identity==checkpoint_from_result(r,b).identity


def test_latest_only_thousand_publications():
    import weakref,gc
    from lcprop.pr.transverse.live_preview import OpticalPassPreview
    box=LatestPreview('run');refs=[]
    for i in range(1000):
        f=OpticalPassPreview('run','s',i,i,i,float(i),np.zeros((2,2)),np.arange(2.),np.arange(2.),0.,(2,2,2),False)
        refs.append(weakref.ref(f));box.publish(f)
    del f;gc.collect()
    assert sum(ref() is not None for ref in refs)==1
    last=box.take();assert last.pass_id==999
    box.publish(replace(last,segment_id='delayed-other-segment',pass_id=1000));assert box.take() is None


def test_lineage_numbering_reload_and_same_time_illumination(tmp_path):
    from lcprop.pr.transverse.continuation import save_checkpoint,load_checkpoint
    from lcprop.pr.transverse.live_preview import next_segment_number
    r=request(2);obs,box=observer();frames=[];box.publish=frames.append
    first=w.run_pr_transverse_timedependent(r,optical_preview=obs)
    assert {f.segment_number for f in frames}=={1}
    cp=checkpoint_from_result(r,first);assert next_segment_number(cp)==2
    save_checkpoint(cp,tmp_path/'checkpoint');cp=load_checkpoint(tmp_path/'checkpoint')
    changed=replace(r,material=replace(r.material,uniform_irradiance_W_cm2=4.))
    second_obs=AcceptedPassObserver(LatestPreview('continuation'),segment_id='second',interval_seconds=0,total_steps=2)
    second_frames=[];second_obs.mailbox.publish=second_frames.append
    second=continue_transverse_td(changed,cp,optical_preview=second_obs)
    plain=continue_transverse_td(changed,cp)
    for name in ('A_final','psi_final','source_intensity_stack'):
        np.testing.assert_array_equal(getattr(second,name),getattr(plain,name))
    assert checkpoint_from_result(changed,second).identity==checkpoint_from_result(changed,plain).identity
    assert second_frames[0].cumulative_time==frames[-1].cumulative_time
    assert second_frames[0].segment_id!=frames[-1].segment_id
    assert {f.segment_number for f in second_frames}=={2}
    assert second_frames[0].total_steps==2
    cp2=checkpoint_from_result(changed,second)
    assert next_segment_number(cp2)==3
    assert next_segment_number(SimpleNamespace(record={})) is None
    assert next_segment_number(SimpleNamespace(record={'lineage':[{'segment_index':9}]})) is None


def test_preservation_requires_bound_checkpoint_and_compatible_state():
    from lcprop.pr.transverse.live_preview import can_preserve_preview
    r=request(1);cp=checkpoint_from_result(r,w.run_pr_transverse_timedependent(r))
    assert can_preserve_preview(r,cp,cp.identity)
    changed=replace(r,material=replace(r.material,uniform_irradiance_W_cm2=4.))
    assert can_preserve_preview(changed,cp,cp.identity)
    assert not can_preserve_preview(r,cp,'different-checkpoint')
    incompatible=replace(r,grid=replace(r.grid,Nx=24))
    assert not can_preserve_preview(incompatible,cp,cp.identity)
    assert not can_preserve_preview(r,None,cp.identity)
