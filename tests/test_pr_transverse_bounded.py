"""Full-x-y resident authority, ownership, and whole-generation transactions."""
from dataclasses import replace
import numpy as np
import pytest
from tests.test_pr_transverse_continuation import request
from lcprop.pr.transverse import workflow as w
from lcprop.pr.transverse.storage import TDStoragePolicy, BoundedStorage, allocation_snapshot
from lcprop.pr.transverse.continuation import (
    continue_transverse_td, checkpoint_from_result, run_continuable_transverse_td,
    ContinuationFailure, save_checkpoint, load_checkpoint,
)
from lcprop.pr.transverse.timedependent_transport_codec import (
    encode_pr_transverse_timedependent_transport_result as encode,
    decode_pr_transverse_timedependent_transport_result as decode,
)
from lcprop.pr.scattering import PRCanonicalScatteringSpec
from lcprop.core.execution import CancellationToken


# Existing native full-transverse float64 backend contract; not a new gate.
TOL = 8e-12


def fixture(*, size=12, steps=3, scattering=False, bias=0.):
    r=request(steps)
    return replace(r, grid=replace(r.grid,Nx=size,Ny=size,z_length_um=25.),
        boundary=replace(r.boundary,applied_field_x=bias),
        scattering=PRCanonicalScatteringSpec(epsilon=.01,transverse_correlation_um=.4,
            realization_seed=17,canonical_dz_um=1.) if scattering else None)


def compare(a,b):
    for name in ('psi_initial','psi_final','source_intensity_stack','A_initial','A_final',
                 'longitudinal_intensity_xz','longitudinal_intensity_yz'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    assert (a.completed_steps,a.time_normalized,a.status)==(b.completed_steps,b.time_normalized,b.status)
    assert a.resolved_profile==b.resolved_profile
    assert a.td_scalar_history==b.td_scalar_history
    for key,v in a.diagnostics.items():
        if key in ('memory_policy','storage_execution'):continue
        other=b.diagnostics[key]
        if isinstance(v,(float,int,np.ndarray)):
            np.testing.assert_allclose(other,v,rtol=TOL,atol=TOL)
        else:assert other==v


@pytest.mark.parametrize('k',[1,2,3,8])
@pytest.mark.parametrize('scattering',[False,True])
@pytest.mark.parametrize('bias',[0.])
def test_resident_bounded(k,scattering,bias):
    r=fixture(scattering=scattering,bias=bias)
    a=w.run_pr_transverse_timedependent(r)
    b=w.run_pr_transverse_timedependent(r,storage_policy=TDStoragePolicy('bounded',k))
    compare(a,b)
    assert b.diagnostics['storage_execution']['max_uploaded_planes']<=min(k,5)


@pytest.mark.parametrize('k',[1,2,4])
def test_moderate_chunk_invariance(k):
    r=fixture(size=64,scattering=True)
    compare(w.run_pr_transverse_timedependent(r),w.run_pr_transverse_timedependent(
        r,storage_policy=TDStoragePolicy('bounded',k)))


@pytest.mark.parametrize('policy',['fast','full'])
def test_movies_observer_and_retention(policy):
    r=fixture(scattering=True)
    snapshots=[]
    def observer(p):
        snapshots.append(p.latest_field_state['psi_current'])
        assert p.latest_field_state['psi_current_backend']=='numpy'
        snapshots[-1][:]=100 # detached owned observation must not feed science
    a=w.run_pr_transverse_timedependent(r,progress_callback=lambda p:None)
    b=w.run_pr_transverse_timedependent(r,storage_policy=TDStoragePolicy('bounded',2),progress_callback=observer)
    compare(a,b)
    assert a.td_preview_movie_metadata==b.td_preview_movie_metadata
    np.testing.assert_array_equal(a.td_preview_movie,b.td_preview_movie)
    for result in (a,b):
        encoded=encode(result,result_policy=policy)
        decoded=decode(encoded.payload.metadata,encoded.payload.arrays)
        assert decoded.td_preview_movie_metadata==result.td_preview_movie_metadata
        if policy=='full':np.testing.assert_array_equal(decoded.psi_final,result.psi_final)
        else:assert decoded.psi_final is None


def test_checkpoint_continuation(tmp_path):
    r=fixture(steps=2,scattering=True)
    full=w.run_pr_transverse_timedependent(replace(r,solver=replace(r.solver,Nt=4)))
    first=run_continuable_transverse_td(r,storage_policy=TDStoragePolicy('bounded',2))
    cp=checkpoint_from_result(r,first);save_checkpoint(cp,tmp_path)
    cp=load_checkpoint(tmp_path)
    out=continue_transverse_td(r,cp,storage_policy=TDStoragePolicy('bounded',1))
    for name in ('psi_final','A_final','source_intensity_stack'):
        np.testing.assert_array_equal(getattr(out,name),getattr(full,name))
    assert out.resolved_profile['continuation_segment']['cumulative_time']==full.time_normalized


@pytest.mark.parametrize('cancel',[False,True])
def test_later_chunk_transaction(monkeypatch,cancel):
    r=fixture(steps=3); policy=TDStoragePolicy('bounded',2)
    one=w.run_pr_transverse_timedependent(replace(r,solver=replace(r.solver,Nt=1)),storage_policy=policy)
    original=w.imex_euler_step;calls=[];token=CancellationToken()
    def step(*args,**kw):
        calls.append(1)
        if len(calls)==5:
            if cancel:token.cancel()
            else:raise RuntimeError('later chunk injected failure')
        return original(*args,**kw)
    monkeypatch.setattr(w,'imex_euler_step',step)
    if cancel:
        out=run_continuable_transverse_td(r,cancellation_token=token,storage_policy=policy)
        assert out.status=='cancelled'
        np.testing.assert_array_equal(out.psi_final,one.psi_final)
        assert out.completed_steps==1
        encoded=encode(out);decode(encoded.payload.metadata,encoded.payload.arrays)
    else:
        with pytest.raises(ContinuationFailure) as failure:
            run_continuable_transverse_td(r,storage_policy=policy)
        np.testing.assert_array_equal(failure.value.checkpoint.psi,one.psi_final)
        assert failure.value.checkpoint.completed_steps==1
        assert str(failure.value.__cause__)=='later chunk injected failure'


def test_complete_pass_before_any_update(monkeypatch):
    r=fixture(steps=1);events=[]
    march=w.advance_pr_published_frozen_slice;step=w.imex_euler_step
    def cell(*a,**kw):events.append('cell');return march(*a,**kw)
    def update(*a,**kw):events.append('update');return step(*a,**kw)
    monkeypatch.setattr(w,'advance_pr_published_frozen_slice',cell)
    monkeypatch.setattr(w,'imex_euler_step',update)
    w.run_pr_transverse_timedependent(r,storage_policy=TDStoragePolicy('bounded',2))
    assert events==['cell']*5+['update']*3+['cell']*5


def test_scattering_exact_cache(monkeypatch):
    r=fixture(scattering=True);saved=[];original=w._apply_canonical_scattering_after_slice
    def apply(*args,**kw):saved.append(kw['phase'].copy());return original(*args,**kw)
    monkeypatch.setattr(w,'_apply_canonical_scattering_after_slice',apply)
    w.run_pr_transverse_timedependent(r);a=saved.copy();saved.clear()
    w.run_pr_transverse_timedependent(r,storage_policy=TDStoragePolicy('bounded',2))
    assert len(a)==len(saved)
    for x,y in zip(a,saved):np.testing.assert_array_equal(x,y)


def test_planning_and_backing_inventory():
    a=np.ones((3,4,5),dtype=np.complex128); view=a.real
    m=allocation_snapshot([view,a])
    assert m['unique_backing_bytes']==a.nbytes
    assert m['logical_array_bytes']==a.nbytes+view.nbytes
    assert m['pool_reserved_bytes'] is None
    p=TDStoragePolicy('bounded',2).plan((80,4096,2048),np.float64)
    q=TDStoragePolicy('bounded',2).plan((160,4096,2048),np.float64)
    assert p['device_array_allowance_bytes']==q['device_array_allowance_bytes']
    assert q['longitudinal_backing_bytes']==2*p['longitudinal_backing_bytes']
    with pytest.raises(ValueError,match='host budget'):
        TDStoragePolicy('bounded',2,1).plan((5,12,12),np.float64)
    with pytest.raises(ValueError,match='unbounded'):
        BoundedStorage(TDStoragePolicy('bounded',1),(5,12,12),np.float64,np).upload(np.zeros((2,12,12)))


@pytest.mark.parametrize('policy',[TDStoragePolicy('bad'),TDStoragePolicy('bounded',0),TDStoragePolicy('bounded',True)])
def test_unknown_or_invalid_storage_rejected(policy):
    with pytest.raises(ValueError):w.run_pr_transverse_timedependent(fixture(),storage_policy=policy)


def test_biased_nonlinear_remains_unsupported():
    for policy in (TDStoragePolicy(),TDStoragePolicy('bounded',2)):
        with pytest.raises(ValueError,match='unbiased'):
            w.run_pr_transverse_timedependent(fixture(bias=.2),storage_policy=policy)


def test_fast_previews_and_checkpoint_never_reconstruct_host_volume(monkeypatch):
    from lcprop.pr.transverse import transport
    r=fixture();out=w.run_pr_transverse_timedependent(r,storage_policy=TDStoragePolicy('bounded',2))
    original=transport.state_from_potential;shapes=[]
    def bounded_only(a,**kw):
        shapes.append(a.shape)
        assert a.ndim==2
        return original(a,**kw)
    monkeypatch.setattr(transport,'state_from_potential',bounded_only)
    encode(out,result_policy='fast')
    checkpoint_from_result(r,out).validate()
    assert shapes


def test_derived_state_arrays_identical():
    from lcprop.pr.transverse.transport import state_from_potential
    r=fixture(scattering=True);a=w.run_pr_transverse_timedependent(r)
    b=w.run_pr_transverse_timedependent(r,storage_policy=TDStoragePolicy('bounded',2))
    p=a.resolved_profile
    kw=dict(dx_normalized=p['dx_normalized'],dy_normalized=p['dy_normalized'])
    sa=state_from_potential(a.psi_final,**kw);sb=state_from_potential(b.psi_final,**kw)
    for name in ('psi','carrier_density','E_x','E_y'):
        np.testing.assert_array_equal(getattr(sa,name),getattr(sb,name))


def test_preexecution_planner_requires_explicit_workspace_and_authorization():
    from lcprop.pr.transverse.storage import plan_td_storage
    shape=(80,128,128); kw=dict(shape=shape,dtype=np.float64,host_budget_bytes=1024**3,fft_workspace_bytes=0)
    assert plan_td_storage(**kw,device_budget_bytes=1024**3).mode=='resident'
    with pytest.raises(ValueError,match='not authorized'):
        plan_td_storage(**kw,device_budget_bytes=32*1024**2)
    p=plan_td_storage(**kw,device_budget_bytes=32*1024**2,allow_bounded=True)
    assert p.mode=='bounded' and 1<=p.chunk_planes<80
    assert p.plan(shape,np.float64)['device_array_allowance_bytes']<=32*1024**2
    with pytest.raises(ValueError,match='one full transverse'):
        plan_td_storage(**kw,device_budget_bytes=1,allow_bounded=True)
