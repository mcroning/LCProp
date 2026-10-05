"""Bounded continuation science and ownership, independent of viewer state."""
from dataclasses import replace
import numpy as np
import pytest
from tests.test_pr_transverse_production import _request
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse import workflow as w
from lcprop.pr.transverse.continuation import (
    checkpoint_from_result,continue_transverse_td,validate_continuation,
    save_checkpoint,load_checkpoint,ContinuationFailure)


def request(steps=2,dt=.001):
    r=_request(steps=steps)
    m=replace(r.material,normalization_identity='pr_integral_total_illumination_mean_irradiance_v1',
        dark_intensity=0.,uniform_background_intensity=0.,
        dark_irradiance_W_cm2=10.,uniform_irradiance_W_cm2=2.)
    return replace(r,material=m,solver=replace(r.solver,dt_normalized=dt))


@pytest.mark.parametrize('precision',['float32','float64'])
@pytest.mark.parametrize('scattering',[False,True])
def test_segment_equivalence_and_save_reopen(tmp_path,precision,scattering,record_property):
    from lcprop.pr.scattering import PRCanonicalScatteringSpec
    from lcprop.pr.transverse.continuation import state_identity
    r=request(5)
    r=replace(r,backend=replace(r.backend,precision=precision),
        scattering=PRCanonicalScatteringSpec(epsilon=.01,transverse_correlation_um=.4,
            realization_seed=17,canonical_dz_um=1.) if scattering else None)
    full=w.run_pr_transverse_timedependent(r)
    first=replace(r,solver=replace(r.solver,Nt=2))
    result=w.run_pr_transverse_timedependent(first)
    cp=checkpoint_from_result(first,result)
    assert not cp.psi.flags.writeable
    next_r=replace(r,solver=replace(r.solver,Nt=3))
    second=continue_transverse_td(next_r,cp)
    np.testing.assert_array_equal(second.psi_initial,result.psi_final)
    for name in ('psi_final','A_final','source_intensity_stack'):
        np.testing.assert_array_equal(getattr(second,name),getattr(full,name))
    assert second.resolved_profile['continuation_segment']['cumulative_time']==full.time_normalized
    save_checkpoint(cp,tmp_path)
    opened=load_checkpoint(tmp_path)
    again=continue_transverse_td(next_r,opened)
    np.testing.assert_array_equal(again.psi_final,second.psi_final)
    assert again.resolved_profile==second.resolved_profile
    record_property("final_material_identity",state_identity(full.psi_final))
    record_property("endpoint_identity",state_identity(full.A_final))
    record_property("source_identity",state_identity(full.source_intensity_stack))


@pytest.mark.parametrize('uniform',[0.,30.])
def test_turnoff_uniform(uniform):
    r=request();out=w.run_pr_transverse_timedependent(r);cp=checkpoint_from_result(r,out)
    dark=replace(r,beams=replace(r.beams,channels=tuple(replace(c,power_mW=0.) for c in r.beams.channels)),
        material=replace(r.material,uniform_irradiance_W_cm2=uniform))
    result=continue_transverse_td(dark,cp)
    np.testing.assert_array_equal(result.psi_initial,out.psi_final)
    assert np.count_nonzero(result.A_initial)==np.count_nonzero(result.A_final)==0
    np.testing.assert_allclose(result.source_intensity_stack,1.,rtol=0,atol=0)
    ref=result.resolved_profile['source_normalization']
    assert ref['reference_power_W']==ref['area_cm2']*(10.+uniform)
    assert ref['uniform_fraction']==uniform/(10.+uniform)
    assert result.diagnostics['physical_state_valid']
    checkpoint_from_result(dark,result).validate()


def test_order_state_and_transaction(monkeypatch):
    r=request(2);out=w.run_pr_transverse_timedependent(r);cp=checkpoint_from_result(r,out)
    original_pass=w._optical_pass; original_step=w.imex_euler_step
    events=[]
    def march(A,psi,**kw):
        events.append('source')
        if len(events)==1:np.testing.assert_array_equal(psi,cp.psi)
        return original_pass(A,psi,**kw)
    def update(psi,source,**kw):
        events.append('update')
        assert events[-2]=='source'
        if events.count('update')==2:raise RuntimeError('injected candidate failure')
        return original_step(psi,source,**kw)
    monkeypatch.setattr(w,'_optical_pass',march);monkeypatch.setattr(w,'imex_euler_step',update)
    one=w.run_pr_transverse_timedependent(replace(r,solver=replace(r.solver,Nt=1)),_continuation_psi=cp.psi)
    events.clear()
    with pytest.raises(ContinuationFailure) as failure:continue_transverse_td(r,cp)
    failed=failure.value.checkpoint
    np.testing.assert_array_equal(failed.psi,one.psi_final)
    assert failed.completed_steps==cp.completed_steps+1
    assert failed.record['status']=='failed'
    assert isinstance(failure.value.__cause__,RuntimeError)
    assert events==['source','update','source','update']


@pytest.mark.parametrize('change',[
    lambda r:replace(r,grid=replace(r.grid,Nx=14)),
    lambda r:replace(r,backend=replace(r.backend,precision='float32')),
    lambda r:replace(r,material=PRMaterialSpec()),
    lambda r:replace(r,solver=replace(r.solver,integrator='explicit_euler_reference')),
    lambda r:replace(r,material=replace(r.material,refractive_index=2.5)),
    lambda r:replace(r,initial_psi=np.zeros((2,12,10))),
])
def test_incompatible(change):
    r=request(1);cp=checkpoint_from_result(r,w.run_pr_transverse_timedependent(r))
    with pytest.raises((ValueError,TypeError)):validate_continuation(change(r),cp)


def test_cancelled_source_and_candidate(monkeypatch):
    from lcprop.core.execution import CancellationToken
    r=request(3);token=CancellationToken()
    out=w.run_pr_transverse_timedependent(r,cancellation_token=token,progress_callback=lambda p:token.cancel())
    assert out.status=='cancelled' and out.completed_steps==1
    cp=checkpoint_from_result(r,out)
    token=CancellationToken();original=w.imex_euler_step
    def cancel_after(*a,**kw):
        candidate=original(*a,**kw);token.cancel();return candidate
    monkeypatch.setattr(w,'imex_euler_step',cancel_after)
    result=continue_transverse_td(r,cp,cancellation_token=token)
    np.testing.assert_array_equal(result.psi_final,cp.psi)
    assert result.completed_steps==0
    assert result.resolved_profile['continuation_segment']['cumulative_time']==out.time_normalized


def independent_constraints(result,r):
    # Stage-3 periodic first-derivative convention, constructed independently.
    p=result.psi_final;dx=r.material.characteristic_wavenumber_per_um*r.grid.x_aperture_um/r.grid.Nx
    dy=r.material.characteristic_wavenumber_per_um*r.grid.y_aperture_um/r.grid.Ny
    kx=2*np.pi*np.fft.fftfreq(r.grid.Nx,d=dx)[:,None]
    ky=2*np.pi*np.fft.fftfreq(r.grid.Ny,d=dy)[None,:]
    if r.grid.Nx%2==0:kx[r.grid.Nx//2]=0
    if r.grid.Ny%2==0:ky[:,r.grid.Ny//2]=0
    fft=lambda a:np.fft.fft2(a,axes=(-2,-1))
    inv=lambda a:np.fft.ifft2(a,axes=(-2,-1)).real
    K=kx*kx+ky*ky
    c=1+inv(K*fft(p));ex=-inv(1j*kx*fft(p));ey=-inv(1j*ky*fft(p))
    weighted=c*result.source_intensity_stack
    jx=weighted*ex-inv(1j*kx*fft(weighted));jy=weighted*ey-inv(1j*ky*fft(weighted))
    F=w.potential_rhs(p,result.source_intensity_stack,dx_normalized=dx,dy_normalized=dy,xp=np)
    metrics={'gauge':float(abs(p.mean(axis=(-2,-1))).max()),
        'neutrality':float(abs(c.mean(axis=(-2,-1))-1).max()),
        'gauss':float(abs(inv(1j*kx*fft(ex)+1j*ky*fft(ey))-c+1).max()),
        'curl':float(abs(inv(1j*kx*fft(ey)-1j*ky*fft(ex))).max()),
        'continuity':float(abs(inv(K*fft(F)+1j*kx*fft(jx)+1j*ky*fft(jy))).max())}
    assert np.isfinite(c).all() and c.min()>0
    assert metrics['gauge']<1e-11 and metrics['neutrality']<1e-11
    assert metrics['gauss']<1e-9 and metrics['curl']<1e-9
    return metrics


@pytest.mark.parametrize('mode',['dark','uniform','power','shift'])
def test_continued_refinement(mode,record_property):
    r=request(10,.002);out=w.run_pr_transverse_timedependent(r);cp=checkpoint_from_result(r,out)
    ch=r.beams.channels[0]
    if mode in ('dark','uniform'):ch=replace(ch,power_mW=0)
    if mode=='power':ch=replace(ch,power_mW=.25)
    if mode=='shift':ch=replace(ch,x0_um=1.)
    r=replace(r,beams=replace(r.beams,channels=(ch,)))
    if mode=='uniform':r=replace(r,material=replace(r.material,uniform_irradiance_W_cm2=50.))
    states=[];constraints=[]
    for n in (16,32,64,128):
        segment=continue_transverse_td(replace(r,solver=replace(r.solver,Nt=n,dt_normalized=.02/n)),cp)
        states.append(segment.psi_final)
        constraints.append(independent_constraints(segment,r))
        assert segment.diagnostics['physical_state_valid']
        np.testing.assert_array_equal(segment.psi_initial,cp.psi)
    errors=[np.linalg.norm(states[i]-states[i+1]) for i in range(3)]
    ratios=[errors[i]/errors[i+1] for i in range(2)]
    record_property(mode+'_ratios',[float(v) for v in ratios])
    record_property(mode+'_successive_errors',[float(v) for v in errors])
    record_property(mode+'_minimum_carrier',segment.diagnostics['carrier_density_minimum'])
    record_property(mode+'_constraints',constraints)
    assert all(1.8<v<2.6 for v in ratios)
    if mode in ('dark','uniform'):assert np.linalg.norm(states[-1])<np.linalg.norm(cp.psi)


def test_result_codec_and_products_dark():
    from lcprop.pr.transverse.timedependent_transport_codec import (
        encode_pr_transverse_timedependent_transport_result as encode,
        decode_pr_transverse_timedependent_transport_result as decode)
    from lcprop.pr.transverse.operations import PR_TRANSVERSE_TIMEDEPENDENT_OPERATION as operation
    r=request(1);cp=checkpoint_from_result(r,w.run_pr_transverse_timedependent(r))
    dark=replace(r,beams=replace(r.beams,channels=(replace(r.beams.channels[0],power_mW=0.),)))
    out=continue_transverse_td(dark,cp)
    operation.to_run_data(out)
    for policy in ('fast','full'):
        payload=encode(out,result_policy=policy).payload
        reopened=decode(payload.metadata,payload.arrays)
        assert reopened.resolved_profile['continuation_segment']==out.resolved_profile['continuation_segment']
        operation.to_run_data(reopened)
        if policy=='full':
            saved=checkpoint_from_result(dark,reopened)
            np.testing.assert_array_equal(continue_transverse_td(dark,saved).psi_final,
                continue_transverse_td(dark,checkpoint_from_result(dark,out)).psi_final)
        else:
            with pytest.raises(ValueError):checkpoint_from_result(dark,reopened)


def test_ordinary_identity_and_integrator_bytes():
    import subprocess,types
    from pathlib import Path
    base='bc2fe7c6029687a2c77f379711107a044f3296e2'
    for p in ('src/lcprop/pr/transverse/transport.py','src/lcprop/pr/transverse/projection.py',
              'src/lcprop/pr/evolution.py'):
        assert Path(p).read_bytes()==subprocess.check_output(['git','show',base+':'+p])
    code=subprocess.check_output(['git','show',base+':src/lcprop/pr/transverse/workflow.py'],text=True)
    module=types.ModuleType('lcprop.pr.transverse._frozen_stage5_base')
    exec(compile(code,'frozen-workflow','exec'),module.__dict__)
    r=request(3);a=w.run_pr_transverse_timedependent(r);b=module.run_pr_transverse_timedependent(r)
    for name in ('psi_initial','psi_final','A_initial','A_final','source_intensity_stack',
                 'longitudinal_intensity_xz','longitudinal_intensity_yz'):
        np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
    np.testing.assert_equal(a.diagnostics,b.diagnostics)
    assert {k:v for k,v in a.resolved_profile.items() if k != "continuation_request"}==b.resolved_profile
    assert a.td_scalar_history==b.td_scalar_history


def test_callbacks_and_samples():
    r=request(4);samples=[]
    full=w.run_pr_transverse_timedependent(r,progress_callback=lambda p:samples.append(p.latest_field_state['psi_current'].copy()))
    first_r=replace(r,solver=replace(r.solver,Nt=2))
    first=w.run_pr_transverse_timedependent(first_r);cp=checkpoint_from_result(first_r,first)
    progress=[];second=continue_transverse_td(first_r,cp,progress_callback=progress.append)
    plain=continue_transverse_td(first_r,cp)
    for a,b in zip(progress,samples[2:]):np.testing.assert_array_equal(a.latest_field_state['psi_current'],b)
    for name in ('psi_final','A_final','source_intensity_stack'):
        np.testing.assert_array_equal(getattr(second,name),getattr(plain,name))
        np.testing.assert_array_equal(getattr(second,name),getattr(full,name))
    assert progress[-1].cumulative_completed_steps==4
    assert progress[-1].cumulative_time==full.time_normalized


def test_gui_locked_editor_and_results(tmp_path):
    from PySide6.QtWidgets import QApplication,QTableWidgetItem
    from lcprop.pr.gui.transverse_continuation import TransverseContinuationDialog
    from lcprop.pr.gui.main_window import PRMainWindow,_continue_transverse_operation
    app=QApplication.instance() or QApplication([])
    r=request(1);cp=checkpoint_from_result(r,w.run_pr_transverse_timedependent(r))
    dialog=TransverseContinuationDialog(cp)
    assert dialog.build_request()==cp.request
    assert 'source_cumulative_time' in dialog.preview.toPlainText()
    dialog.beams.setItem(0,0,QTableWidgetItem('0'))
    dialog.uniform.setText('30000')
    changed=dialog.build_request()
    assert changed.grid==cp.request.grid and changed.backend==cp.request.backend
    assert changed.material.refractive_index==cp.request.material.refractive_index
    assert changed.material.uniform_irradiance_W_cm2==30.
    from lcprop.pr.gui.request_adapter import validate_pr_gui_workflow_request
    validate_pr_gui_workflow_request(changed)
    result=_continue_transverse_operation(changed,checkpoint=cp)
    assert result.run_data is not None
    assert result.result.resolved_profile['continuation_segment']['source_checkpoint']==cp.identity
    window=PRMainWindow();window.last_checkpoint=cp;window._refresh_checkpoint_controls()
    assert window.continue_button.isEnabled()
    window.save_checkpoint_to(tmp_path)
    loaded=window.load_checkpoint_from(tmp_path)
    assert loaded.identity==cp.identity
    window.close();dialog.close()


def test_tamper_and_zero_total_rejected(tmp_path):
    r=request(1);cp=checkpoint_from_result(r,w.run_pr_transverse_timedependent(r))
    with pytest.raises(ValueError):cp.psi.setflags(write=True)
    bad=replace(r,beams=replace(r.beams,channels=(replace(r.beams.channels[0],power_mW=0.),)),
        material=replace(r.material,dark_irradiance_W_cm2=0.,uniform_irradiance_W_cm2=0.))
    with pytest.raises(ValueError,match='total physical illumination'):validate_continuation(bad,cp)
    save_checkpoint(cp,tmp_path)
    a=np.load(tmp_path/'accepted-psi.npy');a.flat[0]+=1e-5;np.save(tmp_path/'accepted-psi.npy',a)
    with pytest.raises(ValueError,match='identity mismatch'):load_checkpoint(tmp_path)


def test_complete_source_binding_and_gui_failure(monkeypatch,tmp_path):
    from PySide6.QtWidgets import QApplication
    from lcprop.pr.gui.main_window import PRMainWindow,_continue_transverse_operation
    from lcprop.pr.scattering import PRCanonicalScatteringSpec
    from lcprop.pr.transverse.continuation import run_continuable_transverse_td
    app=QApplication.instance() or QApplication([])
    r=request(2);out=w.run_pr_transverse_timedependent(r);cp=checkpoint_from_result(r,out)
    with pytest.raises(ValueError,match='complete continuation request'):
        checkpoint_from_result(replace(r,scattering=PRCanonicalScatteringSpec(epsilon=.01,transverse_correlation_um=.4,realization_seed=17,canonical_dz_um=1.)),out)
    window=PRMainWindow();window.last_checkpoint=cp
    original=w.imex_euler_step;count=[0]
    def fail_second(*a,**kw):
        count[0]+=1
        if count[0]==2:raise RuntimeError('bounded failure')
        return original(*a,**kw)
    monkeypatch.setattr(w,'imex_euler_step',fail_second)
    with pytest.raises(ContinuationFailure):
        _continue_transverse_operation(r,checkpoint=cp,progress_callback=window._on_progress)
    assert window.last_checkpoint.completed_steps==cp.completed_steps+1
    assert window.last_checkpoint.time_normalized==(cp.completed_steps+1)*r.solver.dt_normalized
    window.save_checkpoint_to(tmp_path)
    assert load_checkpoint(tmp_path).identity==window.last_checkpoint.identity
    count[0]=0
    with pytest.raises(ContinuationFailure) as failure:
        run_continuable_transverse_td(r)
    assert failure.value.checkpoint.completed_steps==1
    window.close()


def test_canonical_time_blocks():
    r=request(1,.001);out=w.run_pr_transverse_timedependent(r);cp=checkpoint_from_result(r,out)
    from lcprop.pr.transverse.continuation import _time_ledger
    # Isolate accounting from dynamics: same cadence is coalesced, not
    # repeatedly rounded by segment addition; new cadence never rescales past.
    blocks,time=_time_ledger(cp,8,.001)
    assert time==9*.001 and blocks==[{'steps':9,'dt':.001}]
    blocks,time=_time_ledger(cp,2,.0005)
    assert time==.002 and blocks==[{'steps':1,'dt':.001},{'steps':2,'dt':.0005}]


def test_cancel_before_step_and_prelaunch_failure(monkeypatch):
    from lcprop.core.execution import CancellationToken
    r=request(1);cp=checkpoint_from_result(r,w.run_pr_transverse_timedependent(r))
    token=CancellationToken();token.cancel()
    out=continue_transverse_td(r,cp,cancellation_token=token)
    np.testing.assert_array_equal(out.psi_final,cp.psi)
    assert out.completed_steps==0 and out.status=='cancelled'
    original=cp.record['source_normalization']
    changed=replace(r,material=replace(r.material,uniform_irradiance_W_cm2=50.))
    def fail(*args,**kwargs):raise RuntimeError('launch construction failed')
    monkeypatch.setattr(w,'build_launch',fail)
    with pytest.raises(ContinuationFailure) as caught:continue_transverse_td(changed,cp)
    preserved=caught.value.checkpoint
    assert preserved.request==cp.request and preserved.time_normalized==cp.time_normalized
    assert preserved.record['source_normalization']==original
    np.testing.assert_array_equal(preserved.psi,cp.psi)


def test_lineage_codec_rejects_incomplete_metadata():
    from copy import deepcopy
    from lcprop.pr.transverse.timedependent_transport_codec import (
        encode_pr_transverse_timedependent_transport_result as encode,
        decode_pr_transverse_timedependent_transport_result as decode)
    r=request(1);cp=checkpoint_from_result(r,w.run_pr_transverse_timedependent(r))
    out=continue_transverse_td(r,cp);payload=encode(out).payload
    for key in ('source_state','source_normalization','time_blocks','new_normalization'):
        meta=deepcopy(payload.metadata)
        meta['resolved_profile']['continuation_segment'].pop(key)
        with pytest.raises(ValueError,match='continuation segment'):decode(meta,payload.arrays)
