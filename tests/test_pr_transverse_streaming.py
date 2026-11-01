"""Small synthetic CPU integration tests; never Figure 11 inputs."""
from dataclasses import replace
import hashlib
import json
import numpy as np
import pytest
from tests.test_pr_transverse_marching_static import _request
from lcprop.pr.transverse import streaming as s
from lcprop.pr.frozen_scattering import FrozenScatteringPhases, scattering_context
from lcprop.pr.static_streaming import PRStreamingStaticRequest, PRStreamingStaticOptions


def request(n=7, modulation=0.):
    r=_request()
    a=np.broadcast_to(1+modulation*np.cos(2*np.pi*np.arange(n)/n)[:,None],(n,n)).copy()[None].astype(complex)
    return replace(r,grid=replace(r.grid,Nx=n,Ny=n),initial_A=a)


def binding(tmp_path,r):
    context=PRStreamingStaticRequest(grid=r.grid,beams=r.beams,material=r.material,
        backend=r.backend,initial_A=r.initial_A,
        solver=PRStreamingStaticOptions(volume_noise_epsilon=.02,volume_noise_seeds=(11,22,33)))
    a=np.stack([np.full(r.initial_A.shape[1:],.01*i) for i in range(3)])
    p=tmp_path/'phase.npz';np.savez(p,phase=a)
    m=dict(schema=1,units='rad',layout='zxy',dtype='<f8',shape=list(a.shape),indices=[0,1,2],payload_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),context=scattering_context(context),producer='synthetic',coordinates='synthetic')
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(m))
    source=FrozenScatteringPhases.from_npz(p,manifest_path=mp,manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest())
    return s.TransversePhaseBinding(source,context)


@pytest.mark.parametrize('n',[7,8])
@pytest.mark.parametrize('modulation',[0.,.001])
def test_replay_diagnostics_and_y_reduction(n,modulation):
    out=s.run_pr_transverse_streaming(request(n,modulation),stop_after_accepted_cells=2)
    assert out.converged, out
    assert all(out.diagnostics['replay_checks'].values())
    assert out.first_pass.completed_intervals==2
    assert out.diagnostics['physical_Nz']==3
    for d in out.first_pass.diagnostics['planes']:
        assert d['moments']['E_y'][3]<1e-12
        assert d['full_residual']['maximum']<1e-6
        assert d['carrier_minimum']>0
    json.loads(json.dumps(out.diagnostics,allow_nan=False))
    json.loads(json.dumps(out.first_pass.diagnostics['planes'],allow_nan=False))


def test_source_events_replay_identity(tmp_path,monkeypatch):
    r=request();b=binding(tmp_path,r);events=[]
    original=FrozenScatteringPhases.phase
    def phase(self,index):
        events.append(index);return original(self,index)
    monkeypatch.setattr(FrozenScatteringPhases,'phase',phase)
    out=s.run_pr_transverse_streaming(r,phase_binding=b,stop_after_accepted_cells=2)
    assert out.converged
    assert events==[0,1,0,1]
    baseline=s.run_pr_transverse_streaming(r,stop_after_accepted_cells=2)
    assert np.allclose(out.first_pass.A_final,baseline.first_pass.A_final*np.exp(.01j))


def test_explicit_normalization_scale_invariance():
    r=request(modulation=.001)
    a=s.run_pr_transverse_streaming(r,launch_peak_intensity_reference=4.,stop_after_accepted_cells=1)
    b=s.run_pr_transverse_streaming(replace(r,initial_A=2*r.initial_A),launch_peak_intensity_reference=16.,stop_after_accepted_cells=1)
    assert a.converged and b.converged
    assert np.allclose(a.first_pass.A_final,b.first_pass.A_final/2,rtol=1e-13,atol=1e-13)
    assert np.array_equal(a.first_pass.restart_state.previous_psi,b.first_pass.restart_state.previous_psi)


@pytest.mark.parametrize('count',[0,-1,4,True,1.5])
def test_invalid_prefix(count):
    with pytest.raises(ValueError):s.run_pr_transverse_streaming(request(),stop_after_accepted_cells=count)


@pytest.mark.parametrize('peak',[0,-1,float('nan'),float('inf'),True])
def test_invalid_peak(peak):
    with pytest.raises(ValueError):s.run_pr_transverse_streaming(request(),launch_peak_intensity_reference=peak)


def test_binding_mismatch_before_execution(tmp_path):
    r=request();b=binding(tmp_path,r)
    with pytest.raises(ValueError,match='grid mismatch'):
        s.run_pr_transverse_streaming(replace(r,grid=replace(r.grid,x_aperture_um=41)),phase_binding=b)
    wrong=replace(b,context=replace(b.context,solver=replace(b.context.solver,volume_noise_seeds=(12,22,33))))
    with pytest.raises(ValueError):s.run_pr_transverse_streaming(r,phase_binding=wrong)


def test_replay_failure_preserves_first(monkeypatch):
    original=s.run_pr_transverse_static_marching;calls=[]
    def run(*args,**kwargs):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('synthetic replay failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(s,'run_pr_transverse_static_marching',run)
    out=s.run_pr_transverse_streaming(request(),stop_after_accepted_cells=1)
    assert out.status=='replay_failed' and out.first_pass.completed_intervals==1
    assert out.replay_pass is None


def test_long_physical_grid_plane_retention():
    r=request();r=replace(r,grid=replace(r.grid,z_length_um=5000,dz_um=2))
    out=s.run_pr_transverse_streaming(r,stop_after_accepted_cells=1)
    assert out.converged and out.diagnostics['physical_Nz']==2500
    assert out.diagnostics['propagated_distance_um']==2
    for name,value in vars(out.first_pass).items():
        if isinstance(value,np.ndarray):assert value.size<2500


def test_failed_cell_rolls_back_without_phase_or_replay(tmp_path,monkeypatch):
    import lcprop.pr.transverse.marching_static as m
    r=request(modulation=.001);b=binding(tmp_path,r)
    baseline=s.run_pr_transverse_streaming(r,phase_binding=b,stop_after_accepted_cells=1)
    events=[];apply=s.TransversePhaseBinding.apply_to;solve=m.solve_pr_transverse_static_intensity
    evaluate=m._evaluate_local_state
    def invalid(*args,**kwargs):
        value=evaluate(*args,**kwargs)
        return replace(value,physical_state_valid=False) if events else value
    monkeypatch.setattr(m,'_evaluate_local_state',invalid)
    def track(self,field,index,**kwargs):
        events.append(index);return apply(self,field,index,**kwargs)
    def fail(*args,**kwargs):
        value=solve(*args,**kwargs)
        return replace(value,converged=False,status='injected_failure') if events else value
    monkeypatch.setattr(s.TransversePhaseBinding,'apply_to',track)
    monkeypatch.setattr(m,'solve_pr_transverse_static_intensity',fail)
    out=s.run_pr_transverse_streaming(r,phase_binding=b,stop_after_accepted_cells=2)
    assert out.status=='first_pass_failed' and events==[0]
    assert out.replay_pass is None
    assert np.array_equal(out.first_pass.A_final,baseline.first_pass.A_final)


def test_gpu_source_dispatch_without_gpu_execution(monkeypatch):
    from unittest.mock import Mock
    source=Mock(spec=s.GPUIndexedScatteringPhases)
    b=s.TransversePhaseBinding(source,None)
    field=np.ones((1,3,3),complex)
    b.apply_to(field,7,xp=np)
    source.apply_to.assert_called_once_with(field,7,xp=np)


def test_explicit_none_default_parity():
    r=request(modulation=.001)
    a=s.run_pr_transverse_static_marching(r)
    b=s.run_pr_transverse_static_marching(r,launch_peak_intensity_reference=None,phase_binding=None)
    for name,value in vars(a).items():
        if isinstance(value,np.ndarray):assert np.array_equal(value,getattr(b,name))
    assert a.diagnostics==b.diagnostics


def test_reference_mapping():
    r=request()
    ref=PRStreamingStaticRequest(grid=r.grid,beams=r.beams,material=r.material,backend=r.backend,initial_A=r.initial_A)
    out=s.transverse_request_from_reference(ref,options=r.solver)
    assert out.grid is ref.grid and out.initial_A is ref.initial_A
    assert out.material is ref.material
    with pytest.raises(ValueError,match='apodization'):
        s.transverse_request_from_reference(replace(ref,solver=replace(ref.solver,tukey_alpha=.1)),options=r.solver)


def test_even_null_residual_cannot_be_accepted():
    r=request(8)
    a=(1+.1*(-1.)**(np.arange(8)[:,None]+np.arange(8)[None,:]))[None].astype(complex)
    out=s.run_pr_transverse_streaming(replace(r,initial_A=a),stop_after_accepted_cells=1)
    assert out.status=='first_pass_failed'
    assert out.first_pass.completed_intervals==0 and out.replay_pass is None
    assert out.first_pass.failure_state is not None
