"""Constant-gauge discretization: small synthetic/frozen-state CPU checks only."""
from dataclasses import replace
from pathlib import Path
import json,hashlib
import numpy as np
import pytest
from lcprop.pr.transverse.discretization import CONSTANT_GAUGE_V1 as NEW
from lcprop.pr.transverse.static import (PRTransverseStaticMaterialSolverOptions,solve_pr_transverse_static_intensity,
 static_equilibrium_residual,_symbols,_jacobian_action,_precondition,project_production_resolved_modes)
from lcprop.pr.transverse.transport import state_from_potential,spectral_wavevectors,spectral_derivatives
from tests.test_pr_transverse_marching_static import _request
from lcprop.pr.transverse.streaming import run_pr_transverse_streaming


def options(**kw):return PRTransverseStaticMaterialSolverOptions(potential_space=NEW,**kw)


def test_retained_wolfram_solution_and_independent_dense_jacobian():
    path = Path(__file__).parent / 'data/pr_constant_gauge_wolfram.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'a69cdbc93683534912ab6113a1a6accdbe4cc98f3a277066169533e972f57cca'
    w = json.loads(path.read_text())
    shape = tuple(w['shape'])
    intensity = np.array(w['intensity']).reshape(shape)
    reference = np.array(w['solution']).reshape(shape)
    kw = dict(dx_normalized=w['dx'], dy_normalized=w['dy'])
    result = solve_pr_transverse_static_intensity(intensity, options=options(), **kw)
    assert result.converged
    assert np.max(abs(result.psi - reference)) < 1e-8
    assert np.max(abs(result.equilibrium_residual)) < 1e-8

    # Independent periodic cotangent differentiation matrices from reference.wls,
    # rather than constructing the oracle with the production FFT derivative.
    n = 4
    assert shape == (n, n)
    delta = np.arange(n)[:, None] - np.arange(n)[None, :]
    d = np.zeros((n, n))
    nonzero = delta != 0
    d[nonzero] = .5 * (-1.) ** delta[nonzero] / np.tan(np.pi * delta[nonzero] / n)
    dx, dy = np.kron(d, np.eye(n)), np.kron(np.eye(n), d)
    laplacian = -dx @ dx - dy @ dy
    p = np.exp(-reference) / intensity
    p = (p / p.mean()).ravel()
    count = intensity.size
    gauge = np.eye(count) - np.ones((count, count)) / count
    dense = gauge @ (laplacian + np.diag(p) - np.outer(p, p) / count) @ gauge
    denominator, mask = _symbols(shape, h_y=1., xp=np, potential_space=NEW, **kw)
    actual = np.column_stack([
        _jacobian_action(v.reshape(shape), weight=p.reshape(shape),
                         denominator=denominator, null_mask=mask).ravel()
        for v in np.eye(count)
    ])
    assert np.max(abs(actual - dense)) < 1e-14

@pytest.mark.parametrize('shape',[(7,5),(8,5),(7,6),(8,6)])
def test_geometry_jvp_conservation(shape):
 nx,ny=shape;x=np.arange(nx)[:,None];y=np.arange(ny)[None,:]
 psi=.001*np.cos(2*np.pi*x/nx)+.0007*np.sin(2*np.pi*y/ny)
 if nx%2==0:psi=psi+.003*(-1.)**x
 if ny%2==0:psi=psi+.002*(-1.)**y
 psi=psi-psi.mean();kw=dict(dx_normalized=.4,dy_normalized=.6)
 st=state_from_potential(psi,**kw,potential_space=NEW)
 assert np.allclose(st.psi,psi,atol=1e-17)
 kx,ky=spectral_wavevectors(shape,**kw);dex,_=spectral_derivatives(st.E_x,kx=kx,ky=ky);_,dey=spectral_derivatives(st.E_y,kx=kx,ky=ky)
 assert np.max(abs(st.carrier_density-1-dex-dey))<1e-14
 eyx,_=spectral_derivatives(st.E_y,kx=kx,ky=ky);_,exy=spectral_derivatives(st.E_x,kx=kx,ky=ky)
 assert np.max(abs(eyx-exy))<1e-14 and abs(st.carrier_density.mean()-1)<1e-14
 I=np.exp(-psi)/st.carrier_density
 R=static_equilibrium_residual(psi,I,**kw,potential_space=NEW);assert np.max(abs(R))<1e-14
 den,mask=_symbols(shape,**kw,h_y=1,xp=np,potential_space=NEW);assert mask.sum()==1
 v=np.cos(np.arange(nx*ny).reshape(shape));v-=v.mean();w=st.carrier_density
 j=_jacobian_action(v,weight=w,denominator=den,null_mask=mask)
 eps=1e-6
 fd=(static_equilibrium_residual(psi+eps*v,I,**kw,potential_space=NEW)-static_equilibrium_residual(psi-eps*v,I,**kw,potential_space=NEW))/(2*eps)
 assert np.max(abs(fd-j))<2e-9
 assert np.vdot(v,j)>0
 if nx%2==0:
  c=np.broadcast_to((-1.)**x,shape).copy()
  assert np.allclose(_precondition(c,denominator=den,null_mask=mask),c)


def test_retained_cell1_full_convergence():
 p=Path(__file__).parent/'data/pr_constant_gauge_cell1.npz'
 m=json.loads(p.with_suffix('.json').read_text());assert hashlib.sha256(p.read_bytes()).hexdigest()==m['fixture_sha256']
 with np.load(p) as a:
  I=a['intensity'];psi=a['psi'];kw=dict(dx_normalized=float(a['dx']),dy_normalized=float(a['dy']),initial_psi=psi)
  old=solve_pr_transverse_static_intensity(I,**kw);new=solve_pr_transverse_static_intensity(I,**kw,options=options())
 assert old.converged and np.sqrt(np.mean(old.equilibrium_residual**2))>1e-8
 assert new.converged and np.sqrt(np.mean(new.equilibrium_residual**2))<=1e-9
 assert np.max(abs(new.equilibrium_residual))<=1e-8
 assert abs(new.psi.mean())<1e-14
 assert np.max(abs(new.psi-project_production_resolved_modes(new.psi,dx_normalized=kw['dx_normalized'],dy_normalized=kw['dy_normalized'])))>1e-8

@pytest.mark.parametrize('n',[7,8])
def test_one_dimensional_reduction(n):
 x=np.arange(n)[:,None];psi=.002*np.cos(2*np.pi*x/n)
 if n%2==0:psi+=.001*(-1.)**x
 psi=np.broadcast_to(psi,(n,6)).copy();kw=dict(dx_normalized=.4,dy_normalized=.5)
 st=state_from_potential(psi,**kw,potential_space=NEW);I=np.exp(-psi)/st.carrier_density
 out=solve_pr_transverse_static_intensity(I,**kw,options=options())
 assert out.converged and np.max(abs(out.psi-psi))<1e-8
 state=state_from_potential(out.psi,**kw,potential_space=NEW)
 assert np.max(abs(state.E_y))<1e-14


def test_odd_default_parity():
 x=np.arange(7)[:,None];I=np.broadcast_to(1+.01*np.cos(2*np.pi*x/7),(7,5)).copy()
 kw=dict(dx_normalized=.4,dy_normalized=.6)
 old=solve_pr_transverse_static_intensity(I,**kw);new=solve_pr_transverse_static_intensity(I,**kw,options=options())
 assert old.converged and new.converged
 assert np.allclose(old.psi,new.psi,rtol=0,atol=1e-14)
 assert [r.pcg_iterations for r in old.iteration_records]==[r.pcg_iterations for r in new.iteration_records]


def test_checkerboard_marching_replay_and_policy_identity():
 n=8;a=np.exp(.01*(-1.)**np.arange(n)[:,None])*np.ones((1,n));r=_request(initial_A=a[None].astype(complex));r=replace(r,grid=replace(r.grid,Nx=n,Ny=n))
 old=run_pr_transverse_streaming(r,stop_after_accepted_cells=2)
 assert not old.converged
 r=replace(r,solver=replace(r.solver,material_solver=options()))
 new=run_pr_transverse_streaming(r,stop_after_accepted_cells=2)
 assert new.converged and all(new.diagnostics['replay_checks'].values())
 assert new.first_pass.resolved_profile['solver']['material_solver']['potential_space']==NEW
 for d in new.first_pass.diagnostics['planes']:
  assert d['full_residual']['rms']<1e-8 and d['null_residual']['rms']<1e-8
  assert d['carrier_minimum']>0 and abs(d['carrier_mean']-1)<1e-12
  assert d['curl_max']<1e-11 and d['gauss_max']<1e-11
 assert abs(new.first_pass.power_final/new.first_pass.power_initial-1)<1e-10


def test_invalid_and_failure_handling():
 with pytest.raises(ValueError):PRTransverseStaticMaterialSolverOptions(potential_space='bad').validate()
 x=np.arange(8)[:,None];I=np.exp(.1*(-1.)**x)*np.ones((1,6))
 out=solve_pr_transverse_static_intensity(I,dx_normalized=.4,dy_normalized=.5,options=options(max_newton_iterations=1))
 assert not out.converged and np.max(abs(out.equilibrium_residual))>1e-8


def test_new_mode_scattering_mapping_and_rollback(tmp_path,monkeypatch):
 from tests.test_pr_transverse_streaming import request,binding
 from lcprop.pr.frozen_scattering import FrozenScatteringPhases
 from lcprop.pr.transverse import marching_static as m
 r=request(n=8,modulation=.001);r=replace(r,solver=replace(r.solver,material_solver=options()))
 b=binding(tmp_path,r);events=[];original=FrozenScatteringPhases.phase
 def phase(self,index):events.append(index);return original(self,index)
 monkeypatch.setattr(FrozenScatteringPhases,'phase',phase)
 out=run_pr_transverse_streaming(r,phase_binding=b,stop_after_accepted_cells=2)
 assert out.converged and events==[0,1,0,1]
 baseline=run_pr_transverse_streaming(r,stop_after_accepted_cells=1)
 evaluate=m._evaluate_local_state;apply=m._apply_canonical_scattering_after_slice;seen=[0]
 def failed(*a,**kw):
  v=evaluate(*a,**kw);return replace(v,physical_state_valid=False) if seen[0] else v
 def accepted(*a,**kw):seen[0]+=1;return apply(*a,**kw)
 monkeypatch.setattr(m,'_evaluate_local_state',failed);monkeypatch.setattr(m,'_apply_canonical_scattering_after_slice',accepted)
 failure=run_pr_transverse_streaming(r,stop_after_accepted_cells=2)
 assert failure.status=='first_pass_failed' and failure.first_pass.completed_intervals==1
 assert np.array_equal(failure.first_pass.A_final,baseline.first_pass.A_final)
 assert np.array_equal(failure.first_pass.restart_state.previous_psi,baseline.first_pass.restart_state.previous_psi)


def test_restart_mode_identity_and_unsupported_volume():
 from lcprop.pr.transverse.marching_static import run_pr_transverse_static_marching
 from lcprop.pr.transverse.static_workflow import run_pr_transverse_static
 from tests.test_pr_transverse_marching_static import _global_request
 r=_request(initial_A=np.ones((1,16,16),complex))
 old=run_pr_transverse_static_marching(r,stop_after_accepted_intervals=1)
 corrected=replace(r,solver=replace(r.solver,material_solver=options()))
 with pytest.raises(ValueError,match='restart'):
  run_pr_transverse_static_marching(corrected,restart_state=old.restart_state,stop_after_accepted_intervals=1)
 with pytest.raises(ValueError,match='causal'):
  run_pr_transverse_static(_global_request(corrected))


def test_td_corrector_rejects_unreviewed_potential_space():
 from lcprop.pr.transverse.static import solve_pr_transverse_discrete_static_intensity
 with pytest.raises(ValueError,match="TD-discrete"):
  solve_pr_transverse_discrete_static_intensity(np.ones((8,8)),dx_normalized=1,dy_normalized=1,options=options())
