"""Published optical map, unchanged temporal stages, and persisted identity."""
from dataclasses import replace
import inspect
import numpy as np
import pytest
from tests.test_pr_semi_implicit_prototype import _coupled_optical_case
from lcprop.pr import workflow as w
from lcprop.pr import evolution as ev
from lcprop.pr.specs import PR_TD_PUBLISHED_COUPLING, PR_TD_LEGACY_COUPLING
from lcprop.pr.source import channel_peak_intensity_reference, pr_driving_intensity
from lcprop.pr.scattering import PRCanonicalScatteringSpec, PR_CANONICAL_SCATTERING_V2
from lcprop.optics.splitstep import scalar_angular_spectrum_kernel
import lcprop.persistence  # Initialize the public registry before material codecs.
from lcprop.pr.experiment_codec import encode_pr_timedependent_request, decode_pr_timedependent_request


def case():
    req,g,A,E,_ = _coupled_optical_case()
    req=replace(req, initial_A=A, initial_E=E, solver=replace(req.solver,integrator='semi_implicit_trapezoidal'))
    kw=dict(request=req,grid=g,kernel=scalar_angular_spectrum_kernel(g.fxy2_um,dz=g.dz_um,wavelength=.633,n_ref=req.material.refractive_index,xp=np),peak_reference=channel_peak_intensity_reference(A,xp=np),wavelength_um=.633)
    return req,g,A,E,kw


@pytest.mark.parametrize('nz',[1,2,3])
def test_independent_arriving_source_and_full_phase_oracle(nz):
    req,g,A,E,kw=case()
    # Same physical grid; limit the bounded march, not its phase coefficient.
    object.__setattr__(g,'Nz',nz)
    E=E[:nz].copy();original=E.copy();incoming=A.copy()
    actual,I=w._optical_pass(A,E,**kw)
    ref=A.copy();sources=[]
    for k in range(nz):
        ref=np.fft.ifft2(np.fft.fft2(ref,axes=(-2,-1))*kw['kernel'],axes=(-2,-1))
        sources.append(pr_driving_intensity(ref,peak_intensity_reference=kw['peak_reference'],background_intensity=req.material.background_intensity,coherence_groups=req.beams.coherence_groups,xp=np))
        # Published dn = -2*gammaL*E/(L*k_vac); full M coefficient = -2*gammaL*h/L.
        ref *= np.exp(-2j*req.material.gain_length_product*g.dz_um/req.grid.z_length_um*E[k])[None]
    np.testing.assert_allclose(actual,ref,rtol=1e-14,atol=1e-15)
    np.testing.assert_allclose(I,np.array(sources),rtol=1e-14,atol=1e-15)
    np.testing.assert_array_equal(E,original);np.testing.assert_array_equal(A,incoming)


def test_order_and_no_longitudinal_optical_allocation(monkeypatch):
    req,g,A,E,kw=case();events=[];allocations=[]
    for name,label in [('hop_linear_inplace','P'),('pr_driving_intensity','I'),('delta_n_from_E','M'),('apply_response_screen_inplace','phase'),('_apply_canonical_scattering_after_slice','S')]:
        original=getattr(w,name)
        def spy(*a,_fn=original,_label=label,**k):
            events.append(_label);return _fn(*a,**k)
        monkeypatch.setattr(w,name,spy)
    original=np.empty
    def empty(shape,*a,**k):
        allocations.append(tuple(shape) if hasattr(shape,'__len__') else shape)
        return original(shape,*a,**k)
    monkeypatch.setattr(np,'empty',empty)
    w._optical_pass(A,E,**kw)
    assert events==['P','I','M','phase','S']*g.Nz
    assert allocations.count(E.shape)==1  # required source state, no optical volume
    assert 'midpoint' not in inspect.getsource(w.advance_pr_published_frozen_slice).split('"""')[-1]


def test_two_temporal_stages_and_unchanged_corrector(monkeypatch):
    req,g,A,E,kw=case();states=[];sources=[];phases=[]
    req=replace(req,scattering=PRCanonicalScatteringSpec(epsilon=.02,transverse_correlation_um=1.,realization_seed=17,canonical_dz_um=1.,algorithm_version=PR_CANONICAL_SCATTERING_V2))
    kw['request']=req
    original=w._canonical_scattering_phase_for_slice
    def phase(*a,**k):
        p=original(*a,**k);phases.append(p.copy());return p
    monkeypatch.setattr(w,'_canonical_scattering_phase_for_slice',phase)
    def source(state):
        states.append(state.copy());I=w._optical_pass(A,state,**kw)[1];sources.append(I.copy());return I
    dt=req.solver.dt_normalized;dx=req.material.characteristic_wavenumber_per_um*g.dx_um
    actual=ev.semi_implicit_trapezoidal_step(E,source,dt_normalized=dt,applied_field=req.material.applied_field,background_intensity=req.material.background_intensity,dx_normalized=dx,xp=np)
    args=dict(applied_field=req.material.applied_field,background_intensity=req.material.background_intensity,dx_normalized=dx,xp=np)
    implicit,explicit=ev.diffusion_implicit_split(E,sources[0],**args)
    predicted=ev.solve_periodic_variable_diffusion(E+dt*explicit,sources[0],alpha=dt,dx_normalized=dx,xp=np)
    _,ep=ev.diffusion_implicit_split(predicted,sources[1],**args)
    expected=ev.solve_periodic_variable_diffusion(E+.5*dt*(implicit+explicit+ep),sources[1],alpha=.5*dt,dx_normalized=dx,xp=np)
    assert len(states)==2
    np.testing.assert_array_equal(states[0],E);np.testing.assert_array_equal(states[1],predicted)
    np.testing.assert_array_equal(actual,expected)
    for p,q in zip(phases[:g.Nz],phases[g.Nz:]):np.testing.assert_array_equal(p,q)


@pytest.mark.parametrize("steps", [1, 2])
def test_callback_is_observational_and_counts_marches(monkeypatch, steps):
    req,g,A,E,kw=case();req=replace(req,solver=replace(req.solver,Nt=steps))
    calls=[];original=w._optical_pass
    def spy(*a,**k):
        calls.append(k.get('cancellation_stage','final_products'));return original(*a,**k)
    monkeypatch.setattr(w,'_optical_pass',spy)
    plain=w.run_pr_timedependent(req);plain_calls=calls.copy();calls.clear()
    observed=w.run_pr_timedependent(req,progress_callback=lambda _:None)
    np.testing.assert_array_equal(plain.E_final,observed.E_final)
    np.testing.assert_array_equal(plain.A_final,observed.A_final)
    assert plain_calls.count('material_source_optical_z_march')==2*steps
    assert calls.count('material_source_optical_z_march')==2*steps
    assert calls.count('progress_optical_z_march')==steps
    assert plain.diagnostics['source_z_um']==[4.,8.,12.]


def test_old_and_new_experiment_identities_and_gui_rejection():
    req,*_=case();req=replace(req,initial_A=None,initial_E=None)
    payload=encode_pr_timedependent_request(req)
    assert payload['optical_coupling']==PR_TD_PUBLISHED_COUPLING
    assert decode_pr_timedependent_request(payload).optical_coupling==PR_TD_PUBLISHED_COUPLING
    payload.pop('optical_coupling')
    legacy=decode_pr_timedependent_request(payload)
    assert legacy.optical_coupling==PR_TD_LEGACY_COUPLING
    from lcprop.pr.gui.request_adapter import validate_pr_gui_request_representable
    with pytest.raises(ValueError,match='midpoint TD'):validate_pr_gui_request_representable(legacy)


def test_transport_and_checkpoint_identity():
    from lcprop.pr.timedependent_transport_codec import _encode_request_metadata,_decode_request_metadata
    from lcprop.pr.persistence import _request_to_dict,_request_from_dict,PR_CHECKPOINT_SCHEMA_VERSION
    req,*_=case();arrays={}
    data=_encode_request_metadata(req,arrays,prefix='test')
    assert _decode_request_metadata(data,arrays).optical_coupling==PR_TD_PUBLISHED_COUPLING
    data.pop('optical_coupling')
    assert _decode_request_metadata(data,arrays).optical_coupling==PR_TD_LEGACY_COUPLING
    data=_request_to_dict(req)
    assert _request_from_dict(data,schema_version=PR_CHECKPOINT_SCHEMA_VERSION).optical_coupling==PR_TD_PUBLISHED_COUPLING
    data.pop('optical_coupling')
    assert _request_from_dict(data,schema_version=PR_CHECKPOINT_SCHEMA_VERSION).optical_coupling==PR_TD_LEGACY_COUPLING


def test_predictor_source_failure_does_not_mutate_accepted_state():
    req,g,A,E,kw=case();before=E.copy();count=0
    def source(state):
        nonlocal count
        count+=1
        if count==2:raise RuntimeError('predictor source failure')
        return w._optical_pass(A,state,**kw)[1]
    with pytest.raises(RuntimeError,match='predictor source'):
        ev.semi_implicit_trapezoidal_step(E,source,dt_normalized=.001,applied_field=req.material.applied_field,background_intensity=req.material.background_intensity,dx_normalized=req.material.characteristic_wavenumber_per_um*g.dx_um,xp=np)
    np.testing.assert_array_equal(E,before)


@pytest.mark.parametrize('precision',['float32','float64'])
@pytest.mark.parametrize('scatter',[False,True])
def test_exact_commissioned_static_primitive_with_frozen_callback(precision,scatter):
    from types import SimpleNamespace
    from lcprop.core.backend import get_backend
    from lcprop.core.grid import make_grid
    from lcprop.pr.published_static_step import step_published_cell
    req,_,A,E,_=case()
    req=replace(req,backend=replace(req.backend,precision=precision),scattering=(PRCanonicalScatteringSpec(epsilon=.02,transverse_correlation_um=1.,realization_seed=17,canonical_dz_um=1.,algorithm_version=PR_CANONICAL_SCATTERING_V2) if scatter else None))
    b=get_backend(req.backend);g=make_grid(req.grid,xp=np,real_dtype=b.real_dtype)
    A=A.astype(b.complex_dtype);E=E.astype(b.real_dtype);peak=channel_peak_intensity_reference(A,xp=np)
    kernel=scalar_angular_spectrum_kernel(g.fxy2_um,dz=g.dz_um,wavelength=.633,n_ref=req.material.refractive_index,complex_dtype=b.complex_dtype,xp=np)
    actual,I=w._optical_pass(A,E,request=req,grid=g,kernel=kernel,peak_reference=peak,wavelength_um=.633)
    ref=A.copy();sources=[]
    for k in range(g.Nz):
        cell=step_published_cell(ref,material_response=lambda _,k=k:SimpleNamespace(E=E[k],residual=np.zeros_like(E[k])),material_model='frozen_td_stage',grid=g,cell_index=k,z_start_um=k*g.dz_um,dz_um=g.dz_um,interaction_length_um=req.grid.z_length_um,wavelength_um=.633,material=req.material,peak_intensity_reference=peak,backend=req.backend,coherence_groups=req.beams.coherence_groups,scattering=req.scattering)
        ref=cell.A_candidate;sources.append(cell.source_intensity)
    np.testing.assert_array_equal(actual,ref)
    np.testing.assert_array_equal(I,np.array(sources))


def test_no_continuation_across_coupling_identities():
    from lcprop.pr.checkpoint import validate_pr_continuation
    req,*_=case();result=w.run_pr_timedependent(req)
    legacy=replace(result.checkpoint.request,optical_coupling=PR_TD_LEGACY_COUPLING)
    with pytest.raises(ValueError,match='optical_coupling'):
        validate_pr_continuation(legacy,result.checkpoint)


def test_fresh_gui_and_inspect_identify_published_second_order():
    from PySide6.QtWidgets import QApplication
    app=QApplication.instance() or QApplication([])
    from lcprop.pr.gui.main_window import PRMainWindow
    window=PRMainWindow()
    try:
        window.grid_panel.Nx.setValue(8);window.grid_panel.Ny.setValue(8)
        window.grid_panel.dz_um.setValue(10.)
        window.grid_panel.z_length_um.setValue(20.)
        request=window.build_request()
        scientific=request
        assert scientific.optical_coupling==PR_TD_PUBLISHED_COUPLING
        assert scientific.solver.integrator=='semi_implicit_trapezoidal'
        text=window.describe_request(request)
        assert PR_TD_PUBLISHED_COUPLING in text and 'semi_implicit_trapezoidal' in text
    finally:
        window.close()
