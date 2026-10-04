"""Bounded headless optical integration; independent explicit march reference."""
from dataclasses import replace
from types import SimpleNamespace
import subprocess
from pathlib import Path
import numpy as np
import pytest

from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.core.grid import make_grid
from lcprop.optics.splitstep import hop_linear_inplace, scalar_angular_spectrum_kernel
from lcprop.optics.farfield import direction_cosine_spectrum
from lcprop.pr.source import channel_peak_intensity_reference, pr_driving_intensity
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.scattering import PRCanonicalScatteringSpec, PR_CANONICAL_SCATTERING_V2, canonical_scattering_phase_increment
from lcprop.pr.optical_response import delta_n_from_E
from lcprop.pr.unified.specs import (
    PRUnifiedSpatialSpec, PRElectricalClosureSpec, PRMaterialPrecisionSpec,
    UNBIASED,FIXED_FIELD,PRESCRIBED_CURRENT,OPEN_TRANSVERSE,MIXED_PRECISION,
)
from lcprop.pr.unified.state import PRTransportIntensity
from lcprop.pr.unified.static import solve_static_material
from lcprop.pr.unified.projection import electric_field_optical_node
from lcprop.pr.unified import workflow as w


def request(n=2,precision='float64',dimension=1,scatter=False,closure=UNBIASED):
    real=np.dtype(precision);complex_type=np.complex64 if precision=='float32' else np.complex128
    A=np.broadcast_to(1+.03*np.exp(2j*np.pi*np.arange(16)[:,None]/16),(16,4))[None].astype(complex_type).copy()
    spatial=PRUnifiedSpatialSpec((16,),(16.,),batch_axes=('y',),batch_shape=(4,)) if dimension==1 else PRUnifiedSpatialSpec(
        (16,4),(16.,8.),active_axes=('x','y'))
    target=(0.,)*dimension if closure==UNBIASED else (.02,)+(0.,)*(dimension-1)
    return w.UnifiedStaticRequest(GridSpec(Nx=16,Ny=4,x_aperture_um=16.,y_aperture_um=8.,
        z_length_um=2.*n,dz_um=2.),spatial,PRElectricalClosureSpec(closure,dimension,target),A,
        material=PRMaterialSpec(gain_length_product=.1,characteristic_wavenumber_per_um_override=1.),
        precision=PRMaterialPrecisionSpec() if real.name=='float64' else PRMaterialPrecisionSpec(
            MIXED_PRECISION,'float32',output_dtype='float32'),
        scattering=PRCanonicalScatteringSpec(.02,.4,17,1.,PR_CANONICAL_SCATTERING_V2) if scatter else None)


def reference(r):
    dtype=np.dtype(r.precision.state_dtype);g=make_grid(r.grid,real_dtype=dtype)
    A=r.initial_A.copy();peak=channel_peak_intensity_reference(A,xp=np)
    boundaries=[A.copy()];sources=[]
    h=r.grid.dz_um
    kernel=scalar_angular_spectrum_kernel(g.fxy2_um,dz=h,wavelength=r.wavelength_um,
        n_ref=r.material.refractive_index,complex_dtype=A.dtype,xp=np)
    for k in range(round(r.grid.z_length_um/h)):
        hop_linear_inplace(A,kernel,xp=np)
        I=pr_driving_intensity(A,peak_intensity_reference=peak,background_intensity=r.material.background_intensity,xp=np)
        state,_=solve_static_material(PRTransportIntensity(I,r.spatial,r.precision,'numpy',peak,.01,0),closure=r.closure)
        # Independent face arithmetic, including per-column harmonic broadcast.
        b=state.b[...,0] if r.spatial.batch_shape else state.b[0]
        face=b-(np.roll(state.psi,-1,axis=0)-state.psi)/(r.spatial.normalized_lengths[0]/r.grid.Nx)
        E=(face+np.roll(face,1,axis=0))*dtype.type(.5)
        dn=delta_n_from_E(E,gain_length_product=r.material.gain_length_product,
            interaction_length_um=r.grid.z_length_um,wavelength_um=r.wavelength_um)
        A*=np.exp(1j*(2*np.pi/r.wavelength_um)*h*dn)
        if r.scattering:
            phi=canonical_scattering_phase_increment(r.scattering,z_start_um=k*h,dz_um=h,z_length_um=r.grid.z_length_um,
                Nx=g.Nx,Ny=g.Ny,x_aperture_um=r.grid.x_aperture_um,y_aperture_um=r.grid.y_aperture_um,
                real_dtype=dtype,xp=np)
            A*=np.exp(1j*phi)
        boundaries.append(A.copy());sources.append(I.copy())
    return boundaries,sources,state


@pytest.mark.parametrize('precision',['float32','float64'])
@pytest.mark.parametrize('n',[1,2,4])
@pytest.mark.parametrize('scatter',[False,True])
def test_exact_march_coordinates_products_retention(precision,n,scatter):
    r=request(n,precision,scatter=scatter);before=r.initial_A.tobytes()
    boundaries,sources,state=reference(r)
    selection=w.UnifiedProductSelection(True,True,True,True,('x',),('x',),True)
    out=w.run_unified_static(r,selection=selection)
    assert out.status=='completed',out.failure
    assert out.completed_cells==n and out.reached_z_um==2*n
    assert out.boundary_z_um==tuple(np.arange(n+1)*2.)
    assert out.material_z_um==tuple(np.arange(1,n+1)*2.)
    np.testing.assert_array_equal(out.boundary_field,boundaries[-1])
    np.testing.assert_array_equal(out.products['launch'],r.initial_A)
    np.testing.assert_array_equal(out.material_state.psi,state.psi)
    for A,(xz,yz) in zip(boundaries,out.products['boundary_cuts']):
        np.testing.assert_array_equal(xz,A[:,:,out.products['cut_y_index']])
        np.testing.assert_array_equal(yz,A[:,out.products['cut_x_index'],:])
    expected=direction_cosine_spectrum(boundaries[-1],dx_um=1.,dy_um=2.,wavelength_um=.633,refractive_index=2.4)
    np.testing.assert_array_equal(out.products['far_field'].intensity,expected.intensity)
    minimal=w.run_unified_static(r)
    np.testing.assert_array_equal(minimal.boundary_field,out.boundary_field)
    assert minimal.material_state is None and minimal.products['boundary_cuts']==()
    assert minimal.products['launch'] is None and r.initial_A.tobytes()==before
    assert out.identities['projection']=='adjacent_face_arithmetic_to_optical_node_v1'
    for k,row in enumerate(out.ledger):
        assert row['material_weight_um']==row['optical_distance_um']==2
        assert row['canonical_slabs']==((2*k,2*k+2) if scatter else None)


@pytest.mark.parametrize('precision',['float32','float64'])
@pytest.mark.parametrize('dimension',[1,2])
@pytest.mark.parametrize('closure',[UNBIASED,FIXED_FIELD,PRESCRIBED_CURRENT])
def test_bounded_closures_and_dimensionality(precision,dimension,closure,monkeypatch):
    import lcprop.pr.unified.static as core
    called=[];original=core.solve_column
    def column(*a,**kw): called.append(a[0].shape);return original(*a,**kw)
    monkeypatch.setattr(core,'solve_column',column)
    r=request(1,precision,dimension,closure=closure)
    out=w.run_unified_static(r,selection=w.UnifiedProductSelection(material_state=True,
        electric_faces=('x','y') if dimension==2 else ('x',),electric_nodes=('x',)))
    assert out.status=='completed',out.failure
    assert called==([(16,)]*4 if dimension==1 else [])
    assert out.material_state.b.shape==((4,1) if dimension==1 else (2,))


def test_operation_order_source_and_commissioned_driver(monkeypatch):
    from lcprop.pr.published_static_step import step_published_cell
    from lcprop.core.backend import BackendSpec
    r=request(1,scatter=True);events=[];saved={}
    for name,label in [('hop_linear_inplace','P'),('pr_driving_intensity','I'),('solve_static_material','solve'),
                       ('electric_field_optical_node','project'),('delta_n_from_E','dn'),
                       ('apply_response_screen_inplace','apply'),('canonical_scattering_phase_increment','S')]:
        orig=getattr(w,name)
        def wrapper(*a,_orig=orig,_label=label,**kw):
            events.append(_label);v=_orig(*a,**kw)
            if _label=='I': saved['I']=v.copy()
            if _label=='project': saved['E']=v.copy()
            return v
        monkeypatch.setattr(w,name,wrapper)
    out=w.run_unified_static(r)
    assert out.status=='completed',out.failure
    assert events==['P','I','solve','project','dn','apply','S','apply']
    # Same response injected into the unchanged commissioned cell primitive
    # isolates optical arithmetic from different material equations.
    def response(I):
        np.testing.assert_array_equal(I,saved['I'])
        return SimpleNamespace(E=saved['E'],residual=np.zeros_like(I))
    grid=make_grid(r.grid,real_dtype=np.float64)
    old=step_published_cell(r.initial_A,material_response=response,material_model='test_exact_response',grid=grid,
        cell_index=0,z_start_um=0,dz_um=2,interaction_length_um=2,wavelength_um=.633,material=r.material,
        peak_intensity_reference=channel_peak_intensity_reference(r.initial_A,xp=np),backend=BackendSpec('numpy','float64',False),
        scattering=r.scattering)
    np.testing.assert_array_equal(old.A_candidate,out.boundary_field)


@pytest.mark.parametrize('where',['optical','solve','phase','scattering','bookkeeping'])
def test_failure_after_one_cell_preserves_previous_state(where,monkeypatch):
    r=request(3,scatter=True)
    # Keep original L while stopping after the first accepted cell.
    token=CancellationToken();calls=0;original=w._prepare_acceptance
    def stop(*a):
        nonlocal calls
        calls+=1
        if calls==2: token.cancel()
        return original(*a)
    monkeypatch.setattr(w,'_prepare_acceptance',stop)
    previous=w.run_unified_static(r,selection=w.UnifiedProductSelection(material_state=True),cancellation_token=token)
    monkeypatch.setattr(w,'_prepare_acceptance',original)
    target={'optical':'hop_linear_inplace','solve':'solve_static_material','phase':'delta_n_from_E',
            'scattering':'canonical_scattering_phase_increment','bookkeeping':'_prepare_acceptance'}[where]
    original=getattr(w,target);calls=0
    def fail(*a,**kw):
        nonlocal calls
        calls+=1
        if calls==2: raise RuntimeError('injected '+where)
        return original(*a,**kw)
    monkeypatch.setattr(w,target,fail)
    out=w.run_unified_static(r,selection=w.UnifiedProductSelection(material_state=True,boundary_cuts=True))
    assert out.status=='failed' and out.completed_cells==1 and out.reached_z_um==2
    assert out.failure['cell_index']==1 and where in out.reason
    assert out.boundary_z_um==(0.,2.) and out.material_z_um==(2.,)
    assert len(out.products['boundary_cuts'])==2
    for name in ('q','psi','b'): np.testing.assert_array_equal(getattr(out.material_state,name),getattr(previous.material_state,name))
    np.testing.assert_array_equal(out.boundary_field,previous.boundary_field)


@pytest.mark.parametrize('when',['before','solve','phase','scattering','bookkeeping'])
def test_cancel_candidate_never_accepted(when,monkeypatch):
    r=request(2,scatter=True);token=CancellationToken()
    if when=='before': token.cancel()
    else:
        target={'solve':'solve_static_material','phase':'delta_n_from_E','scattering':'canonical_scattering_phase_increment',
                'bookkeeping':'_prepare_acceptance'}[when]
        original=getattr(w,target)
        def cancel(*a,**kw):
            result=original(*a,**kw);token.cancel();return result
        monkeypatch.setattr(w,target,cancel)
    out=w.run_unified_static(r,selection=w.UnifiedProductSelection(launch=True,boundary_cuts=True,material_state=True),cancellation_token=token)
    assert out.status=='cancelled' and out.completed_cells==0 and out.reached_z_um==0
    assert out.boundary_z_um==(0.,) and out.material_z_um==() and out.material_state is None
    np.testing.assert_array_equal(out.boundary_field,r.initial_A)
    assert len(out.products['boundary_cuts'])==1


def test_material_gate_failure_and_early_size_guard(monkeypatch):
    r=request(1);original=w.solve_static_material
    def bad(*a,**kw):
        state,diag=original(*a,**kw)
        return state,replace(diag,observations=tuple((k,100. if k.endswith('gauss_max') else v) for k,v in diag.observations))
    monkeypatch.setattr(w,'solve_static_material',bad)
    out=w.run_unified_static(r)
    assert out.status=='failed' and out.completed_cells==0 and out.failure['stage']=='material_equilibrium'
    r2=request(1,dimension=2)
    r2=replace(r2,grid=replace(r2.grid,Nx=128,Ny=128),spatial=PRUnifiedSpatialSpec((128,128),(16.,8.),active_axes=('x','y')))
    monkeypatch.setattr(w,'get_backend',lambda *a: pytest.fail('guard must precede backend/allocation'))
    with pytest.raises(ValueError,match='12,288-node bound'): w.run_unified_static(r2)


def test_uniform_constant_phase_and_fractional_domain():
    r=request(3,closure=FIXED_FIELD)
    r=replace(r,initial_A=np.ones_like(r.initial_A),grid=replace(r.grid,dz_um=.1,z_length_um=.3))
    out=w.run_unified_static(r,selection=w.UnifiedProductSelection(material_state=True))
    assert out.status=='completed',out.failure
    # Full scalar P_h retains longitudinal carrier phase (not paraxial P_h).
    expected=np.exp(2j*np.pi*r.material.refractive_index*.3/r.wavelength_um
                    -2j*r.material.gain_length_product*.02)
    np.testing.assert_allclose(out.boundary_field,expected,rtol=0,atol=2e-15)
    assert out.reached_z_um==.3 and out.boundary_z_um[-1]==.3
    assert all(v['material_weight_um']==.1 for v in out.ledger)


def test_invalid_launch_no_accepted_boundary_and_metadata_rejection():
    r=request(1);bad=r.initial_A.copy();bad[0,0,0]=np.nan
    out=w.run_unified_static(replace(r,initial_A=bad))
    assert out.status=='failed' and out.boundary_field is None and out.boundary_z_um==()
    for kwargs in ({'workflow_identity':'legacy'},{'projection_identity':'future'},
                   {'backend':'auto'},{'material':replace(r.material,applied_field=1.)}):
        with pytest.raises(ValueError): w.run_unified_static(replace(r,**kwargs))


def test_a7_and_open_transverse_closure_semantics():
    r=request(1)
    r=replace(r,closure=PRElectricalClosureSpec.a7(.04,.01),material=replace(r.material,applied_field=.04))
    out=w.run_unified_static(r,selection=w.UnifiedProductSelection(material_state=True))
    assert out.status=='completed',out.failure
    assert out.identities['closure']['target']==(.0004,)
    r2=request(1,dimension=2,closure=OPEN_TRANSVERSE)
    out2=w.run_unified_static(r2,selection=w.UnifiedProductSelection(material_state=True))
    assert out2.status=='completed',out2.failure
    assert abs(out2.ledger[0]['observations']['mean_current_y'])<=1e-9


def test_zero_gain_matches_only_optical_and_scattering():
    r=request(2,scatter=True);r=replace(r,material=replace(r.material,gain_length_product=0.))
    out=w.run_unified_static(r)
    g=make_grid(r.grid,real_dtype=np.float64);A=r.initial_A.copy()
    kernel=scalar_angular_spectrum_kernel(g.fxy2_um,dz=2.,wavelength=.633,n_ref=2.4,complex_dtype=A.dtype,xp=np)
    for k in range(2):
        hop_linear_inplace(A,kernel,xp=np)
        A*=np.exp(1j*canonical_scattering_phase_increment(r.scattering,z_start_um=2.*k,dz_um=2.,z_length_um=4.,
            Nx=16,Ny=4,x_aperture_um=16.,y_aperture_um=8.,real_dtype=np.float64,xp=np))
    assert out.status=='completed',out.failure
    np.testing.assert_array_equal(out.boundary_field,A)


def test_owned_launch_and_bounded_retained_arrays(monkeypatch):
    r=request(5);expected=w.run_unified_static(r)
    original=w.solve_static_material;calls=0
    def solve(*a,**kw):
        nonlocal calls
        calls+=1
        # Mutation after launch acceptance cannot change owned launch or peak.
        r.initial_A[:]=900
        return original(*a,**kw)
    monkeypatch.setattr(w,'solve_static_material',solve)
    out=w.run_unified_static(r,selection=w.UnifiedProductSelection(boundary_cuts=True))
    assert out.status=='completed' and calls==5
    np.testing.assert_array_equal(out.boundary_field,expected.boundary_field)
    def arrays(v):
        if isinstance(v,np.ndarray): yield v
        elif isinstance(v,dict):
            for a in v.values(): yield from arrays(a)
        elif isinstance(v,(tuple,list)):
            for a in v: yield from arrays(a)
    assert list(arrays(out.ledger))==[]
    for a in arrays(out.products):
        assert a.shape in ((1,16),(1,4),(16,),(4,))
    assert out.material_state is None


def test_no_longitudinal_scientific_volume_allocation(monkeypatch):
    r=request(3)
    for name in ('zeros','empty','ones','full'):
        original=getattr(np,name)
        def guard(shape,*a,_original=original,**kw):
            if isinstance(shape,(tuple,list)):
                assert tuple(shape) not in ((3,16,4),(4,16,4)), 'longitudinal scientific allocation'
            return _original(shape,*a,**kw)
        monkeypatch.setattr(np,name,guard)
    out=w.run_unified_static(r)
    assert out.status=='completed',out.failure


def test_product_failure_preserves_science_and_original_cell_failure(monkeypatch):
    r=request(1);baseline=w.run_unified_static(r)
    def bad(*a,**kw): raise RuntimeError('product failure')
    monkeypatch.setattr(w,'direction_cosine_spectrum',bad)
    out=w.run_unified_static(r,selection=w.UnifiedProductSelection(far_field=True))
    assert out.status=='failed' and out.completed_cells==1 and out.failure['stage']=='products'
    np.testing.assert_array_equal(out.boundary_field,baseline.boundary_field)
    monkeypatch.setattr(w,'solve_static_material',lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('material failure')))
    out=w.run_unified_static(r,selection=w.UnifiedProductSelection(far_field=True))
    assert out.status=='failed' and out.completed_cells==0
    assert out.failure['stage']=='material_equilibrium' and out.reason=='material failure'
    assert out.failure['product_failure']=='product failure'


def test_import_isolation_and_m1_m3_unchanged():
    root=Path(__file__).resolve().parents[1]
    core=['__init__.py','specs.py','state.py','_backend.py','_newton.py','operators.py','static.py']
    for name in core:
        p='src/lcprop/pr/unified/'+name
        assert (root/p).read_bytes()==subprocess.check_output(['git','show','4eca4a116babf901b85cfa3cddd663b4ca1ace88:'+p],cwd=root)
    # M6 already registered the legacy unified workflow. S4-M2 must not alter
    # any existing external dispatch, including Local-I and either TD path.
    paths=subprocess.check_output(['git','ls-files','src'],cwd=root,text=True).splitlines()
    protected={'src/lcprop/pr/published_static.py',
        'src/lcprop/pr/gui/evolution_panel.py','src/lcprop/pr/gui/request_adapter.py',
        'src/lcprop/pr/gui/main_window.py','src/lcprop/pr/runtime_estimator.py',
        'src/lcprop/pr/gui/run_cost.py'}
    for p in paths:
        if '/pr/unified/' not in p and p not in protected:
            assert (root/p).read_bytes()==subprocess.check_output(
                ['git','show','4eca4a116babf901b85cfa3cddd663b4ca1ace88:'+p],cwd=root)



@pytest.mark.parametrize('precision',['float32','float64'])
def test_native_workflow_when_available(precision,monkeypatch):
    cp=pytest.importorskip('cupy')
    try: cp.zeros(1)
    except Exception: pytest.skip('CuPy/CUDA unavailable')
    r=request(1,precision,scatter=True);cpu=w.run_unified_static(r)
    gpu_r=replace(r,initial_A=cp.asarray(r.initial_A),backend='cupy')
    original=cp.asnumpy
    def guard(a,*args,**kw):
        assert a.ndim<=1 and a.nbytes<=256,'prohibited scientific host transfer'
        return original(a,*args,**kw)
    monkeypatch.setattr(cp,'asnumpy',guard)
    gpu=w.run_unified_static(gpu_r,selection=w.UnifiedProductSelection(material_state=True))
    assert gpu.status=='completed',gpu.failure
    np.testing.assert_allclose(original(gpu.boundary_field),cpu.boundary_field,
        rtol=2e-5 if precision=='float32' else 3e-10,atol=2e-6 if precision=='float32' else 3e-11)
