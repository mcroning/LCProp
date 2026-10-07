"""Independent units, quadrature and explicit physical-input contract checks."""
from dataclasses import replace
import numpy as np
import pytest

from lcprop.pr.illumination import (
    INTEGRAL_NORMALIZATION, LEGACY_NORMALIZATION, physical_reference,
    reference_from_metadata, resolve_material_illumination,
)
from lcprop.pr.specs import PRMaterialSpec, material_metadata, material_values


def physical(material=None, dark=.2, uniform=.1):
    return replace(material or PRMaterialSpec(), dark_intensity=0.,
        uniform_background_intensity=0., normalization_identity=INTEGRAL_NORMALIZATION,
        dark_irradiance_W_cm2=dark, uniform_irradiance_W_cm2=uniform)


def reference(a, scale=1., dark=.2, uniform=.1, groups=None, dx=1., dy=1.):
    return physical_reference(a, optical_scale_W_cm2=scale, dx_um=dx, dy_um=dy,
        dark_irradiance_W_cm2=dark, uniform_irradiance_W_cm2=uniform,
        coherence_groups=groups, xp=np)


@pytest.mark.parametrize('dark,uniform', [(None,0.),(0.,None),(-1.,0.),(float('nan'),0.),(True,0.)])
def test_explicit_inputs_reject_missing_or_invalid(dark, uniform):
    with pytest.raises(ValueError): physical(dark=dark,uniform=uniform).validate()


def test_explicit_zero_legacy_and_strict_metadata():
    m=physical(dark=0.,uniform=0.);m.validate()
    assert PRMaterialSpec(**material_values(material_metadata(m)))==m
    old=material_metadata(PRMaterialSpec())
    assert 'normalization_identity' not in old
    assert PRMaterialSpec(**material_values(old)).normalization_identity==LEGACY_NORMALIZATION
    assert PRMaterialSpec(**old).dark_intensity==.01
    for key in ('dark_irradiance_W_cm2','uniform_irradiance_W_cm2','normalization_identity'):
        broken=material_metadata(m);broken.pop(key)
        with pytest.raises(ValueError):material_values(broken)
    with pytest.raises(ValueError):replace(m,dark_intensity=.01).validate()


@pytest.mark.parametrize('power', [1.,.5,.25])
def test_physical_power_and_background_fractions(power):
    # Uniform field numerical integral = 1 µm²; restore mW then convert to W/cm².
    a=np.ones((1,8,8),complex)/8
    r=reference(a,scale=power*1e5,dark=100.,uniform=50.)
    assert r.optical_power_W==pytest.approx(power/1000,rel=2e-15)
    expected=(power/1000)+64e-8*150.
    assert r.reference_power_W==pytest.approx(expected,rel=2e-15)
    assert r.dark_fraction==pytest.approx(100*64e-8/expected)
    assert r.uniform_fraction==pytest.approx(50*64e-8/expected)
    np.testing.assert_allclose(r.driving_intensity(a,coherence_groups=None,xp=np),1.,rtol=2e-15)
    assert reference_from_metadata(r.metadata())==r


def test_coherence_integrates_actual_interference_and_relative_powers():
    a=np.ones((2,8,8),complex);a[1]*=2j
    assert reference(a,groups=('a','a')).optical_power_W==pytest.approx(5*64e-8)
    a[1]*=-1j
    assert reference(a,groups=('a','a')).optical_power_W==pytest.approx(9*64e-8)
    assert reference(a,groups=('a','b')).optical_power_W==pytest.approx(5*64e-8)
    a[1]*=-.5
    assert reference(a,groups=('a','a')).optical_power_W==0
    with pytest.raises(ValueError):reference(a,groups=('a','a'),dark=0,uniform=0)


def test_actual_single_beam_power_scaling_changes_structured_drive():
    from tests.test_pr_transverse_production import _request
    from lcprop.core.grid import make_grid
    from lcprop.optics.launch import build_launch, OpticalLaunchContext
    r=_request(steps=0);g=make_grid(r.grid,xp=np,real_dtype=np.float64)
    fractions=[];peaks=[]
    for power in (1.,.5,.25):
        beams=replace(r.beams,channels=(replace(r.beams.channels[0],power_mW=power),))
        launch=build_launch(beams,g,complex_dtype=np.complex128,
            context=OpticalLaunchContext(g,r.material.refractive_index,r.grid.z_length_um))
        ref=physical_reference(launch.A0,optical_scale_W_cm2=power*1e5,
            dx_um=g.dx_um,dy_um=g.dy_um,dark_irradiance_W_cm2=10.,uniform_irradiance_W_cm2=2.,
            coherence_groups=beams.coherence_groups,xp=np)
        I=ref.driving_intensity(launch.A0,coherence_groups=beams.coherence_groups,xp=np)
        fractions.append(ref.background_fraction);peaks.append(float(I.max()))
        assert float(I.mean())==pytest.approx(1.,abs=2e-15)
    assert fractions[0]<fractions[1]<fractions[2]
    assert peaks[0]>peaks[1]>peaks[2]>1.


@pytest.mark.parametrize('n',[64,128,256,512])
def test_equal_power_shapes_have_same_reference(n):
    x=(np.arange(n)+.5)/n-.5;X,Y=np.meshgrid(x,x,indexing='ij')
    rng=np.random.default_rng(7)
    shapes=[np.exp(-40*(X*X+Y*Y)),np.exp(-40*((X-.3/n)**2+Y*Y)),
        ((abs(X)<.2)&(abs(Y)<.2)).astype(float),
        1+.2*np.cos(2*np.pi*X)*np.cos(2*np.pi*Y),
        (1+.2*np.sin(8*np.pi*X))*np.exp(1j*rng.uniform(-np.pi,np.pi,(n,n)))]
    redistributed=np.ones((n,n));redistributed[0,0]=n*n/2;redistributed[1:]*=(n*n-redistributed[0].sum())/redistributed[1:].sum()
    shapes.append(np.sqrt(redistributed))
    refs=[]
    for a in shapes:
        a=(a/np.sqrt(np.mean(abs(a)**2)))[None].astype(complex)
        r=reference(a,dx=100/n,dy=100/n)
        refs.append(r.reference_power_W)
        assert np.mean(r.driving_intensity(a,coherence_groups=None,xp=np))==pytest.approx(1.,abs=5e-15)
    np.testing.assert_allclose(refs,1.3e-4,rtol=5e-15,atol=0)


def test_fixed_physical_gaussian_quadrature_converges():
    # Analytic finite-square integral, independent of launch/normalization code.
    import math
    expected=(math.sqrt(math.pi/80)*math.erf(math.sqrt(80)/2))**2
    errors=[];fraction_errors=[];source_errors=[]
    for n in (64,128,256,512):
        x=(np.arange(n)+.5)/n-.5
        a=np.exp(-40*(x[:,None]**2+x[None,:]**2))[None].astype(complex)
        r=reference(a,dx=1/n,dy=1/n)
        errors.append(abs(r.optical_power_W/1e-8-expected))
        fraction_errors.append(abs(r.dark_fraction-.2/(expected+.3)))
        source=r.driving_intensity(a,coherence_groups=None,xp=np)
        exact=(abs(a[0])**2+.3)/(expected+.3)
        source_errors.append(float(np.max(abs(source-exact))))
    assert all(b<a for a,b in zip(errors,errors[1:]))
    assert errors[-1]<2e-13
    assert fraction_errors[-1]<fraction_errors[0]
    assert source_errors[-1]<source_errors[0]


def test_same_bandlimited_random_field_and_top_hat_grid_quadrature():
    rng=np.random.default_rng(17)
    modes=[(1,2),(3,-2),(-4,1),(0,0)]
    coeff=rng.normal(size=4)+1j*rng.normal(size=4)
    power=float(np.sum(abs(coeff)**2))
    for n in (64,128,256,512):
        x=(np.arange(n)+.5)/n-.5
        a=sum(c*np.exp(2j*np.pi*(kx*x[:,None]+ky*x[None,:])) for c,(kx,ky) in zip(coeff,modes))
        ref=reference(a[None],dx=1/n,dy=1/n)
        assert ref.optical_power_W==pytest.approx(power*1e-8,rel=5e-15)
        assert ref.dark_fraction==pytest.approx(.2/(power+.3),rel=5e-15)
        # Same discontinuous physical square at all resolutions; quadrature
        # error is bounded by its unresolved edge strips, not a peak criterion.
        top=((abs(x[:,None])<.2)&(abs(x[None,:])<.2)).astype(complex)[None]
        ref=reference(top,dx=1/n,dy=1/n)
        assert abs(ref.optical_power_W/1e-8-.16)<=1.6/n+4/n**2


def test_float32_state_uses_same_physical_units_and_bounded_function_error():
    a=(1+.15*np.cos(np.arange(32)[:,None]))*np.ones((32,8))
    low=a[None].astype(np.complex64);high=low.astype(np.complex128)
    r32=reference(low,scale=1e5);r64=reference(high,scale=1e5)
    I32=r32.driving_intensity(low,coherence_groups=None,xp=np)
    I64=r64.driving_intensity(high,coherence_groups=None,xp=np)
    assert I32.dtype==np.float32 and I64.dtype==np.float64
    np.testing.assert_allclose(I32,I64,rtol=2e-6,atol=2e-7)


@pytest.mark.parametrize('dimension',[1,2])
@pytest.mark.parametrize('biased',[False,True])
def test_unified_physical_material_gates(dimension,biased):
    from tests.test_pr_unified_workflow import request
    from lcprop.pr.unified.specs import UNBIASED,FIXED_FIELD
    from lcprop.pr.unified.workflow import run_unified_static, UnifiedProductSelection
    r=request(n=1,dimension=dimension,closure=FIXED_FIELD if biased else UNBIASED)
    r=replace(r,material=physical(r.material),optical_scale_W_cm2=1.)
    out=run_unified_static(r,selection=UnifiedProductSelection(material_state=True,carrier=True))
    assert out.status=='completed',out.failure
    assert out.identities['source_normalization']['identity']==INTEGRAL_NORMALIZATION
    assert np.min(np.exp(out.material_state.q))>0


def test_both_td_workflows_physical_smoke():
    from tests.test_pr_semi_implicit_prototype import _coupled_optical_case
    from tests.test_pr_transverse_production import _request
    from lcprop.pr.workflow import run_pr_timedependent
    from lcprop.pr.transverse.workflow import run_pr_transverse_timedependent
    r,*_= _coupled_optical_case()
    r=replace(r,material=physical(r.material,dark=10.,uniform=2.),
        solver=replace(r.solver,integrator='semi_implicit_trapezoidal',Nt=2,dt_normalized=1e-4))
    out=run_pr_timedependent(r)
    assert out.completed_steps==2
    assert out.diagnostics['source_normalization']['identity']==INTEGRAL_NORMALIZATION
    r=_request(steps=2)
    r=replace(r,material=physical(r.material,dark=10.,uniform=2.))
    out=run_pr_transverse_timedependent(r)
    assert out.completed_steps==2
    assert out.resolved_profile['source_normalization']['identity']==INTEGRAL_NORMALIZATION


@pytest.mark.parametrize('dimension',[1,2])
def test_physical_unified_codec_and_scalable_overlap(dimension):
    from tests.test_pr_unified_workflow import request
    from lcprop.pr.unified import codec, products
    from lcprop.pr.unified.solver_specs import PRUnifiedSolverSpec, SCALABLE, ITERATIVE_POLICY
    r=request(n=1,dimension=dimension)
    r=replace(r,material=physical(r.material),optical_scale_W_cm2=1.)
    decoded=codec.decode_request(codec.encode_request(r)).materialize()
    assert decoded.material==r.material
    assert decoded.optical_scale_W_cm2==1.
    result=products.run_unified_products(r)
    restored=codec.decode_result(codec.encode_result(result))
    np.testing.assert_array_equal(restored.scientific.boundary_field,result.scientific.boundary_field)
    assert restored.scientific.identities['source_normalization']==result.scientific.identities['source_normalization']
    if dimension==2:
        scalable=replace(r,solver=PRUnifiedSolverSpec(SCALABLE,ITERATIVE_POLICY))
        other=products.run_unified_products(scalable)
        assert other.scientific.status=='completed',other.scientific.failure
        # Existing state64 direct/scalable endpoint contract.
        np.testing.assert_allclose(other.scientific.boundary_field,result.scientific.boundary_field,rtol=1e-8,atol=1e-8)


def test_gui_requires_explicit_physical_inputs_and_legacy_load():
    from PySide6.QtWidgets import QApplication
    from lcprop.pr.gui.material_panel import PRMaterialPanel
    app=QApplication.instance() or QApplication([])
    panel=PRMaterialPanel()
    try:
        assert panel.material().normalization_identity==INTEGRAL_NORMALIZATION
        panel.material().validate()
        assert panel.material().dark_irradiance_W_cm2 == .01
        panel.dark_irradiance.clear()
        with pytest.raises(ValueError):panel.material().validate()
        panel.dark_irradiance.setText('250');panel.uniform_irradiance.setText('0')
        m=panel.material();m.validate()
        assert m.dark_irradiance_W_cm2==.25 and m.uniform_irradiance_W_cm2==0
        panel.set_material(PRMaterialSpec())
        assert panel.material()==PRMaterialSpec()
        panel.set_material(m)
        assert panel.material()==m
    finally: panel.close()


def test_gui_physical_reduced_td_preflight_and_request_roundtrip():
    import lcprop.persistence
    from PySide6.QtWidgets import QApplication
    from lcprop.pr.gui.main_window import PRMainWindow
    from lcprop.pr.experiment_codec import encode_pr_timedependent_request, decode_pr_timedependent_request
    app=QApplication.instance() or QApplication([])
    window=PRMainWindow()
    try:
        window.material_panel.dark_irradiance.setText('not yet entered')
        original=window._capture_experiment_gui_state()
        window.material_panel.dark_irradiance.setText('100')
        window._restore_experiment_gui_state(original)
        assert window.material_panel.dark_irradiance.text()=='not yet entered'
        assert window.material_panel.uniform_irradiance.text()=='0'
        window.material_panel.dark_irradiance.setText('100')
        window.material_panel.uniform_irradiance.setText('0')
        window.grid_panel.Nx.setValue(8);window.grid_panel.Ny.setValue(8)
        request=window.build_request()
        assert request.material.normalization_identity==INTEGRAL_NORMALIZATION
        assert decode_pr_timedependent_request(encode_pr_timedependent_request(request))==request
    finally: window.close()


@pytest.mark.parametrize('dimension',[1,2])
def test_physical_gui_unified_dispatch_matches_headless(dimension):
    from PySide6.QtWidgets import QApplication
    from lcprop.pr.gui.main_window import PRMainWindow
    from tests.test_pr_unified_integration import fresh,apply,science_equal
    from lcprop.pr.unified.specs import UNBIASED
    from lcprop.pr.unified.integration import encode_fresh,decode_fresh,execute_unified
    app=QApplication.instance() or QApplication([])
    window=PRMainWindow()
    try:
        r=fresh(dimension,UNBIASED,'float64')
        r=replace(r,material=physical(r.material,dark=10.,uniform=0.))
        apply(window,r);gui=window.build_request()
        assert encode_fresh(gui)==encode_fresh(r)
        assert decode_fresh(encode_fresh(gui))==gui
        window._set_product_policy('analysis:far_field_intensity')
        actual=window._run_registered(gui).result.run
        expected=execute_unified(r,result_policy='analysis:far_field_intensity').run
        science_equal(actual.scientific,expected.scientific)
        for key in actual.arrays:np.testing.assert_array_equal(actual.arrays[key],expected.arrays[key])
        for key in actual.coordinates:np.testing.assert_array_equal(actual.coordinates[key],expected.coordinates[key])
    finally: window.close()


def test_cupy_physical_primitive_parity_if_available():
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount()<1:pytest.skip('CuPy/CUDA unavailable')
    except (ImportError,RuntimeError):pytest.skip('CuPy/CUDA unavailable')
    for dtype in (np.complex64,np.complex128):
        a=np.random.default_rng(10).normal(size=(2,16,8)).astype(dtype)
        a+=1j*a[::-1]
        host=reference(a,scale=10.)
        device=cp.asarray(a)
        native=physical_reference(device,optical_scale_W_cm2=10.,dx_um=1.,dy_um=1.,
            dark_irradiance_W_cm2=.2,uniform_irradiance_W_cm2=.1,coherence_groups=None,xp=cp)
        tol=2e-6 if dtype==np.complex64 else 8e-12
        np.testing.assert_allclose(cp.asnumpy(native.driving_intensity(device,coherence_groups=None,xp=cp)),
            host.driving_intensity(a,coherence_groups=None,xp=np),rtol=tol,atol=tol)


def test_new_reduced_coupled_temporal_order(monkeypatch, record_property):
    from tests import test_pr_semi_implicit_prototype as authority
    from lcprop.pr.workflow import _optical_pass
    from lcprop.optics.splitstep import scalar_angular_spectrum_kernel
    original=authority._coupled_optical_case
    def case():
        req,g,A,E,_=original()
        ref,coeff=resolve_material_illumination(physical(req.material,dark=10.,uniform=2.),A,
            grid=g,optical_scale_W_cm2=1e5,coherence_groups=req.beams.coherence_groups,xp=np)
        req=replace(req,material=coeff)
        kernel=scalar_angular_spectrum_kernel(g.fxy2_um,dz=g.dz_um,wavelength=.633,
            n_ref=req.material.refractive_index,xp=np)
        def optical(E):
            return _optical_pass(A,E,request=req,grid=g,kernel=kernel,peak_reference=ref,wavelength_um=.633)
        return req,g,A,E,optical
    monkeypatch.setattr(authority,'_coupled_optical_case',case)
    errors=authority._coupled_temporal_convergence_errors()
    orders=np.log2(errors[:-1]/errors[1:]);record_property('orders',orders.tolist())
    assert np.all((orders>1.8)&(orders<2.2))  # Existing coupled temporal contract.


def test_new_transverse_temporal_order(record_property):
    from tests.test_pr_transverse_imex import _smooth_fixed_source_case, _integrate_imex
    psi,optical,dx,dy=_smooth_fixed_source_case()
    fields=np.sqrt(optical)[None].astype(complex)
    ref=reference(fields,dark=.2,uniform=.1)
    source=ref.driving_intensity(fields,coherence_groups=None,xp=np)
    target=_integrate_imex(psi,source,dx=dx,dy=dy,final_time=.02,steps=512)
    errors=[]
    for steps in (16,32,64,128):
        got=_integrate_imex(psi,source,dx=dx,dy=dy,final_time=.02,steps=steps)
        errors.append(float(np.linalg.norm(got-target)/np.linalg.norm(target)))
    ratios=np.array(errors[:-1])/errors[1:];record_property('ratios',ratios.tolist())
    assert np.all((ratios>1.8)&(ratios<2.6))  # Existing IMEX contract.


def test_shared_source_all_three_workflows(monkeypatch):
    from tests.test_pr_transverse_production import _request
    from lcprop.pr import workflow as reduced
    from lcprop.pr.transverse import workflow as transverse
    from lcprop.pr.unified import workflow as unified
    from lcprop.pr.unified.integration import UnifiedFreshRequest, prepare_request
    from lcprop.pr.unified.specs import PRElectricalClosureSpec, UNBIASED
    from lcprop.pr.specs import PRRunRequest, PRSolverOptions
    original=reduced.pr_driving_intensity;seen=[]
    def capture(*a,**kw):
        result=original(*a,**kw);seen.append(result.copy());return result
    monkeypatch.setattr(reduced,'pr_driving_intensity',capture)
    monkeypatch.setattr(unified,'pr_driving_intensity',capture)
    t=_request(steps=0);t=replace(t,grid=replace(t.grid,z_length_um=t.grid.dz_um),
        material=physical(replace(t.material,gain_length_product=0.),dark=10.,uniform=2.))
    fresh=UnifiedFreshRequest(t.grid,t.beams,t.material,PRElectricalClosureSpec(UNBIASED,2,(0.,0.)),backend=t.backend)
    out=unified.run_unified_static(prepare_request(fresh));assert out.status=='completed',out.failure
    I=seen[0];seen.clear()
    reduced.run_pr_timedependent(PRRunRequest(grid=t.grid,beams=t.beams,material=t.material,
        backend=t.backend,solver=PRSolverOptions(Nt=0,integrator='semi_implicit_trapezoidal')))
    assert seen
    for value in seen:np.testing.assert_array_equal(value,I)

    seen.clear();transverse.run_pr_transverse_timedependent(t)
    assert seen
    for value in seen:np.testing.assert_array_equal(value,I)


def test_physical_dark_only_prepared_static_and_a7_provenance():
    from tests.test_pr_unified_workflow import request
    from lcprop.pr.unified import products, codec
    from lcprop.pr.unified.specs import PRElectricalClosureSpec
    from tests.test_pr_unified_codec import rewrite
    r=request(n=1)
    r=replace(r,initial_A=np.zeros_like(r.initial_A),optical_scale_W_cm2=1.,
        material=physical(replace(r.material,applied_field=.2)),
        closure=PRElectricalClosureSpec.a7(.2,0.))
    result=products.run_unified_products(r)
    assert result.scientific.status=='completed',result.scientific.failure
    assert result.scientific.identities['closure']['background_intensity']==1.
    assert result.scientific.identities['closure']['target']==(.2,)
    payload=codec.encode_result(result)
    codec.decode_result(payload)
    def corrupt(meta):
        meta['scientific']['identities']['closure'].update(background_intensity=.5,target=[.1])
        meta['state']['closure'].update(background_intensity=.5,target=[.1])
    with pytest.raises(ValueError,match='A7 current/reference'):
        codec.decode_result(rewrite(payload,corrupt))


@pytest.mark.parametrize('transverse',[False,True])
def test_td_physical_persistence_and_missing_reference_rejection(transverse):
    import lcprop.persistence
    from lcprop.pr.illumination import result_source_inverse
    if transverse:
        from tests.test_pr_transverse_production import _request
        from lcprop.pr.transverse.workflow import run_pr_transverse_timedependent as run
        from lcprop.pr.transverse.timedependent_transport_codec import (
            encode_pr_transverse_timedependent_transport_result as encode,
            decode_pr_transverse_timedependent_transport_result as decode)
        r=_request(steps=1)
    else:
        from tests.test_pr_semi_implicit_prototype import _coupled_optical_case
        from lcprop.pr.workflow import run_pr_timedependent as run
        from lcprop.pr.timedependent_transport_codec import (
            encode_pr_timedependent_transport_result as encode,
            decode_pr_timedependent_transport_result as decode)
        r,*_=_coupled_optical_case()
        r=replace(r,solver=replace(r.solver,integrator='semi_implicit_trapezoidal',Nt=1,dt_normalized=1e-4))
    r=replace(r,material=physical(r.material,dark=10.,uniform=2.))
    out=run(r);inverse=result_source_inverse(out)
    observed=run(r,progress_callback=lambda _: None)
    for key in ('A_final','source_intensity_stack','psi_final' if transverse else 'E_final'):
        np.testing.assert_array_equal(getattr(observed,key),getattr(out,key))
    if transverse:
        assert out.diagnostics['physical_state_valid']
        assert out.diagnostics['carrier_density_minimum']>0
    else:
        from lcprop.pr.evolution import periodic_derivatives_x
        dx=r.material.characteristic_wavenumber_per_um*r.grid.x_aperture_um/r.grid.Nx
        carrier=1+periodic_derivatives_x(out.E_final,dx_normalized=dx,xp=np)[0]
        assert np.all(np.isfinite(carrier)) and np.min(carrier)>0
    for policy in ('fast','full'):
        payload=encode(out,result_policy=policy)
        restored=decode(payload.payload.metadata,payload.payload.arrays)
        np.testing.assert_array_equal(restored.A_final,out.A_final)
        assert result_source_inverse(restored)==inverse
        from copy import deepcopy
        incomplete = deepcopy(payload.payload.metadata)
        incomplete['resolved_profile' if transverse else 'diagnostics'].pop('source_normalization')
        with pytest.raises(ValueError,match='requires illumination provenance'):
            decode(incomplete,payload.payload.arrays)
    if transverse:
        bad=dict(out.resolved_profile);bad.pop('source_normalization')
        broken=replace(out,resolved_profile=bad)
    else:
        bad=dict(out.diagnostics);bad.pop('source_normalization')
        broken=replace(out,diagnostics=bad)
    with pytest.raises(ValueError,match='requires illumination provenance'):
        result_source_inverse(broken)
