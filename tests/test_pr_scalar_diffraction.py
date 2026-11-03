"""Independent scalar-dispersion oracles; no GPU commissioning required."""
from dataclasses import replace
import importlib
import math

import numpy as np
import pytest

from lcprop.optics.splitstep import scalar_angular_spectrum_kernel, linear_kernel, hop_linear
from lcprop.pr.geometry import scalar_kernel_slopes, crossing_beam_channels


@pytest.mark.parametrize('dtype,tol', [(np.complex64, 3e-7), (np.complex128, 3e-13)])
@pytest.mark.parametrize('distance', [-2.3, 0.0, 2.3])
@pytest.mark.parametrize('mode', [(0, 0), (2, 0), (0, -3), (2, -3)])
def test_fourier_modes_against_independent_longitudinal_wavevector(dtype, tol, distance, mode):
    nxy, aperture, wavelength, index = 32, 12.0, 0.8, 1.5
    fx = np.fft.fftfreq(nxy, d=aperture/nxy)
    f2 = fx[:, None]**2 + fx[None, :]**2
    mx, my = mode
    xy = np.arange(nxy) * aperture / nxy
    kx, ky = 2*np.pi*mx/aperture, 2*np.pi*my/aperture
    field = np.exp(1j*(kx*xy[:, None] + ky*xy[None, :]))[None].astype(dtype)
    kernel = scalar_angular_spectrum_kernel(f2, dz=distance, wavelength=wavelength,
                                            n_ref=index, complex_dtype=dtype, xp=np)
    kz = math.sqrt((2*math.pi*index/wavelength)**2-kx*kx-ky*ky)
    assert kernel.dtype == np.dtype(dtype)
    np.testing.assert_allclose(hop_linear(field, kernel), field*np.exp(1j*distance*kz), atol=tol, rtol=tol)


@pytest.mark.parametrize('dtype,tol', [(np.complex64, 3e-7), (np.complex128, 1e-13)])
def test_cutoff_grazing_inverse_and_input_precision(dtype, tol):
    # lambda=n=1 makes q=0 exactly representable. No tolerance widens support.
    f2 = np.array([[0., 0.25, 1., 1.25]], dtype=np.float64)
    for real_dtype in (np.float32, np.float64):
        forward = scalar_angular_spectrum_kernel(f2.astype(real_dtype), dz=1.3,
                                                 wavelength=1, n_ref=1, complex_dtype=dtype)
        backward = scalar_angular_spectrum_kernel(f2.astype(real_dtype), dz=-1.3,
                                                  wavelength=1, n_ref=1, complex_dtype=dtype)
        assert forward[0, 2] == backward[0, 2] == 1
        assert forward[0, 3] == backward[0, 3] == 0
        np.testing.assert_allclose(forward*backward, [[1, 1, 1, 0]], atol=tol, rtol=tol)
    near = np.array([[np.nextafter(1., 0.), 1., np.nextafter(1., 2.)]])
    h = scalar_angular_spectrum_kernel(near, dz=-4, wavelength=1, n_ref=1, complex_dtype=dtype)
    np.testing.assert_allclose(np.abs(h), [[1, 1, 0]], atol=tol)


@pytest.mark.parametrize('dtype,tol', [(np.complex64, 4e-7), (np.complex128, 1e-13)])
def test_retained_field_roundtrip(dtype, tol):
    f = np.fft.fftfreq(24, d=0.3)
    f2 = f[:, None]**2+f[None, :]**2
    spectrum = np.random.default_rng(9).normal(size=(24,24))+1j*np.random.default_rng(10).normal(size=(24,24))
    spectrum[f2 > 1] = 0
    a = np.fft.ifft2(spectrum)[None].astype(dtype)
    h = scalar_angular_spectrum_kernel(f2, dz=3.2, wavelength=1, n_ref=1, complex_dtype=dtype)
    back = scalar_angular_spectrum_kernel(f2, dz=-3.2, wavelength=1, n_ref=1, complex_dtype=dtype)
    np.testing.assert_allclose(hop_linear(hop_linear(a,h),back), a, atol=tol, rtol=tol)


def test_paraxial_limit_has_fourth_order_phase_error():
    k, distance = 2*np.pi, 0.7
    eps = np.array([[0.02, 0.04, 0.08]])
    scalar = scalar_angular_spectrum_kernel(eps**2, dz=distance, wavelength=1, n_ref=1)
    fresnel = linear_kernel(eps**2, dz=distance, wavelength=1, n_ref=1)
    measured = np.angle(scalar*np.exp(-1j*k*distance)/fresnel)
    leading = -distance*k*eps**4/8
    np.testing.assert_allclose(measured, leading, rtol=0.004, atol=1e-14)
    np.testing.assert_allclose(measured[0,1:]/measured[0,:-1], 16, rtol=0.003)


@pytest.mark.parametrize('tilt', [(0.,0.), (0.8,0.), (0.,-0.6), (0.8,-0.6)])
def test_gaussian_centroid_follows_scalar_packet_slope(tilt):
    nxy, aperture, waist, distance = 192, 120., 12., 8.
    x = (np.arange(nxy)-nxy/2)*aperture/nxy
    xx, yy = np.meshgrid(x,x,indexing='ij')
    kx, ky = tilt
    a = np.exp(-(xx**2+yy**2)/waist**2+1j*(kx*xx+ky*yy))[None]
    f = np.fft.fftfreq(nxy,d=aperture/nxy)
    h = scalar_angular_spectrum_kernel(f[:,None]**2+f[None,:]**2, dz=distance, wavelength=1, n_ref=1)
    intensity = np.abs(hop_linear(a,h)[0])**2
    center = np.array([(intensity*xx).sum(), (intensity*yy).sum()])/intensity.sum()
    slope = np.array(tilt)/math.sqrt((2*np.pi)**2-kx*kx-ky*ky)
    # Finite spectral width contributes O((1/(k*w))**2), unlike a delta packet.
    np.testing.assert_allclose(center, distance*slope, atol=0.001, rtol=0.001)
    np.testing.assert_allclose(scalar_kernel_slopes(kx_rad_per_um=kx,ky_rad_per_um=ky,
                               wavelength_um=1,refractive_index=1),slope,atol=1e-15)


@pytest.mark.parametrize('screened', [False, True])
def test_broad_curved_or_screened_field_matches_independent_spectral_sum(screened):
    nxy=20
    x=(np.arange(nxy)-nxy/2)*0.25
    xx,yy=np.meshgrid(x,x,indexing='ij')
    a=np.exp(-(xx**2+yy**2)/0.6**2+1j*(0.7*xx+0.4*yy+2*(xx**2+yy**2)))
    if screened:
        a *= np.where(xx>0, 0.35, 1.)
    f=np.fft.fftfreq(nxy,d=0.25)
    # Avoid exact grazing bins here; representable cutoff has its own test.
    k=2*np.pi/0.93
    kz2=k*k-(2*np.pi*f[:,None])**2-(2*np.pi*f[None,:])**2
    expected_spectrum=np.fft.fft2(a)*np.where(kz2>=0,np.exp(0.3j*np.sqrt(np.maximum(kz2,0))),0)
    h=scalar_angular_spectrum_kernel(f[:,None]**2+f[None,:]**2,dz=0.3,wavelength=0.93,n_ref=1)
    np.testing.assert_allclose(hop_linear(a[None],h)[0],np.fft.ifft2(expected_spectrum),atol=1e-14)


def test_crossing_geometry_uses_both_gradient_components():
    theta, phi, z = 0.4, 0.7, 10.
    channels = crossing_beam_channels(wavelength_um=1,refractive_index=1,
        interaction_length_um=2*z,polar_angles_rad=(theta,theta),azimuths_rad=(phi,phi+np.pi),
        waist_x_um=4,waist_y_um=4)
    assert channels[0].x0_um == pytest.approx(-z*np.tan(theta)*np.cos(phi))
    assert channels[0].y0_um == pytest.approx(-z*np.tan(theta)*np.sin(phi))
    for kx in (2*np.pi, 3*np.pi):
        with pytest.raises(ValueError, match='no finite scalar trajectory'):
            scalar_kernel_slopes(kx_rad_per_um=kx,ky_rad_per_um=0,wavelength_um=1,refractive_index=1)


def test_backend_dispatch_stays_on_supplied_array_namespace():
    class Namespace:
        calls=[]
        float64=np.float64
        def __getattr__(self,name):
            self.calls.append(name)
            assert name in ('asarray','sqrt','maximum','where','exp')
            return getattr(np,name)
    xp=Namespace()
    h=scalar_angular_spectrum_kernel(np.array([[0.,2.]],dtype=np.float32),dz=-1,wavelength=1,n_ref=1,xp=xp)
    assert h.dtype==np.complex64
    assert set(xp.calls)=={'asarray','sqrt','maximum','where','exp'}


@pytest.mark.parametrize('name', ['workflow','static_workflow','transverse.workflow',
                                  'transverse.static_workflow','transverse.marching_static',
                                  'static_streaming','coupling'])
def test_canonical_modules_bind_shared_scalar_primitive(name):
    module=importlib.import_module('lcprop.pr.'+name)
    assert module.scalar_angular_spectrum_kernel is scalar_angular_spectrum_kernel
    assert not hasattr(module,'linear_kernel')


@pytest.mark.parametrize('path', ['reduced_td','reduced_static','transverse_td',
                                  'transverse_static','marching','streaming'])
@pytest.mark.parametrize('substeps', [1, 3])
def test_real_workflows_and_replays_accumulate_scalar_phase(path, substeps):
    from lcprop.core.backend import BackendSpec
    from lcprop.core.beams import BeamChannel, BeamStack
    from lcprop.core.context import GridSpec
    from lcprop.pr.specs import PRRunRequest, PRSolverOptions, PRMaterialSpec
    from lcprop.pr.workflow import run_pr_timedependent
    from lcprop.pr.static_workflow import PRStaticRunRequest, PRStaticWorkflowOptions, run_pr_static
    from lcprop.pr.transverse.specs import PRTransverseRunRequest, PRTransverseSolverOptions
    from lcprop.pr.transverse.workflow import run_pr_transverse_timedependent
    from lcprop.pr.transverse.static_workflow import (
        PRTransverseStaticRunRequest, PRTransverseStaticWorkflowOptions, run_pr_transverse_static)
    from lcprop.pr.transverse.marching_static import (
        PRTransverseMarchingStaticRunRequest, PRTransverseMarchingStaticOptions,
        run_pr_transverse_static_marching)
    from lcprop.pr.static_streaming import PRStreamingStaticRequest, PRStreamingStaticOptions, run_pr_static_streaming

    grid=GridSpec(Nx=12,Ny=12,x_aperture_um=12,y_aperture_um=12,z_length_um=0.6,dz_um=0.2)
    x=np.arange(12)
    kx,ky=2*np.pi/12,4*np.pi/12
    initial=np.exp(1j*(kx*x[:,None]+ky*x[None,:]))[None]
    common=dict(grid=grid,beams=BeamStack(channels=(BeamChannel(wavelength_um=1, w1_um=3, w2_um=3),)),
        material=PRMaterialSpec(refractive_index=1,gain_length_product=0,applied_field=0,dark_intensity=0.2,
                                characteristic_wavenumber_per_um_override=0.2),
        backend=BackendSpec(backend='numpy',precision='float64',verbose=False),initial_A=initial)
    constructors={
        'reduced_td':(PRRunRequest,PRSolverOptions(Nt=2,optical_substeps=substeps),run_pr_timedependent),
        'reduced_static':(PRStaticRunRequest,PRStaticWorkflowOptions(optical_substeps=substeps),run_pr_static),
        'transverse_td':(PRTransverseRunRequest,PRTransverseSolverOptions(Nt=2,optical_substeps=substeps),run_pr_transverse_timedependent),
        'transverse_static':(PRTransverseStaticRunRequest,PRTransverseStaticWorkflowOptions(optical_substeps=substeps),run_pr_transverse_static),
        'marching':(PRTransverseMarchingStaticRunRequest,PRTransverseMarchingStaticOptions(optical_substeps=substeps),run_pr_transverse_static_marching),
        'streaming':(PRStreamingStaticRequest,PRStreamingStaticOptions(coupled=PRStaticWorkflowOptions(optical_substeps=substeps)),run_pr_static_streaming),
    }
    constructor,solver,run=constructors[path]
    result=run(constructor(**common,solver=solver))
    expected=initial*np.exp(0.6j*math.sqrt((2*np.pi)**2-kx*kx-ky*ky))
    np.testing.assert_allclose(result.A_final,expected,atol=3e-12,rtol=3e-12)
    if hasattr(result,'converged'):
        assert result.converged


def test_prepared_response_preserves_noncommuting_order(monkeypatch):
    import lcprop.optics.splitstep as module
    from lcprop.optics.boundaries import TransverseBoundarySpec
    events=[]
    a=np.ones((1,4,4),dtype=complex)
    h=scalar_angular_spectrum_kernel(np.zeros((4,4)),dz=0.1,wavelength=1,n_ref=1)
    monkeypatch.setattr(module,'apply_response_screen_inplace',lambda *args,**kw: events.append('half'))
    monkeypatch.setattr(module,'hop_linear_inplace',lambda *args,**kw: events.append('hop'))
    monkeypatch.setattr(module,'transverse_boundary_mask',lambda *args,**kw: ('sponge',kw['propagation_distance_um']))
    monkeypatch.setattr(module,'apply_transverse_boundary_inplace',lambda a,mask: events.append(mask))
    module.advance_prepared_response(a,kernel=h,half_step_response=np.ones((4,4)),Nsub=2,
        boundary=TransverseBoundarySpec(mode='sponge'),boundary_grid=object(),propagation_distance_um=0.2)
    assert events==['half','hop','half',('sponge',0.1),'half','hop','half',('sponge',0.1),None]


def test_streaming_reference_uses_audited_scalar_formula():
    from types import SimpleNamespace
    from lcprop.pr.static_streaming import _legacy_angular_spectrum_kernel
    f2=np.array([[0.,0.3,0.9,1.7]])
    grid=SimpleNamespace(fxy2_um=f2,dz_um=0.8)
    q=1-(0.7/1.2)**2*f2
    audited=np.where(q>=0,np.exp(2j*np.pi*1.2*grid.dz_um/0.7*np.sqrt(np.maximum(q,0))),0)
    np.testing.assert_allclose(_legacy_angular_spectrum_kernel(grid,wavelength_um=0.7,n_ref=1.2,xp=np),audited,atol=1e-14)


@pytest.mark.parametrize('precision,tol',[('float32',3e-6),('float64',1e-12)])
def test_reduced_workflow_honors_optical_precision(precision,tol):
    from lcprop.core.backend import BackendSpec
    from lcprop.core.beams import BeamChannel,BeamStack
    from lcprop.core.context import GridSpec
    from lcprop.pr.specs import PRRunRequest,PRSolverOptions,PRMaterialSpec
    from lcprop.pr.workflow import run_pr_timedependent
    dtype=np.complex64 if precision=='float32' else np.complex128
    a=np.ones((1,8,8),dtype=dtype)
    request=PRRunRequest(grid=GridSpec(Nx=8,Ny=8,x_aperture_um=16,y_aperture_um=16,z_length_um=1,dz_um=0.5),
        beams=BeamStack(channels=(BeamChannel(wavelength_um=0.9, w1_um=3, w2_um=3),)),
        material=PRMaterialSpec(refractive_index=1.2,gain_length_product=0,applied_field=0),
        solver=PRSolverOptions(Nt=1,optical_substeps=3),
        backend=BackendSpec(backend='numpy',precision=precision,verbose=False),initial_A=a)
    result=run_pr_timedependent(request)
    assert result.A_final.dtype==np.dtype(dtype)
    np.testing.assert_allclose(result.A_final,a*np.exp(2j*np.pi*1.2/0.9),atol=tol,rtol=tol)


@pytest.mark.parametrize('kwargs', [{'wavelength':0},{'n_ref':-1},{'dz':float('nan')},{'complex_dtype':np.float32}])
def test_invalid_kernel_metadata_rejected(kwargs):
    args=dict(dz=1,wavelength=1,n_ref=1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        scalar_angular_spectrum_kernel(np.zeros((2,2)),**args)
