"""Independent optical-first oracles; bounded grids only."""
from dataclasses import replace
import inspect
import subprocess
import numpy as np
import pytest

from tests.test_pr_local_plane_workflow import request as symmetric_request
from lcprop.core.backend import get_backend
from lcprop.core.grid import make_grid
from lcprop.pr.published_static import (
    PublishedStaticRequest, run_published_static, solve_arriving_material,
    PR_PUBLISHED_STATIC_WORKFLOW, PR_PUBLISHED_STATIC_ARITHMETIC,
)
from lcprop.pr.published_static_products import run_published_products
from lcprop.pr.local_plane_products import LocalPlaneProductSelection
from lcprop.pr.published_static_codec import encode_published_products, decode_published_products
from lcprop.pr.transverse.specs import PRTransverseMaterialResponseSpec
from lcprop.pr.source import channel_peak_intensity_reference, pr_driving_intensity
from lcprop.optics.splitstep import hop_linear_inplace, scalar_angular_spectrum_kernel
from lcprop.pr.optical_response import delta_n_from_E
from lcprop.pr.scattering import canonical_scattering_phase_increment
from lcprop.pr.static import PRStaticSolverOptions


def request(n=2, model='field_linear_local_intensity', **kwargs):
    old = symmetric_request(n=n, **kwargs)
    return PublishedStaticRequest(**vars(old), material_solver=PRStaticSolverOptions()) if model == 'field_linear_local_intensity' else replace(
        PublishedStaticRequest(**vars(old)), material_response=PRTransverseMaterialResponseSpec(model=model))


def reference(r):
    b = get_backend(r.backend)
    g = make_grid(r.grid, xp=np, real_dtype=b.real_dtype)
    A = r.initial_A.copy()
    peak = channel_peak_intensity_reference(A, xp=np)
    kernel = scalar_angular_spectrum_kernel(g.fxy2_um, dz=r.grid.dz_um, wavelength=.633,
        n_ref=r.material.refractive_index, complex_dtype=b.complex_dtype, xp=np)
    boundaries, sources, materials = [A.copy()], [], []
    for k in range(round(r.grid.z_length_um/r.grid.dz_um)):
        hop_linear_inplace(A, kernel, xp=np)
        I = pr_driving_intensity(A, peak_intensity_reference=peak,
            background_intensity=r.material.background_intensity, coherence_groups=r.beams.coherence_groups, xp=np)
        E = solve_arriving_material(I, request=r).E
        phase = np.exp(1j*(2*np.pi/.633)*r.grid.dz_um*delta_n_from_E(E,
            gain_length_product=r.material.gain_length_product,
            interaction_length_um=r.grid.z_length_um, wavelength_um=.633))
        A *= phase
        if r.scattering is not None:
            phi = canonical_scattering_phase_increment(r.scattering,
                z_start_um=k*r.grid.dz_um, dz_um=r.grid.dz_um, z_length_um=r.grid.z_length_um,
                Nx=g.Nx, Ny=g.Ny, x_aperture_um=r.grid.x_aperture_um,
                y_aperture_um=r.grid.y_aperture_um, real_dtype=b.real_dtype, xp=np)
            A *= np.exp(1j*phi)
        boundaries.append(A.copy()); sources.append(I.copy()); materials.append(E.copy())
    return boundaries, sources, materials


@pytest.mark.parametrize('model', ['field_linear_local_intensity','nonlinear'])
@pytest.mark.parametrize('n', [1,2,5])
@pytest.mark.parametrize('scattering', [False,True])
def test_exact_reference_coordinates_products_and_retention(model,n,scattering):
    r=request(n,model,scattering=scattering)
    boundaries,sources,materials=reference(r)
    selected=LocalPlaneProductSelection(**dict.fromkeys(vars(LocalPlaneProductSelection()),True))
    out=run_published_products(r,selection=selected)
    s,p=out.scientific,out.products
    assert s.status == 'completed',s.reason
    assert s.workflow_identity == PR_PUBLISHED_STATIC_WORKFLOW
    assert s.arithmetic_identity == PR_PUBLISHED_STATIC_ARITHMETIC
    assert s.completed_cells==n and s.reached_z_um==r.grid.z_length_um
    np.testing.assert_array_equal(p.boundary_z_um,np.arange(n+1)*r.grid.dz_um)
    np.testing.assert_array_equal(p.material_z_um,np.arange(1,n+1)*r.grid.dz_um)
    np.testing.assert_array_equal(s.boundary_field,boundaries[-1])
    ix,iy=np.argmin(abs(p.x_um)),np.argmin(abs(p.y_um))
    np.testing.assert_array_equal(p.cuts['optical'][0],np.asarray(boundaries)[:,:,:,iy])
    np.testing.assert_array_equal(p.cuts['source'][1],np.asarray(sources)[:,ix,:])
    np.testing.assert_array_equal(p.cuts['material'][0],np.asarray(materials)[:,:,iy])
    minimal=run_published_static(r,retain_boundary_field=True)
    np.testing.assert_array_equal(minimal.boundary_field,s.boundary_field)
    decoded=decode_published_products(encode_published_products(out))
    np.testing.assert_array_equal(decoded.products.material_z_um,p.material_z_um)
    np.testing.assert_array_equal(decoded.scientific.far_field.intensity,s.far_field.intensity)
    assert not hasattr(p,'center_z_um')
    for record in s.ledger:
        assert record['material_plane_um']==record['z_end_um']
        assert record['optical_step_um']==r.grid.dz_um
        assert record['material_weight_um']==r.grid.dz_um


@pytest.mark.parametrize('model',['field_linear_local_intensity','nonlinear'])
def test_shared_driver_exact_order(monkeypatch,model):
    import lcprop.pr.published_static_step as step
    import lcprop.pr.published_static as workflow
    calls=[]
    for name,label in [('hop_linear_inplace','P'),('pr_driving_intensity','I'),
                       ('apply_response_screen_inplace','phase'),('canonical_scattering_phase_increment','S')]:
        fn=getattr(step,name)
        def traced(*a,_fn=fn,_label=label,**kw):
            calls.append(_label);return _fn(*a,**kw)
        monkeypatch.setattr(step,name,traced)
    original=workflow.solve_arriving_material
    def material(*a,**kw): calls.append('material');return original(*a,**kw)
    monkeypatch.setattr(workflow,'solve_arriving_material',material)
    out=run_published_static(request(2,model,scattering=True))
    assert out.status=='completed',out.reason
    assert calls==['P','I','material','phase','S','phase']*2
    assert 'advance_pr_slice_with_midpoint_source' not in inspect.getsource(step)


def test_failure_no_unaccepted_products(monkeypatch):
    import lcprop.pr.published_static as workflow
    original=workflow.solve_arriving_material
    count=0
    def fail(I,**kw):
        nonlocal count
        count+=1
        if count==2: raise ValueError('injected material failure')
        return original(I,**kw)
    monkeypatch.setattr(workflow,'solve_arriving_material',fail)
    out=run_published_products(request(3))
    assert out.scientific.status=='failed'
    assert count==2 and out.scientific.completed_cells==1
    assert out.products.boundary_z_um.tolist()==[0,2]
    assert out.products.material_z_um.tolist()==[2]


def test_nonconvergence_fails_before_phase():
    r=replace(request(model='nonlinear'),material_solver=PRStaticSolverOptions(max_iterations=0))
    out=run_published_static(r,retain_boundary_field=True)
    assert out.status=='failed' and out.completed_cells==0
    np.testing.assert_array_equal(out.boundary_field,r.initial_A)
    assert 'material solve failed' in out.reason


def test_model_and_td_bytes_unchanged():
    from pathlib import Path
    paths=['src/lcprop/pr/static.py','src/lcprop/pr/reduced_field_linear.py',
           'src/lcprop/pr/evolution.py','src/lcprop/pr/workflow.py',
           'src/lcprop/pr/transverse/workflow.py','src/lcprop/pr/transverse/transport.py',
           'src/lcprop/pr/local_plane_step.py']
    for path in paths:
        assert Path(path).read_bytes()==subprocess.check_output(['git','show',f'ec84ffbd9230601f1615738229aa9906da539a60:{path}'])


@pytest.mark.parametrize('cells',[0,1,2])
def test_cancelled_coordinates_only_accepted(cells):
    from lcprop.core.execution import CancellationToken
    token=CancellationToken()
    def progress(p):
        if p.completed_cells==cells: token.cancel()
    out=run_published_products(request(3),cancellation_token=token,progress_callback=progress)
    assert out.scientific.status=='cancelled'
    assert out.products.boundary_z_um.tolist()==[2.*k for k in range(cells+1)]
    assert out.products.material_z_um.tolist()==[2.*k for k in range(1,cells+1)]
    assert out.scientific.far_field is None
    decode_published_products(encode_published_products(out))


def test_preview_and_intensity_exact_postreduction():
    from lcprop.pr.visualization import block_average_2d
    from lcprop.pr.longitudinal_cuts import presentation_peak_intensity_reference, _normalize_presentation_cut
    r=request(3)
    boundaries,_,_=reference(r)
    out=run_published_products(r)
    p=out.products
    peak=channel_peak_intensity_reference(boundaries[0],xp=np)
    display=presentation_peak_intensity_reference(boundaries[0],asnumpy=np.asarray)
    assert p.presentation_reference==display
    planes=np.stack([pr_driving_intensity(A,peak_intensity_reference=peak,
        background_intensity=r.material.background_intensity,coherence_groups=r.beams.coherence_groups,xp=np) for A in boundaries])
    ix,iy=np.argmin(abs(p.x_um)),np.argmin(abs(p.y_um))
    for axis,raw in enumerate((planes[:,:,iy],planes[:,ix,:])):
        expected=np.stack([_normalize_presentation_cut(v,display,r.material.background_intensity) for v in raw])
        np.testing.assert_array_equal(expected,p.cuts['intensity'][axis])
    size=min(96,max(1,int(np.sqrt(r.grid.Nx+r.grid.Ny))))
    preview=np.stack([block_average_2d((I.astype(np.float64)-r.material.background_intensity)*display,
        max_x=size,max_y=size)[0].astype(np.float32) for I in planes])
    np.testing.assert_array_equal(preview,p.preview)


@pytest.mark.parametrize('model',['field_linear_local_intensity','nonlinear'])
def test_float32_no_longitudinal_volume(monkeypatch,model):
    r=request(5,model,precision='float32')
    r=replace(r,material_solver=PRStaticSolverOptions(residual_rms_tolerance=2e-6,residual_max_tolerance=1e-5))
    original=np.empty
    def guarded(shape,*args,**kwargs):
        if isinstance(shape,tuple): assert shape != (5,16,8)
        return original(shape,*args,**kwargs)
    monkeypatch.setattr(np,'empty',guarded)
    out=run_published_products(r)
    assert out.scientific.status=='completed',out.scientific.reason
    assert out.products.preview.shape[1]*out.products.preview.shape[2] <= 24
    assert all(not isinstance(v,np.ndarray) for record in out.scientific.ledger for v in record.values())
