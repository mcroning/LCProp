"""Layout/coordinate-only regressions; no native backend."""
import hashlib
from dataclasses import replace
import numpy as np
import pytest
from tests.test_synchronized_fields import app, fixture, select
from lcprop.gui.workspace import Workspace
from lcprop.gui.field_selection import independent_plane_label


def test_plane_expands_and_linked_layout_restores(app):
    w=Workspace();w.resize(1250,1000);w.set_run_data(fixture());w.show();app.processEvents()
    select(w,'volume','psi');app.processEvents()
    split_width=w.image_pane.width()
    select(w,'plane','far');app.processEvents()
    assert w.longitudinal_pane.isHidden()
    assert w.image_pane.width()>split_width
    assert w.image_pane.image_view.ax.get_aspect()==1.
    select(w,'volume','psi');app.processEvents()
    assert not w.longitudinal_pane.isHidden()
    assert w.longitudinal_pane.xz_view._field is not None
    w.close()


@pytest.mark.parametrize('vertical',[False,True])
def test_orientation_coordinates_clicks_guides_and_hashes(app,vertical):
    r=fixture();a=r.fields['psi'].data;before=hashlib.sha256(a.tobytes()).hexdigest()
    w=Workspace();w.set_run_data(r);select(w,'volume','psi');p=w.longitudinal_pane
    p.set_cut_indices(2,1);p.z_plane_slider.setValue(2)
    p.orientation.setCurrentIndex(int(vertical))
    assert (p._ix,p._iy,p._iz)==(2,1,2)
    for view,expected,axis in [(p.xz_view,a[:,:,1],'x'),(p.yz_view,a[:,2,:],'y')]:
        assert np.shares_memory(view._field.data,a)
        np.testing.assert_array_equal(view.image.get_array(),expected if vertical else expected.T)
        assert view.ax.get_xlabel().startswith(axis if vertical else 'z')
        assert view.ax.get_ylabel().startswith('z' if vertical else axis)
    coords=p.xz_view._index_to_display_coordinates(*( (2,1) if vertical else (1,2)))
    assert coords==((1.,3.) if vertical else (3.,1.))
    p.xz_view.positionSelected.emit(*( (2,1) if vertical else (1,2)))
    assert (p._ix,p._iz)==(2,1)
    assert p.xz_view._crosshair_index==((2,1) if vertical else (1,2))
    np.testing.assert_array_equal(w.image_pane.image_view._field.data,a[1])
    assert hashlib.sha256(a.tobytes()).hexdigest()==before
    w.close()


def test_independent_plane_z_requires_explicit_metadata():
    f=replace(fixture().fields['psi'],axes=('x','y'),data=np.ones((4,2)),
              display_name='Optical Intensity x-y')
    assert 'unspecified' in independent_plane_label(f) # volume z vector is not plane provenance
    assert 'z=17 µm' in independent_plane_label(replace(f,coordinates={'selected_z_um':17.}))
    assert 'z=19 µm' in independent_plane_label(replace(f,coordinates={'z':19.},units={'z':'um'}))
    assert 'unspecified' in independent_plane_label(replace(f,coordinates={'z':float('nan')}))


@pytest.mark.parametrize('size',[(800,650),(1250,1000)])
def test_resized_axes_labels_and_colorbar_fit_canvas(app,size):
    w=Workspace();w.resize(*size);w.set_run_data(fixture());select(w,'volume','face');w.show()
    app.processEvents()
    for orientation in (0,1):
        w.longitudinal_pane.orientation.setCurrentIndex(orientation);app.processEvents()
        for view in (w.image_pane.image_view,w.longitudinal_pane.xz_view,w.longitudinal_pane.yz_view):
            view.draw();renderer=view.get_renderer();bounds=view.figure.bbox
            for ax in (view.ax,view.colorbar.ax):
                box=ax.get_tightbbox(renderer)
                assert box.x0 >= -1 and box.y0 >= -1
                assert box.x1 <= bounds.width+1 and box.y1 <= bounds.height+1
    w.close()


def test_tilted_full_transverse_td_motion(app, record_property):
    """One gain-zero material step; presentation regression, not qualification."""
    import json
    from tests.test_pr_transverse_continuation import request
    from lcprop.core.context import GridSpec
    from lcprop.pr.transverse.workflow import run_pr_transverse_timedependent
    from lcprop.pr.transverse.products import pr_transverse_result_to_run_data
    r=request(steps=1,dt=.0001)
    kx=.934993
    beam=replace(r.beams.channels[0], wavelength_um=.633,
                 theta_ext_rad=float(np.arcsin(kx*.633/(2*np.pi))),n_ext=1.,phi_rad=0.)
    r=replace(r,grid=GridSpec(Nx=128,Ny=128,x_aperture_um=200.,y_aperture_um=200.,
                             dz_um=10.,z_length_um=1000.),
              beams=replace(r.beams,channels=(beam,)),
              material=replace(r.material,gain_length_product=0.,refractive_index=2.4))
    result=run_pr_transverse_timedependent(r)
    assert result.status=='completed'
    data=pr_transverse_result_to_run_data(result)
    volume=data.fields['optical_intensity_stack'].data
    before=hashlib.sha256(volume.tobytes()).hexdigest()
    x,z=data.geometry.x,data.geometry.z
    weight=volume.sum(axis=2)
    centers=(weight*x[None,:]).sum(axis=1)/weight.sum(axis=1)
    kz=np.sqrt((2*np.pi*2.4/.633)**2-kx**2)
    expected=kx/kz*(z[-1]-z[0])
    assert centers[-1]-centers[0]==pytest.approx(expected,abs=.1)
    w=Workspace();w.set_run_data(data);select(w,'volume','optical_intensity_stack')
    p=w.longitudinal_pane
    p.set_cut_coordinates(0.,0.)
    cut=p.xz_view._field.data
    cut_centers=(cut*x[None,:]).sum(axis=1)/cut.sum(axis=1)
    assert cut_centers[-1]-cut_centers[0]==pytest.approx(expected,abs=.1)
    p.orientation.setCurrentIndex(1)
    np.testing.assert_array_equal(p.xz_view.image.get_array(),cut)
    assert p.xz_view.ax.get_ylabel().startswith('z')
    assert hashlib.sha256(volume.tobytes()).hexdigest()==before
    assert 'z unspecified' in independent_plane_label(data.fields['optical_intensity_xy'])
    record_property('tilted_beam',json.dumps({'grid':[128,128,100], 'wavelength_um':.633,
        'n':2.4,'kx_rad_per_um':kx,'gain_length':0.,'z_samples_um':[float(z[0]),float(z[-1])],
        'measured_shift_um':float(centers[-1]-centers[0]),'xz_shift_um':float(cut_centers[-1]-cut_centers[0]),
        'expected_between_samples_um':float(expected),'expected_per_mm_um':float(1000*kx/kz),
        'volume_sha256_before_after':before,'material_steps':1}))
    w.close()
