"""Independent geometry, discrete measure, and real Results regressions."""
from dataclasses import replace
import math
import numpy as np
import pytest

from lcprop.core.beams import BeamChannel
from lcprop.optics.physical_launch import resolve_beam_geometry
from lcprop.pr.far_field import far_field_product
from lcprop.pr.far_field_mask import carrier_ellipses, mask_diagnostic, MASK_KEY
from lcprop.pr.products import pr_result_to_run_data
from lcprop.gui.workspace import Workspace
from tests.test_pr_live_results import app, close_widget, no_movie_encoder, collect_test_objects_on_gui_thread


def summary(*beams):
    return {'Nch':len(beams), 'resolved_beams':[resolve_beam_geometry(b,2.4).summary() for b in beams]}


@pytest.mark.parametrize('theta,phi,roll,w1,w2', [
    (0,0,0,20,20),(0,0,0,10,30),(0,0,math.pi/2,10,30),
    (.6,0,0,20,20),(.6,.7,.4,10,30),(-.6,2.1,.9,10,30)])
def test_resolved_covariance_independent_frame(theta,phi,roll,w1,w2):
    b=BeamChannel(theta_ext_rad=theta,phi_rad=phi,psi_rad=roll,w1_um=w1,w2_um=w2)
    e=carrier_ellipses(summary(b))[0]
    k=2*math.pi*2.4/.633
    # Explicit transverse basis (no resolver/frame helper for expectations).
    c,s,cp,sp=math.cos(theta),math.sin(theta),math.cos(phi),math.sin(phi)
    u=np.array([1+(c-1)*cp*cp,(c-1)*cp*sp])
    v=np.array([(c-1)*cp*sp,1+(c-1)*sp*sp])
    a=math.cos(roll)*u+math.sin(roll)*v
    d=-math.sin(roll)*u+math.cos(roll)*v
    expected=(np.outer(a,a)/w1**2+np.outer(d,d)/w2**2)/k**2
    np.testing.assert_allclose(e['covariance_s'],expected,rtol=2e-14,atol=1e-20)
    np.testing.assert_allclose(e['center_s'],[s*cp/2.4,s*sp/2.4],atol=1e-16)


@pytest.mark.parametrize('centers', [[(0,0)],[(.03,-.02)],[(-.03,0),(.03,0)],[(0,0),(.005,0)]])
def test_synthetic_bright_lobes_union_and_independent_integral(centers):
    sx=np.arange(-.1,.101,.002);sy=np.arange(-.08,.081,.002)
    sigma=.004
    # Retained resolved metadata constructed independently of the resolver.
    launch={'Nch':len(centers),'resolved_beams':[
        dict(kx_rad_per_um=x,ky_rad_per_um=y,kz_internal_rad_per_um=math.sqrt(1-x*x-y*y),
             interface_quadratic_per_um2=[[sigma*sigma,0],[0,sigma*sigma]]) for x,y in centers]}
    excluded=np.zeros((len(sx),len(sy)),bool)
    for x,y in centers:excluded |= ((sx[:,None]-x)**2+(sy[None,:]-y)**2)<= (4*sigma)**2
    intensity=np.full(excluded.shape,.01);intensity[excluded]=1e12
    intensity[-5,-4]=7.;intensity[4,-7]=3.
    original=intensity.copy();field=far_field_product(intensity,sx,sy)
    masked,metric=mask_diagnostic(field,launch)
    np.testing.assert_array_equal(np.isnan(masked.data),excluded)
    np.testing.assert_array_equal(masked.data[~excluded],original[~excluded])
    np.testing.assert_array_equal(field.data,original)
    assert metric['off_carrier_field_norm']==pytest.approx(original[~excluded].sum()*.002**2)
    assert metric['off_carrier_fraction']==pytest.approx(original[~excluded].sum()/original.sum())
    assert metric['carrier_count']==len(centers)
    assert masked.coordinates['s_x'] is field.coordinates['s_x']
    assert metric['ideal_single_lobe_enclosed_fraction']==pytest.approx(1-math.exp(-8))


def test_unavailable_and_zero_norm():
    field=far_field_product(np.zeros((3,3)),np.arange(3.),np.arange(3.))
    masked,metric=mask_diagnostic(field,summary(BeamChannel()))
    assert metric['off_carrier_fraction'] is None
    assert metric['off_carrier_field_norm']==0
    with pytest.raises(ValueError,match='unavailable'):mask_diagnostic(field,{})
    with pytest.raises(ValueError):mask_diagnostic(field,summary(BeamChannel()),sigmas=0)


def test_completed_product_autoscale_manual_switch_and_invalidation(app):
    from tests.test_pr_products import _synthetic_pr_result
    r=_synthetic_pr_result();shape=r.A_final.shape
    # Known Fourier coefficients: bright center plus weak off-carrier bin.
    f=np.zeros(shape[1:],complex);f[0,0]=1e6;f[1,1]=3
    a=np.fft.ifft2(f);r=replace(r,A_final=np.stack([a,a*0]),
        launch_summary=dict(r.launch_summary,**summary(BeamChannel(w1_um=100,w2_um=100),BeamChannel(w1_um=100,w2_um=100)),wavelengths_um=[.633,.633],refractive_index=2.4))
    data=pr_result_to_run_data(r);canonical=data.fields['far_field_intensity'];before=canonical.data.copy()
    expected=abs(np.fft.fftshift(f)*8)**2*(2.4/.633)**2
    np.testing.assert_allclose(canonical.data,expected,atol=.001)
    ws=Workspace()
    try:
        ws.set_run_data(data);pane=ws.image_pane
        pane.field_selector.setCurrentIndex(pane.field_selector.findData(MASK_KEY))
        masked=data.fields[MASK_KEY];valid=masked.data[np.isfinite(masked.data)]
        assert pane.image_view.image.get_clim()[1]==pytest.approx(valid.max())
        assert np.ma.getmaskarray(pane.image_view.image.get_array()).sum()==1
        pane.scale_controls.lower.setText('0');pane.scale_controls.upper.setText('10');pane.scale_controls.apply_limits()
        assert pane.image_view.image.get_clim()==(0,10)
        pane.field_selector.setCurrentIndex(pane.field_selector.findData('far_field_intensity'))
        assert pane.image_view.image.get_clim()[1]==pytest.approx(before.max())
        np.testing.assert_array_equal(canonical.data,before)
        pane.field_selector.setCurrentIndex(pane.field_selector.findData(MASK_KEY))
        assert pane.image_view.image.get_clim()==(0,10)
        ws.invalidate_products()
        assert not pane._run_data.fields and ws.diagnostics_view.toPlainText().strip() == 'Workflow: pending'
    finally:close_widget(ws)


def test_gain_pair_completed_full_fast_and_transverse(no_movie_encoder):
    from tests.test_pr_execution import _request
    from lcprop.pr.workflow import run_pr_timedependent
    from lcprop.pr.scattering import PRCanonicalScatteringSpec,PR_CANONICAL_SCATTERING_V2
    from lcprop.pr.specs import PR_EULER_INTEGRATOR
    r=_request(steps=2)
    r=replace(r,initial_A=None,scattering=PRCanonicalScatteringSpec(.02,.4,0,1.,PR_CANONICAL_SCATTERING_V2),
              solver=replace(r.solver,integrator=PR_EULER_INTEGRATOR))
    previous=None
    for gain in (0.,10.):
        result=run_pr_timedependent(replace(r,material=replace(r.material,gain_length_product=gain)))
        data=pr_result_to_run_data(result);ff=data.fields['far_field_intensity'];masked=data.fields[MASK_KEY];metric=data.diagnostics['off_carrier'].values
        np.testing.assert_allclose(ff.data,abs(np.fft.fftshift(np.fft.fft2(result.A_final[0]))*result.grid_summary['dx_um']*result.grid_summary['dy_um'])**2*(2.4/.633)**2,rtol=2e-14,atol=1e-25)
        good=np.isfinite(masked.data);np.testing.assert_array_equal(masked.data[good],ff.data[good])
        cell=np.diff(ff.coordinates['s_x'])[0]*np.diff(ff.coordinates['s_y'])[0]
        assert metric['off_carrier_field_norm']==pytest.approx(ff.data[good].sum()*cell)
        assert metric['off_carrier_fraction']==pytest.approx(ff.data[good].sum()/ff.data.sum())
        if previous:
            assert metric['carriers']==previous[0]
            assert data.diagnostics['pr_workflow'].values['canonical_scattering']==previous[1]
        previous=(metric['carriers'],data.diagnostics['pr_workflow'].values['canonical_scattering'])
        fast=pr_result_to_run_data(replace(result,retention_summary={'policy':'fast','omitted_fields':[]}))
        np.testing.assert_allclose(fast.fields[MASK_KEY].data,masked.data,equal_nan=True)
    from tests.test_pr_transverse_production import _request as transverse_request
    from lcprop.pr.transverse import run_pr_transverse_timedependent,pr_transverse_result_to_run_data
    result=run_pr_transverse_timedependent(transverse_request(steps=1))
    for policy in ('full','fast'):
        data=pr_transverse_result_to_run_data(replace(result,retention_summary={'policy':policy,'omitted_fields':[]}))
        assert MASK_KEY in data.fields and data.diagnostics['off_carrier'].values['status']=='available'


def test_real_worker_new_run_never_reuses_mask_and_disabled_beam(app,monkeypatch,no_movie_encoder):
    from tests.test_pr_gui_main_window import _tiny_window,_wait_for
    from lcprop.pr.live_results import PRLivePreviewPolicy
    w=_tiny_window(app,steps=2);ws=w.results_panel.workspace;seen=[]
    original_run=w.local_runner.run_registered;original_set=ws.set_run_data
    def run(*args,**kwargs):
        kwargs['live_preview_policy']=PRLivePreviewPolicy(interval_seconds=0)
        return original_run(*args,**kwargs)
    def display(data,*,state=None):
        if state in ('Waiting for current result','Current accepted state'):
            assert MASK_KEY not in data.fields and 'off_carrier' not in data.diagnostics
            seen.append(state)
        original_set(data,state=state)
    monkeypatch.setattr(w.local_runner,'run_registered',run);monkeypatch.setattr(ws,'set_run_data',display)
    try:
        stack=w.beam_panel.beam_stack_definition
        disabled=replace(stack.beams[0],name='disabled',enabled=False,theta_ext_rad=.2)
        w.beam_panel.set_beam_stack_definition(replace(stack,beams=(*stack.beams,disabled)))
        w.run_clicked();_wait_for(app,lambda:not w._background_running)
        a=w.last_runner_result.run_data;ma=a.diagnostics['off_carrier'].values
        assert ma['carrier_count']==1
        ws.image_pane.field_selector.setCurrentIndex(ws.image_pane.field_selector.findData(MASK_KEY))
        stack=w.beam_panel.beam_stack_definition
        w.beam_panel.set_beam_stack_definition(replace(stack,beams=(replace(stack.beams[0],theta_ext_rad=.001,w1_um=10),disabled)))
        w.run_clicked();_wait_for(app,lambda:not w._background_running)
        b=w.last_runner_result.run_data;mb=b.diagnostics['off_carrier'].values
        assert mb['carrier_count']==1 and mb['carriers']!=ma['carriers']
        assert ws.result_ownership.text()=='Completed result'
        assert seen.count('Waiting for current result')==2 and seen.count('Current accepted state')==4
        assert 'off_carrier' in ws.diagnostics_view.toPlainText()
    finally:assert w.shutdown_background_run();close_widget(w)


def test_all_excluded_and_missing_geometry_are_explicit(app):
    from lcprop.products.data_model import FieldCollection, DiagnosticCollection
    from lcprop.pr.far_field_mask import add_completed_mask
    from types import SimpleNamespace
    from lcprop.gui.views.display_scale import volume_limits
    field=far_field_product(np.ones((3,3)),np.arange(-1,2)*1e-5,np.arange(-1,2)*1e-5)
    masked,metric=mask_diagnostic(field,summary(BeamChannel()))
    assert np.isnan(masked.data).all() and metric['off_carrier_fraction']==0
    assert volume_limits(masked.data)==(0.,1.)
    fields=FieldCollection([(field.key,field)]);diags=DiagnosticCollection()
    add_completed_mask(fields,diags,SimpleNamespace(launch_summary={}))
    assert MASK_KEY not in fields
    assert diags['off_carrier'].values['status']=='unavailable'
    assert 'geometry' in diags['off_carrier'].values['reason']


def test_rolled_ellipse_exclusion_against_explicit_principal_coordinates():
    # Normal beam, unequal radii, 45-degree roll: use principal coordinates
    # directly, independently of the production inverse-covariance expression.
    axis=np.arange(-30,31)*.0005
    b=BeamChannel(w1_um=15,w2_um=40,psi_rad=math.pi/4)
    field=far_field_product(np.ones((61,61)),axis,axis)
    masked,_=mask_diagnostic(field,summary(b))
    x,y=axis[:,None],axis[None,:];k=2*math.pi*2.4/.633
    u=(x+y)/math.sqrt(2);v=(-x+y)/math.sqrt(2)
    expected=(u*15*k)**2+(v*40*k)**2<=16
    np.testing.assert_array_equal(np.isnan(masked.data),expected)
    untilted,_=mask_diagnostic(field,summary(replace(b,psi_rad=0)))
    assert not np.array_equal(np.isnan(untilted.data),expected)
