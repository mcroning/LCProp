"""PR TD display lifecycle, exact guide coordinates, and output-plane spectra."""
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from lcprop.gui.workspace import Workspace
from lcprop.pr.far_field import completed_far_field
from lcprop.pr.live_results import PRLivePreviewPolicy, reduced_pr_live_to_run_data
from lcprop.pr.products import pr_result_to_run_data
from lcprop.pr.workflow import run_pr_timedependent
from tests.test_pr_live_results import (app, close_widget, snapshot_fixture,
                                      no_movie_encoder, collect_test_objects_on_gui_thread)
from tests.test_pr_gui_main_window import _tiny_window, _wait_for
from tests.test_pr_products import _synthetic_pr_result
from tests.test_pr_execution import _request


def test_non_square_complex_plane_independent_dft_and_parseval():
    result = _synthetic_pr_result()
    nx, ny = result.A_final.shape[1:]
    x, y = np.arange(nx)[:, None], np.arange(ny)[None, :]
    a = (1+x+.3*y)*np.exp(2j*np.pi*(x/nx-y/ny))
    channels = np.stack((a, .3j*a))
    launch = dict(result.launch_summary, wavelengths_um=[.633,.633], refractive_index=2.4)
    result = replace(result, A_final=channels, launch_summary=launch)
    field = pr_result_to_run_data(result).fields['far_field_intensity']
    # Independent direct DFT on each axis, coherent sum BEFORE the transform.
    fx = np.fft.fftshift(np.fft.fftfreq(nx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny))
    dft = np.exp(-2j*np.pi*fx[:,None]*np.arange(nx)) @ channels.sum(axis=0) @ np.exp(-2j*np.pi*np.arange(ny)[:,None]*fy)
    expected = abs(dft*2*4)**2*(2.4/.633)**2
    np.testing.assert_allclose(field.data, expected, rtol=2e-14, atol=1e-20)
    np.testing.assert_allclose(field.coordinates['s_x'], fx*.633/(2*2.4))
    np.testing.assert_allclose(field.coordinates['s_y'], fy*.633/(4*2.4))
    dsx, dsy = .633/(2*2.4*nx), .633/(4*2.4*ny)
    assert field.data.sum()*dsx*dsy == pytest.approx((abs(channels.sum(axis=0))**2).sum()*8)
    assert field.axes == ('s_x','s_y') and 'mW' not in field.value_unit
    incoherent = replace(result, launch_summary=dict(launch, coherence_groups=['a','b']))
    other = completed_far_field(incoherent)
    assert other.data.sum() == pytest.approx(field.data.sum())  # quadrature relative phase
    different = replace(result, A_final=np.stack((a,a)))
    separate = replace(different, launch_summary=dict(launch, coherence_groups=['a','b']))
    np.testing.assert_allclose(completed_far_field(different).data,2*completed_far_field(separate).data)


def lines(view, x, y):
    assert view._vline is not None and view._hline is not None
    np.testing.assert_allclose(view._vline.get_xdata(), [x,x])
    np.testing.assert_allclose(view._hline.get_ydata(), [y,y])
    assert view._vline.get_visible() and view._hline.get_visible()


def test_live_fixed_guides_are_physical_and_survive_every_refresh(app):
    snap,_,_,_ = snapshot_fixture()
    # These fixed physical cuts lie BETWEEN preview samples; don't round them.
    snap = replace(snap, x_cut_um=-1.25, y_cut_um=2.75)
    ws=Workspace();p=ws.longitudinal_pane;xy=ws.image_pane
    try:
        cut_indices=[]
        p.cutChanged.connect(lambda x,y: cut_indices.append((x,y)))
        for step in (1,2,3):
            ws.set_run_data(reduced_pr_live_to_run_data(replace(snap, completed_steps=step)),state='Current accepted state')
            p.z_plane_slider.setValue(2)
            assert p.show_guides.isChecked() and not p.show_guides.isHidden()
            for key in ('live_output_intensity','live_material_plane'):
                xy.field_selector.setCurrentIndex(xy.field_selector.findData(key))
                lines(xy.image_view,-1.25,2.75)
                lines(p.xz_view,snap.z_um[2],-1.25);lines(p.yz_view,snap.z_um[2],2.75)
            xy.field_selector.setCurrentIndex(xy.field_selector.findData('far_field_intensity'))
            assert xy.image_view._vline is None  # Spatial guides cannot label angular axes.
            p.scale_controls.lower.setText('0');p.scale_controls.upper.setText('10');p.scale_controls.apply_limits()
            p.scale_controls.mode.setCurrentIndex(p.scale_controls.mode.findData('auto'))
            ws.resize(1250+step*10,850);app.processEvents()
            p._xz_position_selected(3,1)
            lines(p.xz_view,snap.z_um[3],-1.25)
            # Clicking a fixed cut cannot replace its declared cut by a slider index.
            ws._image_position_selected(0,0)
            lines(p.yz_view,snap.z_um[3],2.75)
        assert not cut_indices  # Fixed physical cuts never emit fabricated indices.
        p.show_guides.setChecked(False)
        ws.set_run_data(reduced_pr_live_to_run_data(snap),state='Current accepted state')
        assert p.xz_view._vline is None and xy.image_view._vline is None
        p.show_guides.setChecked(True)
        lines(p.xz_view,snap.z_um[p._iz],-1.25)
    finally:close_widget(ws)


def test_real_worker_A_to_B_invalidates_before_first_progress(app,monkeypatch,no_movie_encoder):
    from lcprop.pr.live_results import PRLivePreviewPolicy
    import lcprop.pr.gui.main_window as gui
    w=_tiny_window(app,steps=3);ws=w.results_panel.workspace
    # This historical reduced-TD fixture retains its predecessor input semantics.
    w.material_panel.normalization_mode.setCurrentIndex(1)
    w.evolution_panel.set_workflow_id("pr_timedependent")
    entered,release=Event(),Event();deliveries=[];starts=[]
    original_run=w.local_runner.run_registered
    def run(*a,**k):
        # Hold B before it can publish anything; normal worker/slot remain intact.
        if starts:
            entered.set()
            if not release.wait(8):raise RuntimeError('test release timed out')
        starts.append(True)
        k['live_preview_policy']=PRLivePreviewPolicy(interval_seconds=0)
        return original_run(*a,**k)
    original_set=ws.set_run_data
    def set_data(data,*,state=None):
        original_set(data,state=state)
        if state=='Current accepted state':
            deliveries.append((ws._attempt,np.array(data.fields['live_output_intensity'].data),
                               np.array(data.fields['far_field_intensity'].data)))
    monkeypatch.setattr(w.local_runner,'run_registered',run)
    monkeypatch.setattr(ws,'set_run_data',set_data)
    try:
        w.run_clicked();_wait_for(app,lambda:not w._background_running)
        a=w.last_runner_result.run_data
        a_output=a.fields['output_intensity'].data.copy()
        ws.image_pane.field_selector.setCurrentIndex(ws.image_pane.field_selector.findData('far_field_intensity'))
        beam=w.beam_panel.beam_stack_definition
        w.beam_panel.set_beam_stack_definition(replace(beam,beams=(replace(beam.beams[0],w1_um=9.,w2_um=13.),)))
        w.run_clicked();_wait_for(app,entered.is_set)
        assert ws.result_ownership.text()=='Waiting for current result'
        assert not ws.image_pane._run_data.fields and not ws.longitudinal_pane._run_data.fields
        assert ws.image_pane.image_view._field is None
        assert ws.image_pane.image_view.image.get_array().size==0
        assert ws.longitudinal_pane.xz_view.image.get_array().size==0
        ws.resize(1200,850);ws.image_pane.reset_color_scales();app.processEvents()
        assert ws.image_pane.image_view._field is None
        # Saved result A was not destroyed by display invalidation.
        np.testing.assert_array_equal(a.fields['output_intensity'].data,a_output)
        release.set();_wait_for(app,lambda:not w._background_running)
        b=w.last_runner_result.run_data
        assert ws.result_ownership.text().startswith('Completed result')
        assert ws.image_pane.field_selector.currentData()=='far_field_intensity'
        np.testing.assert_array_equal(ws.image_pane.image_view.image.get_array(),b.fields['far_field_intensity'].data.T)
        assert not np.array_equal(a_output,b.fields['output_intensity'].data)
        for attempt in (1,2):assert sum(d[0]==attempt for d in deliveries)==3
        assert not np.array_equal(deliveries[0][2],deliveries[3][2])
    finally:
        release.set();assert w.shutdown_background_run();close_widget(w)


def test_gain_pair_spectra_have_own_output_and_scattering_provenance(no_movie_encoder):
    from lcprop.pr.scattering import PRCanonicalScatteringSpec,PR_CANONICAL_SCATTERING_V2
    from lcprop.pr.specs import PR_EULER_INTEGRATOR
    r=_request(steps=2)
    r=replace(r,initial_A=None,scattering=PRCanonicalScatteringSpec(.02,.4,0,1.,PR_CANONICAL_SCATTERING_V2),
              solver=replace(r.solver,integrator=PR_EULER_INTEGRATOR))
    products=[];provenance=[]
    for gain in (0.,10.):
        events=[]
        result=run_pr_timedependent(replace(r,material=replace(r.material,gain_length_product=gain)),
            progress_callback=events.append,live_preview_policy=PRLivePreviewPolicy(interval_seconds=0))
        product=pr_result_to_run_data(result);f=product.fields['far_field_intensity'];products.append(f)
        dx,dy=result.grid_summary['dx_um'],result.grid_summary['dy_um']
        expected=abs(np.fft.fftshift(np.fft.fft2(result.A_final[0]))*dx*dy)**2*(2.4/.633)**2
        np.testing.assert_allclose(f.data,expected,rtol=2e-14,atol=1e-25)
        np.testing.assert_allclose(events[-1].latest_field_state.far_field_intensity,expected,rtol=2e-14,atol=1e-25)
        provenance.append(product.diagnostics['pr_workflow'].values['canonical_scattering'])
    for axis in ('s_x','s_y'):np.testing.assert_array_equal(products[0].coordinates[axis],products[1].coordinates[axis])
    assert products[0].quantity==products[1].quantity and products[0].value_unit==products[1].value_unit
    assert provenance[0]==provenance[1]
    assert provenance[0]['algorithm_version']==PR_CANONICAL_SCATTERING_V2


def test_full_transverse_completed_far_field(no_movie_encoder):
    from tests.test_pr_transverse_production import _request as request
    from lcprop.pr.transverse import run_pr_transverse_timedependent,pr_transverse_result_to_run_data
    result=run_pr_transverse_timedependent(request(steps=1))
    data=pr_transverse_result_to_run_data(result)
    f=data.fields['far_field_intensity']
    expected=abs(np.fft.fftshift(np.fft.fft2(result.A_final[0]))*4*4)**2*(2.4/.633)**2
    np.testing.assert_allclose(f.data,expected,rtol=2e-14,atol=1e-25)


def test_guides_completed_volume_sliders_and_replacement(app):
    from tests.test_stage1b_transparency import image_data
    ws=Workspace()
    try:
        snap,_,_,_=snapshot_fixture()
        ws.set_run_data(reduced_pr_live_to_run_data(snap),state='Current accepted state')
        ws.set_run_data(image_data(),state='Completed result')
        p=ws.longitudinal_pane;p.set_cut_indices(1,2);p.z_plane_slider.setValue(0)
        for scale in (1.,2.):
            ws.set_run_data(image_data(scale),state='Completed result')
            x,y,z=p.guide_coordinates()
            lines(ws.image_pane.image_view,x,y)
            lines(p.xz_view,z,x);lines(p.yz_view,z,y)
            assert (p._ix,p._iy,p._iz)==(1,2,0)
            field=ws.image_pane.image_view._field
            volume=p._run_data.fields[p.field_selector.currentData()].data
            np.testing.assert_array_equal(field.data,volume[0])
    finally:close_widget(ws)


def test_live_fft_before_decimation_and_bounded_host_transfer(monkeypatch):
    import lcprop.pr.live_results as live
    original=live.direction_cosine_spectrum;shapes=[]
    def spectrum(A,**kwargs):
        shapes.append(A.shape);return original(A,**kwargs)
    monkeypatch.setattr(live,'direction_cosine_spectrum',spectrum)
    snap,policy,_,_=snapshot_fixture(monkeypatch)
    assert shapes==[(1,19,15)]
    assert snap.far_field_intensity.shape==(7,5)
    assert snap.array_bytes==policy.maximum_array_bytes
    assert PRLivePreviewPolicy().maximum_array_bytes==923648
    # Spectrum coordinates are samples of the ORIGINAL FFT grid, not a new FFT.
    ix=np.rint(np.linspace(0,18,7)).astype(int)
    np.testing.assert_array_equal(snap.s_x,(np.fft.fftshift(np.fft.fftfreq(19,2))*(.633/2.4))[ix])
