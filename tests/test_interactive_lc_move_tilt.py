"""Graphical draft requests only: no material or propagation execution."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from lcprop.notebook import interactive_lc as ui
from lcprop.notebook import _beam_position_editor as editor
from lcprop.optics.physical_launch import external_direction_for_exit, resolve_beam_geometry
from tests.test_interactive_lc_dragging import app, enable, event, drag


@pytest.mark.parametrize('theta,phi', [(0.,1.2),(-0.,2.),(.01,.3),(-.01,.3),(.02,3.4)])
def test_canonical_forward_inverse(theta,phi):
    request=ui.build_request()
    beam=replace(request.beams.channels[0],theta_ext_rad=theta,phi_rad=phi)
    point=editor.ray_endpoint(beam,request)
    g=resolve_beam_geometry(beam,request.material.no)
    assert point==(beam.x0_um+request.grid.z_length_um*g.kx/g.kz_internal,
                   beam.y0_um+request.grid.z_length_um*g.ky/g.kz_internal)
    actual=external_direction_for_exit(beam,request.material.no,request.grid.z_length_um,*point)
    assert actual[0]==pytest.approx(theta,abs=1e-15)
    assert actual[1]==pytest.approx(phi,abs=1e-15)
    if theta==0: assert np.signbit(actual[0])==np.signbit(theta)


def test_selectors_and_arrow_highlight(app):
    assert app.active_beam.description=='Active Beam:'
    assert app.action.options==('Move','Tilt') and app.action.value=='Move'
    app.second_enabled.value=True
    e=enable(app)
    assert len(e.arrows)==2
    assert e.arrows[0].arrow_patch.get_linewidth()>e.arrows[1].arrow_patch.get_linewidth()
    app.active_beam.value=1
    assert e.arrows[1].arrow_patch.get_linewidth()>e.arrows[0].arrow_patch.get_linewidth()


@pytest.mark.parametrize('mode',['coherent','incoherent'])
def test_tilt_request_and_launch_parity(app,monkeypatch,mode):
    app.second_enabled.value=True
    app.coherence.value=mode
    app.beam2_controls['psi_rad'].value=.23
    app.beam2_controls['phase_rad'].value=.4
    app.active_beam.value=1
    app.action.value='Tilt'
    e=enable(app)
    before=app.request()
    start=editor.ray_endpoint(before.beams.channels[1],before)
    end=(start[0]-3.75,start[1]+2.125)
    expected=editor.gesture_request(before,1,'Tilt',*end)
    calls=[]; launch=ui.launch_preview
    monkeypatch.setattr(ui,'launch_preview',lambda r:(calls.append(r),launch(r))[1])
    e.press(event(e,*start)); e.motion(event(e,*end))
    assert app.request()==before and not calls
    e.release(event(e,*end))
    assert len(calls)==1 and app.request()==expected
    a,b=before.beams.channels[1],expected.beams.channels[1]
    assert replace(b,theta_ext_rad=a.theta_ext_rad,phi_rad=a.phi_rad)==a
    np.testing.assert_array_equal(launch(app.request())[1].A0,launch(expected)[1].A0)
    np.testing.assert_array_equal(launch(app.request())[2],launch(expected)[2])


def test_unreachable_inverse_rejected():
    request=ui.build_request()
    request=replace(request,grid=replace(request.grid,z_length_um=.001))
    with pytest.raises(ValueError,match='unreachable'):
        editor.gesture_request(request,0,'Tilt',10.,10.)


def test_live_throttle_bound_and_final_refresh(app,monkeypatch):
    app.live_preview.value=True
    e=enable(app)
    now=[0.];e.clock=lambda:now[0]
    calls=[];launch=ui.launch_preview
    monkeypatch.setattr(ui,'launch_preview',lambda r:(calls.append(r),launch(r))[1])
    before=app.request()
    mesh=e.mesh
    e.press(event(e,-20,0))
    for _ in range(100):e.motion(event(e,-18,1))
    assert not calls
    for k in range(25):
        now[0]+=.3
        e.motion(event(e,-18+k*.01,1))
    assert len(calls)==20 and e.live_disabled
    assert app.request()==before and e.mesh is mesh
    e.release(event(e,-15,2))
    assert len(calls)==21 and app.controls['x'].value==-15


def test_slow_live_fallback_cancel_and_resume(app,monkeypatch):
    app.live_preview.value=True
    e=enable(app)
    now=[0.];e.clock=lambda:now[0]
    calls=[]; launch=ui.launch_preview
    def slow(r):
        calls.append(r);now[0]+=.11
        return launch(r)
    monkeypatch.setattr(ui,'launch_preview',slow)
    original=e.mesh.get_array().copy()
    before=app.request()
    e.press(event(e,-20,0));now[0]=.3;e.motion(event(e,-17,1))
    assert len(calls)==1 and e.live_disabled
    now[0]=2;e.motion(event(e,-16,1));assert len(calls)==1
    e.key(SimpleNamespace(key='escape'))
    assert app.request()==before
    np.testing.assert_array_equal(e.mesh.get_array(),original)
    e.press(event(e,-20,0));assert not e.live_disabled


def test_invalid_live_does_not_commit_or_lose_snapshot(app,monkeypatch):
    from tests.test_interactive_lc_multibeam_presentation import snapshot
    saved=snapshot();app.completed=saved
    app.live_preview.value=True
    app.action.value='Tilt'
    e=enable(app)
    now=[0.];e.clock=lambda:now[0]
    before=app.request()
    original=e.mesh.get_array().copy()
    def reject(r):raise ValueError('synthetic sampling rejection')
    monkeypatch.setattr(ui,'launch_preview',reject)
    e.press(event(e,-20,0));now[0]=.3;e.motion(event(e,-18,1))
    assert e.live_disabled and app.request()==before and app.completed is saved
    e.cancel()
    np.testing.assert_array_equal(e.mesh.get_array(),original)


def test_action_change_and_retired_tilt_events(app):
    e=enable(app)
    e.press(event(e,-20,0))
    app.action.value='Tilt'
    assert e.gesture is None
    e.press(event(e,-20,0))
    before=app.request()
    app.close()
    e.motion(event(e,-18,2));e.release(event(e,-18,2))
    assert app.request()==before


def test_zero_crossing_retains_azimuth_and_changes_signed_theta():
    request=ui.build_request()
    beam=replace(request.beams.channels[0],theta_ext_rad=.01,phi_rad=.2)
    tip=editor.ray_endpoint(beam,request)
    theta,phi=external_direction_for_exit(beam,request.material.no,request.grid.z_length_um,
        2*beam.x0_um-tip[0],2*beam.y0_um-tip[1])
    assert theta==pytest.approx(-.01) and phi==pytest.approx(.2)


def test_unreachable_tilt_release_preserves_numeric_controls(app):
    app.beam1_controls['n_ext'].value=.001
    app.action.value='Tilt'
    e=enable(app)
    before=app.request()
    drag(e,(-20,0),(-19,1))
    assert app.request()==before
    assert 'Gesture rejected' in app.editor_status.value


def test_live_tilt_final_refresh_and_export(app,monkeypatch,tmp_path):
    from tests.test_interactive_lc_multibeam_presentation import snapshot
    saved=snapshot();app.completed=saved
    app.action.value='Tilt';app.live_preview.value=True
    e=enable(app);now=[0.];e.clock=lambda:now[0]
    launch=ui.launch_preview;calls=[]
    monkeypatch.setattr(ui,'launch_preview',lambda r:(calls.append(r),launch(r))[1])
    before=app.request()
    e.press(event(e,-20,0));now[0]=.3;e.motion(event(e,-18,1))
    assert len(calls)==1 and app.request()==before
    e.release(event(e,-17,2))
    assert len(calls)==2
    assert app.request()==editor.gesture_request(before,0,'Tilt',-17,2)
    ui.export_completed(app.completed,tmp_path/'after')
    assert (tmp_path/'after/arrays.npz').read_bytes()==saved.arrays_npz
    assert app.completed is saved
