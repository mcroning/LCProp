"""Committed redraw and fixed launch geometry; entrance sampling only."""
import numpy as np
import pytest
from lcprop.notebook import interactive_lc as ui
from tests.test_interactive_lc_dragging import app, event


def geometry(e):
    return np.concatenate([e.figure.get_size_inches(), e.ax.get_xlim(), e.ax.get_ylim(),
        e.ax.get_position().bounds, e.colorbar.ax.get_position().bounds])


def test_enable_disable_and_radius_commit_draw_without_pointer(app,monkeypatch):
    e=app._position_editor
    draws=[]
    draw=e.canvas.draw_idle
    monkeypatch.setattr(e.canvas,'draw_idle',lambda: (draws.append(len(e.markers)),draw())[1])
    for enabled in (True,False,True):
        before=len(draws)
        app.second_enabled.value=enabled
        assert len(draws)>before and draws[-1]==(2 if enabled else 1)
    for c in (*app.controls.values(),*app.beam1_controls.values(),*app.beam2_controls.values()):
        assert c.continuous_update is False
    before=len(draws)
    app.beam2_controls['w1_um'].value=13.5  # One committed trait update, not keystrokes.
    assert len(draws)==before+1  # one completed draft, no cancellation redraw
    np.testing.assert_array_equal(e.mesh.get_array(),ui.launch_preview(app.request())[2].T)
    assert app.preview_host.children==(e.canvas,)


@pytest.mark.parametrize('action',['Move','Tilt'])
def test_drag_geometry_stable_with_dynamic_scaling(app,action):
    from lcprop.notebook._beam_position_editor import ray_endpoint
    e=app._position_editor
    app.second_enabled.value=True
    app.action.value=action
    app.live_preview.value=True
    now=[0.]
    e.clock=lambda:now[0]
    original=geometry(e)
    objects=(e.figure,e.canvas,e.ax,e.colorbar,e.mesh)
    for repeat in range(3):
        request=app.request();beam=request.beams.channels[0]
        start=(beam.x0_um,beam.y0_um) if action=='Move' else ray_endpoint(beam,request)
        e.press(event(e,*start))
        for delta in (.5,1.,1.5):
            now[0]+=1
            e.motion(event(e,start[0]+delta,start[1]+delta))
            e.canvas.draw()
            np.testing.assert_allclose(geometry(e),original,rtol=0,atol=1e-12)
        e.release(event(e,start[0]+1.5,start[1]+1.5))
        assert objects==(e.figure,e.canvas,e.ax,e.colorbar,e.mesh)
        np.testing.assert_allclose(geometry(e),original,rtol=0,atol=1e-12)
    old=e.mesh.get_clim()
    app.controls['w1'].value=4
    assert e.mesh.get_clim()!=old
    np.testing.assert_allclose(geometry(e),original,rtol=0,atol=1e-12)
