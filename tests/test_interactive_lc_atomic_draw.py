"""A committed draft requests rendering only after intensity and geometry agree."""
import numpy as np
import pytest
from tests.test_interactive_lc_dragging import app, event
from lcprop.notebook import interactive_lc as ui


def test_committed_toggle_has_no_intermediate_stale_draw(app, monkeypatch):
    e=app._position_editor
    seen=[]
    def observe():
        canonical=ui.launch_preview(app.request())[2]
        seen.append((len(e.markers), np.array_equal(e.mesh.get_array(),canonical.T)))
    monkeypatch.setattr(e.canvas,'draw_idle',observe)
    for value in (True,False,True):
        seen.clear()
        app.second_enabled.value=value
        assert seen==[(2 if value else 1,True)]
    app.active_beam.value=1
    seen.clear()
    app.second_enabled.value=False
    assert app.active_beam.value==0
    assert seen==[(1,True)]


@pytest.mark.parametrize('action',['Move','Tilt'])
def test_release_only_publishes_final_draft(app,monkeypatch,action):
    from lcprop.notebook._beam_position_editor import ray_endpoint
    e=app._position_editor
    app.action.value=action
    beam=app.request().beams.channels[0]
    start=(beam.x0_um,beam.y0_um) if action=='Move' else ray_endpoint(beam,app.request())
    e.press(event(e,*start))
    e.motion(event(e,start[0]+1,start[1]+1))
    seen=[]
    monkeypatch.setattr(e.canvas,'draw_idle',lambda: seen.append(app.request()))
    e.release(event(e,start[0]+1,start[1]+1))
    assert seen==[app.request()]
    np.testing.assert_array_equal(e.mesh.get_array(),ui.launch_preview(app.request())[2].T)
