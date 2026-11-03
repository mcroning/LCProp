"""Regression for the superseded forced-refresh fix: use frontend scheduling."""
import numpy as np
from tests.test_interactive_lc_frame_delivery import recorded, compose


def test_committed_edits_defer_transmission_to_frontend(recorded, monkeypatch):
    r = recorded
    canvas = r.canvas
    def forbidden(*args, **kwargs):
        raise AssertionError('No forced refresh or synchronous draw during edit')
    for control, value in ((r.app.second_enabled, True),
                           (r.app.beam2_controls['w1_um'], 12.),
                           (r.app.second_enabled, False),
                           (r.app.second_enabled, True)):
        first = len(r.frames)
        event_start = len(r.events)
        with monkeypatch.context() as m:
            m.setattr(canvas, 'draw', forbidden)
            m.setattr(canvas.manager, 'handle_json', forbidden)
            control.value = value
        assert len(r.frames) == first
        assert any(e['direction']=='out' and e['type']=='draw' for e in r.events[event_start:])
        r.dispatch('draw')
        assert len(r.frames) > first
        np.testing.assert_array_equal(compose(r.frames), r.frames[-1]['rgba'])
        np.testing.assert_array_equal(r.editor.mesh.get_array(), r.ui.launch_preview(r.app.request())[2].T)
        assert r.canvas is canvas
