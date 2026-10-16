"""Synthetic pointer tests: no propagation or material solves."""
from types import SimpleNamespace
import sys
import subprocess
import os

import numpy as np
import pytest

from lcprop.notebook import interactive_lc as ui
from lcprop.notebook import _beam_position_editor as dragging


@pytest.fixture
def app(monkeypatch):
    import ipywidgets as W
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    def canvas():
        fig = Figure(figsize=(3.75, 3.75), layout='constrained')
        agg = FigureCanvasAgg(fig)
        widget = W.Box()
        widget.mpl_connect = agg.mpl_connect
        widget.mpl_disconnect = agg.mpl_disconnect
        widget.draw = agg.draw
        widget.draw_idle = agg.draw
        return fig, widget
    monkeypatch.setattr(dragging, 'widget_canvas', canvas)
    monkeypatch.setattr(ui, 'display_figure', lambda *a, **k: None)
    monkeypatch.setattr(ui, 'execute', lambda *a, **k: pytest.fail('Solver forbidden'))
    # Initialization and refresh use only canonical entrance sampling, no propagation.
    obj = ui.InteractiveLC()
    # Simulate the real widget initialization message for the Agg-backed test double.
    obj._position_editor._canvas_message(obj._position_editor.canvas, {'type': 'initialized'}, [])
    obj.live_preview.value = False  # Preserve marker-only Stage 2A coverage.
    yield obj
    obj.close()


def enable(app):
    assert app._position_editor is not None, app.editor_status.value
    return app._position_editor


def event(editor, x, y, button=1, inside=True):
    px, py = editor.ax.transData.transform((x, y))
    return SimpleNamespace(xdata=x, ydata=y, x=px, y=py, button=button,
                           inaxes=editor.ax if inside else None)


def drag(editor, start, end):
    editor.press(event(editor, *start))
    editor.motion(event(editor, *end))
    editor.release(event(editor, *end))


def test_release_atomic_and_canonical_parity(app, monkeypatch):
    e = enable(app)
    original = app.request()
    expected = ui.request_for_controls(base=original, x=-12.125, y=3.625)
    calls = []
    launch = ui.launch_preview
    monkeypatch.setattr(ui, 'launch_preview', lambda r: (calls.append(r), launch(r))[1])
    e.press(event(e, -20, 0))
    e.motion(event(e, -12.125, 3.625))
    assert app.request() == original and not calls
    e.release(event(e, -12.125, 3.625))
    assert len(calls) == 1 and app.request() == expected
    np.testing.assert_array_equal(launch(expected)[1].A0, launch(app.request())[1].A0)
    np.testing.assert_array_equal(launch(expected)[2], launch(app.request())[2])


@pytest.mark.parametrize('mode', ['coherent', 'incoherent'])
def test_overlap_explicit_second_selection(app, mode):
    from dataclasses import replace
    app.second_enabled.value = True
    app.beam2_controls['x0_um'].value = -20
    app.coherence.value = mode
    app.beam2_controls['phase_rad'].value = .7
    app.beam2_controls['psi_rad'].value = .2
    app.active_beam.value = 1
    e = enable(app)
    before = app.request()
    drag(e, (-20, 0), (-9.3, -5.1))
    expected = replace(before, beams=replace(before.beams, channels=(before.beams.channels[0],
        replace(before.beams.channels[1], x0_um=-9.3, y0_um=-5.1))))
    assert app.request() == expected
    np.testing.assert_array_equal(ui.launch_preview(expected)[1].A0, ui.launch_preview(app.request())[1].A0)
    assert app.active_beam.value == 1


def test_invalid_cancel_recovery_and_numeric_outside(app):
    e = enable(app)
    before = app.request()
    for target in [(1000, 0), (0, 1000), (float('nan'), 0)]:
        e.press(event(e, -20, 0))
        e.release(event(e, *target))
        assert app.request() == before and e.gesture is None
    e.press(event(e, -20, 0, button=3))
    assert e.gesture is None
    e.press(event(e, -20, 0))
    e.release(event(e, -10, 2, inside=False))
    assert app.request() == before
    drag(e, (-20, 0), (-10, 2))
    assert app.controls['y'].value == 2
    app.controls['x'].value = 100
    assert app.controls['x'].value == 100
    assert e.markers[0].get_xdata()[0] == 100


def test_mapping_after_resize_zoom_and_grab_offset(app):
    e = enable(app)
    e.figure.set_size_inches(6, 3)
    e.ax.set_xlim(-30, 10); e.ax.set_ylim(-10, 20)
    e.canvas.draw_idle()
    e.press(event(e, -19.9, .1))
    assert e.gesture
    e.release(event(e, -10.9, 4.1))
    assert app.controls['x'].value == pytest.approx(-11)
    assert app.controls['y'].value == pytest.approx(4)


@pytest.mark.parametrize('action', ['reset', 'run', 'load', 'close'])
def test_stale_canvas_ignored_and_snapshot_preserved(app, monkeypatch, action):
    from tests.test_interactive_lc_multibeam_presentation import snapshot
    saved = snapshot()
    app.completed = saved
    app.completed_summary.value = 'Previous completed summary'
    e = enable(app)
    e.press(event(e, -20, 0))
    if action == 'reset': app._restore()
    elif action == 'run':
        monkeypatch.setattr(ui, 'execute', lambda *a, **k: (_ for _ in ()).throw(TimeoutError('synthetic')))
        app._run()
    elif action == 'load':
        with pytest.raises(Exception): app.load('/no-such-experiment.lcprop.json')
    else: app.close()
    before = app.request()
    # Trailing release is cancelled; a complete later gesture may edit draft.
    e.release(event(e, -5, 3))
    assert app.request() == before
    drag(e, (-20, 0), (-5, 3))
    if action == 'close':
        assert app.request() == before and not e.active
    else:
        assert app.controls['x'].value == -5 and e.active
        assert app._position_editor is e
    assert app.completed is saved and app.completed_summary.value == 'Previous completed summary'


def test_pan_escape_disable_and_selection_cancel(app):
    app.second_enabled.value = True
    e = enable(app)
    e.canvas.toolbar = SimpleNamespace(mode='pan/zoom')
    e.press(event(e, -20, 0)); assert e.gesture is None
    e.canvas.toolbar.mode = ''
    e.press(event(e, -20, 0)); e.key(SimpleNamespace(key='escape')); assert e.gesture is None
    e.press(event(e, -20, 0)); app.active_beam.value = 1; assert e.gesture is None
    app.second_enabled.value = False
    assert app.active_beam.options == (('Beam 1', 0),)
    assert len(e.markers) == 1


def test_export_snapshot_identity(app, tmp_path):
    from tests.test_interactive_lc_multibeam_presentation import snapshot
    saved = snapshot()
    app.completed = saved
    ui.export_completed(saved, tmp_path/'before')
    e = enable(app); drag(e, (-20, 0), (-15, 2))
    ui.export_completed(app.completed, tmp_path/'after')
    for name in ('arrays.npz', 'experiment.lcprop.json'):
        assert (tmp_path/'before'/name).read_bytes() == (tmp_path/'after'/name).read_bytes()


def test_missing_optional_backend_static_fallback(app, monkeypatch):
    def missing(): raise ImportError('ipympl missing')
    monkeypatch.setattr(dragging, 'widget_canvas', missing)
    app = ui.InteractiveLC()
    assert app._position_editor is None
    assert 'Numeric controls and static preview remain available' in app.editor_status.value
    assert app.preview_host.children == (app._preview_image,)


def test_core_static_imports_do_not_need_optional_backend():
    code = '''
import sys
for name in ('ipympl','PySide6','launchplane','cupy'):
    sys.modules[name] = None
import lcprop
from lcprop.notebook import InteractiveLC
import lcprop.notebook.interactive_lc as ui
ui.launch_preview = lambda *a: (_ for _ in ()).throw(ValueError('synthetic invalid draft'))
a = InteractiveLC()
assert a._position_editor is None
assert sys.modules['ipympl'] is None
'''
    subprocess.run([sys.executable, '-B', '-c', code], env={**os.environ, 'MPLBACKEND':'Agg'}, check=True)


def test_backend_requires_explicit_activation():
    import matplotlib
    if matplotlib.get_backend().lower() != 'agg':
        pytest.skip('Agg isolation test')
    with pytest.raises(RuntimeError, match='First run %matplotlib widget'):
        dragging.widget_canvas()
    assert matplotlib.get_backend().lower() == 'agg'


def test_disposal_never_reconnects_old_events(app):
    e = enable(app)
    app.close()
    before = app.request()
    drag(e, (-20, 0), (-10, 2))
    app._run()
    app._restore()
    app.load('/not-read')
    assert app.request() == before and not e.active
