"""Shared static/editor launch presentation, without propagation."""
import io
import subprocess
import sys
import os

import numpy as np
import pytest
from PIL import Image
from lcprop.notebook import interactive_lc as ui
from lcprop.notebook import _beam_position_editor as editor
from tests.test_interactive_lc_dragging import app


@pytest.mark.parametrize('mode', ['preserve', 'coherent', 'incoherent'])
def test_static_interactive_pixels_geometry_and_snapshot(app, monkeypatch, mode, tmp_path):
    from lcprop.optics.physical_launch import resolve_beam_geometry
    from tests.test_interactive_lc_multibeam_presentation import snapshot
    monkeypatch.setattr(editor, 'widget_canvas', lambda: (_ for _ in ()).throw(ImportError('absent')))
    static = ui.InteractiveLC()
    try:
        saved = snapshot()
        static.completed = saved
        ui.export_completed(saved, tmp_path/'before')
        for obj in (app, static):
            obj.second_enabled.value = True
            obj.coherence.value = mode
            obj.beam2_controls['x0_um'].value = -10.
            obj.beam2_controls['theta_ext_rad'].value = .02
            obj.beam2_controls['phi_rad'].value = -.4
            obj.beam2_controls['phase_rad'].value = .7
            obj.beam2_controls['w1_um'].value = 12.
            obj.active_beam.value = 1
        assert static.request() == app.request()
        a, b = static._static_preview, app._position_editor
        expected = ui.launch_preview(static.request())[2].T
        for view in (a,b):
            np.testing.assert_array_equal(view.mesh.get_array(), expected)
            assert len(view.markers) == 2
            for i,ch in enumerate(static.request().beams.channels):
                np.testing.assert_array_equal(view.markers[i].get_data(), [[ch.x0_um],[ch.y0_um]])
                g = resolve_beam_geometry(ch, static.request().material.no)
                tip = (ch.x0_um + 500*g.kx/g.kz_internal, ch.y0_um+500*g.ky/g.kz_internal)
                np.testing.assert_allclose(view.arrows[i].xy, tip, rtol=0, atol=1e-14)
            assert view.arrows[1].arrow_patch.get_linewidth() == 2.5
            assert view.arrows[0].arrow_patch.get_linewidth() == 1
            view.figure.canvas.draw()
        # The Agg widget double performs an extra initial constrained-layout pass.
        # Normalize only backend layout boxes; data, norms, artists and physical
        # axis limits remain untouched. Then require pixel-exact rendering.
        for ax_a, ax_b in zip(a.figure.axes, b.figure.axes):
            np.testing.assert_array_equal(ax_a.get_xlim(), ax_b.get_xlim())
            np.testing.assert_array_equal(ax_a.get_ylim(), ax_b.get_ylim())
            ax_b.set_position(ax_a.get_position())
        b.figure.canvas.draw()
        np.testing.assert_array_equal(a.figure.canvas.buffer_rgba(), b.figure.canvas.buffer_rgba())
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(static._preview_image.value))), a.figure.canvas.buffer_rgba())
        static.active_beam.value = 0
        assert a.arrows[0].arrow_patch.get_linewidth() == 2.5
        assert static.completed is saved
        ui.export_completed(static.completed, tmp_path/'after')
        for name in ('arrays.npz','summary.json','experiment.lcprop.json'):
            assert (tmp_path/'before'/name).read_bytes() == (tmp_path/'after'/name).read_bytes()
        static.second_enabled.value = False
        assert len(a.markers) == 1
        assert static.preview_host.children == (static._preview_image,)
    finally:
        static.close()


def test_static_no_optional_backend_or_propagation():
    script = '''
import sys
from lcprop.notebook import interactive_lc as ui
ui.execute = lambda *a, **k: (_ for _ in ()).throw(AssertionError('Propagation'))
a = ui.InteractiveLC()
a.second_enabled.value = True
a.beam2_controls['theta_ext_rad'].value = .01
a.active_beam.value = 1
assert a._position_editor is None
assert len(a._static_preview.arrows) == 2
assert not any(n.split('.')[0] in ('ipympl','PySide6','cupy','launchplane') for n in sys.modules)
a.close()
'''
    subprocess.run([sys.executable,'-c',script], env={**os.environ,'MPLBACKEND':'Agg'},check=True,capture_output=True)
