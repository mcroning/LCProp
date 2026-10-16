"""One launch-preview widget slot, without callback display/output capture."""
import numpy as np
import pytest
from lcprop.notebook import interactive_lc as ui
from tests.test_interactive_lc_dragging import app, enable, drag


def test_persistent_child_no_display(app,monkeypatch):
    import IPython.display
    monkeypatch.setattr(IPython.display,'display',lambda *a,**k:pytest.fail('Unexpected publication'))
    host=app.preview_host
    e=enable(app)
    canvas=e.canvas
    for i in range(4):
        app.controls['x'].value=-19.+i
        app._restore()
        assert e.canvas is canvas and host.children==(canvas,)
        assert e.active and app._position_editor is e
    assert not hasattr(app,'graphical')
    assert app.configure.children[0].children.count(host)==1


def test_numeric_static_reuses_image_and_figure(monkeypatch):
    from lcprop.notebook import _beam_position_editor as editor
    monkeypatch.setattr(editor,'widget_canvas',lambda: (_ for _ in ()).throw(ImportError('no ipympl')))
    app=ui.InteractiveLC()
    image=app._preview_image;fig=app._static_figure;data=bytes(image.value)
    app.controls['x'].value=-12.5
    assert app.preview_host.children==(image,) and app._static_figure is fig
    assert bytes(image.value)!=data
    np.testing.assert_array_equal(app._static_mesh.get_array(),ui.launch_preview(app.request())[2].T)
    app.close()


@pytest.mark.parametrize('size',[(3.2,3.75),(3.75,3.75),(4.5,4.)])
def test_interactive_labels_inside_and_colorbar_separate(app,size):
    e=enable(app);e.figure.set_size_inches(*size)
    from matplotlib.backend_bases import ResizeEvent
    ResizeEvent('resize_event',e.figure.canvas)._process()
    e.canvas.draw_idle()
    renderer=e.figure.canvas.get_renderer()
    bounds=e.figure.bbox
    for label in (e.ax.title,e.ax.xaxis.label,e.ax.yaxis.label,e.colorbar.ax.yaxis.label):
        box=label.get_window_extent(renderer)
        assert bounds.x0<=box.x0 and box.x1<=bounds.x1
        assert bounds.y0<=box.y0 and box.y1<=bounds.y1
    assert e.ax.get_window_extent(renderer).x1<e.colorbar.ax.get_window_extent(renderer).x0


def test_mode_switch_preserves_completed_exports(app,tmp_path):
    from tests.test_interactive_lc_multibeam_presentation import snapshot
    saved=snapshot();app.completed=saved;app.completed_summary.value='unchanged'
    ui.export_completed(saved,tmp_path/'before')
    for action in ('Move','Tilt'):
        app.action.value=action
        e=enable(app)
        start=(app.controls['x'].value,app.controls['y'].value)
        drag(e,start,(start[0]+1,start[1]+1))
        app._restore()
    ui.export_completed(app.completed,tmp_path/'after')
    assert app.completed is saved and app.completed_summary.value=='unchanged'
    for name in ('arrays.npz','experiment.lcprop.json','summary.json'):
        assert (tmp_path/'before'/name).read_bytes()==(tmp_path/'after'/name).read_bytes()
