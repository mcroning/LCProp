from dataclasses import replace
from types import SimpleNamespace
import warnings
import numpy as np
from lcprop.gui.scientific_labels import scientific_text,unit_label,axis_label
from tests.test_pr_standard_movies import artifact


def test_label_semantics_and_existing_mathtext():
    for spelling in ('1/um^2','1/µm²','1/μm²'):
        assert unit_label(spelling)==r'$1/(\mu\mathrm{m})^{2}$'
    assert unit_label('log10 1/um^2')==r'$\log_{10}$ $1/(\mu\mathrm{m})^{2}$'
    assert scientific_text(r'$\psi$')==r'$\psi$'
    assert axis_label('s_x',{'s_x':'1'})==r'$s_x$ (1)'
    assert scientific_text('ψ τ')==r'$\psi$ $\tau$'


def test_real_exported_frames_render_scientific_labels(monkeypatch,tmp_path):
    import lcprop.gui.movie_export as export
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    a=artifact();a=replace(a,display_name='ψ Trajectory',metadata={**a.metadata,'value_unit':'1/um^2','axes':['x','y'],'coordinates':{'x':[-1.,1.],'y':[-1.,1.]}})
    original=a.data.copy();draw=FigureCanvasAgg.draw;rendered=[]
    def capture(canvas,*args,**kwargs):
        result=draw(canvas,*args,**kwargs)
        figure=canvas.figure;ax,bar=figure.axes
        assert bar.get_ylabel()==unit_label('1/um^2')
        assert r'$\tau$' in ax.get_title() and 'Time evolution' in ax.get_title()
        assert r'$\psi$' in ax.get_title()
        assert r'$\mu\mathrm{m}$' in ax.get_xlabel()
        rendered.append(np.asarray(canvas.buffer_rgba()).copy())
        return result
    monkeypatch.setattr(FigureCanvasAgg,'draw',capture)
    def encode(command,**kwargs):
        assert len(kwargs['input'])==2*640*480*3
        return SimpleNamespace(stdout=b'mp4')
    monkeypatch.setattr(export.subprocess,'run',encode)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always');export.save_movie(a,tmp_path/'labels.mp4',ffmpeg_path='/mock/ffmpeg')
    assert len(rendered)==2 and all(frame.std()>0 for frame in rendered)
    assert not any('Glyph' in str(w.message) for w in recorded)
    np.testing.assert_array_equal(a.data,original)
    assert a.metadata['value_unit']=='1/um^2'
