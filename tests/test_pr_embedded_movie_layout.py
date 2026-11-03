from dataclasses import replace
import numpy as np
import pytest
from tests.test_synchronized_fields import app,fixture
from tests.test_pr_standard_movies import artifact
from lcprop.gui.workspace import Workspace


@pytest.mark.parametrize('axes',[('x','y'),('s_x','s_y')])
@pytest.mark.parametrize('size',[(800,750),(1400,950)])
def test_movie_compact_group_aspect_and_restoration(app,axes,size):
    a=artifact();a=replace(a,key='td_trajectory',metadata={**a.metadata,'axes':axes,
         'coordinates':{axes[0]:[-2.,2.],axes[1]:[-1.,1.]}})
    original=a.data.copy();r=replace(fixture(),artifacts={a.key:a});w=Workspace();w.resize(*size);w.set_run_data(r);w.show()
    p=w.trajectory_player;p.show_frame(0);v=w.image_pane.image_view
    app.processEvents();v.draw();image=v.ax.get_window_extent();bar=v.colorbar.ax.get_window_extent();pt=v.figure.dpi/72.
    assert image.width/image.height==pytest.approx(2.)
    assert bar.x0-image.x1==pytest.approx(14*pt,abs=1)
    assert bar.height==pytest.approx(image.height,abs=1)
    assert (image.x0+bar.x1)/2==pytest.approx((v.figure.bbox.width-3*pt)/2,abs=1)
    limits=v.image.get_clim();p.slider.setValue(1);assert v.image.get_clim()==limits
    p.restore_endpoint();assert not v._time_evolution_layout
    np.testing.assert_array_equal(a.data,original);w.close()


def test_export_title_only_change(monkeypatch,tmp_path):
    import lcprop.gui.movie_export as export
    from matplotlib.axes import Axes
    from types import SimpleNamespace
    a=replace(artifact(),display_name='Optical Trajectory');before=a.data.copy();titles=[]
    original=Axes.set_title
    def capture(self,label,*args,**kwargs):
        titles.append(label);return original(self,label,*args,**kwargs)
    monkeypatch.setattr(Axes,'set_title',capture)
    monkeypatch.setattr(export.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout=b'mp4'))
    export.save_movie(a,tmp_path/'test.mp4',ffmpeg_path='/mock/ffmpeg')
    assert len(titles)==2 and all('Time evolution' in t and 'Trajectory' not in t for t in titles)
    assert 'accepted ' in titles[-1] and '$=3' in titles[-1]
    np.testing.assert_array_equal(a.data,before)
