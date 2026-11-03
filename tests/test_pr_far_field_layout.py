"""Angular layout is field-owned, independent of movie/spatial callers."""
from dataclasses import replace
import hashlib
import numpy as np
import pytest
from tests.test_synchronized_fields import app, fixture, select
from tests.test_pr_standard_movies import artifact
from lcprop.gui.workspace import Workspace
from lcprop.products.data_model import FieldData


@pytest.mark.parametrize('size',[(800,750),(1200,800),(1600,1000)])
def test_far_field_and_movie_layout_transitions(app,size):
    r=fixture();a=artifact();r=replace(r,artifacts={'td_trajectory':replace(a,key='td_trajectory')})
    f=FieldData('angular','Far field',a.data[0],('s_x','s_y'),'intensity',
                coordinates={k:np.asarray(v) for k,v in a.metadata['coordinates'].items()})
    r.fields.add(f.key,f)
    before=hashlib.sha256(a.data.tobytes()).hexdigest()
    w=Workspace();w.resize(*size);w.set_run_data(r);w.show();app.processEvents()
    v=w.image_pane.image_view;p=w.trajectory_player
    def check():
        app.processEvents();v.draw()
        image=v.ax.get_window_extent();bar=v.colorbar.ax.get_window_extent()
        width,height=v.figure.bbox.width,v.figure.bbox.height;pt=v.figure.dpi/72.
        assert bar.x0-image.x1==pytest.approx(14*pt,abs=1)
        assert bar.height==pytest.approx(image.height,abs=1)
        # Group centered in plotting area after reserving label gutters.
        assert (image.x0+bar.x1)/2==pytest.approx((52*pt+width-55*pt)/2,abs=1)
        assert image.width/image.height==pytest.approx(2.)
        assert image.height==pytest.approx(min(height-88*pt,(width-130*pt)/2),abs=1)
        assert w.longitudinal_pane.isHidden()
    select(w,'plane','angular');check()
    limits=v.image.get_clim();np.testing.assert_array_equal(v.image.get_array(),a.data[0].T)
    p.show_frame(0);check();global_limits=v.image.get_clim()
    p.slider.setValue(1);check();assert v.image.get_clim()==global_limits
    p.scale.setCurrentIndex(1);check()
    p.restore_endpoint();check();assert v.image.get_clim()==limits
    select(w,'volume','psi');app.processEvents();assert not w.longitudinal_pane.isHidden()
    p.show_frame(0);check();p.restore_endpoint();app.processEvents();assert not w.longitudinal_pane.isHidden()
    assert hashlib.sha256(a.data.tobytes()).hexdigest()==before
    np.testing.assert_array_equal(f.coordinates['s_x'],a.metadata['coordinates']['s_x'])
    w.close()
