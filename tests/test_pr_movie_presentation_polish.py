"""Display-only style and retained movie fixtures."""
from dataclasses import replace
import hashlib
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication,QStyle,QStyleOptionSpinBox,QStyleFactory
from tests.test_synchronized_fields import app
from tests.test_pr_standard_movies import artifact
from lcprop.gui.workspace import Workspace
from lcprop.products.data_model import RunData,Geometry
from lcprop.gui.panels.input_screen_editor import ScreenSpinBox


@pytest.mark.parametrize('points',[10,18,26])
def test_measured_edit_rectangle(app,points):
    for name in QStyleFactory.keys():
        w=ScreenSpinBox();w.setStyle(QStyleFactory.create(name));f=w.font();f.setPointSize(points);w.setFont(f)
        w.setRange(-1e7,1e7);w.setDecimals(6);w.setValue(123.456789);w.resize(w.sizeHint());w.show();app.processEvents()
        opt=QStyleOptionSpinBox();w.initStyleOption(opt)
        rect=w.style().subControlRect(QStyle.ComplexControl.CC_SpinBox,opt,QStyle.SubControl.SC_SpinBoxEditField,w)
        assert rect.height()>=w.lineEdit().sizeHint().height(),(name,points,rect)
        assert w.value()==123.456789
        w.close()


def test_movie_orientation_labels_scales_and_hashes(app):
    a=artifact();a=replace(a,key='td_trajectory',display_name='Optical trajectory',metadata={**a.metadata,'axes':['x','z'],'coordinates':{'x':[-2.,2.],'z':[0.,10.]}})
    before=hashlib.sha256(a.data.tobytes()).hexdigest();w=Workspace();p=w.trajectory_player
    p.set_run_data(RunData(workflow='pr_timedependent',geometry=Geometry(),artifacts={a.key:a}))
    p.slider.setValue(1);p.show_frame(1);v=w.image_pane.image_view;limits=v.image.get_clim()
    assert v.ax.get_xlabel().startswith('z')
    np.testing.assert_array_equal(v.image.get_array(),a.data[1])
    p.orientation.setCurrentIndex(1)
    assert v.ax.get_ylabel().startswith('z')
    np.testing.assert_array_equal(v.image.get_array(),a.data[1].T)
    assert v.image.get_clim()==limits and 'accepted τ=3' in p.label.text()
    assert 'trajectory' not in p.label.text().lower()
    assert p.restore.text()=='Return to spatial field views'
    assert hashlib.sha256(a.data.tobytes()).hexdigest()==before
    w.close()


def test_angular_group_colorbar_stays_adjacent(app):
    a=artifact();base=replace(a,key='td_trajectory');w=Workspace();p=w.trajectory_player
    p.set_run_data(RunData(workflow='pr_timedependent',geometry=Geometry(),artifacts={base.key:base}))
    p.show_frame(0);v=w.image_pane.image_view;v.resize(1200,500);v.show();app.processEvents();v.draw()
    image=v.ax.get_window_extent();bar=v.colorbar.ax.get_window_extent()
    assert abs(image.height-bar.height)<2
    assert 0<bar.x0-image.x1<40
    assert not p.orientation.isEnabled()
    np.testing.assert_array_equal(v._field.coordinates['s_x'],a.metadata['coordinates']['s_x'])
    np.testing.assert_array_equal(v.image.get_array(),a.data[0].T)
    w.close()
