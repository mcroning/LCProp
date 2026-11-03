"""Presentation checks; coordinate-coded retained arrays, no scientific run."""
from dataclasses import replace
import hashlib
from types import SimpleNamespace
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QSizePolicy
from tests.test_synchronized_fields import app, fixture, select
from lcprop.gui.workspace import Workspace
from lcprop.gui.field_selection import independent_plane_label
from lcprop.gui.panels.input_screen_editor import InputScreenEditor
from lcprop.gui.request_transparency import report_failure
from lcprop.pr.gui.app import configure_application_font


def test_td_default_and_explicit_selection_survive_refresh(app):
    r=fixture();f=replace(r.fields['psi'],key='optical_intensity_stack',quantity='physical_optical_intensity',kind='intensity')
    r.fields.add(f.key,f)
    before=hashlib.sha256(f.data.tobytes()).hexdigest()
    w=Workspace();w.set_run_data(r)
    assert w.spatial_selector.currentData()==['volume',f.key]
    select(w,'plane','far');w.spatial_selector.activated.emit(w.spatial_selector.currentIndex())
    w.set_run_data(r)
    assert w.spatial_selector.currentData()==['plane','far']
    assert w.longitudinal_pane.isHidden()
    assert hashlib.sha256(f.data.tobytes()).hexdigest()==before
    w.close()


def test_responsive_layout_and_movie_selector(app):
    w=Workspace();w.set_run_data(fixture());w.resize(820,850);w.show();app.processEvents()
    assert w.fields_splitter.orientation()==Qt.Orientation.Vertical
    assert not w.fields_splitter.childrenCollapsible()
    assert w.trajectory_player.quantity.minimumContentsLength()>=32
    assert w.trajectory_player.quantity.width()>300
    w.resize(1400,1000);app.processEvents()
    assert w.fields_splitter.orientation()==Qt.Orientation.Horizontal
    select(w,'plane','far');app.processEvents()
    assert w.longitudinal_pane.isHidden()
    w.close()


def test_plane_coordinate_not_inherited_from_linked_slider(app):
    r=fixture();plane=replace(r.fields['far'],key='optical_intensity_xy',axes=('x','y'),coordinates={'x':r.geometry.x,'y':r.geometry.y,'z':2.5})
    r.fields.add(plane.key,plane)
    w=Workspace();w.set_run_data(r);select(w,'volume','psi');w.longitudinal_pane.z_plane_slider.setValue(2)
    assert 'z=7' in w.image_pane.image_view._field.display_name
    select(w,'plane',plane.key)
    assert w.image_pane._z_index is None
    assert 'z=2.5' in w.image_pane.image_view._field.display_name
    assert 'Retained optical intensity plane' in independent_plane_label(plane)
    assert 'unspecified' in independent_plane_label(replace(plane,coordinates={}))
    w.close()


def test_spinboxes_follow_font_and_style(app):
    for make in (InputScreenEditor._length_control,InputScreenEditor._position_control):
        w=make(1.25);before=w.value();h=w.sizeHint().height();f=w.font();f.setPointSize(f.pointSize()+10);w.setFont(f)
        assert w.sizePolicy().verticalPolicy()==QSizePolicy.Policy.Fixed
        assert w.sizeHint().height()>h
        assert w.value()==before


def test_font_fallback_and_nyquist_details(app):
    original=app.font()
    try:
        f=QFont(original);f.setFamily('nonexistent-LCProp-test-font');app.setFont(f)
        configure_application_font(app)
        assert app.font().family() in QFontDatabase.families()
    finally:app.setFont(original)
    w=Workspace();window=SimpleNamespace(results_panel=SimpleNamespace(workspace=w))
    details='Invalid request: Nyquist limit exceeded\nfull beam/grid detail 123456'
    report_failure(window,details)
    assert details in w.console.toPlainText()
    # The visible banner keeps the cause, not the final incidental detail line.
    labels=[x.text() for x in w.findChildren(__import__('PySide6.QtWidgets',fromlist=['QLabel']).QLabel)]
    assert any('Nyquist' in x and '123456' not in x for x in labels)
    w.close()
