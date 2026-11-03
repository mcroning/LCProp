"""Input-face intent must survive the real PR worker/result/view lifecycle."""
import math
from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from launchplane.model import BeamDefinition, BeamStackDefinition
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.adapters.launchplane import beam_stack_definition_to_lcprop
from lcprop.optics.physical_launch import resolve_beam_geometry
from tests.test_pr_gui_main_window import _wait_for


@pytest.fixture
def host():
    app = QApplication.instance() or QApplication([])
    window = PRMainWindow()
    window.resize(1200, 850)
    window.grid_panel.Nx.setValue(512)  # Resolve high-angle reciprocal-ray carriers.
    window.grid_panel.Ny.setValue(256)
    window.grid_panel.dz_um.setValue(50)
    window.grid_panel.z_length_um.setValue(100)
    window.grid_panel.x_aperture_um.setValue(200)
    window.grid_panel.y_aperture_um.setValue(200)
    window.material_panel.refractive_index.setValue(2)
    window.evolution_panel.Nt.setValue(2)
    window.tabs.setCurrentWidget(window.beam_panel)
    window.show(); app.processEvents()
    try:
        yield app, window
    finally:
        window.shutdown_background_run()
        window.close(); window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()


def snapshot(window):
    panel = window.beam_panel
    widget = panel.launch_plane_widget
    beams = widget.beam_stack
    channels = beam_stack_definition_to_lcprop(beams).channels
    enabled = [i for i,b in enumerate(beams.beams) if b.enabled]
    assert len(widget._resolved_items) == len(enabled)
    rows = []
    for index, channel, item in zip(enabled, channels, widget._resolved_items):
        geometry = resolve_beam_geometry(channel, panel._preview_n_ref)
        path = item.path()
        points = np.array([(path.elementAt(i).x, -path.elementAt(i).y)
                           for i in range(65)])
        center = np.array([channel.x0_um, channel.y0_um])
        delta = points-center
        np.testing.assert_allclose(np.einsum('ni,ij,nj->n',delta,geometry.interface_quadratic,delta),1.,atol=2e-14)
        assert item.scene() is widget.scene and item.isVisible()
        assert item.zValue() < widget.scene.beam_items[index].zValue()
        rows.append((index, beams.beams[index], geometry.interface_quadratic.copy(), points))
    return (panel._preview_n_ref, panel._interaction_length_um, rows)


def same(before, after):
    assert before[:2] == after[:2]
    assert len(before[2]) == len(after[2])
    for a,b in zip(before[2],after[2]):
        assert a[:2] == b[:2]
        np.testing.assert_array_equal(a[2],b[2])
        np.testing.assert_array_equal(a[3],b[3])


def test_two_beam_real_run_preserves_preview_every_lifecycle_boundary(host, monkeypatch):
    app,w = host; p=w.beam_panel; e=p.launch_plane_widget
    p.set_beam_stack_definition(BeamStackDefinition(beams=(
        BeamDefinition(name='left',x_um=-19,w1_um=20,w2_um=20),
        BeamDefinition(name='right',x_um=19,w1_um=20,w2_um=20))))
    initial = snapshot(w)
    for row in initial[2]:
        np.testing.assert_allclose(np.linalg.eigvalsh(row[2]), [1/400,1/400],atol=1e-16)
    # Real viewport edits through Product's inverse, not fabricated preview data.
    e.full_aperture_button.click(); app.processEvents()
    for index in (0,1):
        item=e.scene.beam_items[index];other=e.scene.beam_items[1-index]
        start=e.view.mapFromScene(item._ray_head.scenePos())
        end=e.view.mapFromScene(other._center_marker.scenePos())
        QTest.mousePress(e.view.viewport(),Qt.LeftButton,pos=start)
        for fraction in (.5,1):
            QTest.mouseMove(e.view.viewport(), start+QPoint(round((end.x()-start.x())*fraction),round((end.y()-start.y())*fraction)))
            app.processEvents()
            assert e.scene.mouseGrabberItem() is item._ray_head
        QTest.mouseRelease(e.view.viewport(),Qt.LeftButton,pos=end)
    before=snapshot(w)
    for row in before[2]:
        assert np.linalg.eigvalsh(row[2])[0] < 1/400
    for update in (lambda:e.object_list.setCurrentRow(0),lambda:e.object_list.setCurrentRow(1),
                   lambda:p.set_aperture(200,200)):
        update(); app.processEvents();same(before,snapshot(w))
    deliveries=[]
    original=w.results_panel.set_run_data
    def deliver(data):
        deliveries.append(snapshot(w));original(data);deliveries.append(snapshot(w))
    monkeypatch.setattr(w.results_panel,'set_run_data',deliver)
    # Real Run -> worker -> operation -> result slot; no replacement result.
    w.run_clicked()
    same(before,snapshot(w))
    assert w._active_request.beams == beam_stack_definition_to_lcprop(e.beam_stack)
    _wait_for(app,lambda:not w._background_running,timeout=30)
    assert w.last_runner_result is not None
    assert w.run_status == 'completed'
    assert deliveries
    for observed in deliveries:same(before,observed)
    same(before,snapshot(w))
    for update in (lambda:w.tabs.setCurrentWidget(w.results_panel),
                   lambda:w.tabs.setCurrentWidget(p),
                   lambda:e.object_list.setCurrentRow(0),lambda:e.object_list.setCurrentRow(1),
                   e.fit_button.click,e.full_aperture_button.click,
                   lambda:w.resize(1000,720),lambda:w.resize(1450,950)):
        update();app.processEvents();same(before,snapshot(w))


@pytest.mark.parametrize('theta,phi',[(0,0),(45,0),(45,90),(-45,0)])
def test_independent_input_face_oracle_and_identity_edits(host,theta,phi):
    app,w=host;p=w.beam_panel;e=p.launch_plane_widget
    beam=BeamDefinition(name='one',w1_um=20,w2_um=20,theta_ext_rad=math.radians(theta),phi_rad=math.radians(phi))
    p.set_beam_stack_definition(BeamStackDefinition(beams=(beam,replace(beam,name='two',x_um=25),replace(beam,name='off',enabled=False))))
    before=snapshot(w)
    u=np.array([math.cos(math.radians(phi)),math.sin(math.radians(phi))]);v=np.array([-u[1],u[0]])
    for row in before[2]:
        xy=row[3]-[row[1].x_um,row[1].y_um]
        np.testing.assert_allclose((xy@u/(20/math.cos(math.radians(theta))))**2+(xy@v/20)**2,1,atol=2e-14)
    p.set_optical_context(n_ref=2.4,interaction_length_um=100)
    after=snapshot(w)
    for a,b in zip(before[2],after[2]):np.testing.assert_array_equal(a[3],b[3])
    e.object_list.setCurrentRow(0);e.duplicate_button.click()
    duplicated=snapshot(w)
    assert len(duplicated[2])==3
    np.testing.assert_array_equal(duplicated[2][0][2],duplicated[2][1][2])
    np.testing.assert_allclose(duplicated[2][1][3]-duplicated[2][0][3],np.full((65,2),5.),atol=1e-14)
    e.object_list.setCurrentRow(1);e.delete_button.click()
    same(after,snapshot(w))


def test_pending_text_changes_intent_only_when_committed(host):
    # Qualify a possible observation mechanism, not a claimed native reproducer.
    app,w=host;p=w.beam_panel;e=p.launch_plane_widget
    p.set_beam_stack_definition(BeamStackDefinition(beams=(BeamDefinition(w1_um=20,w2_um=20),)))
    before=snapshot(w)
    e.theta_spin.lineEdit().setText('45')
    same(before,snapshot(w))
    beams=p.beams()  # The same pending-edit boundary used to construct Run.
    assert beams.channels[0].theta_ext_rad == math.pi/4
    after=snapshot(w)
    assert after[2][0][1].theta_ext_rad == math.pi/4
    np.testing.assert_allclose(np.linalg.eigvalsh(after[2][0][2]),[1/800,1/400],atol=1e-16)
