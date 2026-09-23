"""Real PR host delivery of independently specified laboratory footprints."""
from dataclasses import replace
import math

import numpy as np
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QPointF
from PySide6.QtWidgets import QApplication, QGraphicsRectItem
from launchplane.model import BeamDefinition, BeamStackDefinition
from lcprop.pr.gui.main_window import PRMainWindow


@pytest.fixture
def host():
    app = QApplication.instance() or QApplication([])
    window = PRMainWindow()
    window.resize(1200, 850)
    window.show()
    panel = window.beam_panel
    panel.set_aperture(200., 200.)
    panel.set_preview_grid(128, 128)
    panel.set_beam_stack_definition(BeamStackDefinition(beams=(BeamDefinition(
        name='oracle', w1_um=20., w2_um=20., x_um=0., y_um=0.),)))
    panel.set_optical_context(n_ref=2., interaction_length_um=100.)
    app.processEvents()
    try:
        yield app, window, panel, panel.launch_plane_widget
    finally:
        window.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()


def points(widget):
    assert len(widget._resolved_items) == 1
    item = widget._resolved_items[0]
    assert item.scene() is widget.scene and item.isVisible()
    # Avoid parentItem() introspection: this PySide build transfers ownership
    # of a parentless scene item to Python when that getter returns None.
    # Dropping that wrapper would delete the aperture as a test side effect.
    backgrounds = [i for i in widget.scene.items() if isinstance(i, QGraphicsRectItem)]
    assert backgrounds
    assert all(item.zValue() > i.zValue() for i in backgrounds)
    assert item.zValue() < widget.scene.beam_items[0].zValue()
    assert item.pen().isCosmetic() and item.pen().color().alpha() > 0
    path = item.path()
    return np.array([(path.elementAt(i).x, -path.elementAt(i).y)
                     for i in range(65)])


def test_normal_circle_survives_host_refresh_resize_and_scene_reset(host):
    app, window, panel, widget = host
    for update in (lambda: None,
                   lambda: panel.set_aperture(240., 220.),
                   lambda: panel.set_optical_context(n_ref=1.5, interaction_length_um=80.),
                   lambda: panel.input_screen_editor.refresh_preview(),
                   lambda: window.resize(1450, 950),
                   widget.view.fit_aperture):
        update(); app.processEvents()
        xy = points(widget)
        np.testing.assert_allclose(np.linalg.norm(xy, axis=1), 20., atol=1e-12)
        for expected in [(20.,0.), (-20.,0.), (0.,20.), (0.,-20.)]:
            assert np.min(np.linalg.norm(xy-expected, axis=1)) < 1e-12
        assert widget.scene.beam_items[0].tilt_tip_offset() == QPointF(0., 0.)
        assert 'Host-resolved' in widget.execution_status.text()


@pytest.mark.parametrize('phi', [0., 90., 37.])
def test_oblique_external_ellipse_orientation_and_material_independence(host, phi):
    app, window, panel, widget = host
    widget.object_list.setCurrentRow(0)
    widget.theta_spin.setValue(45.)
    widget.phi_spin.setValue(phi)
    app.processEvents()
    xy = points(widget)
    u = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
    v = np.array([-u[1], u[0]])
    # Independent face-plane ellipse, with external incidence, not internal.
    np.testing.assert_allclose((xy@u/(20/math.cos(math.pi/4)))**2+(xy@v/20)**2,
                               1., atol=1e-12)
    arrow = widget.scene.beam_items[0].tilt_tip_offset()
    np.testing.assert_allclose([arrow.x(), -arrow.y()],
                               80*math.sin(math.pi/4)*u, atol=1e-12)
    before = xy.copy()
    panel.set_optical_context(n_ref=2.4, interaction_length_um=100.)
    np.testing.assert_allclose(points(widget), before, atol=1e-12)
    widget.psi_spin.setValue(73.)
    xy = points(widget)
    np.testing.assert_allclose((xy@u/(20/math.cos(math.pi/4)))**2+(xy@v/20)**2,
                               1., atol=1e-12)


def test_unequal_radii_roll_and_center_parity(host):
    app, window, panel, widget = host
    beam = replace(widget.beam_stack.beams[0], w1_um=10., w2_um=20.,
                   x_um=7., y_um=11., psi_rad=math.pi/2)
    panel.set_beam_stack_definition(BeamStackDefinition(beams=(beam,)))
    xy = points(widget)
    np.testing.assert_allclose(((xy[:,0]-7)/20)**2+((xy[:,1]-11)/10)**2, 1., atol=1e-12)
    assert widget.scene.beam_items[0].pos() == QPointF(7., -11.)
