"""Synthetic Qt events: no host wheel driver or platform assumptions."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import QApplication
from lcprop.gui.views.image_view import ImageView


@pytest.fixture
def view():
    app = QApplication.instance() or QApplication([])
    canvas = ImageView()
    canvas.resize(640, 480)
    canvas.ax.set_xlim(-10, 10)
    canvas.ax.set_ylim(-5, 5)
    canvas.draw()
    yield canvas
    canvas.close()


def wheel(view, delta, *, pixels=0, horizontal=0, inverted=False,
          phase=Qt.ScrollPhase.ScrollUpdate, outside=False):
    # Convert Matplotlib physical pixels into Qt logical top-left coordinates.
    x, y = view.ax.transData.transform((2, 1))
    ratio = view.device_pixel_ratio
    pos = QPointF(x / ratio, (view.figure.bbox.height - y) / ratio)
    if outside:
        pos = QPointF(0, 0)
    event = QWheelEvent(pos, pos, QPoint(0, pixels), QPoint(horizontal, delta),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                        phase, inverted)
    QApplication.sendEvent(view, event)
    return event


@pytest.mark.parametrize('delta', [120, -120, 60, -60, 30, -30, 1, -1, 240])
@pytest.mark.parametrize('pixels', [0, 17])
def test_qt_wheel_fractional_scale_and_cursor_anchor(view, delta, pixels):
    before = np.array([view.ax.get_xlim(), view.ax.get_ylim()])
    event = wheel(view, delta, pixels=pixels)
    after = np.array([view.ax.get_xlim(), view.ax.get_ylim()])
    # Angle values govern zoom even when a trackpad also supplies pixel values.
    scale = 0.9 ** (delta / 120)
    expected = np.array([[2], [1]]) + (before - np.array([[2], [1]])) * scale
    np.testing.assert_allclose(after, expected, atol=1e-12)
    assert event.isAccepted()


def test_notch_is_ten_percent_span_change(view):
    wheel(view, 120)
    assert np.diff(view.ax.get_xlim())[0] == pytest.approx(18)


def test_fractional_composition_and_reciprocal(view):
    initial = np.array([view.ax.get_xlim(), view.ax.get_ylim()])
    for _ in range(8):
        wheel(view, 15)
    after = np.array([view.ax.get_xlim(), view.ax.get_ylim()])
    np.testing.assert_allclose(np.diff(after), np.diff(initial) * .9)
    wheel(view, -120)
    np.testing.assert_allclose([view.ax.get_xlim(), view.ax.get_ylim()], initial)


@pytest.mark.parametrize('phase', [Qt.ScrollPhase.ScrollBegin, Qt.ScrollPhase.ScrollEnd,
                                  Qt.ScrollPhase.ScrollUpdate])
def test_zero_vertical_delta_does_not_zoom(view, phase):
    before = (view.ax.get_xlim(), view.ax.get_ylim())
    event = wheel(view, 0, pixels=23, horizontal=120, phase=phase)
    assert (view.ax.get_xlim(), view.ax.get_ylim()) == before
    assert not event.isAccepted()


def test_qt_direction_is_not_inverted_twice(view):
    wheel(view, 120, inverted=True)
    assert np.diff(view.ax.get_xlim())[0] == pytest.approx(18)


def test_outside_axes_ignored(view):
    before = (view.ax.get_xlim(), view.ax.get_ylim())
    assert not wheel(view, 120, outside=True).isAccepted()
    assert (view.ax.get_xlim(), view.ax.get_ylim()) == before


def test_full_extent_remains_bounded(view):
    view._full_display_extent = (-10, 10, -5, 5)
    wheel(view, -120)
    assert view.ax.get_xlim() == (-10, 10)
    assert view.ax.get_ylim() == (-5, 5)
