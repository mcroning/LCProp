import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from lcprop.products.data_model import to_run_data
from lcprop.workflows import run_static
from lcprop.gui.views.image_pane import ImagePane
from tests.test_all_workflows import make_base_static_request


def test_image_pane_lists_2d_fields():
    app = QApplication.instance() or QApplication([])

    run_data = to_run_data(run_static(make_base_static_request()))
    pane = ImagePane()
    pane.set_run_data(run_data)

    assert pane.field_selector.count() >= 2
    assert pane.field_selector.itemText(0) == "Intensity"
    assert pane.image_view.image is not None


def test_image_pane_can_switch_to_theta():
    app = QApplication.instance() or QApplication([])

    run_data = to_run_data(run_static(make_base_static_request()))
    pane = ImagePane()
    pane.set_run_data(run_data)

    for i in range(pane.field_selector.count()):
        if pane.field_selector.itemText(i) == "Theta":
            pane.field_selector.setCurrentIndex(i)
            break

    assert pane.field_selector.currentText() == "Theta"
    assert pane.image_view.image is not None
