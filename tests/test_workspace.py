import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from lcprop.products.data_model import to_run_data
from lcprop.workflows import run_static
from lcprop.gui.workspace import Workspace
from tests.test_all_workflows import make_base_static_request


def test_workspace_accepts_static_run_data():
    app = QApplication.instance() or QApplication([])
    result = run_static(make_base_static_request())
    run_data = to_run_data(result)

    workspace = Workspace()
    workspace.set_request_summary("test request")
    workspace.append_console("test console")
    workspace.set_run_data(run_data)

    assert workspace.request_summary.toPlainText() == "test request"
    assert "test console" in workspace.console.toPlainText()
    assert "Workflow: static" in workspace.diagnostics_view.toPlainText()
    assert workspace.image_pane.image_view.image is not None
