from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from lcprop.pr.gui.main_window import PRMainWindow


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_run_planning_tab_estimates_current_request_without_execution(app):
    window = PRMainWindow()
    panel = window.resource_estimator_panel
    assert window.tabs.tabText(window.tabs.indexOf(panel)) == "Run Planning"
    assert panel.output.toPlainText() == "No estimate yet."

    backend_before = window.evolution_panel.backend_spec()
    grid_before = (window.grid_panel.Nx.value(), window.grid_panel.Ny.value())
    target_before = window.execution_target_selector.currentData()
    panel.estimate_button.click()

    text = panel.output.toPlainText()
    assert "Model:" in text
    assert "Local Mac / NumPy proxy:" in text
    assert "H200/CuPy:" in text
    assert "Estimated Fast result:" in text
    assert "Estimated Full result:" in text
    assert window.run_status == "idle"
    assert window.last_result is None
    assert window.evolution_panel.backend_spec() == backend_before
    assert (window.grid_panel.Nx.value(), window.grid_panel.Ny.value()) == grid_before
    assert window.execution_target_selector.currentData() == target_before
    window.close()


def test_configuration_change_marks_a_completed_estimate_stale(app):
    window = PRMainWindow()
    panel = window.resource_estimator_panel
    panel.estimate_button.click()
    assert "Calibration:" in panel.output.toPlainText()

    window.grid_panel.Nx.setValue(window.grid_panel.Nx.value() + 2)

    assert panel.output.toPlainText() == "Estimate is stale; recalculate."
    window.close()


def test_invalid_request_is_reported_without_starting_science(app, monkeypatch):
    window = PRMainWindow()
    monkeypatch.setattr(
        window,
        "build_request",
        lambda: (_ for _ in ()).throw(ValueError("invalid planning fixture")),
    )

    window.resource_estimator_panel.estimate_button.click()

    assert window.resource_estimator_panel.output.toPlainText() == (
        "Estimate unavailable: invalid planning fixture"
    )
    assert window.run_status == "idle"
    window.close()
