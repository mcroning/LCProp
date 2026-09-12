from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from lcprop.core.backend import BackendSpec
from lcprop.optics.screens import (
    ChannelLaunchElements,
    IntensityRasterScreen,
    ScreenPlacement,
)
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.pr.gui.request_adapter import validate_pr_transverse_gui_request
from lcprop.pr.image_sources import PRImageSource
from lcprop.pr.transverse.operations import PR_TRANSVERSE_TIMEDEPENDENT_OPERATION
from lcprop.pr.transverse.specs import (
    PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1,
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PRTransverseRunRequest,
    PR_TRANSVERSE_IMEX_EULER,
    PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _window(app, *, steps: int = 1) -> PRMainWindow:
    window = PRMainWindow()
    window.evolution_panel.set_workflow_id(
        PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
    )
    window.grid_panel.Nx.setValue(12)
    window.grid_panel.Ny.setValue(10)
    window.grid_panel.x_aperture_um.setValue(48.0)
    window.grid_panel.y_aperture_um.setValue(40.0)
    window.grid_panel.z_length_um.setValue(10.0)
    window.grid_panel.dz_um.setValue(5.0)
    window.material_panel.applied_field.setValue(0.0)
    window.material_panel.dark_intensity.setValue(0.2)
    window.material_panel.uniform_background_intensity.setValue(0.1)
    window.material_panel.gain_length_product.setValue(0.05)
    window.material_panel.use_characteristic_wavenumber_override.setChecked(True)
    window.material_panel.characteristic_wavenumber_per_um_override.setValue(0.1)
    window.evolution_panel.Nt.setValue(steps)
    window.evolution_panel.dt_normalized.setValue(1.0e-4)
    window.evolution_panel.optical_substeps.setValue(1)
    window.evolution_panel.backend.setCurrentText("numpy")
    window.evolution_panel.precision.setCurrentText("float64")
    return window


def _wait_for(app, predicate, *, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for transverse-TD GUI run")
        time.sleep(0.002)
    app.processEvents()


def test_gui_builds_canonical_transverse_td_request_and_registers_operation(app):
    window = _window(app, steps=3)
    index = window.evolution_panel.workflow.findData(
        PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
    )
    request = window.build_request()
    summary = window.describe_request(request)

    assert window.evolution_panel.workflow.model().item(index).isEnabled()
    assert "Full transverse PR transport" in (
        window.evolution_panel.workflow.itemText(index)
    )
    assert isinstance(request, PRTransverseRunRequest)
    assert request.solver.Nt == 3
    assert request.solver.dt_normalized == pytest.approx(1.0e-4)
    assert request.solver.integrator == PR_TRANSVERSE_IMEX_EULER
    assert request.transport.m_y == 1.0
    assert request.dielectric.h_y == 1.0
    assert request.projection.g_x == 1.0
    assert request.projection.g_y == 0.0
    assert request.initial_A is None
    assert request.initial_psi is None
    assert validate_pr_transverse_gui_request(request).aperture is not None
    assert PR_TRANSVERSE_TIMEDEPENDENT_OPERATION in (
        window.local_runner.registered_operations
    )
    assert "Workflow: pr_transverse_timedependent" in summary
    assert "full 2D transverse zero-flux" in summary
    window.close()


def test_transverse_td_gui_preserves_supported_backend_policy(app):
    window = _window(app)
    window.evolution_panel.backend.setCurrentText("auto")
    with pytest.raises(ValueError, match="explicit NumPy or CuPy"):
        window.build_request()

    window.evolution_panel.backend.setCurrentText("cupy")
    window.evolution_panel.precision.setCurrentText("float32")
    request = window.build_request()
    assert request.backend == BackendSpec("cupy", "float32", False)
    window.close()


def test_gui_builds_linearized_transverse_td_as_experimental(app):
    window = _window(app)
    panel = window.evolution_panel
    response_index = panel.material_response.findData(
        PR_MATERIAL_RESPONSE_LINEARIZED
    )
    panel.material_response.setCurrentIndex(response_index)
    panel.reference_intensity.setValue(1.75)
    panel.transverse_applied_field.setValue(0.3)

    request = window.build_request()
    summary = window.describe_request(request)

    assert request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
    assert request.material_response.reference_intensity == 1.75
    assert request.boundary.profile_id == (
        PR_FULL_TRANSVERSE_PERIODIC_BIASED_CURRENT_V1
    )
    assert request.boundary.applied_field_x == 0.3
    assert "Linearized full transverse [Experimental]" in summary
    assert "exact frozen-source modal update" in summary
    assert not panel.material_response.isHidden()
    assert not panel.reference_intensity.isHidden()
    assert not panel.transverse_applied_field.isHidden()
    panel.set_image_amplification_mode(True)
    workflow_index = panel.workflow.findData(PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW)
    assert "[Experimental]" in panel.workflow.itemText(workflow_index)
    assert "Experimental:" in panel.algorithm_status.text()
    window.close()


def test_transverse_td_gui_worker_dispatches_progress_and_products(app):
    window = _window(app)
    gui_thread = QThread.currentThread()
    calls = []
    original = window.runner.run_registered

    def observed(material_id, workflow_id, request, **kwargs):
        calls.append((material_id, workflow_id, QThread.currentThread()))
        return original(material_id, workflow_id, request, **kwargs)

    window.runner.run_registered = observed
    window.run_button.click()
    _wait_for(app, lambda: not window._background_running)

    assert calls[0][0:2] == ("pr", PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW)
    assert calls[0][2] is not gui_thread
    assert window.last_progress.workflow == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
    assert window.last_progress.completed_units == 1
    assert window.last_progress.current_coordinate == pytest.approx(1.0e-4)
    assert window.last_runner_result.kind == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
    assert window.last_result.status == "completed"
    assert window.run_status == "completed"
    assert window.status_label.text() == "Completed"
    assert {"psi", "P", "E_x", "E_y", "E_active"}.issubset(
        window.last_runner_result.run_data.fields.keys()
    )
    assert window.last_checkpoint is None
    window.close()


def test_transverse_td_experiment_gui_round_trip_preserves_screen_and_backend(
    app,
    tmp_path,
):
    window = _window(app, steps=4)
    screen = IntensityRasterScreen(
        source=PRImageSource.from_array(np.eye(5)),
        placement=ScreenPlacement(
            width_um=20.0,
            height_um=20.0,
            boundary_policy="reject",
        ),
    )
    window.beam_panel.input_screen_editor.set_launch_elements(
        (ChannelLaunchElements(0, (screen,)),)
    )
    window.evolution_panel.set_backend_spec(
        BackendSpec("cupy", "float32", False)
    )
    expected = window.build_request()
    path = tmp_path / "transverse-td.lcprop.json"

    window.save_experiment_to(path)
    window.evolution_panel.set_workflow_id("pr_timedependent")
    loaded = window.load_experiment_from(path)

    assert loaded.request == expected
    assert window.build_request() == expected
    assert expected.backend.backend == "cupy"
    assert expected.backend.precision == "float32"
    assert expected.launch_elements == (ChannelLaunchElements(0, (screen,)),)
    assert "initial_A" not in path.read_text(encoding="utf-8")
    assert "initial_psi" not in path.read_text(encoding="utf-8")
    window.close()


def test_transverse_td_gui_stop_preserves_an_accepted_boundary(app):
    window = _window(app, steps=100)
    window.run_button.click()
    _wait_for(
        app,
        lambda: (
            window.last_progress is not None
            and window.last_progress.completed_units >= 1
        ),
    )
    window.stop_button.click()
    _wait_for(app, lambda: not window._background_running)

    assert window.last_result.status == "cancelled"
    assert 1 <= window.last_result.completed_steps < 100
    assert window.run_status == "stopped"
    assert window.last_result.diagnostics["complete_final_optical_replay"]
    window.close()
