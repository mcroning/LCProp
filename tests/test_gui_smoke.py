import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from lcprop.gui.main_window import LCPropMainWindow
from lcprop.lc.operations import (
    LC_PARAMETER_SWEEP_OPERATION,
    LC_SOLITON_OPERATION,
    LC_STATIC_OPERATION,
    LC_TIMEDEPENDENT_OPERATION,
)
from lcprop.runners.local import LocalRunner


def test_gui_builds_static_request():
    app = QApplication.instance() or QApplication([])
    win = LCPropMainWindow()
    req = win.build_request()

    assert req.grid.Nx == 64
    assert req.grid.z_length_um == 3000.0
    assert req.beams.channels[0].waist_x_um == 3.0
    assert req.beams.channels[0].power_mW == 1.0


def test_lc_gui_registers_and_selects_canonical_operations():
    app = QApplication.instance() or QApplication([])
    win = LCPropMainWindow()

    assert win.runner.registered_operations == (
        LC_STATIC_OPERATION,
        LC_TIMEDEPENDENT_OPERATION,
        LC_SOLITON_OPERATION,
        LC_PARAMETER_SWEEP_OPERATION,
    )
    dispatch = win._experiment_dispatch()
    assert dispatch["Static propagation"][1] is LC_STATIC_OPERATION
    assert dispatch["Time-dependent propagation"][1] is LC_TIMEDEPENDENT_OPERATION
    assert dispatch["Soliton"][1] is LC_SOLITON_OPERATION
    assert dispatch["Soliton existence curve"][1] is LC_PARAMETER_SWEEP_OPERATION


def test_lc_gui_registered_dispatch_does_not_require_compatibility_method(
    monkeypatch,
):
    app = QApplication.instance() or QApplication([])
    win = LCPropMainWindow()
    request = win.build_request()

    def compatibility_path_is_not_canonical(*_args, **_kwargs):
        raise AssertionError("historical LocalRunner.run_static was invoked")

    monkeypatch.setattr(win.runner, "run_static", compatibility_path_is_not_canonical)
    result = win._run_registered(LC_STATIC_OPERATION, request)

    assert result.kind == "static"
    assert result.material_id == "lc"
    assert result.run_data.workflow == "static"


def test_local_runner_static_smoke():
    app = QApplication.instance() or QApplication([])
    win = LCPropMainWindow()
    req = win.build_request()

    result = LocalRunner().run_static(req)

    assert result.kind == "static"
    assert result.result.A_final.shape == (1, 64, 64)
    assert result.result.theta_final.shape == (150, 64, 64)


def test_gui_request_description_contains_key_fields():
    app = QApplication.instance() or QApplication([])
    win = LCPropMainWindow()
    req = win.build_request()
    text = win.describe_request(req)

    assert "Experiment: Static propagation" in text
    assert "Runner: Local CPU" in text
    assert "Beams: 1 enabled, total P=1 mW" in text
    assert "First enabled beam: beam, P=1 mW" in text
    assert "Workflow: local_self_consistent" in text


def test_gui_builds_optional_transverse_refinement_request():
    app = QApplication.instance() or QApplication([])
    win = LCPropMainWindow()
    win.solver_panel.refine_transverse_checkbox.setChecked(True)

    req = win.build_soliton_request()

    assert req.refine_transverse is True
    assert req.transverse_max_outer == 100
    assert req.transverse_theta_steps_per_outer == 50
    assert req.transverse_field_mix == 0.5
    assert req.transverse_theta_mix == 0.5
