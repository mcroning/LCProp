import numpy as np
from PySide6.QtWidgets import QApplication

from lcprop.core.requests import (
    RuntimeOptions,
    TimeDependentRunRequest,
    TimeDependentSolverOptions,
)
from lcprop.workflows import run_static, run_timedependent
from lcprop.gui.main_window import LCPropMainWindow
from tests.test_all_workflows import make_base_static_request


def test_static_default_precision_is_float64():
    req = make_base_static_request()
    result = run_static(req)
    assert result.theta_final.dtype == np.float64
    assert result.A_final.dtype == np.complex128


def test_timedependent_default_theta_precision_is_float64():
    base = make_base_static_request()
    result = run_timedependent(
        TimeDependentRunRequest(
            grid=base.grid,
            material=base.material,
            bias=base.bias,
            beams=base.beams,
            solver=TimeDependentSolverOptions(Nt=1),
            output=base.output,
        )
    )
    assert result.theta_final.dtype == np.float64
    assert result.A_final.dtype == np.complex128

def test_gui_timedependent_request_uses_float64():
    app = QApplication.instance() or QApplication([])
    win = LCPropMainWindow()
    req = win.build_timedependent_request()
    assert req.runtime.precision == "float64"
    
if False:
    def test_float32_runtime_option_still_available():
        base = make_base_static_request()
        req = TimeDependentRunRequest(
            grid=base.grid,
            material=base.material,
            bias=base.bias,
            beams=base.beams,
            solver=TimeDependentSolverOptions(Nt=1),
            output=base.output,
            runtime=RuntimeOptions(precision="float32"),
        )
        result = run_timedependent(req)
        assert result.theta_final.dtype == np.float32
        assert result.A_final.dtype == np.complex64
