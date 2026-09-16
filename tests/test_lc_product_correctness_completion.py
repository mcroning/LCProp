from __future__ import annotations

from dataclasses import replace
import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.lc.gui.main_window import LCPropMainWindow
from lcprop.lc.gui.request_adapter import validate_lc_gui_request_representable
from lcprop.lc.experiment_codec import encode_lc_static_request
from lcprop.lc.operations import LC_STATIC_OPERATION
from lcprop.lc.products import from_static_result, from_timedependent_result
from lcprop.lc.requests import (
    RuntimeOptions,
    SolitonExistenceRequest,
    SolitonRequest,
    StaticSolverOptions,
    StaticWorkflowOptions,
    TimeDependentRunRequest,
    TimeDependentSolverOptions,
)
from lcprop.lc.workflows.static import run_static
from lcprop.lc.workflows.timedependent import (
    continue_timedependent,
    run_timedependent,
    validate_timedependent_continuation,
)
from lcprop.optics.boundaries import TransverseBoundarySpec
from lcprop.runners.slurm import SlurmResourceProfile
from tests.test_all_workflows import make_base_static_request


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _local_static_request(*, precision="float64", boundary=None):
    base = make_base_static_request(power_mW=0.1)
    return replace(
        base,
        grid=GridSpec(
            Nx=12,
            Ny=10,
            dz_um=4.0,
            x_aperture_um=40.0,
            y_aperture_um=50.0,
            z_length_um=8.0,
        ),
        solver=StaticSolverOptions(
            workflow=StaticWorkflowOptions(
                strategy="local_self_consistent",
                theta_solver="picard_cn",
                optics_solver="splitstep",
                coupling="self_consistent",
            ),
            static_max_relax_iterations=1,
            static_max_coupled_passes=1,
            static_residual_rms_tol=1.0e9,
            static_residual_max_tol=1.0e9,
        ),
        runtime=RuntimeOptions(precision=precision),
        optical_boundary=boundary or TransverseBoundarySpec(),
    )


def _td_request(*, precision="float64", boundary=None):
    base = _local_static_request(precision=precision, boundary=boundary)
    return TimeDependentRunRequest(
        grid=base.grid,
        material=base.material,
        bias=base.bias,
        beams=base.beams,
        solver=TimeDependentSolverOptions(Nt=1, dt=7.5e-4),
        output=base.output,
        runtime=base.runtime,
        optical_boundary=base.optical_boundary,
    )


def test_local_self_consistent_boundary_is_effective_and_periodic_is_noop(
    monkeypatch,
):
    periodic_request = _local_static_request()
    periodic = run_static(periodic_request)

    import lcprop.lc.workflows.static as static_workflow

    established_advance = static_workflow.advance_slice_with_midpoint_source

    def without_boundary(A, theta, **kwargs):
        kwargs.pop("boundary")
        kwargs.pop("boundary_grid")
        return established_advance(A, theta, **kwargs)

    monkeypatch.setattr(
        static_workflow,
        "advance_slice_with_midpoint_source",
        without_boundary,
    )
    established_periodic = run_static(periodic_request)
    np.testing.assert_array_equal(periodic.A_final, established_periodic.A_final)
    np.testing.assert_array_equal(
        periodic.intensity_stack,
        established_periodic.intensity_stack,
    )
    np.testing.assert_array_equal(
        periodic.theta_final,
        established_periodic.theta_final,
    )

    monkeypatch.setattr(
        static_workflow,
        "advance_slice_with_midpoint_source",
        established_advance,
    )
    sponge = run_static(
        _local_static_request(
            boundary=TransverseBoundarySpec(
                mode="sponge",
                width_fraction=0.5,
                attenuation_per_um=0.5,
                profile_order=2,
            )
        )
    )
    assert not np.array_equal(sponge.A_final, periodic.A_final)
    assert sponge.power_final < periodic.power_final


def test_td_continuation_requires_unchanged_optical_boundary():
    request = _td_request()
    first = run_timedependent(request)

    validate_timedependent_continuation(request, first.checkpoint)
    continued = continue_timedependent(request, first.checkpoint, 1)
    assert continued.completed_steps == 2

    changed = replace(
        request,
        optical_boundary=TransverseBoundarySpec(
            mode="sponge", attenuation_per_um=0.05
        ),
    )
    with pytest.raises(ValueError, match="incompatible optical boundary"):
        validate_timedependent_continuation(changed, first.checkpoint)


def test_lc_rejects_mixed_wavelengths_but_preserves_equal_wavelength_channels():
    base = _local_static_request()
    channel = base.beams.channels[0]
    equal = replace(
        base,
        beams=BeamStack(
            channels=(
                replace(channel, name="one", power_mW=0.05),
                replace(channel, name="two", power_mW=0.05),
            )
        ),
    )
    assert np.isfinite(np.asarray(run_static(equal).A_final)).all()

    mixed = replace(
        equal,
        beams=BeamStack(
            channels=(
                equal.beams.channels[0],
                replace(equal.beams.channels[1], wavelength_um=0.532),
            )
        ),
    )
    with pytest.raises(ValueError, match="same wavelength"):
        run_static(mixed)
    with pytest.raises(ValueError, match="same wavelength"):
        run_timedependent(
            replace(_td_request(), beams=mixed.beams)
        )
    with pytest.raises(ValueError, match="same wavelength"):
        validate_lc_gui_request_representable(mixed)
    with pytest.raises(ValueError, match="same wavelength"):
        encode_lc_static_request(mixed)


def test_stationary_lc_requests_are_explicitly_periodic_only():
    nonperiodic = replace(
        _local_static_request(),
        optical_boundary=TransverseBoundarySpec(mode="tukey", tukey_alpha=0.4),
    )
    with pytest.raises(ValueError, match="stationary.*periodic"):
        SolitonRequest(base=nonperiodic).validate()
    with pytest.raises(ValueError, match="stationary.*periodic"):
        SolitonExistenceRequest(base=nonperiodic).validate()


def test_lc_volume_coordinates_are_slice_midpoints():
    result = run_static(_local_static_request())
    data = from_static_result(result)

    np.testing.assert_array_equal(data.geometry.z, np.array([2.0, 6.0]))
    np.testing.assert_array_equal(
        [item.z_um for item in result.slice_summaries],
        np.array([2.0, 6.0]),
    )
    np.testing.assert_array_equal(
        data.curves["static_final_residual_rms"].x,
        data.geometry.z,
    )
    assert data.geometry.z[0] != 0.0
    assert data.geometry.z[-1] == result.grid_summary["z_length_um"] - 2.0


def test_numpy_float32_static_and_td_paths_preserve_expected_dtypes():
    static32 = run_static(_local_static_request(precision="float32"))
    static64 = run_static(_local_static_request(precision="float64"))
    assert static32.A_final.dtype == np.complex64
    assert static32.theta_final.dtype == np.float32
    np.testing.assert_allclose(
        static32.theta_final,
        static64.theta_final,
        rtol=2.0e-5,
        atol=2.0e-6,
    )

    td32 = run_timedependent(_td_request(precision="float32"))
    td64 = run_timedependent(_td_request(precision="float64"))
    assert td32.A_final.dtype == np.complex64
    assert td32.theta_final.dtype == np.float32
    np.testing.assert_allclose(
        td32.theta_final,
        td64.theta_final,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    assert from_timedependent_result(td32).geometry.z.tolist() == [2.0, 6.0]


def test_gui_stationary_applicability_and_boundary_restoration(app):
    window = LCPropMainWindow()
    sponge = TransverseBoundarySpec(
        mode="sponge",
        width_fraction=0.2,
        attenuation_per_um=0.03,
        profile_order=3,
    )
    window.beam_panel.set_optical_boundary(sponge)

    window.experiment_panel.set_current_experiment("Soliton")
    assert window.beam_panel.optical_boundary().mode == "periodic"
    assert not window.beam_panel.boundary_mode.isEnabled()
    assert not window.solver_panel.workflow.isEnabled()
    assert not window.solver_panel.max_iterations.isEnabled()
    assert not window.solver_panel.Nt.isEnabled()
    assert window.solver_panel.soliton_mode_selector.isEnabled()
    assert window.solver_panel.refine_transverse_checkbox.isEnabled()
    assert not window.experiment_file_buttons.save_button.isEnabled()
    assert not window.experiment_file_buttons.open_button.isEnabled()
    assert window.build_soliton_request().base.optical_boundary.mode == "periodic"

    window.experiment_panel.set_current_experiment("Static propagation")
    assert window.beam_panel.optical_boundary() == sponge
    assert window.beam_panel.boundary_mode.isEnabled()
    assert window.solver_panel.workflow.isEnabled()
    assert not window.solver_panel.Nt.isEnabled()
    assert window.experiment_file_buttons.save_button.isEnabled()

    window.experiment_panel.set_current_experiment("Time-dependent propagation")
    assert not window.solver_panel.workflow.isEnabled()
    assert window.solver_panel.Nt.isEnabled()
    assert window.solver_panel.dt.isEnabled()
    window.close()


def test_gui_disables_lc_submission_for_gpu_slurm_resource(app, monkeypatch):
    class DummySlurmRunner:
        name = "Slurm"
        supports_parallel_sweeps = False
        registered_operations = ()

    gpu = SlurmResourceProfile(
        "gpu", "gpu", "normal", "00:10:00", 2, 8,
        gpus=1, require_cupy=True,
    )
    cluster = SimpleNamespace(profile=lambda _name: gpu)
    window = LCPropMainWindow()
    window.slurm_runner = DummySlurmRunner()
    slurm_item = window.execution_target_selector.model().item(1)
    assert slurm_item is not None
    slurm_item.setEnabled(True)
    monkeypatch.setattr(
        window.remote_execution_controls,
        "selected_cluster",
        lambda: cluster,
    )
    monkeypatch.setattr(
        window.remote_execution_controls,
        "selected_resource_name",
        lambda: "gpu",
    )
    window.execution_target_selector.setCurrentIndex(1)

    assert not window.run_button.isEnabled()
    assert "NumPy" in window.run_button.toolTip()
    assert "CPU Slurm" in window.run_button.toolTip()
    window.close()


def test_gui_prominently_reports_static_nonconvergence(app):
    request = replace(
        _local_static_request(),
        solver=replace(
            _local_static_request().solver,
            static_residual_rms_tol=0.0,
            static_residual_max_tol=0.0,
        ),
    )
    window = LCPropMainWindow()
    window._active_request = request
    runner_result = window.local_runner.run_operation(
        LC_STATIC_OPERATION,
        request,
    )
    assert runner_result.result.all_slices_converged is False

    window._on_timedependent_finished(runner_result)

    assert window.run_status == "nonconverged"
    assert (
        runner_result.run_data.diagnostics["summary"].values[
            "scientific_status"
        ]
        == "not_converged"
    )
    assert "did not satisfy the convergence qualifications" in (
        window.results_panel.workspace.console.toPlainText()
    )
    window.close()
