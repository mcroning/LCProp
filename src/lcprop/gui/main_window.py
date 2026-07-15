from __future__ import annotations
from dataclasses import replace

import traceback

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTabWidget,
    QCheckBox,
)

from lcprop.products.data_model import to_run_data
from lcprop.runners.local import LocalRunner
from lcprop.gui.panels import (
    ExperimentPanel,
    PhysicsPanel,
    BeamPanel,
    GridPanel,
    SolverPanel,
    SweepPanel,
    ResultsPanel,
)
from lcprop.core.requests import StaticRunRequest, TimeDependentRunRequest, OutputOptions, RuntimeOptions
from lcprop.workflows.soliton import SolitonRequest
from lcprop.workflows.sweep import ParameterSweepRequest

class LCPropMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.runner = LocalRunner()
        self.last_soliton_result = None
        self.setWindowTitle("LCProp")
        # Default to a wide scientific-visualization layout.
        self.setMinimumSize(1200, 760)

        root = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("LCProp"))
        header.addStretch(1)
        header.addWidget(QLabel(f"Runner: {self.runner.name}"))

        self.use_last_soliton = QCheckBox("Start from last soliton")
        self.use_last_soliton.setEnabled(False)
        self.use_last_soliton.setVisible(False)
        header.addWidget(self.use_last_soliton)

        self.run_button = QPushButton()
        self.run_button.clicked.connect(self.run_static_clicked)
        header.addWidget(self.run_button)

        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.experiment_panel = ExperimentPanel()
        self.physics_panel = PhysicsPanel()
        self.grid_panel = GridPanel()
        self.beam_panel = BeamPanel(
            x_aperture_um=self.grid_panel.x_aperture_um.value(),
            y_aperture_um=self.grid_panel.y_aperture_um.value(),
        )
        self.solver_panel = SolverPanel()
        self.sweep_panel = SweepPanel()
        self.results_panel = ResultsPanel()

        self.tabs.addTab(self.experiment_panel, "Experiment")
        self.tabs.addTab(self.physics_panel, "Physics")
        self.tabs.addTab(self.beam_panel, "Beam")
        self.tabs.addTab(self.grid_panel, "Grid")
        self.tabs.addTab(self.solver_panel, "Solver")
        self.sweep_tab_index = self.tabs.addTab(self.sweep_panel, "Sweep")
        self.tabs.addTab(self.results_panel, "Results")
        self.experiment_panel.experimentChanged.connect(self.update_run_button)
        self.grid_panel.x_aperture_um.valueChanged.connect(
            self._sync_beam_aperture
        )
        self.grid_panel.y_aperture_um.valueChanged.connect(
            self._sync_beam_aperture
        )
        self.update_run_button()   
        self.resize(1450, 900)

    def _sync_beam_aperture(self) -> None:
        self.beam_panel.set_aperture(
            self.grid_panel.x_aperture_um.value(),
            self.grid_panel.y_aperture_um.value(),
        )

    def update_run_button(self) -> None:
        experiment = self.experiment_panel.current_experiment()
        self.run_button.setText(f"Run {experiment}")
        self.solver_panel.set_experiment_mode(experiment)
        self._update_sweep_tab(experiment)
        self._update_initial_condition_controls(experiment)

    def _update_sweep_tab(self, experiment: str) -> None:
        sweep_enabled = experiment == "Soliton existence curve"
        self.tabs.setTabEnabled(self.sweep_tab_index, sweep_enabled)
        if not sweep_enabled and self.tabs.currentIndex() == self.sweep_tab_index:
            self.tabs.setCurrentWidget(self.experiment_panel)

    def _update_initial_condition_controls(self, experiment: str) -> None:
        visible = experiment in {"Static propagation", "Time-dependent propagation"}
        enabled = visible and self.last_soliton_result is not None
        self.use_last_soliton.setVisible(visible)
        self.use_last_soliton.setEnabled(enabled)
        if not enabled:
            self.use_last_soliton.setChecked(False)

    def _maybe_apply_last_soliton(self, req, *, allow: bool):
        if not allow or not self.use_last_soliton.isChecked():
            return req
        if self.last_soliton_result is None:
            return req
        return replace(
            req,
            initial_A=self.last_soliton_result.A,
            initial_theta=self.last_soliton_result.theta,
        )

    def build_request(self, *, allow_last_soliton: bool = True) -> StaticRunRequest:
        req = StaticRunRequest(
            grid=self.grid_panel.grid(),
            material=self.physics_panel.material(),
            bias=self.physics_panel.bias(),
            beams=self.beam_panel.beams(),
            solver=self.solver_panel.solver(),
            output=OutputOptions(),
        )
        return self._maybe_apply_last_soliton(req, allow=allow_last_soliton)

    def build_timedependent_request(self) -> TimeDependentRunRequest:
        req = TimeDependentRunRequest(
            grid=self.grid_panel.grid(),
            material=self.physics_panel.material(),
            bias=self.physics_panel.bias(),
            beams=self.beam_panel.beams(),
            solver=self.solver_panel.td_solver(),
            output=OutputOptions(),
            runtime=RuntimeOptions(precision="float64"),
        )
        return self._maybe_apply_last_soliton(req, allow=True)

    def build_soliton_request(self) -> SolitonRequest:
        return SolitonRequest(
            base=self.build_request(allow_last_soliton=False),
            mode=self.solver_panel.soliton_mode(),
            refine_transverse=self.solver_panel.refine_transverse(),
        )

    def build_soliton_existence_request(self) -> ParameterSweepRequest:
        base = SolitonRequest(
            base=self.build_request(allow_last_soliton=False),
            mode=self.solver_panel.soliton_mode(),
        )

        return ParameterSweepRequest(
            experiment="soliton",
            parameter=self.sweep_panel.sweep_parameter(),
            values=self.sweep_panel.sweep_values(),
            base=base,
            continuation=self.sweep_panel.use_continuation(),
            execution=self.sweep_panel.sweep_execution(),
        )

    def _base_static_request(self, req):
        base_req = req
        while hasattr(base_req, "base"):
            base_req = base_req.base
        return base_req

    def describe_request(self, req) -> str:
        base_req = self._base_static_request(req)
        channels = base_req.beams.channels
        first_channel = channels[0]
        total_power_mW = sum(channel.power_mW for channel in channels)
        wavelengths = tuple(
            dict.fromkeys(channel.wavelength_um for channel in channels)
        )
        coherence_groups = tuple(
            dict.fromkeys(channel.coherence_group for channel in channels)
        )
        lines = [
            f"Experiment: {self.experiment_panel.current_experiment()}",
            f"Runner: {self.runner.name}",
            f"Grid: {base_req.grid.Nx} × {base_req.grid.Ny}, Nz≈{round(base_req.grid.z_length_um / base_req.grid.dz_um)}",
            f"Aperture: x={base_req.grid.x_aperture_um:g} µm, y={base_req.grid.y_aperture_um:g} µm",
            f"z length: {base_req.grid.z_length_um:g} µm, dz={base_req.grid.dz_um:g} µm",
            f"Material: ne={base_req.material.ne:g}, no={base_req.material.no:g}, K={base_req.material.K:g}, Δε={base_req.material.delta_epsilon:g}",
            f"Bias: V={base_req.bias.V_bias:g} V, theta_bc={base_req.bias.theta_bc:g} rad",
            f"Beams: {len(channels)} enabled, total P={total_power_mW:g} mW",
            "Wavelengths: " + ", ".join(f"{value:g} µm" for value in wavelengths),
            "Lasers/coherence groups: " + ", ".join(coherence_groups),
            f"First enabled beam: {first_channel.name}, P={first_channel.power_mW:g} mW, "
            f"waists=({first_channel.waist_x_um:g}, {first_channel.waist_y_um:g}) µm, "
            f"λ={first_channel.wavelength_um:g} µm",
            f"Workflow: {base_req.solver.workflow.strategy}",
            f"Initial condition: {'last soliton' if getattr(base_req, 'initial_A', None) is not None or getattr(base_req, 'initial_theta', None) is not None else 'default launch'}",
        ]

        if hasattr(base_req.solver, "resolved_static_max_coupled_passes"):
            lines.extend([
                f"Static max coupled passes: {base_req.solver.resolved_static_max_coupled_passes}",
                f"Static max relax iterations/pass: {base_req.solver.static_max_relax_iterations}",
                f"Static residual RMS tolerance: {base_req.solver.static_residual_rms_tol}",
                f"Static residual max tolerance: {base_req.solver.static_residual_max_tol}",
                f"Static Δθ RMS tolerance: {base_req.solver.resolved_delta_theta_rms_tol}",
                f"Static Δθ max tolerance: {base_req.solver.resolved_delta_theta_max_tol}",
                f"Record static iteration history: {base_req.solver.record_iteration_history}",
            ])

        mode = getattr(req, "mode", None)
        if mode is not None:
            lines.append(f"Soliton mode: {mode}")

        sweep_values = getattr(req, "values", None)
        if sweep_values is not None:
            parameter = getattr(req, "parameter", "parameter")
            lines.append(f"Sweep parameter: {parameter}")
            lines.append("Sweep values: " + ", ".join(f"{p:g}" for p in sweep_values))
            lines.append(f"Continuation: {getattr(req, 'continuation', False)}")
            lines.append(f"Execution: {getattr(req, 'execution', 'sequential')}")
        else:
            powers = getattr(req, "powers_mW", None)
            if powers is not None:
                lines.append("Powers: " + ", ".join(f"{p:g} mW" for p in powers))

        return "\n".join(lines)

    def _experiment_dispatch(self):
        return {
            "Static propagation": (
                self.build_request,
                self.runner.run_static,
                "static workflow",
            ),
            "Time-dependent propagation": (
                self.build_timedependent_request,
                self.runner.run_timedependent,
                "time-dependent workflow",
            ),
            "Soliton": (
                self.build_soliton_request,
                self.runner.run_soliton,
                "soliton workflow",
            ),
            "Soliton existence curve": (
                self.build_soliton_existence_request,
                self.runner.run_parameter_sweep,
                "soliton existence workflow",
            ),
        }

    def run_static_clicked(self):
        self.run_button.setEnabled(False)
        self.run_button.setText("Running…")
        self.tabs.setCurrentWidget(self.results_panel)

        try:
            experiment = self.experiment_panel.current_experiment()
            try:
                request_builder, runner_result_fn, run_label = self._experiment_dispatch()[experiment]
            except KeyError as exc:
                raise ValueError(f"Unsupported experiment: {experiment}") from exc
            req = request_builder()

            self.results_panel.set_request_summary(self.describe_request(req))
            self.results_panel.append_console(f"Running {run_label} with {self.runner.name}...")
            QApplication.processEvents()

            runner_result = runner_result_fn(req)
            result = runner_result.result
            if type(result).__name__ == "SolitonResult":
                self.last_soliton_result = result
                self.results_panel.append_console("Saved this soliton as the current in-memory initial condition.")
            run_data = to_run_data(result)
            self.results_panel.set_run_data(run_data)

            self.results_panel.append_console("")
            self.results_panel.append_console(runner_result.message)
            self.results_panel.append_console("Run complete")
            self._append_result_summary(result, runner_result.kind)

        except Exception:
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())

        finally:
            self.run_button.setEnabled(True)
            self.update_run_button()


    def _append_result_summary(self, result, fallback_method: str) -> None:
        self.results_panel.append_console(f"method: {getattr(result, 'method', fallback_method)}")

        grid_summary = getattr(result, "grid_summary", None)
        if grid_summary is not None:
            self.results_panel.append_console(
                "Nx, Ny, Nz: "
                f"{grid_summary['Nx']}, "
                f"{grid_summary['Ny']}, "
                f"{grid_summary['Nz']}"
            )

        if hasattr(result, "power_initial"):
            self.results_panel.append_console(
                f"normalized_field_integral_initial: {result.power_initial:.8g}"
            )
        if hasattr(result, "power_final"):
            self.results_panel.append_console(
                f"normalized_field_integral_final:   {result.power_final:.8g}"
            )
        if getattr(result, "physical_power_initial_mW", None) is not None:
            self.results_panel.append_console(
                f"physical_power_initial_mW: {result.physical_power_initial_mW:.8g}"
            )
        if getattr(result, "physical_power_final_mW", None) is not None:
            self.results_panel.append_console(
                f"physical_power_final_mW:   {result.physical_power_final_mW:.8g}"
            )

        A = getattr(result, "A_final", getattr(result, "A", None))
        theta = getattr(result, "theta_final", getattr(result, "theta", None))
        if A is not None:
            self.results_panel.append_console(f"A shape:       {A.shape}")
        if theta is not None:
            self.results_panel.append_console(f"theta shape:   {theta.shape}")

        slice_summaries = tuple(getattr(result, "slice_summaries", ()) or ())
        if slice_summaries:
            converged_count = sum(item.converged for item in slice_summaries)
            worst_index = int(result.worst_slice_index)
            worst = slice_summaries[worst_index]
            self.results_panel.append_console("static convergence:")
            self.results_panel.append_console(
                f"  converged slices: {converged_count} / {len(slice_summaries)}"
            )
            self.results_panel.append_console(
                f"  max final residual RMS: {result.max_final_residual_rms:.8g}"
            )
            self.results_panel.append_console(
                f"  median final residual RMS: {result.median_final_residual_rms:.8g}"
            )
            self.results_panel.append_console(
                f"  max final residual max: {result.max_final_residual_max:.8g}"
            )
            self.results_panel.append_console(
                f"  worst slice: {worst.z_index} (z={worst.z_um:g} µm)"
            )
            self.results_panel.append_console(
                "  maximum iterations used: "
                f"{max(item.relaxation_iterations for item in slice_summaries)}"
            )
            self.results_panel.append_console(
                f"  maximum theta: {max(item.theta_max for item in slice_summaries):.8g}"
            )

        mode = getattr(result, "mode", None)
        if mode is not None:
            self.results_panel.append_console(f"mode:          {mode}")

        metrics = getattr(result, "metrics", None)
        if metrics:
            self.results_panel.append_console("metrics:")
            for key in ["converged", "beta", "physical_power_mW", "normalized_field_integral", "final_residual_rms", "final_residual_max", "n_points", "converged_count"]:
                if key in metrics:
                    self.results_panel.append_console(f"  {key}: {metrics[key]}")

        warnings = getattr(result, "warnings", None)
        if warnings:
            self.results_panel.append_console("warnings:")
            for w in warnings:
                self.results_panel.append_console(f"  - {w}")
