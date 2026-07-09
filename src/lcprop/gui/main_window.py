from __future__ import annotations

import traceback

from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTabWidget,
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
        self.setWindowTitle("LCProp")
        # Default to a wide scientific-visualization layout.
        self.setMinimumSize(1200, 760)

        root = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("LCProp"))
        header.addStretch(1)
        header.addWidget(QLabel(f"Runner: {self.runner.name}"))

        self.run_button = QPushButton()
        self.run_button.clicked.connect(self.run_static_clicked)
        header.addWidget(self.run_button)

        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.experiment_panel = ExperimentPanel()
        self.physics_panel = PhysicsPanel()
        self.beam_panel = BeamPanel()
        self.grid_panel = GridPanel()
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
        self.update_run_button()   
        self.resize(1450, 900)

    def update_run_button(self) -> None:
        experiment = self.experiment_panel.current_experiment()
        self.run_button.setText(f"Run {experiment}")
        self.solver_panel.set_experiment_mode(experiment)
        self._update_sweep_tab(experiment)

    def _update_sweep_tab(self, experiment: str) -> None:
        sweep_enabled = experiment == "Soliton existence curve"
        self.tabs.setTabEnabled(self.sweep_tab_index, sweep_enabled)
        if not sweep_enabled and self.tabs.currentIndex() == self.sweep_tab_index:
            self.tabs.setCurrentWidget(self.experiment_panel)

    def build_request(self) -> StaticRunRequest:
        return StaticRunRequest(
            grid=self.grid_panel.grid(),
            material=self.physics_panel.material(),
            bias=self.physics_panel.bias(),
            beams=self.beam_panel.beams(),
            solver=self.solver_panel.solver(),
            output=OutputOptions(),
        )

    def build_timedependent_request(self) -> TimeDependentRunRequest:
        return TimeDependentRunRequest(
            grid=self.grid_panel.grid(),
            material=self.physics_panel.material(),
            bias=self.physics_panel.bias(),
            beams=self.beam_panel.beams(),
            solver=self.solver_panel.td_solver(),
            output=OutputOptions(),
            runtime=RuntimeOptions(precision="float64"),
        )

    def build_soliton_request(self) -> SolitonRequest:
        return SolitonRequest(
            base=self.build_request(),
            mode=self.solver_panel.soliton_mode(),
        )

    def build_soliton_existence_request(self) -> ParameterSweepRequest:
        base = SolitonRequest(
            base=self.build_request(),
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
        ch = base_req.beams.channels[0]
        lines = [
            f"Experiment: {self.experiment_panel.current_experiment()}",
            f"Runner: {self.runner.name}",
            f"Grid: {base_req.grid.Nx} × {base_req.grid.Ny}, Nz≈{round(base_req.grid.z_length_um / base_req.grid.dz_um)}",
            f"Aperture: x={base_req.grid.x_aperture_um:g} µm, y={base_req.grid.y_aperture_um:g} µm",
            f"z length: {base_req.grid.z_length_um:g} µm, dz={base_req.grid.dz_um:g} µm",
            f"Material: ne={base_req.material.ne:g}, no={base_req.material.no:g}, K={base_req.material.K:g}, Δε={base_req.material.delta_epsilon:g}",
            f"Bias: V={base_req.bias.V_bias:g} V, theta_bc={base_req.bias.theta_bc:g} rad",
            f"Beam: P={ch.power_mW:g} mW, waist={ch.waist_x_um:g} µm, λ={ch.wavelength_um:g} µm",
            f"Workflow: {base_req.solver.workflow.strategy}",
        ]

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
            self.results_panel.append_console(f"power_initial: {result.power_initial:.8g}")
        if hasattr(result, "power_final"):
            self.results_panel.append_console(f"power_final:   {result.power_final:.8g}")

        A = getattr(result, "A_final", getattr(result, "A", None))
        theta = getattr(result, "theta_final", getattr(result, "theta", None))
        if A is not None:
            self.results_panel.append_console(f"A shape:       {A.shape}")
        if theta is not None:
            self.results_panel.append_console(f"theta shape:   {theta.shape}")

        mode = getattr(result, "mode", None)
        if mode is not None:
            self.results_panel.append_console(f"mode:          {mode}")

        metrics = getattr(result, "metrics", None)
        if metrics:
            self.results_panel.append_console("metrics:")
            for key in ["converged", "beta", "target_power", "final_residual_rms", "final_residual_max", "n_points", "converged_count"]:
                if key in metrics:
                    self.results_panel.append_console(f"  {key}: {metrics[key]}")

        warnings = getattr(result, "warnings", None)
        if warnings:
            self.results_panel.append_console("warnings:")
            for w in warnings:
                self.results_panel.append_console(f"  - {w}")