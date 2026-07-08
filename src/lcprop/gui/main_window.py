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
    ResultsPanel,
)
from lcprop.core.requests import StaticRunRequest, TimeDependentRunRequest, OutputOptions, RuntimeOptions

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
        self.results_panel = ResultsPanel()

        self.tabs.addTab(self.experiment_panel, "Experiment")
        self.tabs.addTab(self.physics_panel, "Physics")
        self.tabs.addTab(self.beam_panel, "Beam")
        self.tabs.addTab(self.grid_panel, "Grid")
        self.tabs.addTab(self.solver_panel, "Solver")
        self.tabs.addTab(self.results_panel, "Results")
        self.experiment_panel.experimentChanged.connect(self.update_run_button)
        self.update_run_button()   
        self.resize(1450, 900)

    def update_run_button(self) -> None:
        experiment = self.experiment_panel.current_experiment()
        self.run_button.setText(f"Run {experiment}")
        self.solver_panel.set_experiment_mode(experiment)

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

    def describe_request(self, req) -> str:
        ch = req.beams.channels[0]
        return "\n".join([
            f"Experiment: {self.experiment_panel.current_experiment()}",
            f"Runner: {self.runner.name}",
            f"Grid: {req.grid.Nx} × {req.grid.Ny}, Nz≈{round(req.grid.z_length_um / req.grid.dz_um)}",
            f"Aperture: x={req.grid.x_aperture_um:g} µm, y={req.grid.y_aperture_um:g} µm",
            f"z length: {req.grid.z_length_um:g} µm, dz={req.grid.dz_um:g} µm",
            f"Material: ne={req.material.ne:g}, no={req.material.no:g}, K={req.material.K:g}, Δε={req.material.delta_epsilon:g}",
            f"Bias: V={req.bias.V_bias:g} V, theta_bc={req.bias.theta_bc:g} rad",
            f"Beam: P={ch.power_mW:g} mW, waist={ch.waist_x_um:g} µm, λ={ch.wavelength_um:g} µm",
            f"Workflow: {req.solver.workflow.strategy}",
        ])

    def run_static_clicked(self):
        self.run_button.setEnabled(False)
        self.run_button.setText("Running…")
        self.tabs.setCurrentWidget(self.results_panel)

        try:
            experiment = self.experiment_panel.current_experiment()
            if experiment == "Time-dependent propagation":
                req = self.build_timedependent_request()
                run_label = "time-dependent workflow"
                runner_result_fn = self.runner.run_timedependent
            else:
                req = self.build_request()
                run_label = "static workflow"
                runner_result_fn = self.runner.run_static

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
            self.results_panel.append_console(f"method: {getattr(result, 'method', runner_result.kind)}")
            self.results_panel.append_console(
                "Nx, Ny, Nz: "
                f"{result.grid_summary['Nx']}, "
                f"{result.grid_summary['Ny']}, "
                f"{result.grid_summary['Nz']}"
            )
            self.results_panel.append_console(f"power_initial: {result.power_initial:.8g}")
            self.results_panel.append_console(f"power_final:   {result.power_final:.8g}")
            self.results_panel.append_console(f"A_final shape: {result.A_final.shape}")
            self.results_panel.append_console(f"theta shape:   {result.theta_final.shape}")

            if result.warnings:
                self.results_panel.append_console("warnings:")
                for w in result.warnings:
                    self.results_panel.append_console(f"  - {w}")

        except Exception:
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())

        finally:
            self.run_button.setEnabled(True)
            self.update_run_button()
