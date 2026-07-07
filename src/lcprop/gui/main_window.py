from __future__ import annotations

import traceback

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QTextEdit, QDoubleSpinBox, QSpinBox, QComboBox,
    QLabel, QTabWidget,
)

from lcprop.core.context import GridSpec, LCMaterial, BiasSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.requests import StaticRunRequest, StaticSolverOptions, StaticWorkflowOptions, OutputOptions
from lcprop.runners.local import LocalRunner


class LCPropMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.runner = LocalRunner()
        self.setWindowTitle("LCProp")

        root = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("LCProp"))
        header.addStretch(1)
        header.addWidget(QLabel(f"Runner: {self.runner.name}"))
        self.run_button = QPushButton("Run static")
        self.run_button.clicked.connect(self.run_static_clicked)
        header.addWidget(self.run_button)
        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.tabs.addTab(self._build_experiment_tab(), "Experiment")
        self.tabs.addTab(self._build_physics_tab(), "Physics")
        self.tabs.addTab(self._build_beam_tab(), "Beam")
        self.tabs.addTab(self._build_grid_tab(), "Grid")
        self.tabs.addTab(self._build_solver_tab(), "Solver")
        self.tabs.addTab(self._build_results_tab(), "Results")

    def _spin(self, lo, hi, value):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(value)
        return w

    def _dspin(self, lo, hi, value):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(6)
        w.setValue(value)
        return w

    def _build_experiment_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.experiment = QComboBox()
        self.experiment.addItems([
            "Static propagation",
            "Time-dependent propagation",
            "Soliton",
            "Soliton existence curve",
        ])
        form = QFormLayout()
        form.addRow("Experiment", self.experiment)
        layout.addLayout(form)
        layout.addWidget(QLabel("Current milestone: Static propagation is wired to LocalRunner."))
        layout.addStretch(1)
        return page

    def _build_physics_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()

        self.ne = self._dspin(0.1, 10.0, 1.7)
        self.no = self._dspin(0.1, 10.0, 1.5)
        self.K = self._dspin(1e-15, 1e-9, 7e-12)
        self.K.setDecimals(15)
        self.delta_epsilon = self._dspin(0.0, 100.0, 13.0)
        self.V_bias = self._dspin(0.0, 100.0, 0.9144)
        self.theta_bc = self._dspin(0.0, 1.5708, 0.0)

        form.addRow("ne", self.ne)
        form.addRow("no", self.no)
        form.addRow("K (N)", self.K)
        form.addRow("Δε", self.delta_epsilon)
        form.addRow("V bias (V)", self.V_bias)
        form.addRow("theta_bc (rad)", self.theta_bc)
        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def _build_beam_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()

        self.wavelength_um = self._dspin(0.1, 10.0, 0.633)
        self.power_mW = self._dspin(0.0, 1000.0, 1.0)
        self.waist_um = self._dspin(0.1, 1000.0, 3.0)

        form.addRow("Wavelength (µm)", self.wavelength_um)
        form.addRow("Power (mW)", self.power_mW)
        form.addRow("Waist (µm)", self.waist_um)
        layout.addLayout(form)
        layout.addWidget(QLabel("Multichannel beam editor will be added here."))
        layout.addStretch(1)
        return page

    def _build_grid_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()

        self.Nx = self._spin(16, 4096, 64)
        self.Ny = self._spin(16, 4096, 64)
        self.dz_um = self._dspin(0.1, 1000.0, 20.0)
        self.z_length_um = self._dspin(1.0, 100000.0, 3000.0)
        self.x_aperture_um = self._dspin(1.0, 10000.0, 75.0)
        self.y_aperture_um = self._dspin(1.0, 10000.0, 100.0)

        form.addRow("Nx", self.Nx)
        form.addRow("Ny", self.Ny)
        form.addRow("dz (µm)", self.dz_um)
        form.addRow("z length (µm)", self.z_length_um)
        form.addRow("x aperture (µm)", self.x_aperture_um)
        form.addRow("y aperture (µm)", self.y_aperture_um)
        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def _build_solver_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        form = QFormLayout()

        self.workflow = QComboBox()
        self.workflow.addItems(["fixed_theta", "local_self_consistent"])
        self.max_iterations = self._spin(1, 1000, 3)

        form.addRow("Static workflow", self.workflow)
        form.addRow("Max iterations", self.max_iterations)
        layout.addLayout(form)
        layout.addStretch(1)
        return page

    def _build_results_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.request_summary = QTextEdit()
        self.request_summary.setReadOnly(True)
        self.output = QTextEdit()
        self.output.setReadOnly(True)

        layout.addWidget(QLabel("Request summary"))
        layout.addWidget(self.request_summary)
        layout.addWidget(QLabel("Run console"))
        layout.addWidget(self.output)
        return page

    def build_request(self) -> StaticRunRequest:
        if self.workflow.currentText() == "fixed_theta":
            workflow = StaticWorkflowOptions(
                strategy="fixed_theta",
                theta_solver="none",
                optics_solver="splitstep",
                coupling="frozen",
            )
        else:
            workflow = StaticWorkflowOptions(
                strategy="local_self_consistent",
                theta_solver="picard_cn",
                optics_solver="splitstep",
                coupling="self_consistent",
            )

        return StaticRunRequest(
            grid=GridSpec(
                Nx=self.Nx.value(),
                Ny=self.Ny.value(),
                dz_um=self.dz_um.value(),
                x_aperture_um=self.x_aperture_um.value(),
                y_aperture_um=self.y_aperture_um.value(),
                z_length_um=self.z_length_um.value(),
            ),
            material=LCMaterial(
                ne=self.ne.value(),
                no=self.no.value(),
                K=self.K.value(),
                delta_epsilon=self.delta_epsilon.value(),
            ),
            bias=BiasSpec(V_bias=self.V_bias.value(), theta_bc=self.theta_bc.value()),
            beams=BeamStack(
                channels=(
                    BeamChannel(
                        wavelength_um=self.wavelength_um.value(),
                        power_mW=self.power_mW.value(),
                        waist_x_um=self.waist_um.value(),
                        waist_y_um=self.waist_um.value(),
                    ),
                )
            ),
            solver=StaticSolverOptions(
                workflow=workflow,
                max_iterations=self.max_iterations.value(),
            ),
            output=OutputOptions(),
        )

    def describe_request(self, req: StaticRunRequest) -> str:
        ch = req.beams.channels[0]
        return "\n".join([
            "Experiment: Static propagation",
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
        self.tabs.setCurrentIndex(5)

        try:
            req = self.build_request()
            self.request_summary.setPlainText(self.describe_request(req))
            self.output.append(f"Running static workflow with {self.runner.name}...")
            QApplication.processEvents()

            runner_result = self.runner.run_static(req)
            result = runner_result.result

            self.output.append("")
            self.output.append(runner_result.message)
            self.output.append("Run complete")
            self.output.append(f"method: {result.method}")
            self.output.append(f"Nx, Ny, Nz: {result.grid_summary['Nx']}, {result.grid_summary['Ny']}, {result.grid_summary['Nz']}")
            self.output.append(f"power_initial: {result.power_initial:.8g}")
            self.output.append(f"power_final:   {result.power_final:.8g}")
            self.output.append(f"A_final shape: {result.A_final.shape}")
            self.output.append(f"theta shape:   {result.theta_final.shape}")

            if result.warnings:
                self.output.append("warnings:")
                for w in result.warnings:
                    self.output.append(f"  - {w}")

        except Exception:
            self.output.append("ERROR")
            self.output.append(traceback.format_exc())

        finally:
            self.run_button.setEnabled(True)
