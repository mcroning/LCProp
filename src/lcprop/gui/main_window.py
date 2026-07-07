from __future__ import annotations

import traceback

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QFormLayout, QPushButton, QTextEdit, QDoubleSpinBox, QSpinBox, QComboBox, QLabel

from lcprop.core.context import GridSpec, LCMaterial, BiasSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.requests import StaticRunRequest, StaticSolverOptions, StaticWorkflowOptions, OutputOptions
from lcprop.runners.local import LocalRunner


class LCPropMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.runner = LocalRunner()
        self.setWindowTitle("LCProp")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("LCProp — Static Workflow"))

        form = QFormLayout()
        self.Nx = self._spin(16, 4096, 64)
        self.Ny = self._spin(16, 4096, 64)
        self.dz_um = self._dspin(0.1, 1000.0, 20.0)
        self.z_length_um = self._dspin(1.0, 100000.0, 3000.0)
        self.x_aperture_um = self._dspin(1.0, 10000.0, 75.0)
        self.y_aperture_um = self._dspin(1.0, 10000.0, 100.0)
        self.power_mW = self._dspin(0.0, 1000.0, 1.0)
        self.waist_um = self._dspin(0.1, 1000.0, 3.0)
        self.V_bias = self._dspin(0.0, 100.0, 0.9144)
        self.theta_bc = self._dspin(0.0, 1.5708, 0.0)

        self.workflow = QComboBox()
        self.workflow.addItems(["fixed_theta", "local_self_consistent"])
        self.max_iterations = self._spin(1, 1000, 3)

        for label, widget in [
            ("Nx", self.Nx), ("Ny", self.Ny), ("dz (µm)", self.dz_um),
            ("z length (µm)", self.z_length_um), ("x aperture (µm)", self.x_aperture_um),
            ("y aperture (µm)", self.y_aperture_um), ("Power (mW)", self.power_mW),
            ("Waist (µm)", self.waist_um), ("V bias (V)", self.V_bias),
            ("theta_bc (rad)", self.theta_bc), ("Workflow", self.workflow),
            ("Max iterations", self.max_iterations),
        ]:
            form.addRow(label, widget)

        layout.addLayout(form)

        self.run_button = QPushButton("Run static")
        self.run_button.clicked.connect(self.run_static_clicked)
        layout.addWidget(self.run_button)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        layout.addWidget(self.output)

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

    def build_request(self) -> StaticRunRequest:
        if self.workflow.currentText() == "fixed_theta":
            workflow = StaticWorkflowOptions(strategy="fixed_theta", theta_solver="none", optics_solver="splitstep", coupling="frozen")
        else:
            workflow = StaticWorkflowOptions(strategy="local_self_consistent", theta_solver="picard_cn", optics_solver="splitstep", coupling="self_consistent")

        return StaticRunRequest(
            grid=GridSpec(
                Nx=self.Nx.value(), Ny=self.Ny.value(), dz_um=self.dz_um.value(),
                x_aperture_um=self.x_aperture_um.value(),
                y_aperture_um=self.y_aperture_um.value(),
                z_length_um=self.z_length_um.value(),
            ),
            material=LCMaterial(ne=1.7, no=1.5, K=7e-12, delta_epsilon=13.0),
            bias=BiasSpec(V_bias=self.V_bias.value(), theta_bc=self.theta_bc.value()),
            beams=BeamStack(channels=(BeamChannel(wavelength_um=0.633, power_mW=self.power_mW.value(), waist_x_um=self.waist_um.value(), waist_y_um=self.waist_um.value()),)),
            solver=StaticSolverOptions(workflow=workflow, max_iterations=self.max_iterations.value()),
            output=OutputOptions(),
        )

    def run_static_clicked(self):
        self.run_button.setEnabled(False)
        self.output.append(f"Running static workflow with {self.runner.name}...")
        QApplication.processEvents()

        try:
            runner_result = self.runner.run_static(self.build_request())
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
        except Exception:
            self.output.append("ERROR")
            self.output.append(traceback.format_exc())
        finally:
            self.run_button.setEnabled(True)
