from PySide6.QtWidgets import QCheckBox, QComboBox, QFormLayout, QVBoxLayout, QWidget

from lcprop.core.requests import StaticSolverOptions, StaticWorkflowOptions, TimeDependentSolverOptions
from lcprop.gui.panels.helpers import spin_box


class SolverPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.workflow = QComboBox()
        self.workflow.addItems(["local_self_consistent", "fixed_theta"])
        self.workflow.setCurrentText("local_self_consistent")
        self.soliton_mode_selector = QComboBox()
        self.soliton_mode_selector.addItem("Fundamental (00)", "00")
        self.soliton_mode_selector.addItem("Dipole X (10)", "10")
        self.soliton_mode_selector.addItem("Dipole Y (01)", "01")
        self.soliton_mode_selector.addItem("Quadrupole (11)", "11")
        self.refine_transverse_checkbox = QCheckBox(
            "Refine with transverse eigensolver"
        )
        self.refine_transverse_checkbox.setChecked(False)
        self.max_iterations = spin_box(1, 1000, 3)
        self.Nt = spin_box(1, 100000, 2)
        self.dt = spin_box(1, 1000000, 750)

        form.addRow("Static workflow", self.workflow)
        form.addRow("Soliton mode", self.soliton_mode_selector)
        form.addRow(self.refine_transverse_checkbox)
        form.addRow("Max coupled passes", self.max_iterations)
        form.addRow("TD Nt", self.Nt)
        form.addRow("TD dt × 1e6", self.dt)

        layout.addLayout(form)
        layout.addStretch(1)

    def set_experiment_mode(self, experiment: str) -> None:
        """Show only controls relevant to the selected experiment."""
        is_td = experiment == "Time-dependent propagation"
        self.workflow.setEnabled(not is_td)
        self.max_iterations.setEnabled(not is_td)
        is_soliton = experiment in {"Soliton", "Soliton existence curve"}
        self.soliton_mode_selector.setEnabled(is_soliton)
        self.refine_transverse_checkbox.setEnabled(experiment == "Soliton")

    def soliton_mode(self) -> str:
        """Return the selected soliton mode code."""
        return str(self.soliton_mode_selector.currentData())

    def refine_transverse(self) -> bool:
        """Return whether single-soliton transverse refinement is enabled."""
        return self.refine_transverse_checkbox.isChecked()

    def solver(self) -> StaticSolverOptions:
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

        return StaticSolverOptions(
            workflow=workflow,
            max_iterations=self.max_iterations.value(),
            static_max_coupled_passes=self.max_iterations.value(),
        )


    def td_solver(self) -> TimeDependentSolverOptions:
        return TimeDependentSolverOptions(
            Nt=self.Nt.value(),
            dt=self.dt.value() * 1e-6,
            gamma_z=0.0,
        )
