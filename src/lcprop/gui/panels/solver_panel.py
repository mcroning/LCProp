from PySide6.QtWidgets import QComboBox, QFormLayout, QVBoxLayout, QWidget

from lcprop.core.requests import StaticSolverOptions, StaticWorkflowOptions, TimeDependentSolverOptions
from lcprop.gui.panels.helpers import spin_box


class SolverPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.workflow = QComboBox()
        self.workflow.addItems(["fixed_theta", "local_self_consistent"])
        self.max_iterations = spin_box(1, 1000, 3)
        self.Nt = spin_box(1, 100000, 2)
        self.dt = spin_box(1, 1000000, 750)

        form.addRow("Static workflow", self.workflow)
        form.addRow("Max iterations", self.max_iterations)
        form.addRow("TD Nt", self.Nt)
        form.addRow("TD dt × 1e6", self.dt)

        layout.addLayout(form)
        layout.addStretch(1)

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
        )


    def td_solver(self) -> TimeDependentSolverOptions:
        return TimeDependentSolverOptions(
            Nt=self.Nt.value(),
            dt=self.dt.value() * 1e-6,
            gamma_z=0.0,
        )
