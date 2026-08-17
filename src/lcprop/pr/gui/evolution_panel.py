from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QVBoxLayout,
    QWidget,
)

from lcprop.core.backend import BackendSpec
from lcprop.gui.panels.helpers import spin_box
from lcprop.pr.specs import (
    PRSolverOptions,
    PR_EULER_INTEGRATOR,
    PR_SEMI_IMPLICIT_INTEGRATOR,
    PR_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.static_workflow import (
    PRStaticWorkflowOptions,
    PR_STATIC_WORKFLOW,
)


class PREvolutionPanel(QWidget):
    """Workflow, evolution, optical-substep, and backend controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        defaults = PRSolverOptions(
            Nt=10,
            dt_normalized=1e-3,
            optical_substeps=1,
            integrator=PR_SEMI_IMPLICIT_INTEGRATOR,
        )
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.workflow = QComboBox()
        self.workflow.addItem("Time dependent", PR_TIMEDEPENDENT_WORKFLOW)
        self.workflow.addItem("Static (self-consistent)", PR_STATIC_WORKFLOW)
        self.Nt = spin_box(0, 1_000_000_000, defaults.Nt)
        self.dt_normalized = QDoubleSpinBox()
        self.dt_normalized.setRange(1e-12, 1e6)
        self.dt_normalized.setDecimals(12)
        self.dt_normalized.setValue(defaults.dt_normalized)
        self.optical_substeps = spin_box(1, 1_000_000, defaults.optical_substeps)
        self.integrator = QComboBox()
        self.integrator.addItem(
            "Semi-implicit trapezoidal",
            PR_SEMI_IMPLICIT_INTEGRATOR,
        )
        self.integrator.addItem("Explicit Euler (reference)", PR_EULER_INTEGRATOR)
        self.backend = QComboBox()
        self.backend.addItems(("numpy", "auto", "cupy"))
        self.precision = QComboBox()
        self.precision.addItems(("float64", "float32"))
        self.max_coupled_passes = spin_box(1, 1_000_000, 20)

        form.addRow("Workflow", self.workflow)
        form.addRow("Material steps in segment", self.Nt)
        form.addRow("Normalized timestep", self.dt_normalized)
        form.addRow("Material integrator", self.integrator)
        form.addRow("Maximum coupled passes per slice", self.max_coupled_passes)
        form.addRow("Optical substeps per z slice", self.optical_substeps)
        form.addRow("Backend", self.backend)
        form.addRow("Precision", self.precision)
        self._form = form
        self.workflow.currentIndexChanged.connect(
            self._refresh_workflow_controls
        )
        self._refresh_workflow_controls()
        layout.addLayout(form)
        layout.addStretch(1)

    def workflow_id(self) -> str:
        return str(self.workflow.currentData())

    def set_workflow_id(self, workflow_id: str) -> None:
        index = self.workflow.findData(workflow_id)
        if index < 0:
            raise ValueError(f"unsupported PR GUI workflow: {workflow_id!r}")
        self.workflow.setCurrentIndex(index)

    def _set_row_visible(self, widget: QWidget, visible: bool) -> None:
        label = self._form.labelForField(widget)
        if label is not None:
            label.setVisible(visible)
        widget.setVisible(visible)

    def _refresh_workflow_controls(self, *_args) -> None:
        is_time_dependent = self.workflow_id() == PR_TIMEDEPENDENT_WORKFLOW
        for widget in (self.Nt, self.dt_normalized, self.integrator):
            self._set_row_visible(widget, is_time_dependent)
        self._set_row_visible(
            self.max_coupled_passes,
            not is_time_dependent,
        )

    def solver(self) -> PRSolverOptions:
        return PRSolverOptions(
            Nt=self.Nt.value(),
            dt_normalized=self.dt_normalized.value(),
            optical_substeps=self.optical_substeps.value(),
            integrator=self.integrator.currentData(),
        )

    def static_solver(self) -> PRStaticWorkflowOptions:
        """Return the minimal static policy represented by the GUI.

        The material solver remains ``None`` so the static workflow resolves
        its documented precision-aware tolerances.
        """

        return PRStaticWorkflowOptions(
            material_solver=None,
            max_coupled_passes=self.max_coupled_passes.value(),
            optical_substeps=self.optical_substeps.value(),
        )

    def backend_spec(self) -> BackendSpec:
        return BackendSpec(
            backend=self.backend.currentText(),
            precision=self.precision.currentText(),
            verbose=False,
        )

    def set_solver(self, solver: PRSolverOptions) -> None:
        solver.validate()
        self.Nt.setValue(solver.Nt)
        self.dt_normalized.setValue(solver.dt_normalized)
        self.optical_substeps.setValue(solver.optical_substeps)
        integrator_index = self.integrator.findData(solver.integrator)
        if integrator_index < 0:
            raise ValueError("unsupported PR GUI integrator")
        self.integrator.setCurrentIndex(integrator_index)

    def set_static_solver(self, solver: PRStaticWorkflowOptions) -> None:
        solver.validate()
        if solver.material_solver is not None:
            raise ValueError(
                "PR GUI cannot represent an explicit static material solver"
            )
        self.max_coupled_passes.setValue(solver.max_coupled_passes)
        self.optical_substeps.setValue(solver.optical_substeps)

    def set_backend_spec(self, backend: BackendSpec) -> None:
        backend.validate()
        backend_index = self.backend.findText(backend.backend)
        precision_index = self.precision.findText(backend.precision)
        if backend_index < 0 or precision_index < 0:
            raise ValueError("unsupported PR GUI backend specification")
        self.backend.setCurrentIndex(backend_index)
        self.precision.setCurrentIndex(precision_index)


__all__ = ["PREvolutionPanel"]
