from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from lcprop.core.backend import BackendSpec
from lcprop.gui.panels.helpers import spin_box
from lcprop.pr.image_amplification import image_amplification_base_capabilities
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
from lcprop.pr.transverse.specs import PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticWorkflowOptions,
    PR_TRANSVERSE_STATIC_WORKFLOW,
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
        self.workflow.addItem(
            "Static (2D transverse zero-flux)",
            PR_TRANSVERSE_STATIC_WORKFLOW,
        )
        self.workflow.addItem(
            "Static (legacy x-only)",
            PR_STATIC_WORKFLOW,
        )
        self.workflow.addItem(
            "Time dependent (2D transverse) — GUI unavailable",
            PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
        )
        transverse_td_item = self.workflow.model().item(
            self.workflow.findData(PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW)
        )
        transverse_td_item.setEnabled(False)
        transverse_td_item.setToolTip(
            "The prepared-field adapter exists, but the GUI cannot yet "
            "construct the canonical transverse-TD request."
        )
        self.algorithm_status = QLabel()
        self.algorithm_status.setWordWrap(True)
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
        form.addRow("Validation status", self.algorithm_status)
        form.addRow("Material steps in segment", self.Nt)
        form.addRow("Normalized timestep", self.dt_normalized)
        form.addRow("Material integrator", self.integrator)
        form.addRow("Maximum coupled passes per slice", self.max_coupled_passes)
        form.addRow("Optical substeps per z slice", self.optical_substeps)
        form.addRow("Backend", self.backend)
        form.addRow("Precision", self.precision)
        self._form = form
        self._image_amplification_mode = False
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

    def set_image_amplification_mode(self, enabled: bool) -> None:
        """Present the canonical workflow selector as the base algorithm."""

        self._image_amplification_mode = bool(enabled)
        labels = {
            PR_TIMEDEPENDENT_WORKFLOW: "Reduced TD",
            PR_STATIC_WORKFLOW: "Static",
            PR_TRANSVERSE_STATIC_WORKFLOW: "Transverse Static",
            PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW: "Transverse TD",
        }
        statuses = {
            capability.workflow_id: capability.validation_status
            for capability in image_amplification_base_capabilities()
        }
        ordinary_labels = {
            PR_TIMEDEPENDENT_WORKFLOW: "Time dependent",
            PR_TRANSVERSE_STATIC_WORKFLOW: (
                "Static (2D transverse zero-flux)"
            ),
            PR_STATIC_WORKFLOW: "Static (legacy x-only)",
            PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW: (
                "Time dependent (2D transverse) — GUI unavailable"
            ),
        }
        for index in range(self.workflow.count()):
            workflow_id = str(self.workflow.itemData(index))
            if enabled:
                status = statuses[workflow_id]
                badge = (
                    "Validated"
                    if status == "compatible_and_validated"
                    else "Experimental"
                )
                unavailable = (
                    " — unavailable"
                    if workflow_id == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
                    else ""
                )
                text = f"{labels[workflow_id]} [{badge}{unavailable}]"
            else:
                text = ordinary_labels[workflow_id]
            self.workflow.setItemText(index, text)
        label = self._form.labelForField(self.workflow)
        if label is not None:
            label.setText("Algorithm" if enabled else "Workflow")
        self._refresh_workflow_controls()

    def image_amplification_validation_status(self) -> str | None:
        if not self._image_amplification_mode:
            return None
        return next(
            capability.validation_status
            for capability in image_amplification_base_capabilities()
            if capability.workflow_id == self.workflow_id()
        )

    def _set_row_visible(self, widget: QWidget, visible: bool) -> None:
        label = self._form.labelForField(widget)
        if label is not None:
            label.setVisible(visible)
        widget.setVisible(visible)

    def _refresh_workflow_controls(self, *_args) -> None:
        is_time_dependent = self.workflow_id() == PR_TIMEDEPENDENT_WORKFLOW
        is_transverse_static = (
            self.workflow_id() == PR_TRANSVERSE_STATIC_WORKFLOW
        )
        for widget in (self.Nt, self.dt_normalized, self.integrator):
            self._set_row_visible(widget, is_time_dependent)
        self._set_row_visible(
            self.max_coupled_passes,
            not is_time_dependent,
        )
        static_iterations_label = self._form.labelForField(
            self.max_coupled_passes
        )
        if static_iterations_label is not None:
            static_iterations_label.setText(
                "Maximum coupled iterations"
                if is_transverse_static
                else "Maximum coupled passes per slice"
            )
        self._set_row_visible(
            self.algorithm_status,
            self._image_amplification_mode,
        )
        if self._image_amplification_mode:
            status = self.image_amplification_validation_status()
            if status == "compatible_and_validated":
                text = "Validated for the Image Amplification experiment."
            else:
                text = (
                    "Experimental: compatible architecture; specialized "
                    "Image Amplification validation is pending."
                )
            if self.workflow_id() == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW:
                text += (
                    " Selection is disabled because the GUI cannot construct "
                    "its canonical prepared-field request."
                )
            self.algorithm_status.setText(text)

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

    def transverse_static_solver(self) -> PRTransverseStaticWorkflowOptions:
        """Return the bounded 2D zero-flux policy represented by the GUI."""

        return PRTransverseStaticWorkflowOptions(
            max_coupled_iterations=self.max_coupled_passes.value(),
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

    def set_transverse_static_solver(
        self,
        solver: PRTransverseStaticWorkflowOptions,
    ) -> None:
        solver.validate()
        represented = PRTransverseStaticWorkflowOptions(
            max_coupled_iterations=solver.max_coupled_iterations,
            optical_substeps=solver.optical_substeps,
        )
        if solver != represented:
            raise ValueError(
                "PR GUI cannot represent explicit transverse static solver "
                "policy overrides"
            )
        self.max_coupled_passes.setValue(solver.max_coupled_iterations)
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
