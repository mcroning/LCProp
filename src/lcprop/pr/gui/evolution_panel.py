from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
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
    PR_EXACT_MODAL_INTEGRATOR,
    PR_SEMI_IMPLICIT_INTEGRATOR,
    PR_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.scattering import (
    PR_CANONICAL_SCATTERING_V1,
    PR_CANONICAL_SCATTERING_V2,
    PRCanonicalScatteringSpec,
)
from lcprop.pr.static_workflow import (
    PRStaticWorkflowOptions,
    PR_STATIC_WORKFLOW,
)
from lcprop.pr.transverse.specs import (
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PR_MATERIAL_RESPONSE_NONLINEAR,
    PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE,
    PR_TRANSVERSE_IMEX_EULER,
    PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
    PRTransverseMaterialResponseSpec,
    PRTransverseSolverOptions,
)
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

        self.evolution = QComboBox()
        self.evolution.addItem("Static", "static")
        self.evolution.addItem("Time dependent", "time_dependent")
        self.evolution.setCurrentIndex(1)
        self.transport_model = QComboBox()
        self.transport_model.addItem(
            "Reduced x-only (x drift/diffusion)", "reduced_x"
        )
        self.transport_model.addItem(
            "Full transverse (x-y drift/diffusion)", "full_transverse"
        )
        # Retained as the compatibility-facing workflow identity. The visible
        # controls above expose the three independent scientific model axes.
        self.workflow = QComboBox()
        self.workflow.addItem("Time dependent", PR_TIMEDEPENDENT_WORKFLOW)
        self.workflow.addItem(
            "Static — Full transverse PR transport",
            PR_TRANSVERSE_STATIC_WORKFLOW,
        )
        self.workflow.addItem(
            "Static — Reduced x-only PR transport",
            PR_STATIC_WORKFLOW,
        )
        self.workflow.addItem(
            "Time dependent — Full transverse PR transport",
            PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
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
        self.material_response = QComboBox()
        self.material_response.addItem(
            "Fully nonlinear", PR_MATERIAL_RESPONSE_NONLINEAR
        )
        self.material_response.addItem(
            "Linearized",
            PR_MATERIAL_RESPONSE_LINEARIZED,
        )
        self.reference_intensity = QDoubleSpinBox()
        self.reference_intensity.setRange(1e-12, 1e9)
        self.reference_intensity.setDecimals(12)
        self.reference_intensity.setValue(1.0)
        self.transverse_applied_field = QDoubleSpinBox()
        self.transverse_applied_field.setRange(-1e9, 1e9)
        self.transverse_applied_field.setDecimals(12)
        self.transverse_applied_field.setValue(0.0)

        self.scattering_enabled = QCheckBox("Enable canonical volume scattering")
        self.scattering_epsilon = QDoubleSpinBox()
        self.scattering_epsilon.setRange(0.0, 1.0e9)
        self.scattering_epsilon.setDecimals(12)
        self.scattering_epsilon.setValue(1.0e-8)
        self.scattering_correlation_um = QDoubleSpinBox()
        self.scattering_correlation_um.setRange(1.0e-12, 1.0e9)
        self.scattering_correlation_um.setDecimals(12)
        self.scattering_correlation_um.setValue(2.0)
        self.scattering_seed = QDoubleSpinBox()
        self.scattering_seed.setRange(0.0, float(2**32 - 1))
        self.scattering_seed.setDecimals(0)
        self.scattering_seed.setValue(0.0)
        self.scattering_canonical_dz_um = QDoubleSpinBox()
        self.scattering_canonical_dz_um.setRange(1.0e-12, 1.0e9)
        self.scattering_canonical_dz_um.setDecimals(12)
        self.scattering_canonical_dz_um.setValue(1.0)
        self.scattering_algorithm = QComboBox()
        self.scattering_algorithm.addItem(
            "Canonical phase slabs v1", PR_CANONICAL_SCATTERING_V1
        )
        self.scattering_algorithm.addItem(
            "Canonical phase slabs v2 (cross-backend)",
            PR_CANONICAL_SCATTERING_V2,
        )
        self.execution_guidance = QLabel()
        self.execution_guidance.setWordWrap(True)

        form.addRow("Evolution", self.evolution)
        form.addRow("Transport", self.transport_model)
        form.addRow("Algorithm", self.workflow)
        form.addRow("Validation status", self.algorithm_status)
        form.addRow("Material response", self.material_response)
        form.addRow(
            "Linearization intensity I₀ (normalized total transport intensity)",
            self.reference_intensity,
        )
        form.addRow(
            "Transverse applied mean field (normalized)",
            self.transverse_applied_field,
        )
        form.addRow("Material steps in segment", self.Nt)
        form.addRow("Normalized timestep", self.dt_normalized)
        form.addRow("Material integrator", self.integrator)
        form.addRow("Maximum coupled passes per slice", self.max_coupled_passes)
        form.addRow("Optical substeps per z slice", self.optical_substeps)
        form.addRow("Backend", self.backend)
        form.addRow("Precision", self.precision)
        form.addRow("Scattering", self.scattering_enabled)
        form.addRow("Scattering strength ε", self.scattering_epsilon)
        form.addRow(
            "Scattering correlation length (µm)",
            self.scattering_correlation_um,
        )
        form.addRow("Scattering seed", self.scattering_seed)
        form.addRow(
            "Scattering canonical slab Δz (µm)",
            self.scattering_canonical_dz_um,
        )
        form.addRow("Scattering model", self.scattering_algorithm)
        form.addRow("Execution guidance", self.execution_guidance)
        self._form = form
        self._image_amplification_mode = False
        self._syncing_model_controls = False
        self._implicit_local_backend = "numpy"
        self._backend_origin = "implicit_local_default"
        self.workflow.currentIndexChanged.connect(
            self._workflow_identity_changed
        )
        self.evolution.currentIndexChanged.connect(
            self._model_axis_changed
        )
        self.transport_model.currentIndexChanged.connect(
            self._model_axis_changed
        )
        self.material_response.currentIndexChanged.connect(
            self._refresh_workflow_controls
        )
        self.scattering_enabled.toggled.connect(
            self._refresh_workflow_controls
        )
        self.backend.currentIndexChanged.connect(self._backend_selected)
        self.backend.activated.connect(self._backend_selected)
        self._sync_workflow_from_model_axes()
        self._refresh_workflow_controls()
        layout.addLayout(form)
        layout.addStretch(1)

    def workflow_id(self) -> str:
        return str(self.workflow.currentData())

    @staticmethod
    def _workflow_for_axes(evolution: str, transport: str) -> str:
        return {
            ("static", "reduced_x"): PR_STATIC_WORKFLOW,
            ("static", "full_transverse"): PR_TRANSVERSE_STATIC_WORKFLOW,
            ("time_dependent", "reduced_x"): PR_TIMEDEPENDENT_WORKFLOW,
            ("time_dependent", "full_transverse"):
                PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
        }[(evolution, transport)]

    @staticmethod
    def _axes_for_workflow(workflow_id: str) -> tuple[str, str]:
        return {
            PR_STATIC_WORKFLOW: ("static", "reduced_x"),
            PR_TRANSVERSE_STATIC_WORKFLOW: ("static", "full_transverse"),
            PR_TIMEDEPENDENT_WORKFLOW: ("time_dependent", "reduced_x"),
            PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW: (
                "time_dependent",
                "full_transverse",
            ),
        }[workflow_id]

    def _sync_workflow_from_model_axes(self) -> None:
        workflow_id = self._workflow_for_axes(
            str(self.evolution.currentData()),
            str(self.transport_model.currentData()),
        )
        index = self.workflow.findData(workflow_id)
        if index < 0:  # pragma: no cover - fixed internal matrix
            raise ValueError(f"unsupported PR GUI workflow: {workflow_id!r}")
        self.workflow.setCurrentIndex(index)

    def _model_axis_changed(self, *_args) -> None:
        if self._syncing_model_controls:
            return
        self._syncing_model_controls = True
        try:
            self._sync_workflow_from_model_axes()
        finally:
            self._syncing_model_controls = False
        self._refresh_workflow_controls()

    def _workflow_identity_changed(self, *_args) -> None:
        if self._syncing_model_controls:
            return
        self._syncing_model_controls = True
        try:
            evolution, transport = self._axes_for_workflow(self.workflow_id())
            self.evolution.setCurrentIndex(self.evolution.findData(evolution))
            self.transport_model.setCurrentIndex(
                self.transport_model.findData(transport)
            )
        finally:
            self._syncing_model_controls = False
        self._refresh_workflow_controls()

    def set_workflow_id(self, workflow_id: str) -> None:
        index = self.workflow.findData(workflow_id)
        if index < 0:
            raise ValueError(f"unsupported PR GUI workflow: {workflow_id!r}")
        self.workflow.setCurrentIndex(index)
        self._workflow_identity_changed()

    def set_image_amplification_mode(self, enabled: bool) -> None:
        """Present the canonical workflow selector as the base algorithm."""

        self._image_amplification_mode = bool(enabled)
        self._refresh_workflow_labels()
        self._refresh_workflow_controls()

    def _image_amplification_status_for_workflow(self, workflow_id: str) -> str:
        status = next(
            capability.validation_status
            for capability in image_amplification_base_capabilities()
            if capability.workflow_id == workflow_id
        )
        if (
            self.material_response.currentData()
            == PR_MATERIAL_RESPONSE_LINEARIZED
        ):
            return "compatible_validation_pending"
        return status

    def _refresh_workflow_labels(self) -> None:
        labels = {
            PR_TIMEDEPENDENT_WORKFLOW: "Reduced TD",
            PR_STATIC_WORKFLOW: "Static",
            PR_TRANSVERSE_STATIC_WORKFLOW: "Transverse Static",
            PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW: "Transverse TD",
        }
        ordinary_labels = {
            PR_TIMEDEPENDENT_WORKFLOW: "Time dependent",
            PR_TRANSVERSE_STATIC_WORKFLOW: (
                "Static — Full transverse PR transport"
            ),
            PR_STATIC_WORKFLOW: "Static — Reduced x-only PR transport",
            PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW: (
                "Time dependent — Full transverse PR transport"
            ),
        }
        for index in range(self.workflow.count()):
            workflow_id = str(self.workflow.itemData(index))
            if self._image_amplification_mode:
                status = self._image_amplification_status_for_workflow(workflow_id)
                badge = (
                    "Validated"
                    if status == "compatible_and_validated"
                    else "Experimental"
                )
                text = f"{labels[workflow_id]} [{badge}]"
            else:
                text = ordinary_labels[workflow_id]
            self.workflow.setItemText(index, text)

    def image_amplification_validation_status(self) -> str | None:
        if not self._image_amplification_mode:
            return None
        return self._image_amplification_status_for_workflow(self.workflow_id())

    def _set_row_visible(self, widget: QWidget, visible: bool) -> None:
        label = self._form.labelForField(widget)
        if label is not None:
            label.setVisible(visible)
        widget.setVisible(visible)

    def _refresh_workflow_controls(self, *_args) -> None:
        workflow_id = self.workflow_id()
        self._set_row_visible(self.evolution, not self._image_amplification_mode)
        self._set_row_visible(
            self.transport_model, not self._image_amplification_mode
        )
        self._set_row_visible(self.workflow, self._image_amplification_mode)
        is_time_dependent = workflow_id in (
            PR_TIMEDEPENDENT_WORKFLOW,
            PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
        )
        is_transverse_static = (
            workflow_id == PR_TRANSVERSE_STATIC_WORKFLOW
        )
        is_transverse_timedependent = (
            workflow_id == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
        )
        self.material_response.setEnabled(True)
        self._set_row_visible(self.material_response, True)
        is_linearized = (
            self.material_response.currentData()
            == PR_MATERIAL_RESPONSE_LINEARIZED
        )
        self._refresh_workflow_labels()
        self._set_row_visible(self.reference_intensity, is_linearized)
        self._set_row_visible(
            self.transverse_applied_field,
            (is_transverse_static or is_transverse_timedependent)
            and is_linearized,
        )
        self._refresh_integrator_choices()
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
        supports_scattering = workflow_id != PR_STATIC_WORKFLOW
        scattering_details = (
            self.scattering_epsilon,
            self.scattering_correlation_um,
            self.scattering_seed,
            self.scattering_canonical_dz_um,
            self.scattering_algorithm,
        )
        self._set_row_visible(self.scattering_enabled, supports_scattering)
        for widget in scattering_details:
            self._set_row_visible(
                widget,
                supports_scattering and self.scattering_enabled.isChecked(),
            )
        self._set_row_visible(self.algorithm_status, True)
        if self._image_amplification_mode:
            status = self.image_amplification_validation_status()
            if status == "compatible_and_validated":
                text = "Validated for the Image Amplification experiment."
            else:
                text = (
                    "Experimental: compatible architecture; specialized "
                    "Image Amplification validation is pending."
                )
            self.algorithm_status.setText(text)
        else:
            self.algorithm_status.setText(
                "Production model selection; commissioning status is tracked "
                "separately."
            )
        self.execution_guidance.setText(
            "Slurm recommended except for small smoke cases."
            if (
                self.transport_model.currentData() == "full_transverse"
                and not is_linearized
            )
            else (
                "Local-friendly at moderate grids."
                if self.transport_model.currentData() == "full_transverse"
                else (
                    "Local-friendly."
                    if is_linearized
                    else "Local-friendly to moderate."
                )
            )
        )

    def _refresh_integrator_choices(self) -> None:
        workflow_id = self.workflow_id()
        linearized = (
            self.material_response.currentData()
            == PR_MATERIAL_RESPONSE_LINEARIZED
        )
        if workflow_id == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW and linearized:
            # The production workflow performs the exact modal material update;
            # the stored IMEX token is retained for request-schema compatibility.
            choices = (("Exact modal evolution", PR_TRANSVERSE_IMEX_EULER),)
        elif workflow_id == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW:
            choices = (
                ("Spectral IMEX Euler", PR_TRANSVERSE_IMEX_EULER),
                (
                    "Explicit Euler (reference)",
                    PR_TRANSVERSE_EXPLICIT_EULER_REFERENCE,
                ),
            )
        elif workflow_id == PR_TIMEDEPENDENT_WORKFLOW and linearized:
            choices = (("Exact modal evolution", PR_EXACT_MODAL_INTEGRATOR),)
        else:
            choices = (
                ("Semi-implicit trapezoidal", PR_SEMI_IMPLICIT_INTEGRATOR),
                ("Explicit Euler (reference)", PR_EULER_INTEGRATOR),
            )
        current = self.integrator.currentData()
        expected = tuple(value for _label, value in choices)
        current_items = tuple(
            self.integrator.itemData(i) for i in range(self.integrator.count())
        )
        if current_items == expected:
            return
        self.integrator.clear()
        for label, value in choices:
            self.integrator.addItem(label, value)
        index = self.integrator.findData(current)
        self.integrator.setCurrentIndex(index if index >= 0 else 0)

    def solver(self) -> PRSolverOptions:
        return PRSolverOptions(
            Nt=self.Nt.value(),
            dt_normalized=self.dt_normalized.value(),
            optical_substeps=self.optical_substeps.value(),
            integrator=self.integrator.currentData(),
        )

    def transverse_solver(self) -> PRTransverseSolverOptions:
        """Return the canonical full-transverse material-time policy."""

        return PRTransverseSolverOptions(
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

    def transverse_material_response(self) -> PRTransverseMaterialResponseSpec:
        model = str(self.material_response.currentData())
        return PRTransverseMaterialResponseSpec(
            model=model,
            reference_intensity=(
                self.reference_intensity.value()
                if model == PR_MATERIAL_RESPONSE_LINEARIZED
                else None
            ),
        )

    def scattering_spec(self) -> PRCanonicalScatteringSpec | None:
        if (
            not self.scattering_enabled.isChecked()
            or self.workflow_id() == PR_STATIC_WORKFLOW
        ):
            return None
        spec = PRCanonicalScatteringSpec(
            epsilon=self.scattering_epsilon.value(),
            transverse_correlation_um=self.scattering_correlation_um.value(),
            realization_seed=int(self.scattering_seed.value()),
            canonical_dz_um=self.scattering_canonical_dz_um.value(),
            algorithm_version=str(self.scattering_algorithm.currentData()),
        )
        spec.validate()
        return spec

    def set_scattering_spec(
        self, scattering: PRCanonicalScatteringSpec | None
    ) -> None:
        if scattering is None:
            self.scattering_enabled.setChecked(False)
            return
        scattering.validate()
        self.scattering_epsilon.setValue(scattering.epsilon)
        self.scattering_correlation_um.setValue(
            scattering.transverse_correlation_um
        )
        self.scattering_seed.setValue(scattering.realization_seed)
        self.scattering_canonical_dz_um.setValue(scattering.canonical_dz_um)
        index = self.scattering_algorithm.findData(scattering.algorithm_version)
        if index < 0:  # pragma: no cover - spec validation owns this case
            raise ValueError("unsupported PR scattering algorithm")
        self.scattering_algorithm.setCurrentIndex(index)
        self.scattering_enabled.setChecked(True)

    def set_transverse_material_response(
        self,
        response: PRTransverseMaterialResponseSpec,
        *,
        applied_field_x: float,
    ) -> None:
        response.validate()
        index = self.material_response.findData(response.model)
        if index < 0:
            raise ValueError("unsupported transverse material response")
        self.material_response.setCurrentIndex(index)
        if response.reference_intensity is not None:
            self.reference_intensity.setValue(response.reference_intensity)
        self.transverse_applied_field.setValue(applied_field_x)

    def backend_spec(self) -> BackendSpec:
        return BackendSpec(
            backend=self.backend.currentText(),
            precision=self.precision.currentText(),
            verbose=False,
        )

    def _backend_selected(self, *_args) -> None:
        self._backend_origin = "explicit_user_or_loaded"

    @property
    def backend_explicitly_selected(self) -> bool:
        return self._backend_origin == "explicit_user_or_loaded"

    @property
    def backend_origin(self) -> str:
        """Describe whether the backend reflects intent or a reversible default."""

        return self._backend_origin

    def apply_execution_backend_context(
        self,
        *,
        target: str,
        gpu_capable: bool = False,
    ) -> None:
        """Apply reversible target defaults without overwriting user intent."""

        if self.backend_explicitly_selected:
            return
        if target not in ("local", "slurm"):
            raise ValueError(f"unsupported execution target: {target!r}")
        backend = (
            "cupy"
            if target == "slurm" and gpu_capable
            else self._implicit_local_backend
        )
        self.backend.blockSignals(True)
        try:
            self.backend.setCurrentText(backend)
        finally:
            self.backend.blockSignals(False)
        self._backend_origin = (
            "implicit_local_default"
            if target == "local"
            else "automatic_target_default"
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

    def set_transverse_solver(self, solver: PRTransverseSolverOptions) -> None:
        solver.validate()
        self.Nt.setValue(solver.Nt)
        self.dt_normalized.setValue(solver.dt_normalized)
        self.optical_substeps.setValue(solver.optical_substeps)
        integrator_index = self.integrator.findData(solver.integrator)
        if (
            integrator_index < 0
            and self.material_response.currentData()
            == PR_MATERIAL_RESPONSE_LINEARIZED
        ):
            # Preserve an old request's compatibility token while presenting
            # the exact modal evolution that the linearized workflow executes.
            self.integrator.clear()
            self.integrator.addItem("Exact modal evolution", solver.integrator)
            integrator_index = 0
        if integrator_index < 0:
            raise ValueError("unsupported transverse PR GUI integrator")
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
        self.backend.blockSignals(True)
        try:
            self.backend.setCurrentIndex(backend_index)
        finally:
            self.backend.blockSignals(False)
        self._backend_origin = "explicit_user_or_loaded"
        self.precision.setCurrentIndex(precision_index)


__all__ = ["PREvolutionPanel"]
