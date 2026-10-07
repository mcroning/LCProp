from __future__ import annotations

from lcprop.pr.unified.workflow import WORKFLOW_ID as UNIFIED_WORKFLOW
from .unified_controls import UnifiedClosurePanel

from lcprop.pr.published_static import PR_PUBLISHED_STATIC_WORKFLOW

from lcprop.gui.numeric_widgets import CompactDoubleSpinBox

from lcprop.gui.layout import readable_form

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
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
    PR_MATERIAL_RESPONSE_FIELD_LINEAR,
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
        form = readable_form(QFormLayout())

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
        self.dt_normalized = CompactDoubleSpinBox()
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
        self.Nt.setToolTip("Material time steps for Run; additional steps for Continue.")
        form.addRow("Material time steps this run", self.Nt)
        form.addRow("Normalized timestep", self.dt_normalized)
        form.addRow("Material integrator", self.integrator)
        from lcprop.pr.static import PRStaticSolverOptions
        material_defaults = PRStaticSolverOptions()
        self.material_iterations = spin_box(0, 100000, material_defaults.max_iterations)
        self.material_backtracks = spin_box(0, 100000, material_defaults.max_backtracks)
        self.material_rms = CompactDoubleSpinBox()
        self.material_max = CompactDoubleSpinBox()
        for widget, value in ((self.material_rms, material_defaults.residual_rms_tolerance),
                              (self.material_max, material_defaults.residual_max_tolerance)):
            widget.setDecimals(14)
            widget.setRange(1e-14, 1e6)
            widget.setValue(value)
        form.addRow("Material Newton iterations", self.material_iterations)
        form.addRow("Material Newton backtracks", self.material_backtracks)
        form.addRow("Material residual RMS tolerance", self.material_rms)
        form.addRow("Material residual maximum tolerance", self.material_max)
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
        self._material_default_precision = self.precision.currentText()
        self.precision.currentTextChanged.connect(self._update_material_precision_defaults)
        self._form = form
        self._image_amplification_mode = False
        self.legacy_static = False
        self.unified_closure = UnifiedClosurePanel()
        form.addRow(self.unified_closure)
        self.unified_solver = QComboBox()
        self.unified_solver.addItem('Scalable iterative — full x-y', 'pr_unified_connected_scalable_v1')
        self.unified_solver.addItem('Reference/direct — bounded validation', 'pr_unified_connected_direct_v1')
        self.unified_solver.setToolTip(
            'Reference/direct is a numerical solver for bounded validation.\n'
            'It solves the same physical model as the scalable solver.')
        self.unified_solver.currentIndexChanged.connect(self._refresh_workflow_controls)
        form.addRow('Unified material solver', self.unified_solver)
        self.fresh_unified_button=QPushButton('Create fresh unified nonlinear Static request')
        self.fresh_unified_button.clicked.connect(self._fresh_unified)
        form.addRow(self.fresh_unified_button)
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

    def _fresh_unified(self):
        self._loaded_unified_request = None
        self.unified_solver.setCurrentIndex(0)
        self.legacy_static=False
        self.unified_closure.condition.setCurrentIndex(0)
        self._refresh_workflow_controls()

    def workflow_id(self) -> str:
        value = str(self.workflow.currentData())
        if (value in (PR_STATIC_WORKFLOW, PR_TRANSVERSE_STATIC_WORKFLOW)
                and self.material_response.currentData() == PR_MATERIAL_RESPONSE_NONLINEAR
                and not self.legacy_static and not self._image_amplification_mode):
            return UNIFIED_WORKFLOW
        if value == PR_STATIC_WORKFLOW:
            return PR_PUBLISHED_STATIC_WORKFLOW
        return value

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
            PR_PUBLISHED_STATIC_WORKFLOW: ("static", "reduced_x"),
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
        # An explicit dimensionality/evolution change creates a fresh request.
        self.legacy_static = False
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
            evolution, transport = self._axes_for_workflow(str(self.workflow.currentData()))
            self.evolution.setCurrentIndex(self.evolution.findData(evolution))
            self.transport_model.setCurrentIndex(
                self.transport_model.findData(transport)
            )
        finally:
            self._syncing_model_controls = False
        self._refresh_workflow_controls()

    def set_workflow_id(self, workflow_id: str) -> None:
        if workflow_id == UNIFIED_WORKFLOW:
            self.legacy_static = False
            self.set_workflow_id(self._workflow_for_axes('static',str(self.transport_model.currentData())))
            self.material_response.setCurrentIndex(self.material_response.findData(PR_MATERIAL_RESPONSE_NONLINEAR))
            self._refresh_workflow_controls()
            return
        if workflow_id == PR_PUBLISHED_STATIC_WORKFLOW:
            self.set_workflow_id(PR_STATIC_WORKFLOW)
            return
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
        if workflow_id == PR_PUBLISHED_STATIC_WORKFLOW:
            return "unavailable"
        status = next(
            capability.validation_status
            for capability in image_amplification_base_capabilities()
            if capability.workflow_id == workflow_id
        )
        if (
            self.material_response.currentData()
            in (PR_MATERIAL_RESPONSE_LINEARIZED, PR_MATERIAL_RESPONSE_FIELD_LINEAR)
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
            str(self.workflow.currentData()) == PR_TRANSVERSE_STATIC_WORKFLOW
        )
        is_transverse_timedependent = (
            workflow_id == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
        )
        # Distinct physics, not aliases: reduced Eq. (5) versus transverse tangent.
        if is_transverse_static or is_transverse_timedependent:
            choices = (("Nonlinear transverse hopping", PR_MATERIAL_RESPONSE_NONLINEAR),
                       ("Uniform-reference tangent", PR_MATERIAL_RESPONSE_LINEARIZED))
        elif is_time_dependent:
            choices = (("Nonlinear reduced hopping", PR_MATERIAL_RESPONSE_NONLINEAR),)
        else:
            choices = (("Nonlinear reduced hopping", PR_MATERIAL_RESPONSE_NONLINEAR),
                       ("Field-linear (local intensity)", PR_MATERIAL_RESPONSE_FIELD_LINEAR))
        current = self.material_response.currentData()
        if tuple(self.material_response.itemData(i) for i in range(self.material_response.count())) != tuple(v for _, v in choices):
            self.material_response.blockSignals(True)
            self.material_response.clear()
            for label, value in choices:
                self.material_response.addItem(label, value)
            index = self.material_response.findData(current)
            self.material_response.setCurrentIndex(max(index, 0))
            self.material_response.blockSignals(False)
        else:
            for i, (label, _) in enumerate(choices):
                self.material_response.setItemText(i, label)
        workflow_id = self.workflow_id()  # Response choices may have changed with the axes.
        self.material_response.setEnabled(True)
        self._set_row_visible(self.material_response, True)
        is_linearized = (
            self.material_response.currentData()
            == PR_MATERIAL_RESPONSE_LINEARIZED
        )
        self._refresh_workflow_labels()
        self.material_response.setToolTip(
            'Uniform-reference tangent: linearized material model about the specified\n'
            'uniform total reference intensity I0 and mean internal field.\n'
            'Distinct from nonlinear hopping and the reduced field-linear model.'
            if is_linearized else 'Selects the material model, not the numerical solver.')
        self.reference_intensity.setToolTip(
            'Uniform total normalized intensity I0 about which the transverse material model is linearized.')
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
            not is_time_dependent and workflow_id not in (PR_PUBLISHED_STATIC_WORKFLOW, UNIFIED_WORKFLOW),
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
        self._set_row_visible(self.optical_substeps, workflow_id not in (PR_PUBLISHED_STATIC_WORKFLOW, UNIFIED_WORKFLOW))
        local_plane = workflow_id in (PR_PUBLISHED_STATIC_WORKFLOW, UNIFIED_WORKFLOW)
        for widget in (self.material_iterations, self.material_backtracks, self.material_rms, self.material_max):
            self._set_row_visible(widget, workflow_id == PR_PUBLISHED_STATIC_WORKFLOW and self.material_response.currentData() == PR_MATERIAL_RESPONSE_NONLINEAR)
        self.fresh_unified_button.setVisible(self.legacy_static and not is_time_dependent
            and self.material_response.currentData() == PR_MATERIAL_RESPONSE_NONLINEAR
            and not self._image_amplification_mode)
        if workflow_id == UNIFIED_WORKFLOW:
            index=self.material_response.findData(PR_MATERIAL_RESPONSE_NONLINEAR)
            self.material_response.setItemText(index,'Fully nonlinear (unified transport)')
        self.unified_closure.setVisible(workflow_id == UNIFIED_WORKFLOW)
        dimension = 2 if self.transport_model.currentData() == 'full_transverse' else 1
        self._set_row_visible(self.unified_solver, workflow_id == UNIFIED_WORKFLOW and dimension == 2)
        if self.unified_closure.dimension != dimension:
            self.unified_closure.set_dimension(dimension)
        v1 = self.scattering_algorithm.findData(PR_CANONICAL_SCATTERING_V1)
        self.scattering_algorithm.model().item(v1).setEnabled(not local_plane)
        if local_plane and self.scattering_algorithm.currentData() != PR_CANONICAL_SCATTERING_V2:
            self.scattering_algorithm.setCurrentIndex(self.scattering_algorithm.findData(PR_CANONICAL_SCATTERING_V2))
        supports_scattering = True
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
            if status == "unavailable":
                text = "Published optical-first Static currently supports ordinary fresh calculations only."
            elif status == "compatible_and_validated":
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

        if workflow_id == UNIFIED_WORKFLOW:
            self.algorithm_status.setText('Unified nonlinear Static: published optical-first; certified mixed precision. No coupled passes or replay.')
            from .presentation_guidance import unified_support_guidance
            self.execution_guidance.setText(unified_support_guidance())
        elif workflow_id == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW and not is_linearized:
            from .presentation_guidance import TD_ELECTRICAL_GUIDANCE
            self.execution_guidance.setText(self.execution_guidance.text() + '\n' + TD_ELECTRICAL_GUIDANCE)

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

    def _update_material_precision_defaults(self, precision):
        # Same established material-only tolerances as reduced Static. Explicit
        # user values are preserved; coupled optical tolerances do not exist here.
        defaults = {"float64": (1e-10, 1e-9), "float32": (2e-6, 1e-5)}
        previous = defaults[self._material_default_precision]
        for widget, old, new in zip((self.material_rms, self.material_max), previous, defaults[precision]):
            if widget.value() == old:
                widget.setValue(new)
        self._material_default_precision = precision

    def published_material_solver(self):
        from lcprop.pr.static import PRStaticSolverOptions
        return PRStaticSolverOptions(max_iterations=self.material_iterations.value(),
            max_backtracks=self.material_backtracks.value(),
            residual_rms_tolerance=self.material_rms.value(),
            residual_max_tolerance=self.material_max.value())

    def set_published_material_solver(self, solver):
        self.material_iterations.setValue(solver.max_iterations)
        self.material_backtracks.setValue(solver.max_backtracks)
        self.material_rms.setValue(solver.residual_rms_tolerance)
        self.material_max.setValue(solver.residual_max_tolerance)

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
        if not self.scattering_enabled.isChecked():
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
        if self.workflow_id() in (PR_STATIC_WORKFLOW, PR_PUBLISHED_STATIC_WORKFLOW):
            response.validate_reduced_static()
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
