"""Standalone Qt window for the photorefractive workflow."""

from __future__ import annotations

from functools import partial
from time import monotonic
import traceback

from PySide6.QtCore import QCoreApplication, QThread, QTimer, Qt, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QFileDialog,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.gui.panels.results_panel import ResultsPanel
from lcprop.gui.experiment_files import (
    ExperimentFileButtons,
    choose_experiment_open_path,
    choose_experiment_save_path,
    launchplane_presentation_payload,
    launchplane_stack_from_presentation,
    show_experiment_open_warning,
)
from lcprop.gui.workers import WorkflowWorker
from lcprop.gui.remote_execution import (
    RemoteExecutionControls,
    execution_target_selector,
    remote_status_text,
)
from lcprop.transport.status import RemoteRunState, RemoteRunStatus
from lcprop.transport.result_policy import FAST_RESULT_POLICY, FULL_RESULT_POLICY
from lcprop.persistence import (
    load_experiment,
    load_run_checkpoint,
    save_experiment,
    save_run_checkpoint,
)
from lcprop.pr.checkpoint import (
    PRTimeDependentCheckpoint,
    validate_pr_continuation,
)
from lcprop.pr.gui.beam_panel import make_pr_beam_panel
from lcprop.pr.gui.evolution_panel import PREvolutionPanel
from lcprop.pr.gui.grid_panel import PRGridPanel
from lcprop.pr.gui.image_input_panel import (
    PR_GAUSSIAN_INPUT_MODE,
    PR_IMAGE_AMPLIFICATION_INPUT_MODE,
    PRImageInputPanel,
)
from lcprop.pr.gui.material_panel import PRMaterialPanel
from lcprop.pr.gui.run_cost import (
    LocalRunCostAssessment,
    LocalRunCostClass,
    classify_pr_run_cost,
)
from lcprop.pr.gui.request_adapter import (
    apply_pr_request,
    build_pr_request,
    validate_pr_gui_workflow_request,
    validate_pr_gui_request_representable,
)
from lcprop.pr.operations import (
    PR_IMAGE_AMPLIFICATION_OPERATION,
    PR_STATIC_OPERATION,
    PR_TIMEDEPENDENT_OPERATION,
)
from lcprop.pr.specs import (
    PRRunRequest,
    PRRunResult,
    PR_MATERIAL_ID,
    PR_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.image_amplification import (
    PR_IMAGE_AMPLIFICATION_WORKFLOW,
    PRBeamPanelImageAmplificationRunRequest,
    PRImageAmplificationCompositeResult,
    PRImageAmplificationExperimentRequest,
    PRImageAmplificationRunRequest,
    image_amplification_base_capabilities,
    image_amplification_experiment_request,
    prepare_image_amplification_base_request,
    prepare_image_amplification_workflow_request,
    run_image_amplification_experiment,
)
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PRStaticRunResult,
    PR_STATIC_WORKFLOW,
)
from lcprop.pr.transverse.operations import (
    PR_TRANSVERSE_STATIC_OPERATION,
    PR_TRANSVERSE_TIMEDEPENDENT_OPERATION,
)
from lcprop.pr.transverse.specs import (
    PR_MATERIAL_RESPONSE_LINEARIZED,
    PRTransverseRunRequest,
    PRTransverseRunResult,
    PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticRunResult,
    PR_TRANSVERSE_STATIC_WORKFLOW,
)
from lcprop.pr.workflow import continue_pr_timedependent
from lcprop.runners.base import RunnerResult
from lcprop.runners.local import LocalRunner


def _image_amplification_validation_status(capability, base_request) -> str:
    """Resolve IA validation without changing workflow-level capability data."""

    if (
        isinstance(
            base_request,
            (PRTransverseRunRequest, PRTransverseStaticRunRequest),
        )
        and base_request.material_response.model
        == PR_MATERIAL_RESPONSE_LINEARIZED
    ):
        return "compatible_validation_pending"
    return capability.validation_status


def _continue_pr_operation(
    request,
    *,
    checkpoint,
    additional_steps,
    cancellation_token=None,
    progress_callback=None,
) -> RunnerResult:
    """Adapt PR continuation to the workflow-neutral worker result."""

    result = continue_pr_timedependent(
        request,
        checkpoint,
        additional_steps,
        cancellation_token=cancellation_token,
        progress_callback=progress_callback,
    )
    return RunnerResult(
        kind=PR_TIMEDEPENDENT_WORKFLOW,
        result=result,
        message=(
            "Cancelled locally"
            if result.status == "cancelled"
            else "Completed locally"
        ),
        run_data=PR_TIMEDEPENDENT_OPERATION.to_run_data(result),
        material_id=PR_MATERIAL_ID,
    )


class PRMainWindow(QWidget):
    """Focused GUI for the registered PR workflow operations."""

    def __init__(self, *, slurm_runner=None) -> None:
        super().__init__()
        self.local_runner = LocalRunner(
            operations=(
                PR_TIMEDEPENDENT_OPERATION,
                PR_IMAGE_AMPLIFICATION_OPERATION,
                PR_TRANSVERSE_TIMEDEPENDENT_OPERATION,
                PR_TRANSVERSE_STATIC_OPERATION,
                PR_STATIC_OPERATION,
            )
        )
        self._explicit_slurm_runner = slurm_runner
        self.remote_execution_controls = RemoteExecutionControls()
        self.slurm_runner = (
            slurm_runner
            if slurm_runner is not None
            else self.remote_execution_controls.create_runner()
        )
        self.runner = self.local_runner
        self.last_result = None
        self.last_runner_result = None
        self.last_progress: RunProgress | None = None
        self.last_remote_status = None
        self.remote_status_history = []
        self.last_progress_thread = None
        self.last_checkpoint = None
        self.run_status = "idle"
        self._background_running = False
        self._active_request = None
        self._thread: QThread | None = None
        self._worker: WorkflowWorker | None = None
        self._cancellation_token: CancellationToken | None = None
        self._outcome_received = False
        self._thread_done = False
        self._close_requested = False
        self._hydrating_checkpoint = False
        self._hydrating_experiment = False
        self.checkpoint_compatibility_reason: str | None = None

        self.setWindowTitle("LCProp PR")
        self.setMinimumSize(1200, 760)
        self.resize(1450, 900)

        root = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel("LCProp PR"))
        header.addStretch(1)
        header.addWidget(QLabel("Execution:"))
        self.execution_target_selector = execution_target_selector(
            slurm_available=self.slurm_runner is not None
        )
        self.execution_target_selector.currentIndexChanged.connect(
            self._execution_target_changed
        )
        header.addWidget(self.execution_target_selector)
        header.addWidget(QLabel("Result retrieval:"))
        self.result_policy_selector = QComboBox()
        self.result_policy_selector.addItem("Fast / Exploratory", FAST_RESULT_POLICY)
        self.result_policy_selector.addItem("Full", FULL_RESULT_POLICY)
        self.result_policy_selector.setEnabled(False)
        self.result_policy_selector.setToolTip(
            "Fast retrieves optical endpoints and compact diagnostics; Full also "
            "retrieves longitudinal material volumes."
        )
        header.addWidget(self.result_policy_selector)
        self.runner_label = QLabel(f"Runner: {self.runner.name}")
        header.addWidget(self.runner_label)
        self.status_label = QLabel("Idle")
        header.addWidget(self.status_label)
        self.run_button = QPushButton("Run PR")
        self.run_button.clicked.connect(self.run_clicked)
        header.addWidget(self.run_button)
        self.continue_button = QPushButton("Continue")
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self.continue_clicked)
        header.addWidget(self.continue_button)
        self.save_checkpoint_button = QPushButton("Save Checkpoint")
        self.save_checkpoint_button.setEnabled(False)
        self.save_checkpoint_button.clicked.connect(
            self.save_checkpoint_clicked
        )
        header.addWidget(self.save_checkpoint_button)
        self.load_checkpoint_button = QPushButton("Load Checkpoint")
        self.load_checkpoint_button.clicked.connect(
            self.load_checkpoint_clicked
        )
        header.addWidget(self.load_checkpoint_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop_clicked)
        header.addWidget(self.stop_button)
        self.experiment_file_buttons = ExperimentFileButtons(self)
        self.experiment_file_buttons.saveRequested.connect(
            self.save_experiment_clicked
        )
        self.experiment_file_buttons.openRequested.connect(
            self.open_experiment_clicked
        )
        header.addWidget(self.experiment_file_buttons)
        root.addLayout(header)
        root.addWidget(self.remote_execution_controls)
        self.remote_execution_controls.selectionChanged.connect(
            self._remote_profile_changed
        )

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)
        self.material_panel = PRMaterialPanel()
        self.grid_panel = PRGridPanel()
        self.input_panel = PRImageInputPanel()
        self.beam_panel = make_pr_beam_panel(
            x_aperture_um=self.grid_panel.x_aperture_um.value(),
            y_aperture_um=self.grid_panel.y_aperture_um.value(),
        )
        self.input_panel.set_beam_panel(self.beam_panel)
        self.evolution_panel = PREvolutionPanel()
        self.results_panel = ResultsPanel()
        self.tabs.addTab(self.material_panel, "PR Material")
        self.tabs.addTab(self.input_panel, "Input")
        self.tabs.addTab(self.beam_panel, "Beam")
        self.tabs.addTab(self.grid_panel, "Grid")
        self.tabs.addTab(self.evolution_panel, "Evolution")
        self.tabs.addTab(self.results_panel, "Results")

        self.grid_panel.x_aperture_um.valueChanged.connect(
            self._sync_beam_aperture
        )
        self.grid_panel.y_aperture_um.valueChanged.connect(
            self._sync_beam_aperture
        )
        self.input_panel.modeChanged.connect(self._input_mode_changed)
        self.input_panel.configurationChanged.connect(
            self._configuration_changed
        )
        self._input_mode_changed(self.input_panel.mode_id())
        self._connect_checkpoint_compatibility_signals()

    @Slot(str)
    def _input_mode_changed(self, mode_id: str) -> None:
        image_mode = mode_id == PR_IMAGE_AMPLIFICATION_INPUT_MODE
        beam_index = self.tabs.indexOf(self.beam_panel)
        self.tabs.setTabEnabled(beam_index, True)
        self.evolution_panel.set_image_amplification_mode(image_mode)
        if hasattr(self, "checkpoint_compatibility_reason"):
            self._refresh_checkpoint_controls()

    def _sync_beam_aperture(self) -> None:
        self.beam_panel.set_aperture(
            self.grid_panel.x_aperture_um.value(),
            self.grid_panel.y_aperture_um.value(),
        )

    def _connect_checkpoint_compatibility_signals(self) -> None:
        for panel in (
            self.material_panel,
            self.input_panel,
            self.beam_panel,
            self.grid_panel,
            self.evolution_panel,
        ):
            for widget_type in (QSpinBox, QDoubleSpinBox):
                for widget in panel.findChildren(widget_type):
                    widget.valueChanged.connect(self._configuration_changed)
            for widget in panel.findChildren(QComboBox):
                widget.currentIndexChanged.connect(
                    self._configuration_changed
                )
            for widget in panel.findChildren(QCheckBox):
                widget.toggled.connect(self._configuration_changed)
            for widget in panel.findChildren(QLineEdit):
                widget.editingFinished.connect(self._configuration_changed)
        self.beam_panel.launch_plane_widget.beamStackChanged.connect(
            self._configuration_changed
        )
        self._refresh_checkpoint_controls()

    @Slot()
    def _configuration_changed(self, *_args) -> None:
        if (
            self._hydrating_checkpoint
            or self._hydrating_experiment
            or self._background_running
        ):
            return
        self._refresh_checkpoint_controls()

    def _refresh_checkpoint_controls(self) -> None:
        checkpoint = self.last_checkpoint
        workflow_id = self.evolution_panel.workflow_id()
        reason = None
        if self.input_panel.is_image_amplification():
            reason = "Continuation is not available for image amplification."
        elif workflow_id != PR_TIMEDEPENDENT_WORKFLOW:
            reason = "Continuation is available only for time-dependent runs."
        elif checkpoint is None:
            reason = "No PR checkpoint is loaded or retained."
        else:
            try:
                validate_pr_continuation(self.build_request(), checkpoint)
            except Exception as exc:
                reason = str(exc)
        self.checkpoint_compatibility_reason = reason
        compatible = checkpoint is not None and reason is None
        actions_enabled = not self._background_running
        self.continue_button.setEnabled(actions_enabled and compatible)
        self.continue_button.setToolTip("" if compatible else reason or "")
        self.save_checkpoint_button.setEnabled(
            actions_enabled
            and workflow_id == PR_TIMEDEPENDENT_WORKFLOW
            and checkpoint is not None
        )
        if checkpoint is None or workflow_id != PR_TIMEDEPENDENT_WORKFLOW:
            return
        if compatible:
            self.status_label.setText(
                "Checkpoint ready: "
                f"step {checkpoint.completed_steps}, "
                f"t={checkpoint.time_normalized:.6g}"
            )
        else:
            self.status_label.setText(f"Checkpoint incompatible: {reason}")

    def build_request(self):
        """Construct the immutable request represented by the controls."""

        if self.input_panel.is_image_amplification():
            base_request = build_pr_request(
                material_panel=self.material_panel,
                beam_panel=self.beam_panel,
                grid_panel=self.grid_panel,
                evolution_panel=self.evolution_panel,
            )
            return self.input_panel.build_experiment_request(
                base_workflow_id=self.evolution_panel.workflow_id(),
                base_request=base_request,
                launch_configuration=self.beam_panel.launch_configuration(),
            )
        return build_pr_request(
            material_panel=self.material_panel,
            beam_panel=self.beam_panel,
            grid_panel=self.grid_panel,
            evolution_panel=self.evolution_panel,
        )

    def _capture_experiment_gui_state(self):
        workflow_id = self.evolution_panel.workflow_id()
        if workflow_id == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW:
            solver = self.evolution_panel.transverse_solver()
        elif workflow_id == PR_TRANSVERSE_STATIC_WORKFLOW:
            solver = self.evolution_panel.transverse_static_solver()
        elif workflow_id == PR_STATIC_WORKFLOW:
            solver = self.evolution_panel.static_solver()
        else:
            solver = self.evolution_panel.solver()
        return {
            "input_mode": self.input_panel.mode_id(),
            "workflow_id": workflow_id,
            "grid": self.grid_panel.grid(),
            "material": self.material_panel.material(),
            "solver": solver,
            "material_response": (
                self.evolution_panel.transverse_material_response()
            ),
            "scattering": self.evolution_panel.scattering_spec(),
            "transverse_applied_field": (
                self.evolution_panel.transverse_applied_field.value()
                if workflow_id in (
                    PR_TRANSVERSE_STATIC_WORKFLOW,
                    PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
                )
                else 0.0
            ),
            "backend": self.evolution_panel.backend_spec(),
            "beam_stack": self.beam_panel.beam_stack_definition,
            "launch_elements": self.beam_panel.launch_elements(),
            "pump_channel_index": self.input_panel.pump_channel.currentData(),
            "signal_channel_index": self.input_panel.signal_channel.currentData(),
        }

    def _restore_experiment_gui_state(self, state) -> None:
        mode_index = self.input_panel.input_mode.findData(state["input_mode"])
        self.input_panel.input_mode.setCurrentIndex(mode_index)
        self.grid_panel.set_grid(state["grid"])
        self.material_panel.set_material(state["material"])
        workflow_id = state["workflow_id"]
        self.evolution_panel.set_workflow_id(workflow_id)
        self.evolution_panel.set_transverse_material_response(
            state["material_response"],
            applied_field_x=state["transverse_applied_field"],
        )
        if workflow_id == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW:
            self.evolution_panel.set_transverse_solver(state["solver"])
        elif workflow_id == PR_TRANSVERSE_STATIC_WORKFLOW:
            self.evolution_panel.set_transverse_static_solver(state["solver"])
        elif workflow_id == PR_STATIC_WORKFLOW:
            self.evolution_panel.set_static_solver(state["solver"])
        else:
            self.evolution_panel.set_solver(state["solver"])
        self.evolution_panel.set_scattering_spec(state["scattering"])
        self.evolution_panel.set_backend_spec(state["backend"])
        self.beam_panel.set_aperture(
            state["grid"].x_aperture_um,
            state["grid"].y_aperture_um,
        )
        self.beam_panel.set_beam_stack_definition(state["beam_stack"])
        self.beam_panel.input_screen_editor.set_launch_elements(
            state["launch_elements"]
        )
        if (
            state["pump_channel_index"] is not None
            and state["signal_channel_index"] is not None
        ):
            self.input_panel.set_role_indices(
                state["pump_channel_index"],
                state["signal_channel_index"],
            )

    @staticmethod
    def _validate_experiment_request_representable(request) -> None:
        if isinstance(request, PRImageAmplificationExperimentRequest):
            request.validate()
            validate_pr_gui_request_representable(request.base_request)
            if request.base_request.beams != request.launch_configuration.beams:
                raise ValueError(
                    "Image Amplification base and launch beams must agree"
                )
            if (
                request.base_request.launch_elements
                != request.launch_configuration.channel_elements
            ):
                raise ValueError(
                    "Image Amplification base and launch screen plans must agree"
                )
            return
        validate_pr_gui_request_representable(request)

    def save_experiment_to(self, path):
        """Save the active PR experiment without running it."""

        if self._background_running:
            raise RuntimeError("cannot save an experiment while a run is active")
        request = self.build_request()
        self._validate_experiment_request_representable(request)
        return save_experiment(
            request,
            path,
            material_id=PR_MATERIAL_ID,
            workflow_id=self._workflow_id_for_request(request),
            presentation_payload=launchplane_presentation_payload(
                self.beam_panel
            ),
        )

    def load_experiment_from(self, path):
        """Validate and transactionally restore one PR experiment file."""

        if self._background_running:
            raise RuntimeError("cannot open an experiment while a run is active")
        loaded = load_experiment(path, expected_material_id=PR_MATERIAL_ID)
        self._validate_experiment_request_representable(loaded.request)
        prior = self._capture_experiment_gui_state()
        stack = launchplane_stack_from_presentation(loaded)
        self._hydrating_experiment = True
        try:
            image_experiment = isinstance(
                loaded.request,
                PRImageAmplificationExperimentRequest,
            )
            mode = (
                PR_IMAGE_AMPLIFICATION_INPUT_MODE
                if image_experiment
                else PR_GAUSSIAN_INPUT_MODE
            )
            mode_index = self.input_panel.input_mode.findData(mode)
            self.input_panel.input_mode.setCurrentIndex(mode_index)
            ordinary_request = (
                loaded.request.base_request
                if image_experiment
                else loaded.request
            )
            apply_pr_request(
                ordinary_request,
                material_panel=self.material_panel,
                beam_panel=self.beam_panel,
                grid_panel=self.grid_panel,
                evolution_panel=self.evolution_panel,
                beam_stack_definition=stack,
            )
            self.beam_panel.input_screen_editor.set_launch_elements(
                ordinary_request.launch_elements
            )
            if image_experiment:
                self.input_panel.set_role_indices(
                    loaded.request.pump_channel_index,
                    loaded.request.signal_channel_index,
                )
            rebuilt = self.build_request()
            if rebuilt != loaded.request:
                raise ValueError(
                    "loaded PR request is not exactly representable by the GUI"
                )
        except Exception:
            self._restore_experiment_gui_state(prior)
            raise
        finally:
            self._hydrating_experiment = False
        self._refresh_checkpoint_controls()
        return loaded

    @Slot()
    def save_experiment_clicked(self) -> None:
        path = choose_experiment_save_path(self)
        if path is None:
            return
        try:
            saved = self.save_experiment_to(path)
        except Exception:
            self.status_label.setText("Experiment save failed")
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())
            self.tabs.setCurrentWidget(self.results_panel)
            return
        self.status_label.setText("Experiment saved")
        self.results_panel.append_console(f"Saved PR experiment: {saved}")

    @Slot()
    def open_experiment_clicked(self) -> None:
        path = choose_experiment_open_path(self)
        if path is None:
            return
        try:
            loaded = self.load_experiment_from(path)
        except Exception as exc:
            show_experiment_open_warning(self, exc)
            self.status_label.setText("Experiment open failed")
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())
            self.tabs.setCurrentWidget(self.results_panel)
            return
        self.status_label.setText("Experiment opened")
        self.results_panel.append_console(
            f"Opened PR experiment: {loaded.workflow_id}"
        )

    def describe_request(self, request) -> str:
        """Return a durable, unit-explicit summary of one PR request."""

        if isinstance(request, PRImageAmplificationExperimentRequest):
            prepared, _transmission, _grating = (
                prepare_image_amplification_base_request(request)
            )
            capability = next(
                capability
                for capability in image_amplification_base_capabilities()
                if capability.workflow_id == request.base_workflow_id
            )
            assignments = {
                assignment.channel_index: assignment.elements
                for assignment in request.launch_configuration.channel_elements
            }
            screen = assignments[request.signal_channel_index][0]
            channels = request.launch_configuration.beams.channels
            pump = channels[request.pump_channel_index]
            signal = channels[request.signal_channel_index]
            validation_status = _image_amplification_validation_status(
                capability, prepared
            )
            status = (
                "Validated"
                if validation_status == "compatible_and_validated"
                else "Experimental — validation pending"
            )
            lines = [
                "Material: photorefractive",
                f"Workflow: {PR_IMAGE_AMPLIFICATION_WORKFLOW}",
                "Input mode: Image Amplification",
                f"Base PR algorithm: {request.base_workflow_id}",
                f"Algorithm validation: {status}",
                f"Source: {request.source.display_name}",
                (
                    "Image footprint: "
                    f"{screen.placement.width_um:g} × "
                    f"{screen.placement.height_um:g} µm"
                ),
                f"Invert image: {screen.invert}",
                (
                    "Incident channel powers: "
                    f"pump={pump.power_mW:g} mW; "
                    f"signal={signal.power_mW:g} mW"
                ),
                (
                    "Derived incident signal/pump power ratio: "
                    f"{request.incident_signal_to_pump_power_ratio:g}"
                ),
                "Base request:",
            ]
            lines.extend(
                f"  {line}" for line in self.describe_request(prepared).splitlines()
            )
            return "\n".join(lines)

        if isinstance(
            request,
            (
                PRImageAmplificationRunRequest,
                PRBeamPanelImageAmplificationRunRequest,
            ),
        ):
            prepared, _transmission, _grating = (
                prepare_image_amplification_workflow_request(request)
            )
            preflight = validate_pr_gui_workflow_request(prepared)
            source = request.source
            if isinstance(request, PRBeamPanelImageAmplificationRunRequest):
                launch_configuration = request.launch_configuration
                pump = launch_configuration.beams.channels[
                    request.pump_channel_index
                ]
                signal = launch_configuration.beams.channels[
                    request.signal_channel_index
                ]
                signal_elements = next(
                    assignment.elements
                    for assignment in launch_configuration.channel_elements
                    if assignment.channel_index == request.signal_channel_index
                )
                screen = signal_elements[0]
                image_size = (
                    screen.placement.width_um,
                    screen.placement.height_um,
                )
                invert_image = screen.invert
                preprocessing_policy = screen.preprocessing_policy
                incident_powers = (pump.power_mW, signal.power_mW)
                incident_ratio = request.incident_signal_to_pump_power_ratio
            else:
                launch = request.launch
                image_size = (
                    launch.image_physical_size_um,
                    launch.image_physical_size_um,
                )
                invert_image = launch.invert_image
                preprocessing_policy = source.preprocessing_policy
                incident_powers = (
                    launch.pump_incident_power_mW,
                    launch.signal_incident_power_mW,
                )
                incident_ratio = launch.incident_signal_to_pump_power_ratio
            lines = [
                "Material: photorefractive",
                f"Workflow: {PR_IMAGE_AMPLIFICATION_WORKFLOW}",
                "Input mode: Image Amplification",
                f"Runner: {self.runner.name}",
                (
                    f"Grid: {request.grid.Nx} × {request.grid.Ny}, "
                    f"Nz={round(request.grid.z_length_um / request.grid.dz_um)}"
                ),
                (
                    "Periodic aperture: "
                    f"x={request.grid.x_aperture_um:g} µm, "
                    f"y={request.grid.y_aperture_um:g} µm"
                ),
                (
                    f"Source: {source.display_name}; "
                    f"pixels={source.width} × {source.height}; "
                    f"format={source.encoded_format}; SHA-256={source.sha256}"
                ),
                f"Preprocessing: {preprocessing_policy}",
                "Alpha policy: discarded; alpha is not an optical mask",
                (
                    f"Image footprint: {image_size[0]:g} × {image_size[1]:g} µm; "
                    "nearest-neighbor resampling"
                ),
                f"Invert image: {invert_image}",
                (
                    "Incident channel powers: "
                    f"pump={incident_powers[0]:g} mW; "
                    f"signal={incident_powers[1]:g} mW"
                ),
                (
                    "Derived incident signal/pump power ratio: "
                    f"{incident_ratio:g}"
                ),
                (
                    "Prepared incident channel powers: "
                    + ", ".join(
                        f"{channel.power_mW:.8g} mW"
                        for channel in prepared.beams.channels
                    )
                ),
                f"Material steps: {request.solver.Nt}",
                f"Material integrator: {request.solver.integrator}",
                f"Normalized timestep: {request.solver.dt_normalized:g}",
                (
                    "Conservative normalized timestep limit: "
                    f"{preflight.conservative_dt_limit:.8g}"
                ),
                f"Optical substeps per z slice: {request.solver.optical_substeps}",
                (
                    f"Backend: {request.backend.backend}; "
                    f"precision={request.backend.precision}"
                ),
            ]
            if preflight.warnings:
                lines.append("Preflight warnings:")
                lines.extend(f"- {warning}" for warning in preflight.warnings)
            else:
                lines.append("Preflight warnings: none")
            return "\n".join(lines)
        preflight = validate_pr_gui_workflow_request(request)
        workflow_id = self._workflow_id_for_request(request)
        lines = [
            "Material: photorefractive",
            f"Workflow: {workflow_id}",
            f"Runner: {self.runner.name}",
            (
                f"Grid: {request.grid.Nx} × {request.grid.Ny}, "
                f"Nz={round(request.grid.z_length_um / request.grid.dz_um)}"
            ),
            (
                "Periodic aperture: "
                f"x={request.grid.x_aperture_um:g} µm, "
                f"y={request.grid.y_aperture_um:g} µm"
            ),
            (
                f"Interaction length: {request.grid.z_length_um:g} µm; "
                f"optical dz={request.grid.dz_um:g} µm"
            ),
            (
                "Normalized intensities: "
                f"dark={request.material.dark_intensity:g}, "
                "uniform background="
                f"{request.material.uniform_background_intensity:g}"
            ),
            f"Normalized applied field: {request.material.applied_field:g}",
            f"Gain-length product: {request.material.gain_length_product:g}",
            f"Refractive index: {request.material.refractive_index:g}",
            f"Enabled beams: {len(request.beams.channels)}",
        ]
        for index, (channel, group) in enumerate(
            zip(request.beams.channels, request.beams.coherence_groups),
            start=1,
        ):
            lines.append(
                f"Beam {index}: {channel.name}; P={channel.power_mW:g} mW; "
                f"λ={channel.wavelength_um:g} µm; "
                f"waists=({channel.waist_x_um:g}, "
                f"{channel.waist_y_um:g}) µm; "
                f"center=({channel.x0_um:g}, {channel.y0_um:g}) µm; "
                "phase gradients="
                f"({channel.tilt_x_rad_per_um:g}, "
                f"{channel.tilt_y_rad_per_um:g}) rad/µm; "
                f"phase={channel.phase_rad:g} rad; group={group}"
            )
        transverse = isinstance(
            request,
            (PRTransverseRunRequest, PRTransverseStaticRunRequest),
        )
        timedependent = isinstance(
            request,
            (PRRunRequest, PRTransverseRunRequest),
        )
        linearized = (
            request.material_response.model == PR_MATERIAL_RESPONSE_LINEARIZED
        )
        lines.extend([
            f"Evolution: {'Time dependent' if timedependent else 'Static'}",
            (
                "Transport model: Full transverse (x-y drift/diffusion)"
                if transverse
                else "Transport model: Reduced x-only (x drift/diffusion)"
            ),
            (
                "Material response: Linearized"
                if linearized
                else "Material response: Fully nonlinear"
            ),
            "Software status: Production model selection",
        ])
        scattering = getattr(request, "scattering", None)
        if scattering is None:
            lines.append("Canonical volume scattering: disabled")
        else:
            lines.extend([
                "Canonical volume scattering: enabled",
                f"Scattering strength ε: {scattering.epsilon:g}",
                (
                    "Scattering correlation length: "
                    f"{scattering.transverse_correlation_um:g} µm"
                ),
                f"Scattering seed: {scattering.realization_seed}",
                f"Scattering canonical slab Δz: {scattering.canonical_dz_um:g} µm",
                f"Scattering model: {scattering.algorithm_version}",
            ])
        if workflow_id == PR_TIMEDEPENDENT_WORKFLOW:
            lines.extend([
                f"Material steps: {request.solver.Nt}",
                f"Material integrator: {request.solver.integrator}",
                f"Normalized timestep: {request.solver.dt_normalized:g}",
                (
                    "Conservative normalized timestep limit: "
                    f"{preflight.conservative_dt_limit:.8g}"
                ),
            ])
            if linearized:
                lines.append(
                    "Linearization intensity I₀: "
                    f"{request.material_response.reference_intensity:g} "
                    "normalized total transport intensity"
                )
        elif workflow_id == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW:
            lines.extend([
                (
                    "Time-dependent material model: Linearized full transverse"
                    if linearized
                    else "Time-dependent material model: full 2D transverse zero-flux"
                ),
                "Authoritative material state: periodic zero-mean psi",
                "Solved material fields: E_x and E_y",
                "Scalar optical projection: E_active = E_x",
                f"Material steps: {request.solver.Nt}",
                (
                    "Material integrator: exact frozen-source modal update"
                    if linearized
                    else f"Material integrator: {request.solver.integrator}"
                ),
                f"Normalized timestep: {request.solver.dt_normalized:g}",
            ])
            if linearized:
                lines.extend([
                    (
                        "Linearization intensity I₀: "
                        f"{request.material_response.reference_intensity:g} "
                        "normalized total transport intensity"
                    ),
                    (
                        "Transverse applied mean field: "
                        f"{request.boundary.applied_field_x:g} normalized"
                    ),
                    "Electrical ensemble: fixed harmonic mean field",
                ])
        elif workflow_id == PR_TRANSVERSE_STATIC_WORKFLOW:
            lines.extend([
                (
                    "Static material model: Linearized material response"
                    if linearized
                    else "Static material model: Fully nonlinear"
                ),
                "Transport: Full transverse PR transport",
                "Authoritative material state: periodic zero-mean psi",
                "Solved material fields: E_x and E_y",
                "Scalar optical projection: E_active = E_x",
                (
                    "Maximum coupled material/optical iterations: "
                    f"{request.solver.max_coupled_iterations}"
                ),
            ])
            if linearized:
                lines.extend([
                    (
                        "Linearization intensity I₀: "
                        f"{request.material_response.reference_intensity:g} "
                        "normalized total transport intensity"
                    ),
                    (
                        "Transverse applied mean field: "
                        f"{request.boundary.applied_field_x:g} normalized"
                    ),
                    "Electrical ensemble: fixed harmonic mean field",
                ])
        else:
            lines.extend([
                (
                    "Static material model: Linearized material response"
                    if linearized
                    else "Static material model: Fully nonlinear"
                ),
                "Transport: Reduced x-only PR transport",
                (
                    "Maximum coupled passes per z slice: "
                    f"{request.solver.max_coupled_passes}"
                ),
                (
                    "Static material solver: direct centered-difference "
                    "Fourier response"
                    if linearized
                    else (
                        "Static material solver: precision-aware automatic "
                        "defaults"
                    )
                ),
            ])
            if linearized:
                lines.extend([
                    (
                        "Linearization intensity I₀: "
                        f"{request.material_response.reference_intensity:g} "
                        "normalized total transport intensity"
                    ),
                    (
                        "Reduced applied field: "
                        f"{request.material.applied_field:g} normalized"
                    ),
                    "Uniform equilibrium field E₀ = E_app I_b / I₀",
                ])
        lines.extend([
            f"Optical substeps per z slice: {request.solver.optical_substeps}",
            (
                f"Backend: {request.backend.backend}; "
                f"precision={request.backend.precision}"
            ),
        ])
        if preflight.warnings:
            lines.append("Preflight warnings:")
            lines.extend(f"- {warning}" for warning in preflight.warnings)
        else:
            lines.append("Preflight warnings: none")
        return "\n".join(lines)

    @staticmethod
    def _workflow_id_for_request(request) -> str:
        if isinstance(
            request,
            (
                PRImageAmplificationRunRequest,
                PRBeamPanelImageAmplificationRunRequest,
                PRImageAmplificationExperimentRequest,
            ),
        ):
            return PR_IMAGE_AMPLIFICATION_WORKFLOW
        if isinstance(request, PRTransverseStaticRunRequest):
            return PR_TRANSVERSE_STATIC_WORKFLOW
        if isinstance(request, PRTransverseRunRequest):
            return PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW
        if isinstance(request, PRStaticRunRequest):
            return PR_STATIC_WORKFLOW
        if isinstance(request, PRRunRequest):
            return PR_TIMEDEPENDENT_WORKFLOW
        raise TypeError("unsupported PR GUI request type")

    @Slot()
    def _remote_profile_changed(self) -> None:
        if self.remote_execution_controls.slurm_available:
            self.slurm_runner = self.remote_execution_controls.create_runner()
        else:
            self.slurm_runner = self._explicit_slurm_runner
        item = self.execution_target_selector.model().item(1)
        if item is not None:
            item.setEnabled(self.slurm_runner is not None)
            item.setToolTip(
                "" if self.slurm_runner is not None
                else self.remote_execution_controls.unavailable_reason
            )
        if self.execution_target_selector.currentData() == "slurm":
            self._execution_target_changed()

    @Slot()
    def _execution_target_changed(self) -> None:
        target = self.execution_target_selector.currentData()
        self.runner = self.slurm_runner if target == "slurm" else self.local_runner
        if self.runner is None:
            self.runner = self.local_runner
            self.execution_target_selector.setCurrentIndex(0)
        self.runner_label.setText(f"Runner: {self.runner.name}")
        self.result_policy_selector.setEnabled(target == "slurm")
        if target != "slurm":
            self.evolution_panel.apply_execution_backend_context(target="local")
            return
        cluster = self.remote_execution_controls.selected_cluster()
        resource = self.remote_execution_controls.selected_resource_name()
        if cluster is None or resource is None:
            return
        profile = cluster.profile(resource)
        self.evolution_panel.apply_execution_backend_context(
            target="slurm",
            gpu_capable=profile.gpus > 0 or profile.require_cupy,
        )

    def _remote_runner_kwargs(self) -> dict[str, str]:
        values = self.remote_execution_controls.runner_kwargs()
        values["result_policy"] = str(self.result_policy_selector.currentData())
        return values

    def _run_registered(self, request, **kwargs):
        if isinstance(
            request,
            (
                PRBeamPanelImageAmplificationRunRequest,
                PRImageAmplificationExperimentRequest,
            ),
        ):
            composite = (
                request
                if isinstance(request, PRImageAmplificationExperimentRequest)
                else image_amplification_experiment_request(request)
            )
            runner_kwargs = (
                self._remote_runner_kwargs()
                if self.runner is self.slurm_runner
                else None
            )
            return run_image_amplification_experiment(
                self.runner,
                composite,
                cancellation_token=kwargs.get("cancellation_token"),
                progress_callback=kwargs.get("progress_callback"),
                runner_kwargs=runner_kwargs,
            )
        if isinstance(
            request,
            (PRStaticRunRequest, PRTransverseStaticRunRequest),
        ):
            progress_callback = kwargs.get("progress_callback")
            phase_started_at = monotonic()

            def before_product_conversion(_operation, result) -> None:
                if progress_callback is None:
                    return
                transverse = isinstance(result, PRTransverseStaticRunResult)
                completed = int(
                    result.completed_coupled_iterations
                    if transverse
                    else result.completed_slices
                )
                total = int(
                    request.solver.max_coupled_iterations
                    if transverse
                    else result.grid_summary["Nz"]
                )
                progress_callback(
                    RunProgress(
                        workflow=(
                            PR_TRANSVERSE_STATIC_WORKFLOW
                            if transverse
                            else PR_STATIC_WORKFLOW
                        ),
                        status="running",
                        completed_units=completed,
                        total_units=total,
                        current_coordinate=(
                            float(completed)
                            if transverse
                            else float(
                                completed * result.grid_summary["dz_um"]
                            )
                        ),
                        coordinate_name=(
                            "coupled_iteration" if transverse else "z"
                        ),
                        coordinate_unit=("1" if transverse else "um"),
                        elapsed_wall_time=monotonic() - phase_started_at,
                        message="Preparing GUI results...",
                        diagnostics={"phase": "gui_products"},
                    )
                )

            kwargs["_before_product_conversion"] = before_product_conversion
        if self.runner is self.slurm_runner:
            kwargs.update(self._remote_runner_kwargs())
        return self.runner.run_registered(
            PR_MATERIAL_ID,
            self._workflow_id_for_request(request),
            request,
            **kwargs,
        )

    def _slurm_supports_workflow(self, workflow_id: str) -> bool:
        """Return whether the active Slurm composition registers one PR operation."""

        if self.slurm_runner is None:
            return False
        return any(
            operation.key == (PR_MATERIAL_ID, workflow_id)
            for operation in self.slurm_runner.registered_operations
        )

    def save_checkpoint_to(self, run_dir):
        """Save the retained PR checkpoint through the shared dispatcher."""

        if self.last_checkpoint is None:
            raise ValueError("no PR checkpoint is available to save")
        return save_run_checkpoint(self.last_checkpoint, run_dir)

    def load_checkpoint_from(self, run_dir) -> PRTimeDependentCheckpoint:
        """Load, validate, and hydrate one PR checkpoint directory."""

        checkpoint = load_run_checkpoint(run_dir)
        if not isinstance(checkpoint, PRTimeDependentCheckpoint):
            raise TypeError(
                "LCProp PR can load only a pr/pr_timedependent checkpoint"
            )
        self._hydrating_checkpoint = True
        try:
            apply_pr_request(
                checkpoint.request,
                material_panel=self.material_panel,
                beam_panel=self.beam_panel,
                grid_panel=self.grid_panel,
                evolution_panel=self.evolution_panel,
            )
        finally:
            self._hydrating_checkpoint = False
        self.last_checkpoint = checkpoint
        self.results_panel.set_request_summary(
            self.describe_request(checkpoint.request)
        )
        self.results_panel.set_td_time_indicator(
            "Loaded PR checkpoint: "
            f"{checkpoint.time_normalized:.6g} normalized; "
            f"step {checkpoint.completed_steps}/{checkpoint.requested_steps}"
        )
        self._refresh_checkpoint_controls()
        return checkpoint

    @Slot()
    def save_checkpoint_clicked(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select PR checkpoint directory",
        )
        if not directory:
            return
        try:
            saved = self.save_checkpoint_to(directory)
        except Exception:
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())
            return
        self.results_panel.append_console(f"Saved PR checkpoint: {saved}")

    @Slot()
    def load_checkpoint_clicked(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Open PR checkpoint directory",
        )
        if not directory:
            return
        try:
            checkpoint = self.load_checkpoint_from(directory)
        except Exception:
            self.status_label.setText("Checkpoint load failed")
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())
            self.tabs.setCurrentWidget(self.results_panel)
            return
        self.results_panel.append_console(
            "Loaded PR checkpoint: "
            f"step {checkpoint.completed_steps}, "
            f"t={checkpoint.time_normalized:.6g} normalized"
        )

    @Slot()
    def continue_clicked(self) -> None:
        if (
            self._background_running
            or self._close_requested
            or self.last_checkpoint is None
            or self.input_panel.is_image_amplification()
            or self.evolution_panel.workflow_id()
            != PR_TIMEDEPENDENT_WORKFLOW
        ):
            return
        checkpoint = self.last_checkpoint
        try:
            request = self.build_request()
            validate_pr_continuation(request, checkpoint)
            summary = self.describe_request(request)
        except Exception:
            self.status_label.setText("Checkpoint incompatible")
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())
            self._refresh_checkpoint_controls()
            return
        additional_steps = int(request.solver.Nt)
        summary += (
            "\nContinuation checkpoint step: "
            f"{checkpoint.completed_steps}"
            "\nContinuation start time (normalized): "
            f"{checkpoint.time_normalized:.8g}"
            f"\nAdditional material steps: {additional_steps}"
        )
        runner_callable = partial(
            _continue_pr_operation,
            checkpoint=checkpoint,
            additional_steps=additional_steps,
        )
        self._start_background(
            request,
            summary=summary,
            runner_callable=runner_callable,
            run_label="Continuing",
        )

    @Slot()
    def run_clicked(self) -> None:
        if self._background_running or self._close_requested:
            return
        try:
            request = self.build_request()
            summary = self.describe_request(request)
            if self.runner is self.slurm_runner:
                self.remote_execution_controls.validate_backend(
                    request.backend.backend
                )
                workflow_id = self._workflow_id_for_request(request)
                if isinstance(request, PRImageAmplificationExperimentRequest):
                    workflow_id = request.base_workflow_id
                if not self._slurm_supports_workflow(workflow_id):
                    kind = (
                        "base operation"
                        if isinstance(
                            request, PRImageAmplificationExperimentRequest
                        )
                        else "operation"
                    )
                    raise ValueError(
                        "Slurm execution does not support the selected PR "
                        f"{kind} {workflow_id!r}"
                    )
        except Exception:
            self.status_label.setText("Invalid request")
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())
            self.tabs.setCurrentWidget(self.results_panel)
            return

        if not self._local_run_cost_guard(request):
            return

        if (
            self.last_checkpoint is not None
            and self._workflow_id_for_request(request)
            in (PR_TIMEDEPENDENT_WORKFLOW, PR_IMAGE_AMPLIFICATION_WORKFLOW)
        ):
            self.results_panel.append_console(
                "Starting a fresh PR run; previous checkpoint cleared."
            )
            self.last_checkpoint = None
        self._start_background(
            request,
            summary=summary,
            runner_callable=self._run_registered,
            run_label="Running",
        )

    def _request_for_local_cost(self, request):
        if isinstance(
            request,
            (
                PRBeamPanelImageAmplificationRunRequest,
                PRImageAmplificationExperimentRequest,
            ),
        ):
            composite = (
                request
                if isinstance(request, PRImageAmplificationExperimentRequest)
                else image_amplification_experiment_request(request)
            )
            base_request, _transmission, _grating = (
                prepare_image_amplification_base_request(composite)
            )
            return base_request
        if isinstance(request, PRImageAmplificationRunRequest):
            base_request, _transmission, _grating = (
                prepare_image_amplification_workflow_request(request)
            )
            return base_request
        return request

    @staticmethod
    def _local_cost_warning_text(assessment: LocalRunCostAssessment) -> str:
        nx, ny, nz = assessment.grid_shape
        return (
            f"Selected model: {assessment.model_label}\n"
            f"Grid: {nx} x {ny} x {nz}\n"
            "Execution: Local\n\n"
            "Full-transverse nonlinear PR can be very slow locally. "
            "Use Slurm/H200 for large full-transverse nonlinear PR runs.\n\n"
            "Stop is observed at the next safe solver cancellation checkpoint; "
            "an in-progress trial is discarded and the last accepted state is "
            "preserved."
        )

    def _show_potentially_expensive_local_warning(
        self, assessment: LocalRunCostAssessment
    ) -> None:
        QMessageBox.warning(
            self,
            "Potentially expensive local calculation",
            self._local_cost_warning_text(assessment),
        )

    def _confirm_very_expensive_local_run(
        self, assessment: LocalRunCostAssessment
    ) -> str:
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Very expensive local calculation")
        dialog.setText(self._local_cost_warning_text(assessment))
        cancel_button = dialog.addButton(
            "Cancel", QMessageBox.ButtonRole.RejectRole
        )
        slurm_button = dialog.addButton(
            "Use Slurm/H200", QMessageBox.ButtonRole.ActionRole
        )
        run_button = dialog.addButton(
            "Run locally anyway", QMessageBox.ButtonRole.DestructiveRole
        )
        slurm_button.setEnabled(self.slurm_runner is not None)
        dialog.setDefaultButton(cancel_button)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked is run_button:
            return "run_local"
        if clicked is slurm_button:
            return "use_slurm"
        return "cancel"

    def _local_run_cost_guard(self, request) -> bool:
        target = str(self.execution_target_selector.currentData())
        assessment = classify_pr_run_cost(
            self._request_for_local_cost(request), execution_target=target
        )
        if assessment.classification is LocalRunCostClass.NORMAL:
            return True
        if assessment.classification is LocalRunCostClass.POTENTIALLY_EXPENSIVE:
            self._show_potentially_expensive_local_warning(assessment)
            return True

        action = self._confirm_very_expensive_local_run(assessment)
        if action == "run_local":
            return True
        if action == "use_slurm":
            index = self.execution_target_selector.findData("slurm")
            if index >= 0 and self.slurm_runner is not None:
                self.execution_target_selector.setCurrentIndex(index)
                self.results_panel.append_console(
                    "Very expensive local calculation not started; Slurm/H200 "
                    "selected. Press Run to submit explicitly."
                )
        return False

    def _start_background(
        self,
        request,
        *,
        summary: str,
        runner_callable,
        run_label: str,
    ) -> None:
        self.results_panel.reset_field_color_scales()
        self.results_panel.set_request_summary(summary)
        workflow_id = self._workflow_id_for_request(request)
        self.results_panel.append_console(
            f"{run_label} {workflow_id} "
            f"with {self.runner.name}..."
        )
        preflight_request = request
        if isinstance(
            request,
            (
                PRBeamPanelImageAmplificationRunRequest,
                PRImageAmplificationExperimentRequest,
            ),
        ):
            composite = (
                request
                if isinstance(request, PRImageAmplificationExperimentRequest)
                else image_amplification_experiment_request(request)
            )
            preflight_request, _transmission, _grating = (
                prepare_image_amplification_base_request(
                    composite
                )
            )
            capability = next(
                capability
                for capability in image_amplification_base_capabilities()
                if capability.workflow_id == composite.base_workflow_id
            )
            self._append_image_amplification_validation_warning(
                capability, preflight_request
            )
        elif isinstance(request, PRImageAmplificationRunRequest):
            preflight_request, _transmission, _grating = (
                prepare_image_amplification_workflow_request(request)
            )
        preflight = validate_pr_gui_workflow_request(preflight_request)
        for warning in preflight.warnings:
            self.results_panel.append_console(f"WARNING: {warning}")

        self._active_request = request
        self._background_running = True
        self._outcome_received = False
        self._thread_done = False
        self.run_status = "running"
        self.last_progress = None
        self.status_label.setText("Running…")
        self.tabs.setCurrentWidget(self.results_panel)
        self._set_configuration_enabled(False)

        token = CancellationToken()
        thread = QThread(self)
        worker = WorkflowWorker(runner_callable, request, token)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(
            self._on_progress,
            Qt.ConnectionType.BlockingQueuedConnection,
        )
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._cancellation_token = token
        self._thread = thread
        self._worker = worker
        thread.start()

    def _append_image_amplification_validation_warning(
        self,
        capability,
        base_request,
    ) -> None:
        if (
            _image_amplification_validation_status(capability, base_request)
            != "compatible_and_validated"
        ):
            self.results_panel.append_console(
                "WARNING: selected Image Amplification base algorithm is "
                "experimental; specialized validation is pending."
            )

    def _set_configuration_enabled(self, enabled: bool) -> None:
        for panel in (
            self.material_panel,
            self.beam_panel,
            self.grid_panel,
            self.evolution_panel,
        ):
            panel.setEnabled(enabled)
        self.execution_target_selector.setEnabled(enabled)
        self.remote_execution_controls.setEnabled(enabled)
        self.run_button.setEnabled(enabled)
        self.experiment_file_buttons.setEnabled(enabled)
        self.load_checkpoint_button.setEnabled(enabled)
        self.stop_button.setVisible(not enabled)
        self.stop_button.setEnabled(not enabled)
        if enabled:
            self._refresh_checkpoint_controls()
        else:
            self.continue_button.setEnabled(False)
            self.save_checkpoint_button.setEnabled(False)

    @Slot()
    def stop_clicked(self) -> None:
        token = self._cancellation_token
        if not self._background_running or token is None:
            return
        token.cancel()
        self.run_status = "stopping"
        is_transverse_static = isinstance(
            self._active_request,
            PRTransverseStaticRunRequest,
        )
        is_static = is_transverse_static or isinstance(
            self._active_request,
            PRStaticRunRequest,
        )
        self.status_label.setText(
            "Stopping at next safe solver checkpoint…"
            if is_static
            else "Stopping at next safe internal boundary…"
        )
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Stopping…")
        self.results_panel.append_console(
            "Stop requested; discarding the in-progress trial at the next safe "
            "solver checkpoint; the last accepted state will be preserved."
            if is_transverse_static
            else "Stop requested; finishing the current accepted z slice."
            if is_static
            else "Stop requested; stopping at next safe internal boundary."
        )

    @Slot(object)
    def _on_progress(self, progress: RunProgress) -> None:
        self.last_progress = progress
        self.last_progress_thread = QThread.currentThread()
        if isinstance(progress, RemoteRunStatus):
            self.last_remote_status = progress
            self.remote_status_history.append(progress)
            message = remote_status_text(progress)
            self.status_label.setText(
                progress.state.value.replace("_", " ").title()
            )
            self.results_panel.set_td_time_indicator(message)
            self.results_panel.append_console(message)
            return
        if progress.workflow == PR_TRANSVERSE_STATIC_WORKFLOW:
            diagnostics = progress.diagnostics or {}
            if diagnostics.get("phase") == "gui_products":
                self.status_label.setText(progress.message)
                self.results_panel.set_td_time_indicator(progress.message)
                self.results_panel.append_console(
                    f"{progress.message}; "
                    f"elapsed={progress.elapsed_wall_time:.3f} s"
                )
                return
            linearized = diagnostics.get("material_response") == "linearized"
            self.status_label.setText(
                ("Linearized outer iteration " if linearized else "2D zero-flux iteration ")
                +
                f"{progress.completed_units}/{progress.total_units}"
            )
            self.results_panel.set_td_time_indicator(
                "PR transverse static coupled iteration: "
                f"{progress.completed_units}/{progress.total_units}"
            )
            self.results_panel.append_console(
                "PR transverse static progress: "
                f"iteration {progress.completed_units}/"
                f"{progress.total_units}; "
                +
                ("material consistency RMS=" if linearized else "equilibrium RMS=")
                +
                f"{diagnostics.get('material_response_rms', float('nan')):.6g}; "
                f"max={diagnostics.get('material_response_max', float('nan')):.6g}; "
                f"elapsed={progress.elapsed_wall_time:.3f} s"
            )
            return
        if progress.workflow == PR_IMAGE_AMPLIFICATION_WORKFLOW:
            self.status_label.setText(progress.message)
            self.results_panel.set_td_time_indicator(progress.message)
            self.results_panel.append_console(progress.message)
            return
        if progress.workflow == PR_STATIC_WORKFLOW:
            diagnostics = progress.diagnostics or {}
            phase = diagnostics.get("phase", "solve")
            if phase != "solve":
                self.status_label.setText(progress.message)
                if phase == "replay":
                    self.results_panel.set_td_time_indicator(
                        "Independent replay: "
                        f"{progress.completed_units}/{progress.total_units}"
                    )
                else:
                    self.results_panel.set_td_time_indicator(progress.message)
                self.results_panel.append_console(
                    f"{progress.message}; "
                    f"elapsed={progress.elapsed_wall_time:.3f} s"
                )
                return
            self.status_label.setText(
                f"Slice {progress.completed_units}/{progress.total_units}"
            )
            self.results_panel.set_td_time_indicator(
                "PR static z: "
                f"{float(progress.current_coordinate):.6g} µm; "
                f"slices: {progress.completed_units}/{progress.total_units}"
            )
            self.results_panel.append_console(
                "PR static progress: "
                f"slice {progress.completed_units}/{progress.total_units}; "
                f"z={float(progress.current_coordinate):.6g} µm; "
                f"elapsed={progress.elapsed_wall_time:.3f} s"
            )
            return
        self.status_label.setText(
            f"Step {progress.completed_units}/{progress.total_units}"
        )
        self.results_panel.set_td_time_indicator(
            "PR material time: "
            f"{float(progress.current_coordinate):.6g} normalized; "
            f"step {progress.completed_units}/{progress.total_units}"
        )
        self.results_panel.append_console(
            "PR progress: "
            f"step {progress.completed_units}/{progress.total_units}; "
            f"normalized time={float(progress.current_coordinate):.6g}; "
            f"elapsed={progress.elapsed_wall_time:.3f} s"
        )

    @Slot(object)
    def _on_finished(self, runner_result) -> None:
        try:
            if runner_result.material_id != PR_MATERIAL_ID:
                raise ValueError("PR window received a non-PR runner result")
            if runner_result.kind not in (
                PR_TIMEDEPENDENT_WORKFLOW,
                PR_IMAGE_AMPLIFICATION_WORKFLOW,
                PR_STATIC_WORKFLOW,
                PR_TRANSVERSE_STATIC_WORKFLOW,
                PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
            ):
                raise ValueError("PR window received an unexpected workflow")
            if runner_result.run_data is None:
                raise ValueError("PR runner result has no prepared RunData")
            result = runner_result.result
            ordinary_result = (
                result.run_result
                if runner_result.kind == PR_IMAGE_AMPLIFICATION_WORKFLOW
                else result
            )
            self.last_runner_result = runner_result
            self.last_result = result
            if runner_result.kind in (
                PR_STATIC_WORKFLOW,
                PR_TRANSVERSE_STATIC_WORKFLOW,
            ):
                self.status_label.setText("Rendering results...")
                self.status_label.repaint()
                self.results_panel.set_td_time_indicator("Rendering results...")
                self.results_panel.append_console("Rendering results...")
            self.results_panel.set_run_data(runner_result.run_data)
            if runner_result.kind == PR_IMAGE_AMPLIFICATION_WORKFLOW:
                td_result = ordinary_result
                analysis = (
                    result.analysis_result
                    if isinstance(result, PRImageAmplificationCompositeResult)
                    else result
                )
                self.last_checkpoint = None
                if result.status == "cancelled":
                    self.run_status = "stopped"
                    self.status_label.setText("Stopped")
                    prefix = "PR image time at stop"
                    message = "Image-amplification run cancelled"
                elif result.status == "not_converged":
                    self.run_status = "not_converged"
                    self.status_label.setText("Not converged")
                    prefix = "Final PR image time"
                    message = "Base PR solve did not converge"
                elif result.status not in ("completed", "converged"):
                    self.run_status = "failed"
                    self.status_label.setText("Analysis failed")
                    prefix = "Final PR image time"
                    message = getattr(
                        result,
                        "analysis_message",
                        "Image-amplification analysis failed",
                    )
                else:
                    self.run_status = "completed"
                    self.status_label.setText(
                        "Converged" if result.status == "converged" else "Completed"
                    )
                    prefix = "Final PR image time"
                    message = "Image-amplification run complete"
                if isinstance(td_result, (PRRunResult, PRTransverseRunResult)):
                    self.results_panel.set_td_time_indicator(
                        f"{prefix}: {float(td_result.time_normalized):.6g} "
                        f"normalized; steps: {td_result.completed_steps}/"
                        f"{td_result.requested_steps}"
                    )
                elif isinstance(td_result, PRTransverseStaticRunResult):
                    requested = int(
                        td_result.resolved_profile["solver"][
                            "max_coupled_iterations"
                        ]
                    )
                    self.results_panel.set_td_time_indicator(
                        "PR image transverse static: "
                        f"{td_result.completed_coupled_iterations}/"
                        f"{requested} coupled iterations"
                    )
                    diagnostics = td_result.diagnostics
                    self.results_panel.append_console(
                        "Authoritative zero-flux residual: "
                        f"RMS={diagnostics['equilibrium_residual_rms']:.8g}; "
                        f"max={diagnostics['equilibrium_residual_max']:.8g}"
                    )
                elif isinstance(td_result, PRStaticRunResult):
                    total_slices = int(td_result.grid_summary["Nz"])
                    self.results_panel.set_td_time_indicator(
                        "PR image static: "
                        f"{td_result.completed_slices}/{total_slices} slices"
                    )
                else:
                    raise TypeError(
                        "unsupported Image Amplification base result type: "
                        f"{type(td_result).__name__}"
                    )
                if analysis is not None:
                    self.results_panel.append_console(
                        "Image amplification: "
                        f"measured gain={analysis.measured_absolute_signal_gain:.8g}; "
                        f"analytic gain={analysis.analytic_absolute_signal_gain:.8g}; "
                        f"correlation={analysis.image_intensity_correlation:.8g}; "
                        f"NRMSE={analysis.normalized_image_rmse:.8g}"
                    )
            elif runner_result.kind == PR_TIMEDEPENDENT_WORKFLOW:
                self.last_checkpoint = result.checkpoint
                if result.status == "cancelled":
                    self.run_status = "stopped"
                    self.status_label.setText("Stopped")
                    prefix = "PR time at stop"
                    message = "Run cancelled"
                else:
                    self.run_status = "completed"
                    self.status_label.setText("Completed")
                    prefix = "Final PR time"
                    message = "Run complete"
                self.results_panel.set_td_time_indicator(
                    f"{prefix}: {float(result.time_normalized):.6g} normalized; "
                    f"steps: {result.completed_steps}/{result.requested_steps}"
                )
            elif runner_result.kind == PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW:
                self.last_checkpoint = None
                if result.status == "cancelled":
                    self.run_status = "stopped"
                    self.status_label.setText("Stopped")
                    prefix = "PR transverse time at stop"
                    message = "2D zero-flux time-dependent run cancelled"
                else:
                    self.run_status = "completed"
                    self.status_label.setText("Completed")
                    prefix = "Final PR transverse time"
                    message = "2D zero-flux time-dependent run complete"
                self.results_panel.set_td_time_indicator(
                    f"{prefix}: {float(result.time_normalized):.6g} normalized; "
                    f"steps: {result.completed_steps}/{result.requested_steps}"
                )
            elif runner_result.kind == PR_TRANSVERSE_STATIC_WORKFLOW:
                diagnostics = result.diagnostics
                linearized = diagnostics.get("material_response") == "linearized"
                model_name = (
                    "linearized material-response"
                    if linearized
                    else "2D zero-flux"
                )
                if result.status == "cancelled":
                    self.run_status = "stopped"
                    self.status_label.setText("Stopped")
                    message = f"{model_name} static run cancelled"
                elif result.status == "converged":
                    self.run_status = "completed"
                    self.status_label.setText("Converged")
                    message = f"{model_name} static solve converged"
                else:
                    self.run_status = "not_converged"
                    self.status_label.setText("Not converged")
                    message = f"{model_name} static solve did not converge"
                requested = int(
                    result.resolved_profile["solver"][
                        "max_coupled_iterations"
                    ]
                )
                self.results_panel.set_td_time_indicator(
                    "PR transverse static: "
                    f"{result.completed_coupled_iterations}/{requested} "
                    "coupled iterations"
                )
                self.results_panel.append_console(
                    (
                        "Material-response consistency residual: "
                        if linearized
                        else "Authoritative zero-flux residual: "
                    )
                    +
                    f"RMS={diagnostics['equilibrium_residual_rms']:.8g}; "
                    f"max={diagnostics['equilibrium_residual_max']:.8g}"
                )
                self.results_panel.append_console(
                    "Material state diagnostics: "
                    f"max|E_x|={diagnostics['E_x_max_abs']:.8g}; "
                    f"max|E_y|={diagnostics['E_y_max_abs']:.8g}; "
                    f"carrier minimum={diagnostics['carrier_minimum']:.8g}"
                )
                if linearized:
                    self.results_panel.append_console(
                        "Material solve work: analytic Fourier response; "
                        f"calls={diagnostics['material_response_calls']}; "
                        f"termination={diagnostics['termination_reason']}"
                    )
                else:
                    solver_summary = diagnostics["discrete_corrector"]
                    self.results_panel.append_console(
                        "Material solve work: "
                        "Newton="
                        f"{solver_summary['continuum_newton_iterations_attempted']}; "
                        f"PCG={solver_summary['continuum_pcg_iterations']}; "
                        f"termination={diagnostics['termination_reason']}"
                    )
            else:
                total_slices = int(result.grid_summary["Nz"])
                z_reached_um = float(
                    result.completed_slices * result.grid_summary["dz_um"]
                )
                if result.status == "cancelled":
                    self.run_status = "stopped"
                    self.status_label.setText("Stopped")
                    prefix = "PR static z at stop"
                    message = "Static run cancelled"
                elif result.status == "converged":
                    self.run_status = "completed"
                    self.status_label.setText("Converged")
                    prefix = "Final PR static z"
                    message = "Static solve converged"
                else:
                    self.run_status = "not_converged"
                    self.status_label.setText("Not converged")
                    prefix = "Final PR static z"
                    message = "Static solve did not converge"
                self.results_panel.set_td_time_indicator(
                    f"{prefix}: {z_reached_um:.6g} µm; "
                    f"slices: {result.completed_slices}/{total_slices}"
                )
            self.results_panel.append_console(runner_result.message)
            self.results_panel.append_console(message)
            self.results_panel.append_console(
                "Normalized optical power: "
                f"{ordinary_result.power_initial:.8g} -> "
                f"{ordinary_result.power_final:.8g}"
            )
        except Exception:
            self.run_status = "failed"
            self.status_label.setText("Failed")
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())
        self._outcome_received = True
        self._maybe_finish_background()

    @Slot(str)
    def _on_failed(self, formatted_traceback: str) -> None:
        if (
            self.last_remote_status is not None
            and self.last_remote_status.state == RemoteRunState.CANCELLED
        ):
            self.run_status = "stopped"
            self.status_label.setText("Stopped")
            self.results_panel.append_console("Remote job cancelled")
            self._outcome_received = True
            self._maybe_finish_background()
            return
        self.run_status = "failed"
        self.status_label.setText("Failed")
        self.results_panel.append_console("ERROR")
        self.results_panel.append_console(formatted_traceback)
        self._outcome_received = True
        self._maybe_finish_background()

    @Slot()
    def _on_thread_finished(self) -> None:
        thread = self._thread
        if thread is not None:
            # ``finished`` precedes completion of all native-thread cleanup.
            # Join before dropping the last explicit reference or allowing
            # the parent window to be destroyed.
            thread.wait()
        self._thread = None
        self._worker = None
        self._thread_done = True
        self._maybe_finish_background()
        if self._close_requested:
            QTimer.singleShot(0, self.close)

    def _maybe_finish_background(self) -> None:
        if self._outcome_received and self._thread_done:
            self._finish_background()

    def _finish_background(self) -> None:
        self._background_running = False
        self._active_request = None
        self._cancellation_token = None
        self.stop_button.setText("Stop")
        if not self._close_requested:
            self._set_configuration_enabled(True)

    def shutdown_background_run(self, timeout_ms: int = 30000) -> bool:
        """Cooperatively cancel and join the PR worker without GUI deadlock."""

        if not self.remote_execution_controls.shutdown_connection_test(
            min(timeout_ms, 16000)
        ):
            return False

        if self._thread is None:
            return True
        if self._cancellation_token is not None:
            self._cancellation_token.cancel()
        self.run_status = "stopping"
        self.status_label.setText("Stopping…")
        deadline = monotonic() + max(0, timeout_ms) / 1000.0
        while self._thread is not None:
            thread = self._thread
            if not thread.isRunning():
                thread.wait()
                QCoreApplication.processEvents()
                if self._thread is thread:
                    self._thread = None
                    self._worker = None
                    self._background_running = False
                return True
            remaining_ms = int(max(0.0, deadline - monotonic()) * 1000)
            if remaining_ms <= 0:
                return False
            thread.wait(min(10, remaining_ms))
            # Blocking progress delivery requires the GUI thread to service
            # queued signals. Do not retain/use ``thread`` after this call:
            # the finished/deleteLater path may have run.
            QCoreApplication.processEvents()
        return True

    def closeEvent(self, event) -> None:
        thread = self._thread
        if self._background_running or (
            thread is not None and thread.isRunning()
        ):
            first_request = not self._close_requested
            self._close_requested = True
            if self._cancellation_token is not None:
                self._cancellation_token.cancel()
            self.run_status = "stopping"
            self.status_label.setText("Stopping…")
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Stopping…")
            if first_request:
                self.results_panel.append_console(
                    "Close requested; waiting for the active PR workflow to "
                    "finish at a safe boundary."
                )
            event.ignore()
            return
        if thread is not None:
            thread.wait()
        self._close_requested = False
        event.accept()


__all__ = ["PRMainWindow"]
