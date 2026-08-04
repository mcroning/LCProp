"""Standalone Qt window for the photorefractive workflow."""

from __future__ import annotations

from functools import partial
from time import monotonic
import traceback

from PySide6.QtCore import QCoreApplication, QThread, Qt, Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QFileDialog,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.gui.panels.results_panel import ResultsPanel
from lcprop.gui.workers import WorkflowWorker
from lcprop.persistence import load_run_checkpoint, save_run_checkpoint
from lcprop.pr.checkpoint import (
    PRTimeDependentCheckpoint,
    validate_pr_continuation,
)
from lcprop.pr.gui.beam_panel import make_pr_beam_panel
from lcprop.pr.gui.evolution_panel import PREvolutionPanel
from lcprop.pr.gui.grid_panel import PRGridPanel
from lcprop.pr.gui.material_panel import PRMaterialPanel
from lcprop.pr.gui.request_adapter import (
    apply_pr_request,
    build_pr_request,
    validate_pr_gui_request,
)
from lcprop.pr.operations import PR_TIMEDEPENDENT_OPERATION
from lcprop.pr.specs import PR_MATERIAL_ID, PR_TIMEDEPENDENT_WORKFLOW
from lcprop.pr.workflow import continue_pr_timedependent
from lcprop.runners.base import RunnerResult
from lcprop.runners.local import LocalRunner


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
    """Focused GUI for the headless PR time-dependent operation."""

    def __init__(self) -> None:
        super().__init__()
        self.runner = LocalRunner(operations=(PR_TIMEDEPENDENT_OPERATION,))
        self.last_result = None
        self.last_runner_result = None
        self.last_progress: RunProgress | None = None
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
        self._hydrating_checkpoint = False
        self.checkpoint_compatibility_reason: str | None = None

        self.setWindowTitle("LCProp PR")
        self.setMinimumSize(1200, 760)
        self.resize(1450, 900)

        root = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(QLabel("LCProp PR"))
        header.addStretch(1)
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
        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)
        self.material_panel = PRMaterialPanel()
        self.grid_panel = PRGridPanel()
        self.beam_panel = make_pr_beam_panel(
            x_aperture_um=self.grid_panel.x_aperture_um.value(),
            y_aperture_um=self.grid_panel.y_aperture_um.value(),
        )
        self.evolution_panel = PREvolutionPanel()
        self.results_panel = ResultsPanel()
        self.tabs.addTab(self.material_panel, "PR Material")
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
        self._connect_checkpoint_compatibility_signals()

    def _sync_beam_aperture(self) -> None:
        self.beam_panel.set_aperture(
            self.grid_panel.x_aperture_um.value(),
            self.grid_panel.y_aperture_um.value(),
        )

    def _connect_checkpoint_compatibility_signals(self) -> None:
        for panel in (
            self.material_panel,
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
        if self._hydrating_checkpoint or self._background_running:
            return
        self._refresh_checkpoint_controls()

    def _refresh_checkpoint_controls(self) -> None:
        checkpoint = self.last_checkpoint
        reason = None
        if checkpoint is None:
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
            actions_enabled and checkpoint is not None
        )
        if checkpoint is None:
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

        return build_pr_request(
            material_panel=self.material_panel,
            beam_panel=self.beam_panel,
            grid_panel=self.grid_panel,
            evolution_panel=self.evolution_panel,
        )

    def describe_request(self, request) -> str:
        """Return a durable, unit-explicit summary of one PR request."""

        preflight = validate_pr_gui_request(request)
        lines = [
            "Material: photorefractive",
            f"Workflow: {PR_TIMEDEPENDENT_WORKFLOW}",
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
        lines.extend([
            f"Material steps: {request.solver.Nt}",
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
        ])
        if preflight.warnings:
            lines.append("Preflight warnings:")
            lines.extend(f"- {warning}" for warning in preflight.warnings)
        else:
            lines.append("Preflight warnings: none")
        return "\n".join(lines)

    def _run_registered(self, request, **kwargs):
        return self.runner.run_registered(
            PR_MATERIAL_ID,
            PR_TIMEDEPENDENT_WORKFLOW,
            request,
            **kwargs,
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
        if self._background_running or self.last_checkpoint is None:
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
        if self._background_running:
            return
        try:
            request = self.build_request()
            summary = self.describe_request(request)
        except Exception:
            self.status_label.setText("Invalid request")
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())
            self.tabs.setCurrentWidget(self.results_panel)
            return

        if self.last_checkpoint is not None:
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
        self.results_panel.append_console(
            f"{run_label} {PR_TIMEDEPENDENT_WORKFLOW} "
            f"with {self.runner.name}..."
        )
        preflight = validate_pr_gui_request(request)
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

    def _set_configuration_enabled(self, enabled: bool) -> None:
        for panel in (
            self.material_panel,
            self.beam_panel,
            self.grid_panel,
            self.evolution_panel,
        ):
            panel.setEnabled(enabled)
        self.run_button.setEnabled(enabled)
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
        self.status_label.setText("Stopping…")
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Stopping…")
        self.results_panel.append_console(
            "Stop requested; finishing the current material-time step."
        )

    @Slot(object)
    def _on_progress(self, progress: RunProgress) -> None:
        self.last_progress = progress
        self.last_progress_thread = QThread.currentThread()
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
            if runner_result.kind != PR_TIMEDEPENDENT_WORKFLOW:
                raise ValueError("PR window received an unexpected workflow")
            if runner_result.run_data is None:
                raise ValueError("PR runner result has no prepared RunData")
            result = runner_result.result
            self.last_runner_result = runner_result
            self.last_result = result
            self.last_checkpoint = result.checkpoint
            self.results_panel.set_run_data(runner_result.run_data)
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
            self.results_panel.append_console(runner_result.message)
            self.results_panel.append_console(message)
            self.results_panel.append_console(
                "Normalized optical power: "
                f"{result.power_initial:.8g} -> {result.power_final:.8g}"
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
        self.run_status = "failed"
        self.status_label.setText("Failed")
        self.results_panel.append_console("ERROR")
        self.results_panel.append_console(formatted_traceback)
        self._outcome_received = True
        self._maybe_finish_background()

    @Slot()
    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._thread_done = True
        self._maybe_finish_background()

    def _maybe_finish_background(self) -> None:
        if self._outcome_received and self._thread_done:
            self._finish_background()

    def _finish_background(self) -> None:
        self._background_running = False
        self._active_request = None
        self._cancellation_token = None
        self.stop_button.setText("Stop")
        self._set_configuration_enabled(True)

    def shutdown_background_run(self, timeout_ms: int = 30000) -> bool:
        """Cooperatively cancel and join the PR worker without GUI deadlock."""

        thread = self._thread
        if thread is None or not thread.isRunning():
            return True
        if self._cancellation_token is not None:
            self._cancellation_token.cancel()
        self.run_status = "stopping"
        self.status_label.setText("Stopping…")
        deadline = monotonic() + max(0, timeout_ms) / 1000.0
        while thread.isRunning() and monotonic() < deadline:
            QCoreApplication.processEvents()
            thread.wait(10)
        finished = not thread.isRunning()
        if finished:
            self._thread = None
            self._worker = None
            self._background_running = False
        return finished

    def closeEvent(self, event) -> None:
        if self.shutdown_background_run():
            event.accept()
        else:
            self.results_panel.append_console(
                "Close delayed: PR workflow did not stop within the shutdown "
                "timeout."
            )
            event.ignore()


__all__ = ["PRMainWindow"]
