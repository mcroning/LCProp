from __future__ import annotations
from dataclasses import replace
from functools import partial

import traceback

import numpy as np

from PySide6.QtCore import QThread, Qt, Slot
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTabWidget,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
)

from lcprop.products.data_model import (
    from_static_live_state,
    from_timedependent_live_state,
    to_run_data,
)
from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.runners.local import LocalRunner
from lcprop.gui.workers import WorkflowWorker
from lcprop.gui.panels import (
    ExperimentPanel,
    PhysicsPanel,
    BeamPanel,
    GridPanel,
    SolverPanel,
    SweepPanel,
    ResultsPanel,
)
from lcprop.core.requests import StaticRunRequest, TimeDependentRunRequest, OutputOptions, RuntimeOptions
from lcprop.workflows.soliton import SolitonRequest
from lcprop.workflows.sweep import ParameterSweepRequest
from lcprop.workflows.timedependent import timedependent_state_from_static_result

class LCPropMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.runner = LocalRunner()
        self.last_soliton_result = None
        self.last_timedependent_checkpoint = None
        self.last_timedependent_result = None
        self.last_timedependent_progress = None
        self.last_run_progress = None
        self.last_static_checkpoint = None
        self.last_static_result = None
        self._active_td_from_static = False
        self._td_static_reference = None
        self._last_td_progress_thread = None
        self._background_running = False
        self.run_status = "idle"
        self._td_thread = None
        self._td_worker = None
        self._td_cancellation_token = None
        self._td_outcome_received = False
        self._td_thread_done = False
        self.setWindowTitle("LCProp")
        # Default to a wide scientific-visualization layout.
        self.setMinimumSize(1200, 760)

        root = QVBoxLayout(self)

        header = QHBoxLayout()
        header.addWidget(QLabel("LCProp"))
        header.addStretch(1)
        header.addWidget(QLabel(f"Runner: {self.runner.name}"))

        self.use_last_soliton = QCheckBox("Start from last soliton")
        self.use_last_soliton.setEnabled(False)
        self.use_last_soliton.setVisible(False)
        header.addWidget(self.use_last_soliton)

        self.use_last_static = QCheckBox("Start TD from last static result")
        self.use_last_static.setEnabled(False)
        self.use_last_static.setVisible(False)
        header.addWidget(self.use_last_static)

        self.run_button = QPushButton()
        self.run_button.clicked.connect(self.run_static_clicked)
        header.addWidget(self.run_button)

        self.continue_button = QPushButton("Continue")
        self.continue_button.setVisible(False)
        self.continue_button.setEnabled(False)
        self.continue_button.clicked.connect(self.continue_workflow_clicked)
        header.addWidget(self.continue_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop_workflow_clicked)
        header.addWidget(self.stop_button)

        root.addLayout(header)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.experiment_panel = ExperimentPanel()
        self.physics_panel = PhysicsPanel()
        self.grid_panel = GridPanel()
        self.beam_panel = BeamPanel(
            x_aperture_um=self.grid_panel.x_aperture_um.value(),
            y_aperture_um=self.grid_panel.y_aperture_um.value(),
        )
        self.solver_panel = SolverPanel()
        self.sweep_panel = SweepPanel()
        self.results_panel = ResultsPanel()

        self.tabs.addTab(self.experiment_panel, "Experiment")
        self.tabs.addTab(self.physics_panel, "Physics")
        self.tabs.addTab(self.beam_panel, "Beam")
        self.tabs.addTab(self.grid_panel, "Grid")
        self.tabs.addTab(self.solver_panel, "Solver")
        self.sweep_tab_index = self.tabs.addTab(self.sweep_panel, "Sweep")
        self.tabs.addTab(self.results_panel, "Results")
        self.experiment_panel.experimentChanged.connect(self.update_run_button)
        self.grid_panel.x_aperture_um.valueChanged.connect(
            self._sync_beam_aperture
        )
        self.grid_panel.y_aperture_um.valueChanged.connect(
            self._sync_beam_aperture
        )
        self._connect_continuation_invalidation_signals()
        self.update_run_button()   
        self.resize(1450, 900)

    def _sync_beam_aperture(self) -> None:
        self.beam_panel.set_aperture(
            self.grid_panel.x_aperture_um.value(),
            self.grid_panel.y_aperture_um.value(),
        )

    def update_run_button(self) -> None:
        if self._background_running:
            self.run_button.setText("Running…")
            return
        experiment = self.experiment_panel.current_experiment()
        self.run_button.setText(f"Run {experiment}")
        is_continuable = experiment in {
            "Static propagation", "Time-dependent propagation"
        }
        checkpoint = (
            self.last_timedependent_checkpoint
            if experiment == "Time-dependent propagation"
            else self.last_static_checkpoint
        )
        checkpoint_compatible = checkpoint is not None
        if experiment == "Time-dependent propagation" and checkpoint is not None:
            checkpoint_compatible = self._td_checkpoint_matches_current_request(
                checkpoint
            )
        self.continue_button.setVisible(is_continuable)
        self.continue_button.setEnabled(
            is_continuable
            and checkpoint_compatible
            and not self._background_running
        )
        self.solver_panel.set_experiment_mode(experiment)
        self._update_sweep_tab(experiment)
        self._update_initial_condition_controls(experiment)

    def _connect_continuation_invalidation_signals(self) -> None:
        """Invalidate a retained TD checkpoint as soon as its request changes."""

        for panel in (
            self.physics_panel,
            self.beam_panel,
            self.grid_panel,
            self.solver_panel,
        ):
            for widget_type in (QSpinBox, QDoubleSpinBox):
                for widget in panel.findChildren(widget_type):
                    widget.valueChanged.connect(self._configuration_changed)
            for widget in panel.findChildren(QComboBox):
                widget.currentIndexChanged.connect(self._configuration_changed)
            for widget in panel.findChildren(QCheckBox):
                widget.toggled.connect(self._configuration_changed)
            for widget in panel.findChildren(QLineEdit):
                widget.editingFinished.connect(self._configuration_changed)
        self.use_last_soliton.toggled.connect(self._configuration_changed)
        self.use_last_static.toggled.connect(self._configuration_changed)

    def _td_checkpoint_matches_current_request(self, checkpoint) -> bool:
        try:
            self.runner.validate_timedependent_continuation(
                self.build_timedependent_request(), checkpoint
            )
        except Exception:
            return False
        return True

    def _configuration_changed(self, *_args) -> None:
        checkpoint = self.last_timedependent_checkpoint
        if checkpoint is None or self._background_running:
            return
        try:
            request = self.build_timedependent_request()
            self.runner.validate_timedependent_continuation(request, checkpoint)
        except Exception as exc:
            self.last_timedependent_checkpoint = None
            self.results_panel.append_console(
                f"Continuation invalidated: {exc}"
            )
        self.update_run_button()

    def _update_sweep_tab(self, experiment: str) -> None:
        sweep_enabled = experiment == "Soliton existence curve"
        self.tabs.setTabEnabled(self.sweep_tab_index, sweep_enabled)
        if not sweep_enabled and self.tabs.currentIndex() == self.sweep_tab_index:
            self.tabs.setCurrentWidget(self.experiment_panel)

    def _update_initial_condition_controls(self, experiment: str) -> None:
        visible = experiment in {"Static propagation", "Time-dependent propagation"}
        enabled = visible and self.last_soliton_result is not None
        self.use_last_soliton.setVisible(visible)
        self.use_last_soliton.setEnabled(enabled)
        if not enabled:
            self.use_last_soliton.setChecked(False)
        self.use_last_static.setVisible(
            experiment == "Time-dependent propagation"
        )
        self.use_last_static.setEnabled(
            experiment == "Time-dependent propagation"
            and self.last_static_result is not None
            and getattr(self.last_static_result, "status", None) == "completed"
        )
        if not self.use_last_static.isEnabled():
            self.use_last_static.setChecked(False)

    def _maybe_apply_last_soliton(self, req, *, allow: bool):
        if not allow or not self.use_last_soliton.isChecked():
            return req
        if self.last_soliton_result is None:
            return req
        return replace(
            req,
            initial_A=self.last_soliton_result.A,
            initial_theta=self.last_soliton_result.theta,
        )

    def build_request(self, *, allow_last_soliton: bool = True) -> StaticRunRequest:
        req = StaticRunRequest(
            grid=self.grid_panel.grid(),
            material=self.physics_panel.material(),
            bias=self.physics_panel.bias(),
            beams=self.beam_panel.beams(),
            solver=self.solver_panel.solver(),
            output=OutputOptions(),
        )
        return self._maybe_apply_last_soliton(req, allow=allow_last_soliton)

    def build_timedependent_request(self) -> TimeDependentRunRequest:
        req = TimeDependentRunRequest(
            grid=self.grid_panel.grid(),
            material=self.physics_panel.material(),
            bias=self.physics_panel.bias(),
            beams=self.beam_panel.beams(),
            solver=self.solver_panel.td_solver(),
            output=OutputOptions(),
            runtime=RuntimeOptions(precision="float64"),
        )
        req = self._maybe_apply_last_soliton(req, allow=True)
        if self.use_last_static.isChecked():
            if self.last_static_result is None:
                raise ValueError("no completed static result is available")
            req = timedependent_state_from_static_result(
                self.last_static_result, req
            )
        return req

    def build_soliton_request(self) -> SolitonRequest:
        return SolitonRequest(
            base=self.build_request(allow_last_soliton=False),
            mode=self.solver_panel.soliton_mode(),
            refine_transverse=self.solver_panel.refine_transverse(),
        )

    def build_soliton_existence_request(self) -> ParameterSweepRequest:
        base = SolitonRequest(
            base=self.build_request(allow_last_soliton=False),
            mode=self.solver_panel.soliton_mode(),
        )

        return ParameterSweepRequest(
            experiment="soliton",
            parameter=self.sweep_panel.sweep_parameter(),
            values=self.sweep_panel.sweep_values(),
            base=base,
            continuation=self.sweep_panel.use_continuation(),
            execution=self.sweep_panel.sweep_execution(),
        )

    def _base_static_request(self, req):
        base_req = req
        while hasattr(base_req, "base"):
            base_req = base_req.base
        return base_req

    def describe_request(self, req) -> str:
        base_req = self._base_static_request(req)
        channels = base_req.beams.channels
        first_channel = channels[0]
        total_power_mW = sum(channel.power_mW for channel in channels)
        wavelengths = tuple(
            dict.fromkeys(channel.wavelength_um for channel in channels)
        )
        coherence_groups = tuple(
            dict.fromkeys(channel.coherence_group for channel in channels)
        )
        lines = [
            f"Experiment: {self.experiment_panel.current_experiment()}",
            f"Runner: {self.runner.name}",
            f"Grid: {base_req.grid.Nx} × {base_req.grid.Ny}, Nz≈{round(base_req.grid.z_length_um / base_req.grid.dz_um)}",
            f"Aperture: x={base_req.grid.x_aperture_um:g} µm, y={base_req.grid.y_aperture_um:g} µm",
            f"z length: {base_req.grid.z_length_um:g} µm, dz={base_req.grid.dz_um:g} µm",
            f"Material: ne={base_req.material.ne:g}, no={base_req.material.no:g}, K={base_req.material.K:g}, Δε={base_req.material.delta_epsilon:g}",
            f"Bias: V={base_req.bias.V_bias:g} V, theta_bc={base_req.bias.theta_bc:g} rad",
            f"Beams: {len(channels)} enabled, total P={total_power_mW:g} mW",
            "Wavelengths: " + ", ".join(f"{value:g} µm" for value in wavelengths),
            "Lasers/coherence groups: " + ", ".join(coherence_groups),
            f"First enabled beam: {first_channel.name}, P={first_channel.power_mW:g} mW, "
            f"waists=({first_channel.waist_x_um:g}, {first_channel.waist_y_um:g}) µm, "
            f"λ={first_channel.wavelength_um:g} µm",
            f"Workflow: {base_req.solver.workflow.strategy}",
            f"Initial condition: {'last soliton' if getattr(base_req, 'initial_A', None) is not None or getattr(base_req, 'initial_theta', None) is not None else 'default launch'}",
        ]

        if isinstance(req, TimeDependentRunRequest) and self.use_last_static.isChecked():
            source = self.last_static_result
            lines[-1] = "Initial condition: static result"
            lines.extend([
                f"Static source status: {source.status}",
                f"Source z length: {source.grid_summary['z_length_um']} um",
                "Source grid: "
                f"{source.grid_summary['Nx']} x {source.grid_summary['Ny']} x "
                f"{source.grid_summary['Nz']}",
                "Source residual summary: max RMS="
                f"{source.max_final_residual_rms}",
            ])

        if hasattr(base_req.solver, "resolved_static_max_coupled_passes"):
            lines.extend([
                f"Static max coupled passes: {base_req.solver.resolved_static_max_coupled_passes}",
                f"Static max relax iterations/pass: {base_req.solver.static_max_relax_iterations}",
                f"Static residual RMS tolerance: {base_req.solver.static_residual_rms_tol}",
                f"Static residual max tolerance: {base_req.solver.static_residual_max_tol}",
                f"Static Δθ RMS tolerance: {base_req.solver.resolved_delta_theta_rms_tol}",
                f"Static Δθ max tolerance: {base_req.solver.resolved_delta_theta_max_tol}",
                f"Record static iteration history: {base_req.solver.record_iteration_history}",
            ])

        mode = getattr(req, "mode", None)
        if mode is not None:
            lines.append(f"Soliton mode: {mode}")

        sweep_values = getattr(req, "values", None)
        if sweep_values is not None:
            parameter = getattr(req, "parameter", "parameter")
            lines.append(f"Sweep parameter: {parameter}")
            lines.append("Sweep values: " + ", ".join(f"{p:g}" for p in sweep_values))
            lines.append(f"Continuation: {getattr(req, 'continuation', False)}")
            lines.append(f"Execution: {getattr(req, 'execution', 'sequential')}")
        else:
            powers = getattr(req, "powers_mW", None)
            if powers is not None:
                lines.append("Powers: " + ", ".join(f"{p:g} mW" for p in powers))

        return "\n".join(lines)

    def _experiment_dispatch(self):
        return {
            "Static propagation": (
                self.build_request,
                self.runner.run_static,
                "static workflow",
            ),
            "Time-dependent propagation": (
                self.build_timedependent_request,
                self.runner.run_timedependent,
                "time-dependent workflow",
            ),
            "Soliton": (
                self.build_soliton_request,
                self.runner.run_soliton,
                "soliton workflow",
            ),
            "Soliton existence curve": (
                self.build_soliton_existence_request,
                self.runner.run_parameter_sweep,
                "soliton existence workflow",
            ),
        }

    def run_static_clicked(self):
        if self._background_running:
            return
        if self.experiment_panel.current_experiment() == "Time-dependent propagation":
            self._start_timedependent_background()
            return
        if self.experiment_panel.current_experiment() == "Static propagation":
            self._start_static_background()
            return

        self.run_button.setEnabled(False)
        self.run_button.setText("Running…")
        self.tabs.setCurrentWidget(self.results_panel)

        try:
            experiment = self.experiment_panel.current_experiment()
            try:
                request_builder, runner_result_fn, run_label = self._experiment_dispatch()[experiment]
            except KeyError as exc:
                raise ValueError(f"Unsupported experiment: {experiment}") from exc
            req = request_builder()

            self.results_panel.set_request_summary(self.describe_request(req))
            self.results_panel.append_console(f"Running {run_label} with {self.runner.name}...")
            QApplication.processEvents()

            runner_result = runner_result_fn(req)
            result = runner_result.result
            if type(result).__name__ == "SolitonResult":
                self.last_soliton_result = result
                self.results_panel.append_console("Saved this soliton as the current in-memory initial condition.")
            run_data = to_run_data(result)
            self.results_panel.set_run_data(run_data)

            self.results_panel.append_console("")
            self.results_panel.append_console(runner_result.message)
            self.results_panel.append_console("Run complete")
            self._append_result_summary(result, runner_result.kind)

        except Exception:
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())

        finally:
            self.run_button.setEnabled(True)
            self.update_run_button()

    def _set_background_controls_enabled(self, enabled: bool) -> None:
        for panel in (
            self.experiment_panel,
            self.physics_panel,
            self.beam_panel,
            self.grid_panel,
            self.solver_panel,
            self.sweep_panel,
        ):
            panel.setEnabled(enabled)
        self.use_last_soliton.setEnabled(
            enabled and self.last_soliton_result is not None
        )
        self.use_last_static.setEnabled(
            enabled
            and self.experiment_panel.current_experiment()
            == "Time-dependent propagation"
            and self.last_static_result is not None
            and self.last_static_result.status == "completed"
        )
        experiment = self.experiment_panel.current_experiment()
        checkpoint = (
            self.last_timedependent_checkpoint
            if experiment == "Time-dependent propagation"
            else self.last_static_checkpoint
        )
        self.run_button.setEnabled(enabled)
        self.continue_button.setEnabled(
            enabled
            and experiment in {"Static propagation", "Time-dependent propagation"}
            and checkpoint is not None
        )
        self.stop_button.setVisible(not enabled)
        self.stop_button.setEnabled(not enabled)

    def _start_timedependent_background(
        self,
        *,
        request=None,
        runner_callable=None,
        cumulative_start_time: float = 0.0,
        is_continuation: bool = False,
        workflow: str = "timedependent",
    ) -> None:
        try:
            req = self.build_timedependent_request() if request is None else request
            summary = self.describe_request(req)
        except Exception:
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())
            return

        old_checkpoint = (
            self.last_timedependent_checkpoint
            if workflow == "timedependent"
            else self.last_static_checkpoint
        )
        if not is_continuation and old_checkpoint is not None:
            workflow_label = "TD" if workflow == "timedependent" else "static"
            self.results_panel.append_console(
                f"Starting a fresh {workflow_label} run; previous continuation checkpoint cleared."
            )
            if workflow == "timedependent":
                self.last_timedependent_checkpoint = None
            else:
                self.last_static_checkpoint = None

        if workflow == "timedependent" and not is_continuation:
            self._active_td_from_static = self.use_last_static.isChecked()
            self._td_static_reference = None

        if not is_continuation:
            self.results_panel.reset_field_color_scales()

        self._background_running = True
        self.run_status = "running"
        self.last_timedependent_progress = None
        self.last_run_progress = None
        self._td_outcome_received = False
        self._td_thread_done = False
        self._set_background_controls_enabled(False)
        self.run_button.setText("Running…")
        self.tabs.setCurrentWidget(self.results_panel)
        self.results_panel.set_request_summary(summary)
        self.results_panel.append_console(
            f"Running {workflow} workflow with {self.runner.name}..."
        )
        coordinate = "TD time" if workflow == "timedependent" else "Static z"
        unit = "" if workflow == "timedependent" else " um"
        self.results_panel.set_td_time_indicator(
            f"{coordinate}: {self._format_coordinate(cumulative_start_time)}{unit}"
        )

        token = CancellationToken()
        thread = QThread(self)
        worker = WorkflowWorker(
            (self.runner.run_timedependent if workflow == "timedependent" else self.runner.run_static)
            if runner_callable is None else runner_callable,
            req,
            token,
        )
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        # Apply backpressure so every safe-boundary live state is displayed
        # before the worker advances and progress signals cannot form a stale
        # GUI queue during fast tiny runs.
        worker.progress.connect(
            self._on_workflow_progress,
            Qt.ConnectionType.BlockingQueuedConnection,
        )
        worker.finished.connect(self._on_timedependent_finished)
        worker.failed.connect(self._on_timedependent_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self._on_timedependent_thread_finished)
        thread.finished.connect(thread.deleteLater)

        self._td_cancellation_token = token
        self._td_thread = thread
        self._td_worker = worker
        thread.start()

    def _start_static_background(
        self,
        *,
        request=None,
        runner_callable=None,
        z_start_um: float = 0.0,
        is_continuation: bool = False,
    ) -> None:
        req = self.build_request() if request is None else request
        self._start_timedependent_background(
            request=req,
            runner_callable=runner_callable,
            cumulative_start_time=z_start_um,
            is_continuation=is_continuation,
            workflow="static",
        )

    def start_timedependent_continuation(
        self,
        request,
        checkpoint,
        additional_steps: int,
    ) -> None:
        """Start an in-memory continuation without adding saved-run UI."""
        self.runner.validate_timedependent_continuation(request, checkpoint)
        continuation_runner = partial(
            self.runner.continue_timedependent,
            checkpoint=checkpoint,
            additional_steps=additional_steps,
        )
        self._start_timedependent_background(
            request=request,
            runner_callable=continuation_runner,
            cumulative_start_time=checkpoint.current_time,
            is_continuation=True,
        )

    @Slot()
    def continue_workflow_clicked(self) -> None:
        is_td = self.experiment_panel.current_experiment() == "Time-dependent propagation"
        checkpoint = self.last_timedependent_checkpoint if is_td else self.last_static_checkpoint
        if self._background_running or checkpoint is None:
            return
        try:
            request = self.build_timedependent_request() if is_td else self.build_request()
            if is_td:
                self.runner.validate_timedependent_continuation(request, checkpoint)
            else:
                self.runner.validate_static_continuation(request, checkpoint)
        except ValueError as exc:
            if is_td:
                self.last_timedependent_checkpoint = None
            else:
                self.last_static_checkpoint = None
            self.results_panel.append_console(
                f"Continuation invalidated: {exc}"
            )
            self.update_run_button()
            return
        except Exception:
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())
            return

        if is_td:
            self.start_timedependent_continuation(
                request, checkpoint, additional_steps=request.solver.Nt
            )
        else:
            continuation_runner = partial(
                self.runner.continue_static, checkpoint=checkpoint
            )
            self._start_static_background(
                request=request,
                runner_callable=continuation_runner,
                z_start_um=checkpoint.z_reached_um,
                is_continuation=True,
            )

    # Compatibility slot name retained for existing callers/tests.
    continue_timedependent_clicked = continue_workflow_clicked

    @Slot()
    def stop_workflow_clicked(self) -> None:
        token = self._td_cancellation_token
        if not self._background_running or token is None:
            return
        token.cancel()
        self.run_status = "stopping"
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Stopping…")
        self.results_panel.append_console(
            "Stop requested; finishing the current workflow unit."
        )

    stop_timedependent_clicked = stop_workflow_clicked

    @Slot(object)
    def _on_workflow_progress(self, progress: RunProgress) -> None:
        self.last_timedependent_progress = progress
        self.last_run_progress = progress
        self._last_td_progress_thread = QThread.currentThread()
        unit = f" {progress.coordinate_unit}" if progress.coordinate_unit else ""
        if progress.workflow == "static" and progress.latest_field_state is not None:
            self.results_panel.set_run_data(
                from_static_live_state(progress.latest_field_state)
            )
        elif progress.workflow == "timedependent" and progress.latest_field_state is not None:
            if self._active_td_from_static and self._td_static_reference is None:
                state = progress.latest_field_state
                self._td_static_reference = {
                    "intensity_stack": np.asarray(
                        state["initial_intensity_stack"]
                    ).copy(),
                    "theta_stack": np.asarray(state["theta_initial"]).copy(),
                    "theta_bias": np.asarray(state["theta_bias"]).copy(),
                    "output_plane_intensity": np.asarray(
                        state["initial_output_plane_intensity"]
                    ).copy(),
                }
            self.results_panel.set_run_data(
                from_timedependent_live_state(
                    progress.latest_field_state,
                    td_from_static=self._active_td_from_static,
                    static_reference=self._td_static_reference,
                )
            )
        progress_label = "TD" if progress.workflow == "timedependent" else "Static"
        self.results_panel.set_td_time_indicator(
            f"{progress_label}: {progress.coordinate_name} = "
            f"{self._format_coordinate(progress.current_coordinate)}{unit}; "
            f"{progress.completed_units}/{progress.total_units}"
        )
        if progress.workflow == "timedependent":
            self.results_panel.append_console(
                "TD progress: "
                f"segment step {progress.segment_completed_steps}/"
                f"{progress.segment_total_steps}, "
                f"cumulative step {progress.cumulative_completed_steps}, "
                "cumulative t="
                f"{self._format_coordinate(progress.current_coordinate)}, "
                f"elapsed={progress.elapsed_wall_time:.3f} s"
            )
        else:
            self.results_panel.append_console(
                "Static progress: "
                f"slices {progress.completed_units}/{progress.total_units}, "
                f"z={self._format_coordinate(progress.current_coordinate)}{unit}, "
                f"elapsed={progress.elapsed_wall_time:.3f} s"
            )

    _on_timedependent_progress = _on_workflow_progress

    @Slot(object)
    def _on_timedependent_finished(self, runner_result) -> None:
        try:
            result = runner_result.result
            if runner_result.kind == "static":
                self.last_static_result = result
                self.last_static_checkpoint = (
                    result.checkpoint if result.status == "stopped" else None
                )
                self.run_status = (
                    "stopped" if result.status == "stopped" else "completed"
                )
            else:
                self.last_timedependent_result = result
                self.last_timedependent_checkpoint = result.checkpoint
                self.run_status = (
                    "stopped" if result.status == "cancelled" else "completed"
                )
            self._display_runner_result(runner_result)
            if runner_result.kind == "static":
                prefix = "z at stop" if result.status == "stopped" else "Final z"
                self.results_panel.set_td_time_indicator(
                    f"{prefix}: {self._format_coordinate(result.z_reached_um)} um; "
                    f"slices: {result.completed_slices}/{result.total_slices}"
                )
                self.results_panel.append_console(
                    "Run stopped" if result.status == "stopped" else "Run complete"
                )
            elif result.status == "cancelled":
                self.results_panel.set_td_time_indicator(
                    "TD time at stop: "
                    f"{self._format_coordinate(result.checkpoint.current_time)}"
                )
                self.results_panel.append_console("Run cancelled")
            else:
                self.results_panel.set_td_time_indicator(
                    "Final TD time: "
                    f"{self._format_coordinate(result.cumulative_time)}"
                )
                self.results_panel.append_console("Run complete")
        except Exception:
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())
        self._td_outcome_received = True
        self._maybe_finish_timedependent_background()

    @Slot(str)
    def _on_timedependent_failed(self, formatted_traceback: str) -> None:
        self.run_status = "failed"
        self.results_panel.append_console("ERROR")
        self.results_panel.append_console(formatted_traceback)
        self._td_outcome_received = True
        self._maybe_finish_timedependent_background()

    @Slot()
    def _on_timedependent_thread_finished(self) -> None:
        self._td_thread = None
        self._td_worker = None
        self._td_thread_done = True
        self._maybe_finish_timedependent_background()

    def _maybe_finish_timedependent_background(self) -> None:
        if self._td_outcome_received and self._td_thread_done:
            self._finish_timedependent_background()

    def _finish_timedependent_background(self) -> None:
        self._background_running = False
        self._td_cancellation_token = None
        self.stop_button.setText("Stop")
        self._set_background_controls_enabled(True)
        self.update_run_button()

    def _display_runner_result(self, runner_result) -> None:
        result = runner_result.result
        if runner_result.kind == "timedependent" and self._active_td_from_static:
            if self._td_static_reference is None:
                initial_output = getattr(
                    result, "initial_output_plane_intensity", None
                )
                if initial_output is None:
                    raise ValueError(
                        "TD-from-static result lacks its initial output plane"
                    )
                self._td_static_reference = {
                    "intensity_stack": np.asarray(
                        result.initial_intensity_stack
                    ).copy(),
                    "theta_stack": np.asarray(result.theta_initial).copy(),
                    "theta_bias": np.asarray(result.theta_bias).copy(),
                    "output_plane_intensity": np.asarray(initial_output).copy(),
                }
            run_data = to_run_data(
                result,
                td_from_static=True,
                static_reference=self._td_static_reference,
            )
        else:
            run_data = to_run_data(result)
        self.results_panel.set_run_data(run_data)
        self.results_panel.append_console("")
        self.results_panel.append_console(runner_result.message)
        self._append_result_summary(result, runner_result.kind)

    @staticmethod
    def _format_coordinate(value: float) -> str:
        return f"{float(value):.3f}"

    _format_td_time = _format_coordinate


    def _append_result_summary(self, result, fallback_method: str) -> None:
        self.results_panel.append_console(f"method: {getattr(result, 'method', fallback_method)}")

        grid_summary = getattr(result, "grid_summary", None)
        if grid_summary is not None:
            self.results_panel.append_console(
                "Nx, Ny, Nz: "
                f"{grid_summary['Nx']}, "
                f"{grid_summary['Ny']}, "
                f"{grid_summary['Nz']}"
            )

        if hasattr(result, "power_initial"):
            self.results_panel.append_console(
                f"normalized_field_integral_initial: {result.power_initial:.8g}"
            )
        if hasattr(result, "power_final"):
            self.results_panel.append_console(
                f"normalized_field_integral_final:   {result.power_final:.8g}"
            )
        if getattr(result, "physical_power_initial_mW", None) is not None:
            self.results_panel.append_console(
                f"physical_power_initial_mW: {result.physical_power_initial_mW:.8g}"
            )
        if getattr(result, "physical_power_final_mW", None) is not None:
            self.results_panel.append_console(
                f"physical_power_final_mW:   {result.physical_power_final_mW:.8g}"
            )

        if hasattr(result, "cumulative_time"):
            self.results_panel.append_console("TD accounting:")
            self.results_panel.append_console(
                f"  status: {result.status}"
            )
            self.results_panel.append_console(
                "  segment_start_time: "
                f"{self._format_td_time(result.segment_start_time)}"
            )
            self.results_panel.append_console(
                "  segment_elapsed_time: "
                f"{self._format_td_time(result.segment_elapsed_time)}"
            )
            self.results_panel.append_console(
                "  cumulative_time: "
                f"{self._format_td_time(result.cumulative_time)}"
            )
            self.results_panel.append_console(
                f"  prior_completed_steps: {result.prior_completed_steps}"
            )
            self.results_panel.append_console(
                f"  segment_completed_steps: {result.segment_completed_steps}"
            )
            self.results_panel.append_console(
                "  cumulative_completed_steps: "
                f"{result.cumulative_completed_steps}"
            )

        A = getattr(result, "A_final", getattr(result, "A", None))
        theta = getattr(result, "theta_final", getattr(result, "theta", None))
        if A is not None:
            self.results_panel.append_console(f"A shape:       {A.shape}")
        if theta is not None:
            self.results_panel.append_console(f"theta shape:   {theta.shape}")

        slice_summaries = tuple(getattr(result, "slice_summaries", ()) or ())
        if slice_summaries:
            converged_count = sum(item.converged for item in slice_summaries)
            worst_index = int(result.worst_slice_index)
            worst = slice_summaries[worst_index]
            self.results_panel.append_console("static convergence:")
            self.results_panel.append_console(
                f"  converged slices: {converged_count} / {len(slice_summaries)}"
            )
            self.results_panel.append_console(
                f"  max final residual RMS: {result.max_final_residual_rms:.8g}"
            )
            self.results_panel.append_console(
                f"  median final residual RMS: {result.median_final_residual_rms:.8g}"
            )
            self.results_panel.append_console(
                f"  max final residual max: {result.max_final_residual_max:.8g}"
            )
            self.results_panel.append_console(
                f"  worst slice: {worst.z_index} (z={worst.z_um:g} µm)"
            )
            self.results_panel.append_console(
                "  maximum iterations used: "
                f"{max(item.relaxation_iterations for item in slice_summaries)}"
            )
            self.results_panel.append_console(
                f"  maximum theta: {max(item.theta_max for item in slice_summaries):.8g}"
            )

        mode = getattr(result, "mode", None)
        if mode is not None:
            self.results_panel.append_console(f"mode:          {mode}")

        metrics = getattr(result, "metrics", None)
        if metrics:
            self.results_panel.append_console("metrics:")
            for key in ["converged", "beta", "physical_power_mW", "normalized_field_integral", "final_residual_rms", "final_residual_max", "n_points", "converged_count"]:
                if key in metrics:
                    self.results_panel.append_console(f"  {key}: {metrics[key]}")

        warnings = getattr(result, "warnings", None)
        if warnings:
            self.results_panel.append_console("warnings:")
            for w in warnings:
                self.results_panel.append_console(f"  - {w}")
