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

from lcprop.lc.products import (
    from_static_live_state,
    from_timedependent_live_state,
    to_run_data,
)
from lcprop.lc import LC_MATERIAL_ID
from lcprop.lc.operations import (
    LC_CONTINUE_STATIC_OPERATION,
    LC_CONTINUE_TIMEDEPENDENT_OPERATION,
    LC_PARAMETER_SWEEP_OPERATION,
    LC_SOLITON_OPERATION,
    LC_STATIC_OPERATION,
    LC_TIMEDEPENDENT_OPERATION,
)
from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.runners.local import LocalRunner
from lcprop.gui.workers import WorkflowWorker
from lcprop.lc.gui.retained_results import (
    ExperimentFamily,
    RetainedResult,
    RetainedResults,
    TDInitialSourceMode,
    WorkflowKind,
    soliton_result_incompatibility,
    source_incompatibility,
)
from lcprop.lc.gui.panels import (
    ExperimentPanel,
    PhysicsPanel,
    SolverPanel,
    SweepPanel,
)
from lcprop.gui.panels.beam_panel import BeamPanel
from lcprop.gui.panels.grid_panel import GridPanel
from lcprop.gui.panels.results_panel import ResultsPanel
from lcprop.lc.requests import (
    OutputOptions,
    ParameterSweepRequest,
    RuntimeOptions,
    SolitonRequest,
    StaticRunRequest,
    TimeDependentRunRequest,
)
from lcprop.lc.workflows import (
    validate_static_continuation,
    validate_timedependent_continuation,
)
from lcprop.lc.workflows.timedependent import timedependent_state_from_static_result

class LCPropMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.runner = LocalRunner(
            operations=(
                LC_STATIC_OPERATION,
                LC_TIMEDEPENDENT_OPERATION,
                LC_SOLITON_OPERATION,
                LC_PARAMETER_SWEEP_OPERATION,
            )
        )
        self.retained_results = RetainedResults()
        self.last_soliton_result = None
        self.last_soliton_existence_result = None
        self.last_timedependent_checkpoint = None
        self.last_timedependent_result = None
        self.last_timedependent_progress = None
        self.last_run_progress = None
        self.last_static_checkpoint = None
        self.last_static_result = None
        self._active_td_from_static = False
        self._active_td_source_mode = TDInitialSourceMode.BEAM_LAUNCH
        self._active_td_soliton_source = None
        self._td_static_reference = None
        self._last_td_progress_thread = None
        self._background_running = False
        self._active_workflow = None
        self._active_request = None
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

        self.td_initial_condition_label = QLabel("Initial condition:")
        self.td_initial_condition_label.setVisible(False)
        header.addWidget(self.td_initial_condition_label)
        self.td_initial_condition_selector = QComboBox()
        self.td_initial_condition_selector.addItem(
            "Beam pane launch", TDInitialSourceMode.BEAM_LAUNCH
        )
        self.td_initial_condition_selector.addItem(
            "Last standard static result",
            TDInitialSourceMode.STANDARD_STATIC,
        )
        self.td_initial_condition_selector.addItem(
            "Selected soliton result",
            TDInitialSourceMode.SOLITON_RESULT,
        )
        self.td_initial_condition_selector.setVisible(False)
        self.td_initial_condition_selector.currentIndexChanged.connect(
            self._td_initial_source_mode_changed
        )
        header.addWidget(self.td_initial_condition_selector)

        # Hidden compatibility shims for older callers. The combo box above is
        # the sole authoritative source-mode control.
        self.use_last_soliton = QCheckBox("Start TD from a soliton result")
        self.use_last_soliton.setEnabled(False)
        self.use_last_soliton.setVisible(False)
        self.soliton_source_label = QLabel("Soliton source:")
        self.soliton_source_label.setVisible(False)
        header.addWidget(self.soliton_source_label)
        self.soliton_source_selector = QComboBox()
        self.soliton_source_selector.setMinimumContentsLength(24)
        self.soliton_source_selector.setVisible(False)
        self.soliton_source_selector.currentIndexChanged.connect(
            self._soliton_source_changed
        )
        header.addWidget(self.soliton_source_selector)

        self.use_last_static = QCheckBox("Start TD from last static result")
        self.use_last_static.setEnabled(False)
        self.use_last_static.setVisible(False)

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
        self.sweep_panel = SweepPanel(
            parallel_available=bool(
                getattr(self.runner, "supports_parallel_sweeps", False)
            )
        )
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
        self.use_last_soliton.toggled.connect(
            self._legacy_soliton_source_toggled
        )
        self.use_last_static.toggled.connect(
            self._legacy_static_source_toggled
        )
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
        if experiment == "Time-dependent propagation":
            available, reason = self._td_source_mode_availability()
            self.run_button.setEnabled(available)
            self.run_button.setToolTip("" if available else reason)
        else:
            self.run_button.setEnabled(True)
            self.run_button.setToolTip("")

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

    def _td_checkpoint_matches_current_request(self, checkpoint) -> bool:
        try:
            validate_timedependent_continuation(
                self.build_timedependent_request(), checkpoint
            )
        except Exception:
            return False
        return True

    def _configuration_changed(self, *_args) -> None:
        checkpoint = self.last_timedependent_checkpoint
        if self._background_running:
            return
        if checkpoint is not None:
            try:
                request = self.build_timedependent_request()
                validate_timedependent_continuation(request, checkpoint)
            except Exception as exc:
                self.last_timedependent_checkpoint = None
                self.retained_results.last_standard_td_checkpoint = None
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
        self._adopt_legacy_retained_values()
        is_td = experiment == "Time-dependent propagation"
        td_request = self._build_timedependent_base_request() if is_td else None
        self._refresh_soliton_source_selector()

        source = self.retained_results.selected_soliton_source
        soliton_reason = "Run or select a completed soliton first."
        if is_td and source is not None:
            mismatch = source_incompatibility(td_request, source.request)
            if mismatch is None:
                mismatch = soliton_result_incompatibility(
                    td_request, source.result
                )
            valid_partial = bool(
                getattr(
                    source.result,
                    "usable_as_initial_condition",
                    getattr(source.result, "status", "completed")
                    == "completed",
                )
            )
            if mismatch is not None:
                soliton_reason = (
                    f"Selected soliton is incompatible with current {mismatch}."
                )
            elif not valid_partial:
                soliton_reason = "Last stopped soliton has no accepted candidate."
            else:
                soliton_reason = ""

        soliton_enabled = is_td and source is not None and not soliton_reason
        has_sources = self.soliton_source_selector.count() > 0
        self.td_initial_condition_label.setVisible(is_td)
        self.td_initial_condition_selector.setVisible(is_td)
        self.soliton_source_label.setVisible(is_td)
        self.soliton_source_selector.setVisible(is_td)
        self.soliton_source_selector.setEnabled(
            is_td
            and has_sources
            and self.td_initial_source_mode()
            is TDInitialSourceMode.SOLITON_RESULT
        )
        self.soliton_source_selector.setToolTip(
            "Choose a retained single soliton or a completed member of the "
            "latest existence sweep."
            if has_sources
            else "No retained soliton sources are available."
        )

        static_reason = "Run a completed standard static propagation first."
        static_entry = self.retained_results.last_standard_static_result
        if is_td and static_entry is not None:
            try:
                timedependent_state_from_static_result(static_entry.result, td_request)
            except Exception as exc:
                static_reason = str(exc)
            else:
                static_reason = ""
        static_enabled = is_td and static_entry is not None and not static_reason
        self._set_td_source_item_enabled(
            TDInitialSourceMode.BEAM_LAUNCH,
            True,
            "Initialize from the current Beam pane and normal bias state.",
        )
        self._set_td_source_item_enabled(
            TDInitialSourceMode.STANDARD_STATIC,
            static_enabled,
            static_reason
            or "Initialize from the retained standard static result.",
        )
        self._set_td_source_item_enabled(
            TDInitialSourceMode.SOLITON_RESULT,
            soliton_enabled,
            soliton_reason
            or "Initialize from the selected retained soliton.",
        )
        self._sync_legacy_source_checkboxes()

    def _adopt_legacy_retained_values(self) -> None:
        """Keep direct attribute assignment compatible with older callers/tests."""

        if (
            self.last_static_result is not None
            and self.retained_results.last_standard_static_result is None
        ):
            request = getattr(self.last_static_result, "request", None)
            if request is not None:
                self.retained_results.store(
                    family=ExperimentFamily.STANDARD,
                    workflow=WorkflowKind.STATIC,
                    request=request,
                    result=self.last_static_result,
                )
        if (
            self.last_soliton_result is not None
            and self.retained_results.last_soliton_result is None
        ):
            request = getattr(self.last_soliton_result, "request", None)
            if request is not None:
                self.retained_results.store(
                    family=ExperimentFamily.SOLITON,
                    workflow=WorkflowKind.SOLITON,
                    request=request,
                    result=self.last_soliton_result,
                )

    def td_initial_source_mode(self) -> TDInitialSourceMode:
        mode = self.td_initial_condition_selector.currentData()
        return (
            TDInitialSourceMode.BEAM_LAUNCH
            if mode is None
            else TDInitialSourceMode(mode)
        )

    def _set_td_initial_source_mode(self, mode: TDInitialSourceMode) -> None:
        index = self.td_initial_condition_selector.findData(mode)
        if index >= 0:
            self.td_initial_condition_selector.setCurrentIndex(index)

    def _set_td_source_item_enabled(
        self,
        mode: TDInitialSourceMode,
        enabled: bool,
        tooltip: str,
    ) -> None:
        index = self.td_initial_condition_selector.findData(mode)
        item = self.td_initial_condition_selector.model().item(index)
        item.setEnabled(enabled)
        item.setToolTip(tooltip)

    def _td_source_mode_availability(self) -> tuple[bool, str]:
        mode = self.td_initial_source_mode()
        index = self.td_initial_condition_selector.findData(mode)
        item = self.td_initial_condition_selector.model().item(index)
        return bool(item.isEnabled()), str(item.toolTip() or "")

    def _td_initial_source_mode_changed(self, *_args) -> None:
        self._sync_legacy_source_checkboxes()
        self._configuration_changed()

    def _sync_legacy_source_checkboxes(self) -> None:
        mode = self.td_initial_source_mode()
        for checkbox, checked in (
            (
                self.use_last_soliton,
                mode is TDInitialSourceMode.SOLITON_RESULT,
            ),
            (
                self.use_last_static,
                mode is TDInitialSourceMode.STANDARD_STATIC,
            ),
        ):
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)
        self.use_last_soliton.setEnabled(
            self.td_initial_condition_selector.model()
            .item(
                self.td_initial_condition_selector.findData(
                    TDInitialSourceMode.SOLITON_RESULT
                )
            )
            .isEnabled()
        )
        self.use_last_soliton.setToolTip(
            self.td_initial_condition_selector.model()
            .item(
                self.td_initial_condition_selector.findData(
                    TDInitialSourceMode.SOLITON_RESULT
                )
            )
            .toolTip()
        )
        self.use_last_static.setEnabled(
            self.td_initial_condition_selector.model()
            .item(
                self.td_initial_condition_selector.findData(
                    TDInitialSourceMode.STANDARD_STATIC
                )
            )
            .isEnabled()
        )
        self.use_last_static.setToolTip(
            self.td_initial_condition_selector.model()
            .item(
                self.td_initial_condition_selector.findData(
                    TDInitialSourceMode.STANDARD_STATIC
                )
            )
            .toolTip()
        )

    def _legacy_soliton_source_toggled(self, checked: bool) -> None:
        if checked:
            self._set_td_initial_source_mode(
                TDInitialSourceMode.SOLITON_RESULT
            )
        elif self.td_initial_source_mode() is TDInitialSourceMode.SOLITON_RESULT:
            self._set_td_initial_source_mode(TDInitialSourceMode.BEAM_LAUNCH)

    def _legacy_static_source_toggled(self, checked: bool) -> None:
        if checked:
            self._set_td_initial_source_mode(
                TDInitialSourceMode.STANDARD_STATIC
            )
        elif self.td_initial_source_mode() is TDInitialSourceMode.STANDARD_STATIC:
            self._set_td_initial_source_mode(TDInitialSourceMode.BEAM_LAUNCH)

    def _refresh_soliton_source_selector(self) -> None:
        sources = self.retained_results.soliton_sources()
        selected = self.retained_results.selected_soliton_source
        selected_id = None if selected is None else selected.source_id
        self.soliton_source_selector.blockSignals(True)
        self.soliton_source_selector.clear()
        for source in sources:
            self.soliton_source_selector.addItem(source.label, source)
        selected_index = -1
        for index in range(self.soliton_source_selector.count()):
            source = self.soliton_source_selector.itemData(index)
            if source.source_id == selected_id:
                selected_index = index
                break
        if selected_index < 0 and sources and selected_id is None:
            selected_index = 0
        self.soliton_source_selector.setCurrentIndex(selected_index)
        self.soliton_source_selector.blockSignals(False)
        self.retained_results.selected_soliton_source = (
            None
            if selected_index < 0
            else self.soliton_source_selector.itemData(selected_index)
        )

    def _soliton_source_changed(self, index: int) -> None:
        self.retained_results.selected_soliton_source = (
            None
            if index < 0
            else self.soliton_source_selector.itemData(index)
        )
        self._configuration_changed()

    def build_request(self, *, allow_last_soliton: bool = True) -> StaticRunRequest:
        return StaticRunRequest(
            grid=self.grid_panel.grid(),
            material=self.physics_panel.material(),
            bias=self.physics_panel.bias(),
            beams=self.beam_panel.beams(),
            solver=self.solver_panel.solver(),
            output=OutputOptions(),
        )

    def _build_timedependent_base_request(self) -> TimeDependentRunRequest:
        return TimeDependentRunRequest(
            grid=self.grid_panel.grid(),
            material=self.physics_panel.material(),
            bias=self.physics_panel.bias(),
            beams=self.beam_panel.beams(),
            solver=self.solver_panel.td_solver(),
            output=OutputOptions(),
            runtime=RuntimeOptions(precision="float64"),
        )

    def build_timedependent_request(self) -> TimeDependentRunRequest:
        req = self._build_timedependent_base_request()
        mode = self.td_initial_source_mode()
        if mode is TDInitialSourceMode.BEAM_LAUNCH:
            return req
        if mode is TDInitialSourceMode.SOLITON_RESULT:
            source = self.retained_results.selected_soliton_source
            if source is None:
                raise ValueError("no compatible soliton result is available")
            mismatch = source_incompatibility(req, source.request)
            if mismatch is None:
                mismatch = soliton_result_incompatibility(req, source.result)
            if mismatch is not None:
                raise ValueError(
                    f"cannot initialize TD from soliton: incompatible {mismatch}"
                )
            req = replace(
                req,
                beams=self._base_static_request(source.request).beams,
                initial_A=source.result.A,
                initial_theta=source.result.theta,
            )
        elif mode is TDInitialSourceMode.STANDARD_STATIC:
            retained = self.retained_results.last_standard_static_result
            if retained is None:
                raise ValueError("no completed static result is available")
            req = timedependent_state_from_static_result(
                retained.result, req
            )
        else:
            raise ValueError(f"unsupported TD initial-condition mode: {mode}")
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
            max_workers=self.sweep_panel.sweep_worker_count(),
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
            "Initial condition: Beam pane launch",
        ]

        mode = self.td_initial_source_mode()
        if (
            isinstance(req, TimeDependentRunRequest)
            and mode is TDInitialSourceMode.STANDARD_STATIC
        ):
            source = self.retained_results.last_standard_static_result.result
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
        elif (
            isinstance(req, TimeDependentRunRequest)
            and mode is TDInitialSourceMode.SOLITON_RESULT
        ):
            source = self.retained_results.selected_soliton_source
            lines[-1] = f"Initial condition: {source.label}"
            lines.append(f"Soliton source ID: {source.source_id}")
            if source.sweep_id is not None:
                lines.append(f"Source sweep ID: {source.sweep_id}")
                lines.append(
                    f"Source sweep power: {source.requested_power_mW:g} mW"
                )

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
            lines.append(
                f"Configured sweep workers: {getattr(req, 'max_workers', None)}"
            )
        else:
            powers = getattr(req, "powers_mW", None)
            if powers is not None:
                lines.append("Powers: " + ", ".join(f"{p:g} mW" for p in powers))

        return "\n".join(lines)

    def _experiment_dispatch(self):
        return {
            "Static propagation": (
                self.build_request,
                LC_STATIC_OPERATION,
                "static workflow",
            ),
            "Time-dependent propagation": (
                self.build_timedependent_request,
                LC_TIMEDEPENDENT_OPERATION,
                "time-dependent workflow",
            ),
            "Soliton": (
                self.build_soliton_request,
                LC_SOLITON_OPERATION,
                "soliton workflow",
            ),
            "Soliton existence curve": (
                self.build_soliton_existence_request,
                LC_PARAMETER_SWEEP_OPERATION,
                "soliton existence workflow",
            ),
        }

    def _run_registered(self, operation, request, **kwargs):
        """Dispatch one canonical LC operation through the shared runner."""

        runner_result = self.runner.run_registered(
            LC_MATERIAL_ID,
            operation.workflow_id,
            request,
            **kwargs,
        )
        if (
            operation is LC_SOLITON_OPERATION
            and request.refine_transverse
            and runner_result.result.status != "stopped"
        ):
            return replace(
                runner_result,
                message="Completed locally with transverse refinement",
            )
        return runner_result

    def run_static_clicked(self):
        if self._background_running:
            return
        experiment = self.experiment_panel.current_experiment()
        if experiment == "Time-dependent propagation":
            self._start_timedependent_background()
            return
        if experiment == "Static propagation":
            self._start_static_background()
            return
        try:
            request_builder, operation, _run_label = (
                self._experiment_dispatch()[experiment]
            )
            req = request_builder()
        except Exception:
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())
            return
        workflow = (
            "soliton"
            if experiment == "Soliton"
            else "soliton_existence"
        )
        self._start_timedependent_background(
            request=req,
            runner_callable=partial(self._run_registered, operation),
            workflow=workflow,
        )

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
        self.td_initial_condition_selector.setEnabled(enabled)
        self.use_last_soliton.setEnabled(
            enabled and self.retained_results.selected_soliton_source is not None
        )
        self.soliton_source_selector.setEnabled(
            enabled
            and self.soliton_source_selector.isVisible()
            and self.soliton_source_selector.count() > 0
            and self.td_initial_source_mode()
            is TDInitialSourceMode.SOLITON_RESULT
        )
        self.use_last_static.setEnabled(
            enabled
            and self.experiment_panel.current_experiment()
            == "Time-dependent propagation"
            and self.retained_results.last_standard_static_result is not None
            and self.retained_results.last_standard_static_result.result.status
            == "completed"
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
            req = (
                self.build_timedependent_request()
                if request is None
                else request
            )
            summary = self.describe_request(req)
        except Exception:
            self.results_panel.append_console("ERROR")
            self.results_panel.append_console(traceback.format_exc())
            return

        old_checkpoint = (
            self.last_timedependent_checkpoint
            if workflow == "timedependent"
            else self.last_static_checkpoint if workflow == "static" else None
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
            mode = self.td_initial_source_mode()
            self._active_td_source_mode = mode
            self._active_td_from_static = (
                mode is TDInitialSourceMode.STANDARD_STATIC
            )
            self._active_td_soliton_source = (
                self.retained_results.selected_soliton_source
                if mode is TDInitialSourceMode.SOLITON_RESULT
                else None
            )
            self._td_static_reference = None

        if not is_continuation:
            self.results_panel.reset_field_color_scales()

        self._background_running = True
        self._active_workflow = workflow
        self._active_request = req
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
        if workflow == "timedependent" and self._active_td_soliton_source is not None:
            source = self._active_td_soliton_source
            self.results_panel.append_console(
                f"TD soliton source: {source.label} [{source.source_id}]"
            )
        coordinate = {
            "timedependent": "TD time",
            "static": "Static z",
            "soliton": "Soliton iteration",
            "soliton_existence": "Completed powers",
        }[workflow]
        unit = " um" if workflow == "static" else ""
        self.results_panel.set_td_time_indicator(
            f"{coordinate}: {self._format_coordinate(cumulative_start_time)}{unit}"
        )

        token = CancellationToken()
        thread = QThread(self)
        worker = WorkflowWorker(
            partial(
                self._run_registered,
                LC_TIMEDEPENDENT_OPERATION
                if workflow == "timedependent"
                else LC_STATIC_OPERATION,
            )
            if runner_callable is None
            else runner_callable,
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
        validate_timedependent_continuation(request, checkpoint)
        continuation_runner = partial(
            self.runner.run_operation,
            LC_CONTINUE_TIMEDEPENDENT_OPERATION,
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
                validate_timedependent_continuation(request, checkpoint)
            else:
                validate_static_continuation(request, checkpoint)
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
                self.runner.run_operation,
                LC_CONTINUE_STATIC_OPERATION,
                checkpoint=checkpoint,
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
        elif progress.workflow in {"soliton", "soliton_existence"}:
            if progress.latest_field_state is not None:
                self.results_panel.set_run_data(
                    to_run_data(progress.latest_field_state)
                )
        progress_label = {
            "timedependent": "TD",
            "static": "Static",
            "soliton": "Soliton",
            "soliton_existence": "Existence sweep",
        }.get(progress.workflow, progress.workflow)
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
        elif progress.workflow == "static":
            self.results_panel.append_console(
                "Static progress: "
                f"slices {progress.completed_units}/{progress.total_units}, "
                f"z={self._format_coordinate(progress.current_coordinate)}{unit}, "
                f"elapsed={progress.elapsed_wall_time:.3f} s"
            )
        elif progress.workflow == "soliton":
            diagnostics = progress.diagnostics or {}
            beta = diagnostics.get("beta")
            beta_text = "" if beta is None else f", beta={beta:.6g}"
            self.results_panel.set_td_time_indicator(
                "Soliton: iteration "
                f"{progress.completed_units}/{progress.total_units}; "
                f"residual RMS={diagnostics.get('residual_rms', float('nan')):.3g}; "
                f"residual max={diagnostics.get('residual_max', float('nan')):.3g}"
                f"{beta_text}"
            )
        elif progress.workflow == "soliton_existence":
            diagnostics = progress.diagnostics or {}
            status = (
                "converged" if diagnostics.get("converged") else "not converged"
            )
            self.results_panel.set_td_time_indicator(
                "Existence sweep: "
                f"{progress.completed_units}/{progress.total_units} powers; "
                f"current={diagnostics.get('current_power_mW', float('nan')):g} mW; "
                f"{status}"
            )

    _on_timedependent_progress = _on_workflow_progress

    @Slot(object)
    def _on_timedependent_finished(self, runner_result) -> None:
        try:
            result = runner_result.result
            if runner_result.kind == "timedependent":
                provenance = {
                    "initial_source_kind": self._active_td_source_mode.value,
                    "source_result_id": None,
                    "source_sweep_id": None,
                    "source_member_index": None,
                    "source_power_mW": None,
                    "source_beta": None,
                }
                if self._active_td_soliton_source is not None:
                    source = self._active_td_soliton_source
                    provenance.update({
                        "initial_source_kind": source.source_kind.value,
                        "source_result_id": source.result_id,
                        "source_sweep_id": source.sweep_id,
                        "source_member_index": source.requested_index,
                        "source_power_mW": source.requested_power_mW,
                        "source_beta": source.beta,
                    })
                elif (
                    self._active_td_source_mode
                    is TDInitialSourceMode.STANDARD_STATIC
                    and self.retained_results.last_standard_static_result
                    is not None
                ):
                    provenance["source_result_id"] = (
                        self.retained_results.last_standard_static_result.retained_id
                    )
                result = replace(
                    result,
                    provenance=provenance,
                )
                runner_result = replace(runner_result, result=result)
            if runner_result.kind == "static":
                self.last_static_result = result
                if result.status == "completed":
                    self.retained_results.store(
                        family=ExperimentFamily.STANDARD,
                        workflow=WorkflowKind.STATIC,
                        request=self._active_request,
                        result=result,
                    )
                self.last_static_checkpoint = (
                    result.checkpoint if result.status == "stopped" else None
                )
                self.retained_results.last_standard_static_checkpoint = (
                    None
                    if self.last_static_checkpoint is None
                    else RetainedResult(
                        ExperimentFamily.STANDARD,
                        WorkflowKind.STATIC,
                        self._active_request,
                        self.last_static_checkpoint,
                    )
                )
                self.run_status = (
                    "stopped" if result.status == "stopped" else "completed"
                )
            elif runner_result.kind == "timedependent":
                self.last_timedependent_result = result
                td_retained = self.retained_results.store(
                    family=ExperimentFamily.STANDARD,
                    workflow=WorkflowKind.TIMEDEPENDENT,
                    request=self._active_request,
                    result=result,
                )
                self.last_timedependent_checkpoint = result.checkpoint
                self.retained_results.last_standard_td_checkpoint = td_retained
                self.run_status = (
                    "stopped" if result.status == "cancelled" else "completed"
                )
            elif runner_result.kind == "soliton":
                valid = (
                    result.status == "completed"
                    or result.completed_iterations > 0
                )
                if valid:
                    self.last_soliton_result = result
                    self.retained_results.store(
                        family=ExperimentFamily.SOLITON,
                        workflow=WorkflowKind.SOLITON,
                        request=self._active_request,
                        result=result,
                    )
                self.run_status = (
                    "stopped" if result.status == "stopped" else "completed"
                )
            elif runner_result.kind in {
                "parameter_sweep", "soliton_existence"
            }:
                self.last_soliton_existence_result = result
                self.retained_results.store(
                    family=ExperimentFamily.SOLITON,
                    workflow=WorkflowKind.SOLITON_EXISTENCE,
                    request=self._active_request,
                    result=result,
                )
                self.run_status = (
                    "stopped" if result.status == "stopped" else "completed"
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
            elif runner_result.kind == "timedependent" and result.status == "cancelled":
                self.results_panel.set_td_time_indicator(
                    "TD time at stop: "
                    f"{self._format_coordinate(result.checkpoint.current_time)}"
                )
                self.results_panel.append_console("Run cancelled")
            elif runner_result.kind == "timedependent":
                self.results_panel.set_td_time_indicator(
                    "Final TD time: "
                    f"{self._format_coordinate(result.cumulative_time)}"
                )
                self.results_panel.append_console("Run complete")
            elif runner_result.kind == "soliton":
                label = (
                    "Soliton result at Stop"
                    if result.status == "stopped"
                    else "Completed soliton result"
                )
                self.results_panel.set_td_time_indicator(label)
                self.results_panel.append_console(label)
            elif runner_result.kind in {
                "parameter_sweep", "soliton_existence"
            }:
                prefix = (
                    "Stopped soliton existence sweep"
                    if result.status == "stopped"
                    else "Completed soliton existence sweep"
                )
                label = (
                    f"{prefix}: {result.completed_points}/{result.total_points}"
                )
                failed_count = int(result.metrics.get("failed_count", 0))
                if failed_count:
                    label += f"; failed members: {failed_count}"
                self.results_panel.set_td_time_indicator(label)
                self.results_panel.append_console(label)
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
        self._active_workflow = None
        self._active_request = None
        self._active_td_source_mode = TDInitialSourceMode.BEAM_LAUNCH
        self._active_td_soliton_source = None
        self.stop_button.setText("Stop")
        self._set_background_controls_enabled(True)
        self.update_run_button()

    def shutdown_background_run(self, timeout_ms: int = 30000) -> bool:
        """Cooperatively stop and join the centralized workflow thread."""

        thread = self._td_thread
        if thread is None or not thread.isRunning():
            return True
        if self._td_cancellation_token is not None:
            self._td_cancellation_token.cancel()
        self.run_status = "stopping"
        thread.quit()
        finished = thread.wait(timeout_ms)
        if finished:
            self._td_thread = None
            self._td_worker = None
            self._background_running = False
        return bool(finished)

    def closeEvent(self, event) -> None:
        if self.shutdown_background_run():
            event.accept()
        else:
            self.results_panel.append_console(
                "Close delayed: workflow did not stop within the shutdown timeout."
            )
            event.ignore()

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
            run_data = (
                runner_result.run_data
                if runner_result.run_data is not None
                else to_run_data(result)
            )
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
