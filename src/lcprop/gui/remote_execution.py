"""Shared Qt controls for local/remote execution and cluster setup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from lcprop.runners.cluster_connection import ClusterConnectionTester
from lcprop.runners.cluster_profiles import (
    ClusterCatalog,
    ClusterConfigError,
    ClusterProfile,
    compose_ssh_host,
    default_cluster_config_path,
    delete_cluster_profile,
    load_cluster_profiles,
    split_ssh_host,
    select_cluster_profile,
    upsert_cluster_profile,
    write_cluster_profiles,
)
from lcprop.runners.slurm import SlurmResourceProfile
from lcprop.transport.defaults import default_slurm_runner_from_environment
from lcprop.transport.status import RemoteRunState, RemoteRunStatus


@dataclass(frozen=True)
class RemoteExecutionDiscovery:
    catalog: ClusterCatalog
    error: str | None = None


def discover_remote_execution(
    path: str | Path | None = None,
) -> RemoteExecutionDiscovery:
    """Discover optional profiles without allowing them to block Local startup."""
    try:
        return RemoteExecutionDiscovery(load_cluster_profiles(path))
    except ClusterConfigError as exc:
        resolved = Path(path).expanduser() if path else default_cluster_config_path()
        return RemoteExecutionDiscovery(ClusterCatalog(config_path=resolved), str(exc))


def execution_target_selector(*, slurm_available: bool) -> QComboBox:
    selector = QComboBox()
    selector.addItem("Local", "local")
    selector.addItem("Slurm", "slurm")
    item = selector.model().item(1)
    if item is not None:
        item.setEnabled(slurm_available)
    selector.setToolTip(
        "Execution location; independent of the NumPy/CuPy scientific backend."
    )
    return selector


class _ConnectionWorker(QObject):
    finished = Signal(int, object)

    def __init__(self, tester, cluster, resource, generation) -> None:
        super().__init__()
        self.tester, self.cluster, self.resource = tester, cluster, resource
        self.generation = generation
        self.cancelled = Event()

    def cancel(self) -> None:
        self.cancelled.set()

    @Slot()
    def run(self) -> None:
        try:
            result = self.tester.test(
                self.cluster,
                self.resource,
                cancellation_check=self.cancelled.is_set,
            )
        except Exception as exc:
            result = exc
        self.finished.emit(self.generation, result)
        QThread.currentThread().quit()


class RemoteExecutionDialog(QDialog):
    """Material-neutral editor for the existing TOML-backed profile model."""

    catalogSaved = Signal(object)

    def __init__(self, catalog, *, connection_tester=None, parent=None) -> None:
        super().__init__(parent)
        self.catalog = catalog
        self.connection_tester = connection_tester or ClusterConnectionTester()
        self._connection_thread = self._connection_worker = None
        self._connection_generation = 0
        self._connection_result = None
        self._connection_close_pending = False
        self._editing_cluster_name = None
        self._editing_resource_name = None
        self.setWindowTitle("Configure Remote Execution")
        self.setMinimumWidth(620)
        root = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Saved cluster:"))
        self.saved_cluster = QComboBox()
        row.addWidget(self.saved_cluster)
        self.new_cluster_button = QPushButton("New")
        row.addWidget(self.new_cluster_button)
        root.addLayout(row)
        self.saved_cluster.currentIndexChanged.connect(self._load_cluster)
        self.new_cluster_button.clicked.connect(self._new_cluster)

        form = QFormLayout()
        self.profile_name, self.username, self.login_host = (
            QLineEdit(),
            QLineEdit(),
            QLineEdit(),
        )
        self.remote_run_root, self.remote_python, self.source_root = (
            QLineEdit(),
            QLineEdit(),
            QLineEdit(),
        )
        self.poll_interval = QDoubleSpinBox()
        self.poll_interval.setRange(0.1, 3600)
        self.poll_interval.setValue(5)
        self.cleanup_remote_on_success = QCheckBox(
            "Delete remote run artifacts after successful retrieval"
        )
        self.cleanup_remote_on_success.setChecked(True)
        for label, widget in (
            ("Profile name", self.profile_name),
            ("SSH username", self.username),
            ("Login host", self.login_host),
            ("Remote run root", self.remote_run_root),
            ("Remote Python", self.remote_python),
            ("Remote source root", self.source_root),
            ("Polling interval (s)", self.poll_interval),
            ("Successful-run cleanup", self.cleanup_remote_on_success),
        ):
            form.addRow(label, widget)
        root.addLayout(form)

        row = QHBoxLayout()
        row.addWidget(QLabel("Resource profile:"))
        self.saved_resource = QComboBox()
        row.addWidget(self.saved_resource)
        self.new_resource_button = QPushButton("New")
        row.addWidget(self.new_resource_button)
        root.addLayout(row)
        self.saved_resource.currentIndexChanged.connect(self._load_resource)
        self.new_resource_button.clicked.connect(self._new_resource)
        form = QFormLayout()
        self.resource_name, self.partition, self.qos = (
            QLineEdit(),
            QLineEdit(),
            QLineEdit(),
        )
        self.time_limit = QLineEdit("00:15:00")
        self.cpus = QSpinBox()
        self.cpus.setRange(1, 4096)
        self.cpus.setValue(2)
        self.memory_gb = QSpinBox()
        self.memory_gb.setRange(1, 1_000_000)
        self.memory_gb.setValue(8)
        self.gpus = QSpinBox()
        self.gpus.setRange(0, 128)
        self.gres = QLineEdit()
        self.require_cupy = QCheckBox()
        self.minimum_devices = QSpinBox()
        self.minimum_devices.setRange(0, 128)
        self.device_pattern = QLineEdit()
        self.setup_commands = QPlainTextEdit()
        self.setup_commands.setMaximumHeight(72)
        self.setup_commands.setPlaceholderText("Trusted site commands, one per line")
        for label, widget in (
            ("Resource name", self.resource_name),
            ("Partition", self.partition),
            ("QOS", self.qos),
            ("Time limit", self.time_limit),
            ("CPUs", self.cpus),
            ("Memory (GiB)", self.memory_gb),
            ("GPU count", self.gpus),
            ("GRES", self.gres),
            ("Require CuPy", self.require_cupy),
            ("Minimum visible devices (0=unset)", self.minimum_devices),
            ("Expected device pattern", self.device_pattern),
            ("Setup commands", self.setup_commands),
        ):
            form.addRow(label, widget)
        root.addLayout(form)
        note = QLabel(
            "LCProp stores no SSH credentials. System SSH/agent authentication is required."
        )
        note.setWordWrap(True)
        root.addWidget(note)
        self.test_results = QLabel("Not tested")
        self.test_results.setWordWrap(True)
        root.addWidget(self.test_results)
        row = QHBoxLayout()
        self.test_button = QPushButton("Test Connection")
        self.delete_button = QPushButton("Delete Profile")
        self.save_button = QPushButton("Save")
        self.close_button = QPushButton("Close")
        for widget in (self.test_button, self.delete_button):
            row.addWidget(widget)
        row.addStretch(1)
        row.addWidget(self.save_button)
        row.addWidget(self.close_button)
        root.addLayout(row)
        self.test_button.clicked.connect(self.test_connection)
        self.delete_button.clicked.connect(self.delete_profile)
        self.save_button.clicked.connect(self.save_profile)
        self.close_button.clicked.connect(self.close)
        self._configuration_widgets = (
            self.saved_cluster,
            self.new_cluster_button,
            self.profile_name,
            self.username,
            self.login_host,
            self.remote_run_root,
            self.remote_python,
            self.source_root,
            self.poll_interval,
            self.cleanup_remote_on_success,
            self.saved_resource,
            self.new_resource_button,
            self.resource_name,
            self.partition,
            self.qos,
            self.time_limit,
            self.cpus,
            self.memory_gb,
            self.gpus,
            self.gres,
            self.require_cupy,
            self.minimum_devices,
            self.device_pattern,
            self.setup_commands,
            self.delete_button,
            self.save_button,
        )
        self._refresh_clusters()

    def _refresh_clusters(self, selected=None) -> None:
        self.saved_cluster.blockSignals(True)
        self.saved_cluster.clear()
        for cluster in self.catalog.clusters:
            self.saved_cluster.addItem(cluster.name, cluster.name)
        self.saved_cluster.blockSignals(False)
        if self.saved_cluster.count():
            index = self.saved_cluster.findData(
                selected or self.catalog.default_cluster
            )
            self.saved_cluster.setCurrentIndex(max(0, index))
            self._load_cluster()
        else:
            self._new_cluster()

    @Slot()
    def _new_cluster(self) -> None:
        self._editing_cluster_name = None
        self._editing_resource_name = None
        self.saved_cluster.blockSignals(True)
        self.saved_cluster.setCurrentIndex(-1)
        self.saved_cluster.blockSignals(False)
        for widget in (
            self.profile_name,
            self.username,
            self.login_host,
            self.remote_run_root,
            self.remote_python,
            self.source_root,
        ):
            widget.clear()
        self.poll_interval.setValue(5)
        self.cleanup_remote_on_success.setChecked(True)
        self.saved_resource.clear()
        self._new_resource()

    @Slot()
    def _load_cluster(self) -> None:
        name = self.saved_cluster.currentData()
        if not name:
            return
        self._editing_cluster_name = name
        cluster = self.catalog[name]
        username, host = split_ssh_host(cluster.host)
        for widget, value in (
            (self.profile_name, cluster.name),
            (self.username, username),
            (self.login_host, host),
            (self.remote_run_root, cluster.remote_run_root),
            (self.remote_python, cluster.remote_python),
            (self.source_root, cluster.source_root),
        ):
            widget.setText(value)
        self.poll_interval.setValue(cluster.poll_interval)
        self.cleanup_remote_on_success.setChecked(
            cluster.cleanup_remote_on_success
        )
        self.saved_resource.blockSignals(True)
        self.saved_resource.clear()
        for profile in cluster.resource_profiles:
            self.saved_resource.addItem(profile.name, profile.name)
        self.saved_resource.blockSignals(False)
        self.saved_resource.setCurrentIndex(
            max(0, self.saved_resource.findData(cluster.default_resource_profile))
        )
        self._load_resource()

    @Slot()
    def _new_resource(self) -> None:
        self._editing_resource_name = None
        self.saved_resource.blockSignals(True)
        self.saved_resource.setCurrentIndex(-1)
        self.saved_resource.blockSignals(False)
        for widget in (
            self.resource_name,
            self.partition,
            self.qos,
            self.gres,
            self.device_pattern,
        ):
            widget.clear()
        self.time_limit.setText("00:15:00")
        self.cpus.setValue(2)
        self.memory_gb.setValue(8)
        self.gpus.setValue(0)
        self.require_cupy.setChecked(False)
        self.minimum_devices.setValue(0)
        self.setup_commands.clear()

    @Slot()
    def _load_resource(self) -> None:
        cluster_name, profile_name = (
            self.saved_cluster.currentData(),
            self.saved_resource.currentData(),
        )
        if not cluster_name or not profile_name:
            return
        self._editing_resource_name = profile_name
        profile = self.catalog[cluster_name].profile(profile_name)
        for widget, value in (
            (self.resource_name, profile.name),
            (self.partition, profile.partition),
            (self.qos, profile.qos),
            (self.time_limit, profile.time_limit),
            (self.gres, profile.gres or ""),
            (self.device_pattern, profile.expected_device_pattern or ""),
        ):
            widget.setText(value)
        self.cpus.setValue(profile.cpus)
        self.memory_gb.setValue(profile.memory_gb)
        self.gpus.setValue(profile.gpus)
        self.require_cupy.setChecked(profile.require_cupy)
        self.minimum_devices.setValue(profile.minimum_device_count or 0)
        self.setup_commands.setPlainText("\n".join(profile.setup_commands))

    def _form_resource(self) -> SlurmResourceProfile:
        return SlurmResourceProfile(
            self.resource_name.text().strip(),
            self.partition.text().strip(),
            self.qos.text().strip(),
            self.time_limit.text().strip(),
            self.cpus.value(),
            self.memory_gb.value(),
            self.gpus.value(),
            tuple(
                line.strip()
                for line in self.setup_commands.toPlainText().splitlines()
                if line.strip()
            ),
            self.gres.text().strip() or None,
            self.require_cupy.isChecked(),
            self.minimum_devices.value() or None,
            self.device_pattern.text().strip() or None,
        )

    def _form_cluster(self) -> ClusterProfile:
        resource = self._form_resource()
        resources = []
        old_name = self._editing_cluster_name
        if old_name:
            resources = list(self.catalog[old_name].resource_profiles)
        old_resource = self._editing_resource_name
        if old_resource and old_resource != resource.name:
            resources = [value for value in resources if value.name != old_resource]
        for index, old in enumerate(resources):
            if old.name == resource.name:
                resources[index] = resource
                break
        else:
            resources.append(resource)
        return ClusterProfile(
            self.profile_name.text().strip(),
            compose_ssh_host(self.username.text(), self.login_host.text()),
            self.remote_run_root.text().strip(),
            self.remote_python.text().strip(),
            self.source_root.text().strip(),
            tuple(resources),
            self.poll_interval.value(),
            resource.name,
            self.cleanup_remote_on_success.isChecked(),
        )

    @Slot()
    def save_profile(self) -> None:
        try:
            cluster = self._form_cluster()
            old_name = self._editing_cluster_name
            updated = self.catalog
            was_default = old_name == updated.default_cluster
            if old_name and old_name != cluster.name:
                updated = delete_cluster_profile(updated, old_name)
            self.catalog = write_cluster_profiles(
                upsert_cluster_profile(updated, cluster, make_default=was_default)
            )
        except (ClusterConfigError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid remote profile", str(exc))
            return
        self.catalogSaved.emit(self.catalog)
        self._refresh_clusters(cluster.name)
        self.test_results.setText("Profile saved; connection not yet tested.")

    @Slot()
    def delete_profile(self) -> None:
        name = self.saved_cluster.currentData()
        if not name:
            return
        try:
            self.catalog = write_cluster_profiles(
                delete_cluster_profile(self.catalog, name)
            )
        except (ClusterConfigError, KeyError) as exc:
            QMessageBox.warning(self, "Cannot delete profile", str(exc))
            return
        self.catalogSaved.emit(self.catalog)
        self._refresh_clusters()
        self.test_results.setText("Remote profile removed.")

    @Slot()
    def test_connection(self) -> None:
        if self._connection_thread is not None:
            return
        try:
            cluster = self._form_cluster()
        except (ClusterConfigError, ValueError) as exc:
            self.test_results.setText(f"Invalid profile: {exc}")
            return
        self._connection_generation += 1
        generation = self._connection_generation
        self._connection_close_pending = False
        self._set_connection_test_running(True)
        self.test_results.setText("Testing connection…")
        thread = QThread(self)
        worker = _ConnectionWorker(
            self.connection_tester,
            cluster,
            cluster.default_resource_profile,
            generation,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._connection_result_ready)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(
            lambda: self._connection_thread_finished(generation, thread)
        )
        self._connection_thread, self._connection_worker = thread, worker
        thread.start()

    def _set_connection_test_running(self, running: bool) -> None:
        for widget in self._configuration_widgets:
            widget.setEnabled(not running)
        self.test_button.setEnabled(not running)

    @Slot(int, object)
    def _connection_result_ready(self, generation, result) -> None:
        if (
            generation == self._connection_generation
            and not self._connection_close_pending
        ):
            self._connection_result = result

    def _connection_thread_finished(self, generation, thread) -> None:
        thread.wait()
        is_current = (
            generation == self._connection_generation
            and thread is self._connection_thread
        )
        if is_current:
            self._connection_thread = self._connection_worker = None
            if self._connection_close_pending:
                self._connection_close_pending = False
                self._set_connection_test_running(False)
                self.close()
            else:
                self._set_connection_test_running(False)
                result = self._connection_result
                self._connection_result = None
                if isinstance(result, Exception):
                    self.test_results.setText(f"Connection test failed: {result}")
                elif result is not None:
                    self.test_results.setText(
                        "\n".join(
                            f"{'✓' if check.passed else '✗'} "
                            f"{check.name}: {check.detail}"
                            for check in result.checks
                        )
                    )
        thread.deleteLater()

    def shutdown_connection_test(self, timeout_ms: int = 16000) -> bool:
        """Cancel and join the bounded connection worker before destruction."""
        thread = self._connection_thread
        worker = self._connection_worker
        if thread is None:
            return True
        self._connection_generation += 1
        self._connection_result = None
        if worker is not None:
            worker.cancel()
        finished = thread.wait(max(0, timeout_ms))
        if finished:
            try:
                thread.finished.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._connection_thread = self._connection_worker = None
            self._connection_close_pending = False
            self._set_connection_test_running(False)
        return bool(finished)

    def closeEvent(self, event) -> None:
        thread = self._connection_thread
        if thread is not None and thread.isRunning():
            self._connection_close_pending = True
            if self._connection_worker is not None:
                self._connection_worker.cancel()
            self.test_results.setText(
                "Closing after the active connection probe stops…"
            )
            event.ignore()
            return
        if thread is not None:
            thread.wait()
        event.accept()

    def reject(self) -> None:
        """Route Escape through the same worker-safe close policy."""
        self.close()


class RemoteExecutionControls(QWidget):
    selectionChanged = Signal()

    def __init__(
        self, discovery=None, *, config_path=None, runner_factory=None, parent=None
    ) -> None:
        super().__init__(parent)
        self.discovery = discovery or discover_remote_execution(config_path)
        self.runner_factory = runner_factory
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.cluster_selector, self.resource_selector = QComboBox(), QComboBox()
        self.configure_button, self.availability_label = (
            QPushButton("Configure Remote Execution…"),
            QLabel(),
        )
        for label, widget in (
            ("Cluster:", self.cluster_selector),
            ("Resource:", self.resource_selector),
        ):
            layout.addWidget(QLabel(label))
            layout.addWidget(widget)
        layout.addWidget(self.configure_button)
        layout.addWidget(self.availability_label)
        self.cluster_selector.currentIndexChanged.connect(self._cluster_changed)
        self.resource_selector.currentIndexChanged.connect(self.selectionChanged)
        self.configure_button.clicked.connect(self.configure)
        self._dialog = None
        self.refresh(self.discovery)

    @property
    def slurm_available(self):
        return bool(self.discovery.catalog) and self.discovery.error is None

    @property
    def unavailable_reason(self):
        if self.discovery.error:
            return f"Slurm unavailable — invalid cluster config: {self.discovery.error}"
        if not self.discovery.catalog:
            return "Slurm unavailable — no cluster profile configured"
        return ""

    def refresh(self, discovery) -> None:
        old = self.cluster_selector.currentData()
        self.discovery = discovery
        self.cluster_selector.blockSignals(True)
        self.cluster_selector.clear()
        for cluster in discovery.catalog.clusters:
            self.cluster_selector.addItem(cluster.name, cluster.name)
        self.cluster_selector.blockSignals(False)
        if self.cluster_selector.count():
            self.cluster_selector.setCurrentIndex(
                max(
                    0,
                    self.cluster_selector.findData(
                        old or discovery.catalog.default_cluster
                    ),
                )
            )
        self._cluster_changed()
        self.availability_label.setText(
            "Slurm profile configured"
            if self.slurm_available
            else self.unavailable_reason
        )
        self.availability_label.setToolTip(
            self.unavailable_reason or "Connectivity is tested explicitly."
        )
        self.selectionChanged.emit()

    @Slot()
    def _cluster_changed(self) -> None:
        self.resource_selector.blockSignals(True)
        self.resource_selector.clear()
        cluster = self.selected_cluster()
        if cluster:
            for profile in cluster.resource_profiles:
                self.resource_selector.addItem(profile.name, profile.name)
            self.resource_selector.setCurrentIndex(
                max(
                    0, self.resource_selector.findData(cluster.default_resource_profile)
                )
            )
        self.resource_selector.blockSignals(False)
        self.selectionChanged.emit()

    def selected_cluster(self):
        name = self.cluster_selector.currentData()
        return self.discovery.catalog[name] if name else None

    def selected_resource_name(self):
        return self.resource_selector.currentData()

    def runner_kwargs(self) -> dict[str, str]:
        """Return shared runner-selection arguments for a remote dispatch."""
        resource = self.selected_resource_name()
        return {} if resource is None else {"resource_profile": resource}

    def validate_backend(self, backend: str) -> None:
        """Reject an explicit CuPy request on a CPU-only selected resource."""
        cluster = self.selected_cluster()
        resource = self.selected_resource_name()
        if cluster is None or resource is None:
            raise ValueError(self.unavailable_reason or "no remote resource selected")
        profile = cluster.profile(resource)
        if backend == "cupy" and (profile.gpus < 1 or not profile.require_cupy):
            raise ValueError(
                "CuPy scientific backend requires a GPU resource profile with "
                "Require CuPy enabled"
            )

    def create_runner(self):
        cluster = self.selected_cluster()
        if cluster is None:
            return None
        if self.runner_factory is None:
            return default_slurm_runner_from_environment(
                catalog=self.discovery.catalog,
                cluster_name=cluster.name,
            )
        cluster = select_cluster_profile(self.discovery.catalog, cluster.name)
        return self.runner_factory(cluster=cluster)

    @Slot()
    def configure(self) -> None:
        self._dialog = RemoteExecutionDialog(self.discovery.catalog, parent=self)
        self._dialog.catalogSaved.connect(self._catalog_saved)
        self._dialog.show()

    @Slot(object)
    def _catalog_saved(self, catalog) -> None:
        self.refresh(RemoteExecutionDiscovery(catalog))

    def shutdown_connection_test(self, timeout_ms: int = 16000) -> bool:
        if self._dialog is None:
            return True
        return self._dialog.shutdown_connection_test(timeout_ms)


def remote_status_text(status: RemoteRunStatus) -> str:
    parts = ["Runner: Slurm", f"State: {status.state.value.replace('_', ' ').title()}"]
    if status.remote_job_id:
        parts.append(f"Job ID: {status.remote_job_id}")
    parts.append(f"Backend requested: {status.scientific_backend_requested}")
    if status.scientific_backend_resolved != "unresolved":
        parts.append(f"Backend resolved: {status.scientific_backend_resolved}")
    if status.device_summary:
        device = status.device_summary.get("device") or status.device_summary.get(
            "name"
        )
        if device:
            parts.append(f"Device: {device}")
    if status.state == RemoteRunState.RETRIEVING and status.progress_metadata:
        progress = status.progress_metadata
        transferred = progress.get("bytes_transferred")
        total = progress.get("bytes_total")
        percentage = progress.get("percentage")
        rate = progress.get("bytes_per_second")
        current_file = progress.get("current_file")
        if transferred is not None:
            detail = f"Retrieved {int(transferred) / (1024 ** 2):.1f} MiB"
            if total:
                detail += f" / {int(total) / (1024 ** 2):.1f} MiB"
            if percentage is not None:
                detail += f" ({float(percentage):.1f}%)"
            if rate is not None:
                detail += f" at {float(rate) / (1024 ** 2):.2f} MiB/s"
            parts.append(detail)
        if current_file:
            parts.append(f"Current object: {current_file}")
    if status.state.value == "completed":
        if status.remote_cleanup_succeeded:
            parts.append("Remote artifacts cleaned up")
        elif status.remote_cleanup_requested is False:
            parts.append(
                f"Remote artifacts retained at {status.remote_cleanup_target}"
            )
        elif status.remote_cleanup_succeeded is False:
            parts.append(
                "Warning: remote cleanup failed; artifacts retained at "
                f"{status.remote_cleanup_target}"
            )
            if status.remote_cleanup_error:
                parts.append(f"Reason: {status.remote_cleanup_error}")
    return "; ".join(parts)


__all__ = [
    "RemoteExecutionControls",
    "RemoteExecutionDialog",
    "RemoteExecutionDiscovery",
    "discover_remote_execution",
    "execution_target_selector",
    "remote_status_text",
]
