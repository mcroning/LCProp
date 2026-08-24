from __future__ import annotations

import inspect
import os
from pathlib import Path
import shlex
import subprocess
import sys
from threading import Event, Timer
from time import monotonic

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from lcprop.gui.remote_execution import (
    RemoteExecutionControls,
    RemoteExecutionDialog,
    RemoteExecutionDiscovery,
    discover_remote_execution,
)
from lcprop.lc.gui.main_window import LCPropMainWindow
from lcprop.pr.gui.main_window import PRMainWindow
from lcprop.runners.cluster_connection import ClusterConnectionTester
from lcprop.runners.cluster_connection import (
    ClusterConnectionResult,
    ConnectionCheck,
    ConnectionTestRemoteTransport,
)
from lcprop.runners.cluster_profiles import (
    ClusterCatalog,
    ClusterConfigError,
    ClusterProfile,
    compose_ssh_host,
    delete_cluster_profile,
    load_cluster_profiles,
    split_ssh_host,
    upsert_cluster_profile,
    write_cluster_profiles,
)
from lcprop.runners.slurm import SlurmResourceProfile


def _app():
    return QApplication.instance() or QApplication([])


def _wait_for_connection_thread(dialog, timeout_ms=2000):
    thread = dialog._connection_thread
    if thread is None:
        return
    loop = QEventLoop()
    QTimer.singleShot(timeout_ms, loop.quit)
    thread.finished.connect(loop.quit)
    loop.exec()
    QApplication.processEvents()


CPU = SlurmResourceProfile("CPU", "batch", "normal", "00:10:00", 2, 8)
GPU = SlurmResourceProfile(
    "GPU",
    "gpu",
    "normal",
    "00:20:00",
    2,
    16,
    gpus=1,
    gres="gpu:1",
    require_cupy=True,
    minimum_device_count=1,
)


def _cluster(name="alpha", resources=(CPU, GPU)):
    return ClusterProfile(
        name=name,
        host=f"user@login.{name}.edu",
        remote_run_root=f"/scratch/{name}/runs",
        remote_python=f"/scratch/{name}/env/bin/python",
        source_root=f"/scratch/{name}/sources",
        resource_profiles=resources,
        default_resource_profile=resources[0].name,
    )


class RecordingTransport:
    def __init__(self, fail_on=()):
        self.calls = []
        self.bounded_calls = []
        self.fail_on = tuple(fail_on)

    def ssh(self, host, *args):
        self.calls.append((host, args))
        joined = " ".join(args)
        if any(value in joined for value in self.fail_on):
            raise subprocess.CalledProcessError(
                255,
                ["ssh"],
                stderr="Permission denied" if "true" in joined else "missing",
            )
        if args[:2] == ("command", "-v"):
            return f"/usr/bin/{args[2]}"
        if args[-1:] == ("--version",):
            return "Python 3.12.1"
        if "import cupy" in joined:
            return "13.6.0"
        return ""

    def ssh_with_timeout(
        self,
        host,
        *args,
        timeout_seconds,
        cancellation_check,
    ):
        self.bounded_calls.append((host, args, timeout_seconds))
        if cancellation_check():
            raise RuntimeError("connection test cancelled")
        return self.ssh(host, *args)


def test_username_and_login_host_compose_without_credentials():
    assert (
        compose_ssh_host("scientist", "login.example.edu")
        == "scientist@login.example.edu"
    )
    assert split_ssh_host("scientist@login.example.edu") == (
        "scientist",
        "login.example.edu",
    )
    assert compose_ssh_host("", "login.example.edu") == "login.example.edu"
    with pytest.raises(ClusterConfigError):
        compose_ssh_host("bad;user", "login.example.edu")


def test_catalog_save_reload_edit_delete_and_other_profile_preservation(tmp_path):
    path = tmp_path / "clusters.toml"
    alpha, beta = _cluster("alpha"), _cluster("beta", (CPU,))
    catalog = ClusterCatalog((alpha, beta), "alpha", path)
    saved = write_cluster_profiles(catalog)
    assert load_cluster_profiles(path) == saved
    edited = _cluster("alpha", (GPU, CPU))
    saved = write_cluster_profiles(upsert_cluster_profile(saved, edited))
    assert saved["beta"] == beta
    assert tuple(p.name for p in saved["alpha"].resource_profiles) == ("GPU", "CPU")
    saved = write_cluster_profiles(delete_cluster_profile(saved, "alpha"))
    assert tuple(c.name for c in saved.clusters) == ("beta",)
    assert saved.default_cluster == "beta"


def test_atomic_write_failure_preserves_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "clusters.toml"
    catalog = write_cluster_profiles(ClusterCatalog((_cluster(),), "alpha", path))
    original = path.read_bytes()

    def fail_replace(_source, _target):
        raise OSError("synthetic replacement failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(ClusterConfigError, match="cannot save"):
        write_cluster_profiles(catalog)
    assert path.read_bytes() == original


def test_gui_fields_round_trip_through_existing_toml_model(tmp_path):
    _app()
    path = tmp_path / "clusters.toml"
    catalog = write_cluster_profiles(
        ClusterCatalog((_cluster("alpha"), _cluster("beta", (CPU,))), "alpha", path)
    )
    dialog = RemoteExecutionDialog(catalog)
    dialog.remote_run_root.setText("/scratch/alpha/new-runs")
    dialog.poll_interval.setValue(7.5)
    dialog.save_profile()
    reloaded = load_cluster_profiles(path)
    assert reloaded["alpha"].remote_run_root == "/scratch/alpha/new-runs"
    assert reloaded["alpha"].poll_interval == 7.5
    assert reloaded["beta"] == catalog["beta"]
    dialog.close_button.click()


def test_gui_new_cluster_adds_without_replacing_selected_cluster(tmp_path):
    _app()
    path = tmp_path / "clusters.toml"
    catalog = write_cluster_profiles(
        ClusterCatalog((_cluster("alpha"), _cluster("beta", (CPU,))), "alpha", path)
    )
    dialog = RemoteExecutionDialog(catalog)
    dialog.new_cluster_button.click()
    assert dialog.saved_cluster.currentData() is None
    dialog.profile_name.setText("gamma")
    dialog.username.setText("newuser")
    dialog.login_host.setText("login.gamma.edu")
    dialog.remote_run_root.setText("/scratch/gamma/runs")
    dialog.remote_python.setText("/scratch/gamma/env/bin/python")
    dialog.source_root.setText("/scratch/gamma/sources")
    dialog.resource_name.setText("CPU new")
    dialog.partition.setText("batch")
    dialog.qos.setText("normal")
    dialog.save_profile()
    reloaded = load_cluster_profiles(path)
    assert tuple(cluster.name for cluster in reloaded.clusters) == (
        "alpha",
        "beta",
        "gamma",
    )
    assert reloaded["alpha"] == catalog["alpha"]
    dialog.close()


def test_gui_new_resource_adds_without_replacing_selected_resource(tmp_path):
    _app()
    path = tmp_path / "clusters.toml"
    catalog = write_cluster_profiles(ClusterCatalog((_cluster(),), "alpha", path))
    dialog = RemoteExecutionDialog(catalog)
    assert dialog.saved_resource.currentData() == "CPU"
    dialog.new_resource_button.click()
    assert dialog.saved_resource.currentData() is None
    dialog.resource_name.setText("CPU large")
    dialog.partition.setText("batch")
    dialog.qos.setText("normal")
    dialog.time_limit.setText("00:30:00")
    dialog.cpus.setValue(4)
    dialog.memory_gb.setValue(32)
    dialog.save_profile()
    reloaded = load_cluster_profiles(path)
    assert tuple(profile.name for profile in reloaded["alpha"].resource_profiles) == (
        "CPU",
        "GPU",
        "CPU large",
    )
    assert reloaded["alpha"].profile("CPU") == CPU
    dialog.close()


def test_no_profile_and_invalid_profile_leave_local_available(tmp_path):
    _app()
    empty = RemoteExecutionControls(
        RemoteExecutionDiscovery(ClusterCatalog(config_path=tmp_path / "missing"))
    )
    assert not empty.slurm_available
    assert "no cluster profile" in empty.availability_label.text()
    invalid_path = tmp_path / "bad.toml"
    invalid_path.write_text("[invalid", encoding="utf-8")
    invalid = RemoteExecutionControls(discover_remote_execution(invalid_path))
    assert not invalid.slurm_available
    assert "invalid cluster config" in invalid.availability_label.text()
    assert invalid.discovery.catalog.config_path == invalid_path
    empty.close()
    invalid.close()


def test_multiple_cluster_and_resource_selection_is_independent():
    _app()
    catalog = ClusterCatalog((_cluster("alpha"), _cluster("beta", (GPU,))), "alpha")
    controls = RemoteExecutionControls(RemoteExecutionDiscovery(catalog))
    assert controls.cluster_selector.count() == 2
    assert controls.resource_selector.count() == 2
    controls.cluster_selector.setCurrentText("beta")
    assert controls.resource_selector.count() == 1
    assert controls.selected_resource_name() == "GPU"
    controls.close()


def test_cupy_backend_rejected_for_cpu_resource_and_allowed_for_gpu():
    _app()
    controls = RemoteExecutionControls(
        RemoteExecutionDiscovery(ClusterCatalog((_cluster(),), "alpha"))
    )
    with pytest.raises(ValueError, match="CuPy"):
        controls.validate_backend("cupy")
    controls.validate_backend("numpy")
    controls.validate_backend("auto")
    controls.resource_selector.setCurrentText("GPU")
    controls.validate_backend("cupy")
    controls.close()


def test_gui_runner_uses_automatic_deployment_and_preserves_explicit_override(
    tmp_path, monkeypatch
):
    _app()
    controls = RemoteExecutionControls(
        RemoteExecutionDiscovery(ClusterCatalog((_cluster(),), "alpha"))
    )
    automatic = controls.create_runner()
    assert automatic.config.remote_source_path is None
    assert automatic._source_deployment_manager is not None
    monkeypatch.setenv("LCPROP_SLURM_SOURCE_PATH", "/pre/staged/source")
    monkeypatch.setenv("LCPROP_SLURM_SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("LCPROP_SLURM_LOCAL_ARTIFACT_ROOT", str(tmp_path))
    explicit = controls.create_runner()
    assert explicit.config.remote_source_path == "/pre/staged/source"
    assert explicit.config.source_git_sha == "a" * 40
    controls.close()


def test_main_windows_keep_local_default_and_show_profile_controls(
    tmp_path, monkeypatch
):
    _app()
    path = tmp_path / "clusters.toml"
    write_cluster_profiles(ClusterCatalog((_cluster(),), "alpha", path))
    monkeypatch.setenv("LCPROP_CLUSTER_CONFIG", str(path))
    lc = LCPropMainWindow()
    pr = PRMainWindow()
    for window in (lc, pr):
        assert window.execution_target_selector.currentData() == "local"
        assert window.runner is window.local_runner
        assert window.remote_execution_controls.slurm_available
        assert window.remote_execution_controls.selected_resource_name() == "CPU"
        window.close()


@pytest.mark.parametrize("window_type", (LCPropMainWindow, PRMainWindow))
def test_deleting_active_cluster_falls_back_to_local(
    tmp_path, monkeypatch, window_type
):
    _app()
    path = tmp_path / "clusters.toml"
    write_cluster_profiles(ClusterCatalog((_cluster(),), "alpha", path))
    monkeypatch.setenv("LCPROP_CLUSTER_CONFIG", str(path))
    window = window_type()
    window.execution_target_selector.setCurrentIndex(1)
    assert window.runner is window.slurm_runner
    window.remote_execution_controls._catalog_saved(ClusterCatalog(config_path=path))
    assert window.execution_target_selector.currentData() == "local"
    assert window.runner is window.local_runner
    assert window.remote_execution_controls.cluster_selector.count() == 0
    assert window.remote_execution_controls.resource_selector.count() == 0
    assert "no cluster profile" in (
        window.remote_execution_controls.availability_label.text()
    )
    window.close()


def test_connection_success_checks_commands_python_paths_and_optional_cupy():
    transport = RecordingTransport()
    result = ClusterConnectionTester(transport).test(_cluster(), "GPU")
    assert result.passed
    assert {check.name for check in result.checks} == {
        "SSH",
        "Slurm sbatch",
        "Slurm squeue",
        "Slurm sacct",
        "Remote Python",
        "Remote run root",
        "Remote source root",
        "CuPy import",
    }
    argument_lists = [args for _host, args in transport.calls]
    assert not any(
        args and args[0] in {"sbatch", "srun", "scancel"} for args in argument_lists
    )


def test_connection_cupy_probe_preserves_login_setup_and_python_code_quoting():
    gpu = SlurmResourceProfile(
        "Tufts H200",
        "gpu",
        "normal",
        "00:20:00",
        2,
        16,
        gpus=1,
        require_cupy=True,
        setup_commands=("module load cuda/12.9.0",),
    )
    cluster = _cluster(resources=(gpu,))
    transport = RecordingTransport()

    result = ClusterConnectionTester(transport).test(cluster, gpu.name)

    assert result.passed
    _host, arguments = next(
        call for call in transport.calls if "import cupy" in " ".join(call[1])
    )
    assert len(arguments) == 1
    shell_arguments = shlex.split(arguments[0])
    assert shell_arguments[:4] == [
        "bash",
        "-lc",
        'eval "$1"',
        "lcprop-connection-test",
    ]
    assert shell_arguments[4] == (
        "module load cuda/12.9.0; "
        f"{cluster.remote_python} -c 'import cupy; print(cupy.__version__)'"
    )


def test_connection_setup_command_arguments_remain_in_one_bounded_payload():
    setup = "export LCPROP_SITE_LABEL='Tufts H200 commissioned environment'"
    gpu = SlurmResourceProfile(
        "GPU",
        "gpu",
        "normal",
        "00:20:00",
        2,
        16,
        gpus=1,
        require_cupy=True,
        setup_commands=(setup, "module load cuda/12.9.0"),
    )
    cluster = _cluster(resources=(gpu,))
    transport = RecordingTransport()

    assert ClusterConnectionTester(transport).test(cluster, gpu.name).passed

    _host, arguments = next(
        call for call in transport.calls if "import cupy" in " ".join(call[1])
    )
    shell_arguments = shlex.split(arguments[0])
    assert len(arguments) == 1
    assert shell_arguments[4].startswith(setup + "; module load cuda/12.9.0; ")


def test_connection_uses_longer_timeout_only_for_cupy_environment_probe():
    transport = RecordingTransport()

    assert ClusterConnectionTester(transport).test(_cluster(), "GPU").passed

    assert transport.bounded_calls
    for _host, arguments, timeout_seconds in transport.bounded_calls:
        expected = 30.0 if "import cupy" in " ".join(arguments) else 7.5
        assert timeout_seconds == expected


def test_connection_timeout_is_reported_without_raw_subprocess_command():
    class TimeoutTransport(RecordingTransport):
        def ssh_with_timeout(
            self,
            host,
            *args,
            timeout_seconds,
            cancellation_check,
        ):
            if "import cupy" in " ".join(args):
                raise subprocess.TimeoutExpired(["ssh", host, *args], timeout_seconds)
            return super().ssh_with_timeout(
                host,
                *args,
                timeout_seconds=timeout_seconds,
                cancellation_check=cancellation_check,
            )

    result = ClusterConnectionTester(TimeoutTransport()).test(_cluster(), "GPU")

    cupy = next(check for check in result.checks if check.name == "CuPy import")
    assert not cupy.passed
    assert cupy.detail == "timed out after 30 seconds"
    assert "ssh" not in cupy.detail


def test_connection_transport_cancels_active_long_probe_boundedly():
    cancelled = Event()
    timer = Timer(0.05, cancelled.set)
    started = monotonic()
    timer.start()
    try:
        with pytest.raises(RuntimeError, match="connection test cancelled"):
            ConnectionTestRemoteTransport._run_bounded(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                timeout_seconds=30.0,
                cancellation_check=cancelled.is_set,
            )
    finally:
        timer.cancel()
    assert monotonic() - started < 2.0


def test_connection_test_refuses_scheduler_commands_in_gpu_setup():
    unsafe = SlurmResourceProfile(
        "GPU",
        "gpu",
        "normal",
        "00:10:00",
        1,
        8,
        gpus=1,
        require_cupy=True,
        setup_commands=("sbatch forbidden.sbatch",),
    )
    transport = RecordingTransport()
    result = ClusterConnectionTester(transport).test(
        _cluster(resources=(unsafe,)), "GPU"
    )
    cupy = next(check for check in result.checks if check.name == "CuPy import")
    assert not cupy.passed
    assert not any(args and args[0] == "sbatch" for _host, args in transport.calls)


@pytest.mark.parametrize(
    ("failure", "failed_check"),
    (
        ("true", "SSH"),
        ("sbatch", "Slurm sbatch"),
        ("squeue", "Slurm squeue"),
        ("sacct", "Slurm sacct"),
        ("--version", "Remote Python"),
        ("/scratch/alpha/runs", "Remote run root"),
        ("/scratch/alpha/sources", "Remote source root"),
        ("import cupy", "CuPy import"),
    ),
)
def test_connection_failures_are_structured(failure, failed_check):
    result = ClusterConnectionTester(RecordingTransport((failure,))).test(
        _cluster(), "GPU"
    )
    check = next(value for value in result.checks if value.name == failed_check)
    assert not check.passed
    if failed_check == "SSH":
        assert "system SSH" in check.detail


def test_connection_worker_keeps_dialog_call_nonblocking():
    _app()
    catalog = ClusterCatalog((_cluster(),), "alpha")

    class DeferredTester:
        def test(self, *_args, **_kwargs):
            return ClusterConnectionTester(RecordingTransport()).test(_cluster(), "CPU")

    dialog = RemoteExecutionDialog(catalog, connection_tester=DeferredTester())
    dialog.test_connection()
    assert not dialog.test_button.isEnabled()
    _wait_for_connection_thread(dialog)
    assert dialog.test_button.isEnabled()
    dialog.close()


class BlockingTester:
    def __init__(self):
        self.started = Event()
        self.release = Event()
        self.calls = 0

    def test(self, cluster, resource, *, cancellation_check):
        self.calls += 1
        self.started.set()
        while not self.release.wait(0.005):
            if cancellation_check():
                break
        return ClusterConnectionResult(
            cluster.name,
            resource,
            (ConnectionCheck("Synthetic", True, "late result"),),
        )


class BlockingEnvironmentProbeTransport(RecordingTransport):
    def __init__(self):
        super().__init__()
        self.started = Event()
        self.cancelled = Event()

    def ssh_with_timeout(
        self,
        host,
        *args,
        timeout_seconds,
        cancellation_check,
    ):
        if "import cupy" not in " ".join(args):
            return super().ssh_with_timeout(
                host,
                *args,
                timeout_seconds=timeout_seconds,
                cancellation_check=cancellation_check,
            )
        assert timeout_seconds == 30.0
        self.started.set()
        while not cancellation_check():
            self.cancelled.wait(0.005)
        self.cancelled.set()
        raise RuntimeError("connection test cancelled")


def test_connection_test_disables_edits_and_duplicate_launches():
    _app()
    tester = BlockingTester()
    dialog = RemoteExecutionDialog(
        ClusterCatalog((_cluster(),), "alpha"), connection_tester=tester
    )
    dialog.test_connection()
    assert tester.started.wait(1)
    first_thread = dialog._connection_thread
    dialog.test_connection()
    assert dialog._connection_thread is first_thread
    assert tester.calls == 1
    assert not dialog.test_button.isEnabled()
    assert all(not widget.isEnabled() for widget in dialog._configuration_widgets)
    tester.release.set()
    _wait_for_connection_thread(dialog)
    assert dialog.test_button.isEnabled()
    assert all(widget.isEnabled() for widget in dialog._configuration_widgets)
    dialog.close()


def test_connection_test_failure_restores_controls():
    _app()

    class FailingTester:
        def test(self, *_args, **_kwargs):
            raise RuntimeError("synthetic connection failure")

    dialog = RemoteExecutionDialog(
        ClusterCatalog((_cluster(),), "alpha"), connection_tester=FailingTester()
    )
    dialog.test_connection()
    _wait_for_connection_thread(dialog)
    assert dialog.test_button.isEnabled()
    assert all(widget.isEnabled() for widget in dialog._configuration_widgets)
    assert "synthetic connection failure" in dialog.test_results.text()
    dialog.close()


def test_dialog_close_cancels_worker_and_ignores_late_result():
    _app()
    tester = BlockingTester()
    dialog = RemoteExecutionDialog(
        ClusterCatalog((_cluster(),), "alpha"), connection_tester=tester
    )
    dialog.show()
    dialog.test_connection()
    assert tester.started.wait(1)
    thread = dialog._connection_thread
    dialog.close_button.click()
    assert dialog._connection_close_pending
    assert "Closing after" in dialog.test_results.text()
    loop = QEventLoop()
    QTimer.singleShot(2000, loop.quit)
    thread.finished.connect(loop.quit)
    loop.exec()
    QApplication.processEvents()
    assert dialog._connection_thread is None
    assert not dialog.isVisible()
    assert "late result" not in dialog.test_results.text()
    assert all(widget.isEnabled() for widget in dialog._configuration_widgets)


def test_dialog_close_cancels_longer_environment_probe_without_lingering_thread():
    _app()
    transport = BlockingEnvironmentProbeTransport()
    dialog = RemoteExecutionDialog(
        ClusterCatalog((_cluster(resources=(GPU,)),), "alpha"),
        connection_tester=ClusterConnectionTester(transport),
    )
    dialog.show()
    dialog.test_connection()
    assert transport.started.wait(1)
    thread = dialog._connection_thread

    dialog.close_button.click()

    loop = QEventLoop()
    QTimer.singleShot(2000, loop.quit)
    thread.finished.connect(loop.quit)
    loop.exec()
    QApplication.processEvents()
    assert transport.cancelled.is_set()
    assert dialog._connection_thread is None
    assert not dialog.isVisible()


@pytest.mark.parametrize("window_type", (LCPropMainWindow, PRMainWindow))
def test_parent_window_shutdown_joins_active_connection_worker(window_type):
    _app()
    tester = BlockingTester()
    window = window_type()
    dialog = RemoteExecutionDialog(
        ClusterCatalog((_cluster(),), "alpha"),
        connection_tester=tester,
        parent=window.remote_execution_controls,
    )
    window.remote_execution_controls._dialog = dialog
    dialog.test_connection()
    assert tester.started.wait(1)
    assert window.shutdown_background_run(timeout_ms=2000)
    assert dialog._connection_thread is None
    window.close()


@pytest.mark.parametrize("window_type", (LCPropMainWindow, PRMainWindow))
def test_parent_window_shutdown_cancels_longer_environment_probe(window_type):
    _app()
    transport = BlockingEnvironmentProbeTransport()
    window = window_type()
    dialog = RemoteExecutionDialog(
        ClusterCatalog((_cluster(resources=(GPU,)),), "alpha"),
        connection_tester=ClusterConnectionTester(transport),
        parent=window.remote_execution_controls,
    )
    window.remote_execution_controls._dialog = dialog
    dialog.test_connection()
    assert transport.started.wait(1)

    assert window.shutdown_background_run(timeout_ms=2000)

    assert transport.cancelled.is_set()
    assert dialog._connection_thread is None
    window.close()


def test_stale_connection_callback_is_ignored():
    _app()
    dialog = RemoteExecutionDialog(ClusterCatalog((_cluster(),), "alpha"))
    stale = ClusterConnectionResult(
        "alpha", "CPU", (ConnectionCheck("Synthetic", True, "stale"),)
    )
    dialog._connection_generation = 2
    dialog._connection_result_ready(1, stale)
    assert dialog._connection_result is None
    dialog.close()


def test_runner_and_profile_models_have_no_qt_or_material_logic():
    from lcprop.runners import cluster_connection, cluster_profiles

    source = inspect.getsource(cluster_connection) + inspect.getsource(cluster_profiles)
    assert "PySide" not in source and "Qt" not in source
    assert (
        "material_id" not in source
        and '== "lc"' not in source
        and '== "pr"' not in source
    )
