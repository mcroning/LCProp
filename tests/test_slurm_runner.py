import ast
from dataclasses import replace
import inspect
import json
from pathlib import Path
import re
import shlex
from types import SimpleNamespace

import pytest
import lcprop.runners.slurm as slurm_module

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.backend import BackendSpec
from lcprop.core.execution import CancellationToken
from lcprop.lc.operations import LC_STATIC_OPERATION
from lcprop.lc.requests import OutputOptions, StaticRunRequest, StaticSolverOptions
from lcprop.lc.specs import BiasSpec, LCMaterial
from lcprop.runners.slurm import (
    RemoteExecutionError, RemoteRunCancelled, SlurmExecutionConfig,
    SlurmResourceProfile, SlurmRunner, _device_pattern_preflight,
)
from lcprop.runners.source_deployment import (
    ResolvedSourceDeployment,
    SourceDeploymentError,
)
from lcprop.pr.operations import PR_TIMEDEPENDENT_OPERATION
from lcprop.pr.specs import PRMaterialSpec, PRRunRequest, PRSolverOptions
from lcprop.pr.transverse.operations import PR_TRANSVERSE_STATIC_OPERATION
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest, PRTransverseStaticWorkflowOptions,
)
from lcprop.transport.executor import execute_run_directory
from lcprop.transport.defaults import default_transport_registry
from lcprop.transport.envelopes import TransportVerificationError
from lcprop.transport.status import RemoteRunState


SHA = "e366537b0825c0e2c02065c24756336e1bcd774c"
COMMISSIONED_STAGE_D_SHA = "eb9382c0843a25136309728e57d0a7fb4dae333f"
CPU_SMALL = SlurmResourceProfile(
    "CPU small", "batch", "normal", "00:15:00", 2, 8
)
H200_SMALL = SlurmResourceProfile(
    "H200 small",
    "gpu",
    "normal",
    "00:15:00",
    2,
    16,
    gpus=1,
    setup_commands=("module load cuda/12.9.0",),
    gres="gpu:h200:1",
    require_cupy=True,
    minimum_device_count=1,
    expected_device_pattern="H200",
)


def _request():
    return StaticRunRequest(
        grid=GridSpec(
            Nx=8, Ny=8, x_aperture_um=20.0, y_aperture_um=20.0,
            dz_um=2.0, z_length_um=2.0,
        ),
        material=LCMaterial(), bias=BiasSpec(),
        beams=BeamStack(channels=(BeamChannel(
            wavelength_um=0.633, waist_x_um=4.0, waist_y_um=4.0,
        ),)),
        solver=StaticSolverOptions(), output=OutputOptions(),
    )


def _pr_request():
    return PRTransverseStaticRunRequest(
        grid=GridSpec(
            Nx=8, Ny=8, x_aperture_um=20.0, y_aperture_um=20.0,
            dz_um=2.0, z_length_um=2.0,
        ),
        beams=BeamStack(channels=(BeamChannel(
            wavelength_um=0.633, waist_x_um=4.0, waist_y_um=4.0,
            coherence_group="remote-pr",
        ),)),
        material=PRMaterialSpec(
            dark_intensity=0.4, uniform_background_intensity=0.1,
            gain_length_product=1e-3,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRTransverseStaticWorkflowOptions(max_coupled_iterations=2),
        backend=BackendSpec(backend="numpy", precision="float64", verbose=False),
    )


def _pr_td_request():
    return PRRunRequest(
        grid=GridSpec(
            Nx=8,
            Ny=8,
            x_aperture_um=20.0,
            y_aperture_um=20.0,
            dz_um=2.0,
            z_length_um=2.0,
        ),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    wavelength_um=0.633,
                    waist_x_um=4.0,
                    waist_y_um=4.0,
                    coherence_group="remote-pr-td",
                ),
            )
        ),
        material=PRMaterialSpec(
            dark_intensity=0.4,
            uniform_background_intensity=0.1,
            gain_length_product=1e-3,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRSolverOptions(Nt=0, dt_normalized=0.001),
        backend=BackendSpec(
            backend="auto", precision="float32", verbose=False
        ),
    )


class FakeTransport:
    def __init__(self, *, polls=None):
        self.submissions = 0
        self.polls = iter(
            ("PENDING|0:0", "RUNNING|0:0", "COMPLETED|0:0")
            if polls is None else polls
        )
        self.commands = []

    def ssh(self, host, *arguments):
        self.commands.append((host, arguments))
        if arguments == ("cat", "/snapshot/.lcprop-source-sha"):
            return SHA
        if arguments[0] == "sbatch":
            self.submissions += 1
            return "12345"
        if arguments[0] == "sacct":
            return next(self.polls)
        return ""

    def upload(self, host, local, remote):
        self.commands.append((host, ("upload", str(local), remote)))

    def download(self, host, remote, local):
        self.commands.append((host, ("download", remote, str(local))))
        if remote.endswith("/output"):
            assert execute_run_directory(local) == 0
            return
        if remote.endswith("/execution_provenance.json"):
            (Path(local) / "execution_provenance.json").write_text(
                json.dumps({
                    "device": "NVIDIA H200",
                    "device_count": 1,
                    "preflight_backend": "cupy",
                }),
                encoding="utf-8",
            )
            return
        raise AssertionError(f"unexpected download {remote}")


def _cleanup_commands(transport):
    return [
        arguments
        for _host, arguments in transport.commands
        if len(arguments) == 4 and arguments[:3] == ("rm", "-rf", "--")
    ]


class FakeDeploymentManager:
    def __init__(self, *, fail=False):
        self.calls = 0
        self.fail = fail

    def resolve_or_stage(self):
        self.calls += 1
        if self.fail:
            raise SourceDeploymentError("source_dirty", "synthetic dirty source")
        provenance = {
            "source_kind": "committed_git_archive",
            "source_git_sha": SHA,
            "remote_source_path": "/snapshot",
            "source_checksum_sha256": "a" * 64,
            "snapshot_reused": self.calls > 1,
        }
        return ResolvedSourceDeployment(
            source_kind="committed_git_archive",
            source_git_sha=SHA,
            remote_source_path="/snapshot",
            source_checksum="a" * 64,
            reused_existing_snapshot=self.calls > 1,
            provenance=provenance,
        )


def _config(tmp_path, **changes):
    values = dict(
        host="login-p03.pax.tufts.edu",
        remote_run_root="/runs",
        remote_python="/env/bin/python",
        remote_source_path="/snapshot",
        source_git_sha=SHA,
        local_artifact_root=tmp_path,
        resource_profiles=(CPU_SMALL, H200_SMALL),
        default_resource_profile="CPU small",
        poll_interval=0.001,
    )
    values.update(changes)
    return SlurmExecutionConfig(**values)


def test_execution_config_and_resource_profiles_are_strict_and_credential_free(tmp_path):
    config = _config(tmp_path)
    assert config.profile("CPU small") is CPU_SMALL
    assert not hasattr(config, "password")
    with pytest.raises(ValueError):
        _config(tmp_path, host="host; touch bad")
    with pytest.raises(ValueError):
        SlurmResourceProfile("bad", "gpu", "normal", "1h", 1, 1)
    with pytest.raises(ValueError, match="default_resource_profile"):
        _config(tmp_path, default_resource_profile="missing")


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "/runs/bad path",
        "/runs/bad;command",
        "/runs/$(command)",
        "/runs/`command`",
        "/runs/bad\npath",
        "/runs/'quoted'",
        "/runs/bad\\path",
        "/runs/bad|pipe",
        "/runs/bad>redirect",
        "/runs/bad&command",
    ),
)
@pytest.mark.parametrize(
    "field", ("remote_run_root", "remote_python", "remote_source_path")
)
def test_direct_execution_config_rejects_shell_unsafe_remote_paths(
    tmp_path, field, unsafe_path
):
    with pytest.raises(ValueError, match="remote paths"):
        _config(tmp_path, **{field: unsafe_path})


def test_direct_execution_config_accepts_required_safe_posix_path_characters(
    tmp_path,
):
    safe = "/cluster/project/user/lcprop-runs_1.2+gpu@site%5=value:tag,part"
    config = _config(
        tmp_path,
        remote_run_root=safe,
        remote_python=f"{safe}/bin/python",
        remote_source_path=f"{safe}/source",
    )
    assert config.remote_run_root == safe


def test_direct_gpu_profile_requires_cupy():
    with pytest.raises(ValueError, match="require_cupy=true"):
        SlurmResourceProfile(
            "GPU without CuPy",
            "gpu",
            "normal",
            "00:15:00",
            2,
            16,
            gpus=1,
            gres="gpu:1",
            require_cupy=False,
        )


@pytest.mark.parametrize("command", ("module load cuda\nwhoami", "module\0load"))
def test_setup_commands_remain_single_line_and_nul_free(command):
    with pytest.raises(ValueError, match="single lines"):
        SlurmResourceProfile(
            "GPU",
            "gpu",
            "normal",
            "00:15:00",
            2,
            16,
            gpus=1,
            setup_commands=(command,),
            gres="gpu:1",
            require_cupy=True,
        )


def test_shared_runner_has_no_material_specific_resource_branching(tmp_path):
    source = inspect.getsource(SlurmRunner)
    material_literals = {
        node.value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant) and node.value in {"lc", "pr"}
    }
    assert material_literals == set()
    assert "default_resource_profile" in source
    config = _config(tmp_path, default_resource_profile=None)
    runner = SlurmRunner(
        config, (LC_STATIC_OPERATION,), transport=FakeTransport(),
        registry=default_transport_registry(),
    )
    with pytest.raises(ValueError, match="resource_profile is required"):
        runner.run_registered("lc", "static", _request())


@pytest.mark.parametrize("backend", ("numpy", "cupy"))
def test_verified_result_backend_provenance_is_result_type_independent(backend):
    decoded = SimpleNamespace(
        envelope=SimpleNamespace(
            scientific_backend_resolved=backend,
            device_summary={"backend": backend, "device": "test-device"},
        ),
        result=object(),
    )

    resolved, device = slurm_module._verified_result_provenance(decoded)

    assert resolved == backend
    assert device == {"backend": backend, "device": "test-device"}


def test_final_remote_status_uses_verified_cupy_envelope_for_td_result(
    tmp_path, monkeypatch
):
    original_read = slurm_module.read_result_package

    def read_with_verified_cupy_provenance(*args, **kwargs):
        decoded = original_read(*args, **kwargs)
        return replace(
            decoded,
            envelope=replace(
                decoded.envelope,
                scientific_backend_resolved="cupy",
                device_summary={"backend": "cupy", "is_gpu": True},
            ),
        )

    monkeypatch.setattr(
        slurm_module, "read_result_package", read_with_verified_cupy_provenance
    )
    transport = FakeTransport()
    states = []
    runner = SlurmRunner(
        _config(tmp_path),
        (PR_TIMEDEPENDENT_OPERATION,),
        transport=transport,
        registry=default_transport_registry(),
        sleep=lambda _seconds: None,
    )

    completed = runner.run_registered(
        "pr",
        "pr_timedependent",
        _pr_td_request(),
        resource_profile="H200 small",
        progress_callback=states.append,
    )

    assert not hasattr(completed.result, "backend_summary")
    assert states[-1].state == RemoteRunState.COMPLETED
    assert states[-1].scientific_backend_resolved == "cupy"
    assert states[-1].device_summary["backend"] == "cupy"
    assert states[-1].device_summary["device"] == "NVIDIA H200"


def test_commissioning_script_defaults_to_accepted_stage_d_snapshot():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts/checks/slurm_gui_commissioning.py"
    ).read_text(encoding="utf-8")
    assert f'SOURCE_SHA = "{COMMISSIONED_STAGE_D_SHA}"' in script
    assert "lcprop-remote-eb9382c0843a" in script


def test_slurm_runner_submits_exactly_once_and_completes_only_after_reconstruction(tmp_path):
    transport = FakeTransport()
    states = []
    runner = SlurmRunner(
        _config(tmp_path), (LC_STATIC_OPERATION,), transport=transport,
        registry=default_transport_registry(),
        sleep=lambda _seconds: None,
    )
    completed = runner.run_registered(
        "lc", "static", _request(), progress_callback=lambda value: states.append(value)
    )
    assert transport.submissions == 1
    assert completed.run_data is not None
    assert [value.state for value in states] == [
        RemoteRunState.SUBMITTING,
        RemoteRunState.PENDING,
        RemoteRunState.RUNNING,
        RemoteRunState.SCIENTIFICALLY_FINISHED,
        RemoteRunState.RETRIEVING,
        RemoteRunState.VERIFYING,
        RemoteRunState.RECONSTRUCTING,
        RemoteRunState.COMPLETED,
    ]
    assert not states[-2].gui_product_ready
    assert states[-1].gui_product_ready
    assert _cleanup_commands(transport) == [
        ("rm", "-rf", "--", states[-1].remote_artifact_location)
    ]
    assert states[-1].remote_cleanup_requested is True
    assert states[-1].remote_cleanup_succeeded is True
    assert states[-1].remote_cleanup_completed_at is not None
    assert states[-1].remote_cleanup_target == states[-1].remote_artifact_location
    assert states[-1].remote_artifacts_retained is False
    assert states[-1].remote_cleanup_error is None
    assert (Path(states[-1].local_artifact_location) / "output").is_dir()


@pytest.mark.parametrize(
    ("cancel_while", "cancelled_state"),
    (
        ("pending", "CANCELLED"),
        ("pending", "CANCELLED+"),
        ("pending", "CANCELLED by 12345"),
        ("running", "CANCELLED"),
    ),
)
def test_cancellation_is_idempotent_monotonic_and_terminal(
    tmp_path, cancel_while, cancelled_state
):
    polls = (
        ("PENDING|0:0", f"{cancelled_state}|0:15")
        if cancel_while == "pending"
        else ("RUNNING|0:0", f"{cancelled_state}|0:15")
    )
    transport = FakeTransport(polls=polls)
    states = []
    token = CancellationToken()
    if cancel_while == "pending":
        token.cancel()

    def observe(status):
        states.append(status)
        if cancel_while == "running" and status.state == RemoteRunState.RUNNING:
            token.cancel()

    runner = SlurmRunner(
        _config(tmp_path), (LC_STATIC_OPERATION,), transport=transport,
        registry=default_transport_registry(), sleep=lambda _seconds: None,
    )
    with pytest.raises(RemoteRunCancelled) as caught:
        runner.run_registered(
            "lc", "static", _request(), cancellation_token=token,
            progress_callback=observe,
        )
    assert caught.value.category == "cancelled"
    assert states[-1].state == RemoteRunState.CANCELLED
    assert sum(
        1 for _host, arguments in transport.commands
        if arguments and arguments[0] == "scancel"
    ) == 1
    cancel_index = next(
        index for index, value in enumerate(states)
        if value.state == RemoteRunState.CANCEL_REQUESTED
    )
    assert all(
        value.state not in {RemoteRunState.PENDING, RemoteRunState.RUNNING}
        for value in states[cancel_index + 1:]
    )
    assert _cleanup_commands(transport) == []
    assert sum(
        1 for _host, arguments in transport.commands
        if arguments and arguments[0] == "sacct"
    ) == 2


class RetrievalFailureTransport(FakeTransport):
    def download(self, host, remote, local):
        raise OSError("synthetic retrieval failure")


def _run_phase_failure(tmp_path, *, transport=None, operation=None):
    states = []
    active_transport = FakeTransport() if transport is None else transport
    runner = SlurmRunner(
        _config(tmp_path),
        (LC_STATIC_OPERATION if operation is None else operation,),
        transport=active_transport,
        registry=default_transport_registry(),
        sleep=lambda _seconds: None,
    )
    with pytest.raises(RemoteExecutionError) as caught:
        runner.run_registered(
            "lc", "static", _request(), progress_callback=states.append
        )
    assert states[-1].state == RemoteRunState.FAILED
    assert states[-1].failure_reason is not None
    assert states[-1].progress_metadata == {
        "failure_category": caught.value.category
    }
    assert _cleanup_commands(active_transport) == []
    return caught.value.category, states


def test_retrieval_failure_reaches_terminal_failed(tmp_path):
    category, states = _run_phase_failure(
        tmp_path, transport=RetrievalFailureTransport()
    )
    assert category == "retrieval"
    assert RemoteRunState.RETRIEVING in [value.state for value in states]


def test_checksum_verification_failure_reaches_terminal_failed(tmp_path, monkeypatch):
    def fail_verification(*_args, **_kwargs):
        raise TransportVerificationError("synthetic checksum failure")

    monkeypatch.setattr(slurm_module, "read_result_package", fail_verification)
    category, states = _run_phase_failure(tmp_path)
    assert category == "verification"
    assert RemoteRunState.VERIFYING in [value.state for value in states]


def test_canonical_reconstruction_failure_reaches_terminal_failed(
    tmp_path, monkeypatch
):
    def fail_reconstruction(*_args, **_kwargs):
        raise ValueError("synthetic reconstruction failure")

    monkeypatch.setattr(slurm_module, "read_result_package", fail_reconstruction)
    category, states = _run_phase_failure(tmp_path)
    assert category == "reconstruction"
    assert RemoteRunState.RECONSTRUCTING in [value.state for value in states]


def test_product_conversion_failure_reaches_terminal_failed(tmp_path):
    def fail_product(_result):
        raise ValueError("synthetic product failure")

    operation = replace(LC_STATIC_OPERATION, to_run_data=fail_product)
    category, states = _run_phase_failure(tmp_path, operation=operation)
    assert category == "product_conversion"
    assert RemoteRunState.RECONSTRUCTING in [value.state for value in states]


def test_runner_rejects_remote_snapshot_mismatch_before_submission(tmp_path):
    transport = FakeTransport()
    runner = SlurmRunner(
        _config(tmp_path, source_git_sha="0" * 40),
        (LC_STATIC_OPERATION,), transport=transport,
        registry=default_transport_registry(),
    )
    with pytest.raises(RuntimeError, match="remote source SHA mismatch"):
        runner.run_registered("lc", "static", _request())
    assert transport.submissions == 0


def test_runner_resolves_dynamic_source_and_records_deployment_provenance(tmp_path):
    transport = FakeTransport()
    deployment = FakeDeploymentManager()
    config = _config(
        tmp_path,
        remote_source_path=None,
        source_git_sha=None,
        cluster_profile="test-cluster",
    )
    runner = SlurmRunner(
        config,
        (LC_STATIC_OPERATION,),
        transport=transport,
        source_deployment_manager=deployment,
        registry=default_transport_registry(),
        sleep=lambda _seconds: None,
    )
    runner.run_registered("lc", "static", _request())
    assert deployment.calls == 1
    assert transport.submissions == 1
    local_run = next(tmp_path.iterdir())
    envelope = json.loads(
        (local_run / "request/request.json").read_text(encoding="utf-8")
    )
    assert envelope["provenance"] == {
        "cluster_profile": "test-cluster",
        "remote_source_path": "/snapshot",
        "resource_profile": "CPU small",
        "snapshot_reused": False,
        "source_checksum_sha256": "a" * 64,
        "source_git_sha": SHA,
        "source_kind": "committed_git_archive",
    }
    assert f"export PYTHONPATH=/snapshot/src" in (
        local_run / "launch.sbatch"
    ).read_text(encoding="utf-8")


def test_source_deployment_failure_prevents_submission(tmp_path):
    transport = FakeTransport()
    states = []
    runner = SlurmRunner(
        _config(tmp_path, remote_source_path=None, source_git_sha=None),
        (LC_STATIC_OPERATION,),
        transport=transport,
        source_deployment_manager=FakeDeploymentManager(fail=True),
        registry=default_transport_registry(),
    )
    with pytest.raises(RemoteExecutionError, match="source_dirty"):
        runner.run_registered(
            "lc", "static", _request(), progress_callback=states.append
        )
    assert transport.submissions == 0
    assert states[-1].state == RemoteRunState.FAILED
    assert states[-1].progress_metadata == {"failure_category": "source_dirty"}


def test_pr_transverse_static_uses_same_remote_artifacts_and_product_adapter(tmp_path):
    transport = FakeTransport()
    states = []
    runner = SlurmRunner(
        _config(tmp_path), (PR_TRANSVERSE_STATIC_OPERATION,),
        transport=transport, registry=default_transport_registry(),
        sleep=lambda _seconds: None,
    )
    completed = runner.run_registered(
        "pr", "pr_transverse_static", _pr_request(),
        resource_profile="H200 small",
        progress_callback=states.append,
    )
    assert completed.kind == "pr_transverse_static"
    assert completed.run_data.workflow == "pr_transverse_static"
    assert transport.submissions == 1
    assert states[-1].scientific_backend_requested == "numpy"
    assert states[-1].scientific_backend_resolved == "numpy"
    assert states[-1].device_summary["device"] == "NVIDIA H200"
    assert (Path(states[-1].local_artifact_location) / "execution_provenance.json").is_file()
    assert _cleanup_commands(transport) == [
        ("rm", "-rf", "--", states[-1].remote_artifact_location)
    ]
    assert states[-1].remote_cleanup_succeeded is True


def test_cleanup_runs_only_after_product_conversion(tmp_path):
    product_ready = False

    def convert(result):
        nonlocal product_ready
        product_ready = True
        return LC_STATIC_OPERATION.to_run_data(result)

    class GateTransport(FakeTransport):
        def ssh(self, host, *arguments):
            if arguments and arguments[0] == "rm":
                assert product_ready
            return super().ssh(host, *arguments)

    transport = GateTransport()
    operation = replace(LC_STATIC_OPERATION, to_run_data=convert)
    runner = SlurmRunner(
        _config(tmp_path),
        (operation,),
        transport=transport,
        registry=default_transport_registry(),
        sleep=lambda _seconds: None,
    )

    result = runner.run_registered("lc", "static", _request())

    assert result.run_data is not None
    assert product_ready
    assert len(_cleanup_commands(transport)) == 1


def test_cleanup_disabled_retains_successful_remote_artifacts(tmp_path):
    transport = FakeTransport()
    states = []
    runner = SlurmRunner(
        _config(tmp_path, cleanup_remote_on_success=False),
        (LC_STATIC_OPERATION,),
        transport=transport,
        registry=default_transport_registry(),
        sleep=lambda _seconds: None,
    )

    result = runner.run_registered(
        "lc", "static", _request(), progress_callback=states.append
    )

    assert result.run_data is not None
    assert _cleanup_commands(transport) == []
    assert states[-1].state == RemoteRunState.COMPLETED
    assert states[-1].remote_cleanup_requested is False
    assert states[-1].remote_cleanup_succeeded is None
    assert states[-1].remote_artifacts_retained is True
    assert "retained" in states[-1].state_message


def test_cleanup_failure_is_nonblocking_and_preserves_local_result(tmp_path):
    class CleanupFailureTransport(FakeTransport):
        def ssh(self, host, *arguments):
            if arguments and arguments[0] == "rm":
                self.commands.append((host, arguments))
                raise OSError("synthetic cleanup failure")
            return super().ssh(host, *arguments)

    transport = CleanupFailureTransport()
    states = []
    runner = SlurmRunner(
        _config(tmp_path),
        (LC_STATIC_OPERATION,),
        transport=transport,
        registry=default_transport_registry(),
        sleep=lambda _seconds: None,
    )

    result = runner.run_registered(
        "lc", "static", _request(), progress_callback=states.append
    )

    assert result.run_data is not None
    completed = states[-1]
    assert completed.state == RemoteRunState.COMPLETED
    assert completed.gui_product_ready
    assert completed.remote_cleanup_requested is True
    assert completed.remote_cleanup_succeeded is False
    assert completed.remote_artifacts_retained is True
    assert "synthetic cleanup failure" in completed.remote_cleanup_error
    assert "cleanup failed" in completed.state_message
    assert (Path(completed.local_artifact_location) / "output").is_dir()


@pytest.mark.parametrize(
    "poll",
    ("FAILED|1:0", "TIMEOUT|0:0", "OUT_OF_MEMORY|0:0"),
)
def test_scheduler_terminal_failures_never_trigger_cleanup(tmp_path, poll):
    transport = FakeTransport(polls=(poll,))
    runner = SlurmRunner(
        _config(tmp_path),
        (LC_STATIC_OPERATION,),
        transport=transport,
        registry=default_transport_registry(),
        sleep=lambda _seconds: None,
    )

    with pytest.raises(RemoteExecutionError):
        runner.run_registered("lc", "static", _request())

    assert _cleanup_commands(transport) == []


@pytest.mark.parametrize(
    ("run_id", "remote_run_path", "source_path"),
    (
        ("bad", "/runs/bad", "/snapshot"),
        ("lcprop-" + "a" * 32, "/runs", "/snapshot"),
        ("lcprop-" + "a" * 32, "/other/lcprop-" + "a" * 32, "/snapshot"),
        ("lcprop-" + "a" * 32, "/runs/lcprop-*", "/snapshot"),
        (
            "lcprop-" + "a" * 32,
            "/runs/lcprop-" + "a" * 32,
            "/runs/lcprop-" + "a" * 32 + "/source",
        ),
    ),
)
def test_cleanup_rejects_nonexact_or_source_related_targets(
    tmp_path, run_id, remote_run_path, source_path
):
    transport = FakeTransport()
    runner = SlurmRunner(
        _config(tmp_path),
        (LC_STATIC_OPERATION,),
        transport=transport,
        registry=default_transport_registry(),
    )

    with pytest.raises(ValueError):
        runner.cleanup_remote_run(
            run_id,
            remote_run_path,
            remote_source_path=source_path,
        )

    assert _cleanup_commands(transport) == []


def test_cleanup_uses_exact_symlink_safe_rm_operand_and_never_source_snapshot(
    tmp_path,
):
    transport = FakeTransport()
    runner = SlurmRunner(
        _config(tmp_path),
        (LC_STATIC_OPERATION,),
        transport=transport,
        registry=default_transport_registry(),
    )
    run_id = "lcprop-" + "a" * 32
    target = f"/runs/{run_id}"

    runner.cleanup_remote_run(
        run_id,
        target,
        remote_source_path="/sources/git-" + "b" * 40,
    )

    assert _cleanup_commands(transport) == [("rm", "-rf", "--", target)]
    assert not target.endswith("/")
    assert all("/sources" not in value for value in _cleanup_commands(transport)[0])


def test_slurm_script_keeps_execution_and_scientific_backend_separate(tmp_path):
    runner = SlurmRunner(
        _config(tmp_path), (LC_STATIC_OPERATION,), transport=FakeTransport(),
        registry=default_transport_registry(),
    )
    script = runner._script("/runs/test", CPU_SMALL)
    assert "lcprop.transport.executor" in script
    assert "--partition=batch" in script
    assert "#SBATCH --gres" not in script
    assert "#SBATCH --gpus" not in script
    assert "execution_provenance.json" not in script
    assert "nvidia-smi" not in script
    assert "backend" not in script.lower()
    lines = script.splitlines()
    assert max(i for i, line in enumerate(lines) if line.startswith("#SBATCH")) < lines.index(
        "set -euo pipefail"
    )


def test_h200_preflight_precedes_scientific_executor(tmp_path):
    runner = SlurmRunner(
        _config(tmp_path), (PR_TRANSVERSE_STATIC_OPERATION,),
        transport=FakeTransport(), registry=default_transport_registry(),
    )
    lines = runner._script("/runs/test", H200_SMALL).splitlines()
    assert "#SBATCH --partition=gpu" in lines
    assert "#SBATCH --gres=gpu:h200:1" in lines
    preflight_index = next(
        index for index, line in enumerate(lines)
        if "execution_provenance.json" in line
    )
    executor_index = next(
        index for index, line in enumerate(lines)
        if "-m lcprop.transport.executor" in line
    )
    assert preflight_index < executor_index
    assert "re.search" in lines[preflight_index]
    assert "H200" in lines[preflight_index]
    assert "BackendSpec" in lines[preflight_index]
    assert "cupy" in lines[preflight_index]
    assert any("gpu_memory_samples_mib.txt" in line for line in lines)


def _generated_gpu_preflight(tmp_path, profile):
    runner = SlurmRunner(
        _config(
            tmp_path,
            resource_profiles=(profile,),
            default_resource_profile=profile.name,
        ),
        (PR_TRANSVERSE_STATIC_OPERATION,),
        transport=FakeTransport(),
        registry=default_transport_registry(),
    )
    line = next(
        value
        for value in runner._script("/runs/test", profile).splitlines()
        if "execution_provenance.json" in value
    )
    arguments = shlex.split(line)
    return arguments[arguments.index("-c") + 1]


@pytest.mark.parametrize(
    ("pattern", "matching_name", "nonmatching_name"),
    (
        ("H200", "NVIDIA H200", "NVIDIA A100"),
        (r"NVIDIA (?:H200|'Hopper')$", "NVIDIA 'Hopper'", "NVIDIA L40S"),
    ),
)
def test_gpu_preflight_pattern_is_a_safe_literal_and_constrains_device(
    tmp_path, pattern, matching_name, nonmatching_name
):
    profile = replace(H200_SMALL, expected_device_pattern=pattern)
    preflight = _generated_gpu_preflight(tmp_path, profile)
    compile(preflight, "<gpu-preflight>", "exec")
    pattern_check = ";".join(_device_pattern_preflight(pattern))
    exec(pattern_check, {"re": re, "name": matching_name})
    with pytest.raises(AssertionError, match="expected device matching"):
        exec(pattern_check, {"re": re, "name": nonmatching_name})


def test_gpu_preflight_without_expected_pattern_compiles_and_keeps_provenance(tmp_path):
    profile = replace(H200_SMALL, expected_device_pattern=None)
    preflight = _generated_gpu_preflight(tmp_path, profile)
    compile(preflight, "<gpu-preflight>", "exec")
    assert "expected_pattern" not in preflight
    assert "cp.cuda.runtime.getDeviceProperties(0)" in preflight
    assert "device=name" in preflight
    assert "device_count=count" in preflight
    assert "preflight_backend=backend.name" in preflight


def test_cupy_request_is_rejected_by_cpu_profile_before_submission(tmp_path):
    request = replace(
        _pr_request(),
        backend=BackendSpec(backend="cupy", precision="float64", verbose=False),
    )
    transport = FakeTransport()
    runner = SlurmRunner(
        _config(tmp_path),
        (PR_TRANSVERSE_STATIC_OPERATION,),
        transport=transport,
        registry=default_transport_registry(),
    )
    with pytest.raises(ValueError, match="requires a GPU resource profile"):
        runner.run_registered(
            "pr", "pr_transverse_static", request, resource_profile="CPU small"
        )
    assert transport.submissions == 0


def test_generic_gpu_profile_and_setup_commands_are_profile_driven(tmp_path):
    generic = SlurmResourceProfile(
        "Generic GPU",
        "gpu",
        "normal",
        "00:15:00",
        2,
        16,
        gpus=1,
        setup_commands=("module load cuda",),
        gres="gpu:1",
        require_cupy=True,
        minimum_device_count=1,
    )
    runner = SlurmRunner(
        _config(
            tmp_path,
            resource_profiles=(generic,),
            default_resource_profile="Generic GPU",
        ),
        (PR_TRANSVERSE_STATIC_OPERATION,),
        transport=FakeTransport(),
        registry=default_transport_registry(),
    )
    script = runner._script("/runs/test", generic)
    assert "#SBATCH --gres=gpu:1" in script
    assert "module load cuda" in script
    assert "H200" not in script
    assert "execution_provenance.json" in script


def test_shared_runner_has_no_universal_h200_request_or_assertion():
    source = inspect.getsource(SlurmRunner)
    assert "gpu:h200" not in source.lower()
    assert "h200" not in source.lower()
    assert "--query-gpu=name" not in source


def test_unregistered_workflow_is_rejected_without_fallback(tmp_path):
    runner = SlurmRunner(
        _config(tmp_path), (LC_STATIC_OPERATION,), transport=FakeTransport(),
        registry=default_transport_registry(),
    )
    with pytest.raises(KeyError, match="not registered"):
        runner.run_registered("lc", "timedependent", _request())
