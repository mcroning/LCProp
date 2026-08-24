import ast
from dataclasses import replace
import inspect
import json
from pathlib import Path

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
    CPU_SMALL, H200_SMALL, RemoteExecutionError, RemoteRunCancelled,
    SlurmExecutionConfig, SlurmResourceProfile, SlurmRunner,
)
from lcprop.pr.specs import PRMaterialSpec
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


@pytest.mark.parametrize("cancel_while", ("pending", "running"))
def test_cancellation_is_idempotent_monotonic_and_terminal(tmp_path, cancel_while):
    polls = (
        ("PENDING|0:0", "CANCELLED|0:15")
        if cancel_while == "pending"
        else ("RUNNING|0:0", "CANCELLED|0:15")
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


class RetrievalFailureTransport(FakeTransport):
    def download(self, host, remote, local):
        raise OSError("synthetic retrieval failure")


def _run_phase_failure(tmp_path, *, transport=None, operation=None):
    states = []
    runner = SlurmRunner(
        _config(tmp_path),
        (LC_STATIC_OPERATION if operation is None else operation,),
        transport=FakeTransport() if transport is None else transport,
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


def test_slurm_script_keeps_execution_and_scientific_backend_separate(tmp_path):
    runner = SlurmRunner(
        _config(tmp_path), (LC_STATIC_OPERATION,), transport=FakeTransport(),
        registry=default_transport_registry(),
    )
    script = runner._script("/runs/test", CPU_SMALL)
    assert "lcprop.transport.executor" in script
    assert "--partition=batch" in script
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
    assert 'assert "H200" in name' in lines[preflight_index]
    assert 'backend="cupy"' in lines[preflight_index]
    assert any("gpu_memory_samples_mib.txt" in line for line in lines)


def test_unregistered_workflow_is_rejected_without_fallback(tmp_path):
    runner = SlurmRunner(
        _config(tmp_path), (LC_STATIC_OPERATION,), transport=FakeTransport(),
        registry=default_transport_registry(),
    )
    with pytest.raises(KeyError, match="not registered"):
        runner.run_registered("lc", "timedependent", _request())
