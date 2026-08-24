"""Material-neutral Slurm orchestration for portable LCProp operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from typing import Callable, Iterable, Protocol
from uuid import uuid4

from lcprop.runners.base import RunnerResult, WorkflowOperation
from lcprop.transport.io import read_result_package, write_request_package
from lcprop.transport.envelopes import TransportVerificationError
from lcprop.transport.status import (
    RemoteRunState,
    RemoteRunStatus,
    scheduler_state_to_remote_state,
    transition_remote_status,
)


class RemoteExecutionError(RuntimeError):
    """One categorized remote-execution failure after submission."""

    def __init__(self, category: str, reason: str) -> None:
        self.category = category
        self.reason = reason
        super().__init__(f"{category}: {reason}")


class RemoteRunCancelled(RemoteExecutionError):
    """The scheduler confirmed cancellation of one submitted job."""

    def __init__(self, job_id: str) -> None:
        super().__init__("cancelled", f"Slurm job {job_id} was cancelled")


@dataclass(frozen=True)
class SlurmResourceProfile:
    name: str
    partition: str
    qos: str
    time_limit: str
    cpus: int
    memory_gb: int
    gpus: int = 0
    setup_commands: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.partition.strip() or not self.qos.strip():
            raise ValueError("resource profile identifiers must be non-empty")
        if not re.fullmatch(r"\d{1,3}:\d{2}:\d{2}", self.time_limit):
            raise ValueError("time_limit must use HH:MM:SS")
        if self.cpus < 1 or self.memory_gb < 1 or self.gpus < 0:
            raise ValueError("invalid Slurm resource quantity")


CPU_SMALL = SlurmResourceProfile("CPU small", "batch", "normal", "00:15:00", 2, 8)
H200_SMALL = SlurmResourceProfile(
    "H200 small", "gpu", "normal", "00:15:00", 2, 16, 1,
    ("module load cuda/12.9.0",),
)
H200_STANDARD = SlurmResourceProfile(
    "H200 standard", "gpu", "normal", "02:00:00", 4, 64, 1,
    ("module load cuda/12.9.0",),
)


def _remote_path(value: str) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("remote paths must be absolute and normalized")
    return str(path)


@dataclass(frozen=True)
class SlurmExecutionConfig:
    host: str
    remote_run_root: str
    remote_python: str
    remote_source_path: str
    source_git_sha: str
    local_artifact_root: Path
    resource_profiles: tuple[SlurmResourceProfile, ...] = (
        CPU_SMALL, H200_SMALL, H200_STANDARD
    )
    default_resource_profile: str | None = None
    poll_interval: float = 5.0

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+", self.host):
            raise ValueError("host must be an SSH hostname or user@hostname")
        for value in (self.remote_run_root, self.remote_python, self.remote_source_path):
            _remote_path(value)
        if not re.fullmatch(r"[0-9a-f]{40}", self.source_git_sha):
            raise ValueError("source_git_sha must be an exact 40-character SHA")
        if self.poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        names = [profile.name for profile in self.resource_profiles]
        if len(set(names)) != len(names):
            raise ValueError("resource profile names must be unique")
        if (
            self.default_resource_profile is not None
            and self.default_resource_profile not in names
        ):
            raise ValueError("default_resource_profile is not registered")

    def profile(self, name: str) -> SlurmResourceProfile:
        matches = [value for value in self.resource_profiles if value.name == name]
        if len(matches) != 1:
            raise KeyError(f"unknown Slurm resource profile {name!r}")
        return matches[0]


class RemoteTransport(Protocol):
    def ssh(self, host: str, *arguments: str) -> str: ...
    def upload(self, host: str, local: Path, remote: str) -> None: ...
    def download(self, host: str, remote: str, local: Path) -> None: ...


class SubprocessRemoteTransport:
    """Small SSH/SCP adapter; no shell interpolation or credential storage."""

    @staticmethod
    def _run(arguments: list[str]) -> str:
        completed = subprocess.run(
            arguments, check=True, text=True, capture_output=True
        )
        return completed.stdout.strip()

    def ssh(self, host: str, *arguments: str) -> str:
        return self._run(["ssh", "-o", "BatchMode=yes", host, *arguments])

    def upload(self, host: str, local: Path, remote: str) -> None:
        self._run(["scp", "-q", "-r", str(local), f"{host}:{remote}"])

    def download(self, host: str, remote: str, local: Path) -> None:
        local.mkdir(parents=True, exist_ok=True)
        self._run(["scp", "-q", "-r", f"{host}:{remote}", str(local)])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_execution_provenance(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid execution provenance: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("execution provenance must be a JSON object")
    device = value.get("device")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("execution provenance has no device identity")
    return value


class SlurmRunner:
    """Execute registered operations remotely through the transport contract."""

    name = "Slurm"
    supports_parallel_sweeps = False

    def __init__(
        self,
        config: SlurmExecutionConfig,
        operations: Iterable[WorkflowOperation] = (),
        *,
        transport: RemoteTransport | None = None,
        registry,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        operation_values = tuple(operations)
        self._operations = {operation.key: operation for operation in operation_values}
        if len(self._operations) != len(operation_values):
            raise ValueError("duplicate operation registration")
        self._transport = SubprocessRemoteTransport() if transport is None else transport
        self._registry = registry
        self._sleep = sleep

    @property
    def registered_operations(self) -> tuple[WorkflowOperation, ...]:
        return tuple(self._operations.values())

    def register_operation(self, operation: WorkflowOperation) -> None:
        if operation.key in self._operations:
            raise ValueError(f"operation already registered for {operation.key!r}")
        self._operations[operation.key] = operation

    def _script(self, remote_run: str, profile: SlurmResourceProfile) -> str:
        lines = [
            "#!/bin/bash",
            f"#SBATCH --partition={profile.partition}",
            f"#SBATCH --qos={profile.qos}",
            f"#SBATCH --time={profile.time_limit}",
            f"#SBATCH --cpus-per-task={profile.cpus}",
            f"#SBATCH --mem={profile.memory_gb}G",
            f"#SBATCH --output={remote_run}/stdout.txt",
            f"#SBATCH --error={remote_run}/stderr.txt",
        ]
        if profile.gpus:
            lines.append(f"#SBATCH --gres=gpu:h200:{profile.gpus}")
        lines.append("set -euo pipefail")
        lines.extend(profile.setup_commands)
        lines.append(f"export PYTHONPATH={self.config.remote_source_path}/src")
        if profile.gpus:
            preflight = (
                'import json,platform,cupy as cp; '
                'from lcprop.core.backend import BackendSpec,get_backend; '
                'device=cp.cuda.runtime.getDeviceProperties(0); '
                'name=device["name"].decode() if isinstance(device["name"],bytes) else str(device["name"]); '
                'backend=get_backend(BackendSpec(backend="cupy",precision="float64",verbose=False)); '
                'assert "H200" in name, f"expected NVIDIA H200, got {name}"; '
                'assert backend.name == "cupy"; '
                'print(json.dumps({"python":platform.python_version(),"cupy":cp.__version__,'
                '"cuda_runtime":cp.cuda.runtime.runtimeGetVersion(),'
                '"cuda_driver":cp.cuda.runtime.driverGetVersion(),'
                '"device":name,"device_count":cp.cuda.runtime.getDeviceCount(),'
                '"preflight_backend":backend.name},sort_keys=True))'
            )
            lines.append(
                f"{self.config.remote_python} -c '{preflight}' "
                f"> {remote_run}/execution_provenance.json"
            )
        executor = (
            f"{self.config.remote_python} -m lcprop.transport.executor "
            f"--run-dir {remote_run}"
        )
        if profile.gpus:
            lines.extend([
                "(while true; do nvidia-smi --query-compute-apps=used_gpu_memory "
                "--format=csv,noheader,nounits; sleep 0.2; done) "
                f"> {remote_run}/gpu_memory_samples_mib.txt &",
                "monitor_pid=$!",
                "set +e",
                executor,
                "payload_exit=$?",
                "set -e",
                "kill ${monitor_pid} 2>/dev/null || true",
                "wait ${monitor_pid} 2>/dev/null || true",
                "exit ${payload_exit}",
            ])
        else:
            lines.append(executor)
        return "\n".join(lines) + "\n"

    def run_registered(
        self, material_id: str, workflow_id: str, request, *,
        resource_profile: str | None = None, progress_callback=None,
        cancellation_token=None, **_ignored,
    ) -> RunnerResult:
        key = (material_id, workflow_id)
        if key not in self._operations:
            raise KeyError(f"remote operation is not registered for {key!r}")
        profile_name = resource_profile or self.config.default_resource_profile
        if profile_name is None:
            raise ValueError(
                "resource_profile is required when no configuration default exists"
            )
        profile = self.config.profile(profile_name)
        run_id = f"lcprop-{uuid4().hex}"
        local_run = self.config.local_artifact_root / run_id
        local_run.mkdir(parents=True, exist_ok=False)
        remote_run = f"{self.config.remote_run_root}/{run_id}"
        status = RemoteRunStatus(
            run_id=run_id, execution_target="slurm",
            state=RemoteRunState.SUBMITTING,
            state_message="Staging request", resource_profile=profile.name,
            scientific_backend_requested=getattr(getattr(request, "backend", None), "backend", "numpy"),
            remote_artifact_location=remote_run,
            local_artifact_location=str(local_run),
        )
        if progress_callback:
            progress_callback(status)
        actual_remote_sha = self._transport.ssh(
            self.config.host, "cat",
            f"{self.config.remote_source_path}/.lcprop-source-sha",
        )
        if actual_remote_sha != self.config.source_git_sha:
            raise RuntimeError(
                "remote source SHA mismatch: "
                f"expected {self.config.source_git_sha}, got {actual_remote_sha}"
            )
        write_request_package(
            local_run, registry=self._registry, material_id=material_id,
            workflow_id=workflow_id, request=request, run_id=run_id,
            resource_profile=profile.name,
            provenance={
                "local_git_sha": self.config.source_git_sha,
                "remote_source_path": self.config.remote_source_path,
            },
        )
        script = local_run / "launch.sbatch"
        script.write_text(self._script(remote_run, profile), encoding="utf-8")
        self._transport.ssh(self.config.host, "mkdir", "-p", remote_run)
        self._transport.upload(self.config.host, local_run / "request", remote_run)
        self._transport.upload(self.config.host, script, f"{remote_run}/launch.sbatch")
        job_id = self._transport.ssh(
            self.config.host, "sbatch", "--parsable", f"{remote_run}/launch.sbatch"
        ).split(";")[0].strip()
        if not re.fullmatch(r"\d+", job_id):
            raise RuntimeError(f"invalid sbatch job ID {job_id!r}")
        status = transition_remote_status(
            status, RemoteRunState.PENDING, remote_job_id=job_id,
            submitted_at=_utc_now(), state_message="Queued"
        )
        if progress_callback:
            progress_callback(status)

        cancellation_requested = False
        while True:
            if (
                not cancellation_requested
                and cancellation_token is not None
                and cancellation_token.is_cancelled()
            ):
                status = transition_remote_status(status, RemoteRunState.CANCEL_REQUESTED)
                if progress_callback:
                    progress_callback(status)
                self._transport.ssh(self.config.host, "scancel", job_id)
                cancellation_requested = True
            raw = self._transport.ssh(
                self.config.host, "sacct", "-n", "-X", "-j", job_id,
                "--format=State,ExitCode", "--parsable2"
            ).splitlines()
            record = next((line for line in raw if line.strip()), "PENDING|0:0")
            fields = record.split("|")
            scheduler = fields[0].strip()
            exit_code = fields[1].strip() if len(fields) > 1 else None
            mapped = scheduler_state_to_remote_state(scheduler, exit_code=exit_code)
            if status.state == RemoteRunState.CANCEL_REQUESTED and mapped in {
                RemoteRunState.PENDING,
                RemoteRunState.RUNNING,
                RemoteRunState.UNKNOWN,
            }:
                self._sleep(self.config.poll_interval)
                continue
            if (
                mapped == RemoteRunState.SCIENTIFICALLY_FINISHED
                and status.state == RemoteRunState.PENDING
            ):
                status = transition_remote_status(
                    status, RemoteRunState.RUNNING, scheduler_state="RUNNING",
                    started_at=_utc_now(), state_message="Running",
                )
                if progress_callback:
                    progress_callback(status)
            if mapped != status.state:
                status = transition_remote_status(
                    status, mapped, scheduler_state=scheduler,
                    started_at=status.started_at or (_utc_now() if mapped == RemoteRunState.RUNNING else None),
                    scientific_finished_at=_utc_now() if mapped == RemoteRunState.SCIENTIFICALLY_FINISHED else None,
                    state_message=mapped.value.replace("_", " ").title(),
                )
                if progress_callback:
                    progress_callback(status)
            if mapped == RemoteRunState.SCIENTIFICALLY_FINISHED:
                break
            if mapped == RemoteRunState.CANCELLED:
                raise RemoteRunCancelled(job_id)
            if mapped in {
                RemoteRunState.FAILED,
                RemoteRunState.TIMEOUT,
                RemoteRunState.OUT_OF_MEMORY,
            }:
                raise RemoteExecutionError(
                    "scheduler",
                    f"Slurm job {job_id} ended as {scheduler} ({exit_code})",
                )
            self._sleep(self.config.poll_interval)

        status = transition_remote_status(status, RemoteRunState.RETRIEVING, retrieval_started_at=_utc_now())
        if progress_callback:
            progress_callback(status)
        try:
            self._transport.download(self.config.host, f"{remote_run}/output", local_run)
            output_download = local_run / "output"
            nested = output_download / "output"
            if nested.is_dir():
                nested.rename(local_run / "output.retrieved")
                output_download.rmdir()
                (local_run / "output.retrieved").rename(output_download)
            execution_provenance = {}
            if profile.gpus:
                self._transport.download(
                    self.config.host,
                    f"{remote_run}/execution_provenance.json",
                    local_run,
                )
                execution_provenance = _read_execution_provenance(
                    local_run / "execution_provenance.json"
                )
        except Exception as exc:
            status = self._failed_status(
                status, "retrieval", exc, progress_callback=progress_callback
            )
            raise RemoteExecutionError("retrieval", str(exc)) from exc
        status = transition_remote_status(status, RemoteRunState.VERIFYING)
        if progress_callback:
            progress_callback(status)
        status = transition_remote_status(status, RemoteRunState.RECONSTRUCTING)
        if progress_callback:
            progress_callback(status)
        try:
            decoded = read_result_package(local_run, registry=self._registry)
        except TransportVerificationError as exc:
            status = self._failed_status(
                status, "verification", exc, progress_callback=progress_callback
            )
            raise RemoteExecutionError("verification", str(exc)) from exc
        except Exception as exc:
            status = self._failed_status(
                status, "reconstruction", exc, progress_callback=progress_callback
            )
            raise RemoteExecutionError("reconstruction", str(exc)) from exc
        operation = self._operations.get(decoded.codec.key)
        if operation is None:
            exc = ValueError(
                f"no registered product adapter for {decoded.codec.key!r}"
            )
            status = self._failed_status(
                status, "product_conversion", exc,
                progress_callback=progress_callback,
            )
            raise RemoteExecutionError("product_conversion", str(exc)) from exc
        try:
            run_data = operation.to_run_data(decoded.result)
        except Exception as exc:
            status = self._failed_status(
                status, "product_conversion", exc,
                progress_callback=progress_callback,
            )
            raise RemoteExecutionError("product_conversion", str(exc)) from exc
        runner_result = RunnerResult(
            kind=operation.workflow_id,
            result=decoded.result,
            message="Completed remotely",
            run_data=run_data,
            material_id=operation.material_id,
        )
        backend_summary = getattr(decoded.result, "backend_summary", {})
        decoded_backend = getattr(backend_summary, "get", lambda *_: "numpy")(
            "backend", "numpy"
        )
        device_summary = dict(backend_summary) if backend_summary else {}
        device_summary.update(execution_provenance)
        status = transition_remote_status(
            status, RemoteRunState.COMPLETED, completed_at=_utc_now(),
            scientific_backend_resolved=str(decoded_backend),
            device_summary=device_summary or None,
            remote_job_id=job_id, state_message="Completed"
        )
        if progress_callback:
            progress_callback(status)
        return runner_result

    @staticmethod
    def _failed_status(
        status: RemoteRunStatus,
        category: str,
        exc: Exception,
        *,
        progress_callback=None,
    ) -> RemoteRunStatus:
        reason = f"{category}: {type(exc).__name__}: {exc}"
        failed = transition_remote_status(
            status,
            RemoteRunState.FAILED,
            state_message=f"{category.replace('_', ' ').title()} failed",
            failure_reason=reason,
            progress_metadata={"failure_category": category},
        )
        if progress_callback:
            progress_callback(failed)
        return failed


__all__ = [
    "CPU_SMALL", "H200_SMALL", "H200_STANDARD", "RemoteTransport",
    "RemoteExecutionError", "RemoteRunCancelled", "SlurmExecutionConfig",
    "SlurmResourceProfile", "SlurmRunner",
    "SubprocessRemoteTransport",
]
