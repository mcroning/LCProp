"""Material-neutral, non-submitting validation of one cluster profile."""

from __future__ import annotations

from dataclasses import dataclass
import re
import shlex
import subprocess
from typing import Callable
from uuid import uuid4

from lcprop.runners.cluster_profiles import ClusterProfile
from lcprop.runners.slurm import RemoteTransport, SubprocessRemoteTransport


class ConnectionTestRemoteTransport(SubprocessRemoteTransport):
    """SSH transport with a bounded command time for interactive GUI probes."""

    def __init__(self, *, timeout_seconds: float = 7.5) -> None:
        if timeout_seconds <= 0:
            raise ValueError("connection-test timeout must be positive")
        self.timeout_seconds = float(timeout_seconds)

    def _run(self, arguments: list[str]) -> str:
        completed = subprocess.run(
            arguments,
            check=True,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
        )
        return completed.stdout.strip()


@dataclass(frozen=True)
class ConnectionCheck:
    """One independently reportable connection/environment check."""

    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ClusterConnectionResult:
    """Structured result from a connection-only cluster validation."""

    cluster_name: str
    resource_profile: str
    checks: tuple[ConnectionCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def _failure_detail(exc: Exception) -> str:
    text = str(exc).strip()
    if isinstance(exc, subprocess.CalledProcessError):
        text = (exc.stderr or exc.stdout or text).strip()
    lowered = text.lower()
    if "permission denied" in lowered or "authentication" in lowered:
        return (
            "SSH authentication is not available for this profile. Configure "
            "system SSH access or your SSH agent, then test again."
        )
    return text or type(exc).__name__


class ClusterConnectionTester:
    """Run harmless SSH/Python/path probes without invoking the scheduler."""

    def __init__(self, transport: RemoteTransport | None = None) -> None:
        self._transport = (
            ConnectionTestRemoteTransport() if transport is None else transport
        )

    def _check(self, name: str, host: str, *arguments: str) -> ConnectionCheck:
        try:
            output = self._transport.ssh(host, *arguments)
        except Exception as exc:  # structured boundary for subprocess/transport errors
            return ConnectionCheck(name, False, _failure_detail(exc))
        return ConnectionCheck(name, True, output.strip() or "available")

    def _writable_root(
        self,
        name: str,
        host: str,
        root: str,
        cancellation_check: Callable[[], bool],
    ) -> ConnectionCheck:
        probe = f"{root}/.lcprop-connection-test-{uuid4().hex}"
        created = False
        try:
            self._transport.ssh(host, "mkdir", probe)
            created = True
            if cancellation_check():
                return ConnectionCheck(name, False, "connection test cancelled")
            self._transport.ssh(host, "test", "-d", probe)
        except Exception as exc:
            return ConnectionCheck(name, False, _failure_detail(exc))
        finally:
            if created:
                try:
                    self._transport.ssh(host, "rmdir", probe)
                except Exception as exc:
                    return ConnectionCheck(
                        name,
                        False,
                        "writable probe succeeded but cleanup failed: "
                        + _failure_detail(exc),
                    )
        return ConnectionCheck(name, True, f"writable: {root}")

    def test(
        self,
        cluster: ClusterProfile,
        resource_profile: str,
        *,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> ClusterConnectionResult:
        """Test login-node prerequisites only; never call sbatch/srun/scancel."""

        profile = cluster.profile(resource_profile)
        cancelled = cancellation_check or (lambda: False)
        checks: list[ConnectionCheck] = [
            self._check("SSH", cluster.host, "true"),
        ]
        if checks[0].passed:
            for command in ("sbatch", "squeue", "sacct"):
                if cancelled():
                    break
                checks.append(
                    self._check(
                        f"Slurm {command}", cluster.host, "command", "-v", command
                    )
                )
            if not cancelled():
                checks.append(
                    self._check(
                        "Remote Python",
                        cluster.host,
                        cluster.remote_python,
                        "--version",
                    )
                )
            if not cancelled():
                checks.append(
                    self._writable_root(
                        "Remote run root",
                        cluster.host,
                        cluster.remote_run_root,
                        cancelled,
                    )
                )
            if not cancelled():
                checks.append(
                    self._writable_root(
                        "Remote source root",
                        cluster.host,
                        cluster.source_root,
                        cancelled,
                    )
                )
            if profile.require_cupy and not cancelled():
                if any(
                    re.search(r"\b(?:sbatch|srun|scancel)\b", command)
                    for command in profile.setup_commands
                ):
                    checks.append(
                        ConnectionCheck(
                            "CuPy import",
                            False,
                            "resource setup commands may not invoke scheduler jobs "
                            "during Test Connection",
                        )
                    )
                else:
                    python = shlex.quote(cluster.remote_python)
                    code = shlex.quote("import cupy; print(cupy.__version__)")
                    command = "; ".join(
                        (*profile.setup_commands, f"{python} -c {code}")
                    )
                    checks.append(
                        self._check(
                            "CuPy import",
                            cluster.host,
                            "bash",
                            "-lc",
                            shlex.quote(command),
                        )
                    )
        if cancelled():
            checks.append(ConnectionCheck("Connection test", False, "cancelled"))
        return ClusterConnectionResult(
            cluster_name=cluster.name,
            resource_profile=profile.name,
            checks=tuple(checks),
        )


__all__ = [
    "ClusterConnectionResult",
    "ClusterConnectionTester",
    "ConnectionTestRemoteTransport",
    "ConnectionCheck",
]
