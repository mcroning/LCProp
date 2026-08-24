"""Material-neutral, non-submitting validation of one cluster profile."""

from __future__ import annotations

from dataclasses import dataclass
import re
import shlex
import subprocess
from time import monotonic
from typing import Callable
from uuid import uuid4

from lcprop.runners.cluster_profiles import ClusterProfile
from lcprop.runners.slurm import RemoteTransport, SubprocessRemoteTransport


_DEFAULT_PROBE_TIMEOUT_SECONDS = 7.5
_ENVIRONMENT_PROBE_TIMEOUT_SECONDS = 30.0


class _ConnectionProbeCancelled(RuntimeError):
    pass


class ConnectionTestRemoteTransport(SubprocessRemoteTransport):
    """SSH transport with a bounded command time for interactive GUI probes."""

    def __init__(
        self, *, timeout_seconds: float = _DEFAULT_PROBE_TIMEOUT_SECONDS
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("connection-test timeout must be positive")
        self.timeout_seconds = float(timeout_seconds)

    def _run(self, arguments: list[str]) -> str:
        return self._run_bounded(arguments, timeout_seconds=self.timeout_seconds)

    @staticmethod
    def _stop(process: subprocess.Popen) -> None:
        if process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
        try:
            process.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()

    @classmethod
    def _run_bounded(
        cls,
        arguments: list[str],
        *,
        timeout_seconds: float,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> str:
        cancelled = cancellation_check or (lambda: False)
        if cancelled():
            raise _ConnectionProbeCancelled("connection test cancelled")
        process = subprocess.Popen(
            arguments,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = monotonic() + timeout_seconds
        while True:
            if cancelled():
                cls._stop(process)
                raise _ConnectionProbeCancelled("connection test cancelled")
            remaining = deadline - monotonic()
            if remaining <= 0:
                cls._stop(process)
                raise subprocess.TimeoutExpired(arguments, timeout_seconds)
            try:
                stdout, stderr = process.communicate(timeout=min(0.05, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise subprocess.CalledProcessError(
                process.returncode,
                arguments,
                output=stdout,
                stderr=stderr,
            )
        return stdout.strip()

    def ssh_with_timeout(
        self,
        host: str,
        *arguments: str,
        timeout_seconds: float,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> str:
        """Run one cancellable SSH probe with an operation-specific timeout."""

        return self._run_bounded(
            ["ssh", "-o", "BatchMode=yes", host, *arguments],
            timeout_seconds=timeout_seconds,
            cancellation_check=cancellation_check,
        )


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
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"timed out after {float(exc.timeout):g} seconds"
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


def _login_shell_command(commands: tuple[str, ...]) -> str:
    """Return one SSH command with the trusted payload isolated in ``$1``."""

    payload = "; ".join(commands)
    return shlex.join(
        (
            "bash",
            "-lc",
            'eval "$1"',
            "lcprop-connection-test",
            payload,
        )
    )


class ClusterConnectionTester:
    """Run harmless SSH/Python/path probes without invoking the scheduler."""

    def __init__(self, transport: RemoteTransport | None = None) -> None:
        self._transport = (
            ConnectionTestRemoteTransport() if transport is None else transport
        )

    def _ssh(
        self,
        host: str,
        *arguments: str,
        timeout_seconds: float,
        cancellation_check: Callable[[], bool],
    ) -> str:
        bounded = getattr(self._transport, "ssh_with_timeout", None)
        if callable(bounded):
            return bounded(
                host,
                *arguments,
                timeout_seconds=timeout_seconds,
                cancellation_check=cancellation_check,
            )
        return self._transport.ssh(host, *arguments)

    def _check(
        self,
        name: str,
        host: str,
        *arguments: str,
        timeout_seconds: float = _DEFAULT_PROBE_TIMEOUT_SECONDS,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> ConnectionCheck:
        cancelled = cancellation_check or (lambda: False)
        try:
            output = self._ssh(
                host,
                *arguments,
                timeout_seconds=timeout_seconds,
                cancellation_check=cancelled,
            )
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
            self._ssh(
                host,
                "mkdir",
                probe,
                timeout_seconds=_DEFAULT_PROBE_TIMEOUT_SECONDS,
                cancellation_check=cancellation_check,
            )
            created = True
            if cancellation_check():
                return ConnectionCheck(name, False, "connection test cancelled")
            self._ssh(
                host,
                "test",
                "-d",
                probe,
                timeout_seconds=_DEFAULT_PROBE_TIMEOUT_SECONDS,
                cancellation_check=cancellation_check,
            )
        except Exception as exc:
            return ConnectionCheck(name, False, _failure_detail(exc))
        finally:
            if created:
                try:
                    self._ssh(
                        host,
                        "rmdir",
                        probe,
                        timeout_seconds=_DEFAULT_PROBE_TIMEOUT_SECONDS,
                        cancellation_check=lambda: False,
                    )
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
            self._check("SSH", cluster.host, "true", cancellation_check=cancelled),
        ]
        if checks[0].passed:
            for command in ("sbatch", "squeue", "sacct"):
                if cancelled():
                    break
                checks.append(
                    self._check(
                        f"Slurm {command}",
                        cluster.host,
                        "command",
                        "-v",
                        command,
                        cancellation_check=cancelled,
                    )
                )
            if not cancelled():
                checks.append(
                    self._check(
                        "Remote Python",
                        cluster.host,
                        cluster.remote_python,
                        "--version",
                        cancellation_check=cancelled,
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
                    probe = shlex.join(
                        (
                            cluster.remote_python,
                            "-c",
                            "import cupy; print(cupy.__version__)",
                        )
                    )
                    checks.append(
                        self._check(
                            "CuPy import",
                            cluster.host,
                            _login_shell_command((*profile.setup_commands, probe)),
                            timeout_seconds=_ENVIRONMENT_PROBE_TIMEOUT_SECONDS,
                            cancellation_check=cancelled,
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
