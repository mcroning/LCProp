"""Material-neutral remote execution status contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


class RemoteRunState(str, Enum):
    NOT_SUBMITTED = "not_submitted"
    SUBMITTING = "submitting"
    PENDING = "pending"
    RUNNING = "running"
    SCIENTIFICALLY_FINISHED = "scientifically_finished"
    RETRIEVING = "retrieving"
    VERIFYING = "verifying"
    RECONSTRUCTING = "reconstructing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    OUT_OF_MEMORY = "out_of_memory"
    UNKNOWN = "unknown"


_TERMINAL_STATES = {
    RemoteRunState.COMPLETED,
    RemoteRunState.FAILED,
    RemoteRunState.CANCELLED,
    RemoteRunState.TIMEOUT,
    RemoteRunState.OUT_OF_MEMORY,
}

_ALLOWED_TRANSITIONS = {
    RemoteRunState.NOT_SUBMITTED: {
        RemoteRunState.SUBMITTING,
        RemoteRunState.FAILED,
    },
    RemoteRunState.SUBMITTING: {
        RemoteRunState.PENDING,
        RemoteRunState.RUNNING,
        RemoteRunState.FAILED,
        RemoteRunState.CANCEL_REQUESTED,
    },
    RemoteRunState.PENDING: {
        RemoteRunState.RUNNING,
        RemoteRunState.FAILED,
        RemoteRunState.CANCEL_REQUESTED,
        RemoteRunState.CANCELLED,
        RemoteRunState.TIMEOUT,
        RemoteRunState.OUT_OF_MEMORY,
        RemoteRunState.UNKNOWN,
    },
    RemoteRunState.RUNNING: {
        RemoteRunState.SCIENTIFICALLY_FINISHED,
        RemoteRunState.FAILED,
        RemoteRunState.CANCEL_REQUESTED,
        RemoteRunState.CANCELLED,
        RemoteRunState.TIMEOUT,
        RemoteRunState.OUT_OF_MEMORY,
        RemoteRunState.UNKNOWN,
    },
    RemoteRunState.SCIENTIFICALLY_FINISHED: {
        RemoteRunState.RETRIEVING,
        RemoteRunState.FAILED,
    },
    RemoteRunState.RETRIEVING: {
        RemoteRunState.VERIFYING,
        RemoteRunState.FAILED,
    },
    RemoteRunState.VERIFYING: {
        RemoteRunState.RECONSTRUCTING,
        RemoteRunState.FAILED,
    },
    RemoteRunState.RECONSTRUCTING: {
        RemoteRunState.COMPLETED,
        RemoteRunState.FAILED,
    },
    RemoteRunState.CANCEL_REQUESTED: {
        RemoteRunState.CANCELLED,
        RemoteRunState.FAILED,
        RemoteRunState.SCIENTIFICALLY_FINISHED,
        RemoteRunState.TIMEOUT,
        RemoteRunState.OUT_OF_MEMORY,
    },
    RemoteRunState.UNKNOWN: {
        RemoteRunState.PENDING,
        RemoteRunState.RUNNING,
        RemoteRunState.SCIENTIFICALLY_FINISHED,
        RemoteRunState.FAILED,
        RemoteRunState.CANCELLED,
        RemoteRunState.TIMEOUT,
        RemoteRunState.OUT_OF_MEMORY,
    },
}


@dataclass(frozen=True)
class RemoteRunStatus:
    """Immutable runner-to-GUI status update without Qt dependencies."""

    run_id: str
    execution_target: str
    state: RemoteRunState = RemoteRunState.NOT_SUBMITTED
    state_message: str = ""
    remote_job_id: str | None = None
    scheduler_state: str | None = None
    scientific_result_status: str | None = None
    scientific_backend_requested: str = "numpy"
    scientific_backend_resolved: str = "unresolved"
    device_summary: dict[str, Any] | None = None
    resource_profile: str | None = None
    submitted_at: str | None = None
    started_at: str | None = None
    scientific_finished_at: str | None = None
    retrieval_started_at: str | None = None
    completed_at: str | None = None
    progress_metadata: dict[str, Any] | None = None
    failure_reason: str | None = None
    remote_artifact_location: str | None = None
    local_artifact_location: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        if self.execution_target not in {"local", "slurm"}:
            raise ValueError("execution_target must be 'local' or 'slurm'")
        if self.scientific_backend_requested not in {"numpy", "auto", "cupy"}:
            raise ValueError("invalid requested scientific backend")
        if self.scientific_backend_resolved not in {
            "unresolved",
            "numpy",
            "cupy",
        }:
            raise ValueError("invalid resolved scientific backend")

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    @property
    def gui_product_ready(self) -> bool:
        """True only after verified reconstruction and product preparation."""

        return self.state == RemoteRunState.COMPLETED


def transition_remote_status(
    status: RemoteRunStatus,
    state: RemoteRunState,
    **changes: Any,
) -> RemoteRunStatus:
    """Return one validated monotonic transition; repeated updates are safe."""

    if not isinstance(status, RemoteRunStatus):
        raise TypeError("status must be a RemoteRunStatus")
    state = RemoteRunState(state)
    if state == status.state:
        return replace(status, **changes)
    if status.terminal:
        raise ValueError(f"cannot transition terminal state {status.state.value}")
    if state not in _ALLOWED_TRANSITIONS.get(status.state, set()):
        raise ValueError(
            f"invalid remote transition {status.state.value} -> {state.value}"
        )
    return replace(status, state=state, **changes)


def scheduler_state_to_remote_state(
    scheduler_state: str,
    *,
    exit_code: str | None = None,
) -> RemoteRunState:
    """Map a raw Slurm-like state to material-neutral GUI semantics."""

    raw = str(scheduler_state).strip().upper().split("+")[0]
    if raw in {"PENDING", "CONFIGURING", "RESV_DEL_HOLD"} or raw.startswith(
        "REQUEUE"
    ):
        return RemoteRunState.PENDING
    if raw in {"RUNNING", "COMPLETING"}:
        return RemoteRunState.RUNNING
    if raw == "COMPLETED":
        return (
            RemoteRunState.SCIENTIFICALLY_FINISHED
            if exit_code in (None, "0:0", "0")
            else RemoteRunState.FAILED
        )
    if raw == "CANCELLED":
        return RemoteRunState.CANCELLED
    if raw == "TIMEOUT":
        return RemoteRunState.TIMEOUT
    if raw == "OUT_OF_MEMORY":
        return RemoteRunState.OUT_OF_MEMORY
    if raw in {
        "FAILED",
        "BOOT_FAIL",
        "NODE_FAIL",
        "DEADLINE",
        "REVOKED",
        "PREEMPTED",
    }:
        return RemoteRunState.FAILED
    return RemoteRunState.UNKNOWN


__all__ = [
    "RemoteRunState",
    "RemoteRunStatus",
    "scheduler_state_to_remote_state",
    "transition_remote_status",
]
