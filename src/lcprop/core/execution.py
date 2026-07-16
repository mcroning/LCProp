"""Small thread-safe execution controls shared by workflows and front ends."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Any, Literal


RunStatus = Literal[
    "idle", "running", "stopping", "stopped", "completed", "failed"
]


class CancellationToken:
    """Cooperative cancellation flag safe to set from another thread."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class RunProgress:
    """Workflow-neutral progress emitted at a safe continuation boundary."""

    workflow: str
    status: RunStatus
    completed_units: int
    total_units: int
    current_coordinate: float
    coordinate_name: str
    coordinate_unit: str
    elapsed_wall_time: float
    latest_field_state: Any | None = None
    checkpoint_available: bool = False
    message: str = ""
    diagnostics: dict[str, Any] | None = None

    # TD compatibility/accounting names remain cumulative across continuations.
    completed_step: int = 0
    total_steps: int = 0
    current_time: float = 0.0
    prior_completed_steps: int = 0
    segment_completed_steps: int = 0
    segment_total_steps: int = 0
    cumulative_completed_steps: int = 0
    segment_start_time: float = 0.0
    segment_elapsed_time: float = 0.0
    cumulative_time: float = 0.0


# Retain the public name used by the first interruptible TD implementation.
TDProgress = RunProgress


__all__ = ["CancellationToken", "RunProgress", "RunStatus", "TDProgress"]
