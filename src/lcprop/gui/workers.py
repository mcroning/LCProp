"""Qt workers for running LCProp workflows outside the GUI thread."""

from __future__ import annotations

import traceback
from typing import Callable

from PySide6.QtCore import QObject, Signal, Slot


class WorkflowWorker(QObject):
    """Invoke an existing cancellable runner in a worker thread."""

    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, runner: Callable, request, cancellation_token) -> None:
        super().__init__()
        self._runner = runner
        self._request = request
        self._cancellation_token = cancellation_token

    @Slot()
    def run(self) -> None:
        try:
            result = self._runner(
                self._request,
                cancellation_token=self._cancellation_token,
                progress_callback=self.progress.emit,
            )
        except Exception:
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(result)


# Compatibility alias for callers/tests written for the original TD-only worker.
TimeDependentWorker = WorkflowWorker


__all__ = ["TimeDependentWorker", "WorkflowWorker"]
