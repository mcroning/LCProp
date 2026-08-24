"""Shared Qt controls and formatting for local versus remote execution."""

from PySide6.QtWidgets import QComboBox

from lcprop.transport.status import RemoteRunStatus


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


def remote_status_text(status: RemoteRunStatus) -> str:
    parts = ["Runner: Slurm", f"State: {status.state.value.replace('_', ' ').title()}"]
    if status.remote_job_id:
        parts.append(f"Job ID: {status.remote_job_id}")
    parts.append(f"Backend requested: {status.scientific_backend_requested}")
    if status.scientific_backend_resolved != "unresolved":
        parts.append(f"Backend resolved: {status.scientific_backend_resolved}")
    if status.device_summary:
        device = status.device_summary.get("device") or status.device_summary.get("name")
        if device:
            parts.append(f"Device: {device}")
    return "; ".join(parts)


__all__ = ["execution_target_selector", "remote_status_text"]
