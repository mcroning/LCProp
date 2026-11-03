"""In-memory checkpoint state and validation for PR material evolution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from lcprop.core.backend import asnumpy
from lcprop.pr.specs import PRRunRequest, validate_pr_timedependent_configuration


PRCheckpointStatus = Literal["completed", "cancelled"]


def _host_array(value: Any) -> np.ndarray:
    return np.asarray(asnumpy(value))


@dataclass(frozen=True)
class PRTimeDependentCheckpoint:
    """Accepted PR state at a complete normalized material-time boundary."""

    request: PRRunRequest
    E_initial: Any
    E_current: Any
    A0: Any
    completed_steps: int
    requested_steps: int
    time_normalized: float
    grid_summary: dict[str, Any]
    E_dtype: str
    A0_dtype: str
    status: PRCheckpointStatus = "completed"
    segment_lineage: tuple[dict[str, Any], ...] = ()


def validate_pr_checkpoint(checkpoint: PRTimeDependentCheckpoint) -> None:
    """Raise ``ValueError`` when an in-memory PR checkpoint is inconsistent."""

    if not isinstance(checkpoint, PRTimeDependentCheckpoint):
        raise TypeError("checkpoint must be a PRTimeDependentCheckpoint")
    validate_pr_timedependent_configuration(checkpoint.request)
    if checkpoint.status not in ("completed", "cancelled"):
        raise ValueError(f"invalid PR checkpoint status: {checkpoint.status}")
    if int(checkpoint.completed_steps) < 0:
        raise ValueError("checkpoint completed_steps must be nonnegative")
    if int(checkpoint.requested_steps) < int(checkpoint.completed_steps):
        raise ValueError("checkpoint requested_steps must be >= completed_steps")

    if checkpoint.segment_lineage:
        from lcprop.pr.illumination import INTEGRAL_NORMALIZATION, reference_from_metadata
        import json
        if checkpoint.request.material.normalization_identity != INTEGRAL_NORMALIZATION:
            raise ValueError("physical segment lineage cannot label a legacy checkpoint")
        json.dumps(checkpoint.segment_lineage, allow_nan=False)
        end = -1.0
        for segment in checkpoint.segment_lineage:
            if segment.get("identity") != "pr_reduced_accepted_E_physical_illumination_segments_v1":
                raise ValueError("unknown reduced continuation identity")
            start = float(segment["source_cumulative_time"])
            stop = float(segment["cumulative_time"])
            if start < end or stop < start or stop > checkpoint.time_normalized:
                raise ValueError("invalid reduced segment time ordering")
            reference_from_metadata(segment["old_reference"])
            reference_from_metadata(segment["new_reference"])
            end = stop

    E_initial = _host_array(checkpoint.E_initial)
    E_current = _host_array(checkpoint.E_current)
    A0 = _host_array(checkpoint.A0)
    if E_initial.ndim != 3 or E_current.ndim != 3:
        raise ValueError(
            "checkpoint E fields must have shape (Nz, Nx, Ny)"
        )
    if E_current.shape != E_initial.shape:
        raise ValueError("checkpoint E_current shape must match E_initial")
    if A0.ndim != 3:
        raise ValueError("checkpoint A0 must have shape (Nch, Nx, Ny)")
    if A0.shape[1:] != E_current.shape[1:]:
        raise ValueError("checkpoint A0 and E spatial shapes must match")
    if str(E_current.dtype) != checkpoint.E_dtype:
        raise ValueError("checkpoint E dtype metadata does not match array")
    if str(E_initial.dtype) != checkpoint.E_dtype:
        raise ValueError("checkpoint E_initial and E_current dtypes must match")
    if str(A0.dtype) != checkpoint.A0_dtype:
        raise ValueError("checkpoint A0 dtype metadata does not match array")

    expected_shape = (
        int(checkpoint.grid_summary["Nz"]),
        int(checkpoint.grid_summary["Nx"]),
        int(checkpoint.grid_summary["Ny"]),
    )
    if E_current.shape != expected_shape:
        raise ValueError("checkpoint E shape does not match saved grid")

    expected_time = int(checkpoint.completed_steps) * float(
        checkpoint.request.solver.dt_normalized
    )
    if not np.isclose(
        checkpoint.time_normalized,
        expected_time,
        rtol=1e-12,
        atol=1e-15,
    ):
        raise ValueError(
            "checkpoint normalized time is inconsistent with completed steps"
        )


def validate_pr_continuation(
    request: PRRunRequest,
    checkpoint: PRTimeDependentCheckpoint,
) -> None:
    """Raise when ``request`` cannot resume the accepted PR checkpoint."""

    validate_pr_checkpoint(checkpoint)
    original = checkpoint.request
    comparable = (
        (request.grid, original.grid, "grid"),
        (request.material, original.material, "material"),
        (request.beams, original.beams, "beams"),
        (
            request.launch_elements,
            original.launch_elements,
            "launch_elements",
        ),
        (request.backend, original.backend, "backend"),
        (request.scattering, original.scattering, "scattering"),
        (request.optical_boundary, original.optical_boundary, "optical_boundary"),
        (request.optical_coupling, original.optical_coupling, "optical_coupling"),
        (
            request.material_response,
            original.material_response,
            "material_response",
        ),
        (
            replace(request.solver, Nt=0),
            replace(original.solver, Nt=0),
            "solver",
        ),
    )
    for current, saved, label in comparable:
        if current != saved:
            raise ValueError(
                f"cannot continue PR checkpoint with incompatible {label}"
            )

    if request.initial_A is not None:
        checkpoint_A0 = _host_array(checkpoint.A0)
        supplied_A = _host_array(request.initial_A).astype(
            checkpoint_A0.dtype,
            copy=False,
        )
        if supplied_A.shape != checkpoint_A0.shape or not np.array_equal(
            supplied_A,
            checkpoint_A0,
        ):
            raise ValueError(
                "cannot continue PR checkpoint with incompatible initial_A"
            )
    if request.initial_E is not None:
        checkpoint_E_initial = _host_array(checkpoint.E_initial)
        supplied_E = _host_array(request.initial_E).astype(
            checkpoint_E_initial.dtype,
            copy=False,
        )
        if supplied_E.shape != checkpoint_E_initial.shape or not np.array_equal(
            supplied_E,
            checkpoint_E_initial,
        ):
            raise ValueError(
                "cannot continue PR checkpoint with incompatible initial_E"
            )


__all__ = [
    "PRCheckpointStatus",
    "PRTimeDependentCheckpoint",
    "validate_pr_checkpoint",
    "validate_pr_continuation",
]
