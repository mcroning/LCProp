"""LC-owned in-memory result orchestration for GUI continuation sources."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import uuid

import numpy as np


class ExperimentFamily(str, Enum):
    STANDARD = "standard"
    SOLITON = "soliton"


class WorkflowKind(str, Enum):
    STATIC = "static"
    TIMEDEPENDENT = "timedependent"
    SOLITON = "soliton"
    SOLITON_EXISTENCE = "soliton_existence"


class SolitonSourceKind(str, Enum):
    SINGLE = "single"
    SWEEP_MEMBER = "existence_member"


class TDInitialSourceMode(Enum):
    BEAM_LAUNCH = "beam_launch"
    STANDARD_STATIC = "standard_static"
    SOLITON_RESULT = "soliton_result"


@dataclass(frozen=True)
class RetainedResult:
    experiment_family: ExperimentFamily
    workflow_kind: WorkflowKind
    request: Any
    result: Any
    retained_id: str = ""


@dataclass(frozen=True)
class SolitonSource:
    source_id: str
    source_kind: SolitonSourceKind
    label: str
    request: Any
    result: Any
    result_id: str
    sweep_id: str | None = None
    requested_index: int | None = None
    requested_power_mW: float | None = None
    beta: float | None = None


def _source_power_mW(request: Any) -> float | None:
    base = base_physics_request(request)
    beams = getattr(base, "beams", None)
    if beams is None:
        return None
    return float(sum(float(channel.power_mW) for channel in beams.channels))


@dataclass
class RetainedResults:
    last_standard_static_result: RetainedResult | None = None
    last_standard_td_result: RetainedResult | None = None
    last_standard_td_checkpoint: RetainedResult | None = None
    last_single_soliton_result: RetainedResult | None = None
    last_soliton_existence_result: RetainedResult | None = None
    last_standard_static_checkpoint: RetainedResult | None = None
    selected_soliton_source: SolitonSource | None = None

    @property
    def last_soliton_result(self) -> RetainedResult | None:
        return self.last_single_soliton_result

    @last_soliton_result.setter
    def last_soliton_result(self, value: RetainedResult | None) -> None:
        self.last_single_soliton_result = value

    def store(
        self,
        *,
        family: ExperimentFamily,
        workflow: WorkflowKind,
        request: Any,
        result: Any,
    ) -> RetainedResult:
        retained = RetainedResult(
            family,
            workflow,
            request,
            result,
            uuid.uuid4().hex,
        )
        if workflow is WorkflowKind.STATIC:
            self.last_standard_static_result = retained
        elif workflow is WorkflowKind.TIMEDEPENDENT:
            self.last_standard_td_result = retained
        elif workflow is WorkflowKind.SOLITON:
            self.last_single_soliton_result = retained
        elif workflow is WorkflowKind.SOLITON_EXISTENCE:
            self.last_soliton_existence_result = retained
        return retained

    def soliton_sources(self) -> list[SolitonSource]:
        sources: list[SolitonSource] = []
        single = self.last_single_soliton_result
        if single is not None:
            single_status = getattr(single.result, "status", "completed")
            usable = bool(
                getattr(
                    single.result,
                    "usable_as_initial_condition",
                    single_status == "completed",
                )
            )
            if not usable:
                single = None
        if single is not None:
            beta = getattr(single.result, "metrics", {}).get("beta")
            power = _source_power_mW(single.request)
            label = "Single soliton"
            if getattr(single.result, "status", "completed") != "completed":
                label += " (stopped)"
            if power is not None:
                label += f": {power:.6g} mW"
            if beta is not None:
                label += f", β={float(beta):.6g}"
            sources.append(
                SolitonSource(
                    source_id=f"single:{single.retained_id}",
                    source_kind=SolitonSourceKind.SINGLE,
                    label=label,
                    request=single.request,
                    result=single.result,
                    result_id=single.retained_id,
                    requested_power_mW=power,
                    beta=beta,
                )
            )

        sweep = self.last_soliton_existence_result
        if sweep is not None:
            sweep_id = getattr(sweep.result, "sweep_id", sweep.retained_id)
            for member in getattr(sweep.result, "members", ()):
                if member.status != "completed" or member.result is None:
                    continue
                beta = member.beta
                label = f"Sweep: {member.requested_power_mW:.6g} mW"
                if beta is not None:
                    label += f", β={float(beta):.6g}"
                sources.append(
                    SolitonSource(
                        source_id=(
                            f"sweep:{sweep_id}:{member.requested_index}"
                        ),
                        source_kind=SolitonSourceKind.SWEEP_MEMBER,
                        label=label,
                        request=getattr(member.result, "request", sweep.request),
                        result=member.result,
                        result_id=(
                            f"{sweep_id}:{member.requested_index}"
                        ),
                        sweep_id=sweep_id,
                        requested_index=member.requested_index,
                        requested_power_mW=member.requested_power_mW,
                        beta=beta,
                    )
                )
        return sources


def base_physics_request(request: Any) -> Any:
    """Unwrap workflow requests to their grid/material/beam-bearing request."""

    current = request
    while hasattr(current, "base"):
        current = current.base
    return current


def source_incompatibility(request: Any, source_request: Any) -> str | None:
    """Return the first incompatible continuation-source field."""

    current = base_physics_request(request)
    source = base_physics_request(source_request)
    current_grid = getattr(current, "grid", None)
    source_grid = getattr(source, "grid", None)
    for field_name in ("Nx", "Ny", "x_aperture_um", "y_aperture_um"):
        if getattr(current_grid, field_name, None) != getattr(
            source_grid, field_name, None
        ):
            return f"grid.{field_name}"
    for field_name in ("material", "bias", "runtime"):
        if getattr(current, field_name, None) != getattr(source, field_name, None):
            return field_name
    current_beams = getattr(current, "beams", None)
    source_beams = getattr(source, "beams", None)
    if current_beams is None or source_beams is None:
        return "beam representation"
    if len(current_beams.channels) != len(source_beams.channels):
        return "beam channel count"
    if current_beams.coherence_groups != source_beams.coherence_groups:
        return "beam coherence groups"
    for index, (current_channel, source_channel) in enumerate(
        zip(current_beams.channels, source_beams.channels)
    ):
        if current_channel.wavelength_um != source_channel.wavelength_um:
            return f"beam channel {index} wavelength"
    return None


def soliton_result_incompatibility(
    request: Any,
    source_result: Any,
) -> str | None:
    base = base_physics_request(request)
    nx = int(base.grid.Nx)
    ny = int(base.grid.Ny)
    channel_count = len(base.beams.channels)
    A = getattr(source_result, "A", None)
    theta = getattr(source_result, "theta", None)
    if A is None or tuple(A.shape) != (channel_count, nx, ny):
        return "soliton field shape"
    if theta is None or tuple(theta.shape[-2:]) != (nx, ny):
        return "soliton director shape"
    precision = getattr(base.runtime, "precision", "float64")
    if precision == "float64":
        if np.dtype(A.dtype).itemsize < np.dtype(np.complex128).itemsize:
            return "soliton field precision"
        if np.dtype(theta.dtype).itemsize < np.dtype(np.float64).itemsize:
            return "soliton director precision"
    return None


__all__ = [
    "ExperimentFamily",
    "RetainedResult",
    "RetainedResults",
    "SolitonSource",
    "SolitonSourceKind",
    "TDInitialSourceMode",
    "WorkflowKind",
    "base_physics_request",
    "source_incompatibility",
    "soliton_result_incompatibility",
]
