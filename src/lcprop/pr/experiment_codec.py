"""Material-owned experiment request codecs for photorefractive workflows."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from lcprop.core.backend import BackendSpec
from lcprop.core.context import GridSpec
from lcprop.persistence.experiments import (
    ExperimentPayloadError,
    ExperimentRequestCodec,
    ExperimentRuntimeStateError,
    ExperimentSchemaError,
    dataclass_values,
    decode_beam_stack,
    encode_beam_stack,
    require_exact_keys,
    require_mapping,
)
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_MATERIAL_ID,
    PR_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.static import PRStaticSolverOptions
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PRStaticWorkflowOptions,
    PR_STATIC_WORKFLOW,
)
from lcprop.pr.scattering import PRCanonicalScatteringSpec
from lcprop.pr.transverse.specs import (
    PRTransverseBoundaryProfile,
    PRTransverseDielectricProfile,
    PRTransverseProjectionProfile,
    PRTransverseTransportProfile,
)
from lcprop.pr.transverse.static import (
    PRTransverseDiscreteStaticCorrectorOptions,
    PRTransverseStaticMaterialSolverOptions,
)
from lcprop.pr.transverse.static_workflow import (
    PRTransverseStaticRunRequest,
    PRTransverseStaticWorkflowOptions,
    PR_TRANSVERSE_STATIC_WORKFLOW,
)


PR_EXPERIMENT_REQUEST_SCHEMA_VERSION = 1


def _reject_runtime_state(request: PRRunRequest | PRStaticRunRequest) -> None:
    populated = tuple(
        name
        for name in ("initial_A", "initial_E")
        if getattr(request, name) is not None
    )
    if populated:
        raise ExperimentRuntimeStateError(
            "PR experiment requests cannot persist runtime state "
            f"({', '.join(populated)}); use checkpoint persistence for "
            "continuation state"
        )


def _validate_common(request: PRRunRequest | PRStaticRunRequest) -> None:
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.solver.validate()
    request.backend.validate()


def _encode_common(request: PRRunRequest | PRStaticRunRequest) -> dict:
    _reject_runtime_state(request)
    _validate_common(request)
    return {
        "schema_version": PR_EXPERIMENT_REQUEST_SCHEMA_VERSION,
        "grid": asdict(request.grid),
        "material": asdict(request.material),
        "beams": encode_beam_stack(request.beams),
        "solver": asdict(request.solver),
        "backend": asdict(request.backend),
    }


def encode_pr_timedependent_request(request: PRRunRequest) -> dict:
    if not isinstance(request, PRRunRequest):
        raise TypeError("request must be a PRRunRequest")
    return _encode_common(request)


def encode_pr_static_request(request: PRStaticRunRequest) -> dict:
    if not isinstance(request, PRStaticRunRequest):
        raise TypeError("request must be a PRStaticRunRequest")
    return _encode_common(request)


def _validate_transverse_static(request: PRTransverseStaticRunRequest) -> None:
    request.grid.validate()
    request.beams.validate()
    request.material.validate()
    request.transport.validate()
    request.dielectric.validate()
    request.boundary.validate()
    request.projection.validate()
    request.solver.validate()
    request.backend.validate()
    if request.scattering is not None:
        request.scattering.validate()


def encode_pr_transverse_static_request(
    request: PRTransverseStaticRunRequest,
) -> dict:
    """Encode canonical transverse-static inputs without runtime state."""

    if not isinstance(request, PRTransverseStaticRunRequest):
        raise TypeError("request must be a PRTransverseStaticRunRequest")
    populated = tuple(
        name
        for name in ("initial_A", "initial_psi")
        if getattr(request, name) is not None
    )
    if populated:
        raise ExperimentRuntimeStateError(
            "PR transverse-static experiment requests cannot persist runtime "
            f"state ({', '.join(populated)}); use result transport or an "
            "explicit continuation mechanism for runtime state"
        )
    _validate_transverse_static(request)
    return {
        "schema_version": PR_EXPERIMENT_REQUEST_SCHEMA_VERSION,
        "grid": asdict(request.grid),
        "material": asdict(request.material),
        "beams": encode_beam_stack(request.beams),
        "transport": asdict(request.transport),
        "dielectric": asdict(request.dielectric),
        "boundary": asdict(request.boundary),
        "projection": asdict(request.projection),
        "solver": asdict(request.solver),
        "backend": asdict(request.backend),
        "scattering": (
            None if request.scattering is None else asdict(request.scattering)
        ),
    }


def _payload(value: Any) -> dict[str, Any]:
    payload = require_mapping(value, name="PR request_payload")
    require_exact_keys(
        payload,
        required={
            "schema_version",
            "grid",
            "material",
            "beams",
            "solver",
            "backend",
        },
        name="PR request_payload",
    )
    version = payload["schema_version"]
    if type(version) is not int or version != PR_EXPERIMENT_REQUEST_SCHEMA_VERSION:
        raise ExperimentSchemaError(
            f"unsupported PR experiment request schema version: {version!r}"
        )
    return payload


def _decode_common(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "grid": GridSpec(
            **dataclass_values(
                GridSpec,
                payload["grid"],
                name="PR request_payload.grid",
            )
        ),
        "material": PRMaterialSpec(
            **dataclass_values(
                PRMaterialSpec,
                payload["material"],
                name="PR request_payload.material",
            )
        ),
        "beams": decode_beam_stack(payload["beams"]),
        "backend": BackendSpec(
            **dataclass_values(
                BackendSpec,
                payload["backend"],
                name="PR request_payload.backend",
            )
        ),
    }


def decode_pr_timedependent_request(value: Any) -> PRRunRequest:
    payload = _payload(value)
    try:
        request = PRRunRequest(
            **_decode_common(payload),
            solver=PRSolverOptions(
                **dataclass_values(
                    PRSolverOptions,
                    payload["solver"],
                    name="PR request_payload.solver",
                )
            ),
        )
        _validate_common(request)
    except ExperimentPayloadError:
        raise
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(
            f"invalid PR time-dependent request: {exc}"
        ) from exc
    return request


def decode_pr_static_request(value: Any) -> PRStaticRunRequest:
    payload = _payload(value)
    solver_values = dataclass_values(
        PRStaticWorkflowOptions,
        payload["solver"],
        name="PR request_payload.solver",
    )
    material_solver_values = solver_values.pop("material_solver")
    material_solver = None
    if material_solver_values is not None:
        material_solver = PRStaticSolverOptions(
            **dataclass_values(
                PRStaticSolverOptions,
                material_solver_values,
                name="PR request_payload.solver.material_solver",
            )
        )
    try:
        request = PRStaticRunRequest(
            **_decode_common(payload),
            solver=PRStaticWorkflowOptions(
                material_solver=material_solver,
                **solver_values,
            ),
        )
        _validate_common(request)
    except ExperimentPayloadError:
        raise
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(f"invalid PR static request: {exc}") from exc
    return request


def decode_pr_transverse_static_request(
    value: Any,
) -> PRTransverseStaticRunRequest:
    payload = require_mapping(value, name="PR transverse-static request_payload")
    require_exact_keys(
        payload,
        required={
            "schema_version",
            "grid",
            "material",
            "beams",
            "transport",
            "dielectric",
            "boundary",
            "projection",
            "solver",
            "backend",
            "scattering",
        },
        name="PR transverse-static request_payload",
    )
    version = payload["schema_version"]
    if type(version) is not int or version != PR_EXPERIMENT_REQUEST_SCHEMA_VERSION:
        raise ExperimentSchemaError(
            "unsupported PR transverse-static experiment request schema "
            f"version: {version!r}"
        )
    solver_values = dataclass_values(
        PRTransverseStaticWorkflowOptions,
        payload["solver"],
        name="PR transverse-static request_payload.solver",
    )
    material_solver_values = solver_values.pop("material_solver")
    discrete_corrector_values = solver_values.pop("discrete_corrector")
    scattering_values = payload["scattering"]
    try:
        request = PRTransverseStaticRunRequest(
            grid=GridSpec(
                **dataclass_values(
                    GridSpec,
                    payload["grid"],
                    name="PR transverse-static request_payload.grid",
                )
            ),
            beams=decode_beam_stack(payload["beams"]),
            material=PRMaterialSpec(
                **dataclass_values(
                    PRMaterialSpec,
                    payload["material"],
                    name="PR transverse-static request_payload.material",
                )
            ),
            transport=PRTransverseTransportProfile(
                **dataclass_values(
                    PRTransverseTransportProfile,
                    payload["transport"],
                    name="PR transverse-static request_payload.transport",
                )
            ),
            dielectric=PRTransverseDielectricProfile(
                **dataclass_values(
                    PRTransverseDielectricProfile,
                    payload["dielectric"],
                    name="PR transverse-static request_payload.dielectric",
                )
            ),
            boundary=PRTransverseBoundaryProfile(
                **dataclass_values(
                    PRTransverseBoundaryProfile,
                    payload["boundary"],
                    name="PR transverse-static request_payload.boundary",
                )
            ),
            projection=PRTransverseProjectionProfile(
                **dataclass_values(
                    PRTransverseProjectionProfile,
                    payload["projection"],
                    name="PR transverse-static request_payload.projection",
                )
            ),
            solver=PRTransverseStaticWorkflowOptions(
                material_solver=PRTransverseStaticMaterialSolverOptions(
                    **dataclass_values(
                        PRTransverseStaticMaterialSolverOptions,
                        material_solver_values,
                        name=(
                            "PR transverse-static request_payload.solver."
                            "material_solver"
                        ),
                    )
                ),
                discrete_corrector=PRTransverseDiscreteStaticCorrectorOptions(
                    **dataclass_values(
                        PRTransverseDiscreteStaticCorrectorOptions,
                        discrete_corrector_values,
                        name=(
                            "PR transverse-static request_payload.solver."
                            "discrete_corrector"
                        ),
                    )
                ),
                **solver_values,
            ),
            backend=BackendSpec(
                **dataclass_values(
                    BackendSpec,
                    payload["backend"],
                    name="PR transverse-static request_payload.backend",
                )
            ),
            initial_A=None,
            initial_psi=None,
            scattering=(
                None
                if scattering_values is None
                else PRCanonicalScatteringSpec(
                    **dataclass_values(
                        PRCanonicalScatteringSpec,
                        scattering_values,
                        name="PR transverse-static request_payload.scattering",
                    )
                )
            ),
        )
        _validate_transverse_static(request)
    except ExperimentPayloadError:
        raise
    except (TypeError, ValueError) as exc:
        raise ExperimentPayloadError(
            f"invalid PR transverse-static request: {exc}"
        ) from exc
    return request


PR_STATIC_EXPERIMENT_CODEC = ExperimentRequestCodec(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_STATIC_WORKFLOW,
    request_type=PRStaticRunRequest,
    encode_request=encode_pr_static_request,
    decode_request=decode_pr_static_request,
)

PR_TIMEDEPENDENT_EXPERIMENT_CODEC = ExperimentRequestCodec(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_TIMEDEPENDENT_WORKFLOW,
    request_type=PRRunRequest,
    encode_request=encode_pr_timedependent_request,
    decode_request=decode_pr_timedependent_request,
)

PR_TRANSVERSE_STATIC_EXPERIMENT_CODEC = ExperimentRequestCodec(
    material_id=PR_MATERIAL_ID,
    workflow_id=PR_TRANSVERSE_STATIC_WORKFLOW,
    request_type=PRTransverseStaticRunRequest,
    encode_request=encode_pr_transverse_static_request,
    decode_request=decode_pr_transverse_static_request,
)


__all__ = [
    "PR_EXPERIMENT_REQUEST_SCHEMA_VERSION",
    "PR_STATIC_EXPERIMENT_CODEC",
    "PR_TIMEDEPENDENT_EXPERIMENT_CODEC",
    "PR_TRANSVERSE_STATIC_EXPERIMENT_CODEC",
    "decode_pr_static_request",
    "decode_pr_timedependent_request",
    "decode_pr_transverse_static_request",
    "encode_pr_static_request",
    "encode_pr_timedependent_request",
    "encode_pr_transverse_static_request",
]
