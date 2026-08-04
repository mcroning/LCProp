from dataclasses import replace
from typing import Iterable

from lcprop.runners.base import RunnerResult, WorkflowOperation
from lcprop.workflows import (
    continue_static,
    continue_timedependent,
    run_static,
    run_timedependent,
    validate_timedependent_continuation,
    validate_static_continuation,
    run_soliton,
    run_soliton_existence,
    run_parameter_sweep,
)
from lcprop.workflows.soliton_trans import polish_soliton


class LocalRunner:
    name = "Local CPU"
    supports_parallel_sweeps = True

    def __init__(
        self,
        operations: Iterable[WorkflowOperation] = (),
    ) -> None:
        self._operations: dict[tuple[str, str], WorkflowOperation] = {}
        for operation in operations:
            self.register_operation(operation)

    @property
    def registered_operations(self) -> tuple[WorkflowOperation, ...]:
        """Return operations in deterministic registration order."""

        return tuple(self._operations.values())

    def register_operation(self, operation: WorkflowOperation) -> None:
        """Register one explicit material operation for local execution."""

        if not isinstance(operation, WorkflowOperation):
            raise TypeError("operation must be a WorkflowOperation")
        if operation.key in self._operations:
            material_id, workflow_id = operation.key
            raise ValueError(
                "operation is already registered for "
                f"material_id={material_id!r}, workflow_id={workflow_id!r}"
            )
        self._operations[operation.key] = operation

    def run_operation(
        self,
        operation: WorkflowOperation,
        request,
        **kwargs,
    ) -> RunnerResult:
        """Execute an explicit operation and prepare its shared products."""

        if not isinstance(operation, WorkflowOperation):
            raise TypeError("operation must be a WorkflowOperation")
        result = operation.run(request, **kwargs)
        run_data = operation.to_run_data(result)
        status = getattr(result, "status", "completed")
        if status == "cancelled":
            message = "Cancelled locally"
        elif status == "stopped":
            message = "Stopped locally"
        else:
            message = "Completed locally"
        return RunnerResult(
            operation.workflow_id,
            result,
            message,
            run_data=run_data,
            material_id=operation.material_id,
        )

    def run_registered(
        self,
        material_id: str,
        workflow_id: str,
        request,
        **kwargs,
    ) -> RunnerResult:
        """Execute a previously registered operation by its exact key."""

        key = (material_id, workflow_id)
        try:
            operation = self._operations[key]
        except KeyError as exc:
            raise KeyError(
                "no operation registered for "
                f"material_id={material_id!r}, workflow_id={workflow_id!r}"
            ) from exc
        return self.run_operation(operation, request, **kwargs)

    def run_static(self, request, **kwargs) -> RunnerResult:
        result = run_static(request, **kwargs)
        message = (
            "Stopped locally"
            if result.status == "stopped"
            else "Completed locally"
        )
        return RunnerResult("static", result, message)

    def continue_static(self, request, checkpoint, **kwargs) -> RunnerResult:
        result = continue_static(request, checkpoint, **kwargs)
        message = (
            "Stopped locally"
            if result.status == "stopped"
            else "Completed locally"
        )
        return RunnerResult("static", result, message)

    def validate_static_continuation(self, request, checkpoint) -> None:
        validate_static_continuation(request, checkpoint)

    def run_timedependent(self, request, **kwargs) -> RunnerResult:
        result = run_timedependent(request, **kwargs)
        message = (
            "Cancelled locally"
            if result.status == "cancelled"
            else "Completed locally"
        )
        return RunnerResult("timedependent", result, message)

    def continue_timedependent(
        self,
        request,
        checkpoint,
        additional_steps,
        **kwargs,
    ) -> RunnerResult:
        result = continue_timedependent(
            request,
            checkpoint,
            additional_steps,
            **kwargs,
        )
        message = (
            "Cancelled locally"
            if result.status == "cancelled"
            else "Completed locally"
        )
        return RunnerResult("timedependent", result, message)

    def validate_timedependent_continuation(self, request, checkpoint) -> None:
        validate_timedependent_continuation(request, checkpoint)

    def run_soliton(self, request, **kwargs) -> RunnerResult:
        seed = run_soliton(request, **kwargs)
        result = seed
        message = "Stopped locally" if seed.status == "stopped" else "Completed locally"

        if request.refine_transverse and seed.status != "stopped":
            polish_request = replace(
                request,
                theta_steps_per_outer=request.transverse_theta_steps_per_outer,
            )
            result = polish_soliton(
                polish_request,
                seed,
                max_outer=request.transverse_max_outer,
                field_mix=request.transverse_field_mix,
                theta_mix=request.transverse_theta_mix,
                **kwargs,
            )
            message = "Completed locally with transverse refinement"

        return RunnerResult("soliton", result, message)

    def run_soliton_existence(self, request) -> RunnerResult:
        return RunnerResult(
            "soliton_existence",
            run_soliton_existence(request),
            "Completed locally",
        )

    def run_parameter_sweep(self, request, **kwargs) -> RunnerResult:
        result = run_parameter_sweep(request, **kwargs)
        return RunnerResult(
            "parameter_sweep",
            result,
            "Stopped locally" if result.status == "stopped" else "Completed locally",
        )
