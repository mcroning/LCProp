from typing import Any, Callable, Iterable

from lcprop.runners.base import RunnerResult, WorkflowOperation


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
        *args,
        _prepare_products: bool = True,
        _before_product_conversion: (
            Callable[[WorkflowOperation, Any], None] | None
        ) = None,
        **kwargs,
    ) -> RunnerResult:
        """Execute an explicit operation and prepare its shared products."""

        if not isinstance(operation, WorkflowOperation):
            raise TypeError("operation must be a WorkflowOperation")
        result = operation.run(request, *args, **kwargs)
        if _prepare_products and _before_product_conversion is not None:
            _before_product_conversion(operation, result)
        run_data = operation.to_run_data(result) if _prepare_products else None
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
        """Compatibility-only LC static entry point."""

        from lcprop.lc.operations import LC_STATIC_OPERATION

        return self._run_lc_compatibility(LC_STATIC_OPERATION, request, **kwargs)

    def continue_static(self, request, checkpoint, **kwargs) -> RunnerResult:
        """Compatibility-only LC static continuation entry point."""

        from lcprop.lc.operations import LC_CONTINUE_STATIC_OPERATION

        return self._run_lc_compatibility(
            LC_CONTINUE_STATIC_OPERATION,
            request,
            checkpoint,
            **kwargs,
        )

    def validate_static_continuation(self, request, checkpoint) -> None:
        """Compatibility-only LC continuation validation."""

        from lcprop.lc.workflows import validate_static_continuation

        validate_static_continuation(request, checkpoint)

    def run_timedependent(self, request, **kwargs) -> RunnerResult:
        """Compatibility-only LC time-dependent entry point."""

        from lcprop.lc.operations import LC_TIMEDEPENDENT_OPERATION

        return self._run_lc_compatibility(
            LC_TIMEDEPENDENT_OPERATION,
            request,
            **kwargs,
        )

    def continue_timedependent(
        self,
        request,
        checkpoint,
        additional_steps,
        **kwargs,
    ) -> RunnerResult:
        """Compatibility-only LC time-dependent continuation entry point."""

        from lcprop.lc.operations import LC_CONTINUE_TIMEDEPENDENT_OPERATION

        return self._run_lc_compatibility(
            LC_CONTINUE_TIMEDEPENDENT_OPERATION,
            request,
            checkpoint,
            additional_steps,
            **kwargs,
        )

    def validate_timedependent_continuation(self, request, checkpoint) -> None:
        """Compatibility-only LC continuation validation."""

        from lcprop.lc.workflows import validate_timedependent_continuation

        validate_timedependent_continuation(request, checkpoint)

    def run_soliton(self, request, **kwargs) -> RunnerResult:
        """Compatibility-only LC soliton entry point."""

        from lcprop.lc.operations import LC_SOLITON_OPERATION

        runner_result = self._run_lc_compatibility(
            LC_SOLITON_OPERATION,
            request,
            **kwargs,
        )
        if request.refine_transverse and runner_result.result.status != "stopped":
            return RunnerResult(
                runner_result.kind,
                runner_result.result,
                "Completed locally with transverse refinement",
            )
        return runner_result

    def run_soliton_existence(self, request) -> RunnerResult:
        """Compatibility-only LC soliton-existence entry point."""

        from lcprop.lc.operations import LC_SOLITON_EXISTENCE_OPERATION

        return self._run_lc_compatibility(
            LC_SOLITON_EXISTENCE_OPERATION,
            request,
        )

    def run_parameter_sweep(self, request, **kwargs) -> RunnerResult:
        """Compatibility-only LC parameter-sweep entry point."""

        from lcprop.lc.operations import LC_PARAMETER_SWEEP_OPERATION

        return self._run_lc_compatibility(
            LC_PARAMETER_SWEEP_OPERATION,
            request,
            **kwargs,
        )

    def _run_lc_compatibility(
        self,
        operation: WorkflowOperation,
        request,
        *args,
        **kwargs,
    ) -> RunnerResult:
        """Delegate a historical LC method through canonical execution.

        Legacy methods historically omitted prepared products and material
        identity. Preserve that public result shape while sharing all workflow
        execution, cancellation, progress, errors, and status handling with
        :meth:`run_operation`.
        """

        canonical = self.run_operation(
            operation,
            request,
            *args,
            _prepare_products=False,
            **kwargs,
        )
        return RunnerResult(
            canonical.kind,
            canonical.result,
            canonical.message,
        )
