from dataclasses import replace

from lcprop.runners.base import RunnerResult
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

    def run_static(self, request, **kwargs) -> RunnerResult:
        result = run_static(request, **kwargs)
        message = "Stopped locally" if result.status == "stopped" else "Completed locally"
        return RunnerResult("static", result, message)

    def continue_static(self, request, checkpoint, **kwargs) -> RunnerResult:
        result = continue_static(request, checkpoint, **kwargs)
        message = "Stopped locally" if result.status == "stopped" else "Completed locally"
        return RunnerResult("static", result, message)

    def validate_static_continuation(self, request, checkpoint) -> None:
        validate_static_continuation(request, checkpoint)


    def run_timedependent(self, request, **kwargs) -> RunnerResult:
        result = run_timedependent(request, **kwargs)
        message = "Cancelled locally" if result.status == "cancelled" else "Completed locally"
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
        message = "Cancelled locally" if result.status == "cancelled" else "Completed locally"
        return RunnerResult("timedependent", result, message)

    def validate_timedependent_continuation(self, request, checkpoint) -> None:
        validate_timedependent_continuation(request, checkpoint)

    def run_soliton(self, request) -> RunnerResult:
        seed = run_soliton(request)
        result = seed
        message = "Completed locally"

        if request.refine_transverse:
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
            )
            message = "Completed locally with transverse refinement"

        return RunnerResult("soliton", result, message)

    def run_soliton_existence(self, request) -> RunnerResult:
        return RunnerResult(
            "soliton_existence",
            run_soliton_existence(request),
            "Completed locally",
        )

    def run_parameter_sweep(self, request) -> RunnerResult:
        return RunnerResult(
            "parameter_sweep",
            run_parameter_sweep(request),
            "Completed locally",
        )
