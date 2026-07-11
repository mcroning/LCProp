from dataclasses import replace

from lcprop.runners.base import RunnerResult
from lcprop.workflows import (
    run_static,
    run_timedependent,
    run_soliton,
    run_soliton_existence,
    run_parameter_sweep,
)
from lcprop.workflows.soliton_trans import polish_soliton

class LocalRunner:
    name = "Local CPU"

    def run_static(self, request) -> RunnerResult:
        return RunnerResult("static", run_static(request), "Completed locally")


    def run_timedependent(self, request) -> RunnerResult:
        return RunnerResult("timedependent", run_timedependent(request), "Completed locally")

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
