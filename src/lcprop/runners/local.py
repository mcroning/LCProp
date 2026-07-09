from lcprop.runners.base import RunnerResult
from lcprop.workflows import (
    run_static,
    run_timedependent,
    run_soliton,
    run_soliton_existence,
    run_parameter_sweep,
)

class LocalRunner:
    name = "Local CPU"

    def run_static(self, request) -> RunnerResult:
        return RunnerResult("static", run_static(request), "Completed locally")


    def run_timedependent(self, request) -> RunnerResult:
        return RunnerResult("timedependent", run_timedependent(request), "Completed locally")

    def run_soliton(self, request) -> RunnerResult:
        return RunnerResult(
            "soliton",
            run_soliton(request),
            "Completed locally",
        )

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
