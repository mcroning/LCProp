from lcprop.runners.base import RunnerResult
from lcprop.workflows import run_static, run_timedependent

class LocalRunner:
    name = "Local CPU"

    def run_static(self, request) -> RunnerResult:
        return RunnerResult("static", run_static(request), "Completed locally")


    def run_timedependent(self, request) -> RunnerResult:
        return RunnerResult("timedependent", run_timedependent(request), "Completed locally")
