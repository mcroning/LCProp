from lcprop.runners.base import RunnerResult, WorkflowOperation

__all__ = ["RunnerResult", "WorkflowOperation", "LocalRunner"]


def __getattr__(name):
    # Importing the operation protocol must not select an execution adapter.
    if name == "LocalRunner":
        from lcprop.runners.local import LocalRunner
        return LocalRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
