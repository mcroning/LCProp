from dataclasses import dataclass
from typing import Any, Callable, Protocol


RunCallable = Callable[..., Any]
ProductAdapter = Callable[[Any], Any]


@dataclass(frozen=True)
class WorkflowOperation:
    """Explicit in-tree composition of one material workflow operation."""

    material_id: str
    workflow_id: str
    run: RunCallable
    to_run_data: ProductAdapter

    def __post_init__(self) -> None:
        for name in ("material_id", "workflow_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be a non-empty, trimmed string")
        if not callable(self.run):
            raise TypeError("run must be callable")
        if not callable(self.to_run_data):
            raise TypeError("to_run_data must be callable")

    @property
    def key(self) -> tuple[str, str]:
        return self.material_id, self.workflow_id


@dataclass(frozen=True)
class RunnerResult:
    kind: str
    result: Any
    message: str = ""
    run_data: Any | None = None
    material_id: str | None = None


class Runner(Protocol):
    name: str
    supports_parallel_sweeps: bool

    def register_operation(self, operation: WorkflowOperation) -> None: ...

    def run_registered(
        self,
        material_id,
        workflow_id,
        request,
        **kwargs,
    ) -> RunnerResult: ...

    def run_static(self, request, **kwargs) -> RunnerResult: ...

    def continue_static(self, request, checkpoint, **kwargs) -> RunnerResult: ...

    def validate_static_continuation(self, request, checkpoint) -> None: ...

    def run_timedependent(self, request, **kwargs) -> RunnerResult: ...

    def continue_timedependent(
        self,
        request,
        checkpoint,
        additional_steps,
        **kwargs,
    ) -> RunnerResult: ...

    def validate_timedependent_continuation(self, request, checkpoint) -> None: ...
