from dataclasses import dataclass
from typing import Any, Protocol

@dataclass(frozen=True)
class RunnerResult:
    kind: str
    result: Any
    message: str = ""

class Runner(Protocol):
    name: str
    def run_static(self, request) -> RunnerResult: ...
