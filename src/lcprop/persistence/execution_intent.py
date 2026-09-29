"""Portable user intent, separate from scientific inputs and runtime observations."""
from dataclasses import asdict, dataclass
from collections.abc import Mapping

from lcprop.core.backend import BackendSpec


@dataclass(frozen=True)
class ExecutionIntent:
    target: str
    cluster_profile: str | None
    resource_profile: str | None
    requested_backend: str
    precision: str
    retrieval_policy: str

    def validate(self) -> None:
        if self.target not in ("local", "slurm"):
            raise ValueError("execution intent target must be local or slurm")
        BackendSpec(backend=self.requested_backend, precision=self.precision).validate()
        if self.retrieval_policy not in ("fast", "full"):
            raise ValueError("invalid execution intent retrieval policy")
        for value in (self.cluster_profile, self.resource_profile):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError("execution profile identity must be a nonempty name")
        if self.target == "slurm" and not (self.cluster_profile and self.resource_profile):
            raise ValueError("Slurm intent requires cluster and resource profile identities")
        if self.target == "local" and (self.cluster_profile or self.resource_profile):
            raise ValueError("Local intent must not name remote profiles")

    def to_payload(self):
        self.validate()
        return asdict(self)

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, Mapping) or set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("invalid execution intent fields")
        result = cls(**payload)
        result.validate()
        return result


def validate_execution_intent(intent, request):
    if intent is None:
        return None
    if not isinstance(intent, ExecutionIntent):
        intent = ExecutionIntent.from_payload(intent)
    intent.validate()
    scientific = getattr(request, "base_request", request)
    if (intent.requested_backend, intent.precision) != (
        scientific.backend.backend, scientific.backend.precision,
    ):
        raise ValueError("execution intent backend/precision differs from scientific request")
    return intent
