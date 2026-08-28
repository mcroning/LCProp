"""Material-neutral result-retention policies for remote execution."""

from __future__ import annotations

from enum import Enum


class ResultRetrievalPolicy(str, Enum):
    """Select the portable result projection returned by remote execution."""

    FAST = "fast"
    FULL = "full"


def normalize_result_policy(value: str | ResultRetrievalPolicy) -> str:
    """Return one validated serialized result-policy identifier."""

    try:
        return ResultRetrievalPolicy(value).value
    except (TypeError, ValueError) as exc:
        raise ValueError("result policy must be 'fast' or 'full'") from exc


FAST_RESULT_POLICY = ResultRetrievalPolicy.FAST.value
FULL_RESULT_POLICY = ResultRetrievalPolicy.FULL.value


__all__ = [
    "FAST_RESULT_POLICY",
    "FULL_RESULT_POLICY",
    "ResultRetrievalPolicy",
    "normalize_result_policy",
]
