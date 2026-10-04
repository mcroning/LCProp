"""Material-neutral result-retention policies for remote execution."""

from __future__ import annotations

from enum import Enum


class ResultRetrievalPolicy(str, Enum):
    """Select the portable result projection returned by remote execution."""

    FAST = "fast"
    FULL = "full"
    INTERACTIVE = "interactive"
    ANALYSIS = "analysis"


def normalize_result_policy(value: str | ResultRetrievalPolicy) -> str:
    """Return one validated serialized result-policy identifier."""

    if isinstance(value, str) and value.startswith("analysis:"):
        names = value.split(":", 1)[1].split(",")
        if not names or len(set(names)) != len(names) or not set(names) <= set(ANALYSIS_PRODUCTS):
            raise ValueError("invalid Analysis product selection")
        return "analysis:" + ",".join(sorted(names))
    try:
        return ResultRetrievalPolicy(value).value
    except (TypeError, ValueError) as exc:
        raise ValueError("result policy must be fast, full, interactive, or analysis with valid product names") from exc


FAST_RESULT_POLICY = ResultRetrievalPolicy.FAST.value
FULL_RESULT_POLICY = ResultRetrievalPolicy.FULL.value


__all__ = [
    "FAST_RESULT_POLICY",
    "FULL_RESULT_POLICY",
    "ResultRetrievalPolicy",
    "normalize_result_policy",
]

# Serialized operational selection; never part of a scientific request.
ANALYSIS_PRODUCTS = ("unified_intensity_volume", "unified_potential_volume", "unified_carrier_volume", "unified_optical_field_volume", "unified_face_x_volume", "unified_face_y_volume", "input_intensity", "output_intensity", "far_field_intensity", "complex_input", "complex_output")


def static_product_selection(value):
    policy = normalize_result_policy(value)
    if policy == "full":
        return "full", ()
    if policy.startswith("analysis:"):
        return "analysis", tuple(policy.split(":", 1)[1].split(","))
    if policy == "analysis":
        return "analysis", ()
    return "interactive", ()  # Historical fast intent, new execution only.
