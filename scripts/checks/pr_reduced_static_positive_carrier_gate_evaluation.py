#!/usr/bin/env python3
"""Evaluate a strict positive-carrier Newton candidate gate without production changes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from lcprop.pr.evolution import hopping_rhs, periodic_derivatives_x
from lcprop.pr.static import (
    PRStaticIterationRecord,
    PRStaticResult,
    batched_newton_direction,
    fixed_intensity_jacobian_rows,
    solve_pr_static_intensity_batched,
)
from scripts.checks.pr_reduced_static_max_residual_guard_evaluation import (
    GuardFixture,
    evaluation_fixtures,
)


@dataclass(frozen=True)
class CarrierVariantSolve:
    """One solver result with candidate and accepted-state carrier telemetry."""

    result: PRStaticResult
    trials: tuple[dict[str, Any], ...]
    accepted_carrier_minima: tuple[float, ...]
    wall_time_s: float


def _metrics(residual: np.ndarray) -> tuple[float, float]:
    return (
        float(np.sqrt(np.mean(residual * residual))),
        float(np.max(np.abs(residual))),
    )


def _carrier_minimum(field: np.ndarray, *, dx_normalized: float) -> float:
    derivative, _ = periodic_derivatives_x(
        field,
        dx_normalized=dx_normalized,
        xp=np,
    )
    return float(np.min(1.0 + derivative))


def solve_variant(
    fixture: GuardFixture,
    *,
    enforce_positive_carrier: bool,
) -> CarrierVariantSolve:
    """Run production Newton algebra with one harness-only admissibility switch."""

    options = fixture.options
    options.validate()
    intensity = np.asarray(fixture.intensity, dtype=np.float64).copy()
    state = np.asarray(fixture.initial_E, dtype=np.float64).copy()
    if intensity.shape != state.shape or intensity.ndim not in (2, 3):
        raise ValueError("fixture intensity and initial_E must have one valid shape")

    def residual_at(candidate: np.ndarray) -> np.ndarray:
        return hopping_rhs(
            candidate,
            intensity,
            applied_field=fixture.applied_field,
            background_intensity=fixture.background_intensity,
            dx_normalized=fixture.dx_normalized,
            xp=np,
        )

    started = perf_counter()
    trials: list[dict[str, Any]] = []
    records: list[PRStaticIterationRecord] = []
    accepted_carriers = [
        _carrier_minimum(state, dx_normalized=fixture.dx_normalized)
    ]
    residual = residual_at(state)
    residual_rms, residual_max = _metrics(residual)
    records.append(PRStaticIterationRecord(0, residual_rms, residual_max, 0.0))

    for iteration in range(int(options.max_iterations) + 1):
        if (
            residual_rms <= float(options.residual_rms_tolerance)
            and residual_max <= float(options.residual_max_tolerance)
        ):
            result = PRStaticResult(
                E=state.copy(),
                converged=True,
                status="converged",
                iterations=iteration,
                residual_rms=residual_rms,
                residual_max=residual_max,
                records=tuple(records),
                message="full discrete PR residual converged",
            )
            break
        if iteration == int(options.max_iterations):
            result = PRStaticResult(
                E=state.copy(),
                converged=False,
                status="max_iterations",
                iterations=int(options.max_iterations),
                residual_rms=residual_rms,
                residual_max=residual_max,
                records=tuple(records),
                message="maximum Newton iterations reached before residual convergence",
            )
            break

        lower, diagonal, upper = fixed_intensity_jacobian_rows(
            state,
            intensity,
            dx_normalized=fixture.dx_normalized,
        )
        try:
            direction = batched_newton_direction(
                residual,
                lower,
                diagonal,
                upper,
                xp=np,
            )
        except np.linalg.LinAlgError:
            result = PRStaticResult(
                E=state.copy(),
                converged=False,
                status="singular_jacobian",
                iterations=iteration,
                residual_rms=residual_rms,
                residual_max=residual_max,
                records=tuple(records),
                message="fixed-intensity PR Jacobian is singular",
            )
            break

        step_scale = 1.0
        accepted = False
        for backtrack in range(int(options.max_backtracks) + 1):
            candidate = state + step_scale * direction
            candidate_carrier_minimum = _carrier_minimum(
                candidate,
                dx_normalized=fixture.dx_normalized,
            )
            carrier_admissible = (
                math.isfinite(candidate_carrier_minimum)
                and candidate_carrier_minimum > 0.0
            )
            candidate_residual = residual_at(candidate)
            candidate_rms, candidate_max = _metrics(candidate_residual)
            required_rms = (
                1.0 - float(options.armijo_fraction) * step_scale
            ) * residual_rms
            finite = bool(np.all(np.isfinite(candidate))) and math.isfinite(
                candidate_rms
            )
            production_accepts = finite and candidate_rms <= required_rms
            carrier_gate_rejects = (
                enforce_positive_carrier and not carrier_admissible
            )
            accepted = production_accepts and not carrier_gate_rejects
            trials.append(
                {
                    "iteration": iteration,
                    "backtrack": backtrack,
                    "step_scale": step_scale,
                    "residual_before_rms": residual_rms,
                    "residual_before_max": residual_max,
                    "candidate_residual_rms": candidate_rms,
                    "candidate_residual_max": candidate_max,
                    "required_residual_rms": required_rms,
                    "candidate_carrier_minimum": candidate_carrier_minimum,
                    "carrier_admissible": carrier_admissible,
                    "finite": finite,
                    "production_accepts": production_accepts,
                    "carrier_gate_rejects": carrier_gate_rejects,
                    "accepted": accepted,
                }
            )
            if accepted:
                break
            step_scale *= 0.5
            if step_scale < float(options.minimum_step_scale):
                break

        if not accepted:
            result = PRStaticResult(
                E=state.copy(),
                converged=False,
                status="line_search_failed",
                iterations=iteration,
                residual_rms=residual_rms,
                residual_max=residual_max,
                records=tuple(records),
                message="damped Newton step did not satisfy candidate acceptance",
            )
            break

        state = candidate
        residual = candidate_residual
        residual_rms = candidate_rms
        residual_max = candidate_max
        accepted_carriers.append(candidate_carrier_minimum)
        records.append(
            PRStaticIterationRecord(
                iteration + 1,
                residual_rms,
                residual_max,
                step_scale,
            )
        )

    for index, trial in enumerate(trials):
        if not trial["carrier_gate_rejects"]:
            continue
        later_accepted = next(
            (
                later
                for later in trials[index + 1 :]
                if later["iteration"] == trial["iteration"] and later["accepted"]
            ),
            None,
        )
        trial["smaller_admissible_candidate_accepted"] = later_accepted is not None
        trial["accepted_backtracked_step_scale"] = (
            later_accepted["step_scale"] if later_accepted is not None else None
        )

    return CarrierVariantSolve(
        result=result,
        trials=tuple(trials),
        accepted_carrier_minima=tuple(accepted_carriers),
        wall_time_s=perf_counter() - started,
    )


def _field_sha256(field: np.ndarray) -> str:
    values = np.ascontiguousarray(field)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def _variant_summary(solve: CarrierVariantSolve) -> dict[str, Any]:
    result = solve.result
    rejected = [trial for trial in solve.trials if not trial["accepted"]]
    gate_rejected = [trial for trial in solve.trials if trial["carrier_gate_rejects"]]
    candidate_minima = [trial["candidate_carrier_minimum"] for trial in solve.trials]
    return {
        "converged": result.converged,
        "termination_reason": result.status,
        "newton_iterations": result.iterations,
        "total_line_search_rejects": len(rejected),
        "carrier_gate_rejection_count": len(gate_rejected),
        "production_acceptable_carrier_gate_rejection_count": sum(
            trial["production_accepts"] for trial in gate_rejected
        ),
        "accepted_step_sizes": [record.step_scale for record in result.records[1:]],
        "residual_rms_history": [record.residual_rms for record in result.records],
        "residual_max_history": [record.residual_max for record in result.records],
        "accepted_carrier_minimum_history": list(solve.accepted_carrier_minima),
        "minimum_candidate_carrier_density": min(candidate_minima, default=None),
        "final_residual_rms": result.residual_rms,
        "final_residual_max": result.residual_max,
        "final_minimum_carrier_density": solve.accepted_carrier_minima[-1],
        "mean_E": float(np.mean(result.E)),
        "field_shape": list(result.E.shape),
        "field_dtype": str(result.E.dtype),
        "field_sha256": _field_sha256(result.E),
        "wall_time_s": solve.wall_time_s,
    }


def evaluate_fixture(fixture: GuardFixture) -> dict[str, Any]:
    """Compare production acceptance and the positive-carrier candidate gate."""

    original = solve_variant(fixture, enforce_positive_carrier=False)
    gated = solve_variant(fixture, enforce_positive_carrier=True)
    production = solve_pr_static_intensity_batched(
        fixture.intensity,
        applied_field=fixture.applied_field,
        background_intensity=fixture.background_intensity,
        dx_normalized=fixture.dx_normalized,
        initial_E=fixture.initial_E,
        options=fixture.options,
        xp=np,
    )
    if not np.array_equal(original.result.E, production.E):
        raise AssertionError(f"original harness field differs for {fixture.name}")
    if original.result.records != production.records:
        raise AssertionError(f"original harness history differs for {fixture.name}")
    if (
        original.result.converged != production.converged
        or original.result.status != production.status
    ):
        raise AssertionError(f"original harness status differs for {fixture.name}")

    difference = np.asarray(original.result.E) - np.asarray(gated.result.E)
    denominator = float(np.linalg.norm(gated.result.E))
    gate_rejections = [
        trial for trial in gated.trials if trial["carrier_gate_rejects"]
    ]
    return {
        "fixture": fixture.name,
        "description": fixture.description,
        "input_shape": list(fixture.intensity.shape),
        "input_minimum": float(np.min(fixture.intensity)),
        "input_maximum": float(np.max(fixture.intensity)),
        "applied_field": fixture.applied_field,
        "background_intensity": fixture.background_intensity,
        "dx_normalized": fixture.dx_normalized,
        "original": _variant_summary(original),
        "positive_carrier_gate": _variant_summary(gated),
        "carrier_gate_rejections": gate_rejections,
        "both_converged": original.result.converged and gated.result.converged,
        "final_field_bitwise_equal": bool(
            np.array_equal(original.result.E, gated.result.E)
        ),
        "final_field_relative_l2": (
            float(np.linalg.norm(difference) / denominator)
            if denominator > 0.0
            else float(np.linalg.norm(difference))
        ),
        "final_field_maximum_absolute_difference": float(
            np.max(np.abs(difference))
        ),
        "original_harness_matches_production_bitwise": True,
    }


def _policy_summary(values: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    return {
        "converged": values["converged"],
        "termination_reason": values["termination_reason"],
        "newton_iterations": values["newton_iterations"],
        "total_line_search_rejects": values[
            "total_line_search_rejects"
            if "total_line_search_rejects" in values
            else "total_line_search_backtracks"
        ],
        "final_minimum_carrier_density": values[
            "final_minimum_carrier_density"
            if "final_minimum_carrier_density" in values
            else "minimum_carrier_density"
        ],
        "field_sha256": values["field_sha256"],
        "policy": prefix,
    }


def run_evaluation(repository: Path = Path(".")) -> dict[str, Any]:
    """Run all seven fixtures and add the committed three-policy comparison."""

    rows = [evaluate_fixture(fixture) for fixture in evaluation_fixtures(repository)]
    prior_path = (
        repository
        / "results/pr_reduced_static_max_residual_guard_evaluation_2026-09-14/"
        "evaluation.json"
    )
    with prior_path.open(encoding="utf-8") as stream:
        prior_rows = {row["fixture"]: row for row in json.load(stream)}

    three_policy = []
    for row in rows[-2:]:
        prior = prior_rows[row["fixture"]]
        if row["original"]["field_sha256"] != prior["original"]["field_sha256"]:
            raise AssertionError("current production result differs from prior evidence")
        three_policy.append(
            {
                "fixture": row["fixture"],
                "original_production": _policy_summary(
                    row["original"], prefix="production RMS Armijo"
                ),
                "rejected_maximum_residual_guard": _policy_summary(
                    prior["guarded"], prefix="RMS Armijo plus nonincreasing max residual"
                ),
                "positive_carrier_gate": _policy_summary(
                    row["positive_carrier_gate"],
                    prefix="RMS Armijo plus strict positive candidate carrier",
                ),
            }
        )

    return {
        "schema_version": 1,
        "analysis": "reduced-static positive-carrier candidate gate A/B evaluation",
        "fixtures": rows,
        "difficult_case_three_policy_comparison": three_policy,
    }


def main() -> int:
    print(json.dumps(run_evaluation(), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
