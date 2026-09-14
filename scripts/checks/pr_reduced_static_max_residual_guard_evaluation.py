#!/usr/bin/env python3
"""Bounded A/B evaluation of the reduced-static maximum-residual guard."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch
from lcprop.pr.evolution import hopping_rhs, periodic_derivatives_x
from lcprop.pr.source import channel_peak_intensity_reference, pr_driving_intensity
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.static import (
    PRStaticIterationRecord,
    PRStaticResult,
    PRStaticSolverOptions,
    batched_newton_direction,
    fixed_intensity_jacobian_rows,
    solve_pr_static_intensity_batched,
)


@dataclass(frozen=True)
class GuardFixture:
    """One prescribed-intensity material problem used by both variants."""

    name: str
    description: str
    intensity: np.ndarray
    initial_E: np.ndarray
    applied_field: float
    background_intensity: float
    dx_normalized: float
    options: PRStaticSolverOptions


@dataclass(frozen=True)
class VariantSolve:
    """One solver result plus complete line-search telemetry."""

    result: PRStaticResult
    trials: tuple[dict[str, Any], ...]
    wall_time_s: float


def _metrics(residual: np.ndarray) -> tuple[float, float]:
    return (
        float(np.sqrt(np.mean(residual * residual))),
        float(np.max(np.abs(residual))),
    )


def solve_variant(fixture: GuardFixture, *, enforce_max_guard: bool) -> VariantSolve:
    """Run the production Newton algebra with a harness-only A/B predicate."""

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
            candidate_residual = residual_at(candidate)
            candidate_rms, candidate_max = _metrics(candidate_residual)
            required_rms = (
                1.0 - float(options.armijo_fraction) * step_scale
            ) * residual_rms
            finite = bool(np.all(np.isfinite(candidate))) and math.isfinite(
                candidate_rms
            )
            armijo_accepts = finite and candidate_rms <= required_rms
            maximum_accepts = candidate_max <= residual_max
            accepted = armijo_accepts and (
                maximum_accepts or not enforce_max_guard
            )
            trials.append(
                {
                    "iteration": iteration,
                    "backtrack": backtrack,
                    "step_scale": step_scale,
                    "residual_before_rms": residual_rms,
                    "residual_before_max": residual_max,
                    "candidate_residual_rms": candidate_rms,
                    "candidate_residual_max": candidate_max,
                    "merit_before": 0.5 * residual_rms * residual_rms,
                    "candidate_merit": 0.5 * candidate_rms * candidate_rms,
                    "required_residual_rms": required_rms,
                    "finite": finite,
                    "armijo_accepts": armijo_accepts,
                    "maximum_accepts": maximum_accepts,
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
                message="damped Newton step did not reduce residual RMS",
            )
            break

        state = candidate
        residual = candidate_residual
        residual_rms = candidate_rms
        residual_max = candidate_max
        records.append(
            PRStaticIterationRecord(
                iteration + 1,
                residual_rms,
                residual_max,
                step_scale,
            )
        )

    return VariantSolve(
        result=result,
        trials=tuple(trials),
        wall_time_s=perf_counter() - started,
    )


def _field_sha256(field: np.ndarray) -> str:
    values = np.ascontiguousarray(field)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def _variant_summary(solve: VariantSolve, fixture: GuardFixture) -> dict[str, Any]:
    result = solve.result
    derivative, _ = periodic_derivatives_x(
        result.E,
        dx_normalized=fixture.dx_normalized,
        xp=np,
    )
    accepted = [trial for trial in solve.trials if trial["accepted"]]
    rejected = [trial for trial in solve.trials if not trial["accepted"]]
    return {
        "converged": result.converged,
        "termination_reason": result.status,
        "newton_iterations": result.iterations,
        "total_line_search_backtracks": len(rejected),
        "accepted_step_sizes": [record.step_scale for record in result.records[1:]],
        "residual_rms_history": [record.residual_rms for record in result.records],
        "residual_max_history": [record.residual_max for record in result.records],
        "accepted_rms_decrease_max_increase_count": sum(
            trial["candidate_residual_rms"] < trial["residual_before_rms"]
            and trial["candidate_residual_max"] > trial["residual_before_max"]
            for trial in accepted
        ),
        "final_residual_rms": result.residual_rms,
        "final_residual_max": result.residual_max,
        "minimum_carrier_density": float(np.min(1.0 + derivative)),
        "mean_E": float(np.mean(result.E)),
        "field_shape": list(result.E.shape),
        "field_dtype": str(result.E.dtype),
        "field_sha256": _field_sha256(result.E),
        "wall_time_s": solve.wall_time_s,
    }


def evaluate_fixture(fixture: GuardFixture) -> dict[str, Any]:
    """Compare original and guarded acceptance on exactly one fixture."""

    original = solve_variant(fixture, enforce_max_guard=False)
    guarded = solve_variant(fixture, enforce_max_guard=True)
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

    left = np.asarray(original.result.E)
    right = np.asarray(guarded.result.E)
    difference = left - right
    denominator = float(np.linalg.norm(right))
    both_converged = original.result.converged and guarded.result.converged
    guard_rejections = [
        trial
        for trial in guarded.trials
        if trial["armijo_accepts"] and not trial["maximum_accepts"]
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
        "initial_minimum_carrier_density": float(
            np.min(
                1.0
                + periodic_derivatives_x(
                    fixture.initial_E,
                    dx_normalized=fixture.dx_normalized,
                    xp=np,
                )[0]
            )
        ),
        "original": _variant_summary(original, fixture),
        "guarded": _variant_summary(guarded, fixture),
        "guard_rejection_count": len(guard_rejections),
        "guard_rejections": guard_rejections,
        "both_converged": both_converged,
        "final_field_bitwise_equal": bool(np.array_equal(left, right)),
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


def periodic_fixture(
    name: str,
    *,
    modulation: float,
    kg_rad_per_um: float = math.pi,
    grid_size: int = 512,
) -> GuardFixture:
    material = PRMaterialSpec()
    length_um = 2.0 * math.pi / kg_rad_per_um
    dx_um = length_um / grid_size
    x_um = np.arange(grid_size, dtype=np.float64) * dx_um
    intensity = (1.0 + modulation * np.sin(kg_rad_per_um * x_um))[:, None]
    return GuardFixture(
        name=name,
        description=(
            "one exact periodic material grating; "
            f"kg={kg_rad_per_um:.15g} rad/um, m={modulation:.3g}"
        ),
        intensity=intensity,
        initial_E=np.zeros_like(intensity),
        applied_field=0.0,
        background_intensity=0.0,
        dx_normalized=material.characteristic_wavenumber_per_um * dx_um,
        options=PRStaticSolverOptions(
            max_iterations=80,
            residual_rms_tolerance=1.0e-12,
            residual_max_tolerance=1.0e-11,
        ),
    )


def optical_source_fixture() -> GuardFixture:
    """Return a frozen material subproblem from the optical-coupling source path."""

    grid_spec = GridSpec(
        Nx=96,
        Ny=12,
        x_aperture_um=48.0,
        y_aperture_um=24.0,
        dz_um=5.0,
        z_length_um=10.0,
    )
    grid = make_grid(grid_spec, xp=np, real_dtype=np.float64)
    carrier = 2.0 * math.pi * 3.0 / grid_spec.x_aperture_um
    beams = BeamStack(
        channels=(
            BeamChannel(
                name="pump",
                power_mW=1.0,
                waist_x_um=18.0,
                waist_y_um=10.0,
                tilt_x_rad_per_um=carrier,
                coherence_group="laser",
            ),
            BeamChannel(
                name="signal",
                power_mW=0.2,
                waist_x_um=18.0,
                waist_y_um=10.0,
                tilt_x_rad_per_um=-carrier,
                coherence_group="laser",
            ),
        )
    )
    launch = build_launch(beams, grid, complex_dtype=np.complex128)
    reference = channel_peak_intensity_reference(launch.A0, xp=np)
    background = 0.1
    intensity = pr_driving_intensity(
        launch.A0,
        peak_intensity_reference=reference,
        background_intensity=background,
        coherence_groups=launch.coherence_groups,
        xp=np,
    )
    return GuardFixture(
        name="optical_two_beam_source",
        description=(
            "frozen source produced by the production coherent two-beam optical "
            "launch and PR source-normalization path"
        ),
        intensity=np.asarray(intensity),
        initial_E=np.zeros_like(intensity),
        applied_field=0.4,
        background_intensity=background,
        dx_normalized=PRMaterialSpec().characteristic_wavenumber_per_um * grid.dx_um,
        options=PRStaticSolverOptions(max_iterations=40),
    )


def localized_stress_fixture() -> GuardFixture:
    """Return a deterministic finite stress case with positive initial state factor."""

    rng = np.random.default_rng(201)
    intensity = 0.01 + np.exp(rng.normal(0.0, 0.5, size=(32, 32)))
    initial = rng.normal(0.0, 0.35, size=intensity.shape)
    derivative, _ = periodic_derivatives_x(initial, dx_normalized=1.0, xp=np)
    if float(np.min(1.0 + derivative)) <= 0.0:
        raise AssertionError("stress fixture must retain positive 1+dE/dx")
    return GuardFixture(
        name="localized_finite_stress",
        description=(
            "deterministic positive-intensity localized-residual stress fixture "
            "with positive initial 1+dE/dx"
        ),
        intensity=intensity,
        initial_E=initial,
        applied_field=0.0,
        background_intensity=0.0,
        dx_normalized=1.0,
        options=PRStaticSolverOptions(),
    )


def historical_snapshot_fixture(repository: Path = Path(".")) -> GuardFixture:
    snapshot = (
        repository
        / "outputs/pr_cw_diagnostic_2266990/products/Cw/first_pass_incremental/"
        "failure_snapshot.npz"
    )
    with np.load(snapshot) as saved:
        intensity = saved["fixed_intensity"].copy()
        initial = saved["material_initial_state"].copy()
    return GuardFixture(
        name="historical_cw_z3720",
        description=(
            "retained 1024-square fixed-source failure snapshot from the Cw "
            "optical-coupled march at z=3720 um"
        ),
        intensity=intensity,
        initial_E=initial,
        applied_field=0.0,
        background_intensity=0.0,
        dx_normalized=4.182513268935555,
        options=PRStaticSolverOptions(max_iterations=40),
    )


def evaluation_fixtures(repository: Path = Path(".")) -> tuple[GuardFixture, ...]:
    return (
        periodic_fixture("weak_periodic", modulation=0.01),
        periodic_fixture("moderate_periodic", modulation=0.4),
        periodic_fixture("high_periodic", modulation=0.95),
        periodic_fixture(
            "sharp_low_frequency_periodic",
            modulation=0.95,
            kg_rad_per_um=0.25,
        ),
        optical_source_fixture(),
        localized_stress_fixture(),
        historical_snapshot_fixture(repository),
    )


def run_evaluation(repository: Path = Path(".")) -> list[dict[str, Any]]:
    return [evaluate_fixture(fixture) for fixture in evaluation_fixtures(repository)]


def main() -> int:
    print(json.dumps(run_evaluation(), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
