import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import lcprop.pr.static as static_module
from lcprop.pr.evolution import periodic_derivatives_x
from lcprop.pr.static import PRStaticSolverOptions, solve_pr_static_intensity_batched
from scripts.checks.pr_reduced_static_max_residual_guard_evaluation import (
    historical_snapshot_fixture,
    localized_stress_fixture,
    optical_source_fixture,
    periodic_fixture,
)


REPOSITORY = Path(__file__).resolve().parents[1]
EVALUATION_PATH = (
    REPOSITORY
    / "results/pr_reduced_static_positive_carrier_gate_evaluation_2026-09-14/"
    "evaluation.json"
)
SNAPSHOT_PATH = (
    REPOSITORY
    / "outputs/pr_cw_diagnostic_2266990/products/Cw/first_pass_incremental/"
    "failure_snapshot.npz"
)
SNAPSHOT_SHA256 = "e2dc8c9aede6096e8c01745053c895c5f2e2635f67af8cb18c3d52bcca69b067"


def _field_sha256(field: np.ndarray) -> str:
    values = np.ascontiguousarray(field)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def _evaluation_rows() -> dict[str, dict]:
    with EVALUATION_PATH.open(encoding="utf-8") as stream:
        evidence = json.load(stream)
    return {row["fixture"]: row for row in evidence["fixtures"]}


def _solve_fixture(fixture):
    return solve_pr_static_intensity_batched(
        fixture.intensity,
        applied_field=fixture.applied_field,
        background_intensity=fixture.background_intensity,
        dx_normalized=fixture.dx_normalized,
        initial_E=fixture.initial_E,
        options=fixture.options,
        xp=np,
    )


def _minimum_carrier(field: np.ndarray, *, dx_normalized: float) -> float:
    derivative, _ = periodic_derivatives_x(
        field,
        dx_normalized=dx_normalized,
        xp=np,
    )
    return float(np.min(1.0 + derivative))


def _assert_matches_committed_gate_oracle(result, expected: dict) -> None:
    assert result.converged == expected["converged"]
    assert result.status == expected["termination_reason"]
    assert result.iterations == expected["newton_iterations"]
    assert result.residual_rms == expected["final_residual_rms"]
    assert result.residual_max == expected["final_residual_max"]
    assert [record.step_scale for record in result.records[1:]] == expected[
        "accepted_step_sizes"
    ]
    assert [record.residual_rms for record in result.records] == expected[
        "residual_rms_history"
    ]
    assert [record.residual_max for record in result.records] == expected[
        "residual_max_history"
    ]
    assert _field_sha256(result.E) == expected["field_sha256"]


def _run_controlled_line_search(
    monkeypatch,
    *,
    direction_scale,
    max_backtracks=3,
    candidate_residual_value=0.0,
):
    evaluated_states = []

    def controlled_residual(state, *_args, **_kwargs):
        evaluated_states.append(np.asarray(state).copy())
        if np.array_equal(state, np.zeros_like(state)):
            return np.ones_like(state)
        return np.full_like(state, candidate_residual_value)

    direction = np.array([[0.0], [0.0], [0.0], [direction_scale]])
    monkeypatch.setattr(static_module, "hopping_rhs", controlled_residual)
    monkeypatch.setattr(
        static_module,
        "batched_newton_direction",
        lambda *_args, **_kwargs: direction.copy(),
    )
    result = solve_pr_static_intensity_batched(
        np.ones((4, 1)),
        applied_field=0.0,
        background_intensity=0.0,
        dx_normalized=1.0,
        options=PRStaticSolverOptions(
            max_iterations=1,
            residual_rms_tolerance=1.0e-12,
            residual_max_tolerance=1.0e-12,
            max_backtracks=max_backtracks,
        ),
        xp=np,
    )
    return result, evaluated_states


@pytest.mark.parametrize(
    ("direction_scale", "expected_step_scale", "rejected_carrier_minima"),
    (
        pytest.param(1.0, 1.0, (), id="physical-full-step"),
        pytest.param(2.0, 0.5, (0.0,), id="zero-carrier-full-step"),
        pytest.param(4.0, 0.25, (-1.0, 0.0), id="negative-then-zero"),
    ),
)
def test_candidate_admissibility_precedes_unchanged_armijo_merit(
    monkeypatch,
    direction_scale,
    expected_step_scale,
    rejected_carrier_minima,
):
    result, evaluated_states = _run_controlled_line_search(
        monkeypatch,
        direction_scale=direction_scale,
    )

    assert result.converged
    assert result.records[-1].step_scale == expected_step_scale
    assert len(evaluated_states) == 2
    assert np.array_equal(evaluated_states[-1], result.E)
    assert _minimum_carrier(result.E, dx_normalized=1.0) > 0.0
    full_direction = np.array([[0.0], [0.0], [0.0], [direction_scale]])
    observed_rejected_minima = tuple(
        _minimum_carrier(
            step_scale * full_direction,
            dx_normalized=1.0,
        )
        for step_scale in (1.0, 0.5)[: len(rejected_carrier_minima)]
    )
    assert observed_rejected_minima == rejected_carrier_minima


def test_exhausted_nonpositive_candidates_fail_with_last_physical_state(monkeypatch):
    result, evaluated_states = _run_controlled_line_search(
        monkeypatch,
        direction_scale=16.0,
        max_backtracks=2,
    )

    assert not result.converged
    assert result.status == "line_search_failed"
    assert result.message == (
        "damped Newton line search failed to find an acceptable candidate"
    )
    assert result.iterations == 0
    assert len(evaluated_states) == 1
    assert np.array_equal(result.E, np.zeros((4, 1)))
    assert _minimum_carrier(result.E, dx_normalized=1.0) == 1.0


def test_nonfinite_candidate_is_rejected_before_residual_evaluation(monkeypatch):
    result, evaluated_states = _run_controlled_line_search(
        monkeypatch,
        direction_scale=np.inf,
        max_backtracks=1,
    )

    assert not result.converged
    assert result.status == "line_search_failed"
    assert result.message == (
        "damped Newton line search failed to find an acceptable candidate"
    )
    assert len(evaluated_states) == 1
    assert np.array_equal(result.E, np.zeros((4, 1)))


def test_exhausted_armijo_search_uses_truthful_neutral_failure_message(monkeypatch):
    result, evaluated_states = _run_controlled_line_search(
        monkeypatch,
        direction_scale=1.0,
        max_backtracks=2,
        candidate_residual_value=1.0,
    )

    assert not result.converged
    assert result.status == "line_search_failed"
    assert result.message == (
        "damped Newton line search failed to find an acceptable candidate"
    )
    assert len(evaluated_states) == 4
    assert np.array_equal(result.E, np.zeros((4, 1)))
    assert _minimum_carrier(result.E, dx_normalized=1.0) == 1.0


def test_five_ordinary_fixtures_remain_bitwise_equal_to_pre_gate_baseline():
    rows = _evaluation_rows()
    fixtures = (
        periodic_fixture("weak_periodic", modulation=0.01),
        periodic_fixture("moderate_periodic", modulation=0.4),
        periodic_fixture("high_periodic", modulation=0.95),
        periodic_fixture(
            "sharp_low_frequency_periodic",
            modulation=0.95,
            kg_rad_per_um=0.25,
        ),
        optical_source_fixture(),
    )
    for fixture in fixtures:
        row = rows[fixture.name]
        assert row["final_field_bitwise_equal"]
        assert row["positive_carrier_gate"]["carrier_gate_rejection_count"] == 0
        result = _solve_fixture(fixture)
        _assert_matches_committed_gate_oracle(result, row["original"])
        _assert_matches_committed_gate_oracle(result, row["positive_carrier_gate"])


def test_stress_fixture_matches_gate_oracle_and_not_rejected_maximum_guard():
    rows = _evaluation_rows()
    fixture = localized_stress_fixture()
    row = rows[fixture.name]
    result = _solve_fixture(fixture)

    _assert_matches_committed_gate_oracle(result, row["positive_carrier_gate"])
    assert not result.converged
    assert result.status == "line_search_failed"
    assert result.iterations == 15
    assert _minimum_carrier(
        result.E,
        dx_normalized=fixture.dx_normalized,
    ) == row["positive_carrier_gate"]["final_minimum_carrier_density"]
    comparison = next(
        item
        for item in json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))[
            "difficult_case_three_policy_comparison"
        ]
        if item["fixture"] == fixture.name
    )
    assert _field_sha256(result.E) != comparison["rejected_maximum_residual_guard"][
        "field_sha256"
    ]


def test_historical_snapshot_matches_gate_oracle_when_supplied():
    if not SNAPSHOT_PATH.is_file():
        pytest.skip("separately supplied 3720 um historical snapshot is unavailable")
    assert hashlib.sha256(SNAPSHOT_PATH.read_bytes()).hexdigest() == SNAPSHOT_SHA256

    rows = _evaluation_rows()
    fixture = historical_snapshot_fixture(REPOSITORY)
    row = rows[fixture.name]
    result = _solve_fixture(fixture)

    _assert_matches_committed_gate_oracle(result, row["positive_carrier_gate"])
    assert not result.converged
    assert result.status == "line_search_failed"
    assert result.iterations == 14
    assert _minimum_carrier(
        result.E,
        dx_normalized=fixture.dx_normalized,
    ) == row["positive_carrier_gate"]["final_minimum_carrier_density"]
    residual_maxima = [record.residual_max for record in result.records]
    assert any(
        later > earlier
        for earlier, later in zip(residual_maxima, residual_maxima[1:])
    )
