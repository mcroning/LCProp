import numpy as np

from scripts.checks.pr_reduced_static_max_residual_guard_evaluation import (
    localized_stress_fixture,
    optical_source_fixture,
    periodic_fixture,
)
from scripts.checks.pr_reduced_static_positive_carrier_gate_evaluation import (
    evaluate_fixture,
)


def test_positive_carrier_gate_is_bitwise_inert_on_ordinary_fixtures():
    fixtures = (
        periodic_fixture("weak_test", modulation=0.01),
        periodic_fixture("moderate_test", modulation=0.4),
        periodic_fixture("high_test", modulation=0.95),
        periodic_fixture(
            "sharp_low_frequency_test",
            modulation=0.95,
            kg_rad_per_um=0.25,
        ),
        optical_source_fixture(),
    )

    for fixture in fixtures:
        comparison = evaluate_fixture(fixture)
        original = comparison["original"]
        gated = comparison["positive_carrier_gate"]

        assert original["converged"]
        assert gated["converged"]
        assert gated["carrier_gate_rejection_count"] == 0
        assert comparison["final_field_bitwise_equal"]
        assert comparison["final_field_relative_l2"] == 0.0
        assert comparison["final_field_maximum_absolute_difference"] == 0.0
        assert original["accepted_step_sizes"] == gated["accepted_step_sizes"]
        assert original["residual_rms_history"] == gated["residual_rms_history"]
        assert original["residual_max_history"] == gated["residual_max_history"]
        assert comparison["original_harness_matches_production_bitwise"]


def test_positive_carrier_gate_fails_honestly_at_physical_boundary():
    comparison = evaluate_fixture(localized_stress_fixture())
    original = comparison["original"]
    gated = comparison["positive_carrier_gate"]

    assert original["termination_reason"] == "line_search_failed"
    assert original["final_minimum_carrier_density"] < 0.0
    assert gated["termination_reason"] == "line_search_failed"
    assert not gated["converged"]
    assert gated["newton_iterations"] == 15
    assert gated["total_line_search_rejects"] == 158
    assert gated["carrier_gate_rejection_count"] == 158
    assert gated["production_acceptable_carrier_gate_rejection_count"] == 156
    assert gated["final_minimum_carrier_density"] > 0.0
    assert min(gated["accepted_carrier_minimum_history"]) > 0.0
    assert np.isfinite(gated["final_residual_rms"])
    assert np.isfinite(gated["final_residual_max"])
    assert comparison["original_harness_matches_production_bitwise"]

    events = comparison["carrier_gate_rejections"]
    assert any(event["smaller_admissible_candidate_accepted"] for event in events)
    assert not events[-1]["smaller_admissible_candidate_accepted"]
    assert all(event["candidate_carrier_minimum"] <= 0.0 for event in events)
