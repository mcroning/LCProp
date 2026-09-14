import numpy as np

from scripts.checks.pr_reduced_static_max_residual_guard_evaluation import (
    localized_stress_fixture,
    optical_source_fixture,
    periodic_fixture,
)
from scripts.checks.pr_reduced_static_positive_carrier_gate_evaluation import (
    solve_variant,
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
        original = solve_variant(fixture, enforce_positive_carrier=False)
        gated = solve_variant(fixture, enforce_positive_carrier=True)

        assert original.result.converged
        assert gated.result.converged
        assert not any(trial["carrier_gate_rejects"] for trial in gated.trials)
        assert np.array_equal(original.result.E, gated.result.E)
        assert original.result.records == gated.result.records


def test_positive_carrier_gate_fails_honestly_at_physical_boundary():
    fixture = localized_stress_fixture()
    original = solve_variant(fixture, enforce_positive_carrier=False)
    gated = solve_variant(fixture, enforce_positive_carrier=True)
    rejected = [trial for trial in gated.trials if not trial["accepted"]]
    events = [trial for trial in gated.trials if trial["carrier_gate_rejects"]]

    assert original.result.status == "line_search_failed"
    assert original.accepted_carrier_minima[-1] < 0.0
    assert gated.result.status == "line_search_failed"
    assert not gated.result.converged
    assert gated.result.iterations == 15
    assert len(rejected) == 158
    assert len(events) == 158
    assert sum(event["production_accepts"] for event in events) == 156
    assert gated.accepted_carrier_minima[-1] > 0.0
    assert min(gated.accepted_carrier_minima) > 0.0
    assert np.isfinite(gated.result.residual_rms)
    assert np.isfinite(gated.result.residual_max)

    assert any(event["smaller_admissible_candidate_accepted"] for event in events)
    assert not events[-1]["smaller_admissible_candidate_accepted"]
    assert all(event["candidate_carrier_minimum"] <= 0.0 for event in events)
