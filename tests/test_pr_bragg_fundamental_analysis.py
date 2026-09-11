import json
import math
from pathlib import Path

import pytest

from scripts.checks.pr_bragg_fundamental_analysis import (
    analyze_rows,
    compare_with_superseded,
    coefficient_from_polar,
    wrap_phase_radians,
    write_outputs,
)


def _row(model, magnitude, phase, *, signal_gain=None):
    return {
        "study": "frozen_intensity_material",
        "fixture": "broad_beam",
        "geometry": "x",
        "N": 1024,
        "m": 0.5,
        "model": model,
        "resolution_K": "resolved",
        "harmonic_1_magnitude": magnitude,
        "harmonic_1_phase_rad": phase,
        "harmonic_2_relative_to_K": 0.2,
        "harmonic_3_relative_to_K": 0.03,
        "pump_gain": None,
        "signal_gain": signal_gain,
        "carrier_separation_quality": None,
    }


@pytest.mark.parametrize(
    ("angle", "expected"),
    (
        (0.0, 0.0),
        (math.pi, -math.pi),
        (-math.pi, -math.pi),
        (3.0 * math.pi, -math.pi),
        (-3.0 * math.pi, -math.pi),
        (2.0 * math.pi + 0.2, 0.2),
    ),
)
def test_phase_wrap_uses_documented_principal_interval(angle, expected):
    assert wrap_phase_radians(angle) == pytest.approx(expected, abs=1e-15)


def test_coefficient_reconstruction_preserves_quadrature_sign():
    positive = coefficient_from_polar(0.25, math.pi / 2.0)
    negative = coefficient_from_polar(0.25, -math.pi / 2.0)

    assert positive.real == pytest.approx(0.0, abs=2e-17)
    assert positive.imag == pytest.approx(0.25)
    assert negative.real == pytest.approx(0.0, abs=2e-17)
    assert negative.imag == pytest.approx(-0.25)


def test_analysis_pairs_models_and_computes_complex_comparison():
    rows = [
        _row("reduced_linearized", 0.1, math.pi - 0.02),
        _row("reduced_nonlinear", 0.125, -math.pi + 0.03),
    ]

    result = analyze_rows(rows)

    assert len(result) == 1
    comparison = result[0]
    assert comparison["R_K"] == pytest.approx(1.25)
    assert comparison["delta_phase_rad"] == pytest.approx(0.05)
    assert comparison["real_nonlinear"] == pytest.approx(
        0.125 * math.cos(-math.pi + 0.03)
    )
    assert comparison["imag_linearized"] == pytest.approx(
        0.1 * math.sin(math.pi - 0.02)
    )


def test_analysis_rejects_incomplete_model_pair():
    with pytest.raises(ValueError, match="incomplete"):
        analyze_rows([_row("reduced_nonlinear", 0.1, 0.0)])


def test_analysis_rejects_zero_linearized_fundamental():
    rows = [
        _row("reduced_nonlinear", 0.1, 0.0),
        _row("reduced_linearized", 0.0, 0.0),
    ]

    with pytest.raises(ValueError, match="zero"):
        analyze_rows(rows)


def test_superseded_comparison_reports_roundoff_without_scientific_change():
    old = analyze_rows(
        [
            _row("reduced_nonlinear", 0.125, math.pi / 2.0),
            _row("reduced_linearized", 0.1, math.pi / 2.0),
        ]
    )
    new = [
        {
            **old[0],
            "magnitude_nonlinear": old[0]["magnitude_nonlinear"] + 1e-15,
        }
    ]

    comparison = compare_with_superseded(new, old)

    assert comparison["rows_compared"] == 1
    assert comparison["changed_numeric_values"] == 1
    assert comparison["within_expected_roundoff"] is True


def test_output_manifest_checksums_compact_evidence(tmp_path):
    rows = analyze_rows(
        [
            _row("reduced_nonlinear", 0.125, math.pi / 2.0),
            _row("reduced_linearized", 0.1, math.pi / 2.0),
        ]
    )
    write_outputs(
        tmp_path,
        rows=rows,
        sweep_commit="abc123",
        inputs=[{"path": "input.json", "sha256": "deadbeef"}],
        superseded_comparison=None,
        make_plots=False,
    )

    document = json.loads((tmp_path / "summary.json").read_text())
    manifest = document["manifest"]
    assert manifest["sweep_commit"] == "abc123"
    assert manifest["inputs"] == [{"path": "input.json", "sha256": "deadbeef"}]
    assert manifest["evidence_artifact_sha256"]["summary.csv"]
    assert manifest["scientific_payload_sha256"]


def test_regenerated_evidence_preserves_scientific_conclusions():
    path = Path(
        "results/pr_bragg_fundamental_analysis_2026-09-10/summary.json"
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["manifest"]["sweep_commit"] == (
        "1e5fb7b18961a8536d2cfa3904cc8db4d401c24f"
    )
    material = [
        row
        for row in document["rows"]
        if row["study"] == "frozen_intensity_material"
    ]

    for geometry, expected_ratio in (
        ("x", 1.0792897531048915),
        ("45deg", 1.0756676819445068),
    ):
        primary = next(
            row
            for row in material
            if row["geometry"] == geometry
            and row["N"] == 1024
            and row["m"] == 0.5
        )
        assert primary["R_K"] == expected_ratio
        assert primary["magnitude_nonlinear"] > primary["magnitude_linearized"]
        assert primary["delta_phase_rad"] == 0.0

        modulation = sorted(
            (
                row
                for row in material
                if row["geometry"] == geometry and row["N"] == 1024
            ),
            key=lambda row: row["m"],
        )
        assert [row["R_K"] for row in modulation] == sorted(
            row["R_K"] for row in modulation
        )
        assert modulation[0]["R_K"] - 1.0 < 3.0e-4
        assert all(row["delta_phase_rad"] == 0.0 for row in modulation)
        scaled = [(row["R_K"] - 1.0) / row["m"] ** 2 for row in modulation]
        assert scaled[1] == pytest.approx(scaled[0], rel=0.01)
        assert modulation[-1]["second_harmonic_ratio_nonlinear"] > modulation[0][
            "second_harmonic_ratio_nonlinear"
        ]
        assert modulation[-1]["third_harmonic_ratio_nonlinear"] > modulation[0][
            "third_harmonic_ratio_nonlinear"
        ]

        resolution = sorted(
            (
                row
                for row in material
                if row["geometry"] == geometry and row["m"] == 0.5
            ),
            key=lambda row: row["N"],
        )
        assert [row["N"] for row in resolution] == [256, 512, 1024]
        assert abs(resolution[-1]["R_K"] - resolution[-2]["R_K"]) < 7.0e-4

    production = [
        row for row in document["rows"] if row["study"] == "production_static"
    ]
    assert production
    assert all(row["R_K"] > 1.0 for row in production)
    assert all(
        row["signal_gain_nonlinear"] > row["signal_gain_linearized"]
        for row in production
    )
