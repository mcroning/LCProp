import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.checks.pr_reduced_material_harmonic_sweep import (
    FIELDNAMES,
    GRID_SIZE,
    MODULATIONS,
    analyze_modulation,
    harmonic_amplitude,
    harmonic_coefficient,
    periodic_intensity,
    run_sweep,
    small_modulation_orders,
    write_csv,
)


@pytest.fixture(scope="module")
def sweep_rows():
    return run_sweep()


def test_prescribed_intensity_is_one_exact_positive_period():
    intensity = periodic_intensity(0.95)[:, 0]
    spectrum = np.fft.fft(intensity - np.mean(intensity)) / GRID_SIZE

    assert intensity.shape == (GRID_SIZE,)
    assert np.mean(intensity) == pytest.approx(1.0, abs=2e-16)
    assert np.min(intensity) == pytest.approx(0.05, abs=2e-16)
    assert 2.0 * abs(spectrum[1]) == pytest.approx(0.95, abs=2e-16)
    inactive = spectrum.copy()
    inactive[[1, GRID_SIZE - 1]] = 0.0
    assert np.max(np.abs(inactive)) < 2e-16


def test_exact_fft_bin_convention_recovers_amplitude_and_phase():
    x = 2.0 * math.pi * np.arange(GRID_SIZE) / GRID_SIZE
    field = (0.4 + 0.3 * np.cos(x) + 0.1 * np.sin(2.0 * x))[:, None]

    assert harmonic_amplitude(field, 1) == pytest.approx(0.3, abs=2e-16)
    assert harmonic_amplitude(field, 2) == pytest.approx(0.1, abs=2e-16)
    assert np.angle(harmonic_coefficient(field, 1)) == pytest.approx(
        0.0, abs=2e-16
    )
    assert np.angle(harmonic_coefficient(field, 2)) == pytest.approx(
        -math.pi / 2.0, abs=2e-15
    )


def test_sweep_converges_and_preserves_primary_values(sweep_rows):
    assert [row["m"] for row in sweep_rows] == list(MODULATIONS)
    assert all(row["nonlinear_iterations"] <= 4 for row in sweep_rows)
    assert all(row["nonlinear_residual_rms"] < 3e-13 for row in sweep_rows)
    assert all(row["nonlinear_residual_max"] < 1e-12 for row in sweep_rows)
    assert sweep_rows[0]["A1_ratio"] == pytest.approx(
        1.0000272821687135, rel=0.0, abs=2e-15
    )
    assert sweep_rows[-1]["A1_ratio"] == pytest.approx(
        1.643537533585462, rel=0.0, abs=3e-15
    )
    assert all(abs(row["delta_phase_rad"]) < 3e-15 for row in sweep_rows)


def test_nonlinearity_strengthens_fundamental_and_generates_harmonics(sweep_rows):
    ratios = [row["A1_ratio"] for row in sweep_rows]
    second = [row["A2_over_A1_nonlinear"] for row in sweep_rows]
    third = [row["A3_over_A1_nonlinear"] for row in sweep_rows]

    assert ratios == sorted(ratios)
    assert second == sorted(second)
    assert third == sorted(third)
    assert all(ratio > 1.0 for ratio in ratios)
    assert second[-1] == pytest.approx(0.403667410407777, rel=2e-14)
    assert third[-1] == pytest.approx(0.19168597911877178, rel=2e-14)


def test_small_modulation_orders_match_linearization(sweep_rows):
    orders = small_modulation_orders(sweep_rows)

    assert orders["A1_ratio_minus_one"] == pytest.approx(2.0, abs=0.01)
    assert orders["difference_rms"] == pytest.approx(2.0, abs=0.01)
    assert orders["relative_L2_error"] == pytest.approx(1.0, abs=0.01)
    assert orders["A2"] == pytest.approx(2.0, abs=0.01)
    assert orders["A3"] == pytest.approx(3.0, abs=0.01)


def test_compact_csv_round_trips_all_rows(tmp_path, sweep_rows):
    path = tmp_path / "summary.csv"
    write_csv(path, sweep_rows)

    with path.open(newline="", encoding="utf-8") as handle:
        restored = list(csv.DictReader(handle))
    assert tuple(restored[0]) == FIELDNAMES
    assert len(restored) == len(MODULATIONS)
    assert float(restored[-1]["A1_ratio"]) == sweep_rows[-1]["A1_ratio"]


def test_clean_evidence_manifest_and_retained_profiles_are_consistent():
    root = Path("results/pr_reduced_material_harmonic_sweep_2026-09-11")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["production_source_commit"] == (
        "f534f0c7dabcae963688b8561bbfeb01b04f3869"
    )
    assert manifest["production_source_clean"] is True
    assert manifest["production_source_status_porcelain"] == ""
    assert manifest["harness_sha256"] == hashlib.sha256(
        Path("scripts/checks/pr_reduced_material_harmonic_sweep.py").read_bytes()
    ).hexdigest()
    for relative, expected in manifest["artifact_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected

    with (root / "representative_profiles.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        profiles = list(csv.DictReader(handle))
    with (root / "representative_spectra.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        spectra = list(csv.DictReader(handle))
    assert len(profiles) == 3 * GRID_SIZE
    assert len(spectra) == 3 * 3
    for modulation in (0.1, 0.6, 0.95):
        selected = [row for row in profiles if float(row["m"]) == modulation]
        nonlinear = np.asarray(
            [float(row["E_nonlinear"]) for row in selected]
        )[:, None]
        linearized = np.asarray(
            [float(row["E_linearized"]) for row in selected]
        )[:, None]
        for harmonic in (1, 2, 3):
            retained = next(
                row
                for row in spectra
                if float(row["m"]) == modulation
                and int(row["harmonic"]) == harmonic
            )
            assert complex(
                float(retained["nonlinear_real"]),
                float(retained["nonlinear_imag"]),
            ) == pytest.approx(harmonic_coefficient(nonlinear, harmonic))
            assert complex(
                float(retained["linearized_real"]),
                float(retained["linearized_imag"]),
            ) == pytest.approx(harmonic_coefficient(linearized, harmonic))

    comparison = manifest["provisional_comparison"]
    assert comparison["scientific_conclusions_changed"] is False
    assert set(comparison["maximum_absolute_metric_differences"].values()) == {
        0.0
    }
    assert set(comparison["absolute_exponent_differences"].values()) == {0.0}
