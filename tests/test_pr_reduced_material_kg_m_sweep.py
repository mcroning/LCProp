import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.checks.pr_reduced_material_kg_m_sweep import (
    BASELINE_COMMIT,
    CHARACTERISTIC_WAVENUMBER_PER_UM,
    GRID_SIZE,
    MODULATIONS,
    RESOLUTION_GRID_SIZES,
    RESOLUTION_WAVENUMBERS,
    WAVENUMBERS_RAD_PER_UM,
    canonical_sha256,
    harmonic_coefficient,
    periodic_intensity,
    solve_case,
)


EVIDENCE_ROOT = Path("results/pr_reduced_material_kg_m_sweep_2026-09-11")


def _read_csv(name):
    with (EVIDENCE_ROOT / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _typed_summary(rows):
    integer_fields = {"grid_size", "nonlinear_iterations"}
    return [
        {
            key: int(value) if key in integer_fields else float(value)
            for key, value in row.items()
        }
        for row in rows
    ]


def _typed_fits(rows):
    return [{key: float(value) for key, value in row.items()} for row in rows]


def test_frequency_grid_uses_natural_dimensionless_scale():
    assert CHARACTERISTIC_WAVENUMBER_PER_UM == pytest.approx(
        4.282893587390008, rel=0.0, abs=0.0
    )
    assert math.pi in WAVENUMBERS_RAD_PER_UM
    assert CHARACTERISTIC_WAVENUMBER_PER_UM in WAVENUMBERS_RAD_PER_UM
    dimensionless = np.asarray(WAVENUMBERS_RAD_PER_UM) / (
        CHARACTERISTIC_WAVENUMBER_PER_UM
    )
    assert dimensionless[0] < 0.1
    assert 1.0 in dimensionless
    assert dimensionless[-1] > 1.8


@pytest.mark.parametrize("kg", WAVENUMBERS_RAD_PER_UM)
def test_one_period_intensity_puts_grating_on_exact_fft_bin(kg):
    intensity, length_um, dx_um = periodic_intensity(kg, 0.95, GRID_SIZE)
    spectrum = np.fft.fft(intensity[:, 0] - np.mean(intensity)) / GRID_SIZE

    assert length_um == pytest.approx(2.0 * math.pi / kg, rel=2e-16)
    assert dx_um == pytest.approx(length_um / GRID_SIZE, rel=2e-16)
    assert 2.0 * abs(spectrum[1]) == pytest.approx(0.95, abs=3e-16)
    inactive = spectrum.copy()
    inactive[[1, GRID_SIZE - 1]] = 0.0
    assert np.max(np.abs(inactive)) < 3e-16


def test_clean_operator_case_matches_authoritative_pi_row():
    row = solve_case(math.pi, 0.1, 256)

    assert row["R_K"] == pytest.approx(1.0027427939584972, rel=3e-13)
    assert abs(row["delta_phase_rad"]) < 3e-15
    assert row["nonlinear_residual_rms"] <= row["residual_rms_tolerance"]
    assert row["nonlinear_residual_max"] <= row["residual_max_tolerance"]


def test_map_is_superlinear_and_preserves_original_point():
    rows = _typed_summary(_read_csv("summary.csv"))
    assert len(rows) == len(WAVENUMBERS_RAD_PER_UM) * len(MODULATIONS)
    assert all(row["R_K"] > 1.0 for row in rows)
    original = next(
        row
        for row in rows
        if row["kg_rad_per_um"] == math.pi and row["m"] == 0.95
    )
    maximum = max(rows, key=lambda row: row["R_K"])

    assert original["R_K"] == pytest.approx(1.643537533585462, abs=3e-15)
    assert maximum["kg_over_kD"] == 1.0
    assert maximum["m"] == 0.95
    assert maximum["R_K"] == pytest.approx(1.678413903674545, abs=3e-15)
    assert max(abs(row["delta_phase_rad"]) for row in rows) < 5e-15
    assert all(
        row["nonlinear_residual_rms"] <= row["residual_rms_tolerance"]
        and row["nonlinear_residual_max"] <= row["residual_max_tolerance"]
        for row in rows
    )


def test_small_modulation_fits_recover_expected_orders():
    fits = _typed_fits(_read_csv("small_m_fits.csv"))
    assert len(fits) == len(WAVENUMBERS_RAD_PER_UM)
    assert all(
        abs(row["R_K_minus_one_exponent"] - 2.0) < 0.003
        for row in fits
    )
    assert all(abs(row["A2_exponent"] - 2.0) < 0.003 for row in fits)
    assert all(abs(row["A3_exponent"] - 3.0) < 0.004 for row in fits)
    assert all(row["C_fit_relative_rms"] < 0.001 for row in fits)
    assert min(row["C_RK"] for row in fits) > 0.25


def test_resolution_rows_show_second_order_convergence():
    rows = _typed_summary(_read_csv("resolution.csv"))
    assert len(rows) == len(RESOLUTION_WAVENUMBERS) * len(RESOLUTION_GRID_SIZES)
    for kg in RESOLUTION_WAVENUMBERS:
        selected = sorted(
            (row for row in rows if row["kg_rad_per_um"] == kg),
            key=lambda row: row["grid_size"],
        )
        differences = [
            abs(selected[0]["R_K"] - selected[1]["R_K"]),
            abs(selected[1]["R_K"] - selected[2]["R_K"]),
        ]
        assert differences[0] / differences[1] == pytest.approx(4.0, abs=0.01)
        assert differences[1] / abs(selected[2]["R_K"]) < 3.1e-5
        assert all(
            row["nonlinear_residual_rms"] <= row["residual_rms_tolerance"]
            and row["nonlinear_residual_max"] <= row["residual_max_tolerance"]
            for row in selected
        )


def test_manifest_hashes_all_clean_source_evidence():
    manifest = json.loads((EVIDENCE_ROOT / "manifest.json").read_text())
    summary = _typed_summary(_read_csv("summary.csv"))
    fits = _typed_fits(_read_csv("small_m_fits.csv"))
    resolution = _typed_summary(_read_csv("resolution.csv"))

    assert manifest["production_source"] == {
        "clean_after": True,
        "clean_before": True,
        "commit": BASELINE_COMMIT,
        "status_porcelain_after": "",
        "status_porcelain_before": "",
    }
    assert manifest["harness_sha256"] == hashlib.sha256(
        Path("scripts/checks/pr_reduced_material_kg_m_sweep.py").read_bytes()
    ).hexdigest()
    assert manifest["canonical_sha256"] == {
        "scientific_rows": canonical_sha256(summary),
        "small_m_fits": canonical_sha256(fits),
        "resolution_rows": canonical_sha256(resolution),
    }
    for relative, expected in manifest["artifacts"].items():
        assert hashlib.sha256(
            (EVIDENCE_ROOT / relative).read_bytes()
        ).hexdigest() == expected
