import hashlib
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.checks import pr_kwak_reduced_material_harmonic_sweep as sweep


def test_kwak_parameter_mapping_and_exact_period():
    x_um, intensity, dx_um = sweep.periodic_intensity(0.95)
    spectrum = np.fft.fft(intensity[:, 0] - np.mean(intensity)) / sweep.GRID_SIZE
    assert sweep.KG_RAD_PER_UM == pytest.approx(2.855993321445266, abs=5e-15)
    assert x_um[-1] + dx_um == pytest.approx(sweep.PERIOD_UM, abs=2e-15)
    assert 2.0 * abs(spectrum[1]) == pytest.approx(0.95, abs=3e-16)
    inactive = spectrum.copy(); inactive[[1, -1]] = 0.0
    assert np.max(np.abs(inactive)) < 3e-16
    assert sweep.MATERIAL.relative_permittivity == 513.0
    assert sweep.MATERIAL.mobile_charge_density_m3 == 2.0e22
    assert sweep.MATERIAL.temperature_K == 293.0


def test_exact_bin_fourier_convention():
    phase = 2.0 * math.pi * np.arange(64) / 64
    field = (0.2 + 0.4 * np.cos(phase) + 0.3 * np.sin(2 * phase))[:, None]
    assert sweep.harmonic_coefficient(field, 1) == pytest.approx(0.2 + 0j, abs=2e-16)
    assert sweep.harmonic_coefficient(field, 2) == pytest.approx(-0.15j, abs=2e-16)


def test_global_intensity_scaling_cancels_from_ratios():
    unit = sweep.solve_case(0.4, 1.0e-4, 256, intensity_scale=1.0)
    scaled = sweep.solve_case(0.4, 1.0e-4, 256, intensity_scale=7.0)
    for key in ("R1", "A2_over_A1", "A3_over_A1", "phase_diff_rad", "relative_L2"):
        assert scaled[key] == pytest.approx(unit[key], rel=2e-13, abs=2e-14)


def test_retained_evidence_is_closed_and_scientifically_consistent():
    root = (
        Path(__file__).resolve().parents[1]
        / "results/pr_kwak_reduced_material_harmonic_sweep_2026-09-12"
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["production_commit"] == sweep.PRODUCTION_COMMIT
    assert manifest["source_clean_before_overlay"] is True
    assert manifest["production_status_porcelain"] == ""
    assert manifest["analysis_harness_sha256"] == hashlib.sha256(
        (
            Path(__file__).resolve().parents[1]
            / "scripts/checks/pr_kwak_reduced_material_harmonic_sweep.py"
        ).read_bytes()
    ).hexdigest()
    for relative, expected in manifest["artifact_sha256"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected

    def typed_rows(name):
        with (root / name).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["iterations"] = int(row["iterations"])
            row["Nx"] = int(row["Nx"])
            row["converged"] = row["converged"] == "True"
            for key in row.keys() - {"iterations", "Nx", "converged"}:
                row[key] = float(row[key])
        return rows

    canonical = sweep.canonical_sha256({
        "primary": typed_rows("summary.csv"),
        "dark": typed_rows("dark_sensitivity.csv"),
        "resolution": typed_rows("resolution.csv"),
        "fits": json.loads((root / "small_m_fits.json").read_text()),
    })
    assert canonical == manifest["canonical_scientific_data_sha256"]

    primary = np.genfromtxt(root / "summary.csv", delimiter=",", names=True)
    assert len(primary) == len(sweep.MODULATIONS)
    assert np.all(primary["R1"] > 1.0)
    assert np.max(np.abs(primary["phase_diff_rad"])) < 5e-14
    assert primary[-1]["R1"] == pytest.approx(2.3682394573747225, rel=2e-14)
    fits = json.loads((root / "small_m_fits.json").read_text(encoding="utf-8"))
    assert fits["R1_minus_one_exponent"] == pytest.approx(2.0, abs=0.02)
    assert fits["A2_exponent"] == pytest.approx(2.0, abs=0.02)
    assert fits["A3_exponent"] == pytest.approx(3.0, abs=0.03)

    resolution = np.genfromtxt(root / "resolution.csv", delimiter=",", names=True)
    for m, dark in sweep.RESOLUTION_CASES:
        rows = resolution[(resolution["m"] == m) & (resolution["Id_over_I0"] == dark)]
        assert list(rows["Nx"].astype(int)) == [256, 512, 1024]
        assert abs(rows[2]["R1"] - rows[1]["R1"]) < abs(rows[1]["R1"] - rows[0]["R1"])

    dark = np.genfromtxt(root / "dark_sensitivity.csv", delimiter=",", names=True)
    for m in sweep.DARK_MODULATIONS:
        rows = dark[dark["m"] == m]
        assert np.all(np.diff(rows["R1"]) < 0.0)
        assert np.all(rows["R1"] > 1.0)
