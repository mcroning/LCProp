import hashlib
import json
import math

import numpy as np
import pytest

from scripts.checks.pr_linearization_resolution_sweep import (
    APERTURE_UM,
    GRATING_MAGNITUDE_RAD_PER_UM,
    GRATING_PERIOD_UM,
    TOTAL_POWER_MW,
    REFERENCE_INTENSITY,
    broad_beam_intensity,
    carrier_geometry,
    complex_harmonic,
    observed_order,
    powers_for_visibility,
    resolution_metadata,
    run_material_case,
    visibility_from_powers,
    write_outputs,
)
from lcprop.pr.carrier_power import carrier_power_diagnostic


def _mode_field(shape, index, power, *, dx_um, dy_um):
    spectrum = np.zeros(shape, dtype=np.complex128)
    spectrum[index] = math.sqrt(power * np.prod(shape) / (dx_um * dy_um))
    return np.fft.ifft2(spectrum)


def test_exact_x_and_45_degree_carrier_geometry():
    x_control = carrier_geometry("x")
    diagonal = carrier_geometry("45deg")

    assert x_control.beam_1_k_rad_per_um == pytest.approx((-math.pi / 2.0, 0.0))
    assert x_control.beam_2_k_rad_per_um == pytest.approx((math.pi / 2.0, 0.0))
    assert x_control.delta_k_rad_per_um == pytest.approx((math.pi, 0.0))
    component = math.pi / math.sqrt(2.0)
    assert diagonal.delta_k_rad_per_um == pytest.approx((component, -component))
    for geometry in (x_control, diagonal):
        assert geometry.grating_magnitude_rad_per_um == pytest.approx(
            GRATING_MAGNITUDE_RAD_PER_UM
        )
        assert geometry.grating_period_um == pytest.approx(GRATING_PERIOD_UM)
        assert geometry.beam_1_center_um == pytest.approx(
            tuple(-value for value in geometry.beam_2_center_um)
        )
        assert max(map(abs, geometry.beam_1_center_um)) < APERTURE_UM / 2.0


@pytest.mark.parametrize("visibility", (1.0, 0.5, 0.125, 0.03125))
def test_visibility_construction_keeps_total_power_fixed(visibility):
    powers = powers_for_visibility(visibility)

    assert sum(powers) == pytest.approx(TOTAL_POWER_MW, abs=2e-15)
    assert visibility_from_powers(*powers) == pytest.approx(visibility, rel=2e-14)


def test_resolution_metadata_matches_harmonic_sampling_gate():
    expected = {
        256: (2.56, 1.28, 2.56 / 3.0, "marginal", "underresolved", "underresolved"),
        512: (5.12, 2.56, 5.12 / 3.0, "resolved", "marginal", "underresolved"),
        1024: (10.24, 5.12, 10.24 / 3.0, "resolved", "resolved", "marginal"),
    }
    for size, values in expected.items():
        metadata = resolution_metadata(size)
        assert metadata["dx_um"] == pytest.approx(APERTURE_UM / size)
        assert metadata["points_per_K"] == pytest.approx(values[0])
        assert metadata["points_per_2K"] == pytest.approx(values[1])
        assert metadata["points_per_3K"] == pytest.approx(values[2])
        assert metadata["resolution_K"] == values[3]
        assert metadata["resolution_2K"] == values[4]
        assert metadata["resolution_3K"] == values[5]

    x_support = resolution_metadata(512, geometry="x")
    diagonal_support = resolution_metadata(512, geometry="45deg")
    assert x_support["periodic_fundamental_mode_components"] == pytest.approx(
        (100.0, 0.0)
    )
    assert x_support["periodic_fundamental_commensurate"]
    assert diagonal_support["periodic_fundamental_mode_components"] == pytest.approx(
        (100.0 / math.sqrt(2.0), -100.0 / math.sqrt(2.0))
    )
    assert not diagonal_support["periodic_fundamental_commensurate"]


@pytest.mark.parametrize("geometry", ("x", "45deg"))
@pytest.mark.parametrize("visibility", (0.5, 0.125, 0.03125))
def test_broad_beam_fixture_keeps_discrete_mean_intensity_fixed(
    geometry, visibility
):
    intensity, _x, _y = broad_beam_intensity(
        256,
        visibility,
        geometry=geometry,
        collapse_x_control_y=True,
    )

    assert np.mean(intensity) == pytest.approx(REFERENCE_INTENSITY, abs=3e-16)
    assert np.min(intensity) > 0.0


@pytest.mark.parametrize("geometry", ("x", "45deg"))
def test_retained_material_fixture_uses_literal_square_grid(geometry):
    intensity, x_um, y_um = broad_beam_intensity(
        256,
        0.125,
        geometry=geometry,
    )

    assert intensity.shape == (256, 256)
    assert x_um.shape == (256,)
    assert y_um.shape == (256,)


def test_harmonic_extraction_recovers_known_complex_coefficients():
    nx, ny = 80, 48
    x = np.arange(nx) * APERTURE_UM / nx
    y = np.arange(ny) * APERTURE_UM / ny
    wavevector = (2.0 * math.pi * 3.0 / APERTURE_UM, 0.0)
    coefficients = (
        0.12 * np.exp(0.3j),
        0.035 * np.exp(-0.7j),
        0.008 * np.exp(1.1j),
    )
    field = np.full((nx, ny), 0.4)
    for harmonic, coefficient in enumerate(coefficients, start=1):
        phase = harmonic * wavevector[0] * x[:, None]
        field += 2.0 * np.real(coefficient * np.exp(1j * phase))

    actual = tuple(
        complex_harmonic(
            field,
            x_um=x,
            y_um=y,
            wavevector_rad_per_um=wavevector,
            harmonic=harmonic,
            window=False,
        )
        for harmonic in (1, 2, 3)
    )
    np.testing.assert_allclose(actual, coefficients, rtol=0.0, atol=2e-16)


def test_cheap_reduced_fixture_has_quadratic_error_and_linear_response():
    modulations = (0.2, 0.1, 0.05, 0.025)
    nonlinear_rows = []
    for modulation in modulations:
        rows = run_material_case(
            N=256,
            visibility=modulation,
            geometry_name="x",
            include_full_linearized=False,
        )
        nonlinear = next(row for row in rows if row["model"] == "reduced_nonlinear")
        assert nonlinear["converged"]
        assert nonlinear["pump_power_input_normalized"] is None
        assert nonlinear["signal_power_output_normalized"] is None
        assert "pump_power_input" not in nonlinear
        assert "signal_power_output" not in nonlinear
        nonlinear_rows.append(nonlinear)

    errors = [row["material_abs_L2_vs_linearized"] for row in nonlinear_rows]
    responses = [row["linear_response_rms"] for row in nonlinear_rows]
    assert observed_order(modulations, errors) == pytest.approx(2.0, abs=0.08)
    assert observed_order(modulations, responses) == pytest.approx(1.0, abs=1e-10)
    scaled = np.asarray(errors) / np.square(modulations)
    assert np.max(scaled) / np.min(scaled) < 1.08


def test_compact_output_manifest_checksums_scientific_payload(tmp_path):
    rows = [
        {
            "study": "frozen_intensity_material",
            "geometry": "x",
            "N": 32,
            "m": 0.125,
            "model": "reduced_nonlinear",
            "material_abs_L2_vs_linearized": 0.001,
            "linear_response_rms": 0.02,
        }
    ]

    write_outputs(
        tmp_path,
        rows=rows,
        manifest={
            "production_source_commit": "abc123",
            "production_source_clean": True,
            "harness_sha256": "harness-sha",
        },
        make_plots=False,
    )

    document = json.loads((tmp_path / "summary.json").read_text())
    csv_sha = hashlib.sha256((tmp_path / "summary.csv").read_bytes()).hexdigest()
    payload_bytes = json.dumps(
        {"rows": document["rows"], "summary": document["summary"]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert document["manifest"]["production_source_clean"] is True
    assert document["manifest"]["evidence_artifact_sha256"] == {
        "summary.csv": csv_sha
    }
    assert document["manifest"]["scientific_payload_sha256"] == (
        hashlib.sha256(payload_bytes).hexdigest()
    )


def test_carrier_power_fixture_is_well_separated():
    size = 256
    shape = (size, size)
    spacing = APERTURE_UM / size
    first_index = (size - 50, 0)
    second_index = (50, 0)
    initial = np.stack((
        _mode_field(shape, first_index, 0.75, dx_um=spacing, dy_um=spacing),
        _mode_field(shape, second_index, 0.25, dx_um=spacing, dy_um=spacing),
    ))
    geometry = carrier_geometry("x")
    diagnostic = carrier_power_diagnostic(
        initial,
        initial,
        dx_um=spacing,
        dy_um=spacing,
        coherence_groups=("shared", "shared"),
        carrier_channels=(
            {
                "name": "Pump",
                "kx_rad_per_um": geometry.beam_1_k_rad_per_um[0],
                "ky_rad_per_um": geometry.beam_1_k_rad_per_um[1],
            },
            {
                "name": "Signal",
                "kx_rad_per_um": geometry.beam_2_k_rad_per_um[0],
                "ky_rad_per_um": geometry.beam_2_k_rad_per_um[1],
            },
        ),
    )

    assert diagnostic["status"] == "ok"
    assert diagnostic["carrier_separation_quality"] > 0.999999
    np.testing.assert_allclose(diagnostic["carrier_power_input"], (0.75, 0.25))
    np.testing.assert_allclose(diagnostic["carrier_gain"], (1.0, 1.0))
    assert abs(diagnostic["carrier_power_balance_error"]) < 2e-15
