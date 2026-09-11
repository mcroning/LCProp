#!/usr/bin/env python3
"""Material-only periodic reduced-PR nonlinear/linearized harmonic sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import subprocess
from typing import Any, Sequence

import numpy as np

from lcprop.pr.reduced_linearized import (
    PRReducedLinearizedSpec,
    solve_pr_reduced_linearized_intensity,
)
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.static import PRStaticSolverOptions, solve_pr_static_intensity_batched


BASELINE_COMMIT = "f534f0c7dabcae963688b8561bbfeb01b04f3869"
GRID_SIZE = 512
DOMAIN_LENGTH_UM = 2.0
REFERENCE_INTENSITY = 1.0
APPLIED_FIELD = 0.0
BACKGROUND_INTENSITY = 0.0
WAVEVECTOR_RAD_PER_UM = math.pi
MODULATIONS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95)
REPRESENTATIVE_MODULATIONS = (0.1, 0.6, 0.95)
FIELDNAMES = (
    "m",
    "A1_nonlinear",
    "A1_linearized",
    "A1_ratio",
    "A2_over_A1_nonlinear",
    "A3_over_A1_nonlinear",
    "delta_phase_rad",
    "relative_L2_error",
    "difference_rms",
    "nonlinear_iterations",
    "nonlinear_residual_rms",
    "nonlinear_residual_max",
)


def periodic_intensity(modulation: float) -> np.ndarray:
    """Return I0*[1+m*sin(k*x)] on one endpoint-excluded period."""

    m = float(modulation)
    if not math.isfinite(m) or not 0.0 < m < 1.0:
        raise ValueError("modulation must be finite and in (0, 1)")
    dx_um = DOMAIN_LENGTH_UM / GRID_SIZE
    x_um = np.arange(GRID_SIZE, dtype=np.float64) * dx_um
    return (
        REFERENCE_INTENSITY
        * (1.0 + m * np.sin(WAVEVECTOR_RAD_PER_UM * x_um))
    )[:, None]


def harmonic_coefficient(field: np.ndarray, harmonic: int) -> complex:
    """Return c_n=N^-1 sum_j[(E_j-mean(E))*exp(-i*2*pi*n*j/N)]."""

    values = np.asarray(field, dtype=np.float64)
    if values.shape != (GRID_SIZE, 1):
        raise ValueError(f"field must have shape ({GRID_SIZE}, 1)")
    n = int(harmonic)
    if n not in (1, 2, 3):
        raise ValueError("harmonic must be 1, 2, or 3")
    centered = values[:, 0] - np.mean(values[:, 0])
    return complex(np.fft.fft(centered)[n] / GRID_SIZE)


def harmonic_amplitude(field: np.ndarray, harmonic: int) -> float:
    """Return the real-field sinusoid amplitude A_n=2*|c_n|."""

    return 2.0 * abs(harmonic_coefficient(field, harmonic))


def wrap_phase_radians(value: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def analyze_modulation(modulation: float) -> dict[str, Any]:
    """Solve and compare one prescribed frozen-intensity material case."""

    row, _profiles, _spectra = solve_modulation(modulation)
    return row


def solve_modulation(
    modulation: float,
    *,
    retain_profiles: bool = False,
) -> tuple[dict[str, Any], list[dict[str, float]], list[dict[str, float]]]:
    """Solve one case and optionally retain compact profile/spectrum evidence."""

    intensity = periodic_intensity(modulation)
    dx_um = DOMAIN_LENGTH_UM / GRID_SIZE
    dx_normalized = (
        PRMaterialSpec().characteristic_wavenumber_per_um * dx_um
    )
    nonlinear = solve_pr_static_intensity_batched(
        intensity,
        applied_field=APPLIED_FIELD,
        background_intensity=BACKGROUND_INTENSITY,
        dx_normalized=dx_normalized,
        options=PRStaticSolverOptions(
            max_iterations=40,
            residual_rms_tolerance=1.0e-12,
            residual_max_tolerance=1.0e-11,
        ),
    )
    if not nonlinear.converged:
        raise RuntimeError(
            f"nonlinear solve failed for m={modulation}: {nonlinear.status}"
        )
    linearized = solve_pr_reduced_linearized_intensity(
        intensity,
        spec=PRReducedLinearizedSpec(
            reference_intensity=REFERENCE_INTENSITY,
            applied_field=APPLIED_FIELD,
            background_intensity=BACKGROUND_INTENSITY,
            dx_normalized=dx_normalized,
        ),
    )
    nonlinear_coefficients = tuple(
        harmonic_coefficient(nonlinear.E, harmonic) for harmonic in (1, 2, 3)
    )
    linearized_fundamental = harmonic_coefficient(linearized.E, 1)
    a1_nonlinear = 2.0 * abs(nonlinear_coefficients[0])
    a1_linearized = 2.0 * abs(linearized_fundamental)
    difference = np.asarray(nonlinear.E) - np.asarray(linearized.E)
    row = {
        "m": float(modulation),
        "A1_nonlinear": a1_nonlinear,
        "A1_linearized": a1_linearized,
        "A1_ratio": a1_nonlinear / a1_linearized,
        "A2_over_A1_nonlinear": (
            abs(nonlinear_coefficients[1]) / abs(nonlinear_coefficients[0])
        ),
        "A3_over_A1_nonlinear": (
            abs(nonlinear_coefficients[2]) / abs(nonlinear_coefficients[0])
        ),
        "delta_phase_rad": wrap_phase_radians(
            np.angle(nonlinear_coefficients[0])
            - np.angle(linearized_fundamental)
        ),
        "relative_L2_error": float(
            np.linalg.norm(difference) / np.linalg.norm(linearized.E)
        ),
        "difference_rms": float(np.sqrt(np.mean(difference * difference))),
        "nonlinear_iterations": nonlinear.iterations,
        "nonlinear_residual_rms": nonlinear.residual_rms,
        "nonlinear_residual_max": nonlinear.residual_max,
    }
    profiles: list[dict[str, float]] = []
    spectra: list[dict[str, float]] = []
    if retain_profiles:
        x_um = np.arange(GRID_SIZE, dtype=np.float64) * dx_um
        profiles = [
            {
                "m": float(modulation),
                "x_um": float(x_value),
                "E_nonlinear": float(nonlinear.E[index, 0]),
                "E_linearized": float(linearized.E[index, 0]),
            }
            for index, x_value in enumerate(x_um)
        ]
        for harmonic in (1, 2, 3):
            nonlinear_coefficient = nonlinear_coefficients[harmonic - 1]
            linearized_coefficient = harmonic_coefficient(
                linearized.E, harmonic
            )
            spectra.append({
                "m": float(modulation),
                "harmonic": harmonic,
                "nonlinear_real": nonlinear_coefficient.real,
                "nonlinear_imag": nonlinear_coefficient.imag,
                "nonlinear_amplitude": 2.0 * abs(nonlinear_coefficient),
                "nonlinear_phase_rad": float(np.angle(nonlinear_coefficient)),
                "linearized_real": linearized_coefficient.real,
                "linearized_imag": linearized_coefficient.imag,
                "linearized_amplitude": 2.0 * abs(linearized_coefficient),
                "linearized_phase_rad": float(np.angle(linearized_coefficient)),
            })
    return row, profiles, spectra


def run_sweep() -> list[dict[str, Any]]:
    """Run the fixed nine-point material-only sweep."""

    return [analyze_modulation(modulation) for modulation in MODULATIONS]


def run_evidence() -> tuple[
    list[dict[str, Any]],
    list[dict[str, float]],
    list[dict[str, float]],
]:
    """Run every solver case once while retaining selected compact evidence."""

    rows: list[dict[str, Any]] = []
    profiles: list[dict[str, float]] = []
    spectra: list[dict[str, float]] = []
    for modulation in MODULATIONS:
        row, retained_profiles, retained_spectra = solve_modulation(
            modulation,
            retain_profiles=modulation in REPRESENTATIVE_MODULATIONS,
        )
        rows.append(row)
        profiles.extend(retained_profiles)
        spectra.extend(retained_spectra)
    return rows, profiles, spectra


def observed_power(modulations: Sequence[float], values: Sequence[float]) -> float:
    """Return the log-log least-squares power for positive samples."""

    x = np.asarray(modulations, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if x.shape != y.shape or x.size < 2 or np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("power-law samples must be matched, positive, and nonempty")
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def small_modulation_orders(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Fit asymptotic powers over the prescribed m<=0.1 subset."""

    small = [row for row in rows if row["m"] <= 0.1]
    modulations = [row["m"] for row in small]
    return {
        "A1_ratio_minus_one": observed_power(
            modulations, [row["A1_ratio"] - 1.0 for row in small]
        ),
        "difference_rms": observed_power(
            modulations, [row["difference_rms"] for row in small]
        ),
        "relative_L2_error": observed_power(
            modulations, [row["relative_L2_error"] for row in small]
        ),
        "A2": observed_power(
            modulations,
            [
                row["A1_nonlinear"] * row["A2_over_A1_nonlinear"]
                for row in small
            ],
        ),
        "A3": observed_power(
            modulations,
            [
                row["A1_nonlinear"] * row["A3_over_A1_nonlinear"]
                for row in small
            ],
        ),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write the compact deterministic result table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_dict_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows supplied for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_summary_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def compare_with_provisional(
    rows: Sequence[dict[str, Any]], provisional_csv: Path
) -> dict[str, Any]:
    """Quantify clean-source differences from the provisional dirty-tree CSV."""

    provisional = _read_summary_csv(provisional_csv)
    if [row["m"] for row in provisional] != [row["m"] for row in rows]:
        raise ValueError("provisional modulation sequence does not match")
    metrics = (
        "A1_ratio",
        "A2_over_A1_nonlinear",
        "A3_over_A1_nonlinear",
        "relative_L2_error",
        "delta_phase_rad",
    )
    differences = {
        metric: max(
            abs(float(clean[metric]) - float(old[metric]))
            for clean, old in zip(rows, provisional)
        )
        for metric in metrics
    }
    clean_orders = small_modulation_orders(rows)
    provisional_orders = small_modulation_orders(provisional)
    order_differences = {
        key: abs(clean_orders[key] - provisional_orders[key])
        for key in clean_orders
    }
    return {
        "provisional_csv_sha256": _sha256(provisional_csv),
        "maximum_absolute_metric_differences": differences,
        "clean_small_m_exponents": clean_orders,
        "provisional_small_m_exponents": provisional_orders,
        "absolute_exponent_differences": order_differences,
        "scientific_conclusions_changed": (
            max(differences.values(), default=0.0) > 1.0e-12
            or max(order_differences.values(), default=0.0) > 1.0e-12
        ),
    }


def _write_figures(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    profiles: Sequence[dict[str, float]],
    spectra: Sequence[dict[str, float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    modulation = np.asarray([row["m"] for row in rows])

    figure, axis = plt.subplots(figsize=(5.6, 3.8))
    axis.plot(
        modulation,
        [row["A1_ratio"] for row in rows],
        "o-",
        label="NL / linearized",
    )
    axis.axhline(
        1.0, color="0.35", linestyle="--", linewidth=1.0, label="unity"
    )
    axis.set(xlabel="Modulation depth m", ylabel="A1,NL / A1,lin")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "fundamental_ratio_vs_modulation.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.6, 3.8))
    axis.plot(
        modulation,
        [row["A2_over_A1_nonlinear"] for row in rows],
        "o-",
        label="A2 / A1",
    )
    axis.plot(
        modulation,
        [row["A3_over_A1_nonlinear"] for row in rows],
        "s-",
        label="A3 / A1",
    )
    axis.set(xlabel="Modulation depth m", ylabel="Nonlinear harmonic ratio")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "harmonic_ratios_vs_modulation.png", dpi=180)
    plt.close(figure)

    small = [row for row in rows if row["m"] <= 0.1]
    small_m = np.asarray([row["m"] for row in small])
    orders = small_modulation_orders(rows)
    scaling_series = (
        (
            "A1 ratio - 1",
            [row["A1_ratio"] - 1.0 for row in small],
            orders["A1_ratio_minus_one"],
            "o",
        ),
        (
            "RMS(E_NL - E_lin)",
            [row["difference_rms"] for row in small],
            orders["difference_rms"],
            "s",
        ),
        (
            "A2",
            [
                row["A1_nonlinear"] * row["A2_over_A1_nonlinear"]
                for row in small
            ],
            orders["A2"],
            "^",
        ),
        (
            "A3",
            [
                row["A1_nonlinear"] * row["A3_over_A1_nonlinear"]
                for row in small
            ],
            orders["A3"],
            "d",
        ),
    )
    figure, axis = plt.subplots(figsize=(6.2, 4.2))
    for label, values, order, marker in scaling_series:
        axis.loglog(
            small_m, values, marker=marker, label=f"{label}: m^{order:.3f}"
        )
    axis.set(xlabel="Modulation depth m", ylabel="Magnitude")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(figure_dir / "small_m_scaling.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(6.2, 7.6), sharex=True)
    for axis, selected_m in zip(axes, REPRESENTATIVE_MODULATIONS):
        selected = [row for row in profiles if row["m"] == selected_m]
        axis.plot(
            [row["x_um"] for row in selected],
            [row["E_nonlinear"] for row in selected],
            label="nonlinear",
        )
        axis.plot(
            [row["x_um"] for row in selected],
            [row["E_linearized"] for row in selected],
            "--",
            label="linearized",
        )
        axis.set_ylabel("E(x)")
        axis.set_title(f"m = {selected_m:g}", loc="left", fontsize=10)
        axis.grid(True, alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("x (um)")
    figure.tight_layout()
    figure.savefig(figure_dir / "nonlinear_vs_linearized_profiles.png", dpi=180)
    plt.close(figure)

    strong = [
        row
        for row in spectra
        if row["m"] == max(REPRESENTATIVE_MODULATIONS)
    ]
    harmonics = np.asarray([int(row["harmonic"]) for row in strong])
    figure, axis = plt.subplots(figsize=(5.6, 3.8))
    width = 0.34
    axis.bar(
        harmonics - width / 2,
        [row["nonlinear_amplitude"] for row in strong],
        width,
        label="nonlinear",
    )
    axis.bar(
        harmonics + width / 2,
        [row["linearized_amplitude"] for row in strong],
        width,
        label="linearized",
    )
    axis.set(xticks=harmonics, xlabel="Harmonic n", ylabel="Amplitude A_n")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "harmonic_spectrum_strong_modulation.png", dpi=180)
    plt.close(figure)


def write_evidence(
    output_dir: Path,
    *,
    rows: Sequence[dict[str, Any]],
    profiles: Sequence[dict[str, float]],
    spectra: Sequence[dict[str, float]],
    provenance: dict[str, Any],
    provisional_comparison: dict[str, Any],
) -> None:
    """Write clean compact evidence, figures, and their checksum manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", rows)
    _write_dict_csv(output_dir / "representative_profiles.csv", profiles)
    _write_dict_csv(output_dir / "representative_spectra.csv", spectra)
    _write_figures(output_dir, rows, profiles, spectra)
    artifacts = [
        output_dir / "summary.csv",
        output_dir / "representative_profiles.csv",
        output_dir / "representative_spectra.csv",
        *sorted((output_dir / "figures").glob("*.png")),
    ]
    manifest = {
        **provenance,
        "analysis": "reduced 1-D material-only nonlinear/linearized harmonic sweep",
        "fixture": {
            "I0": REFERENCE_INTENSITY,
            "applied_field": APPLIED_FIELD,
            "background_intensity": BACKGROUND_INTENSITY,
            "k_rad_per_um": WAVEVECTOR_RAD_PER_UM,
            "domain_length_um": DOMAIN_LENGTH_UM,
            "N": GRID_SIZE,
            "modulations": list(MODULATIONS),
            "representative_modulations": list(REPRESENTATIVE_MODULATIONS),
        },
        "scientific_rows_sha256": _canonical_sha256(list(rows)),
        "profile_rows_sha256": _canonical_sha256(list(profiles)),
        "spectrum_rows_sha256": _canonical_sha256(list(spectra)),
        "artifact_sha256": {
            path.relative_to(output_dir).as_posix(): _sha256(path)
            for path in artifacts
        },
        "small_m_exponents": small_modulation_orders(rows),
        "provisional_comparison": provisional_comparison,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "matplotlib_version": version("matplotlib"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _production_provenance(expected_commit: str) -> dict[str, Any]:
    actual = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), text=True
    ).strip()
    if actual != expected_commit:
        raise RuntimeError(f"HEAD {actual} does not match {expected_commit}")
    status = subprocess.check_output(
        ("git", "status", "--porcelain=v1"), text=True
    )
    if status:
        raise RuntimeError("production source checkout is not clean")
    return {
        "production_source_commit": actual,
        "production_source_clean": True,
        "production_source_status_porcelain": status,
        "harness_sha256": _sha256(Path(__file__)),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provisional-csv", type=Path, required=True)
    parser.add_argument("--expected-commit", default=BASELINE_COMMIT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    provenance = _production_provenance(args.expected_commit)
    rows, profiles, spectra = run_evidence()
    comparison = compare_with_provisional(rows, args.provisional_csv)
    write_evidence(
        args.output_dir,
        rows=rows,
        profiles=profiles,
        spectra=spectra,
        provenance=provenance,
        provisional_comparison=comparison,
    )
    for row in rows:
        print(
            f"m={row['m']:.2f} A1_NL={row['A1_nonlinear']:.10g} "
            f"A1_lin={row['A1_linearized']:.10g} "
            f"ratio={row['A1_ratio']:.10g} "
            f"A2/A1={row['A2_over_A1_nonlinear']:.10g} "
            f"A3/A1={row['A3_over_A1_nonlinear']:.10g} "
            f"Delta_phi={row['delta_phase_rad']:.3e} "
            f"rel_L2={row['relative_L2_error']:.10g}"
        )
    orders = small_modulation_orders(rows)
    print(
        "small-m powers: "
        + ", ".join(f"{name}={value:.6f}" for name, value in orders.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
