#!/usr/bin/env python3
"""Map reduced-PR material nonlinearity over grating frequency and contrast."""

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


BASELINE_COMMIT = "c3edb6f46fb82a0e158b98fa599f49f406687a13"
GRID_SIZE = 512
REFERENCE_INTENSITY = 1.0
APPLIED_FIELD = 0.0
BACKGROUND_INTENSITY = 0.0
CHARACTERISTIC_WAVENUMBER_PER_UM = (
    PRMaterialSpec().characteristic_wavenumber_per_um
)
WAVENUMBERS_RAD_PER_UM = (
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    math.pi,
    CHARACTERISTIC_WAVENUMBER_PER_UM,
    6.0,
    8.0,
)
MODULATIONS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95)
LOW_MODULATION_MAX = 0.1
RESOLUTION_WAVENUMBERS = (0.25, math.pi, 8.0)
RESOLUTION_GRID_SIZES = (256, 512, 1024)
PRIMARY_RESIDUAL_RMS_TOLERANCE = 1.0e-12
PRIMARY_RESIDUAL_MAX_TOLERANCE = 1.0e-11
FINE_RESIDUAL_RMS_TOLERANCE = 5.0e-12
FINE_RESIDUAL_MAX_TOLERANCE = 2.0e-11
SELECTED_WAVENUMBERS = (
    0.25,
    1.0,
    math.pi,
    CHARACTERISTIC_WAVENUMBER_PER_UM,
    8.0,
)

SUMMARY_FIELDS = (
    "kg_rad_per_um",
    "kg_over_kD",
    "domain_length_um",
    "grid_size",
    "dx_um",
    "dx_normalized",
    "m",
    "A1_linearized",
    "A1_nonlinear",
    "R_K",
    "R_K_minus_one",
    "A2_over_A1_nonlinear",
    "A3_over_A1_nonlinear",
    "A4_over_A1_nonlinear",
    "A5_over_A1_nonlinear",
    "delta_phase_rad",
    "relative_L2_error",
    "residual_rms_tolerance",
    "residual_max_tolerance",
    "nonlinear_iterations",
    "nonlinear_residual_rms",
    "nonlinear_residual_max",
)
FIT_FIELDS = (
    "kg_rad_per_um",
    "kg_over_kD",
    "C_RK",
    "R_K_minus_one_exponent",
    "C_fit_relative_rms",
    "A2_exponent",
    "A3_exponent",
)
RESOLUTION_FIELDS = SUMMARY_FIELDS


def repository_state(path: Path) -> dict[str, Any]:
    """Return exact HEAD and porcelain status for a production checkout."""

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": head, "status_porcelain": status, "clean": status == ""}


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(rows: Sequence[dict[str, Any]]) -> str:
    """Hash scientific records independently of CSV formatting."""

    payload = json.dumps(
        list(rows), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def periodic_intensity(
    kg_rad_per_um: float, modulation: float, grid_size: int
) -> tuple[np.ndarray, float, float]:
    """Return one exact period and its physical length and sample spacing."""

    kg = float(kg_rad_per_um)
    m = float(modulation)
    n = int(grid_size)
    if not math.isfinite(kg) or kg <= 0.0:
        raise ValueError("kg_rad_per_um must be finite and positive")
    if not math.isfinite(m) or not 0.0 < m < 1.0:
        raise ValueError("modulation must be finite and in (0, 1)")
    if n < 16:
        raise ValueError("grid_size must be at least 16")
    length_um = 2.0 * math.pi / kg
    dx_um = length_um / n
    x_um = np.arange(n, dtype=np.float64) * dx_um
    intensity = REFERENCE_INTENSITY * (1.0 + m * np.sin(kg * x_um))
    return intensity[:, None], length_um, dx_um


def harmonic_coefficient(field: np.ndarray, harmonic: int) -> complex:
    """Return c_n=N^-1 sum_j[(E_j-mean(E))*exp(-i*2*pi*n*j/N)]."""

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("field must have shape (N, 1)")
    n = int(harmonic)
    if n < 1 or n > 5:
        raise ValueError("harmonic must be in [1, 5]")
    centered = values[:, 0] - np.mean(values[:, 0])
    return complex(np.fft.fft(centered)[n] / values.shape[0])


def wrap_phase_radians(value: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def solve_case(
    kg_rad_per_um: float,
    modulation: float,
    grid_size: int = GRID_SIZE,
    *,
    residual_rms_tolerance: float = PRIMARY_RESIDUAL_RMS_TOLERANCE,
    residual_max_tolerance: float = PRIMARY_RESIDUAL_MAX_TOLERANCE,
) -> dict[str, Any]:
    """Solve one frozen-intensity nonlinear/linearized material pair."""

    intensity, length_um, dx_um = periodic_intensity(
        kg_rad_per_um, modulation, grid_size
    )
    dx_normalized = CHARACTERISTIC_WAVENUMBER_PER_UM * dx_um
    nonlinear = solve_pr_static_intensity_batched(
        intensity,
        applied_field=APPLIED_FIELD,
        background_intensity=BACKGROUND_INTENSITY,
        dx_normalized=dx_normalized,
        options=PRStaticSolverOptions(
            max_iterations=80,
            residual_rms_tolerance=residual_rms_tolerance,
            residual_max_tolerance=residual_max_tolerance,
        ),
    )
    if not nonlinear.converged:
        raise RuntimeError(
            "nonlinear solve failed for "
            f"kg={kg_rad_per_um}, m={modulation}, N={grid_size}: "
            f"{nonlinear.status}"
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
    nonlinear_coefficients = {
        harmonic: harmonic_coefficient(nonlinear.E, harmonic)
        for harmonic in range(1, 6)
    }
    linearized_fundamental = harmonic_coefficient(linearized.E, 1)
    a1_nonlinear = 2.0 * abs(nonlinear_coefficients[1])
    a1_linearized = 2.0 * abs(linearized_fundamental)
    difference = np.asarray(nonlinear.E) - np.asarray(linearized.E)
    ratio = a1_nonlinear / a1_linearized
    return {
        "kg_rad_per_um": float(kg_rad_per_um),
        "kg_over_kD": float(
            kg_rad_per_um / CHARACTERISTIC_WAVENUMBER_PER_UM
        ),
        "domain_length_um": length_um,
        "grid_size": int(grid_size),
        "dx_um": dx_um,
        "dx_normalized": dx_normalized,
        "m": float(modulation),
        "A1_linearized": a1_linearized,
        "A1_nonlinear": a1_nonlinear,
        "R_K": ratio,
        "R_K_minus_one": ratio - 1.0,
        "A2_over_A1_nonlinear": abs(nonlinear_coefficients[2])
        / abs(nonlinear_coefficients[1]),
        "A3_over_A1_nonlinear": abs(nonlinear_coefficients[3])
        / abs(nonlinear_coefficients[1]),
        "A4_over_A1_nonlinear": abs(nonlinear_coefficients[4])
        / abs(nonlinear_coefficients[1]),
        "A5_over_A1_nonlinear": abs(nonlinear_coefficients[5])
        / abs(nonlinear_coefficients[1]),
        "delta_phase_rad": wrap_phase_radians(
            np.angle(nonlinear_coefficients[1])
            - np.angle(linearized_fundamental)
        ),
        "relative_L2_error": float(
            np.linalg.norm(difference) / np.linalg.norm(linearized.E)
        ),
        "residual_rms_tolerance": float(residual_rms_tolerance),
        "residual_max_tolerance": float(residual_max_tolerance),
        "nonlinear_iterations": int(nonlinear.iterations),
        "nonlinear_residual_rms": float(nonlinear.residual_rms),
        "nonlinear_residual_max": float(nonlinear.residual_max),
    }


def observed_power(x: Sequence[float], y: Sequence[float]) -> float:
    """Return a log-log least-squares exponent for positive samples."""

    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if (
        x_values.shape != y_values.shape
        or x_values.size < 2
        or np.any(x_values <= 0.0)
        or np.any(y_values <= 0.0)
    ):
        raise ValueError("power-law samples must be matched and positive")
    return float(np.polyfit(np.log(x_values), np.log(y_values), 1)[0])


def fit_small_modulation(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fit C(k_g) and perturbative powers over m<=0.1 for each frequency."""

    fits: list[dict[str, Any]] = []
    for kg in WAVENUMBERS_RAD_PER_UM:
        selected = [
            row
            for row in rows
            if row["kg_rad_per_um"] == kg and row["m"] <= LOW_MODULATION_MAX
        ]
        m = np.asarray([row["m"] for row in selected])
        enhancement = np.asarray([row["R_K_minus_one"] for row in selected])
        m_squared = m * m
        coefficient = float(
            np.dot(m_squared, enhancement) / np.dot(m_squared, m_squared)
        )
        fit_residual = enhancement - coefficient * m_squared
        fits.append({
            "kg_rad_per_um": float(kg),
            "kg_over_kD": float(kg / CHARACTERISTIC_WAVENUMBER_PER_UM),
            "C_RK": coefficient,
            "R_K_minus_one_exponent": observed_power(m, enhancement),
            "C_fit_relative_rms": float(
                np.linalg.norm(fit_residual) / np.linalg.norm(enhancement)
            ),
            "A2_exponent": observed_power(
                m,
                [
                    row["A1_nonlinear"]
                    * row["A2_over_A1_nonlinear"]
                    for row in selected
                ],
            ),
            "A3_exponent": observed_power(
                m,
                [
                    row["A1_nonlinear"]
                    * row["A3_over_A1_nonlinear"]
                    for row in selected
                ],
            ),
        })
    return fits


def run_sweep() -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
]:
    """Run the fixed map and limited resolution checks."""

    rows = [
        solve_case(kg, modulation, GRID_SIZE)
        for kg in WAVENUMBERS_RAD_PER_UM
        for modulation in MODULATIONS
    ]
    fits = fit_small_modulation(rows)
    resolution: list[dict[str, Any]] = []
    for kg in RESOLUTION_WAVENUMBERS:
        for grid_size in RESOLUTION_GRID_SIZES:
            if grid_size == GRID_SIZE:
                resolution.append(next(
                    row.copy()
                    for row in rows
                    if row["kg_rad_per_um"] == kg and row["m"] == 0.95
                ))
            else:
                resolution.append(solve_case(
                    kg,
                    0.95,
                    grid_size,
                    residual_rms_tolerance=(
                        FINE_RESIDUAL_RMS_TOLERANCE
                        if grid_size == 1024
                        else PRIMARY_RESIDUAL_RMS_TOLERANCE
                    ),
                    residual_max_tolerance=(
                        FINE_RESIDUAL_MAX_TOLERANCE
                        if grid_size == 1024
                        else PRIMARY_RESIDUAL_MAX_TOLERANCE
                    ),
                ))
    return rows, fits, resolution


def _write_csv(
    path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _mesh(rows: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray([
        [
            next(
                row[field]
                for row in rows
                if row["kg_rad_per_um"] == kg and row["m"] == modulation
            )
            for kg in WAVENUMBERS_RAD_PER_UM
        ]
        for modulation in MODULATIONS
    ])


def _centers_to_edges(values: Sequence[float]) -> np.ndarray:
    centers = np.asarray(values, dtype=np.float64)
    mids = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate((
        [centers[0] - (mids[0] - centers[0])],
        mids,
        [centers[-1] + (centers[-1] - mids[-1])],
    ))


def write_figures(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    fits: Sequence[dict[str, Any]],
) -> None:
    """Create publication-quality map and selected-curve figures."""

    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    kg_edges = _centers_to_edges(WAVENUMBERS_RAD_PER_UM)
    m_edges = _centers_to_edges(MODULATIONS)
    heatmaps = (
        ("R_K_minus_one", r"Fundamental enhancement $R_K-1$", "fundamental_enhancement_heatmap.png"),
        ("A2_over_A1_nonlinear", r"Nonlinear $A_2/A_1$", "second_harmonic_heatmap.png"),
        ("A3_over_A1_nonlinear", r"Nonlinear $A_3/A_1$", "third_harmonic_heatmap.png"),
    )
    for field, label, filename in heatmaps:
        figure, axis = plt.subplots(figsize=(6.5, 4.3))
        image = axis.pcolormesh(
            kg_edges,
            m_edges,
            _mesh(rows, field),
            shading="flat",
            cmap="viridis",
        )
        if field == "R_K_minus_one":
            axis.plot(math.pi, 0.95, marker="*", ms=13, color="white", mec="black")
        axis.set(xlabel=r"$k_g$ (rad/$\mu$m)", ylabel="Modulation depth m")
        figure.colorbar(image, ax=axis, label=label)
        figure.tight_layout()
        figure.savefig(figure_dir / filename, dpi=200)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.2, 4.1))
    axis.plot(
        [row["kg_over_kD"] for row in fits],
        [row["C_RK"] for row in fits],
        "o-",
    )
    axis.axvline(1.0, color="0.45", ls="--", lw=1.0, label=r"$k_g/k_D=1$")
    axis.set(
        xlabel=r"Dimensionless frequency $k_g/k_D$",
        ylabel=r"Small-$m$ coefficient $C(k_g)$",
    )
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "small_m_coefficient_vs_frequency.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 4.3))
    for kg in SELECTED_WAVENUMBERS:
        selected = [row for row in rows if row["kg_rad_per_um"] == kg]
        axis.plot(
            [row["m"] for row in selected],
            [row["R_K"] for row in selected],
            marker="o",
            ms=3.5,
            label=rf"$k_g={kg:.4g}$",
        )
    axis.axhline(1.0, color="0.45", ls="--", lw=1.0)
    axis.set(xlabel="Modulation depth m", ylabel=r"$R_K$")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(figure_dir / "selected_fundamental_ratio_curves.png", dpi=200)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(6.5, 7.0), sharex=True)
    for kg in SELECTED_WAVENUMBERS:
        selected = [row for row in rows if row["kg_rad_per_um"] == kg]
        axes[0].plot(
            [row["m"] for row in selected],
            [row["A2_over_A1_nonlinear"] for row in selected],
            marker="o",
            ms=3,
            label=rf"$k_g={kg:.4g}$",
        )
        axes[1].plot(
            [row["m"] for row in selected],
            [row["A3_over_A1_nonlinear"] for row in selected],
            marker="o",
            ms=3,
        )
    axes[0].set_ylabel(r"$A_2/A_1$")
    axes[1].set(xlabel="Modulation depth m", ylabel=r"$A_3/A_1$")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(figure_dir / "selected_harmonic_ratio_curves.png", dpi=200)
    plt.close(figure)


def write_evidence(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    fits: Sequence[dict[str, Any]],
    resolution: Sequence[dict[str, Any]],
    provenance: dict[str, Any],
) -> None:
    """Write compact evidence and its checksum manifest."""

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "summary.csv", rows, SUMMARY_FIELDS)
    _write_csv(output_dir / "small_m_fits.csv", fits, FIT_FIELDS)
    _write_csv(output_dir / "resolution.csv", resolution, RESOLUTION_FIELDS)
    write_figures(output_dir, rows, fits)
    artifact_paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": 1,
        "study": "reduced_pr_material_kg_m_nonlinearity_map",
        "production_source": provenance,
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "configuration": {
            "characteristic_wavenumber_per_um": CHARACTERISTIC_WAVENUMBER_PER_UM,
            "dimensionless_frequency": "kg_over_kD = kg_rad_per_um / kD",
            "kg_rad_per_um": list(WAVENUMBERS_RAD_PER_UM),
            "modulations": list(MODULATIONS),
            "grid_size": GRID_SIZE,
            "one_exact_period": True,
            "reference_intensity": REFERENCE_INTENSITY,
            "applied_field": APPLIED_FIELD,
            "background_intensity": BACKGROUND_INTENSITY,
            "resolution_kg_rad_per_um": list(RESOLUTION_WAVENUMBERS),
            "resolution_grid_sizes": list(RESOLUTION_GRID_SIZES),
            "primary_residual_tolerances": {
                "rms": PRIMARY_RESIDUAL_RMS_TOLERANCE,
                "max": PRIMARY_RESIDUAL_MAX_TOLERANCE,
            },
            "N1024_resolution_residual_tolerances": {
                "rms": FINE_RESIDUAL_RMS_TOLERANCE,
                "max": FINE_RESIDUAL_MAX_TOLERANCE,
            },
        },
        "canonical_sha256": {
            "scientific_rows": canonical_sha256(rows),
            "small_m_fits": canonical_sha256(fits),
            "resolution_rows": canonical_sha256(resolution),
        },
        "artifacts": {
            str(path.relative_to(output_dir)): sha256_file(path)
            for path in artifact_paths
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": version("matplotlib"),
        },
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_before = repository_state(Path.cwd())
    if source_before["commit"] != BASELINE_COMMIT:
        raise SystemExit(
            f"production HEAD {source_before['commit']} != {BASELINE_COMMIT}"
        )
    if not source_before["clean"]:
        raise SystemExit("production source is dirty; refusing authoritative run")
    rows, fits, resolution = run_sweep()
    source_after = repository_state(Path.cwd())
    if source_after != source_before:
        raise SystemExit("production source changed during authoritative run")
    write_evidence(
        args.output_dir,
        rows,
        fits,
        resolution,
        {
            "commit": source_before["commit"],
            "clean_before": source_before["clean"],
            "clean_after": source_after["clean"],
            "status_porcelain_before": source_before["status_porcelain"],
            "status_porcelain_after": source_after["status_porcelain"],
        },
    )
    print(f"wrote {len(rows)} sweep rows to {args.output_dir}")


if __name__ == "__main__":
    main()
