#!/usr/bin/env python3
"""Kwak-parameter, material-only reduced-PR harmonic sweep."""

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
import sys
from typing import Any, Sequence

import numpy as np

from lcprop.pr.reduced_linearized import (
    PRReducedLinearizedSpec,
    solve_pr_reduced_linearized_intensity,
)
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.static import PRStaticSolverOptions, solve_pr_static_intensity_batched


PRODUCTION_COMMIT = "d21a95e816d6750bab2e7b5e9f18cfc1ece6e5cf"
WAVELENGTH_NM = 514.5
PERIOD_UM = 2.2
KG_RAD_PER_UM = 2.0 * math.pi / PERIOD_UM
REFERENCE_INTENSITY = 1.0
APPLIED_FIELD = 0.0
RELATIVE_PERMITTIVITY = 513.0
MOBILE_CHARGE_DENSITY_M3 = 2.0e22
TEMPERATURE_K = 293.0
GRID_SIZE = 512
MODULATIONS = (
    0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60,
    0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999,
)
DARK_RATIOS = (0.0, 1.0e-6, 1.0e-4, 1.0e-2)
DARK_MODULATIONS = (0.95, 0.99, 0.999)
RESOLUTION_GRID_SIZES = (256, 512, 1024)
RESOLUTION_CASES = ((0.95, 0.0), (0.999, 0.0), (0.999, 1.0e-4))

MATERIAL = PRMaterialSpec(
    dark_intensity=0.0,
    uniform_background_intensity=0.0,
    applied_field=APPLIED_FIELD,
    relative_permittivity=RELATIVE_PERMITTIVITY,
    mobile_charge_density_m3=MOBILE_CHARGE_DENSITY_M3,
    temperature_K=TEMPERATURE_K,
)
KD_PER_UM = MATERIAL.characteristic_wavenumber_per_um

SUMMARY_FIELDS = (
    "m", "Id_over_I0", "A1_lin", "A1_NL", "R1", "R1_minus_one",
    "A2_over_A1", "A3_over_A1", "A4_over_A1", "A5_over_A1",
    "phase_diff_rad", "relative_L2", "residual_rms", "residual_max",
    "iterations", "converged", "Nx", "domain_length_um", "kg_rad_per_um",
    "kg_over_kD", "dx_um", "dx_normalized",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def repository_state(path: Path) -> dict[str, Any]:
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain=v1"), cwd=path, check=True,
        capture_output=True, text=True,
    ).stdout
    return {"commit": head, "status_porcelain": status, "clean": status == ""}


def periodic_intensity(
    modulation: float, grid_size: int = GRID_SIZE, *, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray, float]:
    m = float(modulation)
    n = int(grid_size)
    if not 0.0 < m < 1.0 or not math.isfinite(m):
        raise ValueError("modulation must be finite and in (0, 1)")
    if n < 16:
        raise ValueError("grid_size must be at least 16")
    dx_um = PERIOD_UM / n
    x_um = np.arange(n, dtype=np.float64) * dx_um
    intensity = float(scale) * REFERENCE_INTENSITY * (
        1.0 + m * np.sin(KG_RAD_PER_UM * x_um)
    )
    return x_um, intensity[:, None], dx_um


def harmonic_coefficient(field: np.ndarray, harmonic: int) -> complex:
    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("field must have shape (Nx, 1)")
    n = int(harmonic)
    if not 1 <= n <= 5:
        raise ValueError("harmonic must be in [1, 5]")
    centered = values[:, 0] - np.mean(values[:, 0])
    return complex(np.fft.fft(centered)[n] / values.shape[0])


def wrap_phase(value: float) -> float:
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def solve_case(
    modulation: float,
    dark_ratio: float = 0.0,
    grid_size: int = GRID_SIZE,
    *,
    intensity_scale: float = 1.0,
) -> dict[str, Any]:
    _x_um, optical_intensity, dx_um = periodic_intensity(
        modulation, grid_size, scale=intensity_scale
    )
    optical_reference = REFERENCE_INTENSITY * float(intensity_scale)
    background = float(dark_ratio) * optical_reference
    intensity = optical_intensity + background
    reference = optical_reference + background
    dx_normalized = KD_PER_UM * dx_um
    nonlinear = solve_pr_static_intensity_batched(
        intensity,
        applied_field=APPLIED_FIELD,
        background_intensity=background,
        dx_normalized=dx_normalized,
        options=PRStaticSolverOptions(
            max_iterations=100,
            residual_rms_tolerance=1.0e-12,
            residual_max_tolerance=1.0e-11,
        ),
    )
    if not nonlinear.converged:
        raise RuntimeError(
            f"nonlinear solve failed: m={modulation}, Id/I0={dark_ratio}, "
            f"Nx={grid_size}, status={nonlinear.status}"
        )
    linearized = solve_pr_reduced_linearized_intensity(
        intensity,
        spec=PRReducedLinearizedSpec(
            reference_intensity=reference,
            applied_field=APPLIED_FIELD,
            background_intensity=background,
            dx_normalized=dx_normalized,
        ),
    )
    coefficients = {
        n: harmonic_coefficient(nonlinear.E, n) for n in range(1, 6)
    }
    linear_coefficient = harmonic_coefficient(linearized.E, 1)
    a1_nl = 2.0 * abs(coefficients[1])
    a1_lin = 2.0 * abs(linear_coefficient)
    difference = np.asarray(nonlinear.E) - np.asarray(linearized.E)
    row = {
        "m": float(modulation),
        "Id_over_I0": float(dark_ratio),
        "A1_lin": a1_lin,
        "A1_NL": a1_nl,
        "R1": a1_nl / a1_lin,
        "R1_minus_one": a1_nl / a1_lin - 1.0,
        "A2_over_A1": abs(coefficients[2]) / abs(coefficients[1]),
        "A3_over_A1": abs(coefficients[3]) / abs(coefficients[1]),
        "A4_over_A1": abs(coefficients[4]) / abs(coefficients[1]),
        "A5_over_A1": abs(coefficients[5]) / abs(coefficients[1]),
        "phase_diff_rad": wrap_phase(
            np.angle(coefficients[1]) - np.angle(linear_coefficient)
        ),
        "relative_L2": float(
            np.linalg.norm(difference) / np.linalg.norm(linearized.E)
        ),
        "residual_rms": float(nonlinear.residual_rms),
        "residual_max": float(nonlinear.residual_max),
        "iterations": int(nonlinear.iterations),
        "converged": bool(nonlinear.converged),
        "Nx": int(grid_size),
        "domain_length_um": PERIOD_UM,
        "kg_rad_per_um": KG_RAD_PER_UM,
        "kg_over_kD": KG_RAD_PER_UM / KD_PER_UM,
        "dx_um": dx_um,
        "dx_normalized": dx_normalized,
    }
    return row


def run_all() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    primary = [solve_case(m) for m in MODULATIONS]
    primary_by_m = {row["m"]: row for row in primary}
    dark = []
    for dark_ratio in DARK_RATIOS:
        for m in DARK_MODULATIONS:
            dark.append(
                primary_by_m[m] if dark_ratio == 0.0 else solve_case(m, dark_ratio)
            )
    resolution = []
    for m, dark_ratio in RESOLUTION_CASES:
        for grid_size in RESOLUTION_GRID_SIZES:
            if grid_size == GRID_SIZE and dark_ratio == 0.0:
                resolution.append(primary_by_m[m])
            else:
                resolution.append(solve_case(m, dark_ratio, grid_size))
    return primary, dark, resolution


def small_m_fits(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    small = [row for row in rows if row["m"] <= 0.1]
    m = np.asarray([row["m"] for row in small], dtype=np.float64)
    correction = np.asarray([row["R1_minus_one"] for row in small])
    c_value = float(np.dot(m * m, correction) / np.dot(m * m, m * m))
    a2 = np.asarray([row["A1_NL"] * row["A2_over_A1"] for row in small])
    a3 = np.asarray([row["A1_NL"] * row["A3_over_A1"] for row in small])
    return {
        "C_R1": c_value,
        "R1_minus_one_exponent": float(np.polyfit(np.log(m), np.log(correction), 1)[0]),
        "A2_exponent": float(np.polyfit(np.log(m), np.log(a2), 1)[0]),
        "A3_exponent": float(np.polyfit(np.log(m), np.log(a3), 1)[0]),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_figures(
    output_dir: Path,
    primary: Sequence[dict[str, Any]],
    dark: Sequence[dict[str, Any]],
    resolution: Sequence[dict[str, Any]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    m = np.asarray([row["m"] for row in primary])

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(m, [row["R1"] for row in primary], "o-", label="reduced nonlinear / linearized")
    ax.axhline(1.0, color="0.35", linestyle="--", label="small-signal linear")
    ax.plot(m, 1.0 / (1.0 + 1.70 * m), label="Kwak empirical")
    ax.plot(m, (1.0 - np.exp(-2.58 * m)) / (2.58 * m), label="Refregier/Kwak alternative")
    ax.set(xlabel="Modulation depth m", ylabel="Fundamental ratio")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figure_dir / "fundamental_ratio_comparison.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for key, label, marker in (("A2_over_A1", "A2/A1", "o"), ("A3_over_A1", "A3/A1", "s"), ("A4_over_A1", "A4/A1", "^"), ("A5_over_A1", "A5/A1", "d")):
        ax.plot(m, [row[key] for row in primary], marker + "-", label=label)
    ax.set(xlabel="Modulation depth m", ylabel="Nonlinear harmonic ratio")
    ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(figure_dir / "harmonic_ratios.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(m, [row["phase_diff_rad"] for row in primary], "o-")
    ax.set(xlabel="Modulation depth m", ylabel="Fundamental phase difference (rad)")
    ax.grid(True, alpha=0.25); fig.tight_layout()
    fig.savefig(figure_dir / "fundamental_phase_difference.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for dark_ratio in DARK_RATIOS:
        selected = [row for row in dark if row["Id_over_I0"] == dark_ratio]
        ax.plot([row["m"] for row in selected], [row["R1"] for row in selected], "o-", label=f"Id/I0={dark_ratio:g}")
    ax.set(xlabel="Modulation depth m", ylabel="Fundamental ratio R1")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figure_dir / "dark_background_sensitivity.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for m_value, dark_ratio in RESOLUTION_CASES:
        selected = [row for row in resolution if row["m"] == m_value and row["Id_over_I0"] == dark_ratio]
        ax.plot([row["Nx"] for row in selected], [row["R1"] for row in selected], "o-", label=f"m={m_value:g}, Id/I0={dark_ratio:g}")
    ax.set(xlabel="Nx", ylabel="Fundamental ratio R1")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(figure_dir / "resolution_comparison.png", dpi=180); plt.close(fig)


def write_evidence(
    output_dir: Path,
    primary: Sequence[dict[str, Any]],
    dark: Sequence[dict[str, Any]],
    resolution: Sequence[dict[str, Any]],
    *,
    source_dir: Path,
    main_repo: Path,
    report_path: Path,
    test_path: Path,
) -> dict[str, Any]:
    write_csv(output_dir / "summary.csv", primary)
    write_csv(output_dir / "dark_sensitivity.csv", dark)
    write_csv(output_dir / "resolution.csv", resolution)
    fits = small_m_fits(primary)
    (output_dir / "small_m_fits.json").write_text(
        json.dumps(fits, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_figures(output_dir, primary, dark, resolution)
    artifacts = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    source = repository_state(source_dir)
    if source != {"commit": PRODUCTION_COMMIT, "status_porcelain": "", "clean": True}:
        raise RuntimeError(f"authoritative source is not exact and clean: {source}")
    normal_repository = repository_state(main_repo)
    manifest = {
        "analysis": "Kwak-parameter reduced PR material harmonic sweep",
        "production_commit": source["commit"],
        "source_clean_before_overlay": source["clean"],
        "production_status_porcelain": source["status_porcelain"],
        "analysis_harness_path": str(Path(__file__).resolve()),
        "analysis_harness_sha256": sha256_file(Path(__file__)),
        "analysis_test_path": str(test_path.resolve()),
        "analysis_test_sha256": sha256_file(test_path),
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "normal_repository": normal_repository,
        "normal_repository_status_sha256": hashlib.sha256(
            normal_repository["status_porcelain"].encode("utf-8")
        ).hexdigest(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": version("scipy"),
        "matplotlib_version": version("matplotlib"),
        "fixture": {
            "wavelength_nm_context_only": WAVELENGTH_NM,
            "period_um": PERIOD_UM,
            "kg_rad_per_um": KG_RAD_PER_UM,
            "relative_permittivity": RELATIVE_PERMITTIVITY,
            "mobile_charge_density_m3": MOBILE_CHARGE_DENSITY_M3,
            "temperature_K": TEMPERATURE_K,
            "characteristic_wavenumber_per_um": KD_PER_UM,
            "kg_over_kD": KG_RAD_PER_UM / KD_PER_UM,
            "I0": REFERENCE_INTENSITY,
            "applied_field": APPLIED_FIELD,
            "primary_Id_over_I0": 0.0,
            "dark_mapping": "driving_intensity=I0*(1+m*sin(Kx))+Id; background_intensity=Id",
            "modulations": list(MODULATIONS),
        },
        "superseded_attempts": [
            "Python 3.9 import failed before solver execution",
            "initial completed sweep omitted Id from driving intensity and was replaced after source-convention review",
        ],
        "canonical_scientific_data_sha256": canonical_sha256({
            "primary": list(primary), "dark": list(dark), "resolution": list(resolution), "fits": fits,
        }),
        "artifact_sha256": {
            path.relative_to(output_dir).as_posix(): sha256_file(path)
            for path in artifacts
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--main-repo", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = repository_state(args.source_dir)
    if source["commit"] != PRODUCTION_COMMIT or not source["clean"]:
        raise RuntimeError(f"production checkout failed preflight: {source}")
    primary, dark, resolution = run_all()
    write_evidence(
        args.output_dir, primary, dark, resolution,
        source_dir=args.source_dir, main_repo=args.main_repo,
        report_path=args.report_path, test_path=args.test_path,
    )
    for row in primary:
        print(f"m={row['m']:.3f} R1={row['R1']:.12f} A2/A1={row['A2_over_A1']:.9f} A3/A1={row['A3_over_A1']:.9f} phase={row['phase_diff_rad']:.3e}")
    print(json.dumps(small_m_fits(primary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
