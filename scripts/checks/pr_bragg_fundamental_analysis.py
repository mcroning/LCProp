#!/usr/bin/env python3
"""Analyze retained Bragg-matched PR fundamental coefficients.

The script reads compact outputs from the headless linearization sweep. It does
not invoke a material solver, optical propagation, a scheduler, or a remote
execution target.
"""

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
from typing import Any, Iterable, Sequence

import numpy as np


NONLINEAR_MODEL = "reduced_nonlinear"
LINEARIZED_MODEL = "reduced_linearized"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(payload: Any) -> str:
    return _sha256_bytes(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def wrap_phase_radians(value: float) -> float:
    """Wrap an angle to the principal interval [-pi, pi)."""

    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def coefficient_from_polar(magnitude: float, phase_radians: float) -> complex:
    """Reconstruct a complex coefficient from retained magnitude and phase."""

    return complex(float(magnitude) * np.exp(1j * float(phase_radians)))


def _model_pairs(rows: Iterable[dict[str, Any]]) -> list[tuple[dict, dict]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in rows:
        model = row.get("model")
        if model not in (NONLINEAR_MODEL, LINEARIZED_MODEL):
            continue
        key = (
            row.get("study"),
            row.get("fixture"),
            row.get("geometry"),
            int(row["N"]),
            float(row["m"]),
        )
        grouped.setdefault(key, {})[model] = row
    pairs = []
    for key in sorted(
        grouped,
        key=lambda item: (item[0], item[1], item[2], item[3], item[4]),
    ):
        models = grouped[key]
        if set(models) != {NONLINEAR_MODEL, LINEARIZED_MODEL}:
            raise ValueError(f"incomplete nonlinear/linearized pair for {key}")
        pairs.append((models[NONLINEAR_MODEL], models[LINEARIZED_MODEL]))
    return pairs


def analyze_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair retained reduced-model rows and compute complex-K comparisons."""

    output: list[dict[str, Any]] = []
    for nonlinear, linearized in _model_pairs(rows):
        magnitude_nl = nonlinear.get("harmonic_1_magnitude")
        phase_nl = nonlinear.get("harmonic_1_phase_rad")
        magnitude_lin = linearized.get("harmonic_1_magnitude")
        phase_lin = linearized.get("harmonic_1_phase_rad")
        if None in (magnitude_nl, phase_nl, magnitude_lin, phase_lin):
            raise ValueError("the fundamental coefficient is unavailable")
        coefficient_nl = coefficient_from_polar(magnitude_nl, phase_nl)
        coefficient_lin = coefficient_from_polar(magnitude_lin, phase_lin)
        if abs(coefficient_lin) == 0.0:
            raise ValueError("the linearized fundamental coefficient is zero")
        output.append({
            "study": nonlinear["study"],
            "fixture": nonlinear["fixture"],
            "geometry": nonlinear["geometry"],
            "N": int(nonlinear["N"]),
            "m": float(nonlinear["m"]),
            "resolution_K": nonlinear["resolution_K"],
            "magnitude_nonlinear": abs(coefficient_nl),
            "phase_nonlinear_rad": float(np.angle(coefficient_nl)),
            "real_nonlinear": coefficient_nl.real,
            "imag_nonlinear": coefficient_nl.imag,
            "magnitude_linearized": abs(coefficient_lin),
            "phase_linearized_rad": float(np.angle(coefficient_lin)),
            "real_linearized": coefficient_lin.real,
            "imag_linearized": coefficient_lin.imag,
            "R_K": abs(coefficient_nl) / abs(coefficient_lin),
            "delta_phase_rad": wrap_phase_radians(
                np.angle(coefficient_nl) - np.angle(coefficient_lin)
            ),
            "second_harmonic_ratio_nonlinear": nonlinear.get(
                "harmonic_2_relative_to_K"
            ),
            "third_harmonic_ratio_nonlinear": nonlinear.get(
                "harmonic_3_relative_to_K"
            ),
            "pump_gain_nonlinear": nonlinear.get("pump_gain"),
            "pump_gain_linearized": linearized.get("pump_gain"),
            "signal_gain_nonlinear": nonlinear.get("signal_gain"),
            "signal_gain_linearized": linearized.get("signal_gain"),
            "carrier_separation_quality": nonlinear.get(
                "carrier_separation_quality"
            ),
        })
    return output


def _repo_root() -> Path:
    return Path(
        subprocess.check_output(
            ("git", "rev-parse", "--show-toplevel"), text=True
        ).strip()
    ).resolve()


def _verify_sweep_input(
    path: Path,
    *,
    repo_root: Path,
    sweep_commit: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Verify one compact 02_30 input against Git and its own manifest."""

    resolved = path.resolve()
    relative = resolved.relative_to(repo_root).as_posix()
    committed_bytes = subprocess.check_output(
        ("git", "show", f"{sweep_commit}:{relative}"), cwd=repo_root
    )
    actual_bytes = resolved.read_bytes()
    if actual_bytes != committed_bytes:
        raise ValueError(f"02_30 input differs from {sweep_commit}: {relative}")

    document = json.loads(actual_bytes)
    rows = document.get("rows")
    manifest = document.get("manifest")
    if not isinstance(rows, list) or not isinstance(manifest, dict):
        raise ValueError(f"missing rows or manifest in {relative}")
    if manifest.get("production_source_clean") is not True:
        raise ValueError(f"02_30 input lacks clean-source provenance: {relative}")
    if manifest.get("production_source_status_porcelain") != "":
        raise ValueError(f"02_30 production source was not clean: {relative}")

    expected_payload_sha = manifest.get("scientific_payload_sha256")
    actual_payload_sha = _canonical_sha256(
        {"rows": rows, "summary": document.get("summary")}
    )
    if actual_payload_sha != expected_payload_sha:
        raise ValueError(f"02_30 scientific payload checksum mismatch: {relative}")

    artifact_hashes = manifest.get("evidence_artifact_sha256")
    if not isinstance(artifact_hashes, dict):
        raise ValueError(f"02_30 artifact checksums are unavailable: {relative}")
    for filename, expected_sha in artifact_hashes.items():
        artifact = resolved.parent / filename
        if not artifact.is_file() or _sha256_path(artifact) != expected_sha:
            raise ValueError(f"02_30 artifact checksum mismatch: {artifact}")

    metadata = {
        "path": relative,
        "sha256": _sha256_bytes(actual_bytes),
        "matches_sweep_commit": True,
        "production_source_commit": manifest.get("production_source_commit"),
        "production_source_clean": True,
        "sweep_harness_sha256": manifest.get("harness_sha256"),
        "sweep_scientific_payload_sha256": expected_payload_sha,
    }
    return rows, manifest, metadata


def compare_with_superseded(
    rows: Sequence[dict[str, Any]], superseded_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Quantify regenerated-versus-superseded scalar differences."""

    key_fields = ("study", "fixture", "geometry", "N", "m")

    def keyed(items: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
        return {tuple(row[field] for field in key_fields): row for row in items}

    current = keyed(rows)
    previous = keyed(superseded_rows)
    if current.keys() != previous.keys():
        raise ValueError("superseded evidence row keys do not match regenerated rows")

    numeric_values = 0
    changed_numeric_values = 0
    max_absolute = (0.0, None, None)
    max_relative = (0.0, None, None)
    changed_by_study: dict[str, int] = {}
    nonnumeric_differences: list[str] = []
    for key in sorted(current):
        new_row = current[key]
        old_row = previous[key]
        if new_row.keys() != old_row.keys():
            raise ValueError(f"superseded evidence fields differ for {key}")
        for field in new_row:
            new_value = new_row[field]
            old_value = old_row[field]
            if (
                isinstance(new_value, (int, float))
                and not isinstance(new_value, bool)
                and isinstance(old_value, (int, float))
                and not isinstance(old_value, bool)
            ):
                numeric_values += 1
                difference = abs(float(new_value) - float(old_value))
                if difference != 0.0:
                    changed_numeric_values += 1
                    study = str(new_row["study"])
                    changed_by_study[study] = changed_by_study.get(study, 0) + 1
                if difference > max_absolute[0]:
                    max_absolute = (difference, key, field)
                scale = max(abs(float(new_value)), abs(float(old_value)))
                if scale >= 1.0e-14:
                    relative = difference / scale
                    if relative > max_relative[0]:
                        max_relative = (relative, key, field)
            elif new_value != old_value:
                nonnumeric_differences.append(f"{key}:{field}")

    return {
        "rows_compared": len(current),
        "numeric_values_compared": numeric_values,
        "changed_numeric_values": changed_numeric_values,
        "changed_numeric_values_by_study": changed_by_study,
        "nonnumeric_differences": nonnumeric_differences,
        "max_absolute_difference": max_absolute[0],
        "max_absolute_difference_row": list(max_absolute[1]) if max_absolute[1] else None,
        "max_absolute_difference_field": max_absolute[2],
        "max_relative_difference_values_ge_1e-14": max_relative[0],
        "max_relative_difference_row": list(max_relative[1]) if max_relative[1] else None,
        "max_relative_difference_field": max_relative[2],
        "within_expected_roundoff": (
            not nonnumeric_differences
            and max_absolute[0] <= 1.0e-12
            and max_relative[0] <= 1.0e-12
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_outputs(
    output_dir: Path,
    *,
    rows: list[dict[str, Any]],
    sweep_commit: str,
    inputs: Sequence[dict[str, Any]],
    superseded_comparison: dict[str, Any] | None,
    make_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if make_plots:
        _write_plots(output_dir, rows)
    evidence_paths = [output_dir / "summary.csv", *sorted(output_dir.glob("*.png"))]
    manifest = {
        "analysis": "Bragg-matched reduced active material field",
        "fourier_convention": (
            "c(K)=sum(w*(E-mean_w(E))*exp(-i*K.r))/sum(w); "
            "rectangular weights for x and Hann-product weights for 45deg"
        ),
        "phase_interval": "[-pi, pi)",
        "sweep_commit": sweep_commit,
        "analysis_script_sha256": _sha256_path(Path(__file__)),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "matplotlib_version": version("matplotlib") if make_plots else None,
        "inputs": list(inputs),
        "evidence_artifact_sha256": {
            path.name: _sha256_path(path) for path in evidence_paths
        },
        "scientific_payload_sha256": _canonical_sha256(_json_safe(rows)),
        "superseded_evidence_comparison": superseded_comparison,
    }
    document = {"manifest": manifest, "rows": _json_safe(rows)}
    (output_dir / "summary.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    material = [row for row in rows if row["study"] == "frozen_intensity_material"]
    production = [row for row in rows if row["study"] == "production_static"]

    def material_plot(
        filename: str,
        ylabel: str,
        fields: Sequence[tuple[str, str]],
    ) -> None:
        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        for geometry in ("x", "45deg"):
            for size in (256, 512, 1024):
                selected = sorted(
                    (
                        row
                        for row in material
                        if row["geometry"] == geometry and row["N"] == size
                    ),
                    key=lambda row: row["m"],
                )
                for field, suffix in fields:
                    axis.plot(
                        [row["m"] for row in selected],
                        [row[field] for row in selected],
                        marker="o",
                        label=f"{geometry}, N={size}{suffix}",
                    )
        axis.set_xscale("log")
        axis.set(xlabel="Visibility m", ylabel=ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=6)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)

    material_plot("R_K_vs_modulation.png", "R_K", (("R_K", ""),))
    material_plot(
        "delta_phase_vs_modulation.png",
        "Delta phase (rad)",
        (("delta_phase_rad", ""),),
    )
    material_plot(
        "fundamental_magnitude_vs_modulation.png",
        "|E_hat(K)|",
        (
            ("magnitude_nonlinear", ", nonlinear"),
            ("magnitude_linearized", ", linearized"),
        ),
    )

    if production:
        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        for geometry, marker in (("x", "o"), ("45deg", "s")):
            selected = [row for row in production if row["geometry"] == geometry]
            for model, magnitude, gain in (
                ("nonlinear", "magnitude_nonlinear", "signal_gain_nonlinear"),
                ("linearized", "magnitude_linearized", "signal_gain_linearized"),
            ):
                axis.scatter(
                    [row[magnitude] for row in selected],
                    [row[gain] for row in selected],
                    marker=marker,
                    label=f"{geometry}, {model}",
                )
        axis.set(xlabel="|E_hat(K)|", ylabel="Signal carrier gain")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(
            output_dir / "carrier_gain_vs_fundamental_magnitude.png", dpi=150
        )
        plt.close(figure)

        figure, axis = plt.subplots(figsize=(7.0, 4.5))
        for geometry, marker in (("x", "o"), ("45deg", "s")):
            selected = [row for row in production if row["geometry"] == geometry]
            axis.scatter(
                [row["delta_phase_rad"] for row in selected],
                [
                    row["signal_gain_nonlinear"]
                    - row["signal_gain_linearized"]
                    for row in selected
                ],
                marker=marker,
                label=geometry,
            )
        axis.set(
            xlabel="Delta phase (rad)",
            ylabel="Signal gain: nonlinear - linearized",
        )
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(
            output_dir / "carrier_gain_difference_vs_delta_phase.png", dpi=150
        )
        plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-json", type=Path, required=True)
    parser.add_argument(
        "--production-json", type=Path, action="append", default=[]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-sweep-commit", required=True)
    parser.add_argument("--superseded-json", type=Path)
    parser.add_argument("--plots", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = _repo_root()
    current_commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=repo_root, text=True
    ).strip()
    if current_commit != args.expected_sweep_commit:
        raise ValueError(
            f"HEAD {current_commit} does not match expected sweep commit "
            f"{args.expected_sweep_commit}"
        )
    all_rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for path in (args.material_json, *args.production_json):
        rows, _manifest, metadata = _verify_sweep_input(
            path,
            repo_root=repo_root,
            sweep_commit=args.expected_sweep_commit,
        )
        all_rows.extend(rows)
        inputs.append(metadata)
    analyzed = analyze_rows(all_rows)
    if not analyzed:
        raise ValueError("no complete nonlinear/linearized pairs found")
    superseded_comparison = None
    if args.superseded_json is not None:
        superseded_document = json.loads(
            args.superseded_json.read_text(encoding="utf-8")
        )
        superseded_comparison = compare_with_superseded(
            analyzed, superseded_document["rows"]
        )
        superseded_comparison["sha256"] = _sha256_path(args.superseded_json)
    write_outputs(
        args.output_dir,
        rows=analyzed,
        sweep_commit=args.expected_sweep_commit,
        inputs=inputs,
        superseded_comparison=superseded_comparison,
        make_plots=args.plots,
    )
    material = [row for row in analyzed if row["study"] == "frozen_intensity_material"]
    for geometry in ("x", "45deg"):
        primary = next(
            row
            for row in material
            if row["geometry"] == geometry and row["N"] == 1024 and row["m"] == 0.5
        )
        print(
            f"{geometry} N=1024 m=0.5: R_K={primary['R_K']:.9f}, "
            f"Delta_phase={primary['delta_phase_rad']:.9g} rad"
        )
    print(f"wrote {len(analyzed)} paired rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
