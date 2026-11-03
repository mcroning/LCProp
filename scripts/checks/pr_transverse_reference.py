#!/usr/bin/env python3
"""Run the small CPU transverse-PR reference sanity case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lcprop.pr.transverse_reference import (
    TransverseReferenceOptions,
    run_transverse_reference,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("pr_transverse_reference_output"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    Nx, Ny = 64, 56
    dx = dy = 0.5
    x = (np.arange(Nx) - Nx / 2.0) * dx
    y = (np.arange(Ny) - Ny / 2.0) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    intensity = 0.1 + np.exp(
        -((X - 1.5) ** 2 / 12.0 + (Y + 2.0) ** 2 / 4.0)
    )
    isotropic_options = TransverseReferenceOptions(
        dx_normalized=dx,
        dy_normalized=dy,
        dt_normalized=0.002,
        steps=100,
        m_y=1.0,
        h_y=1.0,
        applied_field_x=0.0,
    )
    anisotropic_options = TransverseReferenceOptions.from_barium_titanate_c_axis(
        dx_normalized=dx,
        dy_normalized=dy,
        dt_normalized=0.002,
        steps=100,
        c_axis_xz_angle_deg=45.0,
        m_y=1.0,
        applied_field_x=0.0,
    )
    isotropic = run_transverse_reference(
        np.zeros_like(intensity),
        intensity,
        options=isotropic_options,
    )
    anisotropic = run_transverse_reference(
        np.zeros_like(intensity),
        intensity,
        options=anisotropic_options,
    )
    isotropic_final = isotropic.final_state
    anisotropic_final = anisotropic.final_state
    dielectric = anisotropic_options.dielectric
    assert dielectric is not None

    arrays_path = args.output_dir / "transverse_reference_fields.npz"
    np.savez_compressed(
        arrays_path,
        x_normalized=x,
        y_normalized=y,
        intensity=intensity,
        psi=isotropic_final.psi,
        carrier_perturbation=isotropic_final.carrier_density - 1.0,
        E_x=isotropic_final.E_x,
        E_y=isotropic_final.E_y,
        anisotropic_psi=anisotropic_final.psi,
        anisotropic_carrier_perturbation=(
            anisotropic_final.carrier_density - 1.0
        ),
        anisotropic_E_x=anisotropic_final.E_x,
        anisotropic_E_y=anisotropic_final.E_y,
    )

    fields = (
        (intensity, "Driving intensity $I$", "viridis"),
        (
            isotropic_final.carrier_density - 1.0,
            "Isotropic $P-1$",
            "RdBu_r",
        ),
        (isotropic_final.E_x, "Isotropic $E_x$", "RdBu_r"),
        (isotropic_final.E_y, "Isotropic $E_y$", "RdBu_r"),
        (intensity, "Driving intensity $I$", "viridis"),
        (
            anisotropic_final.carrier_density - 1.0,
            "BaTiO$_3$ 45° $P-1$",
            "RdBu_r",
        ),
        (anisotropic_final.E_x, "BaTiO$_3$ 45° $E_x$", "RdBu_r"),
        (anisotropic_final.E_y, "BaTiO$_3$ 45° $E_y$", "RdBu_r"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    extent = (y[0], y[-1], x[0], x[-1])
    shared_limits = (
        None,
        max(
            float(np.max(np.abs(isotropic_final.carrier_density - 1.0))),
            float(np.max(np.abs(anisotropic_final.carrier_density - 1.0))),
        ),
        max(
            float(np.max(np.abs(isotropic_final.E_x))),
            float(np.max(np.abs(anisotropic_final.E_x))),
        ),
        max(
            float(np.max(np.abs(isotropic_final.E_y))),
            float(np.max(np.abs(anisotropic_final.E_y))),
        ),
    )
    for index, (ax, (field, title, cmap)) in enumerate(zip(axes.flat, fields)):
        if cmap == "RdBu_r":
            limit = shared_limits[index % 4]
            assert limit is not None
            image = ax.imshow(
                field,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=cmap,
                vmin=-limit,
                vmax=limit,
            )
        else:
            image = ax.imshow(
                field,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap=cmap,
            )
        ax.set_title(title)
        ax.set_xlabel("y (normalized)")
        ax.set_ylabel("x (normalized)")
        fig.colorbar(image, ax=ax, pad=0.02)
    fig.suptitle("Isotropic and rotated-dielectric transverse PR references")
    figure_path = args.output_dir / "transverse_reference_fields.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    isotropic_diagnostics = isotropic.diagnostics
    anisotropic_diagnostics = anisotropic.diagnostics

    def field_metrics(field):
        return {
            "rms": float(np.sqrt(np.mean(field**2))),
            "max_abs": float(np.max(np.abs(field))),
        }

    def relative_difference(first, second):
        return float(np.linalg.norm(first - second) / np.linalg.norm(first))

    metrics = {
        "model": "transverse_potential_dielectric_comparison",
        "authoritative_state": "psi",
        "grid": {"Nx": Nx, "Ny": Ny, "dx": dx, "dy": dy},
        "solver": {
            "integrator": "explicit_euler",
            "dt_normalized": isotropic_options.dt_normalized,
            "steps": isotropic_options.steps,
            "time_normalized": isotropic.time_normalized,
        },
        "isotropic_dielectric": {
            "m_y": isotropic_options.m_y,
            "h_y": isotropic_options.h_y,
        },
        "rotated_barium_titanate_dielectric": {
            "c_axis_xz_angle_deg": dielectric.c_axis_xz_angle_deg,
            "epsilon_a": dielectric.epsilon_a,
            "epsilon_c": dielectric.epsilon_c,
            "epsilon_xx": dielectric.epsilon_xx,
            "epsilon_yy": dielectric.epsilon_yy,
            "h_y": dielectric.h_y,
            "m_y": anisotropic_options.m_y,
        },
        "applied_field": [isotropic_options.applied_field_x, 0.0],
        "active_field_projection": [1.0, 0.0],
        "isotropic": {
            "E_x": field_metrics(isotropic_final.E_x),
            "E_y": field_metrics(isotropic_final.E_y),
            "carrier_perturbation_rms": field_metrics(
                isotropic_final.carrier_density - 1.0
            )["rms"],
            "carrier_relative_drift": (
                isotropic_diagnostics.carrier_relative_drift
            ),
        },
        "anisotropic": {
            "E_x": field_metrics(anisotropic_final.E_x),
            "E_y": field_metrics(anisotropic_final.E_y),
            "carrier_perturbation_rms": field_metrics(
                anisotropic_final.carrier_density - 1.0
            )["rms"],
            "carrier_relative_drift": (
                anisotropic_diagnostics.carrier_relative_drift
            ),
        },
        "relative_differences": {
            "E_x": relative_difference(
                isotropic_final.E_x,
                anisotropic_final.E_x,
            ),
            "E_y": relative_difference(
                isotropic_final.E_y,
                anisotropic_final.E_y,
            ),
        },
        "outputs": {"figure": figure_path.name, "arrays": arrays_path.name},
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
