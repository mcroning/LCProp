#!/usr/bin/env python3
"""Sweep PRProp noise strength at dz=20 um for the V=0, theta_bc=pi/4 case."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import tempfile

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lcprop-matplotlib-cache"),
)

import matplotlib.pyplot as plt

import validation_01_longitudinal_step_far_field as validation
from lcprop.core.backend import asnumpy
from lcprop.core.grid import make_grid


DZ_UM = 20.0
EPS_VALUES = (0.0, 0.001, 0.002, 0.005, 0.01)
SIGMA_UM = 0.4
BIAS_CASE = "zero_voltage_pi4_boundary"
WORKFLOW = "self_consistent"
LOG_DYNAMIC_RANGE_DECADES = 12.0


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise-seed", type=int, default=54321)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repo_root
            / "outputs"
            / "validation_02_noise_eps_sweep_dz20_V_0_theta_bc_pi4_seed_54321"
        ),
    )
    return parser.parse_args()


def save_figure(
    path: Path,
    runs,
    x_um: np.ndarray,
    y_um: np.ndarray,
    kx: np.ndarray,
    ky: np.ndarray,
) -> None:
    global_peak = max(
        float(np.max(run.far_intensity_normalized)) for run in runs
    )
    log_vmax = math.log10(global_peak)
    log_vmin = log_vmax - LOG_DYNAMIC_RANGE_DECADES
    log_floor = 10.0**log_vmin

    figure = plt.figure(figsize=(18.0, 8.3), constrained_layout=True)
    layout = figure.add_gridspec(2, 10, height_ratios=(0.82, 1.0))
    near_axis = figure.add_subplot(layout[0, 0:5])
    far_axis = figure.add_subplot(layout[0, 5:10])
    image_axes = [
        figure.add_subplot(layout[1, 2 * index : 2 * index + 2])
        for index in range(len(runs))
    ]

    central_y = int(np.argmin(np.abs(y_um)))
    central_ky = int(np.argmin(np.abs(ky)))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(runs)))
    for run, color in zip(runs, colors, strict=True):
        label = f"eps={run.noise_eps:g}"
        near_axis.plot(
            x_um,
            run.near_intensity[:, central_y],
            color=color,
            label=label,
        )
        far_axis.semilogy(
            kx,
            np.maximum(
                run.far_intensity_normalized[:, central_ky],
                log_floor,
            ),
            color=color,
            label=label,
        )

    near_axis.set(
        title="Final near-field central x cut",
        xlabel="x (µm)",
        ylabel="Intensity (µm⁻²)",
    )
    far_axis.set(
        title="Final normalized far-field central kx cut",
        xlabel="kx (rad/µm)",
        ylabel="Spectral power density",
    )
    near_axis.grid(alpha=0.25)
    far_axis.grid(alpha=0.25)
    near_axis.legend()
    far_axis.legend()

    image = None
    for run, axis in zip(runs, image_axes, strict=True):
        log_far = np.log10(
            np.maximum(run.far_intensity_normalized, log_floor)
        )
        image = axis.imshow(
            log_far.T,
            origin="lower",
            extent=(kx[0], kx[-1], ky[0], ky[-1]),
            vmin=log_vmin,
            vmax=log_vmax,
            cmap="magma",
            aspect="equal",
            interpolation="nearest",
        )
        axis.set(
            title=f"eps={run.noise_eps:g}",
            xlabel="kx (rad/µm)",
            ylabel="ky (rad/µm)",
        )

    assert image is not None
    colorbar = figure.colorbar(image, ax=image_axes, shrink=0.92, pad=0.01)
    colorbar.set_label("log10 normalized spectral power density")
    figure.suptitle(
        "LCProp static self-consistent noise-strength sweep\n"
        f"dz=20 µm, V=0, theta_bc=π/4, sigma=0.4 µm, "
        f"seed={runs[0].noise_seed}"
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = (
        "validation_02_noise_eps_sweep_dz20_"
        f"V_0_theta_bc_pi4_seed_{args.noise_seed}"
    )
    npz_path = output_dir / f"{stem}.npz"
    png_path = output_dir / f"{stem}.png"

    runs = []
    kx = None
    ky = None
    for eps in EPS_VALUES:
        print(f"running eps={eps:g}, dz={DZ_UM:g} um ...", flush=True)
        run, case_kx, case_ky = validation.execute_case(
            DZ_UM,
            WORKFLOW,
            noise_mode="independent",
            noise_eps=eps,
            noise_sigma_um=SIGMA_UM,
            noise_seed=args.noise_seed,
            bias_case=BIAS_CASE,
        )
        runs.append(run)
        if kx is None:
            kx = case_kx
            ky = case_ky
        else:
            np.testing.assert_array_equal(kx, case_kx)
            np.testing.assert_array_equal(ky, case_ky)
        print(f"completed eps={eps:g} in {run.elapsed_s:.3f} s")

    assert kx is not None
    assert ky is not None
    reference = runs[0].far_intensity_normalized
    reference_norm = np.linalg.norm(reference)
    far_l2_from_eps0 = np.asarray(
        [
            np.linalg.norm(run.far_intensity_normalized - reference)
            / reference_norm
            for run in runs
        ]
    )

    request = validation.make_request(DZ_UM, WORKFLOW, BIAS_CASE)
    grid = make_grid(request.grid, real_dtype=np.float64)
    x_um = np.asarray(asnumpy(grid.x_um))
    y_um = np.asarray(asnumpy(grid.y_um))
    dxdy = float(grid.dx_um * grid.dy_um)
    dkx = float(abs(kx[1] - kx[0]))
    dky = float(abs(ky[1] - ky[0]))
    near_integrals = np.asarray(
        [np.sum(run.near_intensity) * dxdy for run in runs]
    )
    far_integrals = np.asarray(
        [
            np.sum(run.far_intensity_normalized) * dkx * dky
            for run in runs
        ]
    )
    np.testing.assert_allclose(near_integrals, 1.0, rtol=0, atol=5e-13)
    np.testing.assert_allclose(far_integrals, 1.0, rtol=0, atol=5e-13)

    central_ky = int(np.argmin(np.abs(ky)))
    np.savez_compressed(
        npz_path,
        noise_eps=np.asarray(EPS_VALUES),
        noise_sigma_um=np.asarray(SIGMA_UM),
        noise_seed=np.asarray(args.noise_seed, dtype=np.int64),
        dz_um=np.asarray(DZ_UM),
        n_steps=np.asarray([run.n_steps for run in runs], dtype=np.int64),
        elapsed_s=np.asarray([run.elapsed_s for run in runs]),
        final_power_mW=np.asarray([run.final_power_mW for run in runs]),
        near_rms_x_um=np.asarray([run.near_rms_x_um for run in runs]),
        far_rms_kx_rad_per_um=np.asarray(
            [run.far_rms_kx_rad_per_um for run in runs]
        ),
        outer_kx_power_fraction=np.asarray(
            [run.outer_kx_power_fraction for run in runs]
        ),
        far_l2_from_eps0=far_l2_from_eps0,
        representative_phase_mean_rad=np.asarray(
            [run.representative_phase_mean_rad for run in runs]
        ),
        representative_phase_rms_rad=np.asarray(
            [run.representative_phase_rms_rad for run in runs]
        ),
        unconverged_slices=np.asarray(
            [run.unconverged_slices for run in runs],
            dtype=np.int64,
        ),
        max_final_residual_rms=np.asarray(
            [run.max_final_residual_rms for run in runs]
        ),
        max_final_residual_max=np.asarray(
            [run.max_final_residual_max for run in runs]
        ),
        near_integral=near_integrals,
        far_integral=far_integrals,
        x_um=x_um,
        y_um=y_um,
        kx_rad_per_um=kx,
        ky_rad_per_um=ky,
        final_field=np.stack([run.final_field for run in runs]),
        near_field_intensity=np.stack(
            [run.near_intensity for run in runs]
        ),
        far_field_intensity_normalized=np.stack(
            [run.far_intensity_normalized for run in runs]
        ),
        central_far_field_cut=np.stack(
            [
                run.far_intensity_normalized[:, central_ky]
                for run in runs
            ]
        ),
        representative_phase_screen=np.stack(
            [run.representative_phase for run in runs]
        ),
        representative_phase_spectrum=np.stack(
            [run.representative_phase_spectrum for run in runs]
        ),
    )
    save_figure(png_path, runs, x_um, y_um, kx, ky)

    print()
    print(
        "eps      phase_mean_rad  phase_rms_rad  final_power_mW  "
        "near_rms_x_um  far_rms_kx  outer_fraction  far_L2_from_eps0  "
        "unconverged"
    )
    for run, l2 in zip(runs, far_l2_from_eps0, strict=True):
        print(
            f"{run.noise_eps:7.4f}  "
            f"{run.representative_phase_mean_rad:14.7e}  "
            f"{run.representative_phase_rms_rad:13.7e}  "
            f"{run.final_power_mW:14.9f}  "
            f"{run.near_rms_x_um:13.7f}  "
            f"{run.far_rms_kx_rad_per_um:10.7f}  "
            f"{run.outer_kx_power_fraction:14.7e}  "
            f"{l2:16.9e}  "
            f"{run.unconverged_slices}/{run.n_steps}"
        )
    print(f"npz_path = {npz_path}")
    print(f"png_path = {png_path}")


if __name__ == "__main__":
    main()
