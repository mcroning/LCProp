#!/usr/bin/env python3
"""Compare production adaptive substeps with validation-04 frozen-theta replays."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lcprop-matplotlib-cache"),
)

import matplotlib.pyplot as plt

import validation_01_longitudinal_step_far_field as validation
from lcprop.core.backend import asnumpy
from lcprop.core.grid import make_grid
from lcprop.lc import run_static
from lcprop.optics.splitstep import total_intensity
from lcprop.products.diagnostics import rms_widths


LOG_DYNAMIC_RANGE_DECADES = 12.0


def radial_profile(intensity, kx, ky, spacing):
    radius = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    edges = np.arange(
        0.0,
        float(np.max(radius)) + 2.0 * spacing,
        spacing,
    )
    weighted, _ = np.histogram(
        radius.ravel(),
        bins=edges,
        weights=intensity.ravel(),
    )
    counts, _ = np.histogram(radius.ravel(), bins=edges)
    mean = np.divide(
        weighted,
        counts,
        out=np.zeros_like(weighted),
        where=counts > 0,
    )
    return 0.5 * (edges[:-1] + edges[1:]), mean


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    earlier_path = (
        repo_root
        / "outputs"
        / "validation_04_optical_substepping"
        / "validation_04_optical_substepping_data.npz"
    )
    if not earlier_path.exists():
        raise FileNotFoundError(
            "run validation_04_optical_substepping.py before this comparison"
        )
    output_dir = (
        repo_root
        / "outputs"
        / "validation_05_production_substeps_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    far_png = output_dir / "validation_05_far_field_comparison.png"
    radial_png = output_dir / "validation_05_radial_comparison.png"
    npz_path = output_dir / "validation_05_comparison_data.npz"

    earlier = np.load(earlier_path)
    earlier_nsub = np.asarray(earlier["Nsub"], dtype=np.int64)
    earlier_far = np.asarray(earlier["far_field_intensity_normalized"])
    earlier_near = np.asarray(earlier["near_field_intensity"])
    earlier_theta = np.asarray(earlier["frozen_theta_stack"])
    earlier_kx = np.asarray(earlier["kx_rad_per_um"])
    earlier_ky = np.asarray(earlier["ky_rad_per_um"])

    request = validation.make_request(
        20.0,
        "self_consistent",
        "zero_voltage_pi4_boundary",
    )
    grid = make_grid(request.grid, xp=np, real_dtype=np.float64)
    started = perf_counter()
    result = run_static(request)
    elapsed_s = perf_counter() - started
    diagnostics = dict(result.provenance)
    print(json.dumps(diagnostics, indent=2, sort_keys=True))

    final_field = np.asarray(asnumpy(result.A_final[0])).copy()
    near = np.asarray(
        asnumpy(
            total_intensity(
                result.A_final,
                coherence_groups=request.beams.coherence_groups,
                xp=grid.xp,
            )
        )
    )
    kx, ky, far, dkx, dky = validation.centered_far_field(
        final_field,
        dx_um=grid.dx_um,
        dy_um=grid.dy_um,
        xp=np,
    )
    kx = np.asarray(kx)
    ky = np.asarray(ky)
    far = np.asarray(far)
    np.testing.assert_array_equal(kx, earlier_kx)
    np.testing.assert_array_equal(ky, earlier_ky)
    near_rms_x_um, _ = rms_widths(near, grid)
    far_rms_kx, outer_fraction = validation.far_field_metrics(
        far,
        kx,
        dkx=dkx,
        dky=dky,
        xp=np,
    )

    far_relative_l2 = np.asarray(
        [
            np.linalg.norm(far - reference)
            / np.linalg.norm(reference)
            for reference in earlier_far
        ]
    )
    near_relative_l2 = np.asarray(
        [
            np.linalg.norm(near - reference)
            / np.linalg.norm(reference)
            for reference in earlier_near
        ]
    )
    production_theta = np.asarray(asnumpy(result.theta_final))
    theta_difference = production_theta - earlier_theta
    theta_difference_rms = float(
        np.sqrt(np.mean(theta_difference * theta_difference))
    )
    theta_difference_max = float(np.max(np.abs(theta_difference)))

    all_far = [*earlier_far, far]
    global_peak = max(float(np.max(item)) for item in all_far)
    vmax = math.log10(global_peak)
    vmin = vmax - LOG_DYNAMIC_RANGE_DECADES
    floor = 10.0**vmin
    figure, axes = plt.subplots(
        1,
        5,
        figsize=(18.0, 4.5),
        constrained_layout=True,
    )
    image = None
    labels = [
        *(f"frozen θ\nNsub={value}" for value in earlier_nsub),
        (
            "production SC\n"
            f"Nsub={diagnostics['optical_Nsub']}"
        ),
    ]
    for axis, intensity, label in zip(
        axes, all_far, labels, strict=True
    ):
        image = axis.imshow(
            np.log10(np.maximum(intensity, floor)).T,
            origin="lower",
            extent=(kx[0], kx[-1], ky[0], ky[-1]),
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
            interpolation="nearest",
            aspect="equal",
        )
        axis.set(
            title=label,
            xlabel="kx (rad/µm)",
            ylabel="ky (rad/µm)",
        )
    assert image is not None
    colorbar = figure.colorbar(image, ax=axes, shrink=0.82, pad=0.02)
    colorbar.set_label("log10 normalized spectral power density")
    figure.suptitle(
        "Production adaptive self-consistency vs frozen-theta controls"
    )
    figure.savefig(far_png, dpi=190, bbox_inches="tight")
    plt.close(figure)

    spacing = min(dkx, dky)
    figure, axis = plt.subplots(
        figsize=(9.2, 6.0),
        constrained_layout=True,
    )
    for intensity, label in zip(all_far, labels, strict=True):
        radial_k, radial_mean = radial_profile(
            intensity, kx, ky, spacing
        )
        positive = radial_mean[radial_mean > 0.0]
        display = np.maximum(
            radial_mean,
            float(np.max(positive)) * 1e-14,
        )
        axis.semilogy(radial_k, display, label=label.replace("\n", " "))
    axis.set(
        xlabel="radial k (rad/µm)",
        ylabel="Azimuthal mean spectral density",
        title="Far-field radial profiles",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.savefig(radial_png, dpi=190, bbox_inches="tight")
    plt.close(figure)

    np.savez_compressed(
        npz_path,
        production_diagnostics_json=np.asarray(
            json.dumps(diagnostics, sort_keys=True)
        ),
        production_elapsed_s=np.asarray(elapsed_s),
        production_final_power_mW=np.asarray(
            result.physical_power_final_mW
        ),
        production_near_rms_x_um=np.asarray(near_rms_x_um),
        production_far_rms_kx_rad_per_um=np.asarray(far_rms_kx),
        production_outer_kx_power_fraction=np.asarray(outer_fraction),
        production_unconverged_slices=np.asarray(
            sum(not item.converged for item in result.slice_summaries)
        ),
        earlier_Nsub=earlier_nsub,
        far_relative_l2_from_earlier=far_relative_l2,
        near_relative_l2_from_earlier=near_relative_l2,
        theta_difference_rms_rad=np.asarray(theta_difference_rms),
        theta_difference_max_rad=np.asarray(theta_difference_max),
        kx_rad_per_um=kx,
        ky_rad_per_um=ky,
        production_final_field=final_field,
        production_near_intensity=near,
        production_far_intensity_normalized=far,
        production_theta_stack=production_theta,
    )

    print()
    print(
        "earlier_Nsub  far_relative_L2  near_relative_L2"
    )
    for nsub, far_l2, near_l2 in zip(
        earlier_nsub,
        far_relative_l2,
        near_relative_l2,
        strict=True,
    ):
        print(f"{nsub:12d}  {far_l2:15.9f}  {near_l2:16.9f}")
    print(f"production_elapsed_s = {elapsed_s:.6f}")
    print(
        "production_final_power_mW = "
        f"{float(result.physical_power_final_mW):.12f}"
    )
    print(f"production_near_rms_x_um = {near_rms_x_um:.12f}")
    print(
        "production_far_rms_kx_rad_per_um = "
        f"{far_rms_kx:.12f}"
    )
    print(
        "production_outer_kx_power_fraction = "
        f"{outer_fraction:.12e}"
    )
    print(
        "production_unconverged_slices = "
        f"{sum(not item.converged for item in result.slice_summaries)}"
    )
    print(f"theta_difference_rms_rad = {theta_difference_rms:.12e}")
    print(f"theta_difference_max_rad = {theta_difference_max:.12e}")
    print(f"far_field_figure = {far_png}")
    print(f"radial_figure = {radial_png}")
    print(f"npz_path = {npz_path}")


if __name__ == "__main__":
    main()
