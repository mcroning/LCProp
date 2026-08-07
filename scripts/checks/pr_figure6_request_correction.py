#!/usr/bin/env python3
"""Validate the published Figure 6 launch request before propagation.

This check applies a similarity-scaled version of the exact published request
to the checksummed Air Force chart.  It compares the carrier-isolated launch
intensity with the rendered Figure 6a data panel, records the complete image
preprocessing path, and retains the previous non-inverted request as a control.
It does not propagate either request or evaluate Figure 6b.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from scipy.spatial import cKDTree

from lcprop.core.grid import make_grid
from lcprop.pr.image_amplification import (
    isolate_signal_carrier,
    make_image_amplification_request,
    paper_figure6_spec,
    signal_carrier_mask,
)


AF_CHART_SHA256 = (
    "e424f1070587d79d2a91f4a3bd14e7c11e80adfe7c723e45432d9d4fc5bb85b7"
)
LOCAL_SIMILARITY_SCALE = 0.25
PUBLISHED_INPUT_LIMIT = 0.5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_air_force_chart(path: Path) -> np.ndarray:
    digest = _sha256(path)
    if digest != AF_CHART_SHA256:
        raise ValueError(
            f"Air Force chart SHA-256 mismatch: expected {AF_CHART_SHA256}, "
            f"received {digest}"
        )
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def _inverse_viridis_data_panel(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    rgb = rgb[95:734, 282:921] / 255.0
    table = matplotlib.colormaps["viridis"](
        np.linspace(0.0, 1.0, 4096)
    )[:, :3]
    _, indices = cKDTree(table).query(rgb.reshape(-1, 3))
    return indices.reshape(rgb.shape[:2]) / 4095.0


def _scaled_launch_spec():
    exact = paper_figure6_spec()
    scale = LOCAL_SIMILARITY_SCALE
    return replace(
        exact,
        Nx=int(round(exact.Nx * scale)),
        Ny=int(round(exact.Ny * scale)),
        x_aperture_um=exact.x_aperture_um * scale,
        y_aperture_um=exact.y_aperture_um * scale,
        interaction_length_um=exact.interaction_length_um * scale,
        dz_um=exact.dz_um * scale,
        wavelength_um=exact.wavelength_um * scale,
        positive_mode_index=int(round(exact.positive_mode_index * scale)),
        beam_waist_um=exact.beam_waist_um * scale,
    )


def _carrier_isolated_input(image: np.ndarray, spec):
    request, transmission, _grating, _gain = make_image_amplification_request(
        image, spec
    )
    grid = make_grid(request.grid, real_dtype=np.float64)
    pump_kx = request.beams.channels[0].tilt_x_rad_per_um
    signal_kx = request.beams.channels[1].tilt_x_rad_per_um
    mask = signal_carrier_mask(
        grid,
        pump_kx_rad_per_um=pump_kx,
        signal_kx_rad_per_um=signal_kx,
    )
    coherent = np.sum(np.asarray(request.initial_A), axis=0)
    signal = isolate_signal_carrier(coherent, mask)
    return request, transmission, np.abs(signal) ** 2


def _presentation_raster(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    oriented = np.fliplr(values.T)
    return zoom(
        oriented,
        (shape[0] / oriented.shape[0], shape[1] / oriented.shape[1]),
        order=1,
    )


def _robust_unit(values: np.ndarray) -> np.ndarray:
    low, high = np.percentile(values, (0.5, 99.5))
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    correlation = float(
        np.corrcoef(reference.ravel(), candidate.ravel())[0, 1]
    )
    rmse = float(np.sqrt(np.mean((reference - candidate) ** 2)))
    ref_y, ref_x = np.gradient(reference)
    got_y, got_x = np.gradient(candidate)
    edge_correlation = float(
        np.corrcoef(
            np.hypot(ref_x, ref_y).ravel(),
            np.hypot(got_x, got_y).ravel(),
        )[0, 1]
    )
    return {
        "pearson_correlation": correlation,
        "normalized_rmse": rmse,
        "edge_correlation": edge_correlation,
    }


def _best_horizontal_registration(
    reference: np.ndarray, candidate: np.ndarray
) -> tuple[np.ndarray, int, dict[str, float]]:
    best = None
    for pixels in range(-24, 25):
        shifted = np.roll(candidate, pixels, axis=1)
        measured = _metrics(reference, shifted)
        score = measured["pearson_correlation"]
        if best is None or score > best[0]:
            best = score, pixels, shifted, measured
    assert best is not None
    return best[2], best[1], best[3]


def _save_panel(
    values: np.ndarray,
    path: Path,
    *,
    title: str,
    vmax: float,
) -> None:
    fig, axis = plt.subplots(figsize=(5.2, 5.7), constrained_layout=True)
    image = axis.imshow(
        values,
        origin="upper",
        extent=(-2000.0, 2000.0, -2000.0, 2000.0),
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
    )
    axis.set_title(title)
    axis.set_xlabel(r"x ($\mu$m)")
    axis.set_ylabel(r"y ($\mu$m)")
    axis.set_aspect("equal")
    fig.colorbar(
        image,
        ax=axis,
        orientation="horizontal",
        label="normalized intensity",
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--published-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    image = _load_air_force_chart(args.image)
    published = _inverse_viridis_data_panel(args.published_input)
    published_unit = _robust_unit(published)

    corrected_spec = _scaled_launch_spec()
    previous_spec = replace(corrected_spec, invert_image=False)
    corrected_request, corrected_transmission, corrected = _carrier_isolated_input(
        image, corrected_spec
    )
    previous_request, previous_transmission, previous = _carrier_isolated_input(
        image, previous_spec
    )

    corrected = _robust_unit(_presentation_raster(corrected, published.shape))
    previous = _robust_unit(_presentation_raster(previous, published.shape))
    corrected, corrected_shift, corrected_metrics = _best_horizontal_registration(
        published_unit, corrected
    )
    previous, previous_shift, previous_metrics = _best_horizontal_registration(
        published_unit, previous
    )

    if corrected_metrics["pearson_correlation"] <= 0.75:
        raise AssertionError("corrected Figure 6 input correlation is too low")
    if previous_metrics["pearson_correlation"] >= 0.0:
        raise AssertionError(
            "non-inverted control does not preserve opposite polarity"
        )
    if corrected_request.grid != previous_request.grid:
        raise AssertionError("polarity control changed the grid")
    if corrected_request.material != previous_request.material:
        raise AssertionError("polarity control changed the PR material")
    if corrected_request.solver != previous_request.solver:
        raise AssertionError("polarity control changed solver controls")

    corrected_absolute_display = corrected * PUBLISHED_INPUT_LIMIT
    _save_panel(
        corrected_absolute_display,
        args.output / "corrected_input_panel.png",
        title="LCProp corrected Figure 6 input",
        vmax=PUBLISHED_INPUT_LIMIT,
    )

    residual = np.abs(published_unit - corrected)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    for axis, values, title, cmap in (
        (axes[0], published_unit, "Published Figure 6a", "viridis"),
        (axes[1], corrected, "Corrected LCProp request", "viridis"),
        (axes[2], residual, "Absolute normalized residual", "magma"),
    ):
        panel = axis.imshow(
            values,
            origin="upper",
            extent=(-2000.0, 2000.0, -2000.0, 2000.0),
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(title)
        axis.set_xlabel(r"x ($\mu$m)")
        axis.set_ylabel(r"y ($\mu$m)")
        axis.set_aspect("equal")
        fig.colorbar(panel, ax=axis, orientation="horizontal", fraction=0.05)
    fig.savefig(args.output / "corrected_input_comparison.png", dpi=180)
    plt.close(fig)

    exact_spec = paper_figure6_spec()
    record = {
        "schema": "lcprop.pr_figure6_request_correction.v1",
        "classification": "local launch-request validation",
        "output_panel_evaluated": False,
        "output_panel_reason": (
            "No corrected full-scale propagation was authorized; Figure 6b "
            "evaluation is gated on a successful corrected full-scale run."
        ),
        "sources": {
            "air_force_chart": {
                "path": str(args.image),
                "sha256": _sha256(args.image),
            },
            "published_figure6a": {
                "path": str(args.published_input),
                "sha256": _sha256(args.published_input),
                "data_crop_pixels_yx": [[95, 734], [282, 921]],
            },
        },
        "exact_benchmark_spec": asdict(exact_spec),
        "static_unused_compatibility_fields": ["Nt", "dt_normalized"],
        "local_similarity_validation": {
            "scale": LOCAL_SIMILARITY_SCALE,
            "spec": asdict(corrected_spec),
            "preserved_dimensionless_quantities": [
                "external incidence angle",
                "samples per two-beam grating period",
                "beam-waist to aperture ratio",
                "interaction-length to aperture ratio",
                "image-size to beam-waist ratio",
            ],
        },
        "preprocessing": [
            "Pillow grayscale conversion",
            "normalize source intensity to [0, 1]",
            "invert source intensity",
            "pad non-square source to even square with transparent value 1",
            "rotate source into LCProp (x, y) array convention",
            "nearest-neighbor resize without prefiltering",
            "place at the image-bearing signal launch center",
            "multiply signal amplitude by sqrt(transmission)",
            "renormalize signal peak ratio and total launch power",
            "isolate signal with the unchanged nearest-carrier Fourier mask",
            "transpose and reflect persisted raster into published orientation",
        ],
        "controls": {
            "only_spec_difference": "invert_image: false -> true",
            "grid_equal": corrected_request.grid == previous_request.grid,
            "material_equal": corrected_request.material == previous_request.material,
            "solver_equal": corrected_request.solver == previous_request.solver,
            "corrected_transmission_range": [
                float(np.min(corrected_transmission)),
                float(np.max(corrected_transmission)),
            ],
            "previous_transmission_range": [
                float(np.min(previous_transmission)),
                float(np.max(previous_transmission)),
            ],
        },
        "published_input_presentation": {
            "observable": "carrier-isolated signal intensity",
            "field_of_view_um": [-2000.0, 2000.0, -2000.0, 2000.0],
            "colormap": "viridis",
            "scale": "linear",
            "limits": [0.0, PUBLISHED_INPUT_LIMIT],
        },
        "metrics": {
            "corrected": corrected_metrics,
            "corrected_horizontal_registration_pixels": corrected_shift,
            "non_inverted_control": previous_metrics,
            "non_inverted_horizontal_registration_pixels": previous_shift,
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
