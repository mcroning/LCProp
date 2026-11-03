#!/usr/bin/env python3
"""Run a small deterministic Figure 6 streaming/reference comparison.

This is a local numerical-path check, not the paper-scale calculation. It
uses the Air Force chart and executes the three explicitly named static paths
on a bounded grid. The exact contract remains in ``paper_figure6_spec()``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from lcprop.pr.image_amplification import (
    paper_figure6_spec,
    run_streaming_image_amplification,
)
from lcprop.pr.static_streaming import (
    PRStreamingStaticOptions,
    PR_STREAMING_FULL_NONLINEAR_REFERENCE,
    PR_STREAMING_LEGACY_LINEARIZED_REFERENCE,
    PR_STREAMING_PRODUCTION,
)
from lcprop.pr.static_workflow import PRStaticWorkflowOptions


AF_CHART_SHA256 = "e424f1070587d79d2a91f4a3bd14e7c11e80adfe7c723e45432d9d4fc5bb85b7"


def _load_image(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != AF_CHART_SHA256:
        raise ValueError(
            f"Air Force chart SHA-256 mismatch: expected {AF_CHART_SHA256}, "
            f"got {digest}"
        )
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def _scaled_spec():
    return replace(
        paper_figure6_spec(),
        Nx=64,
        Ny=32,
        x_aperture_um=96.0,
        y_aperture_um=48.0,
        interaction_length_um=40.0,
        dz_um=10.0,
        positive_mode_index=2,
        beam_waist_um=19.2,
        saturated_small_signal_gain=None,
        gain_length_product_override=-0.2,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    image = _load_image(args.image)
    spec = _scaled_spec()
    modes = (
        PR_STREAMING_LEGACY_LINEARIZED_REFERENCE,
        PR_STREAMING_FULL_NONLINEAR_REFERENCE,
        PR_STREAMING_PRODUCTION,
    )
    results = {}
    fields = {}
    for mode in modes:
        result = run_streaming_image_amplification(
            image,
            spec,
            solver=PRStreamingStaticOptions(
                coupled=PRStaticWorkflowOptions(max_coupled_passes=20),
                mode=mode,
                tukey_alpha=spec.tukey_alpha,
                volume_noise_epsilon=spec.volume_noise_epsilon,
                volume_noise_correlation_um=spec.volume_noise_correlation_um,
                volume_noise_seed=spec.volume_noise_seed,
                volume_noise_seeds=spec.volume_noise_seeds,
            ),
        )
        fields[mode] = result.run_result.A_final
        results[mode] = {
            "status": result.run_result.status,
            "gain": result.measured_absolute_signal_gain,
            "analytic_gain": result.analytic_absolute_signal_gain,
            "correlation": result.image_intensity_correlation,
            "normalized_rmse": result.normalized_image_rmse,
            "power_drift": result.normalized_power_relative_drift,
            "governing_residual_final_moments": (
                result.run_result.first_pass.residual_moments[-1].tolist()
            ),
            "full_nonlinear_residual_final_moments": (
                result.run_result.first_pass.full_nonlinear_residual_moments[
                    -1
                ].tolist()
            ),
            "deterministic_replay": result.run_result.replay_diagnostics,
        }
    comparisons = {}
    for left, right in zip(modes[:-1], modes[1:]):
        difference = fields[left] - fields[right]
        comparisons[f"{left}__vs__{right}"] = {
            "relative_l2": float(
                np.linalg.norm(difference) / np.linalg.norm(fields[right])
            ),
            "max_abs": float(np.max(np.abs(difference))),
        }
    payload = {
        "classification": "scaled local numerical-path comparison",
        "image_sha256": AF_CHART_SHA256,
        "figure6_exact_contract": {
            "shape": [16384, 2048, 2175],
            "saturated_small_signal_gain": 4000.0,
            "input_peak_ratio": 1.0,
            "invert_image": True,
            "volume_noise": "disabled",
        },
        "scaled_results": results,
        "mode_comparisons": comparisons,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
