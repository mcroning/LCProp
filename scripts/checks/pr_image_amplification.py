"""Run the scaled PR image-amplification benchmark on a supplied image.

The source images used by PRProp3D are available from
https://github.com/mcroning/sample_images. Example::

    python scripts/checks/pr_image_amplification.py \
        --image "/path/to/sample_images/AF Res Chart.png" --dz-sweep
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np

from lcprop.pr.image_amplification import (
    PRImageAmplificationSpec,
    paper_figure4_spec,
    run_image_amplification,
)


def _read_grayscale(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - research dependency
        raise RuntimeError(
            "Pillow is required only by this image-loading check script"
        ) from error
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def _print_paper_scale() -> None:
    spec = paper_figure4_spec()
    Nz = int(round(spec.interaction_length_um / spec.dz_um))
    state_bytes = spec.Nx * spec.Ny * Nz * 8
    print("paper_figure4_spec=", spec)
    print(f"paper_Nz={Nz}")
    print(f"one_float64_E_state_GiB={state_bytes / 1024**3:.6g}")
    print(
        "paper_scale_status=parameter contract only; current unbatched "
        "workflow is not suitable for this allocation"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--invert",
        action="store_true",
        help="invert a dark-background target such as an MNIST digit",
    )
    parser.add_argument(
        "--dz-sweep",
        action="store_true",
        help="run the scaled case at dz=20, 10, and 5 um",
    )
    parser.add_argument(
        "--show-paper-scale",
        action="store_true",
        help="print the paper Figure 4 contract and memory estimate",
    )
    args = parser.parse_args()

    image_bytes = args.image.read_bytes()
    image = _read_grayscale(args.image)
    print(f"image={args.image}")
    print(f"image_sha256={hashlib.sha256(image_bytes).hexdigest()}")
    print(f"image_shape={image.shape}")
    if args.show_paper_scale:
        _print_paper_scale()

    base = PRImageAmplificationSpec(invert_image=bool(args.invert))
    dz_values = (20.0, 10.0, 5.0) if args.dz_sweep else (base.dz_um,)
    print(
        "dz_um,analytic_gain,measured_gain,correlation,nrmse,"
        "normalized_power_drift,runtime_s"
    )
    for dz_um in dz_values:
        result = run_image_amplification(
            image,
            replace(base, dz_um=dz_um),
        )
        print(
            f"{dz_um:g},{result.analytic_absolute_signal_gain:.12g},"
            f"{result.measured_absolute_signal_gain:.12g},"
            f"{result.image_intensity_correlation:.12g},"
            f"{result.normalized_image_rmse:.12g},"
            f"{result.normalized_power_relative_drift:.12g},"
            f"{result.runtime_s:.6g}"
        )


if __name__ == "__main__":
    main()
