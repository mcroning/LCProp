"""Bounded, explicitly non-authoritative PR visualization products."""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
from typing import Any, Callable, Mapping, Sequence

import numpy as np


FAST_MPR_TARGET_BYTES = 4 * 1024 * 1024
FAST_MPR_MAX_X = 96
FAST_MPR_MAX_Y = 96
FAST_MPR_MAX_Z = FAST_MPR_TARGET_BYTES // (
    FAST_MPR_MAX_X * FAST_MPR_MAX_Y * np.dtype(np.float32).itemsize
)
TD_MOVIE_MAX_FRAMES = 36
TD_MOVIE_MAX_X = 128
TD_MOVIE_MAX_Y = 128


@dataclass(frozen=True)
class PRIntensityPreview:
    intensity: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PRTDPreviewMovie:
    data: np.ndarray | None
    metadata: dict[str, Any]
    warning: str | None = None


def _bin_edges(size: int, maximum: int) -> np.ndarray:
    count = min(int(size), int(maximum))
    if count < 1:
        raise ValueError("preview dimensions must be positive")
    return np.linspace(0, size, count + 1, dtype=np.int64)


def block_average_2d(
    values: Any, *, max_x: int, max_y: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average deterministic contiguous cell blocks without decimation."""

    source = np.asarray(values)
    if source.ndim != 2 or source.dtype.hasobject:
        raise ValueError("block averaging requires a numeric 2-D array")
    x_edges = _bin_edges(source.shape[0], max_x)
    y_edges = _bin_edges(source.shape[1], max_y)
    source64 = np.asarray(source, dtype=np.float64)
    summed_x = np.add.reduceat(source64, x_edges[:-1], axis=0)
    summed_xy = np.add.reduceat(summed_x, y_edges[:-1], axis=1)
    counts = np.diff(x_edges)[:, None] * np.diff(y_edges)[None, :]
    reduced = summed_xy / counts
    return reduced, x_edges, y_edges


def _block_centers(coordinates: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.mean(coordinates[start:stop], dtype=np.float64)
        for start, stop in zip(edges[:-1], edges[1:])
    ])


def make_fast_intensity_preview(
    source_intensity_stack: Any,
    *,
    grid_summary: Mapping[str, Any],
    peak_intensity_reference: float,
    background_intensity: float,
    asnumpy: Callable[[Any], Any] = np.asarray,
    z_offset_fraction: float = 0.0,
) -> PRIntensityPreview:
    """Create the bounded final-state MPR preview from actual computed planes."""

    requested_nz, nx, ny = (
        int(grid_summary[name]) for name in ("Nz", "Nx", "Ny")
    )
    shape = tuple(getattr(source_intensity_stack, "shape", ()))
    if len(shape) != 3 or shape[1:] != (nx, ny) or not 0 < shape[0] <= requested_nz:
        raise ValueError("source intensity must match the completed grid volume")
    nz = int(shape[0])
    reference = float(peak_intensity_reference)
    background = float(background_intensity)
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("peak intensity reference must be finite and positive")
    if not np.isfinite(background) or background < 0.0:
        raise ValueError("background intensity must be finite and nonnegative")

    z_indices = np.unique(
        np.rint(np.linspace(0, nz - 1, min(nz, FAST_MPR_MAX_Z))).astype(int)
    )
    planes: list[np.ndarray] = []
    x_edges = y_edges = None
    for iz in z_indices:
        physical = (
            np.asarray(asnumpy(source_intensity_stack[int(iz)]), dtype=np.float64)
            - background
        ) * reference
        plane, x_edges, y_edges = block_average_2d(
            physical, max_x=FAST_MPR_MAX_X, max_y=FAST_MPR_MAX_Y
        )
        planes.append(plane.astype(np.float32))
    assert x_edges is not None and y_edges is not None
    intensity = np.stack(planes)
    dx = float(grid_summary["dx_um"])
    dy = float(grid_summary["dy_um"])
    dz = float(grid_summary["dz_um"])
    x = (np.arange(nx) - 0.5 * (nx - 1)) * dx
    y = (np.arange(ny) - 0.5 * (ny - 1)) * dy
    x_preview = _block_centers(x, x_edges)
    y_preview = _block_centers(y, y_edges)
    z_preview = (z_indices + float(z_offset_fraction)) * dz
    metadata = {
        "visualization_only": True,
        "quantity": "physical_optical_intensity",
        "value_unit": "1/um^2",
        "original_grid": {"Nz": nz, "Nx": nx, "Ny": ny},
        "requested_grid": {"Nz": requested_nz, "Nx": nx, "Ny": ny},
        "preview_grid": {
            "Nz": int(intensity.shape[0]),
            "Nx": int(intensity.shape[1]),
            "Ny": int(intensity.shape[2]),
        },
        "original_spacing_um": {"dz": dz, "dx": dx, "dy": dy},
        "preview_coordinates_um": {
            "z": z_preview.tolist(),
            "x": x_preview.tolist(),
            "y": y_preview.tolist(),
        },
        "retained_z_indices": z_indices.tolist(),
        "z_reduction": "uniformly_spaced_actual_planes_no_averaging",
        "transverse_reduction": "contiguous_block_arithmetic_mean",
        "x_block_bounds": x_edges.tolist(),
        "y_block_bounds": y_edges.tolist(),
        "normalization": (
            "(dimensionless_PR_driving_intensity - material_background) * "
            "launch_peak_intensity_reference"
        ),
        "dtype": str(intensity.dtype),
        "payload_bytes": int(intensity.nbytes),
        "target_payload_bytes": FAST_MPR_TARGET_BYTES,
    }
    if intensity.nbytes > FAST_MPR_TARGET_BYTES:
        raise AssertionError("Fast MPR preview exceeded its byte budget")
    return PRIntensityPreview(intensity=intensity, metadata=metadata)


def validate_fast_intensity_preview(
    intensity: Any | None, metadata: Any | None
) -> bool:
    """Validate optional preview data; return false for legacy absence."""

    if intensity is None and metadata is None:
        return False
    if not isinstance(intensity, np.ndarray) or intensity.dtype != np.float32:
        raise ValueError("Fast MPR preview must be float32")
    if intensity.ndim != 3 or not isinstance(metadata, Mapping):
        raise ValueError("Fast MPR preview requires 3-D data and metadata")
    if metadata.get("visualization_only") is not True:
        raise ValueError("Fast MPR preview must be visualization-only")
    shape = metadata.get("preview_grid", {})
    expected = tuple(int(shape[name]) for name in ("Nz", "Nx", "Ny"))
    if intensity.shape != expected:
        raise ValueError("Fast MPR preview shape disagrees with metadata")
    coordinates = metadata.get("preview_coordinates_um", {})
    for name, size in zip(("z", "x", "y"), intensity.shape):
        values = np.asarray(coordinates.get(name), dtype=np.float64)
        if values.shape != (size,) or not np.all(np.isfinite(values)):
            raise ValueError("Fast MPR preview coordinates are invalid")
    if intensity.nbytes > int(metadata.get("target_payload_bytes", -1)):
        raise ValueError("Fast MPR preview exceeds recorded byte budget")
    return True


def td_movie_frame_indices(total_steps: int) -> np.ndarray:
    """Choose bounded accepted material-time frames including both endpoints."""

    count = int(total_steps) + 1
    if count < 1:
        raise ValueError("total_steps must be nonnegative")
    return np.unique(
        np.rint(np.linspace(0, total_steps, min(count, TD_MOVIE_MAX_FRAMES))).astype(int)
    )


def downsample_td_movie_frame(intensity_xy: Any) -> np.ndarray:
    reduced, _, _ = block_average_2d(
        intensity_xy, max_x=TD_MOVIE_MAX_X, max_y=TD_MOVIE_MAX_Y
    )
    return reduced.astype(np.float32)


def encode_td_preview_movie(
    frames: Sequence[np.ndarray],
    *,
    frame_indices: Sequence[int],
    material_times: Sequence[float],
    original_grid: Mapping[str, Any],
    original_cadence: float,
    ffmpeg_path: str | None = None,
) -> PRTDPreviewMovie:
    """Encode a bounded fixed-scale MP4; failure is explicitly non-scientific."""

    if not frames:
        return PRTDPreviewMovie(
            None,
            {"visualization_only": True, "status": "not_generated_no_frames"},
        )
    stack = np.asarray(frames, dtype=np.float32)
    if stack.ndim != 3:
        raise ValueError("TD preview frames must have shape (frame, x, y)")
    if len(frame_indices) != len(stack) or len(material_times) != len(stack):
        raise ValueError("TD preview frame provenance lengths disagree")
    finite = stack[np.isfinite(stack)]
    vmin = 0.0
    vmax = float(np.max(finite)) if finite.size else 1.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = 1.0
    scaled = np.clip((stack - vmin) / (vmax - vmin), 0.0, 1.0)
    images = np.rint(255.0 * scaled).astype(np.uint8)
    # ffmpeg's broadly compatible yuv420p output requires even dimensions.
    pad_x = images.shape[1] % 2
    pad_y = images.shape[2] % 2
    if pad_x or pad_y:
        images = np.pad(images, ((0, 0), (0, pad_x), (0, pad_y)), mode="edge")
    executable = ffmpeg_path or shutil.which("ffmpeg")
    metadata = {
        "visualization_only": True,
        "status": "pending_encoding",
        "quantity": "output_plane_physical_optical_intensity",
        "value_unit": "1/um^2",
        "original_spatial_grid": dict(original_grid),
        "preview_frame_grid": {
            "Nx": int(stack.shape[1]), "Ny": int(stack.shape[2])
        },
        "encoded_frame_grid": {
            "Nx": int(images.shape[1]), "Ny": int(images.shape[2])
        },
        "original_td_cadence_normalized": float(original_cadence),
        "retained_frame_indices": [int(value) for value in frame_indices],
        "material_times_normalized": [float(value) for value in material_times],
        "spatial_reduction": "contiguous_block_arithmetic_mean",
        "fixed_color_limits": [vmin, vmax],
        "normalization": "physical optical intensity from launched channel fields",
        "container": "mp4",
        "codec": "h264",
        "frame_rate_fps": 6,
    }
    if executable is None:
        warning = "TD preview MP4 unavailable: ffmpeg executable not found"
        metadata.update({"status": "encoding_failed", "warning": warning})
        return PRTDPreviewMovie(None, metadata, warning)
    command = [
        executable, "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray",
        "-s:v", f"{images.shape[2]}x{images.shape[1]}", "-r", "6", "-i", "pipe:0",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-movflags", "frag_keyframe+empty_moov", "-f", "mp4", "pipe:1",
    ]
    try:
        completed = subprocess.run(
            command,
            input=np.ascontiguousarray(images).tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=60,
        )
        data = np.frombuffer(completed.stdout, dtype=np.uint8).copy()
        if data.size == 0:
            raise RuntimeError("ffmpeg produced an empty MP4")
    except Exception as exc:
        warning = f"TD preview MP4 encoding failed: {type(exc).__name__}: {exc}"
        metadata.update({"status": "encoding_failed", "warning": warning})
        return PRTDPreviewMovie(None, metadata, warning)
    metadata.update({"status": "encoded", "encoded_bytes": int(data.nbytes)})
    return PRTDPreviewMovie(data, metadata)


__all__ = [
    "FAST_MPR_MAX_X",
    "FAST_MPR_MAX_Y",
    "FAST_MPR_MAX_Z",
    "FAST_MPR_TARGET_BYTES",
    "PRIntensityPreview",
    "PRTDPreviewMovie",
    "TD_MOVIE_MAX_FRAMES",
    "block_average_2d",
    "downsample_td_movie_frame",
    "encode_td_preview_movie",
    "make_fast_intensity_preview",
    "td_movie_frame_indices",
    "validate_fast_intensity_preview",
]
