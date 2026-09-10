"""Carrier-resolved Fourier power diagnostics for coherent PR beams."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from lcprop.core.backend import asnumpy


CarrierCenter = tuple[float, float]
TiePolicy = Literal["half", "first", "second"]
_SEPARATION_QUALITY_THRESHOLD = 0.9
_INPUT_POWER_RELATIVE_FLOOR = 1e-12
_INPUT_POWER_EPSILON_MULTIPLIER = 64.0


@dataclass(frozen=True)
class CarrierPartition:
    """Two-carrier Voronoi weights on an unshifted FFT grid."""

    weights: np.ndarray
    kx_rad_per_um: np.ndarray
    ky_rad_per_um: np.ndarray
    centers_k_rad_per_um: tuple[CarrierCenter, CarrierCenter]
    tie_policy: TiePolicy
    tie_pixel_count: int


def nearest_carrier_partition(
    shape: tuple[int, int],
    *,
    dx_um: float,
    dy_um: float,
    centers_k_rad_per_um: Sequence[CarrierCenter],
    tie_policy: TiePolicy = "half",
) -> CarrierPartition:
    """Partition FFT samples by the perpendicular bisector of two carriers.

    The returned weights use the repository's unshifted ``fft2`` ordering.
    With ``tie_policy="half"``, every bisector pixel contributes equally to
    both carriers and the two weights sum to one at every Fourier sample.
    """

    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("shape must contain positive (Nx, Ny)")
    if not math.isfinite(float(dx_um)) or float(dx_um) <= 0.0:
        raise ValueError("dx_um must be finite and positive")
    if not math.isfinite(float(dy_um)) or float(dy_um) <= 0.0:
        raise ValueError("dy_um must be finite and positive")
    if len(centers_k_rad_per_um) != 2:
        raise ValueError("exactly two carrier centers are required")
    centers = tuple(
        (float(center[0]), float(center[1]))
        for center in centers_k_rad_per_um
    )
    if any(
        not math.isfinite(component)
        for center in centers
        for component in center
    ):
        raise ValueError("carrier centers must be finite")
    if tie_policy not in ("half", "first", "second"):
        raise ValueError("tie_policy must be 'half', 'first', or 'second'")

    nx, ny = (int(value) for value in shape)
    kx = 2.0 * math.pi * np.fft.fftfreq(nx, d=float(dx_um))[:, None]
    ky = 2.0 * math.pi * np.fft.fftfreq(ny, d=float(dy_um))[None, :]
    first_distance = (kx - centers[0][0]) ** 2 + (ky - centers[0][1]) ** 2
    second_distance = (kx - centers[1][0]) ** 2 + (ky - centers[1][1]) ** 2
    first_closer = first_distance < second_distance
    ties = first_distance == second_distance
    if tie_policy == "half":
        first = first_closer.astype(float) + 0.5 * ties
    elif tie_policy == "first":
        first = (first_closer | ties).astype(float)
    else:
        first = first_closer.astype(float)
    weights = np.stack((first, 1.0 - first), axis=0)
    return CarrierPartition(
        weights=weights,
        kx_rad_per_um=kx[:, 0],
        ky_rad_per_um=ky[0],
        centers_k_rad_per_um=(centers[0], centers[1]),
        tie_policy=tie_policy,
        tie_pixel_count=int(np.count_nonzero(ties)),
    )


def _spectral_power(
    field: np.ndarray,
    weights: np.ndarray,
    *,
    dx_um: float,
    dy_um: float,
) -> np.ndarray:
    transformed = np.fft.fft2(field)
    scale = float(dx_um) * float(dy_um) / field.size
    density = np.abs(transformed) ** 2 * scale
    return np.sum(weights * density[None, ...], axis=(-2, -1), dtype=np.float64)


def _spatial_power(field: np.ndarray, *, dx_um: float, dy_um: float) -> float:
    return float(
        np.sum(np.abs(field) ** 2, dtype=np.float64)
        * float(dx_um)
        * float(dy_um)
    )


def _input_power_relative_threshold(field: np.ndarray) -> float:
    """Return a deterministic, dtype-aware relative gain-stability floor."""

    real_dtype = np.asarray(field.real).dtype
    epsilon = (
        float(np.finfo(real_dtype).eps)
        if np.issubdtype(real_dtype, np.floating)
        else float(np.finfo(np.float64).eps)
    )
    return max(
        _INPUT_POWER_RELATIVE_FLOOR,
        _INPUT_POWER_EPSILON_MULTIPLIER * epsilon,
    )


def _not_applicable(reason: str) -> dict[str, Any]:
    return {
        "status": "not_applicable",
        "reason": reason,
        "carrier_partition_method": "two_carrier_perpendicular_bisector",
        "carrier_tie_policy": "equal_half_weight",
        "rows": [],
    }


def carrier_channels_from_beams(beams) -> list[dict[str, Any]]:
    """Return compact input carrier metadata from a validated BeamStack."""

    beams.validate()
    return [
        {
            "name": str(channel.name),
            "kx_rad_per_um": float(channel.tilt_x_rad_per_um),
            "ky_rad_per_um": float(channel.tilt_y_rad_per_um),
        }
        for channel in beams.channels
    ]


def carrier_power_diagnostic(
    A_initial,
    A_final,
    *,
    dx_um: float,
    dy_um: float,
    coherence_groups: Sequence[str],
    carrier_channels: Sequence[Mapping[str, Any]],
    physical_total_power_mW: float | None = None,
) -> dict[str, Any]:
    """Measure two coherent input-carrier regions at input and output.

    Channel arrays are used only to form the input separation-quality metric.
    The reported powers come from the Fourier transform of the coherently
    combined group field, so they do not measure channel-lineage norms.
    """

    initial = np.asarray(asnumpy(A_initial))
    final = np.asarray(asnumpy(A_final))
    if initial.ndim != 3 or final.shape != initial.shape:
        raise ValueError("A_initial and A_final must share shape (Nch, Nx, Ny)")
    if initial.shape[0] != 2:
        return _not_applicable("exactly two optical channels are required")
    if len(coherence_groups) != 2:
        return _not_applicable("coherence metadata must match two channels")
    if str(coherence_groups[0]) != str(coherence_groups[1]):
        return _not_applicable(
            "the two channels belong to different coherence groups"
        )
    if len(carrier_channels) != 2:
        return _not_applicable("two input carrier centers are unavailable")

    try:
        centers = tuple(
            (
                float(channel["kx_rad_per_um"]),
                float(channel["ky_rad_per_um"]),
            )
            for channel in carrier_channels
        )
        labels = tuple(
            str(channel.get("name") or f"Carrier {index + 1}")
            for index, channel in enumerate(carrier_channels)
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _not_applicable(f"invalid input carrier metadata: {exc}")

    partition = nearest_carrier_partition(
        initial.shape[-2:],
        dx_um=dx_um,
        dy_um=dy_um,
        centers_k_rad_per_um=centers,
        tie_policy="half",
    )
    combined_initial = initial[0] + initial[1]
    combined_final = final[0] + final[1]
    input_power = _spectral_power(
        combined_initial, partition.weights, dx_um=dx_um, dy_um=dy_um
    )
    output_power = _spectral_power(
        combined_final, partition.weights, dx_um=dx_um, dy_um=dy_um
    )
    delta_power = output_power - input_power
    total_input = _spatial_power(combined_initial, dx_um=dx_um, dy_um=dy_um)
    total_output = _spatial_power(combined_final, dx_um=dx_um, dy_um=dy_um)
    input_partition_error = float(np.sum(input_power) - total_input)
    output_partition_error = float(np.sum(output_power) - total_output)
    input_power_relative_threshold = _input_power_relative_threshold(initial)
    input_power_threshold = input_power_relative_threshold * max(total_input, 0.0)
    insufficient_input_power = input_power <= input_power_threshold

    isolated_input_power = np.asarray([
        _spectral_power(
            initial[index],
            partition.weights,
            dx_um=dx_um,
            dy_um=dy_um,
        )
        for index in range(2)
    ])
    isolated_totals = np.sum(isolated_input_power, axis=1)
    wrong_side_fraction = np.divide(
        (isolated_input_power[0, 1], isolated_input_power[1, 0]),
        isolated_totals,
        out=np.full(2, 0.5, dtype=float),
        where=isolated_totals > 0.0,
    )
    separation_quality = float(1.0 - np.max(wrong_side_fraction))
    separated = bool(separation_quality >= _SEPARATION_QUALITY_THRESHOLD)
    gains: list[float | None] = []
    gain_statuses: list[str] = []
    gain_unavailable_reasons: list[str | None] = []
    for before, after, insufficient in zip(
        input_power, output_power, insufficient_input_power
    ):
        if bool(insufficient):
            gains.append(None)
            gain_statuses.append("insufficient_input_carrier_power")
            gain_unavailable_reasons.append(
                "input carrier power is at or below the relative stability threshold"
            )
        elif not separated:
            gains.append(None)
            gain_statuses.append("insufficient_separation")
            gain_unavailable_reasons.append(
                "input carrier spectra are insufficiently separated"
            )
        else:
            gains.append(float(after / before))
            gain_statuses.append("available")
            gain_unavailable_reasons.append(None)

    any_insufficient_input = bool(np.any(insufficient_input_power))
    if any_insufficient_input:
        status = "insufficient_input_carrier_power"
        reason = (
            "one or more input carrier powers are at or below the relative "
            "stability threshold"
        )
    elif not separated:
        status = "insufficient_separation"
        reason = "input carrier spectra are insufficiently separated"
    else:
        status = "ok"
        reason = None

    physical_scale = None
    if physical_total_power_mW is not None:
        candidate = float(physical_total_power_mW)
        if math.isfinite(candidate) and candidate > 0.0:
            physical_scale = candidate
    input_mw = None if physical_scale is None else input_power * physical_scale
    output_mw = None if physical_scale is None else output_power * physical_scale
    delta_mw = None if physical_scale is None else delta_power * physical_scale

    rows = []
    for index, label in enumerate(labels):
        rows.append({
            "carrier": label,
            "input_power_normalized": float(input_power[index]),
            "output_power_normalized": float(output_power[index]),
            "gain": gains[index],
            "gain_status": gain_statuses[index],
            "gain_unavailable_reason": gain_unavailable_reasons[index],
            "delta_power_normalized": float(delta_power[index]),
            "input_power_mW": (
                None if input_mw is None else float(input_mw[index])
            ),
            "output_power_mW": (
                None if output_mw is None else float(output_mw[index])
            ),
            "delta_power_mW": (
                None if delta_mw is None else float(delta_mw[index])
            ),
        })

    return {
        "status": status,
        "reason": reason,
        "warning": None if status == "ok" else reason,
        "carrier_power_input": [float(value) for value in input_power],
        "carrier_power_output": [float(value) for value in output_power],
        "carrier_gain": gains,
        "carrier_gain_status": gain_statuses,
        "carrier_gain_unavailable_reason": gain_unavailable_reasons,
        "carrier_delta_power": [float(value) for value in delta_power],
        "carrier_power_input_mW": (
            None if input_mw is None else [float(value) for value in input_mw]
        ),
        "carrier_power_output_mW": (
            None if output_mw is None else [float(value) for value in output_mw]
        ),
        "carrier_delta_power_mW": (
            None if delta_mw is None else [float(value) for value in delta_mw]
        ),
        "carrier_centers_k": [list(center) for center in centers],
        "carrier_partition_method": "two_carrier_perpendicular_bisector",
        "carrier_tie_policy": "equal_half_weight",
        "carrier_tie_pixel_count": partition.tie_pixel_count,
        "carrier_separation_quality": separation_quality,
        "carrier_wrong_side_fraction": [
            float(value) for value in wrong_side_fraction
        ],
        "carrier_separation_quality_threshold": _SEPARATION_QUALITY_THRESHOLD,
        "carrier_input_power_relative_threshold": input_power_relative_threshold,
        "carrier_input_power_threshold": input_power_threshold,
        "carrier_power_sum_input": float(np.sum(input_power)),
        "carrier_power_sum_output": float(np.sum(output_power)),
        "carrier_power_sum_drift": float(np.sum(delta_power)),
        "coherent_group_power_input": total_input,
        "coherent_group_power_output": total_output,
        "coherent_group_power_drift": float(total_output - total_input),
        "carrier_power_balance_error": float(
            np.sum(delta_power) - (total_output - total_input)
        ),
        "parseval_partition_error_input": input_partition_error,
        "parseval_partition_error_output": output_partition_error,
        "normalized_power_unit": "normalized optical integral",
        "physical_power_unit": "mW" if physical_scale is not None else None,
        "analyzed_coherence_group": str(coherence_groups[0]),
        "rows": rows,
    }


def carrier_power_diagnostic_from_summary(
    A_initial,
    A_final,
    *,
    grid_summary: Mapping[str, Any],
    launch_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the diagnostic from compact result endpoint metadata."""

    channels = launch_summary.get("carrier_channels")
    if not isinstance(channels, Sequence):
        return _not_applicable("input carrier-center metadata is unavailable")
    groups = launch_summary.get("coherence_groups")
    if not isinstance(groups, Sequence):
        return _not_applicable("coherence-group metadata is unavailable")
    return carrier_power_diagnostic(
        A_initial,
        A_final,
        dx_um=float(grid_summary["dx_um"]),
        dy_um=float(grid_summary["dy_um"]),
        coherence_groups=groups,
        carrier_channels=channels,
        physical_total_power_mW=launch_summary.get("physical_total_power_mW"),
    )


__all__ = [
    "CarrierPartition",
    "carrier_channels_from_beams",
    "carrier_power_diagnostic",
    "carrier_power_diagnostic_from_summary",
    "nearest_carrier_partition",
]
