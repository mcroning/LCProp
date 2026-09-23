"""Build multichannel launch fields from beam specifications.

This module converts human-facing ``BeamStack`` objects into optical
channel stacks consumed by the algorithms.

It does not propagate fields, build FFT kernels, solve material state, or
manage products.

Array convention
----------------
Launch fields have shape ``(Nch, Nx, Ny)``.
A single beam is still a one-channel stack.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from lcprop.core.beams import BeamStack
from lcprop.optics.physical_launch import (
    CAPTURE_TOLERANCE, MINIMUM_FORWARD_COSINE, SPECTRAL_COSINE_RELATIVE_TOLERANCE,
    NONPROPAGATING_NORM_TOLERANCE, NYQUIST_EDGE_NORM_TOLERANCE,
    resolve_beam_geometry, sample_resolved_beam, scalar_flux_diagnostic,
)
from lcprop.optics.sampling import qualify_launch_sampling
from lcprop.core.grid import RuntimeGrid
from lcprop.optics.screens import (
    ChannelLaunchElements,
    apply_channel_launch_elements,
)

Array = Any


@dataclass(frozen=True)
class OpticalLaunchContext:
    """Material-neutral geometry needed to construct an entrance field."""

    grid: RuntimeGrid
    n_ref: float
    interaction_length_um: float
    propagation_sign: int = 1
    propagation_convention: str = "angular_spectrum_forward_z"

    def validate(self) -> None:
        if not np.isfinite(float(self.n_ref)) or float(self.n_ref) <= 0.0:
            raise ValueError("n_ref must be finite and positive")
        if not np.isfinite(float(self.interaction_length_um)) or float(
            self.interaction_length_um
        ) <= 0.0:
            raise ValueError("interaction_length_um must be finite and positive")
        if self.propagation_sign not in (-1, 1):
            raise ValueError("propagation_sign must be -1 or 1")
        if self.propagation_convention != "angular_spectrum_forward_z":
            raise ValueError("unsupported propagation convention")



@dataclass(frozen=True)
class LaunchResult:
    """Prepared multichannel optical launch."""

    A0: Array
    physical_powers_mW: Array
    power_fractions: Array
    physical_total_power_mW: float
    wavelengths_um: Array
    coherence: str
    coherence_groups: tuple[str, ...]
    post_element_physical_powers_mW: Array | None = None
    post_element_total_power_mW: float | None = None
    channel_throughput_fractions: Array | None = None
    resolved_geometry: tuple = ()
    power_metadata: dict | None = None

    def summary(self) -> dict:
        # Explicit groups are authoritative for multichannel interference.
        # Retain the legacy request label for a single channel, where coherent
        # and incoherent construction are numerically identical.
        effective_coherence = self.coherence
        if len(self.coherence_groups) > 1:
            effective_coherence = (
                "coherent"
                if len(set(self.coherence_groups)) < len(self.coherence_groups)
                else "incoherent"
            )
        post_element_powers = (
            self.physical_powers_mW
            if self.post_element_physical_powers_mW is None
            else self.post_element_physical_powers_mW
        )
        post_element_total = (
            self.physical_total_power_mW
            if self.post_element_total_power_mW is None
            else self.post_element_total_power_mW
        )
        throughput = (
            np.ones_like(np.asarray(_to_numpy(self.physical_powers_mW)))
            if self.channel_throughput_fractions is None
            else np.asarray(_to_numpy(self.channel_throughput_fractions))
        )
        qualification = (self.power_metadata or {}).get("post_screen_spectral_qualification", ())
        available = ([item["narrow_band_available"] for item in qualification]
                     if qualification else [True]*int(self.A0.shape[0]))
        return {
            "Nch": int(self.A0.shape[0]),
            "coherence": effective_coherence,
            "coherence_groups": list(self.coherence_groups),
            "physical_channel_powers_mW": [float(x) for x in np.asarray(_to_numpy(self.physical_powers_mW)).ravel()],
            "physical_total_power_mW": float(self.physical_total_power_mW),
            "post_element_channel_powers_mW": [
                float(x) if valid else None for x, valid in
                zip(np.asarray(_to_numpy(post_element_powers)).ravel(), available)
            ],
            "post_element_total_power_mW": float(post_element_total) if all(available) else None,
            "channel_throughput_fractions": [
                float(x) if valid else None for x, valid in zip(throughput.ravel(), available)
            ],
            "power_fractions": [float(x) for x in np.asarray(_to_numpy(self.power_fractions)).ravel()],
            "field_normalization": "physical_irradiance_carrier_cosine_v1",
            "power_normalization": self.power_metadata,
            "resolved_beams": [geometry.summary() for geometry in self.resolved_geometry],
            "wavelengths_um": [float(x) for x in np.asarray(_to_numpy(self.wavelengths_um)).ravel()],
        }


def _to_numpy(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore

        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def gaussian_channel(
    ch,
    grid: RuntimeGrid,
    *,
    power_fraction: float,
    complex_dtype: Any,
    context: OpticalLaunchContext | None = None,
) -> Array:
    """Sample an analytically normalized external physical beam on the face."""
    if context is None:
        raise ValueError("physical launch requires an explicit material OpticalLaunchContext")
    context.validate()
    if context.grid is not grid:
        raise ValueError("OpticalLaunchContext grid must be the launch grid")
    geometry = resolve_beam_geometry(ch, context.n_ref)
    return sample_resolved_beam(ch, geometry, grid, power_fraction, complex_dtype)


def build_launch(
    beams: BeamStack,
    grid: RuntimeGrid,
    *,
    complex_dtype: Any = np.complex64,
    launch_elements: tuple[ChannelLaunchElements, ...] = (),
    context: OpticalLaunchContext | None = None,
) -> LaunchResult:
    """Build and optionally transform ``A0`` from an incident ``BeamStack``.

    Beam powers describe the incident fields. Ordered passive launch elements
    are applied only after those fields have been normalized, and the resulting
    stack is never renormalized.
    """

    beams.validate()
    if context is None:
        raise ValueError("physical launch requires an explicit material OpticalLaunchContext")
    if context is not None:
        context.validate()
        if context.grid is not grid:
            raise ValueError("OpticalLaunchContext grid must be the launch grid")
    qualify_launch_sampling(beams, grid, context.n_ref,
                            launch_elements=launch_elements).require_valid()
    xp = grid.xp

    physical_powers_mW = xp.asarray(
        [float(ch.power_mW) for ch in beams.channels],
        dtype=grid.real_dtype,
    )
    physical_total_power_mW = float(sum(float(ch.power_mW) for ch in beams.channels))
    if physical_total_power_mW <= 0.0:
        raise ValueError("beam stack total physical power must be positive")
    power_fractions = physical_powers_mW / physical_total_power_mW

    fields = [
        gaussian_channel(
            ch,
            grid,
            power_fraction=float(power_fractions[index]),
            complex_dtype=complex_dtype,
            context=context,
        )
        for index, ch in enumerate(beams.channels)
    ]

    A0 = xp.stack(fields, axis=0).astype(complex_dtype, copy=False)
    geometry = tuple(resolve_beam_geometry(ch, context.n_ref) for ch in beams.channels)
    cosines = np.array([item.cosine_internal for item in geometry])
    captured = channel_power_integrals(A0, grid)*physical_total_power_mW*cosines
    pre_diagnostics = [scalar_flux_diagnostic(field, grid, item, physical_total_power_mW)
                       for field, item in zip(A0, geometry)]
    external_diagnostics = [scalar_flux_diagnostic(field, grid,
        replace(item, k_internal=2*np.pi*channel.n_ext/channel.wavelength_um,
                cosine_internal=item.cosine_external), physical_total_power_mW)
        for field, item, channel in zip(A0, geometry, beams.channels)]
    A0 = apply_channel_launch_elements(A0, grid, launch_elements)
    post_diagnostics = [scalar_flux_diagnostic(field, grid, item, physical_total_power_mW)
                        for field, item in zip(A0, geometry)]

    for pre, post, external in zip(pre_diagnostics, post_diagnostics, external_diagnostics):
        if not external["narrow_band_available"]:
            reason = "external spectrum is outside the narrow-band ideal-interface approximation"
            pre["reasons"].append(reason)
            post["reasons"].append(reason)
            pre["narrow_band_available"] = post["narrow_band_available"] = False
    group_flux = {}
    for group in dict.fromkeys(beams.coherence_groups):
        indices = [i for i, label in enumerate(beams.coherence_groups) if label == group]
        if len({beams.channels[i].wavelength_um for i in indices}) != 1:
            group_flux[group] = {"available": False, "reason": "coherent group has different wavelengths"}
        else:
            field = xp.sum(A0[indices], axis=0)
            diagnostic = scalar_flux_diagnostic(field, grid, geometry[indices[0]], physical_total_power_mW)
            group_flux[group] = {"available": True,
                "scalar_axial_current_mW": diagnostic["scalar_axial_current_mW"],
                "qualification": "homogeneous isotropic scalar reference; no single carrier cosine assigned to group"}

    post_element_powers_numpy = (
        channel_power_integrals(A0, grid) * physical_total_power_mW * cosines
    )
    incident_powers_numpy = np.asarray(
        [float(ch.power_mW) for ch in beams.channels],
        dtype=float,
    )
    throughput_numpy = np.divide(
        post_element_powers_numpy,
        incident_powers_numpy,
        out=np.zeros_like(post_element_powers_numpy),
        where=incident_powers_numpy > 0.0,
    )
    post_element_physical_powers_mW = xp.asarray(
        post_element_powers_numpy,
        dtype=grid.real_dtype,
    )
    channel_throughput_fractions = xp.asarray(
        throughput_numpy,
        dtype=grid.real_dtype,
    )

    wavelengths_um = xp.asarray(
        [float(ch.wavelength_um) for ch in beams.channels],
        dtype=grid.real_dtype,
    )

    return LaunchResult(
        A0=A0,
        physical_powers_mW=physical_powers_mW,
        power_fractions=power_fractions,
        physical_total_power_mW=physical_total_power_mW,
        wavelengths_um=wavelengths_um,
        coherence=beams.coherence,
        coherence_groups=beams.coherence_groups,
        post_element_physical_powers_mW=post_element_physical_powers_mW,
        post_element_total_power_mW=float(np.sum(post_element_powers_numpy)),
        channel_throughput_fractions=channel_throughput_fractions,
        resolved_geometry=geometry,
        power_metadata={
            "convention": "irradiance_amplitude_unit_interface_power_transmission",
            "capture_scope": "configured beam launch before any explicitly supplied initial_A override",
            "prepared_field_qualification": "explicit initial_A must already use the declared irradiance units; its provenance is supplied by the caller",
            "power_scale_mW": physical_total_power_mW,
            "material_reference_index": float(context.n_ref),
            "interface_qualification": "isotropic scalar reference; anisotropic/vector interface not modeled",
            "power_estimate": "central-direction normal flux; consult spectral qualification",
            "captured_pre_screen_mW": captured.tolist(),
            "post_screen_central_direction_estimate_mW": post_element_powers_numpy.tolist(),
            "requested_to_post_screen_estimate": throughput_numpy.tolist(),
            "capture_fraction": np.divide(captured, incident_powers_numpy,
                out=np.zeros_like(captured), where=incident_powers_numpy > 0).tolist(),
            "screen_transmission_of_captured": np.divide(post_element_powers_numpy, captured,
                out=np.zeros_like(captured), where=captured > 0).tolist(),
            "external_spectral_qualification": external_diagnostics,
            "coherent_group_scalar_flux": group_flux,
            "pre_screen_spectral_qualification": pre_diagnostics,
            "post_screen_spectral_qualification": post_diagnostics,
            "capture_warnings": [
                "finite aperture loss or quadrature error; no renormalization"
                if p > 0 and abs(c/p-1) > CAPTURE_TOLERANCE else ""
                for c, p in zip(captured, incident_powers_numpy)],
            "tolerances": {
                "capture": CAPTURE_TOLERANCE,
                "minimum_forward_cosine": MINIMUM_FORWARD_COSINE,
                "cosine_relative_rms": SPECTRAL_COSINE_RELATIVE_TOLERANCE,
                "nonpropagating_norm_fraction": NONPROPAGATING_NORM_TOLERANCE,
                "nyquist_edge_norm_fraction": NYQUIST_EDGE_NORM_TOLERANCE,
            },
            "unweighted_norm_to_mW_available": False,
            "unweighted_norm_to_mW_reason": "oblique/diffracting irradiance requires angular flux weighting",
        },
    )


def normalized_power(A0: Array, grid: RuntimeGrid) -> float:
    """Return the normalized channel integral ``sum_c integral |A_c|^2``."""
    xp = grid.xp
    p = xp.sum(xp.abs(A0) ** 2) * float(grid.dx_um) * float(grid.dy_um)
    return float(_to_numpy(p))


def channel_power_integrals(A0: Array, grid: RuntimeGrid) -> np.ndarray:
    """Return normalized per-channel field integrals, independent of coherence."""
    xp = grid.xp
    values = xp.sum(xp.abs(A0) ** 2, axis=(-2, -1)) * float(grid.dx_um) * float(grid.dy_um)
    return np.asarray(_to_numpy(values), dtype=float)


def reconstructed_physical_powers_mW(
    A0: Array,
    grid: RuntimeGrid,
    launch: LaunchResult,
) -> np.ndarray:
    """Return per-channel homogeneous scalar axial current, not vector power.

    Material-induced angle changes prevent inferring this from a fixed cosine
    or an unweighted norm. Coherent group totals require summing fields first.
    """
    if len(launch.resolved_geometry) != len(A0):
        raise ValueError("resolved physical launch geometry is required for scalar flux")
    return np.array([scalar_flux_diagnostic(field, grid, geometry,
                     launch.physical_total_power_mW)["scalar_axial_current_mW"]
                     for field, geometry in zip(A0, launch.resolved_geometry)])


def total_power(A0: Array, grid: RuntimeGrid) -> float:
    """Compatibility alias for :func:`normalized_power`.

    This value is a dimensionless normalized field integral, not milliwatts.
    """
    return normalized_power(A0, grid)


__all__ = [
    "Array",
    "LaunchResult",
    "OpticalLaunchContext",
    "gaussian_channel",
    "build_launch",
    "channel_power_integrals",
    "normalized_power",
    "reconstructed_physical_powers_mW",
    "total_power",
]
