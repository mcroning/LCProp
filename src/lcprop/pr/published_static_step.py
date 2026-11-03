"""Published optical-first cell driver shared by direct and nonlinear material callbacks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from lcprop.core.backend import BackendSpec, get_backend
from lcprop.core.grid import RuntimeGrid
from lcprop.optics.splitstep import (
    apply_response_screen_inplace, hop_linear_inplace, scalar_angular_spectrum_kernel,
)
from lcprop.pr.optical_response import delta_n_from_E
from lcprop.pr.scattering import (
    PR_CANONICAL_SCATTERING_V2, PRCanonicalScatteringSpec,
    canonical_scattering_phase_increment, canonical_slab_range,
)
from lcprop.pr.source import pr_driving_intensity
from lcprop.pr.specs import PRMaterialSpec


@dataclass(frozen=True)
class LocalPlaneCellResult:
    """Bounded candidate state, not a workflow checkpoint or acceptance claim."""

    A_candidate: Any
    E: Any
    source_intensity: Any
    diagnostics: dict[str, Any]


def _finite(value, *, xp, name):
    if not bool(xp.all(xp.isfinite(value)).item()):
        raise ValueError(f"{name} must be finite")


def step_published_cell(
    A,
    *,
    material_response,
    material_model: str,
    grid: RuntimeGrid,
    cell_index: int,
    z_start_um: float,
    dz_um: float,
    interaction_length_um: float,
    wavelength_um: float,
    material: PRMaterialSpec,
    peak_intensity_reference: float,
    backend: BackendSpec,
    coherence_groups: tuple[str, ...] | None = None,
    scattering: PRCanonicalScatteringSpec | None = None,
    residual_rms_tolerance: float = 1e-5,
    residual_max_tolerance: float = 1e-4,
) -> LocalPlaneCellResult:
    """Published cell with requested width h and transactional input ownership.

    material_response consumes the arriving transport intensity and returns E
    and its model-specific residual. Only the optical driver applies M and S.
    The binary64 domain policy and canonical slab addressing match the reviewed
    cell primitive; metadata canonicalization never changes the requested h.
    """
    material.validate()
    resolved = get_backend(backend)
    xp = resolved.xp
    if grid.xp is not xp or not isinstance(A, xp.ndarray):
        raise TypeError("field and grid must reside on the requested backend")
    if A.dtype != xp.dtype(resolved.complex_dtype):
        raise TypeError("field must have the execution complex dtype")
    if A.ndim != 3 or A.shape[0] < 1 or A.shape[1:] != (grid.Nx, grid.Ny):
        raise ValueError("field must have shape (Nch, Nx, Ny) matching grid")
    if grid.real_dtype != resolved.real_dtype:
        raise TypeError("grid must have the execution real dtype")
    if isinstance(cell_index, bool) or int(cell_index) != cell_index or cell_index < 0:
        raise ValueError("cell_index must be a nonnegative integer")
    z, h, length = map(float, (z_start_um, dz_um, interaction_length_um))
    for name, value in (("dz", h), ("length", length),
                        ("wavelength", wavelength_um),
                        ("peak reference", peak_intensity_reference),
                        ("residual RMS tolerance", residual_rms_tolerance),
                        ("residual max tolerance", residual_max_tolerance)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    computed_end = z + h
    if (not math.isfinite(z) or not math.isfinite(computed_end)
            or z < 0 or z >= length or computed_end <= z):
        raise ValueError("cell must lie within the original interaction domain")
    # Four half-ULP contributions (z, h, their addition, declared L) are
    # bounded by twice the largest spacing. No user-scale absolute tolerance.
    endpoint_bound = 2 * max(map(math.ulp, (z, h, computed_end, length)))
    at_domain_end = abs(computed_end - length) <= endpoint_bound
    if computed_end > length and not at_domain_end:
        raise ValueError("cell must lie within the original interaction domain")
    end = length if at_domain_end else computed_end
    scatter_identity = None
    if scattering is not None:
        scattering.validate()
        if scattering.algorithm_version != PR_CANONICAL_SCATTERING_V2:
            raise ValueError("published-order stepping requires canonical scattering V2")
        slabs = canonical_slab_range(
            scattering, z_start_um=z, dz_um=h, z_length_um=length,
        )
        if at_domain_end:
            domain_slabs = canonical_slab_range(
                scattering, z_start_um=0., dz_um=length, z_length_um=length,
            )
            if slabs.stop != domain_slabs.stop:
                raise ValueError("canonical endpoint disagrees with scattering slab range")
        scatter_identity = {
            "spec": asdict(scattering), "slab_start": slabs.start,
            "slab_stop_exclusive": slabs.stop, "z_length_um": length,
            "Nx": grid.Nx, "Ny": grid.Ny,
            "x_aperture_um": grid.spec.x_aperture_um,
            "y_aperture_um": grid.spec.y_aperture_um,
            "dtype": str(xp.dtype(resolved.real_dtype)),
            "backend": resolved.name,
            "placement": "cell_right_endpoint_after_full_material_phase",
        }
    _finite(A, xp=xp, name="incoming field")
    candidate = A.copy()
    kernel = scalar_angular_spectrum_kernel(
        grid.fxy2_um, dz=h, wavelength=wavelength_um,
        n_ref=material.refractive_index, complex_dtype=resolved.complex_dtype, xp=xp,
    )
    hop_linear_inplace(candidate, kernel, xp=xp)
    intensity = pr_driving_intensity(
        candidate, peak_intensity_reference=peak_intensity_reference,
        background_intensity=material.background_intensity,
        coherence_groups=coherence_groups, xp=xp,
    )
    source_min = float(xp.min(intensity).item())
    source_max = float(xp.max(intensity).item())
    solution = material_response(intensity)
    solver_evidence = getattr(solution, "solver_evidence", None)
    E, residual = solution.E, solution.residual
    _finite(intensity, xp=xp, name="source")
    _finite(E, xp=xp, name="material")
    _finite(residual, xp=xp, name="material residual")
    residual_max = float(xp.max(xp.abs(residual)).item())
    # Scale before squaring: finite residuals must not overflow the RMS sum.
    residual_rms = (residual_max * float(xp.sqrt(xp.mean(
        (residual / residual_max) ** 2, dtype=xp.float64,
    )).item()) if residual_max else 0.0)
    if residual_rms > residual_rms_tolerance or residual_max > residual_max_tolerance:
        raise ValueError("material residual exceeds cell validity limits")
    del residual, solution
    dn = delta_n_from_E(
        E, gain_length_product=material.gain_length_product,
        interaction_length_um=length, wavelength_um=wavelength_um,
    )
    phase = xp.exp(1j * (2 * math.pi / wavelength_um) * h * dn)
    del dn
    _finite(phase, xp=xp, name="material phase")
    apply_response_screen_inplace(candidate, phase, xp=xp)
    del phase
    if scattering is not None:
        noise = canonical_scattering_phase_increment(
            scattering, z_start_um=z, dz_um=h, z_length_um=length,
            Nx=grid.Nx, Ny=grid.Ny,
            x_aperture_um=grid.spec.x_aperture_um,
            y_aperture_um=grid.spec.y_aperture_um,
            real_dtype=resolved.real_dtype, xp=xp,
        )
        _finite(noise, xp=xp, name="scattering phase")
        phase = xp.exp(1j * noise)
        del noise
        apply_response_screen_inplace(candidate, phase, xp=xp)
        del phase
    _finite(candidate, xp=xp, name="candidate boundary field")
    return LocalPlaneCellResult(candidate, E, intensity, {
        "cell_index": int(cell_index), "z_start_um": z, "z_end_um": end,
        "computed_z_end_um": computed_end,
        "endpoint_roundoff_bound_um": endpoint_bound,
        "endpoint_canonicalized": end != computed_end,
        "material_plane_um": end, "material_weight_um": h,
        "optical_step_um": h,
        "stage": "candidate_boundary_after_full_material_and_scattering",
        "source_min": source_min, "source_max": source_max,
        "source_finite": True, "material_finite": True, "boundary_finite": True,
        "residual_rms": residual_rms, "residual_max": residual_max,
        "residual_rms_tolerance": float(residual_rms_tolerance),
        "residual_max_tolerance": float(residual_max_tolerance),
        "peak_intensity_reference": float(peak_intensity_reference),
        "scattering": scatter_identity,
        "material_model": material_model, "material_solver": solver_evidence,
    })
