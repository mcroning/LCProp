"""Partition-independent photorefractive volume-scattering increments.

The trusted PRProp3D model applies independent, transversely correlated phase
screens whose unfiltered variance is proportional to ``epsilon / Ns``.  The
least-assumptive partition-independent interpretation is therefore a
white-in-z random phase measure: a screen represents the phase increment over
an interval, not a point sample of a smooth three-dimensional field.

This module represents that measure with deterministic canonical physical-z
slabs.  Every canonical slab has a coordinate-addressed seed, and a material
interval receives the sum of all canonical increments that it contains.  A
coarse interval is consequently the same realization as the corresponding
set of fine intervals without retaining a three-dimensional array.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np


PR_CANONICAL_SCATTERING_ALGORITHM = "canonical_phase_slabs_v1"
_SEED_TAG = 0x4C435052  # ASCII-derived fixed namespace tag: "LCPR"


@dataclass(frozen=True)
class PRCanonicalScatteringSpec:
    """Configuration for a partition-independent white-in-z phase process.

    ``epsilon`` is the expected accumulated phase variance over the complete
    longitudinal domain under the trusted PRProp3D transverse-filter
    normalization.  ``canonical_dz_um`` defines the physical z slabs that
    address independent increments.  Every requested material interval must
    be an integer union of canonical slabs.
    """

    epsilon: float
    transverse_correlation_um: float
    realization_seed: int
    canonical_dz_um: float

    def validate(self) -> None:
        epsilon = float(self.epsilon)
        correlation = float(self.transverse_correlation_um)
        canonical_dz = float(self.canonical_dz_um)
        if not math.isfinite(epsilon) or epsilon < 0.0:
            raise ValueError("epsilon must be finite and nonnegative")
        if not math.isfinite(correlation) or correlation <= 0.0:
            raise ValueError(
                "transverse_correlation_um must be finite and positive"
            )
        if (
            int(self.realization_seed) != self.realization_seed
            or not 0 <= int(self.realization_seed) <= 2**32 - 1
        ):
            raise ValueError("realization_seed must be an unsigned 32-bit integer")
        if not math.isfinite(canonical_dz) or canonical_dz <= 0.0:
            raise ValueError("canonical_dz_um must be finite and positive")


def _aligned_index(value_um: float, quantum_um: float, *, name: str) -> int:
    value = float(value_um)
    quantum = float(quantum_um)
    index = int(round(value / quantum))
    represented = index * quantum
    tolerance = 1e-10 * max(1.0, abs(value), abs(represented))
    if not math.isclose(value, represented, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{name} must align with canonical_dz_um")
    return index


def canonical_slab_range(
    spec: PRCanonicalScatteringSpec,
    *,
    z_start_um: float,
    dz_um: float,
    z_length_um: float,
) -> range:
    """Return canonical slab indices in the half-open material interval."""

    spec.validate()
    z_start = float(z_start_um)
    dz = float(dz_um)
    length = float(z_length_um)
    if not math.isfinite(z_start) or z_start < 0.0:
        raise ValueError("z_start_um must be finite and nonnegative")
    if not math.isfinite(dz) or dz <= 0.0:
        raise ValueError("dz_um must be finite and positive")
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("z_length_um must be finite and positive")

    quantum = float(spec.canonical_dz_um)
    slab_count = _aligned_index(length, quantum, name="z_length_um")
    first = _aligned_index(z_start, quantum, name="z_start_um")
    last = _aligned_index(z_start + dz, quantum, name="z_start_um + dz_um")
    if first < 0 or last <= first or last > slab_count:
        raise ValueError("requested scattering interval lies outside the z domain")
    return range(first, last)


def canonical_slab_seed(spec: PRCanonicalScatteringSpec, slab_index: int) -> int:
    """Return the deterministic seed addressed by one physical-z slab."""

    spec.validate()
    index = int(slab_index)
    if index != slab_index or index < 0:
        raise ValueError("slab_index must be a nonnegative integer")
    return int(
        np.random.SeedSequence(
            [int(spec.realization_seed), _SEED_TAG, index]
        ).generate_state(1)[0]
    )


def _gaussian_filter(raw, *, sigma: tuple[float, float], xp: Any):
    if getattr(xp, "__name__", "") == "cupy":
        from cupyx.scipy.ndimage import gaussian_filter  # type: ignore
    else:
        from scipy.ndimage import gaussian_filter

    return gaussian_filter(raw, sigma=sigma)


def canonical_scattering_phase_increment(
    spec: PRCanonicalScatteringSpec,
    *,
    z_start_um: float,
    dz_um: float,
    z_length_um: float,
    Nx: int,
    Ny: int,
    x_aperture_um: float,
    y_aperture_um: float,
    real_dtype: Any,
    xp: Any,
):
    """Return the phase increment for one material interval.

    Random arrays are generated on the requested NumPy/CuPy backend.  Only the
    scalar slab seed is derived on the host.  Memory is ``O(Nx*Ny)`` and does
    not depend on the number of longitudinal slabs.
    """

    slabs = canonical_slab_range(
        spec,
        z_start_um=z_start_um,
        dz_um=dz_um,
        z_length_um=z_length_um,
    )
    nx = int(Nx)
    ny = int(Ny)
    x_aperture = float(x_aperture_um)
    y_aperture = float(y_aperture_um)
    if nx < 1 or ny < 1:
        raise ValueError("Nx and Ny must be positive")
    if x_aperture <= 0.0 or y_aperture <= 0.0:
        raise ValueError("transverse apertures must be positive")

    sigma_x = float(spec.transverse_correlation_um) * nx / x_aperture
    sigma_y = float(spec.transverse_correlation_um) * ny / y_aperture
    if sigma_x <= 0.0 or sigma_y <= 0.0:
        raise ValueError("resolved transverse filter widths must be positive")

    raw = xp.zeros((nx, ny), dtype=real_dtype)
    if float(spec.epsilon) == 0.0:
        return raw
    scale = math.sqrt(
        float(spec.epsilon)
        * float(spec.canonical_dz_um)
        / float(z_length_um)
        * 4.0
        * math.pi
    )
    for slab_index in slabs:
        seed = canonical_slab_seed(spec, slab_index)
        raw += (
            xp.random.RandomState(seed)
            .normal(0.0, scale, size=(nx, ny))
            .astype(real_dtype)
        )
    return _gaussian_filter(raw, sigma=(sigma_x, sigma_y), xp=xp) * math.sqrt(
        sigma_x * sigma_y
    )


def canonical_scattering_provenance(
    spec: PRCanonicalScatteringSpec,
    *,
    z_length_um: float,
    Nx: int,
    Ny: int,
    x_aperture_um: float,
    y_aperture_um: float,
    real_dtype: Any,
    xp: Any,
) -> dict[str, Any]:
    """Return compact manifest data sufficient to regenerate a realization."""

    spec.validate()
    length = float(z_length_um)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("z_length_um must be finite and positive")
    if int(Nx) < 1 or int(Ny) < 1:
        raise ValueError("Nx and Ny must be positive")
    if float(x_aperture_um) <= 0.0 or float(y_aperture_um) <= 0.0:
        raise ValueError("transverse apertures must be positive")
    slab_count = _aligned_index(
        length,
        float(spec.canonical_dz_um),
        name="z_length_um",
    )
    seeds = np.asarray(
        [canonical_slab_seed(spec, index) for index in range(slab_count)],
        dtype="<u4",
    )
    backend_name = getattr(xp, "__name__", type(xp).__name__)
    backend_version = getattr(xp, "__version__", "unknown")
    if backend_name == "cupy":
        filter_library = f"cupyx.scipy from cupy {backend_version}"
    else:
        import scipy

        filter_library = f"scipy {scipy.__version__}"
    configuration = {
        "mode": "partition_independent_canonical_phase_slabs",
        "algorithm_version": PR_CANONICAL_SCATTERING_ALGORITHM,
        "epsilon_total_phase_variance": float(spec.epsilon),
        "transverse_correlation_um": float(spec.transverse_correlation_um),
        "realization_seed": int(spec.realization_seed),
        "canonical_dz_um": float(spec.canonical_dz_um),
        "z_domain_um": [0.0, float(z_length_um)],
        "transverse_shape": [int(Nx), int(Ny)],
        "transverse_aperture_um": [
            float(x_aperture_um),
            float(y_aperture_um),
        ],
        "coordinate_convention": (
            "canonical slab i is [i*canonical_dz_um, "
            "(i+1)*canonical_dz_um); material screens are interval sums"
        ),
        "normalization_convention": (
            "each canonical unfiltered increment has variance "
            "epsilon*canonical_dz_um/z_length_um*4*pi before the trusted "
            "Gaussian filter and sqrt(sigma_x*sigma_y) factor"
        ),
        "longitudinal_model": "independent white-in-z phase increments",
        "transverse_filter": "scipy-compatible gaussian_filter default boundary",
        "transverse_filter_library": filter_library,
        "rng_addressing": "SeedSequence([realization_seed, LCPR_tag, slab_index])",
        "rng_backend": f"{backend_name}.random.RandomState",
        "rng_backend_version": str(backend_version),
        "seedsequence_numpy_version": np.__version__,
        "real_dtype": str(np.dtype(real_dtype)),
    }
    encoded = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        **configuration,
        "canonical_slab_count": slab_count,
        "canonical_seed_sha256_le_u32": hashlib.sha256(seeds.tobytes()).hexdigest(),
        "first_canonical_seed": int(seeds[0]),
        "last_canonical_seed": int(seeds[-1]),
        "configuration_sha256": hashlib.sha256(encoded).hexdigest(),
    }


__all__ = [
    "PR_CANONICAL_SCATTERING_ALGORITHM",
    "PRCanonicalScatteringSpec",
    "canonical_scattering_phase_increment",
    "canonical_scattering_provenance",
    "canonical_slab_range",
    "canonical_slab_seed",
]
