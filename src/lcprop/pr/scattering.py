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


PR_CANONICAL_SCATTERING_V1 = "canonical_phase_slabs_v1"
PR_CANONICAL_SCATTERING_V2 = "canonical_phase_slabs_v2_cross_backend"
# Historical public name remains the v1 identity for source compatibility.
PR_CANONICAL_SCATTERING_ALGORITHM = PR_CANONICAL_SCATTERING_V1
_SEED_TAG = 0x4C435052  # ASCII-derived fixed namespace tag: "LCPR"
_UINT32_SCALE = 1.0 / 4294967296.0


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
    algorithm_version: str = PR_CANONICAL_SCATTERING_V1

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
        if self.algorithm_version not in (
            PR_CANONICAL_SCATTERING_V1,
            PR_CANONICAL_SCATTERING_V2,
        ):
            raise ValueError("unsupported canonical scattering algorithm_version")


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


def _hash_u32(counter, *, seed: int, stream: int, xp: Any):
    """Return a counter-addressed uint32 hash on NumPy or CuPy."""

    value = counter ^ xp.asarray(seed, dtype=xp.uint32)
    value ^= xp.asarray((stream * 0x9E3779B9) & 0xFFFFFFFF, dtype=xp.uint32)
    value ^= value >> xp.uint32(16)
    value *= xp.uint32(0x7FEB352D)
    value ^= value >> xp.uint32(15)
    value *= xp.uint32(0x846CA68B)
    value ^= value >> xp.uint32(16)
    return value


def canonical_white_noise_field(
    shape: tuple[int, int], *, seed: int, real_dtype: Any, xp: Any
):
    """Generate one backend-independent counter-addressed normal field.

    Integer hashes define the realization exactly. Box--Muller conversion is
    evaluated with backend-native float64 elementwise operations and then cast
    to the requested precision; no backend RNG state or host field is used.
    """

    nx, ny = (int(shape[0]), int(shape[1]))
    if nx < 1 or ny < 1:
        raise ValueError("white-noise shape entries must be positive")
    counter = xp.arange(nx * ny, dtype=xp.uint32)
    first = _hash_u32(counter, seed=int(seed), stream=1, xp=xp)
    second = _hash_u32(counter, seed=int(seed), stream=2, xp=xp)
    u1 = (first.astype(xp.float64) + 0.5) * _UINT32_SCALE
    u2 = (second.astype(xp.float64) + 0.5) * _UINT32_SCALE
    normal = xp.sqrt(-2.0 * xp.log(u1)) * xp.cos(2.0 * math.pi * u2)
    return normal.reshape(nx, ny).astype(real_dtype, copy=False)


def _reflect_indices(length: int, offset: int, *, xp: Any):
    """Return SciPy ``mode='reflect'`` indices for one correlation offset."""

    coordinate = xp.arange(int(length), dtype=xp.int64) + int(offset)
    folded = coordinate % (2 * int(length))
    return xp.where(folded < length, folded, 2 * int(length) - 1 - folded)


def _gaussian_kernel(sigma: float) -> tuple[np.ndarray, np.ndarray]:
    radius = int(4.0 * float(sigma) + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    if radius == 0:
        return offsets, np.ones(1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets.astype(np.float64) / float(sigma)) ** 2)
    weights /= np.sum(weights)
    return offsets, weights


def _correlate_axis(field, *, sigma: float, axis: int, xp: Any):
    offsets, weights = _gaussian_kernel(sigma)
    if offsets.size == 1:
        return field.copy()
    result = xp.zeros_like(field)
    length = field.shape[axis]
    for offset, weight in zip(offsets, weights):
        indices = _reflect_indices(length, int(offset), xp=xp)
        result += xp.asarray(weight, dtype=field.dtype) * xp.take(
            field, indices, axis=axis
        )
    return result


def _gaussian_filter(raw, *, sigma: tuple[float, float], xp: Any):
    """Apply one PR-owned separable Gaussian with SciPy reflect semantics."""

    filtered = _correlate_axis(raw, sigma=float(sigma[0]), axis=0, xp=xp)
    return _correlate_axis(filtered, sigma=float(sigma[1]), axis=1, xp=xp)


def _legacy_gaussian_filter(raw, *, sigma: tuple[float, float], xp: Any):
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
        if spec.algorithm_version == PR_CANONICAL_SCATTERING_V1:
            raw += (
                xp.random.RandomState(seed)
                .normal(0.0, scale, size=(nx, ny))
                .astype(real_dtype)
            )
        else:
            raw += canonical_white_noise_field(
                (nx, ny), seed=seed, real_dtype=real_dtype, xp=xp
            ) * scale
    filter_function = (
        _legacy_gaussian_filter
        if spec.algorithm_version == PR_CANONICAL_SCATTERING_V1
        else _gaussian_filter
    )
    return filter_function(raw, sigma=(sigma_x, sigma_y), xp=xp) * math.sqrt(
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
    if spec.algorithm_version == PR_CANONICAL_SCATTERING_V1:
        if backend_name == "cupy":
            filter_library = f"cupyx.scipy from cupy {backend_version}"
        else:
            import scipy

            filter_library = f"scipy {scipy.__version__}"
        transverse_filter = "scipy-compatible gaussian_filter default boundary"
        rng_backend = f"{backend_name}.random.RandomState"
        rng_backend_version = str(backend_version)
    else:
        filter_library = "lcprop.pr.scattering backend-native primitives"
        transverse_filter = (
            "PR-owned separable Gaussian, truncate=4, half-sample reflect boundary"
        )
        rng_backend = "PR-owned uint32 counter hash plus Box-Muller float64"
        rng_backend_version = PR_CANONICAL_SCATTERING_V2
    configuration = {
        "mode": "partition_independent_canonical_phase_slabs",
        "algorithm_version": spec.algorithm_version,
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
        "transverse_filter": transverse_filter,
        "transverse_filter_library": filter_library,
        "rng_addressing": "SeedSequence([realization_seed, LCPR_tag, slab_index])",
        "rng_backend": rng_backend,
        "rng_backend_version": rng_backend_version,
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
    "PR_CANONICAL_SCATTERING_V1",
    "PR_CANONICAL_SCATTERING_V2",
    "PRCanonicalScatteringSpec",
    "canonical_white_noise_field",
    "canonical_scattering_phase_increment",
    "canonical_scattering_provenance",
    "canonical_slab_range",
    "canonical_slab_seed",
]
