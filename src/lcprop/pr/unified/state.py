"""Borrowed runtime-array records with metadata-only structural validation.

No constructor or validator copies, scans, normalizes or transfers array data.
Frozen records do not make their buffers immutable. The owner must keep borrowed
buffers alive and unmodified while a consumer uses the record. Physical acceptance
and transactional ownership are responsibilities of a future solver, not M1.
"""
from dataclasses import dataclass
import sys
from typing import Any

from lcprop.pr.illumination import LEGACY_NORMALIZATION, INTEGRAL_NORMALIZATION

from .specs import (
    DIAGNOSTICS_ID, MATERIAL_ID, NORMALIZATION_ID, NORMALIZED_UNITS, STATE_ID,
    TRANSPORT_INTENSITY_ID, PRElectricalClosureSpec, PRMaterialPrecisionSpec,
    PRUnifiedSpatialSpec, _real,
)


def _array_metadata(array, *, shape, dtype, backend, name):
    if backend not in ("numpy", "cupy"):
        raise ValueError("an explicit resolved numpy or cupy backend is required")
    module = sys.modules.get(backend)
    array_type = getattr(module, "ndarray", None)
    if array_type is None or not isinstance(array, array_type):
        raise ValueError(f"{name} must already be a {backend} ndarray")
    if tuple(array.shape) != shape:
        raise ValueError(f"{name} shape must be {shape}")
    if array.dtype.name != dtype or not array.dtype.isnative:
        raise ValueError(f"{name} must have native {dtype} dtype")
    return array.device.id if backend == "cupy" else None


def _specs(spatial, precision):
    if not isinstance(spatial, PRUnifiedSpatialSpec) or not isinstance(precision, PRMaterialPrecisionSpec):
        raise TypeError("explicit unified spatial and precision specifications required")
    spatial.validate()
    precision.validate()


@dataclass(frozen=True, eq=False)
class PRUnifiedMaterialState:
    """q and psi have field_shape; b has batch_shape + (active dimension,).

    q represents log carrier; that interpretation alone proves neither finite
    exp(q) nor satisfaction of Gauss/neutrality/gauge. Arrays are borrowed.
    Equality is object identity, never an implicit elementwise array comparison.
    """

    q: Any
    psi: Any
    b: Any
    spatial: PRUnifiedSpatialSpec
    closure: PRElectricalClosureSpec
    precision: PRMaterialPrecisionSpec
    backend: str
    state_id: str = STATE_ID
    material_id: str = MATERIAL_ID

    def validate_structure(self):
        _specs(self.spatial, self.precision)
        if self.state_id != STATE_ID or self.material_id != MATERIAL_ID:
            raise ValueError("unsupported state/material identity")
        if not isinstance(self.closure, PRElectricalClosureSpec):
            raise TypeError("explicit electrical closure required")
        self.closure.validate_spatial(self.spatial)
        devices = []
        for name, shape in (("q", self.spatial.field_shape), ("psi", self.spatial.field_shape),
                            ("b", self.spatial.harmonic_shape)):
            devices.append(_array_metadata(getattr(self, name), shape=shape,
                dtype=self.precision.state_dtype, backend=self.backend, name=name))
        if len(set(devices)) != 1:
            raise ValueError("state arrays must be on the same device")


@dataclass(frozen=True, eq=False)
class PRTransportIntensity:
    """Borrowed total I, with optical/reference + dark + uniform provenance.

    Validation checks scalar metadata and layout only. It does not recompute I,
    inspect its values, or certify that the caller used the declared convention.
    Dark intensity is not a carrier floor; peak reference is not dark intensity.
    The historical scalar field name ``peak_intensity_reference`` is retained
    for compatibility. Under integral normalization it carries the positive
    physical mean reference irradiance in W/cm², not an optical peak. The dark
    and uniform values here are derived dimensionless fractions; physical
    input irradiances and the complete reference live in workflow provenance.
    """

    values: Any
    spatial: PRUnifiedSpatialSpec
    precision: PRMaterialPrecisionSpec
    backend: str
    peak_intensity_reference: float
    dark_intensity: float
    uniform_background: float
    intensity_id: str = TRANSPORT_INTENSITY_ID
    normalization_id: str = NORMALIZATION_ID
    units: str = NORMALIZED_UNITS

    def validate_structure(self):
        _specs(self.spatial, self.precision)
        if (self.intensity_id != TRANSPORT_INTENSITY_ID or self.normalization_id not in (LEGACY_NORMALIZATION, INTEGRAL_NORMALIZATION)
                or self.units != NORMALIZED_UNITS):
            raise ValueError("unsupported transport intensity/normalization identity or units")
        _real(self.peak_intensity_reference, "transport reference", positive=True)
        _real(self.dark_intensity, "dark intensity", nonnegative=True)
        _real(self.uniform_background, "uniform background", nonnegative=True)
        _real(self.total_background, "total background", nonnegative=True)
        _array_metadata(self.values, shape=self.spatial.field_shape,
            dtype=self.precision.state_dtype, backend=self.backend, name="transport intensity")

    @property
    def total_background(self):
        return _real(self.dark_intensity, "dark intensity", nonnegative=True) + _real(
            self.uniform_background, "uniform background", nonnegative=True)


@dataclass(frozen=True)
class PRMaterialDiagnostics:
    """Named host scalar observations, not a physical-acceptance certificate.

    Values/limits are supplied by a future evaluator. M1 never compares them or
    derives a passed flag. 'reported' only means the caller supplied metadata.
    """

    observations: tuple[tuple[str, float], ...] = ()
    limits: tuple[tuple[str, float], ...] = ()
    status: str = "not_evaluated"
    diagnostics_id: str = DIAGNOSTICS_ID
    units: str = NORMALIZED_UNITS

    def __post_init__(self):
        for field in ("observations", "limits"):
            pairs = tuple((name, _real(value, field)) for name, value in getattr(self, field))
            object.__setattr__(self, field, pairs)
        self.validate_structure()

    def validate_structure(self):
        if self.diagnostics_id != DIAGNOSTICS_ID or self.units != NORMALIZED_UNITS:
            raise ValueError("unsupported diagnostics identity or units")
        if self.status not in ("not_evaluated", "reported"):
            raise ValueError("diagnostic metadata does not certify acceptance")
        if self.status == "not_evaluated" and (self.observations or self.limits):
            raise ValueError("not_evaluated diagnostics must be empty")
        for pairs in (self.observations, self.limits):
            names = [name for name, _ in pairs]
            if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
                raise ValueError("diagnostic scalar names must be nonempty and unique")
        if any(value < 0 for _, value in self.limits):
            raise ValueError("diagnostic limits must be nonnegative")
