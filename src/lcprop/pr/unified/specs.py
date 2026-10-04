"""Versioned, immutable metadata for the future unified material core.

These contracts do not select a workflow, instantiate a backend, or solve physics.
"""
from dataclasses import dataclass
import math
from numbers import Integral, Real


SPATIAL_ID = "periodic_compatible_fitted_flux_v1"
MATERIAL_ID = "pr_unified_nonlinear_hopping_v1"
NORMALIZED_UNITS = "pr_normalized_material_units_v1"
UNBIASED = "periodic_unbiased_zero_flux_v1"
FIXED_FIELD = "periodic_fixed_mean_field_v1"
PRESCRIBED_CURRENT = "periodic_prescribed_mean_current_v1"
OPEN_TRANSVERSE = "periodic_fixed_x_field_zero_y_current_v1"
A7_CURRENT = "periodic_a7_reservoir_current_1d_v1"
MIXED_PRECISION = "state32_linear64_bernoulli64_v1"
POSITIVE_PRECISION = "state32_carrier64_coeff64_linear64_bernoulli64_v3"
DOUBLE_PRECISION = "state64_linear64_bernoulli64_v1"
TRANSPORT_INTENSITY_ID = "pr_local_total_transport_intensity_v1"
NORMALIZATION_ID = "pr_channel_peak_reference_v1"
STATE_ID = "pr_unified_q_psi_b_state_v1"
DIAGNOSTICS_ID = "pr_unified_material_diagnostics_v1"


def _real(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real host scalar")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f"invalid {name}")
    return value


def _count(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class PRUnifiedSpatialSpec:
    """Active axes precede batch axes; only independent reduced y columns batch.

    Reduced: (Nx,) or (Nx, Ny), with a separate gauge and b for each y column.
    Full x-y: (Nx, Ny), with one connected-plane gauge and harmonic vector.
    """

    active_shape: tuple[int, ...]
    normalized_lengths: tuple[float, ...]
    active_axes: tuple[str, ...] = ("x",)
    batch_shape: tuple[int, ...] = ()
    batch_axes: tuple[str, ...] = ()
    spatial_id: str = SPATIAL_ID
    boundary: str = "periodic"
    transport_profile: str = "isotropic"
    units: str = NORMALIZED_UNITS

    def __post_init__(self):
        for name in ("active_shape", "normalized_lengths", "active_axes", "batch_shape", "batch_axes"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        self.validate()

    def validate(self):
        if (self.spatial_id != SPATIAL_ID or self.boundary != "periodic"
                or self.transport_profile != "isotropic" or self.units != NORMALIZED_UNITS):
            raise ValueError("unsupported spatial identity, boundary, profile or units")
        if self.active_axes not in (("x",), ("x", "y")):
            raise ValueError("active axes must be x or x-y")
        if len(self.active_shape) != self.dimension or len(self.normalized_lengths) != self.dimension:
            raise ValueError("active shape/length dimension mismatch")
        for n in self.active_shape:
            _count(n, "active size", 2)
        for length in self.normalized_lengths:
            _real(length, "normalized length", positive=True)
        if self.batch_axes:
            if self.active_axes != ("x",) or self.batch_axes != ("y",) or len(self.batch_shape) != 1:
                raise ValueError("only reduced independent y-column batching is supported")
            _count(self.batch_shape[0], "batch size")
        elif self.batch_shape:
            raise ValueError("batch shape requires explicit batch axes")

    @property
    def dimension(self):
        return len(self.active_axes)

    @property
    def field_shape(self):
        return self.active_shape + self.batch_shape

    @property
    def harmonic_shape(self):
        return self.batch_shape + (self.dimension,)

    @property
    def reduction_axes(self):
        """Axes for future gauge/neutrality reductions; never batch axes."""
        return tuple(range(self.dimension))


@dataclass(frozen=True)
class PRElectricalClosureSpec:
    """Named Static experiments; targets are normalized and shared per batch.

    Fixed-field target is b; prescribed-current target is mean(J).
    Open-transverse target is (b_x, 0), where zero constrains mean(J_y), not b_y.
    A7 stores reservoir E_app and I_b as well as target J_ext=E_app*I_b.
    No arbitrary U/V matrices, circuit loads, or dynamic closure are supported.
    """

    identity: str
    dimension: int
    target: tuple[float, ...]
    reservoir_field: float | None = None
    background_intensity: float | None = None
    units: str = NORMALIZED_UNITS

    def __post_init__(self):
        object.__setattr__(self, "target", tuple(_real(v, "closure target") for v in self.target))
        self.validate()

    def validate(self):
        if _count(self.dimension, "closure dimension") not in (1, 2):
            raise ValueError("closure dimension must be 1 or 2")
        if self.units != NORMALIZED_UNITS or len(self.target) != self.dimension:
            raise ValueError("closure units/target dimension mismatch")
        if self.identity not in (UNBIASED, FIXED_FIELD, PRESCRIBED_CURRENT, OPEN_TRANSVERSE, A7_CURRENT):
            raise ValueError("unknown or unsupported electrical closure")
        if self.identity == UNBIASED and any(v != 0 for v in self.target):
            raise ValueError("unbiased zero-flux requires a zero target")
        if self.identity == OPEN_TRANSVERSE and (self.dimension != 2 or self.target[1] != 0):
            raise ValueError("open transverse requires (prescribed b_x, zero mean J_y)")
        if self.identity == A7_CURRENT:
            field = _real(self.reservoir_field, "A7 reservoir field")
            background = _real(self.background_intensity, "A7 total background", nonnegative=True)
            current = _real(field * background, "A7 derived current")
            if self.dimension != 1 or self.target != (current,):
                raise ValueError("A7 requires 1D current target E_app * I_b")
        elif self.reservoir_field is not None or self.background_intensity is not None:
            raise ValueError("reservoir metadata belongs only to A7")

    def validate_spatial(self, spatial: PRUnifiedSpatialSpec):
        self.validate()
        spatial.validate()
        if self.dimension != spatial.dimension:
            raise ValueError("closure and active transport dimensions disagree")

    @classmethod
    def a7(cls, reservoir_field, background_intensity):
        field = _real(reservoir_field, "A7 reservoir field")
        background = _real(background_intensity, "A7 total background", nonnegative=True)
        return cls(A7_CURRENT, 1, (field * background,), field, background)


@dataclass(frozen=True)
class PRMaterialPrecisionSpec:
    """Explicit state/solve/Bernoulli policy; no backend resolution."""

    identity: str = DOUBLE_PRECISION
    state_dtype: str = "float64"
    linear_dtype: str = "float64"
    bernoulli_dtype: str = "float64"
    output_dtype: str = "float64"

    def __post_init__(self):
        self.validate()

    def validate(self):
        expected = {MIXED_PRECISION: "float32", POSITIVE_PRECISION: "float32", DOUBLE_PRECISION: "float64"}.get(self.identity)
        if (expected is None or self.state_dtype != expected or self.output_dtype != expected
                or self.linear_dtype != "float64" or self.bernoulli_dtype != "float64"):
            raise ValueError("unknown or contradictory material precision policy")

    @property
    def carrier_dtype(self):
        """Derived scientific and quantitative product dtype; never a cast hint."""
        return "float64" if self.identity == POSITIVE_PRECISION else self.state_dtype

    @property
    def coefficient_dtype(self):
        return self.carrier_dtype
