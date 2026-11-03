"""Physical illumination authority shared by PR Static and TD.

Irradiances are W/cm², powers W, aperture dimensions µm.  No Gaussian
assumptions, physical-time calibration, or host scientific-array conversion.
"""
from dataclasses import asdict, dataclass, replace
import math
from numbers import Real

LEGACY_NORMALIZATION = "pr_channel_peak_reference_v1"
INTEGRAL_NORMALIZATION = "pr_integral_total_illumination_mean_irradiance_v1"


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} requires an explicit numeric value")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class PhysicalIlluminationReference:
    """Fixed launch/segment reference; derived fractions are not inputs."""
    optical_scale_W_cm2: float
    area_cm2: float
    optical_power_W: float
    dark_irradiance_W_cm2: float
    uniform_irradiance_W_cm2: float
    reference_power_W: float
    reference_irradiance_W_cm2: float
    identity: str = INTEGRAL_NORMALIZATION

    def __post_init__(self):
        if self.identity != INTEGRAL_NORMALIZATION:
            raise ValueError("unknown physical illumination identity")
        for key in self.__dataclass_fields__:
            if key != "identity": nonnegative(getattr(self, key), key)
        if self.area_cm2 <= 0 or self.reference_irradiance_W_cm2 <= 0:
            raise ValueError("positive area/reference required")
        if self.reference_power_W != (self.optical_power_W + self.area_cm2 * (
                self.dark_irradiance_W_cm2 + self.uniform_irradiance_W_cm2)):
            raise ValueError("inconsistent physical reference power")
        if self.reference_irradiance_W_cm2 != self.reference_power_W / self.area_cm2:
            raise ValueError("inconsistent physical reference irradiance")

    @property
    def dark_fraction(self):
        return self.dark_irradiance_W_cm2 / self.reference_irradiance_W_cm2

    @property
    def uniform_fraction(self):
        return self.uniform_irradiance_W_cm2 / self.reference_irradiance_W_cm2

    @property
    def background_fraction(self):
        return self.dark_fraction + self.uniform_fraction

    def driving_intensity(self, fields, *, coherence_groups, xp):
        from lcprop.optics.splitstep import total_intensity
        physical = total_intensity(fields, coherence_groups=coherence_groups, xp=xp)
        physical = physical * self.optical_scale_W_cm2
        return (physical + self.dark_irradiance_W_cm2
                + self.uniform_irradiance_W_cm2) / self.reference_irradiance_W_cm2

    def metadata(self):
        return dict(asdict(self), dark_fraction=self.dark_fraction,
                    uniform_fraction=self.uniform_fraction,
                    quadrature="periodic_cell_centered_rectangle_v1",
                    time_units="characteristic_time")

    def __float__(self):
        """Numerical-field scale for existing observational inverse-source products."""
        if self.optical_scale_W_cm2 <= 0:
            raise ValueError("zero optical scale has no inverse-source representation")
        return self.reference_irradiance_W_cm2 / self.optical_scale_W_cm2


def physical_reference(fields, *, optical_scale_W_cm2, dx_um, dy_um,
                       dark_irradiance_W_cm2, uniform_irradiance_W_cm2,
                       coherence_groups, xp):
    """Integrate actual coherent irradiance, then add physical uniform powers.

    For the existing power-fraction launch, optical_scale_W_cm2 is
    total_power_mW * 1e5 (mW/µm² -> W/cm²). Restoring this physical scale
    precedes integration and prevents loss of absolute beam-power semantics.
    Arbitrary prepared arrays require their explicit physical scale.
    """
    from lcprop.optics.splitstep import total_intensity
    scale = nonnegative(optical_scale_W_cm2, "optical field scale")
    dark = nonnegative(dark_irradiance_W_cm2, "dark irradiance")
    uniform = nonnegative(uniform_irradiance_W_cm2, "uniform irradiance")
    dx = nonnegative(dx_um, "dx"); dy = nonnegative(dy_um, "dy")
    if dx == 0 or dy == 0 or fields.ndim != 3 or min(fields.shape) < 1:
        raise ValueError("nonempty channel plane and positive spacings required")
    irradiance = total_intensity(fields, coherence_groups=coherence_groups, xp=xp) * scale
    if not bool(xp.all(xp.isfinite(irradiance)).item()):
        raise ValueError("nonfinite physical optical irradiance")
    pixel_cm2 = dx * dy * 1e-8
    area = pixel_cm2 * fields.shape[-2] * fields.shape[-1]
    power = float(xp.sum(irradiance, dtype=xp.float64).item()) * pixel_cm2
    reference_power = power + area * (dark + uniform)
    reference = reference_power / area
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("total illumination reference must be finite and positive")
    return PhysicalIlluminationReference(scale, area, power, dark, uniform,
                                         reference_power, reference)


def resolve_material_illumination(material, fields, *, grid, optical_scale_W_cm2,
                                  coherence_groups, xp):
    """Resolve physical inputs once; return unchanged-equation runtime coefficients.

    The returned material is execution-local only. Persist the original physical
    input spec and reference metadata, never this derived legacy-shaped adapter.
    """
    material.validate()
    if material.normalization_identity == LEGACY_NORMALIZATION:
        from .source import channel_peak_intensity_reference
        return channel_peak_intensity_reference(fields, xp=xp), material
    ref = physical_reference(fields, optical_scale_W_cm2=optical_scale_W_cm2,
        dx_um=grid.dx_um, dy_um=grid.dy_um, coherence_groups=coherence_groups, xp=xp,
        dark_irradiance_W_cm2=material.dark_irradiance_W_cm2,
        uniform_irradiance_W_cm2=material.uniform_irradiance_W_cm2)
    runtime = replace(material, normalization_identity=LEGACY_NORMALIZATION,
        dark_irradiance_W_cm2=None, uniform_irradiance_W_cm2=None,
        dark_intensity=ref.dark_fraction, uniform_background_intensity=ref.uniform_fraction)
    return ref, runtime


def result_source_inverse(result):
    """Return numerical optical-field reference/background for presentation only."""
    diagnostics = getattr(result, "diagnostics", {})
    profile = getattr(result, "resolved_profile", {})
    material = profile.get("material")
    checkpoint = getattr(result, "checkpoint", None)
    if checkpoint is not None:
        material = asdict(checkpoint.request.material)
    identity = diagnostics.get("normalization_identity", LEGACY_NORMALIZATION)
    if material is not None:
        identity = material.get("normalization_identity", LEGACY_NORMALIZATION)
    if identity not in (LEGACY_NORMALIZATION, INTEGRAL_NORMALIZATION):
        raise ValueError("unknown result normalization identity")
    record = diagnostics.get("source_normalization")
    if record is None:
        record = profile.get("source_normalization")
    if record is None:
        if identity == INTEGRAL_NORMALIZATION:
            raise ValueError("physical TD result requires illumination provenance")
        return None
    if identity != INTEGRAL_NORMALIZATION:
        raise ValueError("physical reference conflicts with legacy result identity")
    ref = reference_from_metadata(record)
    if material is not None and (material.get("dark_irradiance_W_cm2") != ref.dark_irradiance_W_cm2
            or material.get("uniform_irradiance_W_cm2") != ref.uniform_irradiance_W_cm2):
        raise ValueError("physical TD reference conflicts with material input")
    # Exact zero physical optical power has identically zero optical products.
    return (0.0 if ref.optical_scale_W_cm2 == 0 else float(ref)), ref.background_fraction


def reference_from_metadata(record):
    names = PhysicalIlluminationReference.__dataclass_fields__
    if not isinstance(record, dict) or set(record) != set(names) | {
            "dark_fraction", "uniform_fraction", "quadrature", "time_units"}:
        raise ValueError("complete physical illumination provenance required")
    ref = PhysicalIlluminationReference(**{key: record[key] for key in names})
    if ref.metadata() != record:
        raise ValueError("contradictory derived illumination metadata")
    return ref
