"""PR-owned coherent image-amplification benchmarks and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from time import perf_counter
from typing import Any, Literal

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken, RunProgress
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch, channel_power_integrals
from lcprop.optics.launch_configuration import LaunchConfiguration
from lcprop.optics.screens import (
    ChannelLaunchElements,
    EVEN_SQUARE_NEAREST_TRANSPARENT_V1,
    IntensityRasterScreen,
    ScreenPlacement,
    apply_passive_field_transmittance,
    intensity_transmission_to_field_transmittance,
    prepare_intensity_raster_transmission,
)
from lcprop.optics.splitstep import linear_kernel
from lcprop.pr.coupling import analytic_plane_wave_gain_length
from lcprop.pr.geometry import crossing_beam_channels
from lcprop.pr.image_sources import (
    PR_IMAGE_PREPROCESSING_POLICY_V1,
    PRImageSource,
)
from lcprop.pr.specs import (
    PRMaterialSpec,
    PR_MATERIAL_ID,
    PRRunRequest,
    PRRunResult,
    PRSolverOptions,
    PR_SEMI_IMPLICIT_INTEGRATOR,
    PR_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.workflow import run_pr_timedependent
from lcprop.pr.static_streaming import (
    PRStreamingStaticOptions,
    PRStreamingStaticRequest,
    PRStreamingStaticResult,
    run_pr_static_streaming,
)


PR_IMAGE_AMPLIFICATION_WORKFLOW = "pr_image_amplification"

PR_IMAGE_ANALYSIS_STAGES = (
    "carrier_isolation",
    "output_back_propagation",
    "zero_response_propagation",
    "reference_back_propagation",
    "metric_construction",
    "product_augmentation",
)


@dataclass(frozen=True)
class PRImageAmplificationBaseCapability:
    """Small, explicit adapter contract for an ordinary PR operation."""

    workflow_id: str
    request_type: type
    result_type: type
    launch_adapter: Literal["declarative_elements", "prepared_field"]
    validation_status: Literal[
        "compatible_and_validated", "compatible_validation_pending"
    ]


@dataclass(frozen=True)
class PRImageAmplificationExperimentRequest:
    """Image analysis composed over one selected ordinary PR request.

    ``base_request`` remains authoritative for material, solver, grid, and
    backend settings.  The experiment owns only launch composition and channel
    roles, avoiding a second copy of algorithm-specific solver controls.
    """

    base_workflow_id: str
    base_request: Any
    launch_configuration: LaunchConfiguration
    pump_channel_index: int
    signal_channel_index: int

    @property
    def source(self):
        return _image_screen_and_channels(self)[0].source

    @property
    def grid(self):
        return self.base_request.grid

    @property
    def material(self):
        return self.base_request.material

    @property
    def backend(self):
        return self.base_request.backend

    @property
    def incident_signal_to_pump_power_ratio(self) -> float:
        channels = self.launch_configuration.beams.channels
        return float(channels[self.signal_channel_index].power_mW) / float(
            channels[self.pump_channel_index].power_mW
        )

    def validate(self) -> None:
        prepare_image_amplification_base_request(self)


@dataclass(frozen=True)
class PRImageAmplificationCompositeResult:
    """Ordinary base result plus independently classified image analysis."""

    request: PRImageAmplificationExperimentRequest
    base_runner_result: Any
    analysis_result: PRImageAmplificationResult | None
    analysis_status: Literal["completed", "cancelled", "not_run", "failed"]
    analysis_message: str

    @property
    def run_result(self):
        return self.base_runner_result.result

    @property
    def base_status(self) -> str:
        return str(getattr(self.base_runner_result.result, "status", "failed"))

    @property
    def status(self) -> str:
        if self.base_status not in ("completed", "converged"):
            return self.base_status
        if self.analysis_status != "completed":
            return self.analysis_status
        return self.base_status


@dataclass(frozen=True)
class PRImageAmplificationSpec:
    """Finite-image two-beam benchmark parameters.

    The default is a scaled CPU case. :func:`paper_figure4_spec` returns the
    research-scale geometry reported in the source paper.
    """

    Nx: int = 64
    Ny: int = 32
    x_aperture_um: float = 64.0
    y_aperture_um: float = 32.0
    interaction_length_um: float = 100.0
    dz_um: float = 10.0
    wavelength_um: float = 0.633
    refractive_index: float = 2.4
    positive_mode_index: int = 2
    beam_waist_um: float = 12.0
    image_size_factor: float = 1.0
    input_peak_ratio: float = 1e-3
    saturated_small_signal_gain: float | None = 10.0
    signal_gain_sign: int = 1
    dark_intensity: float = 0.05
    applied_field: float = 0.0
    characteristic_wavenumber_per_um: float | None = 0.5
    relative_permittivity: float = 2500.0
    mobile_charge_density_m3: float = 6.4e22
    temperature_K: float = 293.0
    gain_length_product_override: float | None = None
    tukey_alpha: float = 0.0
    volume_noise_epsilon: float = 0.0
    volume_noise_correlation_um: float = 0.4
    volume_noise_seed: int | None = None
    volume_noise_seeds: tuple[int, ...] | None = None
    Nt: int = 500
    dt_normalized: float = 0.05
    invert_image: bool = True
    coherence_group: str = "pr-image-amplification"


@dataclass(frozen=True)
class PRImageLaunchSpec:
    """Image-bearing launch controls with pre-element incident powers."""

    wavelength_um: float = 0.633
    positive_mode_index: int = 2
    beam_waist_x_um: float = 12.0
    beam_waist_y_um: float = 12.0
    image_physical_size_um: float = 12.0
    pump_incident_power_mW: float = 1.0
    signal_incident_power_mW: float = 1e-3
    invert_image: bool = False
    require_full_footprint: bool = False
    coherence_group: str = "pr-image-amplification"
    preprocessing_policy: str = PR_IMAGE_PREPROCESSING_POLICY_V1

    def validate(self, *, grid: GridSpec, refractive_index: float) -> None:
        if self.preprocessing_policy != PR_IMAGE_PREPROCESSING_POLICY_V1:
            raise ValueError("unsupported PR image preprocessing policy")
        for name in (
            "wavelength_um",
            "beam_waist_x_um",
            "beam_waist_y_um",
            "image_physical_size_um",
            "pump_incident_power_mW",
            "signal_incident_power_mW",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.coherence_group, str) or not self.coherence_group:
            raise ValueError("coherence_group must be non-empty")
        mode = int(self.positive_mode_index)
        if mode <= 0 or 4 * mode >= int(grid.Nx):
            raise ValueError(
                "positive_mode_index must place the two-beam grating below Nyquist"
            )
        kx = 2.0 * math.pi * mode / float(grid.x_aperture_um)
        k_medium = (
            2.0 * math.pi * float(refractive_index) / float(self.wavelength_um)
        )
        if kx >= k_medium:
            raise ValueError("selected carrier is not a propagating mode")

    @property
    def incident_total_power_mW(self) -> float:
        return float(self.pump_incident_power_mW) + float(
            self.signal_incident_power_mW
        )

    @property
    def incident_signal_to_pump_power_ratio(self) -> float:
        return float(self.signal_incident_power_mW) / float(
            self.pump_incident_power_mW
        )


@dataclass(frozen=True)
class PRImageAmplificationRunRequest:
    """Portable-in-design local request for the composite image workflow."""

    grid: GridSpec
    material: PRMaterialSpec
    solver: PRSolverOptions
    backend: BackendSpec
    source: PRImageSource
    launch: PRImageLaunchSpec

    def validate(self) -> None:
        self.grid.validate()
        self.material.validate()
        self.solver.validate()
        self.backend.validate()
        if not isinstance(self.source, PRImageSource):
            raise TypeError("source must be a PRImageSource")
        self.launch.validate(
            grid=self.grid,
            refractive_index=float(self.material.refractive_index),
        )


@dataclass(frozen=True)
class PRBeamPanelImageAmplificationRunRequest:
    """Physical image experiment composed from the shared BeamPanel launch.

    Channel indices refer to the canonical enabled-channel ordering in the
    immutable :class:`LaunchConfiguration`; roles are deliberately not stored
    on the generic beam model.
    """

    grid: GridSpec
    material: PRMaterialSpec
    solver: PRSolverOptions
    backend: BackendSpec
    launch_configuration: LaunchConfiguration
    pump_channel_index: int
    signal_channel_index: int

    def validate(self) -> None:
        self.grid.validate()
        self.material.validate()
        self.solver.validate()
        self.backend.validate()
        if not isinstance(self.launch_configuration, LaunchConfiguration):
            raise TypeError("launch_configuration must be a LaunchConfiguration")
        beams = self.launch_configuration.beams
        if len(beams.channels) != 2:
            raise ValueError(
                "Image Amplification requires exactly two enabled beam channels"
            )
        for name in ("pump_channel_index", "signal_channel_index"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value < len(beams.channels):
                raise ValueError(
                    f"{name} must identify a canonical enabled beam channel"
                )
        if self.pump_channel_index == self.signal_channel_index:
            raise ValueError("pump and signal channels must be distinct")
        pump = beams.channels[self.pump_channel_index]
        signal = beams.channels[self.signal_channel_index]
        if float(pump.power_mW) <= 0.0 or float(signal.power_mW) <= 0.0:
            raise ValueError("pump and signal incident powers must be positive")
        groups = beams.coherence_groups
        if groups[self.pump_channel_index] != groups[self.signal_channel_index]:
            raise ValueError(
                "pump and signal channels must share one coherence group"
            )
        if not math.isclose(
            float(pump.wavelength_um),
            float(signal.wavelength_um),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("pump and signal wavelengths must match")
        if not math.isclose(
            float(pump.tilt_y_rad_per_um),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ) or not math.isclose(
            float(signal.tilt_y_rad_per_um),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                "Image Amplification C1 requires both carriers in the x-z plane"
            )
        if not math.isclose(
            float(pump.tilt_x_rad_per_um),
            -float(signal.tilt_x_rad_per_um),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Image Amplification C1 requires symmetric pump/signal x carriers"
            )
        assignments = {
            assignment.channel_index: assignment.elements
            for assignment in self.launch_configuration.channel_elements
        }
        if assignments.get(self.pump_channel_index, ()):
            raise ValueError("the selected pump channel must not carry an image screen")
        signal_elements = assignments.get(self.signal_channel_index, ())
        if len(signal_elements) != 1 or not isinstance(
            signal_elements[0], IntensityRasterScreen
        ):
            raise ValueError(
                "the selected signal channel must carry exactly one "
                "IntensityRasterScreen"
            )

    @property
    def source(self):
        assignments = {
            assignment.channel_index: assignment.elements
            for assignment in self.launch_configuration.channel_elements
        }
        return assignments[self.signal_channel_index][0].source

    @property
    def incident_signal_to_pump_power_ratio(self) -> float:
        channels = self.launch_configuration.beams.channels
        return float(channels[self.signal_channel_index].power_mW) / float(
            channels[self.pump_channel_index].power_mW
        )


@dataclass(frozen=True)
class PRImageAmplificationResult:
    """Optical, gain, and image-fidelity evidence from one benchmark."""

    request: Any
    run_result: Any
    image_transmission: np.ndarray
    signal_carrier_mask: np.ndarray
    input_signal_field: np.ndarray
    output_signal_field: np.ndarray
    backpropagated_signal_field: np.ndarray
    zero_response_backpropagated_signal_field: np.ndarray
    transverse_phase_gradients_rad_per_um: tuple[float, float]
    normalized_grating_wavenumber: float
    gain_length_product: float
    analytic_gamma_p_L: float
    analytic_absolute_signal_gain: float
    measured_absolute_signal_gain: float
    measured_gain_reference_signal_power_normalized: float
    output_isolated_signal_power_normalized: float
    measured_gain_reference_signal_power_mW: float
    output_isolated_signal_power_mW: float
    image_intensity_correlation: float
    zero_response_image_intensity_correlation: float
    normalized_image_rmse: float
    normalized_power_relative_drift: float
    incident_channel_powers_mW: tuple[float, float]
    post_element_channel_powers_mW: tuple[float, float]
    incident_total_power_mW: float
    post_element_total_power_mW: float
    signal_throughput_fraction: float
    transparency_policy: str
    pr_workflow_runtime_s: float
    reconstruction_optical_runtime_s: float
    reconstruction_runtime_s: float
    runtime_s: float
    image_request: (
        PRImageAmplificationRunRequest
        | PRBeamPanelImageAmplificationRunRequest
        | PRImageAmplificationExperimentRequest
        | None
    ) = None

    @property
    def status(self) -> str:
        return self.run_result.status


@dataclass(frozen=True)
class PRStreamingImageAmplificationResult:
    """Figure-style observables from a bounded-memory static PR march."""

    request: PRStreamingStaticRequest
    run_result: PRStreamingStaticResult
    image_transmission: np.ndarray
    input_signal_field: np.ndarray
    output_signal_field: np.ndarray
    backpropagated_signal_field: np.ndarray
    output_total_intensity: np.ndarray
    normalized_grating_wavenumber: float
    gain_length_product: float
    analytic_absolute_signal_gain: float
    measured_absolute_signal_gain: float
    image_intensity_correlation: float
    normalized_image_rmse: float
    normalized_power_relative_drift: float


def image_amplification_base_capabilities() -> tuple[
    PRImageAmplificationBaseCapability, ...
]:
    """Return the bounded capability adapters for current PR operations.

    Compatibility means the operation can supply the complex launch/output
    fields and physical metadata required by the common postprocessor.  It is
    deliberately distinct from scientific validation for this experiment.
    """

    from lcprop.pr.static_workflow import (
        PRStaticRunRequest,
        PRStaticRunResult,
        PR_STATIC_WORKFLOW,
    )
    from lcprop.pr.transverse.specs import (
        PRTransverseRunRequest,
        PRTransverseRunResult,
        PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
    )
    from lcprop.pr.transverse.static_workflow import (
        PRTransverseStaticRunRequest,
        PRTransverseStaticRunResult,
        PR_TRANSVERSE_STATIC_WORKFLOW,
    )

    return (
        PRImageAmplificationBaseCapability(
            PR_TIMEDEPENDENT_WORKFLOW,
            PRRunRequest,
            PRRunResult,
            "declarative_elements",
            "compatible_and_validated",
        ),
        PRImageAmplificationBaseCapability(
            PR_STATIC_WORKFLOW,
            PRStaticRunRequest,
            PRStaticRunResult,
            "declarative_elements",
            "compatible_validation_pending",
        ),
        PRImageAmplificationBaseCapability(
            PR_TRANSVERSE_STATIC_WORKFLOW,
            PRTransverseStaticRunRequest,
            PRTransverseStaticRunResult,
            "declarative_elements",
            "compatible_and_validated",
        ),
        PRImageAmplificationBaseCapability(
            PR_TRANSVERSE_TIMEDEPENDENT_WORKFLOW,
            PRTransverseRunRequest,
            PRTransverseRunResult,
            "prepared_field",
            "compatible_validation_pending",
        ),
    )


def _base_capability(workflow_id: str) -> PRImageAmplificationBaseCapability:
    for capability in image_amplification_base_capabilities():
        if capability.workflow_id == workflow_id:
            return capability
    raise ValueError(
        f"PR workflow {workflow_id!r} does not expose the image-amplification "
        "optical-result capability"
    )


def _image_screen_and_channels(request):
    launch = request.launch_configuration
    pump_index = int(request.pump_channel_index)
    signal_index = int(request.signal_channel_index)
    assignments = {
        assignment.channel_index: assignment.elements
        for assignment in launch.channel_elements
    }
    signal_elements = assignments.get(signal_index, ())
    if len(signal_elements) != 1 or not isinstance(
        signal_elements[0], IntensityRasterScreen
    ):
        raise ValueError(
            "the selected signal channel must carry exactly one "
            "IntensityRasterScreen"
        )
    return (
        signal_elements[0],
        launch.beams.channels[pump_index],
        launch.beams.channels[signal_index],
    )


def _validate_composite_launch(request) -> None:
    launch = request.launch_configuration
    if not isinstance(launch, LaunchConfiguration):
        raise TypeError("launch_configuration must be a LaunchConfiguration")
    beams = launch.beams
    if len(beams.channels) != 2:
        raise ValueError(
            "Image Amplification requires exactly two enabled beam channels"
        )
    for name in ("pump_channel_index", "signal_channel_index"):
        value = getattr(request, name)
        if type(value) is not int or not 0 <= value < len(beams.channels):
            raise ValueError(f"{name} must identify a canonical enabled beam channel")
    if request.pump_channel_index == request.signal_channel_index:
        raise ValueError("pump and signal channels must be distinct")
    screen, pump, signal = _image_screen_and_channels(request)
    if any(float(channel.power_mW) <= 0.0 for channel in (pump, signal)):
        raise ValueError("pump and signal incident powers must be positive")
    groups = beams.coherence_groups
    if groups[request.pump_channel_index] != groups[request.signal_channel_index]:
        raise ValueError("pump and signal channels must share one coherence group")
    if not math.isclose(
        float(pump.wavelength_um),
        float(signal.wavelength_um),
        abs_tol=1e-15,
    ):
        raise ValueError("pump and signal wavelengths must match")
    if not math.isclose(
        float(pump.tilt_y_rad_per_um), 0.0, abs_tol=1e-15
    ) or not math.isclose(
        float(signal.tilt_y_rad_per_um), 0.0, abs_tol=1e-15
    ):
        raise ValueError("Image Amplification requires both carriers in the x-z plane")
    if not math.isclose(
        float(pump.tilt_x_rad_per_um),
        -float(signal.tilt_x_rad_per_um),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Image Amplification requires symmetric pump/signal x carriers"
        )
    assignments = {
        assignment.channel_index: assignment.elements
        for assignment in launch.channel_elements
    }
    if assignments.get(request.pump_channel_index, ()):
        raise ValueError("the selected pump channel must not carry an image screen")
    if screen.preprocessing_policy != EVEN_SQUARE_NEAREST_TRANSPARENT_V1:
        raise ValueError("unsupported image-screen preprocessing policy")


def image_amplification_experiment_request(
    request: PRBeamPanelImageAmplificationRunRequest,
    *,
    base_workflow_id: str = PR_TIMEDEPENDENT_WORKFLOW,
    base_request: Any | None = None,
) -> PRImageAmplificationExperimentRequest:
    """Compose the C1 BeamPanel request over one ordinary PR operation."""

    request.validate()
    if base_request is None:
        if base_workflow_id != PR_TIMEDEPENDENT_WORKFLOW:
            raise ValueError("a base_request is required for non-default PR algorithms")
        base_request = PRRunRequest(
            grid=request.grid,
            beams=request.launch_configuration.beams,
            material=request.material,
            solver=request.solver,
            backend=request.backend,
        )
    composite = PRImageAmplificationExperimentRequest(
        base_workflow_id=base_workflow_id,
        base_request=base_request,
        launch_configuration=request.launch_configuration,
        pump_channel_index=request.pump_channel_index,
        signal_channel_index=request.signal_channel_index,
    )
    capability = _base_capability(base_workflow_id)
    if not isinstance(base_request, capability.request_type):
        raise TypeError(
            f"{base_workflow_id} requires {capability.request_type.__name__}"
        )
    _validate_composite_launch(composite)
    return composite


def prepare_image_amplification_base_request(
    request: PRImageAmplificationExperimentRequest,
) -> tuple[Any, np.ndarray, float]:
    """Deterministically transform an experiment into an ordinary PR request."""

    if not isinstance(request, PRImageAmplificationExperimentRequest):
        raise TypeError("request must be a PRImageAmplificationExperimentRequest")
    capability = _base_capability(request.base_workflow_id)
    if not isinstance(request.base_request, capability.request_type):
        raise TypeError(
            f"{request.base_workflow_id} requires {capability.request_type.__name__}"
        )
    _validate_composite_launch(request)
    base = request.base_request
    for attribute in ("grid", "material", "backend"):
        if not hasattr(base, attribute):
            raise TypeError(f"base request lacks required {attribute} capability")
    grid = make_grid(base.grid, real_dtype=np.float64)
    screen, pump, signal = _image_screen_and_channels(request)
    transmission = prepare_intensity_raster_transmission(
        screen.source.grayscale,
        grid,
        placement=screen.placement,
        invert=screen.invert,
        preprocessing_policy=screen.preprocessing_policy,
    )
    normalized_grating = (
        float(signal.tilt_x_rad_per_um) - float(pump.tilt_x_rad_per_um)
    ) / float(base.material.characteristic_wavenumber_per_um)
    replacement = {
        "beams": request.launch_configuration.beams,
        "initial_A": None,
    }
    if capability.launch_adapter == "declarative_elements":
        replacement["launch_elements"] = request.launch_configuration.channel_elements
    else:
        prepared = build_launch(
            request.launch_configuration.beams,
            grid,
            complex_dtype=np.complex128,
            launch_elements=request.launch_configuration.channel_elements,
        )
        replacement["initial_A"] = np.asarray(prepared.A0).copy()
    return replace(base, **replacement), transmission, normalized_grating


def paper_absolute_signal_gain(
    *,
    input_ratio: float,
    gamma_p_L: float,
) -> float:
    """Return the paper's finite-ratio absolute signal gain.

    This is Equation (11) in the published paper,
    ``G0=(1+r)*exp(2*gamma_p*L)/(1+r*exp(2*gamma_p*L))``.
    """

    ratio = float(input_ratio)
    coupling = float(gamma_p_L)
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ValueError("input_ratio must be finite and nonnegative")
    if not math.isfinite(coupling):
        raise ValueError("gamma_p_L must be finite")
    exponential = math.exp(2.0 * coupling)
    return (1.0 + ratio) * exponential / (1.0 + ratio * exponential)


def paper_figure4_spec() -> PRImageAmplificationSpec:
    """Return the research-scale parameters reported for paper Figure 4.

    The external incidence angle is 7.56 degrees. On the 4 mm periodic
    aperture it lies essentially on Fourier mode 1024; selecting that exact
    integer mode preserves periodicity and gives eight samples per grating
    period. The source paper's Tukey apodization is intentionally not included
    because LCProp's current PR workflow has periodic transverse boundaries.
    ``Nt`` and ``dt_normalized`` are provisional LCProp material-time controls,
    not parameters reported for the paper's static Figure 4 calculation.
    """

    external_angle = math.radians(7.56)
    mode = round(4000.0 * math.sin(external_angle) / 0.514)
    return PRImageAmplificationSpec(
        Nx=16384,
        Ny=2048,
        x_aperture_um=4000.0,
        y_aperture_um=4000.0,
        interaction_length_um=4350.0,
        dz_um=2.0,
        wavelength_um=0.514,
        refractive_index=2.4,
        positive_mode_index=mode,
        beam_waist_um=3400.0,
        image_size_factor=1.0,
        input_peak_ratio=1e-5,
        saturated_small_signal_gain=4000.0,
        dark_intensity=0.01,
        characteristic_wavenumber_per_um=None,
        Nt=250,
        dt_normalized=0.01,
        invert_image=False,
    )


def paper_figure6_spec() -> PRImageAmplificationSpec:
    """Return the published large-signal Figure 6 benchmark contract.

    Figure 6 uses the Figure 4 geometry with equal incident peak intensities,
    the published Tukey window, and no scattering noise.  The Air Force chart
    is supplied by the caller and inverted before square padding and
    nearest-neighbor placement so that its launch intensity has the published
    dark chart field with bright bars and labels.  This explicit preprocessing
    polarity is part of the benchmark request, not a production default.

    The paper's supplementary ``Figure_3_4_6.json`` contains an unrelated
    no-image, noisy 3 mm by 1 mm saved run and is not used to define this
    caption-derived contract. ``Nt`` and ``dt_normalized`` remain compatibility
    fields of :class:`PRImageAmplificationSpec`; the static streaming workflow
    does not consume them.
    """

    external_angle = math.radians(7.56)
    mode = round(4000.0 * math.sin(external_angle) / 0.514)
    return PRImageAmplificationSpec(
        Nx=16384,
        Ny=2048,
        x_aperture_um=4000.0,
        y_aperture_um=4000.0,
        interaction_length_um=4350.0,
        dz_um=2.0,
        wavelength_um=0.514,
        refractive_index=2.4,
        positive_mode_index=mode,
        beam_waist_um=3400.0,
        image_size_factor=1.0,
        input_peak_ratio=1.0,
        saturated_small_signal_gain=4000.0,
        dark_intensity=0.01,
        applied_field=0.0,
        characteristic_wavenumber_per_um=None,
        relative_permittivity=2500.0,
        mobile_charge_density_m3=6.4e22,
        temperature_K=293.0,
        gain_length_product_override=None,
        tukey_alpha=0.05,
        volume_noise_epsilon=0.0,
        volume_noise_correlation_um=0.4,
        volume_noise_seed=None,
        volume_noise_seeds=None,
        Nt=250,
        dt_normalized=0.01,
        invert_image=True,
    )


def prepare_image_transmission(
    image_intensity,
    grid,
    *,
    center_x_um: float,
    center_y_um: float,
    physical_size_um: float,
    invert: bool = False,
    require_full_footprint: bool = False,
) -> np.ndarray:
    """Place a legacy-compatible real intensity transparency on the grid.

    Input arrays use conventional image indexing ``(row_y, column_x)``. They
    are padded to a square with transparent pixels, rotated into LCProp's
    ``(x, y)`` array convention, nearest-neighbor resized, and centered on the
    selected beam. Outside the image footprint the transparency is one, as in
    PRProp3D. The launch-field amplitude multiplier is the square root of this
    returned intensity transmission.
    """

    size_um = float(physical_size_um)
    if not math.isfinite(size_um) or size_um <= 0.0:
        raise ValueError("physical_size_um must be finite and positive")
    return prepare_intensity_raster_transmission(
        image_intensity,
        grid,
        placement=ScreenPlacement(
            center_x_um=float(center_x_um),
            center_y_um=float(center_y_um),
            width_um=size_um,
            height_um=size_um,
            resampling="nearest",
            outside_intensity_transmission=1.0,
            boundary_policy="reject" if require_full_footprint else "clip",
        ),
        invert=invert,
        preprocessing_policy=EVEN_SQUARE_NEAREST_TRANSPARENT_V1,
    )


def _resolved_characteristic_wavenumber(spec: PRImageAmplificationSpec) -> float:
    override = spec.characteristic_wavenumber_per_um
    if override is not None:
        return float(override)
    return PRMaterialSpec().characteristic_wavenumber_per_um


def _gain_geometry(
    spec: PRImageAmplificationSpec,
) -> tuple[float, float, float, float]:
    mode = int(spec.positive_mode_index)
    if mode <= 0 or 4 * mode >= int(spec.Nx):
        raise ValueError(
            "positive_mode_index must place the two-beam grating below Nyquist"
        )
    kx = 2.0 * math.pi * mode / float(spec.x_aperture_um)
    k_medium = 2.0 * math.pi * float(spec.refractive_index) / float(spec.wavelength_um)
    if kx >= k_medium:
        raise ValueError("selected carrier is not a propagating mode")
    internal_angle = math.asin(kx / k_medium)
    k0 = _resolved_characteristic_wavenumber(spec)
    normalized_grating = -2.0 * kx / k0
    coupling_factor = (
        2.0
        * normalized_grating
        / (math.cos(internal_angle) * (1.0 + normalized_grating**2))
    )
    if int(spec.signal_gain_sign) not in (-1, 1):
        raise ValueError("signal_gain_sign must be +1 or -1")
    override = spec.gain_length_product_override
    if override is None:
        gamma_p_L = (
            int(spec.signal_gain_sign)
            * 0.5
            * math.log(float(spec.saturated_small_signal_gain))
        )
        gain_length_product = gamma_p_L / coupling_factor
    else:
        gain_length_product = float(override)
        if not math.isfinite(gain_length_product):
            raise ValueError("gain_length_product_override must be finite")
    return kx, internal_angle, normalized_grating, gain_length_product


def prepare_image_amplification_workflow_request(
    request: (
        PRImageAmplificationRunRequest | PRBeamPanelImageAmplificationRunRequest
    ),
) -> tuple[PRRunRequest, np.ndarray, float]:
    """Apply the passive image element to an incident-power-normalized launch.

    The incident two-channel stack has unit normalized integral. Applying the
    image may reduce that integral; no post-element renormalization occurs.
    """

    if isinstance(request, PRBeamPanelImageAmplificationRunRequest):
        request.validate()
        grid = make_grid(request.grid, real_dtype=np.float64)
        launch = request.launch_configuration
        prepared = build_launch(
            launch.beams,
            grid,
            complex_dtype=np.complex128,
            launch_elements=launch.channel_elements,
        )
        assignments = {
            assignment.channel_index: assignment.elements
            for assignment in launch.channel_elements
        }
        screen = assignments[request.signal_channel_index][0]
        transmission = prepare_intensity_raster_transmission(
            screen.source.grayscale,
            grid,
            placement=screen.placement,
            invert=screen.invert,
            preprocessing_policy=screen.preprocessing_policy,
        )
        pump = launch.beams.channels[request.pump_channel_index]
        signal = launch.beams.channels[request.signal_channel_index]
        normalized_grating = (
            float(signal.tilt_x_rad_per_um)
            - float(pump.tilt_x_rad_per_um)
        ) / float(request.material.characteristic_wavenumber_per_um)
        workflow_request = PRRunRequest(
            grid=request.grid,
            beams=launch.beams,
            material=request.material,
            solver=request.solver,
            backend=request.backend,
            initial_A=np.asarray(prepared.A0).copy(),
        )
        return workflow_request, transmission, normalized_grating
    if not isinstance(request, PRImageAmplificationRunRequest):
        raise TypeError("unsupported image-amplification request type")
    request.validate()
    launch_spec = request.launch
    grid = make_grid(request.grid, real_dtype=np.float64)
    mode = int(launch_spec.positive_mode_index)
    kx = 2.0 * math.pi * mode / float(request.grid.x_aperture_um)
    k_medium = (
        2.0
        * math.pi
        * float(request.material.refractive_index)
        / float(launch_spec.wavelength_um)
    )
    internal_angle = math.asin(kx / k_medium)
    normalized_grating = (
        -2.0 * kx / float(request.material.characteristic_wavenumber_per_um)
    )
    channels = crossing_beam_channels(
        wavelength_um=float(launch_spec.wavelength_um),
        refractive_index=float(request.material.refractive_index),
        interaction_length_um=float(request.grid.z_length_um),
        polar_angles_rad=(internal_angle, internal_angle),
        azimuths_rad=(0.0, math.pi),
        waist_x_um=float(launch_spec.beam_waist_x_um),
        waist_y_um=float(launch_spec.beam_waist_y_um),
        powers_mW=(
            float(launch_spec.pump_incident_power_mW),
            float(launch_spec.signal_incident_power_mW),
        ),
        coherence_group=launch_spec.coherence_group,
        names=("pump", "image signal"),
    )
    preliminary_beams = BeamStack(channels=channels, coherence="coherent")
    half_size = 0.5 * float(launch_spec.image_physical_size_um)
    if bool(launch_spec.require_full_footprint) and (
        abs(float(channels[1].x0_um)) + half_size
        > 0.5 * float(request.grid.x_aperture_um)
        or abs(float(channels[1].y0_um)) + half_size
        > 0.5 * float(request.grid.y_aperture_um)
    ):
        raise ValueError(
            "image footprint extends outside the simulation aperture; "
            "increase the aperture, reduce image size, or adjust the launch"
        )
    screen = IntensityRasterScreen(
        source=request.source,
        placement=ScreenPlacement(
            center_x_um=float(channels[1].x0_um),
            center_y_um=float(channels[1].y0_um),
            width_um=float(launch_spec.image_physical_size_um),
            height_um=float(launch_spec.image_physical_size_um),
            resampling="nearest",
            outside_intensity_transmission=1.0,
            boundary_policy=(
                "reject" if launch_spec.require_full_footprint else "clip"
            ),
        ),
        invert=bool(launch_spec.invert_image),
        preprocessing_policy=EVEN_SQUARE_NEAREST_TRANSPARENT_V1,
    )
    transmission = prepare_intensity_raster_transmission(
        screen.source.grayscale,
        grid,
        placement=screen.placement,
        invert=screen.invert,
        preprocessing_policy=screen.preprocessing_policy,
    )
    preliminary = build_launch(
        preliminary_beams,
        grid,
        complex_dtype=np.complex128,
        launch_elements=(
            ChannelLaunchElements(channel_index=1, elements=(screen,)),
        ),
    )
    A0 = np.asarray(preliminary.A0).copy()
    beams = BeamStack(channels=channels, coherence="coherent")
    workflow_request = PRRunRequest(
        grid=request.grid,
        beams=beams,
        material=request.material,
        solver=request.solver,
        backend=request.backend,
        initial_A=A0,
    )
    return workflow_request, transmission, normalized_grating


def image_amplification_run_request(
    image_intensity,
    spec: PRImageAmplificationSpec = PRImageAmplificationSpec(),
    *,
    backend: BackendSpec | None = None,
    total_power_mW: float = 1.0,
) -> PRImageAmplificationRunRequest:
    """Translate the established benchmark specification to the composite API."""

    ratio = float(spec.input_peak_ratio)
    saturated_gain = spec.saturated_small_signal_gain
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("input_peak_ratio must be finite and positive")
    if spec.gain_length_product_override is None and (
        saturated_gain is None
        or not math.isfinite(float(saturated_gain))
        or float(saturated_gain) <= 1.0
    ):
        raise ValueError(
            "saturated_small_signal_gain must be greater than one when "
            "gain_length_product_override is not supplied"
        )
    _kx, _angle, _grating, gain_length = _gain_geometry(spec)
    grid = GridSpec(
        Nx=int(spec.Nx),
        Ny=int(spec.Ny),
        x_aperture_um=float(spec.x_aperture_um),
        y_aperture_um=float(spec.y_aperture_um),
        z_length_um=float(spec.interaction_length_um),
        dz_um=float(spec.dz_um),
    )
    material = PRMaterialSpec(
        dark_intensity=float(spec.dark_intensity),
        applied_field=float(spec.applied_field),
        gain_length_product=gain_length,
        refractive_index=float(spec.refractive_index),
        relative_permittivity=float(spec.relative_permittivity),
        mobile_charge_density_m3=float(spec.mobile_charge_density_m3),
        temperature_K=float(spec.temperature_K),
        characteristic_wavenumber_per_um_override=(
            None
            if spec.characteristic_wavenumber_per_um is None
            else float(spec.characteristic_wavenumber_per_um)
        ),
    )
    resolved_backend = backend or BackendSpec(
        backend="numpy", precision="float64", verbose=False
    )
    composite = PRImageAmplificationRunRequest(
        grid=grid,
        material=material,
        solver=PRSolverOptions(
            Nt=int(spec.Nt),
            dt_normalized=float(spec.dt_normalized),
            integrator=PR_SEMI_IMPLICIT_INTEGRATOR,
        ),
        backend=resolved_backend,
        source=PRImageSource.from_array(image_intensity),
        launch=PRImageLaunchSpec(
            wavelength_um=float(spec.wavelength_um),
            positive_mode_index=int(spec.positive_mode_index),
            beam_waist_x_um=float(spec.beam_waist_um),
            beam_waist_y_um=float(spec.beam_waist_um),
            image_physical_size_um=(
                float(spec.image_size_factor) * float(spec.beam_waist_um)
            ),
            pump_incident_power_mW=float(total_power_mW) / (1.0 + ratio),
            signal_incident_power_mW=(
                float(total_power_mW) * ratio / (1.0 + ratio)
            ),
            invert_image=bool(spec.invert_image),
            coherence_group=spec.coherence_group,
        ),
    )
    composite.validate()
    return composite


def make_image_amplification_request(
    image_intensity,
    spec: PRImageAmplificationSpec = PRImageAmplificationSpec(),
) -> tuple[PRRunRequest, np.ndarray, float, float]:
    """Create an image-bearing signal and crossing coherent pump request."""

    composite = image_amplification_run_request(image_intensity, spec)
    request, transmission, normalized_grating = (
        prepare_image_amplification_workflow_request(composite)
    )
    grid = make_grid(request.grid, real_dtype=np.float64)
    A0 = np.asarray(request.initial_A).copy()
    pump_peak = float(np.max(np.abs(A0[0]) ** 2))
    signal_peak = float(np.max(np.abs(A0[1]) ** 2))
    if signal_peak <= 0.0:
        raise ValueError("image transparency blocks the complete signal beam")
    A0[1] *= math.sqrt(float(spec.input_peak_ratio) * pump_peak / signal_peak)
    dxdy = float(grid.dx_um) * float(grid.dy_um)
    A0 /= math.sqrt(float(np.sum(np.abs(A0) ** 2)) * dxdy)
    powers = channel_power_integrals(A0, grid)
    resolved_channels = tuple(
        replace(channel, power_mW=float(power))
        for channel, power in zip(request.beams.channels, powers)
    )
    request = replace(
        request,
        beams=BeamStack(channels=resolved_channels, coherence="coherent"),
        initial_A=A0,
    )
    return (
        request,
        transmission,
        normalized_grating,
        float(composite.material.gain_length_product),
    )


def signal_carrier_mask(
    grid,
    *,
    pump_kx_rad_per_um: float,
    signal_kx_rad_per_um: float,
) -> np.ndarray:
    """Return the PRProp3D nearest-carrier Fourier partition for the signal."""

    kx = 2.0 * math.pi * np.fft.fftfreq(grid.Nx, d=float(grid.dx_um))[:, None]
    ky = 2.0 * math.pi * np.fft.fftfreq(grid.Ny, d=float(grid.dy_um))[None, :]
    signal_distance = (kx - float(signal_kx_rad_per_um)) ** 2 + ky**2
    pump_distance = (kx - float(pump_kx_rad_per_um)) ** 2 + ky**2
    return signal_distance <= pump_distance


def isolate_signal_carrier(field, mask: np.ndarray) -> np.ndarray:
    """Isolate one angular carrier from a coherent two-beam field."""

    supplied = np.asarray(field)
    if supplied.ndim != 2 or supplied.shape != mask.shape:
        raise ValueError("field and mask must have the same two-dimensional shape")
    return np.fft.ifft2(np.fft.fft2(supplied) * mask)


def _linear_propagate(
    field: np.ndarray, grid, *, distance_um: float, request
) -> np.ndarray:
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=float(distance_um),
        wavelength=float(request.beams.channels[0].wavelength_um),
        n_ref=float(request.material.refractive_index),
        xp=np,
    )
    return np.fft.ifft2(np.fft.fft2(field) * kernel)


def _intensity_metrics(
    reference_field: np.ndarray,
    measured_field: np.ndarray,
    roi: np.ndarray,
) -> tuple[float, float]:
    reference = np.abs(reference_field[roi]) ** 2
    measured = np.abs(measured_field[roi]) ** 2
    reference_centered = reference - np.mean(reference)
    measured_centered = measured - np.mean(measured)
    denominator = float(
        np.linalg.norm(reference_centered) * np.linalg.norm(measured_centered)
    )
    if denominator == 0.0:
        raise ValueError("image-fidelity ROI has zero intensity variance")
    correlation = float(
        np.vdot(reference_centered, measured_centered).real / denominator
    )
    measured_sum = float(np.sum(measured))
    if measured_sum <= 0.0:
        raise ValueError("measured image has zero power in the fidelity ROI")
    scaled_measured = measured * (float(np.sum(reference)) / measured_sum)
    normalized_rmse = float(
        np.linalg.norm(scaled_measured - reference) / np.linalg.norm(reference)
    )
    return correlation, normalized_rmse


def _analyze_prepared_image_amplification(
    request,
    run_result,
    transmission: np.ndarray,
    normalized_grating: float,
    *,
    analytic_input_ratio: float,
    wavelength_um: float,
    image_request: (
        PRImageAmplificationRunRequest
        | PRBeamPanelImageAmplificationRunRequest
        | PRImageAmplificationExperimentRequest
        | None
    ),
    transparency_policy: str,
    incident_channel_powers_mW: tuple[float, float],
    pump_channel_index: int = 0,
    signal_channel_index: int = 1,
    pr_workflow_runtime_s: float,
    benchmark_started_at: float,
    stage_callback=None,
) -> PRImageAmplificationResult:
    """Apply the common image analysis to one capable ordinary PR result."""

    grid = make_grid(request.grid, real_dtype=np.float64)
    pump_kx = float(
        request.beams.channels[pump_channel_index].tilt_x_rad_per_um
    )
    signal_kx = float(
        request.beams.channels[signal_channel_index].tilt_x_rad_per_um
    )
    mask = signal_carrier_mask(
        grid,
        pump_kx_rad_per_um=pump_kx,
        signal_kx_rad_per_um=signal_kx,
    )
    prepared_A = np.asarray(run_result.A_initial)
    coherent_input = np.sum(prepared_A, axis=0)
    input_signal = isolate_signal_carrier(coherent_input, mask)
    if stage_callback is not None:
        stage_callback("carrier_isolation")
    reconstruction_started_at = perf_counter()
    coherent_output = np.sum(np.asarray(run_result.A_final), axis=0)
    output_signal = isolate_signal_carrier(coherent_output, mask)
    reconstruction_optical_runtime = 0.0
    optical_started_at = perf_counter()
    backpropagated = _linear_propagate(
        output_signal,
        grid,
        distance_um=-float(request.grid.z_length_um),
        request=request,
    )
    reconstruction_optical_runtime += perf_counter() - optical_started_at
    if stage_callback is not None:
        stage_callback("output_back_propagation")

    optical_started_at = perf_counter()
    zero_output = _linear_propagate(
        coherent_input,
        grid,
        distance_um=float(request.grid.z_length_um),
        request=request,
    )
    reconstruction_optical_runtime += perf_counter() - optical_started_at
    if stage_callback is not None:
        stage_callback("zero_response_propagation")
    zero_signal = isolate_signal_carrier(zero_output, mask)
    optical_started_at = perf_counter()
    zero_backpropagated = _linear_propagate(
        zero_signal,
        grid,
        distance_um=-float(request.grid.z_length_um),
        request=request,
    )
    reconstruction_optical_runtime += perf_counter() - optical_started_at
    if stage_callback is not None:
        stage_callback("reference_back_propagation")
    signal_channel_intensity = (
        np.abs(prepared_A[signal_channel_index]) ** 2
    )
    roi = signal_channel_intensity > 1e-4 * float(np.max(signal_channel_intensity))
    correlation, normalized_rmse = _intensity_metrics(
        input_signal,
        backpropagated,
        roi,
    )
    zero_correlation, _ = _intensity_metrics(
        input_signal,
        zero_backpropagated,
        roi,
    )
    if stage_callback is not None:
        stage_callback("metric_construction")

    dxdy = float(grid.dx_um) * float(grid.dy_um)
    input_signal_power = float(np.sum(np.abs(input_signal) ** 2) * dxdy)
    output_signal_power = float(np.sum(np.abs(output_signal) ** 2) * dxdy)
    measured_gain = output_signal_power / input_signal_power
    internal_angle = math.asin(
        pump_kx
        * float(wavelength_um)
        / (2.0 * math.pi * float(request.material.refractive_index))
    )
    analytic_gamma = analytic_plane_wave_gain_length(
        gain_length_product=float(request.material.gain_length_product),
        signed_grating_k_normalized=normalized_grating,
        internal_half_angle_rad=internal_angle,
    )
    analytic_gain = paper_absolute_signal_gain(
        input_ratio=float(analytic_input_ratio),
        gamma_p_L=analytic_gamma,
    )
    power_drift = float(
        (run_result.power_final - run_result.power_initial) / run_result.power_initial
    )
    reconstruction_runtime = perf_counter() - reconstruction_started_at
    incident_powers = tuple(float(value) for value in incident_channel_powers_mW)
    incident_total = float(sum(incident_powers))
    prepared_channel_powers = channel_power_integrals(prepared_A, grid)
    post_element_powers = (
        float(prepared_channel_powers[pump_channel_index]) * incident_total,
        float(prepared_channel_powers[signal_channel_index]) * incident_total,
    )
    signal_throughput = (
        post_element_powers[1] / incident_powers[1]
        if incident_powers[1] > 0.0
        else 0.0
    )

    return PRImageAmplificationResult(
        request=request,
        run_result=run_result,
        image_transmission=transmission,
        signal_carrier_mask=mask,
        input_signal_field=input_signal,
        output_signal_field=output_signal,
        backpropagated_signal_field=backpropagated,
        zero_response_backpropagated_signal_field=zero_backpropagated,
        transverse_phase_gradients_rad_per_um=(pump_kx, signal_kx),
        normalized_grating_wavenumber=normalized_grating,
        gain_length_product=float(request.material.gain_length_product),
        analytic_gamma_p_L=analytic_gamma,
        analytic_absolute_signal_gain=analytic_gain,
        measured_absolute_signal_gain=measured_gain,
        measured_gain_reference_signal_power_normalized=input_signal_power,
        output_isolated_signal_power_normalized=output_signal_power,
        measured_gain_reference_signal_power_mW=(
            input_signal_power * incident_total
        ),
        output_isolated_signal_power_mW=(
            output_signal_power * incident_total
        ),
        image_intensity_correlation=correlation,
        zero_response_image_intensity_correlation=zero_correlation,
        normalized_image_rmse=normalized_rmse,
        normalized_power_relative_drift=power_drift,
        incident_channel_powers_mW=incident_powers,
        post_element_channel_powers_mW=post_element_powers,
        incident_total_power_mW=incident_total,
        post_element_total_power_mW=float(sum(post_element_powers)),
        signal_throughput_fraction=signal_throughput,
        transparency_policy=transparency_policy,
        pr_workflow_runtime_s=pr_workflow_runtime_s,
        reconstruction_optical_runtime_s=reconstruction_optical_runtime,
        reconstruction_runtime_s=reconstruction_runtime,
        runtime_s=perf_counter() - benchmark_started_at,
        image_request=image_request,
    )


def _run_prepared_image_amplification(
    request: PRRunRequest,
    transmission: np.ndarray,
    normalized_grating: float,
    *,
    analytic_input_ratio: float,
    wavelength_um: float,
    image_request: (
        PRImageAmplificationRunRequest
        | PRBeamPanelImageAmplificationRunRequest
        | None
    ),
    transparency_policy: str,
    incident_channel_powers_mW: tuple[float, float],
    pump_channel_index: int = 0,
    signal_channel_index: int = 1,
    cancellation_token: CancellationToken | None = None,
    progress_callback=None,
) -> PRImageAmplificationResult:
    """Preserve the historical direct reduced-TD execution entry point."""

    benchmark_started_at = perf_counter()
    workflow_started_at = perf_counter()
    run_result = run_pr_timedependent(
        request,
        cancellation_token=cancellation_token,
        progress_callback=progress_callback,
    )
    return _analyze_prepared_image_amplification(
        request,
        run_result,
        transmission,
        normalized_grating,
        analytic_input_ratio=analytic_input_ratio,
        wavelength_um=wavelength_um,
        image_request=image_request,
        transparency_policy=transparency_policy,
        incident_channel_powers_mW=incident_channel_powers_mW,
        pump_channel_index=pump_channel_index,
        signal_channel_index=signal_channel_index,
        pr_workflow_runtime_s=perf_counter() - workflow_started_at,
        benchmark_started_at=benchmark_started_at,
    )


def run_image_amplification_request(
    image_request: (
        PRImageAmplificationRunRequest | PRBeamPanelImageAmplificationRunRequest
    ),
    *,
    cancellation_token: CancellationToken | None = None,
    progress_callback=None,
) -> PRImageAmplificationResult:
    """Execute a physical incident-power image request through existing PR TD."""

    request, transmission, normalized_grating = (
        prepare_image_amplification_workflow_request(image_request)
    )
    if isinstance(image_request, PRBeamPanelImageAmplificationRunRequest):
        launch_configuration = image_request.launch_configuration
        pump_index = image_request.pump_channel_index
        signal_index = image_request.signal_channel_index
        pump = launch_configuration.beams.channels[pump_index]
        signal = launch_configuration.beams.channels[signal_index]
        analytic_ratio = image_request.incident_signal_to_pump_power_ratio
        wavelength_um = float(signal.wavelength_um)
        incident_powers = (float(pump.power_mW), float(signal.power_mW))
    else:
        launch = image_request.launch
        pump_index = 0
        signal_index = 1
        analytic_ratio = launch.incident_signal_to_pump_power_ratio
        wavelength_um = launch.wavelength_um
        incident_powers = (
            launch.pump_incident_power_mW,
            launch.signal_incident_power_mW,
        )
    return _run_prepared_image_amplification(
        request,
        transmission,
        normalized_grating,
        analytic_input_ratio=analytic_ratio,
        wavelength_um=wavelength_um,
        image_request=image_request,
        transparency_policy="passive_intensity_transmission_v1",
        incident_channel_powers_mW=incident_powers,
        pump_channel_index=pump_index,
        signal_channel_index=signal_index,
        cancellation_token=cancellation_token,
        progress_callback=progress_callback,
    )


class _ImageAnalysisCancelled(RuntimeError):
    pass


def run_image_amplification_experiment(
    runner,
    request: PRImageAmplificationExperimentRequest,
    *,
    cancellation_token: CancellationToken | None = None,
    progress_callback=None,
    runner_kwargs: dict[str, Any] | None = None,
):
    """Run one image experiment through exactly one ordinary PR operation.

    Base-operation progress and the cancellation token are forwarded without
    translation.  Only the bounded post-processing stages emit image-specific
    progress.
    """

    from lcprop.pr.products import augment_pr_image_amplification_run_data
    from lcprop.runners.base import RunnerResult

    base_request, transmission, normalized_grating = (
        prepare_image_amplification_base_request(request)
    )
    capability = _base_capability(request.base_workflow_id)
    benchmark_started_at = perf_counter()
    base_started_at = perf_counter()
    kwargs = dict(runner_kwargs or {})
    kwargs["cancellation_token"] = cancellation_token
    kwargs["progress_callback"] = progress_callback
    base_runner_result = runner.run_registered(
        PR_MATERIAL_ID,
        request.base_workflow_id,
        base_request,
        **kwargs,
    )
    base_runtime = perf_counter() - base_started_at
    base_result = base_runner_result.result
    if not isinstance(base_result, capability.result_type):
        raise TypeError(
            f"{request.base_workflow_id} returned {type(base_result).__name__}; "
            f"expected {capability.result_type.__name__}"
        )
    for attribute in (
        "A_initial",
        "A_final",
        "grid_summary",
        "power_initial",
        "power_final",
        "status",
    ):
        if not hasattr(base_result, attribute):
            raise TypeError(
                f"base result lacks required image-amplification {attribute} capability"
            )

    base_status = str(base_result.status)
    analysis_result = None
    analysis_status: Literal["completed", "cancelled", "not_run", "failed"]
    analysis_message: str
    if base_status in ("cancelled", "failed"):
        analysis_status = "not_run"
        analysis_message = f"analysis not run because base status is {base_status}"
    elif cancellation_token is not None and cancellation_token.is_cancelled():
        analysis_status = "cancelled"
        analysis_message = "image analysis cancelled before carrier isolation"
    else:
        stage_started_at = perf_counter()

        def stage_completed(stage: str) -> None:
            if progress_callback is not None:
                index = PR_IMAGE_ANALYSIS_STAGES.index(stage) + 1
                progress_callback(
                    RunProgress(
                        workflow=PR_IMAGE_AMPLIFICATION_WORKFLOW,
                        status="running",
                        completed_units=index,
                        total_units=len(PR_IMAGE_ANALYSIS_STAGES),
                        current_coordinate=float(index),
                        coordinate_name="analysis_stage",
                        coordinate_unit="1",
                        elapsed_wall_time=perf_counter() - stage_started_at,
                        message=(
                            "Post-processing image amplification: "
                            f"{stage.replace('_', ' ')}"
                        ),
                        diagnostics={"phase": "image_postprocessing", "stage": stage},
                    )
                )
            if cancellation_token is not None and cancellation_token.is_cancelled():
                raise _ImageAnalysisCancelled(stage)

        screen, pump, signal = _image_screen_and_channels(request)
        channels = request.launch_configuration.beams.channels
        incident_powers = (
            float(channels[request.pump_channel_index].power_mW),
            float(channels[request.signal_channel_index].power_mW),
        )
        try:
            analysis_result = _analyze_prepared_image_amplification(
                base_request,
                base_result,
                transmission,
                normalized_grating,
                analytic_input_ratio=incident_powers[1] / incident_powers[0],
                wavelength_um=float(signal.wavelength_um),
                image_request=request,
                transparency_policy="passive_intensity_transmission_v1",
                incident_channel_powers_mW=incident_powers,
                pump_channel_index=request.pump_channel_index,
                signal_channel_index=request.signal_channel_index,
                pr_workflow_runtime_s=base_runtime,
                benchmark_started_at=benchmark_started_at,
                stage_callback=stage_completed,
            )
            analysis_status = "completed"
            analysis_message = "image analysis completed"
        except _ImageAnalysisCancelled as exc:
            analysis_status = "cancelled"
            analysis_message = f"image analysis cancelled after {exc}"
        except Exception as exc:  # preserve the successful base result for diagnosis
            analysis_status = "failed"
            analysis_message = f"image analysis failed: {type(exc).__name__}: {exc}"

    if analysis_result is not None:
        try:
            if cancellation_token is not None and cancellation_token.is_cancelled():
                raise _ImageAnalysisCancelled("metric_construction")
            run_data = augment_pr_image_amplification_run_data(
                analysis_result,
                base_runner_result.run_data,
            )
            if progress_callback is not None:
                progress_callback(
                    RunProgress(
                        workflow=PR_IMAGE_AMPLIFICATION_WORKFLOW,
                        status="running",
                        completed_units=len(PR_IMAGE_ANALYSIS_STAGES),
                        total_units=len(PR_IMAGE_ANALYSIS_STAGES),
                        current_coordinate=float(len(PR_IMAGE_ANALYSIS_STAGES)),
                        coordinate_name="analysis_stage",
                        coordinate_unit="1",
                        elapsed_wall_time=perf_counter() - benchmark_started_at,
                        message=(
                            "Post-processing image amplification: "
                            "product augmentation"
                        ),
                        diagnostics={
                            "phase": "image_postprocessing",
                            "stage": "product_augmentation",
                        },
                    )
                )
        except _ImageAnalysisCancelled as exc:
            analysis_result = None
            analysis_status = "cancelled"
            analysis_message = f"image analysis cancelled after {exc}"
            run_data = base_runner_result.run_data
        except Exception as exc:
            analysis_status = "failed"
            analysis_message = (
                "image product augmentation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            run_data = base_runner_result.run_data
    else:
        run_data = base_runner_result.run_data
    composite = PRImageAmplificationCompositeResult(
        request=request,
        base_runner_result=base_runner_result,
        analysis_result=analysis_result,
        analysis_status=analysis_status,
        analysis_message=analysis_message,
    )
    return RunnerResult(
        kind=PR_IMAGE_AMPLIFICATION_WORKFLOW,
        result=composite,
        message=analysis_message,
        run_data=run_data,
        material_id=PR_MATERIAL_ID,
    )


def run_image_amplification(
    image_intensity,
    spec: PRImageAmplificationSpec = PRImageAmplificationSpec(),
    *,
    backend: BackendSpec | None = None,
) -> PRImageAmplificationResult:
    """Run and measure the established finite image-amplification benchmark."""

    request, transmission, normalized_grating, _gain_length = (
        make_image_amplification_request(image_intensity, spec)
    )
    if backend is not None:
        request = replace(request, backend=backend)
    return _run_prepared_image_amplification(
        request,
        transmission,
        normalized_grating,
        analytic_input_ratio=float(spec.input_peak_ratio),
        wavelength_um=float(spec.wavelength_um),
        image_request=None,
        transparency_policy="historical_normalized_benchmark",
        incident_channel_powers_mW=tuple(
            float(channel.power_mW) for channel in request.beams.channels
        ),
    )


def run_streaming_image_amplification(
    image_intensity,
    spec: PRImageAmplificationSpec = PRImageAmplificationSpec(),
    *,
    solver: PRStreamingStaticOptions | None = None,
    backend: BackendSpec | None = None,
) -> PRStreamingImageAmplificationResult:
    """Run the static streaming workflow through the established measurement chain.

    The raw coherent output intensity is retained for a Figure 6-style output
    panel.  Gain and fidelity use the same Fourier carrier partition and
    matched-field back-propagation as the transient image benchmark; they do
    not spatially divide coherently overlapping beams.
    """

    transient_request, transmission, normalized_grating, gain_length = (
        make_image_amplification_request(image_intensity, spec)
    )
    selected_backend = transient_request.backend if backend is None else backend
    if solver is None:
        solver = PRStreamingStaticOptions(
            tukey_alpha=float(spec.tukey_alpha),
            volume_noise_epsilon=float(spec.volume_noise_epsilon),
            volume_noise_correlation_um=float(spec.volume_noise_correlation_um),
            volume_noise_seed=spec.volume_noise_seed,
            volume_noise_seeds=spec.volume_noise_seeds,
        )
    request = PRStreamingStaticRequest(
        grid=transient_request.grid,
        beams=transient_request.beams,
        material=transient_request.material,
        solver=solver,
        backend=selected_backend,
        initial_A=transient_request.initial_A,
    )
    grid = make_grid(request.grid, real_dtype=np.float64)
    pump_kx = float(request.beams.channels[0].tilt_x_rad_per_um)
    signal_kx = float(request.beams.channels[1].tilt_x_rad_per_um)
    mask = signal_carrier_mask(
        grid,
        pump_kx_rad_per_um=pump_kx,
        signal_kx_rad_per_um=signal_kx,
    )
    coherent_input = np.sum(np.asarray(request.initial_A), axis=0)
    input_signal = isolate_signal_carrier(coherent_input, mask)
    run_result = run_pr_static_streaming(request)
    coherent_output = np.sum(np.asarray(run_result.A_final), axis=0)
    output_signal = isolate_signal_carrier(coherent_output, mask)
    backpropagated = _linear_propagate(
        output_signal,
        grid,
        distance_um=-float(request.grid.z_length_um),
        request=request,
    )
    signal_channel_intensity = np.abs(np.asarray(request.initial_A)[1]) ** 2
    roi = signal_channel_intensity > 1e-4 * float(np.max(signal_channel_intensity))
    correlation, normalized_rmse = _intensity_metrics(
        input_signal,
        backpropagated,
        roi,
    )
    dxdy = float(grid.dx_um) * float(grid.dy_um)
    input_signal_power = float(np.sum(np.abs(input_signal) ** 2) * dxdy)
    output_signal_power = float(np.sum(np.abs(output_signal) ** 2) * dxdy)
    internal_angle = math.asin(
        pump_kx
        * float(spec.wavelength_um)
        / (2.0 * math.pi * float(spec.refractive_index))
    )
    analytic_gamma = analytic_plane_wave_gain_length(
        gain_length_product=gain_length,
        signed_grating_k_normalized=normalized_grating,
        internal_half_angle_rad=internal_angle,
    )
    analytic_gain = paper_absolute_signal_gain(
        input_ratio=float(spec.input_peak_ratio),
        gamma_p_L=analytic_gamma,
    )
    return PRStreamingImageAmplificationResult(
        request=request,
        run_result=run_result,
        image_transmission=transmission,
        input_signal_field=input_signal,
        output_signal_field=output_signal,
        backpropagated_signal_field=backpropagated,
        output_total_intensity=np.abs(coherent_output) ** 2,
        normalized_grating_wavenumber=normalized_grating,
        gain_length_product=gain_length,
        analytic_absolute_signal_gain=analytic_gain,
        measured_absolute_signal_gain=output_signal_power / input_signal_power,
        image_intensity_correlation=correlation,
        normalized_image_rmse=normalized_rmse,
        normalized_power_relative_drift=(
            run_result.power_final - run_result.power_initial
        )
        / run_result.power_initial,
    )


__all__ = [
    "PR_IMAGE_AMPLIFICATION_WORKFLOW",
    "PRBeamPanelImageAmplificationRunRequest",
    "PRImageAmplificationRunRequest",
    "PRImageAmplificationResult",
    "PRImageAmplificationSpec",
    "PRImageLaunchSpec",
    "PRStreamingImageAmplificationResult",
    "apply_passive_field_transmittance",
    "image_amplification_run_request",
    "intensity_transmission_to_field_transmittance",
    "isolate_signal_carrier",
    "make_image_amplification_request",
    "paper_absolute_signal_gain",
    "paper_figure4_spec",
    "paper_figure6_spec",
    "prepare_image_amplification_workflow_request",
    "prepare_image_transmission",
    "run_image_amplification",
    "run_image_amplification_request",
    "run_streaming_image_amplification",
    "signal_carrier_mask",
]
