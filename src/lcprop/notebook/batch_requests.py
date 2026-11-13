"""Request adapters for the four notebook routes."""
from dataclasses import replace
import math

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.persistence import LC_STATIC_EXPERIMENT_CODEC, PR_TIMEDEPENDENT_EXPERIMENT_CODEC
from lcprop.lc.operations import LC_STATIC_OPERATION, LC_TIMEDEPENDENT_OPERATION
from lcprop.lc.requests import TimeDependentRunRequest, TimeDependentSolverOptions, validate_single_wavelength_lc_beams
from lcprop.lc.experiment_codec import LC_TIMEDEPENDENT_EXPERIMENT_CODEC
from lcprop.lc.transport_codec import LC_STATIC_TRANSPORT_CODEC
from lcprop.notebook.lc_static_demo import build_request
from lcprop.optics.launch import OpticalLaunchContext, build_launch
from lcprop.optics.sampling import qualify_launch_sampling
from lcprop.optics.splitstep import total_intensity
from lcprop.pr.gui.request_adapter import validate_pr_gui_request, validate_pr_static_gui_request
from lcprop.pr.operations import PR_TIMEDEPENDENT_OPERATION, PR_STATIC_OPERATION
from lcprop.pr.static_workflow import PRStaticRunRequest
from lcprop.pr.static_transport_codec import PR_STATIC_TRANSPORT_CODEC
from lcprop.pr.experiment_codec import PR_STATIC_EXPERIMENT_CODEC
from lcprop.pr.specs import PRMaterialSpec, PRRunRequest, PRSolverOptions
from lcprop.pr.timedependent_transport_codec import PR_TIMEDEPENDENT_TRANSPORT_CODEC
from lcprop.products.data_model import make_field

ROUTES = {
    "lc_td": (LC_TIMEDEPENDENT_OPERATION, LC_TIMEDEPENDENT_EXPERIMENT_CODEC, None),
    "pr_static": (PR_STATIC_OPERATION, PR_STATIC_EXPERIMENT_CODEC, PR_STATIC_TRANSPORT_CODEC),
    "lc": (LC_STATIC_OPERATION, LC_STATIC_EXPERIMENT_CODEC, LC_STATIC_TRANSPORT_CODEC),
    "pr": (PR_TIMEDEPENDENT_OPERATION, PR_TIMEDEPENDENT_EXPERIMENT_CODEC,
           PR_TIMEDEPENDENT_TRANSPORT_CODEC),
}
BEAM_CONTROLS = {
    "x0_um": "x center (µm)", "y0_um": "y center (µm)",
    "w1_um": "w1 radius (µm)", "w2_um": "w2 radius (µm)",
    "power_mW": "Incident power (mW)", "theta_ext_rad": "External θ (rad)",
    "phi_rad": "Azimuth (rad)", "psi_rad": "Roll (rad)", "phase_rad": "Phase (rad)",
}
PR_MATERIAL_CONTROLS = {
    "gain_length_product": "Gain-length product",
    "dark_irradiance_W_cm2": "Dark irradiance (W/cm²)",
    "uniform_irradiance_W_cm2": "Uniform irradiance (W/cm²)",
}


def material_for(route):
    return ROUTES[route][0].material_id


def preset(route):
    if route == "lc_td":
        base = preset("lc")
        return TimeDependentRunRequest(
            base.grid, base.material, base.bias, base.beams, TimeDependentSolverOptions(),
            base.output, base.runtime, optical_boundary=base.optical_boundary,
        )
    if route == "pr_static":
        base = preset("pr")
        return PRStaticRunRequest(base.grid, base.beams, base.material,
                                  optical_boundary=base.optical_boundary)

    if route == "lc":
        request = build_request()
        return replace(request, bias=replace(request.bias, V_bias=0.9144))
    if route != "pr":
        raise ValueError(f"Unsupported route: {route}")
    return PRRunRequest(
        grid=GridSpec(Nx=64, Ny=48, x_aperture_um=80., y_aperture_um=60.,
                      dz_um=1., z_length_um=1000.),
        beams=BeamStack(channels=(BeamChannel(name="demo", w1_um=12., w2_um=12.,
                                             coherence_group="demo-laser"),)),
        material=PRMaterialSpec.physical(dark_irradiance_W_cm2=0.01,
                                        uniform_irradiance_W_cm2=0.,
                                        gain_length_product=0.1),
        solver=PRSolverOptions(Nt=3, dt_normalized=0.001,
                               integrator="semi_implicit_trapezoidal"),
        backend=BackendSpec(backend="auto", precision="float64", verbose=False),
    )


def beam_stack(original, channels, coherence):
    channels = tuple(channels)
    if coherence == "coherent":
        channels = tuple(replace(ch, coherence_group=channels[0].coherence_group)
                         for ch in channels)
    elif coherence == "incoherent" and len(channels) == 2:
        if channels[0].coherence_group == channels[1].coherence_group:
            channels = (channels[0], replace(
                channels[1], coherence_group=channels[0].coherence_group + ":beam-2"))
    return replace(original, channels=channels, coherence=coherence or original.coherence)


def refractive_index(route, request):
    return request.material.no if material_for(route) == "lc" else request.material.refractive_index


def validate(route, request, *, sampling=qualify_launch_sampling,
             pr_preflight=validate_pr_gui_request):
    """Reject unsupported state before codec admission, without normalizing it."""
    reference = preset(route)
    if type(request) is not type(reference):
        raise ValueError("Unsupported request type")
    lc = material_for(route) == "lc"
    excluded = ("initial_A", "initial_theta") if lc else ("initial_A", "initial_E", "scattering")
    for name in excluded:
        if getattr(request, name) is not None:
            raise ValueError(f"{name} is outside this notebook workflow")
    if not lc and request.launch_elements:
        raise ValueError("Launch elements are unavailable in this notebook")
    request.grid.validate()
    request.material.validate()
    request.beams.validate()
    request.optical_boundary.validate()
    if request.optical_boundary != reference.optical_boundary:
        raise ValueError("Only the preset optical boundary is editable in this batch")
    channels = request.beams.channels
    if not 1 <= len(channels) <= 2:
        raise ValueError("Use one or two Gaussian beams")
    for channel in channels:
        channel.validate()
        if (channel.profile != "collimated_gaussian" or channel.wavelength_um != 0.633
                or channel.n_ext != 1.0):
            raise ValueError("Use 0.633 µm collimated Gaussian beams in air")
    if sum(ch.power_mW for ch in channels) <= 0:
        raise ValueError("Total beam power must be positive")

    solver = request.solver
    editable = {"lc": {"max_iterations", "static_max_relax_iterations", "static_max_coupled_passes", "static_residual_rms_tol", "static_residual_max_tol"},
                "lc_td": {"Nt", "dt", "gamma_z", "max_picard_iter", "tolerance_update"},
                "pr_static": {"max_coupled_passes", "optical_substeps", "residual_rms_tolerance", "residual_max_tolerance"},
                "pr": {"Nt", "dt_normalized", "optical_substeps", "integrator"}}[route]
    if any(getattr(solver, name) != getattr(reference.solver, name)
           for name in solver.__dataclass_fields__ if name not in editable):
        raise ValueError("Unsupported hidden solver setting")
    integers = {
        "lc": ("max_iterations", "static_max_relax_iterations", "static_max_coupled_passes"),
        "lc_td": ("Nt", "max_picard_iter"),
        "pr_static": ("max_coupled_passes", "optical_substeps"),
        "pr": ("Nt", "optical_substeps"),
    }[route]
    for name in integers:
        value = getattr(solver, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("dt", "dt_normalized", "tolerance_update", "static_residual_rms_tol",
                 "static_residual_max_tol", "residual_rms_tolerance", "residual_max_tolerance"):
        value = getattr(solver, name, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")

    if lc:
        validate_single_wavelength_lc_beams(request.beams)
        if request.material != reference.material:
            raise ValueError("Unsupported hidden LC material settings")
        if not math.isfinite(request.bias.V_bias):
            raise ValueError("Applied voltage must be finite")
        if replace(request.bias, V_bias=reference.bias.V_bias) != reference.bias:
            raise ValueError("Unsupported hidden LC bias settings")
        request.bias.validate()
        request.runtime.validate()
        if request.runtime != reference.runtime or request.output != reference.output:
            raise ValueError("Unsupported LC runtime or output settings")
        if solver.workflow != reference.solver.workflow:
            raise ValueError("Unsupported LC solver strategy")
        if route == "lc_td" and (not math.isfinite(solver.gamma_z) or solver.gamma_z < 0):
            raise ValueError("gamma_z must be finite and nonnegative")
        check = sampling(request.beams, request.grid, refractive_index(route, request),
                         boundary=request.optical_boundary)
        check.require_valid()
        warnings = check.warnings
    else:
        request.backend.validate()
        if request.backend.precision != "float64" or request.backend.verbose:
            raise ValueError("The notebook requires float64 without backend verbosity")
        if request.material_response != reference.material_response:
            raise ValueError("Only the reduced nonlinear response is available")
        if route == "pr_static":
            warnings = validate_pr_static_gui_request(request).aperture.warnings
        else:
            report = pr_preflight(request)
            warnings = report.aperture.warnings
            if getattr(report, "timestep_assessment", None) == "pending":
                warnings = (*warnings, "Timestep assessment is pending until engine validation")
    return ROUTES[route][1].encode_request(request), tuple(warnings)


def preview(route, request):
    """Sample the canonical entrance plane without propagating."""
    spec = replace(request.grid, Nx=min(request.grid.Nx, 128),
                   Ny=min(request.grid.Ny, 128))
    index = refractive_index(route, request)
    qualify_launch_sampling(request.beams, spec, index,
                            boundary=request.optical_boundary).require_valid()
    grid = make_grid(spec, real_dtype=np.float64)
    launch = build_launch(request.beams, grid, complex_dtype=np.complex128,
                          context=OpticalLaunchContext(
        grid=grid, n_ref=index, interaction_length_um=spec.z_length_um))
    data = total_intensity(launch.A0, coherence_groups=launch.coherence_groups)
    return make_field("input_preview", "Draft input intensity preview", data,
                      ("x", "y"), "intensity", {"x": "um", "y": "um"},
                      value_unit="1/µm²", coordinates={"x": grid.x_um, "y": grid.y_um})
