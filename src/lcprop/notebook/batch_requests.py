"""Authoritative request adapters for the bounded CPU notebook routes."""
from dataclasses import replace
import math

import numpy as np

from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.persistence import LC_STATIC_EXPERIMENT_CODEC, PR_TIMEDEPENDENT_EXPERIMENT_CODEC
from lcprop.lc.operations import LC_STATIC_OPERATION
from lcprop.lc.transport_codec import LC_STATIC_TRANSPORT_CODEC
from lcprop.notebook.lc_static_demo import build_request
from lcprop.optics.launch import OpticalLaunchContext, build_launch
from lcprop.optics.sampling import qualify_launch_sampling
from lcprop.optics.splitstep import total_intensity
from lcprop.pr.gui.request_adapter import validate_pr_gui_request
from lcprop.pr.operations import PR_TIMEDEPENDENT_OPERATION
from lcprop.pr.specs import PRMaterialSpec, PRRunRequest, PRSolverOptions
from lcprop.pr.timedependent_transport_codec import PR_TIMEDEPENDENT_TRANSPORT_CODEC
from lcprop.products.data_model import make_field

ROUTES = {
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


def preset(route):
    if route == "lc":
        request = build_request()
        return replace(request, bias=replace(request.bias, V_bias=0.9144))
    if route != "pr":
        raise ValueError(f"Unsupported route: {route}")
    return PRRunRequest(
        grid=GridSpec(Nx=64, Ny=48, x_aperture_um=80., y_aperture_um=60.,
                      dz_um=1., z_length_um=20.),
        beams=BeamStack(channels=(BeamChannel(name="demo", w1_um=12., w2_um=12.,
                                             coherence_group="demo-laser"),)),
        material=PRMaterialSpec.physical(dark_irradiance_W_cm2=0.01,
                                        uniform_irradiance_W_cm2=0.,
                                        gain_length_product=0.1),
        solver=PRSolverOptions(Nt=3, dt_normalized=0.001,
                               integrator="semi_implicit_trapezoidal"),
        backend=BackendSpec(backend="numpy", precision="float64", verbose=False),
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
    return request.material.no if route == "lc" else request.material.refractive_index


def validate(route, request, *, sampling=qualify_launch_sampling,
             pr_preflight=validate_pr_gui_request):
    """Reject unsupported state before codec admission, without normalizing it."""
    reference = preset(route)
    if type(request) is not type(reference):
        raise ValueError("Unsupported request type")
    excluded = ("initial_A", "initial_theta") if route == "lc" else (
        "initial_A", "initial_E", "scattering")
    for name in excluded:
        if getattr(request, name) is not None:
            raise ValueError(f"{name} is outside this batch")
    if route == "pr" and request.launch_elements:
        raise ValueError("launch_elements are outside this batch")
    request.beams.validate()
    channels = request.beams.channels
    if not 1 <= len(channels) <= 2:
        raise ValueError("Use one or two Gaussian beams")
    for channel in channels:
        channel.validate()
        if channel.profile != "collimated_gaussian" or channel.wavelength_um != 0.633:
            raise ValueError("Only 0.633 µm collimated Gaussian beams are supported")
        if channel.n_ext != 1.0:
            raise ValueError("The external refractive index is fixed at 1")
    if sum(ch.power_mW for ch in channels) <= 0:
        raise ValueError("Total beam power must be positive")
    fixed = replace(request, beams=reference.beams)
    if route == "lc":
        if not math.isfinite(request.bias.V_bias):
            raise ValueError("Voltage must be finite")
        fixed = replace(fixed, bias=replace(request.bias, V_bias=reference.bias.V_bias))
    else:
        solver = request.solver
        if type(solver.Nt) is not int or solver.Nt not in (1, 2, 3) or not 0 < solver.dt_normalized <= 0.001:
            raise ValueError("Use 1–3 material steps and 0 < normalized dt ≤ 0.001")
        fixed = replace(
            fixed,
            solver=replace(solver, Nt=reference.solver.Nt,
                           dt_normalized=reference.solver.dt_normalized),
            material=replace(request.material, **{
                name: getattr(reference.material, name) for name in PR_MATERIAL_CONTROLS
            }),
        )
    for name in reference.__dataclass_fields__:
        if getattr(fixed, name) != getattr(reference, name):
            raise ValueError(f"{name} differs from the bounded {route.upper()} preset")
    encoded = ROUTES[route][1].encode_request(request)
    if route == "pr":
        return encoded, tuple(pr_preflight(request).aperture.warnings)
    check = sampling(request.beams, request.grid, refractive_index(route, request),
                     boundary=request.optical_boundary)
    check.require_valid()
    return encoded, tuple(check.warnings)


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
