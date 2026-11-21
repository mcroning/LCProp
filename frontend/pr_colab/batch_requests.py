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
from lcprop.pr.experiment_codec import (
    PR_TRANSVERSE_STATIC_EXPERIMENT_CODEC, PR_TRANSVERSE_TIMEDEPENDENT_EXPERIMENT_CODEC,
)
from lcprop.pr.transverse.operations import (
    PR_TRANSVERSE_STATIC_OPERATION, PR_TRANSVERSE_TIMEDEPENDENT_OPERATION,
)
from lcprop.pr.transverse.specs import PRTransverseRunRequest
from lcprop.pr.transverse.static_workflow import PRTransverseStaticRunRequest
from lcprop.pr.transverse.transport_codec import PR_TRANSVERSE_STATIC_TRANSPORT_CODEC
from lcprop.pr.transverse.timedependent_transport_codec import (
    PR_TRANSVERSE_TIMEDEPENDENT_TRANSPORT_CODEC,
)

TRANSVERSE_ROUTES = {"pr_xy_static", "pr_xy_td"}

ROUTES = {
    "pr_xy_static": (PR_TRANSVERSE_STATIC_OPERATION, PR_TRANSVERSE_STATIC_EXPERIMENT_CODEC,
                     PR_TRANSVERSE_STATIC_TRANSPORT_CODEC),
    "pr_xy_td": (PR_TRANSVERSE_TIMEDEPENDENT_OPERATION, PR_TRANSVERSE_TIMEDEPENDENT_EXPERIMENT_CODEC,
                 PR_TRANSVERSE_TIMEDEPENDENT_TRANSPORT_CODEC),
    "lc_td": (LC_TIMEDEPENDENT_OPERATION, LC_TIMEDEPENDENT_EXPERIMENT_CODEC, None),
    "pr_static": (PR_STATIC_OPERATION, PR_STATIC_EXPERIMENT_CODEC, PR_STATIC_TRANSPORT_CODEC),
    "lc": (LC_STATIC_OPERATION, LC_STATIC_EXPERIMENT_CODEC, LC_STATIC_TRANSPORT_CODEC),
    "pr": (PR_TIMEDEPENDENT_OPERATION, PR_TIMEDEPENDENT_EXPERIMENT_CODEC,
           PR_TIMEDEPENDENT_TRANSPORT_CODEC),
}
# Additional desktop Static operations share the same native engine codecs.
from lcprop.persistence.experiments import ExperimentRequestCodec
from lcprop.pr.unified.integration import (UnifiedFreshRequest, UNIFIED_OPERATION,
    UNIFIED_TRANSPORT_CODEC, encode_fresh, decode_fresh)
from lcprop.pr.unified.specs import PRElectricalClosureSpec, UNBIASED
from lcprop.pr.published_static import PublishedStaticRequest
from lcprop.pr.published_static_integration import (PUBLISHED_STATIC_OPERATION,
    PUBLISHED_STATIC_TRANSPORT_CODEC, encode_published_static_request, decode_published_static_request)
for key, operation, typ, encoder, decoder, transport in (
    ('pr_unified',UNIFIED_OPERATION,UnifiedFreshRequest,encode_fresh,decode_fresh,UNIFIED_TRANSPORT_CODEC),
    ('pr_published',PUBLISHED_STATIC_OPERATION,PublishedStaticRequest,encode_published_static_request,
     decode_published_static_request,PUBLISHED_STATIC_TRANSPORT_CODEC)):
    ROUTES[key]=(operation,ExperimentRequestCodec('pr',operation.workflow_id,typ,encoder,decoder),transport)

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
    if route in {"pr_unified", "pr_published"}:
        base=preset("pr")
        if route == "pr_unified":
            return UnifiedFreshRequest(base.grid,base.beams,base.material,PRElectricalClosureSpec(UNBIASED,1,(0.,)))
        return PublishedStaticRequest(base.grid,base.beams,base.material)
    if route == "lc_td":
        base = preset("lc")
        return TimeDependentRunRequest(
            base.grid, base.material, base.bias, base.beams, TimeDependentSolverOptions(),
            base.output, base.runtime, optical_boundary=base.optical_boundary,
        )
    if route in TRANSVERSE_ROUTES:
        base = preset("pr")
        grid = replace(base.grid, Nx=24, Ny=16, z_length_um=8.)
        request_type = PRTransverseStaticRunRequest if route == "pr_xy_static" else PRTransverseRunRequest
        request = request_type(grid, base.beams, replace(base.material, applied_field=0.0))
        if route == "pr_xy_td":
            request = replace(request, solver=replace(request.solver, Nt=3))
        return request
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
        backend=BackendSpec(backend="numpy", precision="float64", verbose=False),
    )


def beam_stack(original, channels, coherence):
    channels = tuple(channels)
    if coherence == "coherent":
        channels = tuple(replace(ch, coherence_group=channels[0].coherence_group)
                         for ch in channels)
    elif coherence == "incoherent":
        channels = tuple(replace(ch,coherence_group=f"independent:{i}:{ch.name}")
                         for i,ch in enumerate(channels))
    return replace(original, channels=channels, coherence=coherence or original.coherence)


def refractive_index(route, request):
    return request.material.no if material_for(route) == "lc" else request.material.refractive_index


def validate(route, request, **_):
    """Desktop numerical preflight and native codec; no notebook input whitelist."""
    from lcprop.pr.gui.request_adapter import validate_pr_gui_workflow_request
    if route not in {key for key in ROUTES if key.startswith("pr")}:
        raise ValueError("This workbench accepts PR only")
    if type(request) is not type(preset(route)):
        raise ValueError("Request type does not match the selected workflow")
    # Auto is intent, not a numerical backend. Check CPU compatibility here;
    # the actual resolved backend is validated again before worker dispatch.
    concrete = replace(request, backend=replace(request.backend, backend=(
        "numpy" if request.backend.backend == "auto" else request.backend.backend)))
    try:
        report = validate_pr_gui_workflow_request(concrete)
    except ValueError:
        if request.backend.backend != "auto":raise
        concrete=replace(request,backend=replace(request.backend,backend="cupy"))
        report=validate_pr_gui_workflow_request(concrete)
    return ROUTES[route][1].encode_request(request), tuple(report.warnings)


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
