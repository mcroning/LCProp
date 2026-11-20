"""Bounded PR entrance visualization, independent of scientific admission."""
from dataclasses import replace

import numpy as np

from lcprop.core.grid import make_grid
from lcprop.notebook.batch_requests import refractive_index
from lcprop.optics.launch import OpticalLaunchContext, build_launch
from lcprop.optics.sampling import qualify_launch_sampling
from lcprop.optics.splitstep import total_intensity
from lcprop.products.data_model import make_field

MAX_PREVIEW_SAMPLES = 1024 * 1024  # Includes channels; not a scientific-grid policy.


def editing_grid(route, request):
    if route not in {"pr", "pr_static"}:
        raise ValueError("The PR workbench supports reduced Static and TD only")
    request.grid.validate()
    index = refractive_index(route, request)
    nx, ny = min(request.grid.Nx, 128), min(request.grid.Ny, 128)
    while True:
        spec = replace(request.grid, Nx=nx, Ny=ny)
        if nx * ny * max(1, len(request.beams.channels)) > MAX_PREVIEW_SAMPLES:
            raise ValueError("Editing preview exceeds its bounded display budget. "
                             "Your scientific grid and beam settings are unchanged.")
        report = qualify_launch_sampling(request.beams, spec, index,
                                        boundary=request.optical_boundary)
        bandwidth = tuple(w for w in report.warnings if "Gaussian spectral" in w)
        if not report.errors and not bandwidth:
            return spec
        messages = (*report.errors, *bandwidth)
        grow_x = any(", x:" in message for message in messages)
        grow_y = any(", y:" in message for message in messages)
        if not grow_x and not grow_y:
            report.require_valid()
            raise ValueError("Cannot resolve editing preview sampling")
        nx, ny = (2 * nx if grow_x else nx), (2 * ny if grow_y else ny)


def editing_preview(route, request):
    spec = editing_grid(route, request)
    grid = make_grid(spec, real_dtype=np.float64)
    launch = build_launch(request.beams, grid, complex_dtype=np.complex128,
                          context=OpticalLaunchContext(
                              grid=grid, n_ref=refractive_index(route, request),
                              interaction_length_um=spec.z_length_um))
    intensity = total_intensity(launch.A0, coherence_groups=launch.coherence_groups)
    return make_field("input_preview", "Draft input intensity (editing preview)", intensity,
                      ("x", "y"), "intensity", {"x": "um", "y": "um"},
                      value_unit="1/µm²", coordinates={"x": grid.x_um, "y": grid.y_um})
