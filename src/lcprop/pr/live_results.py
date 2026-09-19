"""Bounded Local presentation of an already accepted reduced PR TD state.

No integration, optical propagation, checkpoint, or retained history lives here.
Sampling is point sampling at recorded physical coordinates, not a new solver grid.
"""
from dataclasses import dataclass
import math

import numpy as np

from lcprop.core.backend import asnumpy
from lcprop.optics.splitstep import total_intensity
from lcprop.products.data_model import (
    CurveCollection, DiagnosticCollection, DiagnosticData, FieldCollection,
    Geometry, RunData, make_field,
)
from lcprop.pr.specs import PR_TIMEDEPENDENT_WORKFLOW


@dataclass(frozen=True)
class PRLivePreviewPolicy:
    """Presentation-only limits; never change the accepted-step cadence."""

    max_x: int = 128
    max_y: int = 128
    max_z: int = 256
    interval_seconds: float = 0.5

    def __post_init__(self):
        for value in (self.max_x, self.max_y, self.max_z):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("preview dimensions must be positive integers")
        if not math.isfinite(self.interval_seconds) or self.interval_seconds < 0:
            raise ValueError("preview interval must be finite and nonnegative")

    @property
    def maximum_array_bytes(self):
        # Four float64 images plus three float64 coordinate vectors.
        x, y, z = self.max_x, self.max_y, self.max_z
        return 8 * (2*x*y + z*x + z*y + x + y + z)


@dataclass(frozen=True)
class PRLiveSnapshot:
    output_intensity: np.ndarray
    material_plane: np.ndarray
    optical_xz: np.ndarray
    optical_yz: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray
    z_um: np.ndarray
    material_z_um: float
    x_cut_um: float
    y_cut_um: float
    completed_steps: int
    segment_completed_steps: int
    requested_steps: int
    time_normalized: float
    scalar_values: dict
    material_response: str

    @property
    def array_bytes(self):
        return sum(value.nbytes for value in (
            self.output_intensity, self.material_plane, self.optical_xz,
            self.optical_yz, self.x_um, self.y_um, self.z_um,
        ))


def _indices(size, maximum):
    return np.rint(np.linspace(0, size - 1, min(size, maximum))).astype(int)


def _detached(value):
    array = np.array(asnumpy(value), dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def reduced_pr_live_snapshot(*, E, A, source, grid, groups, peak_reference,
                             background, policy, completed_steps,
                             segment_completed_steps, requested_steps,
                             time_normalized, scalar_values, material_response):
    """Slice on the compute backend before any host transfer.

    Source cuts retain the existing PR slice-average optical-intensity convention.
    Only a bounded 2-D advanced-index result is transferred for each image.
    """
    xp = grid.xp
    nx, ny, nz = grid.Nx, grid.Ny, grid.Nz
    ix, iy, iz = (_indices(nx, policy.max_x), _indices(ny, policy.max_y),
                  _indices(nz, policy.max_z))
    bx, by, bz = xp.asarray(ix), xp.asarray(iy), xp.asarray(iz)
    x = (np.arange(nx) - 0.5*(nx-1)) * grid.dx_um
    y = (np.arange(ny) - 0.5*(ny-1)) * grid.dy_um
    cx, cy, cz = int(np.argmin(abs(x))), int(np.argmin(abs(y))), nz // 2
    # Intensity is calculated only for the sampled transverse optical field.
    intensity = total_intensity(A[:, bx[:, None], by[None, :]],
                                coherence_groups=groups, xp=xp)
    xz = (source[bz[:, None], bx[None, :], cy] - background) * peak_reference
    yz = (source[bz[:, None], cx, by[None, :]] - background) * peak_reference
    return PRLiveSnapshot(
        _detached(intensity), _detached(E[cz, bx[:, None], by[None, :]]),
        _detached(xz), _detached(yz), _detached(x[ix]), _detached(y[iy]),
        _detached(iz * grid.dz_um), float(cz * grid.dz_um),
        float(x[cx]), float(y[cy]), int(completed_steps),
        int(segment_completed_steps), int(requested_steps), float(time_normalized),
        dict(scalar_values), str(material_response),
    )


def reduced_pr_live_to_run_data(snapshot: PRLiveSnapshot) -> RunData:
    """Adapt presentation buffers directly, without fabricating a final result."""
    s = snapshot
    geometry = Geometry(x=s.x_um, y=s.y_um, z=s.z_um)
    units = {"x": "um", "y": "um", "z": "um"}
    fields = FieldCollection()
    fields.add("live_output_intensity", make_field(
        "live_output_intensity", "Current accepted output intensity",
        s.output_intensity, ("x", "y"), "intensity", units,
        quantity="normalized_intensity", value_unit="1/µm²",
        coordinates={"x": s.x_um, "y": s.y_um}, initially_selected=True,
    ))
    fields.add("live_material_plane", make_field(
        "live_material_plane", f"Accepted PR field at z={s.material_z_um:g} µm",
        s.material_plane, ("x", "y"), "pr_space_charge", units,
        quantity="E", value_unit="1", colormap="coolwarm",
        coordinates={"x": s.x_um, "y": s.y_um, "material_z_um": s.material_z_um},
    ))
    for key, partner, axes, array in (
        ("live_optical_xz", "live_optical_yz", ("z", "x"), s.optical_xz),
        ("live_optical_yz", "live_optical_xz", ("z", "y"), s.optical_yz),
    ):
        fields.add(key, make_field(
            key, "Accepted optical intensity — sampled fixed cuts", array,
            axes, "intensity", units, "longitudinal",
            quantity="physical_optical_intensity", value_unit="1/µm²",
            source_volume_key="live_optical_cuts",
            coordinates={"paired_cut_key": partner, "x_cut_um": s.x_cut_um,
                         "y_cut_um": s.y_cut_um, "z": s.z_um,
                         "x": s.x_um, "y": s.y_um},
        ))
    summary = dict(s.scalar_values, material_time_normalized=s.time_normalized,
                   completed_material_steps=s.completed_steps,
                   segment_completed_steps=s.segment_completed_steps,
                   requested_material_steps=s.requested_steps,
                   material_response=s.material_response,
                   material_response_validation=("locally_validated" if s.material_response == "linearized" else "validated"),
                   visualization_only=True, snapshot_array_bytes=s.array_bytes,
                   sampling="point samples at explicit physical coordinates",
                   optical_observation="complete replay through the same accepted E",
                   longitudinal_sampling="existing slice-average optical intensity",
                   material_plane_z_um=s.material_z_um)
    return RunData(workflow=PR_TIMEDEPENDENT_WORKFLOW, geometry=geometry,
                   fields=fields, curves=CurveCollection(),
                   diagnostics=DiagnosticCollection([
                       ("live_accepted_state", DiagnosticData(
                           "live_accepted_state", "Live accepted-state snapshot", summary))]))
