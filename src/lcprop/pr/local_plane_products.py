"""Selected streaming products for the boundary-canonical local-plane workflow.

No dispatch, persistence, continuation or longitudinal scientific volumes.
Boundary and material-center coordinates are explicit and never conflated.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from lcprop.core.backend import asnumpy, get_backend
from lcprop.pr.local_plane_workflow import LocalPlaneRunResult, run_local_intensity_planes
from lcprop.pr.longitudinal_cuts import (
    centered_transverse_coordinates, presentation_peak_intensity_reference,
    _normalize_presentation_cut,
)
from lcprop.pr.reduced_field_linear import PRReducedFieldLinearSpec, reduced_field_linear_residual
from lcprop.pr.source import channel_peak_intensity_reference, pr_driving_intensity
from lcprop.pr.visualization import block_average_2d, _block_centers

LOCAL_PLANE_PRODUCTS_SCHEMA = 'pr_local_plane_products_v1'


@dataclass(frozen=True)
class LocalPlaneProductSelection:
    intensity_cuts: bool = True
    optical_cuts: bool = False
    material_cuts: bool = False
    source_cuts: bool = False
    residual_cuts: bool = False
    preview: bool = True
    endpoint: bool = False
    far_field: bool = True

    def validate(self):
        if any(type(v) is not bool for v in vars(self).values()):
            raise TypeError('product selections must be booleans')


@dataclass(frozen=True)
class LocalPlaneProducts:
    boundary_z_um: np.ndarray
    center_z_um: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray
    # intensity: (Nb,Nx)/(Nb,Ny); optical: (Nb,Nch,Nx)/(Nb,Nch,Ny).
    # material/source/residual: (Nc,Nx)/(Nc,Ny).
    cuts: dict[str, tuple[np.ndarray, np.ndarray]]
    preview: np.ndarray | None
    preview_x_um: np.ndarray | None
    preview_y_um: np.ndarray | None
    presentation_reference: float | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LocalPlaneProductResult:
    scientific: LocalPlaneRunResult
    products: LocalPlaneProducts


class _Collector:
    """One pending reduced record; publish only on confirmed accepted progress.

    Observations are borrowed. Only copied cuts/reduced preview survive their
    callback. Final reconciliation discards any unaccepted candidate, including
    bookkeeping failures after observation. No cell data are recomputed by a
    second scientific march. Diagnostic residual evaluation has no feedback.
    """
    def __init__(self, request, selection):
        self.request, self.selection = request, selection
        self.pending = None
        self.records = []
        self.reference = self.scientific_reference = None
        self.x, self.y = centered_transverse_coordinates(dict(
            Nx=request.grid.Nx, Ny=request.grid.Ny,
            dx_um=request.grid.x_aperture_um/request.grid.Nx,
            dy_um=request.grid.y_aperture_um/request.grid.Ny))
        self.ix, self.iy = int(np.argmin(abs(self.x))), int(np.argmin(abs(self.y)))
        self.preview_x = self.preview_y = None

    def pair(self, values):
        return (np.array(asnumpy(values[..., :, self.iy]), copy=True),
                np.array(asnumpy(values[..., self.ix, :]), copy=True))

    def observe(self, frame):
        selection, request = self.selection, self.request
        xp = get_backend(request.backend).xp
        launch = frame.kind == 'launch_boundary'
        needs_intensity = selection.intensity_cuts or selection.preview
        if launch and needs_intensity:
            self.reference = presentation_peak_intensity_reference(frame.boundary_field, asnumpy=asnumpy)
            self.scientific_reference = channel_peak_intensity_reference(frame.boundary_field, xp=xp)
        cuts = {}
        preview = None
        if selection.optical_cuts:
            cuts['optical'] = self.pair(frame.boundary_field)
        if needs_intensity:
            source = pr_driving_intensity(frame.boundary_field,
                peak_intensity_reference=self.scientific_reference,
                background_intensity=request.material.background_intensity,
                coherence_groups=request.beams.coherence_groups, xp=xp)
            if selection.intensity_cuts:
                cuts['intensity'] = tuple(_normalize_presentation_cut(a, self.reference,
                    request.material.background_intensity) for a in self.pair(source))
            if selection.preview:
                # Same certified NumPy float64 physical-intensity arithmetic and
                # contiguous-block reduction. Area <= Nx+Ny, including tiny grids;
                # collecting these reduced planes is never a scientific volume.
                physical = (np.asarray(asnumpy(source), dtype=np.float64)
                            - request.material.background_intensity) * self.reference
                maximum = min(96, max(1, int(np.sqrt(request.grid.Nx + request.grid.Ny))))
                reduced, xe, ye = block_average_2d(physical, max_x=maximum, max_y=maximum)
                preview = reduced.astype(np.float32)
                self.preview_x, self.preview_y = _block_centers(self.x, xe), _block_centers(self.y, ye)
        if not launch:
            if selection.material_cuts:
                cuts['material'] = self.pair(frame.material_field)
            if selection.source_cuts:
                cuts['source'] = self.pair(frame.source_intensity)
            if selection.residual_cuts:
                spec = PRReducedFieldLinearSpec(request.material.applied_field,
                    request.material.background_intensity,
                    request.material.characteristic_wavenumber_per_um * (request.grid.x_aperture_um/request.grid.Nx))
                residual = reduced_field_linear_residual(frame.material_field,
                    frame.source_intensity, spec=spec, xp=xp)
                cuts['residual'] = self.pair(residual)
        self.pending = dict(index=frame.cell_index, boundary=frame.z_end_um,
            center=frame.material_plane_um, cuts=cuts, preview=preview)

    def progress(self, progress):
        record = self.pending
        if record is None:
            return
        count = 0 if record['index'] is None else record['index'] + 1
        if count != progress.completed_cells or record['boundary'] != progress.reached_z_um:
            raise ValueError('observation/acceptance coordinate mismatch')
        if len(self.records) != count:
            raise ValueError('observation/acceptance sequence mismatch')
        self.records.append(record)
        self.pending = None

    def finish(self, scientific):
        self.pending = None  # Never infer acceptance from an observation alone.
        records = self.records[:scientific.completed_cells + 1]
        boundary = np.asarray([r['boundary'] for r in records], dtype=np.float64)
        centers = np.asarray([r['center'] for r in records[1:]], dtype=np.float64)
        cuts = {}
        for name in ('intensity', 'optical', 'material', 'source', 'residual'):
            if not getattr(self.selection, name + '_cuts'):
                continue
            samples = records if name in ('intensity', 'optical') else records[1:]
            prefix = (len(self.request.beams.channels),) if name == 'optical' else ()
            real_dtype = np.dtype(self.request.backend.precision)
            dtype = (np.dtype('complex64' if real_dtype == np.float32 else 'complex128')
                     if name == 'optical' else real_dtype)
            cuts[name] = tuple(np.stack([r['cuts'][name][axis] for r in samples]) if samples
                else np.empty((0,) + prefix + (size,), dtype=dtype)
                for axis, size in enumerate((len(self.x), len(self.y))))
        preview = np.stack([r['preview'] for r in records]) if records and self.selection.preview else None
        # Reference/preview geometry is not published for a failed launch observer.
        reference = self.reference if records else None
        return LocalPlaneProducts(boundary, centers, self.x.copy(), self.y.copy(), cuts, preview,
            self.preview_x if preview is not None else None, self.preview_y if preview is not None else None,
            reference, dict(schema=LOCAL_PLANE_PRODUCTS_SCHEMA,
                boundary_quantity='optical_boundary', center_quantity='pre_kick_material_source_residual',
                preview_visualization_only=True, preview_coordinate_axis='boundary_z_um',
                x_cut_um=float(self.x[self.ix]), y_cut_um=float(self.y[self.iy]),
                presentation_normalization='(driving_intensity - material_background) * numpy_launch_reference',
                scientific_reference=self.scientific_reference if records else None,
                channel_count=len(self.request.beams.channels), real_dtype=self.request.backend.precision,
                selection=vars(self.selection).copy()))


def run_local_plane_products(request, *, selection=LocalPlaneProductSelection(),
                             cancellation_token=None, progress_callback=None):
    """Separate headless entry point; old Static dispatch/codecs are untouched.

    Cuts have all accepted explicit coordinates. Preview has every accepted
    boundary but only <= Nx+Ny float32 pixels per plane. Transient construction
    uses bounded transverse workspaces (one host plane for exact preview
    arithmetic); no longitudinal scientific arrays. GPU launch reference uses
    bounded host chunks. Presentation reference never enters the scientific run.
    """
    selection.validate()
    collector = _Collector(request, selection)
    def progress(event):
        collector.progress(event)
        if progress_callback is not None:
            progress_callback(event)
    scientific = run_local_intensity_planes(request,
        retain_boundary_field=selection.endpoint, cancellation_token=cancellation_token,
        progress_callback=progress, observation_callback=collector.observe)
    if not selection.far_field:
        scientific = replace(scientific, far_field=None, far_field_z_um=None)
    return LocalPlaneProductResult(scientific, collector.finish(scientific))
