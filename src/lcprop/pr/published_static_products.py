"""Streaming published-order products with explicit right-endpoint coordinates."""
from dataclasses import dataclass, replace
import numpy as np
from lcprop.pr.local_plane_products import LocalPlaneProductSelection, LocalPlaneProductResult, _Collector
from lcprop.pr.published_static import run_published_static, material_residual

PUBLISHED_STATIC_PRODUCTS_SCHEMA = "pr_published_static_products_v1"


@dataclass(frozen=True)
class PublishedStaticProducts:
    boundary_z_um: np.ndarray
    material_z_um: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray
    cuts: dict
    preview: np.ndarray | None
    preview_x_um: np.ndarray | None
    preview_y_um: np.ndarray | None
    presentation_reference: float | None
    metadata: dict


class _PublishedCollector(_Collector):
    def residual(self, frame, xp):
        return material_residual(frame.material_field, frame.source_intensity,
                                 request=self.request, xp=xp)

    def finish(self, scientific):
        # The shared collector stores explicit observed coordinates; it never
        # infers a midpoint. Only the old workflow calls those samples centers.
        old = super().finish(scientific)
        values = vars(old).copy()
        values['material_z_um'] = values.pop('center_z_um')
        metadata = dict(values['metadata'])
        metadata.pop('center_quantity')
        metadata.update(schema=PUBLISHED_STATIC_PRODUCTS_SCHEMA,
            material_quantity='right_endpoint_arriving_source_material_residual',
            material_coordinate_axis='material_z_um',
            material_model=self.request.material_response.model)
        values['metadata'] = metadata
        return PublishedStaticProducts(**values)


def run_published_products(request, *, selection=LocalPlaneProductSelection(),
                           cancellation_token=None, progress_callback=None):
    selection.validate()
    collector = _PublishedCollector(request, selection)
    def progress(event):
        collector.progress(event)
        if progress_callback is not None:
            progress_callback(event)
    scientific = run_published_static(request, retain_boundary_field=selection.endpoint,
        cancellation_token=cancellation_token, progress_callback=progress,
        observation_callback=collector.observe)
    if not selection.far_field:
        scientific = replace(scientific, far_field=None, far_field_z_um=None)
    return LocalPlaneProductResult(scientific, collector.finish(scientific))
