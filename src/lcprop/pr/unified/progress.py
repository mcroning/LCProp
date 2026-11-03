"""Detached ephemeral post-acceptance presentation; never a scientific state.

Callbacks may retain or mutate this payload without changing accepted arrays.
No launch/candidate event is emitted. A callback exception fails presentation
after acceptance; cancellation requests take effect before the next cell.
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np

from lcprop.optics.splitstep import total_intensity
from lcprop.pr.visualization import downsample_td_movie_frame, _bin_edges


@dataclass(frozen=True)
class AcceptedProgress:
    completed_cells: int
    total_cells: int
    z_um: float
    cell_index: int
    diagnostics: dict
    preview: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray


def notify_accepted(observer, field, grid, groups, record, completed, total, xp):
    # Direct transfer API: do not swallow instrumentation/transfer exceptions.
    transfer = np.asarray if xp is np else xp.asnumpy
    intensity = total_intensity(field, coherence_groups=groups, xp=xp)
    preview = downsample_td_movie_frame(intensity, xp=xp, asnumpy=transfer)
    coordinates = []
    for axis in (grid.x_um, grid.y_um):
        edges = _bin_edges(axis.size, 128)
        centers = (axis[xp.asarray(edges[:-1])] + axis[xp.asarray(edges[1:] - 1)]) / 2
        coordinates.append(np.array(transfer(centers), copy=True))
    observer(AcceptedProgress(completed, total, record['z_end_um'],
        record['cell_index'], deepcopy({k:v for k,v in record['observations'].items() if 'iteration_' not in k}),
        preview, *coordinates))
