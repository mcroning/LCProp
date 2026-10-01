"""Unregistered headless, transverse-streaming Local-I Static workflow.

No GUI, codecs, continuation, old Static dispatch or longitudinal products.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Callable

import numpy as np

from lcprop.core.backend import BackendSpec, asnumpy, get_backend
from lcprop.core.beams import BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.execution import CancellationToken
from lcprop.core.grid import make_grid
from lcprop.optics.boundaries import TransverseBoundarySpec
from lcprop.optics.farfield import DirectionCosineSpectrum, direction_cosine_spectrum
from lcprop.optics.launch import OpticalLaunchContext, build_launch
from lcprop.optics.launch_configuration import LaunchConfiguration, reject_prepared_launch_conflict
from lcprop.optics.screens import ChannelLaunchElements
from lcprop.pr.local_plane_step import step_local_intensity_cell
from lcprop.pr.scattering import (
    PR_CANONICAL_SCATTERING_V2, PRCanonicalScatteringSpec, canonical_slab_range,
)
from lcprop.pr.source import channel_peak_intensity_reference
from lcprop.pr.specs import PRMaterialSpec
from lcprop.pr.transverse.specs import (
    PR_MATERIAL_RESPONSE_FIELD_LINEAR, PRTransverseMaterialResponseSpec,
)

PR_LOCAL_PLANE_WORKFLOW = "pr_static_local_intensity_planes_v1"
PR_LOCAL_PLANE_ARITHMETIC = "per_cell_half_linear_full_local_phase_v1"


@dataclass(frozen=True)
class LocalPlaneRunRequest:
    grid: GridSpec
    beams: BeamStack
    material: PRMaterialSpec = PRMaterialSpec()
    backend: BackendSpec = BackendSpec("numpy", "float64", False)
    material_response: PRTransverseMaterialResponseSpec = field(default_factory=lambda:
        PRTransverseMaterialResponseSpec(model=PR_MATERIAL_RESPONSE_FIELD_LINEAR))
    optical_boundary: TransverseBoundarySpec = TransverseBoundarySpec()
    scattering: PRCanonicalScatteringSpec | None = None
    launch_elements: tuple[ChannelLaunchElements, ...] = ()
    initial_A: Any | None = None
    residual_rms_tolerance: float = 1e-5
    residual_max_tolerance: float = 1e-4


@dataclass(frozen=True)
class LocalPlaneProgress:
    completed_cells: int
    requested_cells: int
    reached_z_um: float
    last_cell: dict[str, Any] | None


@dataclass(frozen=True)
class LocalPlaneObservation:
    """Borrowed launch/boundary or candidate-cell data, valid during callback.

    Arrays are isolated backend copies, not mutable scientific buffers. Copy
    explicitly inside the callback to retain data; retaining borrowed arrays
    or using them asynchronously after return is unsupported. Mutation fails
    the cell transaction. Coordinates describe a qualified *candidate*, whose
    acceptance is subsequently announced by scalar progress, not by this hook.
    The launch_boundary event has z_end_um=0, no cell/start/center, and None
    source/material arrays. It precedes cancellation checks and all cell calls.
    Initial progress confirms successful launch observation/identity checks.
    """
    cell_index: int | None
    z_start_um: float | None
    z_end_um: float
    material_plane_um: float | None
    _arrays: tuple[Any, Any | None, Any | None] | None = field(repr=False)
    kind: str = "cell_candidate"

    def _get(self, index):
        if self._arrays is None:
            raise RuntimeError("observation expired; copy arrays inside the callback")
        return self._arrays[index]

    @property
    def boundary_field(self):
        """Optical boundary at z_end_um."""
        return self._get(0)

    @property
    def source_intensity(self):
        """Source at material_plane_um."""
        return self._get(1)

    @property
    def material_field(self):
        """Material solution at material_plane_um."""
        return self._get(2)


def _observe_cell(callback, cell, record, xp):
    originals = (cell.A_candidate, cell.source_intensity, cell.E)
    _observe_arrays(callback, originals, xp, record['cell_index'],
        record['z_start_um'], record['z_end_um'], record['material_plane_um'], "cell_candidate")


def _observe_arrays(callback, originals, xp, index, start, end, center, kind):
    copies = tuple(value.copy(order="C") if value is not None else None for value in originals)
    observation = LocalPlaneObservation(index, start, end, center, copies, kind)
    try:
        callback(observation)  # Return value never participates in acceptance.
        for original, observed in zip(originals, copies):
            if original is None:
                continue
            if (observed.shape != original.shape or observed.dtype != original.dtype
                    or not bool(xp.array_equal(
                        xp.ascontiguousarray(original).view(xp.uint8),
                        xp.ascontiguousarray(observed).view(xp.uint8)))):
                raise ValueError("observation hook mutated borrowed array")
    finally:
        object.__setattr__(observation, '_arrays', None)


def _append_cell_record(ledger, record):
    ledger.append(record)


@dataclass(frozen=True)
class LocalPlaneRunResult:
    status: str
    reason: str
    requested_cells: int | None
    completed_cells: int
    reached_z_um: float
    last_material_plane_um: float | None
    last_residual_rms: float | None
    last_residual_max: float | None
    launch_identity: dict[str, Any] | None
    ledger: tuple[dict[str, Any], ...]
    failure: dict[str, Any] | None
    boundary_field: Any | None
    far_field: DirectionCosineSpectrum | None
    far_field_z_um: float | None
    workflow_identity: str = PR_LOCAL_PLANE_WORKFLOW
    arithmetic_identity: str = PR_LOCAL_PLANE_ARITHMETIC


def validate_local_plane_domain(length_um: float, dz_um: float) -> int:
    """Exact intended integer coverage under the committed binary64 end policy.

    Use index multiplication, never accumulated z addition. This scalar-only
    preflight validates every cell before any launch or scientific execution.
    """
    length, h = float(length_um), float(dz_um)
    if not all(math.isfinite(v) and v > 0 for v in (length, h)):
        raise ValueError("domain length and cell width must be finite and positive")
    ratio = length / h
    if not math.isfinite(ratio) or ratio > 2**53:
        raise ValueError("cell count is not representable in binary64 metadata")
    n = round(ratio)
    if n < 1:
        raise ValueError("domain must contain at least one complete cell")
    for k in range(n):
        z = k*h
        end = z+h
        if not math.isfinite(end) or z >= length or end <= z:
            raise ValueError("domain has a nonadvancing or out-of-domain cell")
        bound = 2*max(map(math.ulp, (z, h, end, length)))
        terminal = abs(end-length) <= bound
        if (k == n-1 and not terminal) or (k < n-1 and (terminal or end >= length)):
            raise ValueError("domain length is not an integer number of requested cells")
    return n


def _compact(value):
    """Copy JSON-sized scalar records; refuse arrays/nonfinite values."""
    def scalar(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError("compact evidence cannot retain arrays or runtime objects")
    return json.loads(json.dumps(value, default=scalar, allow_nan=False))


def _launch_hash(A) -> str:
    # <=1 MiB per transfer; no full launch plane is materialized on the host.
    digest = hashlib.sha256()
    flat = A.reshape(-1)
    count = max(1, (1024*1024)//A.dtype.itemsize)
    for start in range(0, flat.size, count):
        digest.update(asnumpy(flat[start:start+count]).tobytes(order="C"))
    return digest.hexdigest()


def _failure_record(stage, index, exc):
    functions = []
    tb = exc.__traceback__
    while tb is not None:
        functions.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    reason = str(exc)
    kind = "exception"
    if stage == "cell_step":
        if (isinstance(exc, ValueError) and reason.startswith((
                "material residual", "material must be finite",
                "local total transport intensity", "source must be finite"))):
            kind = "material_invalid"
        elif "solve_pr_reduced_field_linear_intensity" in functions:
            kind = "material_failure"
        elif "canonical_scattering_phase_increment" in functions or "scattering phase" in reason:
            kind = "scattering_failure"
        elif "finite" in reason:
            kind = "nonfinite"
    return dict(stage=stage, cell_index=index, kind=kind,
                exception_type=type(exc).__name__, reason=reason,
                origin_function=functions[-1] if functions else None)


def run_local_intensity_planes(
    request: LocalPlaneRunRequest,
    *,
    retain_boundary_field: bool = False,
    cancellation_token: CancellationToken | None = None,
    progress_callback: Callable[[LocalPlaneProgress], None] | None = None,
    observation_callback: Callable[[LocalPlaneObservation], Any] | None = None,
) -> LocalPlaneRunResult:
    """Run once; failures become compact evidence, never scientific retries.

    Cancellation before acceptance discards a complete candidate. Accepted
    input remains owned and unchanged. Progress is emitted after launch and
    accepted boundaries only, and never contains arrays. Caller mutation of a
    supplied launch after the initial progress event cannot affect this run.

    Optional synchronous launch observation precedes cell 0. Cell observation
    runs before acceptance, while center and boundary data remain live. Its
    return value is ignored. It uses bounded
    transverse backend copies and exact byte comparison (including signed
    zeros); no observation arrays survive the callback unless the caller
    explicitly copies them. No-hook execution makes no observation copies.
    GPU callbacks must use the current stream or join their work before return.

    Arrays returned on the execution backend are transferred to caller
    ownership, never reused by another invocation. No result retains request
    initial_A, center source/material planes, or a longitudinal volume. Exact
    launch hashing streams bounded host chunks (cumulative launch-size traffic
    on GPU). No final spectrum is exposed for a failed/cancelled result.
    """
    accepted = spectrum = launch_identity = None
    requested = None
    completed = 0
    reached = 0.0
    ledger = []
    failure = None
    status, reason = "failed", "execution did not complete"
    stage, attempted = "request_validation", None
    cancelled = lambda: cancellation_token is not None and cancellation_token.is_cancelled()
    try:
        if not isinstance(request, LocalPlaneRunRequest):
            raise TypeError("requires LocalPlaneRunRequest; old Static requests are not reinterpreted")
        request.grid.validate()
        request.beams.validate()
        request.material.validate()
        request.backend.validate()
        request.material_response.validate()
        if request.material_response.model != PR_MATERIAL_RESPONSE_FIELD_LINEAR:
            raise ValueError("local-plane workflow supports only Local-I field-linear Static")
        request.optical_boundary.validate()
        if request.optical_boundary.mode != "periodic":
            raise ValueError("local-plane workflow supports periodic boundaries only")
        if request.grid.Nx < 3:
            raise ValueError("Local-I material requires Nx >= 3")
        for v in (request.residual_rms_tolerance, request.residual_max_tolerance):
            if not math.isfinite(v) or v <= 0:
                raise ValueError("residual limits must be finite and positive")
        LaunchConfiguration(request.beams, request.launch_elements)
        reject_prepared_launch_conflict(request.initial_A, request.launch_elements)
        wavelengths = tuple(float(c.wavelength_um) for c in request.beams.channels)
        if any(v != wavelengths[0] for v in wavelengths[1:]):
            raise ValueError("local-plane workflow requires one shared wavelength")
        length, h = float(request.grid.z_length_um), float(request.grid.dz_um)
        stage = "domain_validation"
        requested = validate_local_plane_domain(length, h)
        if request.scattering is not None:
            stage = "scattering_validation"
            if request.scattering.algorithm_version != PR_CANONICAL_SCATTERING_V2:
                raise ValueError("local-plane workflow requires canonical V2 scattering")
            previous_stop = 0
            for k in range(requested):
                slabs = canonical_slab_range(request.scattering, z_start_um=k*h,
                                             dz_um=h, z_length_um=length)
                if slabs.start != previous_stop:
                    raise ValueError("scattering cells have a gap or duplicated slab")
                previous_stop = slabs.stop
            domain = canonical_slab_range(request.scattering, z_start_um=0.,
                                          dz_um=length, z_length_um=length)
            if previous_stop != domain.stop:
                raise ValueError("scattering cells do not cover the domain")
        stage = "launch"
        backend = get_backend(request.backend)
        xp = backend.xp
        grid = make_grid(request.grid, xp=xp, real_dtype=backend.real_dtype)
        launch = build_launch(request.beams, grid, complex_dtype=backend.complex_dtype,
            launch_elements=request.launch_elements,
            context=OpticalLaunchContext(grid, request.material.refractive_index, length))
        if request.initial_A is None:
            initial = launch.A0.copy(order="C")
        elif isinstance(request.initial_A, np.ndarray):
            snapshot = np.array(request.initial_A, dtype=backend.complex_dtype, order="C", copy=True)
            initial = xp.asarray(snapshot)
            if backend.is_gpu:
                xp.cuda.get_current_stream().synchronize()
            del snapshot
        else:
            initial = xp.asarray(request.initial_A, dtype=backend.complex_dtype).copy(order="C")
        del launch
        if initial.shape != (len(wavelengths), grid.Nx, grid.Ny):
            raise ValueError("initial_A shape does not match the accepted launch")
        if not bool(xp.all(xp.isfinite(initial)).item()):
            raise ValueError("accepted launch must be finite")
        reference = channel_peak_intensity_reference(initial, xp=xp)
        if not math.isfinite(reference):
            raise ValueError("scientific launch reference must be finite")
        launch_identity = dict(shape=list(initial.shape), dtype=str(initial.dtype),
            sha256=_launch_hash(initial), peak_intensity_reference=reference,
            source="generated" if request.initial_A is None else "supplied_owned_snapshot",
            backend=backend.name)
        accepted = initial
        del initial
        if observation_callback is not None:
            stage = "launch_observation"
            _observe_arrays(observation_callback, (accepted, None, None), xp,
                            None, None, 0.0, None, "launch_boundary")

        def progress():
            if progress_callback is not None:
                progress_callback(LocalPlaneProgress(completed, requested, reached,
                    _compact(ledger[-1]) if ledger else None))

        stage = "progress"
        progress()
        status, reason = "completed", "all cells and final spectrum completed"
        for k in range(requested):
            if cancelled():
                status, reason = "cancelled", "cancelled before next cell"
                break
            attempted, stage = k, "cell_step"
            cell = step_local_intensity_cell(accepted, grid=grid, cell_index=k,
                z_start_um=k*h, dz_um=h, interaction_length_um=length,
                wavelength_um=wavelengths[0], material=request.material,
                peak_intensity_reference=reference, backend=request.backend,
                coherence_groups=request.beams.coherence_groups, scattering=request.scattering,
                residual_rms_tolerance=request.residual_rms_tolerance,
                residual_max_tolerance=request.residual_max_tolerance)
            stage = "cell_diagnostics"
            norm = float((xp.sum(xp.abs(cell.A_candidate)**2, dtype=xp.float64)
                          *grid.dx_um*grid.dy_um).item())
            if not math.isfinite(norm):
                raise ValueError("candidate channel norm must be finite")
            record = _compact(dict(cell.diagnostics, accepted_channel_norm=norm))
            if observation_callback is not None and not cancelled():
                stage = "cell_observation"
                _observe_cell(observation_callback, cell, record, xp)
            if cancelled():
                del cell
                status, reason = "cancelled", "completed candidate discarded before acceptance"
                break
            stage = "cell_bookkeeping"
            next_completed, next_reached = k+1, record['z_end_um']
            next_accepted = cell.A_candidate
            try:
                _append_cell_record(ledger, record)
            except Exception:
                del ledger[completed:]
                raise
            # All fallible qualification/bookkeeping precedes state replacement.
            accepted = next_accepted
            completed = next_completed
            reached = next_reached
            del next_accepted
            del cell
            attempted = None
            stage = "progress"
            progress()
        if status == "completed":
            if cancelled():
                status, reason = "cancelled", "cancelled before final spectrum"
            else:
                stage = "final_far_field"
                candidate_spectrum = direction_cosine_spectrum(accepted,
                    dx_um=grid.dx_um, dy_um=grid.dy_um, wavelength_um=wavelengths[0],
                    refractive_index=request.material.refractive_index,
                    coherence_groups=request.beams.coherence_groups, xp=xp)
                if (candidate_spectrum.intensity.shape != (grid.Nx, grid.Ny)
                        or candidate_spectrum.intensity.dtype != xp.dtype(backend.real_dtype)
                        or candidate_spectrum.s_x.shape != (grid.Nx,)
                        or candidate_spectrum.s_y.shape != (grid.Ny,)):
                    raise ValueError("final spectrum shape/dtype mismatch")
                for value in (candidate_spectrum.intensity, candidate_spectrum.s_x, candidate_spectrum.s_y):
                    if not bool(xp.all(xp.isfinite(value)).item()):
                        raise ValueError("final spectrum must be finite")
                if cancelled():
                    status, reason = "cancelled", "cancelled during final spectrum construction"
                else:
                    spectrum = candidate_spectrum
    except Exception as exc:
        failure = _failure_record(stage, attempted, exc)
        status, reason = "failed", str(exc)
        spectrum = None
    last = ledger[-1] if ledger else {}
    return LocalPlaneRunResult(status, reason, requested, completed, reached,
        last.get('material_plane_um'), last.get('residual_rms'), last.get('residual_max'),
        launch_identity, tuple(ledger), failure,
        accepted if retain_boundary_field else None, spectrum,
        reached if spectrum is not None else None)
