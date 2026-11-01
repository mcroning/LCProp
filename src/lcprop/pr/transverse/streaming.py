"""Opt-in plane-local isotropic zero-flux model with indexed phases and replay.

This fourth model is not a tensor-faithful BaTiO3 model. The material solver,
optical response and existing reduced streaming models are not replaced.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from numbers import Integral, Real
from typing import Any

import numpy as np

from lcprop.core.backend import scalar_float
from lcprop.core.grid import round_nz
from lcprop.pr.frozen_scattering import FrozenScatteringPhases
from lcprop.pr.disk_scattering import DiskScatteringPhases
from lcprop.pr.indexed_scattering import IndexedScatteringPhases
from lcprop.pr.gpu_indexed_scattering import GPUIndexedScatteringPhases
from lcprop.pr.static_streaming import PRStreamingStaticRequest
from lcprop.pr.transverse.marching_static import (
    PRTransverseMarchingStaticOptions, PRTransverseMarchingStaticRunRequest,
    PRTransverseMarchingStaticRunResult, run_pr_transverse_static_marching,
    _validate_request,
)
from lcprop.pr.transverse.static import project_production_resolved_modes
from lcprop.pr.transverse.transport import (
    state_from_potential, spectral_wavevectors, spectral_derivatives,
)
from lcprop.pr.transverse.diagnostics import state_diagnostics

PR_TRANSVERSE_STREAMING_V1 = "full_transverse_isotropic_zero_flux_streaming_v1"
_PHASE_TYPES = (FrozenScatteringPhases, DiskScatteringPhases,
                IndexedScatteringPhases, GPUIndexedScatteringPhases)


@dataclass(frozen=True)
class TransversePhaseBinding:
    """Bind an existing verified source to its original physical/seed request.

    ``context`` is metadata for phase validation, not a reduced model execution.
    Its grid, material, beams and numerical backend must match the fourth model.
    No descriptor, seed schedule, normalization or runtime binding is rewritten.
    """
    source: Any
    context: PRStreamingStaticRequest

    def validate_for(self, request, *, required_cells):
        if not isinstance(self.source, _PHASE_TYPES):
            raise TypeError("source must be a supported hash-verified phase source")
        if not isinstance(self.context, PRStreamingStaticRequest):
            raise TypeError("phase context must be PRStreamingStaticRequest")
        if request.scattering is not None:
            raise ValueError("indexed and canonical scattering cannot be combined")
        for name in ("grid", "material", "beams"):
            if asdict(getattr(request, name)) != asdict(getattr(self.context, name)):
                raise ValueError(f"phase binding {name} mismatch")
        if (request.backend.backend, request.backend.precision) != (
                self.context.backend.backend, self.context.backend.precision):
            raise ValueError("phase binding backend mismatch")
        self.context.grid.validate()
        self.context.material.validate()
        self.context.beams.validate()
        self.context.solver.validate()
        self.source.validate_for(self.context, required_cells=required_cells)

    def apply_to(self, field, index, *, xp):
        if isinstance(self.source, FrozenScatteringPhases):
            # Identical arithmetic and boundary placement to reduced streaming.
            field *= xp.exp(1j * self.source.phase(index))[None, :, :]
        else:
            self.source.apply_to(field, index, xp=xp)

    def provenance(self):
        return self.source.provenance()


def transverse_request_from_reference(
    reference: PRStreamingStaticRequest, *,
    options: PRTransverseMarchingStaticOptions,
    initial_psi=None,
) -> PRTransverseMarchingStaticRunRequest:
    """Copy physical inputs only; require explicit fourth-model solver controls."""
    if not isinstance(reference, PRStreamingStaticRequest):
        raise TypeError("reference must be PRStreamingStaticRequest")
    if reference.solver.tukey_alpha != 0:
        raise ValueError("fourth model comparison requires disabled apodization")
    if options.optical_substeps != reference.solver.coupled.optical_substeps:
        raise ValueError("optical substep counts must match the reference")
    request = PRTransverseMarchingStaticRunRequest(
        grid=reference.grid, beams=reference.beams, material=reference.material,
        backend=reference.backend, initial_A=reference.initial_A,
        initial_psi=initial_psi, solver=options,
    )
    _validate_request(request)
    return request


def _plane_diagnostics(evaluation, *, interval_index, material_records,
                       dx_normalized, dy_normalized, request, xp):
    """Scalar observations only. Temporary arrays never escape this function."""
    kw = dict(dx_normalized=dx_normalized, dy_normalized=dy_normalized,
              h_y=request.dielectric.h_y, xp=xp)
    state = state_from_potential(
        evaluation.psi, potential_space=request.solver.material_solver.potential_space, **kw)
    out = state_diagnostics(state, **kw)
    out["carrier_integral"] = float(out.pop("carrier_integrals_per_z"))
    out.update(interval_index=int(interval_index),
               carrier_minimum=evaluation.carrier_minimum,
               carrier_mean=evaluation.carrier_mean,
               potential_mean=evaluation.potential_mean)

    def metrics(value):
        return dict(rms=scalar_float(xp.sqrt(xp.mean(value * value))),
                    maximum=scalar_float(xp.max(xp.abs(value))))

    def moments(value):
        return [scalar_float(xp.sum(value)), scalar_float(xp.sum(value * value)),
                scalar_float(xp.min(value)), scalar_float(xp.max(value))]

    projected = project_production_resolved_modes(evaluation.residual, **kw)
    out["full_residual"] = metrics(evaluation.residual)
    out["projected_residual"] = metrics(projected)
    out["null_residual"] = metrics(evaluation.residual - projected)
    out["moments"] = {name: moments(value) for name, value in (
        ("psi", state.psi), ("carrier", state.carrier_density),
        ("E_x", state.E_x), ("E_y", state.E_y),
        ("source", evaluation.source), ("residual", evaluation.residual))}
    kx, ky = spectral_wavevectors(state.psi.shape, dx_normalized=dx_normalized,
                                  dy_normalized=dy_normalized, xp=xp)
    pi = state.carrier_density * evaluation.source
    gx, gy = spectral_derivatives(pi, kx=kx, ky=ky, xp=xp)
    jx = gx - pi * state.E_x
    jy = request.transport.m_y * (gy - pi * state.E_y)
    out["flux_x"] = metrics(jx)
    out["flux_y"] = metrics(jy)
    jxx, _ = spectral_derivatives(jx, kx=kx, ky=ky, xp=xp)
    _, jyy = spectral_derivatives(jy, kx=kx, ky=ky, xp=xp)
    out["flux_divergence"] = metrics(jxx + jyy)
    out["flux_note"] = "production spectral product-rule diagnostic; not a replacement equilibrium equation"
    out["material_history"] = [asdict(record) for record in material_records]
    return out


@dataclass(frozen=True)
class PRTransverseStreamingResult:
    first_pass: PRTransverseMarchingStaticRunResult
    replay_pass: PRTransverseMarchingStaticRunResult | None
    converged: bool
    status: str
    diagnostics: dict[str, Any]


def run_pr_transverse_streaming(
    request: PRTransverseMarchingStaticRunRequest, *,
    launch_peak_intensity_reference: float | None = None,
    phase_binding: TransversePhaseBinding | None = None,
    stop_after_accepted_cells: int | None = None,
    replay_rtol: float = 1e-11,
    replay_atol: float = 1e-12,
) -> PRTransverseStreamingResult:
    """Run the fourth model with bounded retention and independent replay.

    Original L/Nz and the initial optical/material seed are never shortened or
    replaced by the preceding pass's state. Failures preserve the first result.
    The full equilibrium residual gates this opt-in model (including null modes);
    the legacy marcher's default projected acceptance remains unchanged.
    """
    _validate_request(request)
    if request.backend.precision != "float64":
        raise ValueError("fourth-model qualification requires float64")
    if request.optical_boundary.mode != "periodic":
        raise ValueError("fourth-model comparison requires periodic optical boundaries")
    if request.scattering is not None:
        raise ValueError("use an explicit indexed phase binding, not canonical scattering")
    for value in (replay_rtol, replay_atol):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not math.isfinite(value) or value <= 0):
            raise ValueError("replay tolerances must be finite and positive")
    nz = round_nz(request.grid.z_length_um, request.grid.dz_um)
    if stop_after_accepted_cells is not None and (
        isinstance(stop_after_accepted_cells, (bool, np.bool_))
        or not isinstance(stop_after_accepted_cells, Integral)
        or not 1 <= stop_after_accepted_cells <= nz
    ):
        raise ValueError("accepted-cell prefix must be an integer in [1, Nz]")
    target = nz if stop_after_accepted_cells is None else int(stop_after_accepted_cells)
    bounded = replace(request, solver=replace(
        request.solver, record_material_stacks=False,
        record_residual_stack=False, record_optical_audit_fields=False,
    ))
    kwargs = dict(stop_after_accepted_intervals=target,
                  launch_peak_intensity_reference=launch_peak_intensity_reference,
                  phase_binding=phase_binding, record_streaming_diagnostics=True,
                  require_full_residual=True)
    first = run_pr_transverse_static_marching(bounded, **kwargs)
    info = dict(model=PR_TRANSVERSE_STREAMING_V1, requested_cells=target,
                physical_Nz=nz, physical_length_um=request.grid.z_length_um,
                propagated_distance_um=first.completed_intervals*request.grid.dz_um,
                partial_domain_prefix=target<nz, replay_performed=False,
                retention="plane-local GPU state; scalar per-cell histories; host final fields",
                tensor_faithful_BaTiO3=False)

    def succeeded(result):
        return (result.failed_interval is None and result.completed_intervals == target
                and all(s.converged for s in result.interval_summaries)
                and np.isfinite(result.A_final).all())

    if not succeeded(first):
        return PRTransverseStreamingResult(first, None, False, "first_pass_failed", info)
    try:
        second = run_pr_transverse_static_marching(bounded, **kwargs)
    except Exception as error:
        info["replay_error"] = f"{type(error).__name__}: {error}"
        return PRTransverseStreamingResult(first, None, False, "replay_failed", info)
    info["replay_performed"] = True
    info["replay_accepted_cells"] = second.completed_intervals
    info["field_max_abs_difference"] = float(np.max(np.abs(first.A_final-second.A_final)))
    checks = dict(replay_converged=succeeded(second),
                  original_launch=np.array_equal(first.A_initial,second.A_initial),
                  request_identity=first.resolved_profile["request_fingerprint"] == second.resolved_profile["request_fingerprint"],
                  field=np.allclose(first.A_final,second.A_final,rtol=replay_rtol,atol=replay_atol),
                  material_moments=True, convergence_decisions=True)
    one, two = first.diagnostics["planes"], second.diagnostics["planes"]
    checks["material_moments"] = len(one)==len(two)==target and all(
        np.allclose(a["moments"][key],b["moments"][key],rtol=replay_rtol,atol=replay_atol)
        for a,b in zip(one,two) for key in a["moments"])
    checks["convergence_decisions"] = len(one)==len(two)==target and all(
        [(h['newton_iteration'],h['pcg_iterations'],h['backtracks'],h['step_scale'],h['accepted']) for h in a['material_history']]
        == [(h['newton_iteration'],h['pcg_iterations'],h['backtracks'],h['step_scale'],h['accepted']) for h in b['material_history']]
        for a,b in zip(one,two))
    info["replay_checks"] = {k:bool(v) for k,v in checks.items()}
    ok = all(checks.values())
    return PRTransverseStreamingResult(first, second, ok,
        "prefix_converged" if ok and target<nz else "converged" if ok else "replay_failed", info)


__all__ = ["PR_TRANSVERSE_STREAMING_V1", "TransversePhaseBinding",
           "PRTransverseStreamingResult", "transverse_request_from_reference",
           "run_pr_transverse_streaming"]
