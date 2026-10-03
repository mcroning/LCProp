"""Unregistered headless unified published-order Static march (M4 only).

Requires an explicit backend-resident prepared launch; does not build GUI
requests, register dispatch, encode results, or offer continuation. All arrays
returned are runtime backend arrays, never implicit host presentation payloads.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from typing import Any

from lcprop.core.backend import BackendSpec, get_backend
from lcprop.core.beams import normalize_coherence_groups
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.farfield import direction_cosine_spectrum
from lcprop.optics.splitstep import (
    apply_response_screen_inplace, hop_linear_inplace, scalar_angular_spectrum_kernel,
)
from lcprop.pr.optical_response import delta_n_from_E
from lcprop.pr.scattering import (
    PR_CANONICAL_SCATTERING_V2, PRCanonicalScatteringSpec,
    canonical_scattering_phase_increment, canonical_slab_range,
)
from lcprop.pr.source import channel_peak_intensity_reference, pr_driving_intensity
from lcprop.pr.specs import PRMaterialSpec
from .projection import PROJECTION_ID, electric_field_face, electric_field_optical_node
from .specs import (
    A7_CURRENT, MATERIAL_ID, PRMaterialPrecisionSpec, PRElectricalClosureSpec,
    PRUnifiedSpatialSpec,
)
from .state import PRTransportIntensity
from .static import MAX_CONNECTED_NODES, solve_static_material


WORKFLOW_ID = "pr_static_unified_published_optical_first_v1"
ARITHMETIC_ID = "per_cell_full_linear_arriving_unified_material_full_phase_scattering_v1"


@dataclass(frozen=True)
class UnifiedStaticRequest:
    """Borrow initial_A until synchronous launch acceptance copies it.

    The caller must not concurrently mutate the launch. Its dtype/device must
    already match execution: no hidden upload, normalization or dtype coercion.
    Spatial lengths must use the material's normalized coordinate scale.
    Electrical closure is authoritative; legacy applied_field is either zero
    or, for explicit A7 only, the identical reservoir field.
    """
    grid: GridSpec
    spatial: PRUnifiedSpatialSpec
    closure: PRElectricalClosureSpec
    initial_A: Any
    material: PRMaterialSpec = PRMaterialSpec()
    precision: PRMaterialPrecisionSpec = PRMaterialPrecisionSpec()
    backend: str = "numpy"
    wavelength_um: float = .633
    coherence_groups: tuple[str, ...] | None = None
    scattering: PRCanonicalScatteringSpec | None = None
    workflow_identity: str = WORKFLOW_ID
    arithmetic_identity: str = ARITHMETIC_ID
    projection_identity: str = PROJECTION_ID


@dataclass(frozen=True)
class UnifiedProductSelection:
    """Bounded outputs only: optional launch, cuts, and last material plane.

    Accepted endpoint is always retained. No longitudinal material volumes.
    Cuts are raw complex optical samples, not normalized display products.
    Native face flux products and preview schemas are deferred.
    """
    launch: bool = False
    boundary_cuts: bool = False
    material_state: bool = False
    carrier: bool = False
    electric_faces: tuple[str, ...] = ()
    electric_nodes: tuple[str, ...] = ()
    far_field: bool = False


@dataclass(frozen=True)
class UnifiedStaticResult:
    status: str
    reason: str
    completed_cells: int
    reached_z_um: float
    boundary_field: Any | None
    material_state: Any | None
    boundary_z_um: tuple[float, ...]
    material_z_um: tuple[float, ...]
    ledger: tuple[dict, ...]
    products: dict
    identities: dict
    failure: dict | None


def _finite(a, xp, name):
    if not bool(xp.all(xp.isfinite(a)).item()):
        raise ValueError(f"nonfinite {name}")


def _validate(r, selection):
    if (r.workflow_identity != WORKFLOW_ID or r.arithmetic_identity != ARITHMETIC_ID
            or r.projection_identity != PROJECTION_ID):
        raise ValueError("unknown unified workflow/arithmetic/projection identity")
    r.grid.validate(); r.spatial.validate(); r.precision.validate()
    r.material.validate(); r.closure.validate_spatial(r.spatial)
    if r.backend not in ("numpy", "cupy"):
        raise ValueError("explicit backend required; no scientific fallback")
    if not math.isfinite(r.wavelength_um) or r.wavelength_um <= 0:
        raise ValueError("positive finite wavelength required")
    if r.spatial.field_shape != (r.grid.Nx, r.grid.Ny):
        raise ValueError("optical plane and active/batch material shape disagree")
    lengths = (r.material.characteristic_wavenumber_per_um*r.grid.x_aperture_um,)
    if r.spatial.dimension == 2:
        lengths += (r.material.characteristic_wavenumber_per_um*r.grid.y_aperture_um,)
        if math.prod(r.spatial.active_shape) > MAX_CONNECTED_NODES:
            raise ValueError("connected material plane exceeds bounded direct-solver scope")
    if r.spatial.normalized_lengths != lengths:
        raise ValueError("material/optical normalized lengths disagree")
    expected_bias = r.closure.reservoir_field if r.closure.identity == A7_CURRENT else 0.
    if r.material.applied_field != expected_bias:
        raise ValueError("legacy applied field conflicts with explicit electrical closure")
    if r.closure.identity == A7_CURRENT and r.closure.background_intensity != r.material.background_intensity:
        raise ValueError("A7 reservoir and transport backgrounds disagree")
    for components in (selection.electric_faces, selection.electric_nodes):
        if len(set(components)) != len(components) or any(c not in r.spatial.active_axes for c in components):
            raise ValueError("invalid requested active field components")
    length, h = float(r.grid.z_length_um), float(r.grid.dz_um)
    count = round(length/h)
    if count < 1 or abs(count*h-length) > 2*max(math.ulp(length), math.ulp(count*h)):
        raise ValueError("domain must contain an integral number of requested cells")
    if r.scattering is not None:
        r.scattering.validate()
        if r.scattering.algorithm_version != PR_CANONICAL_SCATTERING_V2:
            raise ValueError("canonical V2 scattering required")
        whole = canonical_slab_range(r.scattering, z_start_um=0., dz_um=length, z_length_um=length)
        stop = whole.start
        for k in range(count):
            slabs = canonical_slab_range(r.scattering, z_start_um=k*h, dz_um=h, z_length_um=length)
            if slabs.start != stop:
                raise ValueError("scattering slab gap or overlap")
            stop = slabs.stop
        if stop != whole.stop:
            raise ValueError("scattering/domain endpoint disagreement")
    return count


def _material_gate(state, diagnostics):
    """Require the M3 reported physical gates, not merely M1 structure."""
    state.validate_structure(); diagnostics.validate_structure()
    if diagnostics.status != "reported":
        raise ValueError("material diagnostics missing")
    obs, limits = dict(diagnostics.observations), dict(diagnostics.limits)
    prefixes = ([f"column_{j}." for j in range(state.spatial.batch_shape[0])]
                if state.spatial.batch_shape else
                (["column_0."] if state.spatial.dimension == 1 else [""]))
    for prefix in prefixes:
        for key in ("gauss_rms", "gauss_max", "flux_divergence_rms", "flux_divergence_max",
                    "closure_rms", "closure_max", "neutrality", "gauge"):
            name = prefix+key
            if name not in limits or name not in obs or abs(obs[name]) > limits[name]:
                raise ValueError(f"material gate failed: {name}")
        if (obs.get(prefix+"finite") != 1 or obs.get(prefix+"converged") != 1
                or obs.get(prefix+"carrier_min", 0) <= 0):
            raise ValueError("material finite/positive/convergence gate failed")
    for name, limit in limits.items():
        if name not in obs or abs(obs[name]) > limit:
            raise ValueError(f"material gate failed: {name}")
    # Drop Newton histories from the per-cell ledger; retain named final gates.
    return {k: v for k, v in obs.items() if "iteration_" not in k}, limits


def _cuts(A, ix, iy):
    return A[:, :, iy].copy(), A[:, ix, :].copy()


def _prepare_acceptance(ledger, record, cuts, new_cut):
    """All fallible bookkeeping precedes the single accepted-state replacement."""
    return ledger+(record,), cuts+((new_cut,) if new_cut is not None else ())


def run_unified_static(request, *, selection=UnifiedProductSelection(), cancellation_token=None):
    """Run fresh prepared launch; return truthful last accepted state on failure.

    Cancellation is checked before a cell and after its complete candidate and
    bookkeeping, before acceptance. Material solves are indivisible M3 calls.
    No progress/array callback, replay, continuation or external observer exists.
    """
    count = _validate(request, selection)  # Invalid metadata raises before allocation.
    r = request
    backend = get_backend(BackendSpec(r.backend, r.precision.state_dtype, False))
    xp = backend.xp
    A = r.initial_A
    if (not isinstance(A, xp.ndarray) or A.ndim != 3 or A.shape[0] < 1
            or A.shape[1:] != r.spatial.field_shape or A.dtype != xp.dtype(backend.complex_dtype)):
        raise ValueError("prepared launch must match execution backend, shape and dtype")
    context = A.device if r.backend == "cupy" else nullcontext()
    with context:
        return _run(r, selection, count, backend, cancellation_token)


def _run(r, selection, count, backend, token):
    xp = backend.xp
    accepted = material_state = launch = None
    completed = 0
    reached = 0.
    ledger = cuts = ()
    products = {}
    status, reason, failure = "completed", "all cells accepted", None
    stage, attempted = "launch", None
    identities = dict(workflow=r.workflow_identity, arithmetic=r.arithmetic_identity,
        projection=r.projection_identity, material=MATERIAL_ID,
        spatial=asdict(r.spatial), closure=asdict(r.closure), precision=asdict(r.precision),
        backend=r.backend, electro_optic="existing_scalar_x_delta_n",
        optical_boundary="periodic", wavelength_um=r.wavelength_um,
        material_parameters=asdict(r.material), grid=asdict(r.grid),
        scattering=asdict(r.scattering) if r.scattering else None)
    cancelled = lambda: token is not None and token.is_cancelled()
    try:
        grid = make_grid(r.grid, xp=xp, real_dtype=backend.real_dtype)
        groups = normalize_coherence_groups(r.initial_A.shape[0], coherent=False,
                                             coherence_groups=r.coherence_groups)
        identities['coherence_groups'] = groups
        candidate = r.initial_A.copy()
        _finite(candidate, xp, "launch")
        peak = channel_peak_intensity_reference(candidate, xp=xp)
        if not math.isfinite(peak):
            raise ValueError("nonfinite launch reference")
        identities['peak_intensity_reference'] = peak
        ix, iy = int(xp.argmin(abs(grid.x_um)).item()), int(xp.argmin(abs(grid.y_um)).item())
        initial_cut = (_cuts(candidate, ix, iy),) if selection.boundary_cuts else ()
        launch_copy = candidate.copy() if selection.launch else None
        accepted, cuts, launch = candidate, initial_cut, launch_copy
        del candidate, launch_copy
        h, length = float(r.grid.dz_um), float(r.grid.z_length_um)
        kernel = scalar_angular_spectrum_kernel(grid.fxy2_um, dz=h,
            wavelength=r.wavelength_um, n_ref=r.material.refractive_index,
            complex_dtype=backend.complex_dtype, xp=xp)
        for k in range(count):
            if cancelled():
                status, reason = "cancelled", "cancelled before cell"
                break
            attempted = k
            stage = "optical_hop"
            candidate = accepted.copy()
            hop_linear_inplace(candidate, kernel, xp=xp)
            stage = "arriving_intensity"
            intensity = pr_driving_intensity(candidate, peak_intensity_reference=peak,
                background_intensity=r.material.background_intensity, coherence_groups=groups, xp=xp)
            transport = PRTransportIntensity(intensity, r.spatial, r.precision, r.backend,
                peak, r.material.dark_intensity, r.material.uniform_background_intensity)
            stage = "material_equilibrium"
            state, diag = solve_static_material(transport, closure=r.closure)
            observations, limits = _material_gate(state, diag)
            stage = "optical_projection"
            E = electric_field_optical_node(state, identity=r.projection_identity)
            _finite(E, xp, "optical field")
            stage = "material_phase"
            dn = delta_n_from_E(E, gain_length_product=r.material.gain_length_product,
                interaction_length_um=length, wavelength_um=r.wavelength_um)
            phase = xp.exp(1j*(2*math.pi/r.wavelength_um)*h*dn)
            _finite(phase, xp, "material phase")
            apply_response_screen_inplace(candidate, phase, xp=xp)
            del dn, phase, E
            stage = "scattering"
            slabs = None
            if r.scattering is not None:
                slab_range = canonical_slab_range(r.scattering, z_start_um=k*h, dz_um=h, z_length_um=length)
                slabs = (slab_range.start, slab_range.stop)
                noise = canonical_scattering_phase_increment(r.scattering,
                    z_start_um=k*h, dz_um=h, z_length_um=length,
                    Nx=grid.Nx, Ny=grid.Ny, x_aperture_um=r.grid.x_aperture_um,
                    y_aperture_um=r.grid.y_aperture_um, real_dtype=backend.real_dtype, xp=xp)
                _finite(noise, xp, "scattering phase")
                apply_response_screen_inplace(candidate, xp.exp(1j*noise), xp=xp)
                del noise
            _finite(candidate, xp, "candidate boundary")
            stage = "bookkeeping"
            end = length if k+1 == count else (k+1)*h
            record = dict(cell_index=k, z_start_um=k*h, z_end_um=end,
                material_z_um=end, optical_distance_um=h, material_weight_um=h,
                canonical_slabs=slabs, observations=observations, limits=limits)
            new_cut = _cuts(candidate, ix, iy) if selection.boundary_cuts else None
            next_ledger, next_cuts = _prepare_acceptance(ledger, record, cuts, new_cut)
            if cancelled():
                status, reason = "cancelled", "cancelled before candidate acceptance"
                break
            accepted, material_state, completed, reached, ledger, cuts = (
                candidate, state, k+1, end, next_ledger, next_cuts)
            del candidate, state, transport, intensity, diag, next_ledger, next_cuts
            attempted = None
    except Exception as exc:
        status, reason = "failed", str(exc)
        failure = dict(stage=stage, cell_index=attempted, type=type(exc).__name__, reason=str(exc))
    # Postprocessing uses only the accepted boundary/material state, never a
    # rejected candidate. Product failure cannot roll science forward/backward.
    try:
        if accepted is not None:
            products = dict(launch=launch, boundary_cuts=cuts,
                cut_x_index=ix, cut_y_index=iy, x_um=grid.x_um, y_um=grid.y_um)
            if selection.far_field:
                spectrum = direction_cosine_spectrum(accepted, dx_um=grid.dx_um, dy_um=grid.dy_um,
                    wavelength_um=r.wavelength_um, refractive_index=r.material.refractive_index,
                    coherence_groups=groups, xp=xp)
                for value in (spectrum.intensity, spectrum.s_x, spectrum.s_y):
                    _finite(value, xp, "final spectrum")
                products['far_field'] = spectrum
                products['far_field_z_um'] = reached
            if material_state is not None:
                if selection.carrier:
                    products['carrier_node'] = xp.exp(material_state.q)
                for c in selection.electric_faces:
                    products[f'electric_field_{c}_face'] = electric_field_face(material_state, component=c)
                for c in selection.electric_nodes:
                    products[f'electric_field_{c}_optical_node'] = electric_field_optical_node(material_state, component=c)
    except Exception as exc:
        if failure is None:
            failure = dict(stage="products", cell_index=None, type=type(exc).__name__, reason=str(exc))
        else:
            failure['product_failure'] = str(exc)
        status, reason = "failed", failure['reason']
    boundaries = ((0.,)+tuple(v['z_end_um'] for v in ledger)) if accepted is not None else ()
    return UnifiedStaticResult(status, reason, completed, reached, accepted,
        material_state if selection.material_state else None, boundaries,
        tuple(v['material_z_um'] for v in ledger), ledger, products, identities, failure)
