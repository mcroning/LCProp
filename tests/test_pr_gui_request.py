from __future__ import annotations
import math

from dataclasses import replace
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from lcprop.adapters import (
    beam_stack_definition_to_lcprop,
    beam_stack_to_launchplane,
)
from lcprop.core.backend import BackendSpec
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.optics.boundaries import TransverseBoundarySpec
from lcprop.pr.gui.beam_panel import make_pr_beam_panel
from lcprop.pr.gui.evolution_panel import PREvolutionPanel
from lcprop.pr.gui.grid_panel import PR_DEFAULT_GRID, PRGridPanel
from lcprop.pr.gui.material_panel import PRMaterialPanel
from lcprop.pr.gui.request_adapter import (
    apply_pr_request,
    build_pr_request,
    validate_pr_gui_request,
    validate_pr_static_gui_request,
)
from lcprop.pr.specs import (
    PRMaterialSpec,
    PRRunRequest,
    PRSolverOptions,
    PR_TIMEDEPENDENT_WORKFLOW,
)
from lcprop.pr.specs import PR_EULER_INTEGRATOR, PR_SEMI_IMPLICIT_INTEGRATOR
from lcprop.pr.static_workflow import (
    PRStaticRunRequest,
    PR_STATIC_WORKFLOW,
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _controls(app):
    grid_panel = PRGridPanel()
    return (
        PRMaterialPanel(),
        make_pr_beam_panel(
            x_aperture_um=grid_panel.x_aperture_um.value(),
            y_aperture_um=grid_panel.y_aperture_um.value(),
        ),
        grid_panel,
        PREvolutionPanel(),
    )


def _build(controls):
    material_panel, beam_panel, grid_panel, evolution_panel = controls
    return build_pr_request(
        material_panel=material_panel,
        beam_panel=beam_panel,
        grid_panel=grid_panel,
        evolution_panel=evolution_panel,
    )


def _apply(request, controls) -> None:
    material_panel, beam_panel, grid_panel, evolution_panel = controls
    apply_pr_request(
        request,
        material_panel=material_panel,
        beam_panel=beam_panel,
        grid_panel=grid_panel,
        evolution_panel=evolution_panel,
    )


def test_pr_gui_defaults_are_pr_owned_valid_and_well_sampled(app):
    controls = _controls(app)

    request = _build(controls)
    preflight = validate_pr_gui_request(request)

    assert controls[-1].workflow_id() == PR_TIMEDEPENDENT_WORKFLOW
    assert request.grid == PR_DEFAULT_GRID
    assert request.grid != GridSpec()
    assert request.solver == PRSolverOptions(
        Nt=10,
        dt_normalized=1e-3,
        optical_substeps=1,
        integrator=PR_SEMI_IMPLICIT_INTEGRATOR,
    )
    assert request.backend == BackendSpec(
        backend="numpy",
        precision="float64",
        verbose=False,
    )
    assert request.initial_A is None
    assert request.initial_E is None
    assert preflight.warnings == ()
    assert request.solver.dt_normalized < preflight.conservative_dt_limit
    assert min(
        value
        for pair in preflight.aperture.samples_per_waist
        for value in pair
    ) >= 6.0
    for plane_margins in preflight.aperture.boundary_margins_um.values():
        assert min(
            margin
            for beam_margins in plane_margins
            for margin in beam_margins.values()
        ) > 0.0


def test_pr_gui_build_and_apply_preserve_selected_optical_boundary(app):
    source = _controls(app)
    sponge = TransverseBoundarySpec(
        mode="sponge",
        width_fraction=0.2,
        attenuation_per_um=0.075,
        profile_order=3,
    )
    source[1].set_optical_boundary(sponge)

    request = _build(source)
    restored = _controls(app)
    _apply(request, restored)

    assert request.optical_boundary == sponge
    assert restored[1].optical_boundary() == sponge


def test_static_selection_builds_only_static_semantics_and_auto_tolerances(app):
    controls = _controls(app)
    evolution_panel = controls[-1]
    evolution_panel.set_workflow_id(PR_STATIC_WORKFLOW)
    evolution_panel.max_coupled_passes.setValue(9)
    evolution_panel.optical_substeps.setValue(3)
    evolution_panel.precision.setCurrentText("float32")

    request = _build(controls)
    preflight = validate_pr_static_gui_request(request)

    assert isinstance(request, PRStaticRunRequest)
    assert request.solver.material_solver is None
    assert request.backend.precision == "float32"
    assert request.solver.max_coupled_passes == 9
    assert request.solver.optical_substeps == 3
    assert not hasattr(request.solver, "Nt")
    assert not hasattr(request.solver, "dt_normalized")
    assert not hasattr(request.solver, "integrator")
    assert preflight.warnings == ()
    assert evolution_panel.Nt.isHidden()
    assert evolution_panel.dt_normalized.isHidden()
    assert evolution_panel.integrator.isHidden()
    assert not evolution_panel.max_coupled_passes.isHidden()


def test_beam_stack_inverse_mapping_preserves_optical_fields_and_signs(app):
    stack = BeamStack(
        channels=(
            BeamChannel(
                name='positive-x',
                wavelength_um=0.532,
                power_mW=1.25,
                x0_um=-4.0,
                y0_um=3.0,
                phase_rad=0.7,
                coherence_group='laser-a',
                w1_um=18.0,
                w2_um=21.0,
                theta_ext_rad=math.asin(math.hypot(0.031, -0.047) * 0.532 / (2 * math.pi)),
                phi_rad=math.atan2(-0.047, 0.031) % (2 * math.pi),
            ),
            BeamChannel(
                name='negative-x',
                wavelength_um=0.532,
                power_mW=0.75,
                x0_um=5.0,
                y0_um=-2.0,
                phase_rad=-0.4,
                coherence_group='laser-a',
                w1_um=22.0,
                w2_um=19.0,
                theta_ext_rad=math.asin(math.hypot(-0.029, 0.041) * 0.532 / (2 * math.pi)),
                phi_rad=math.atan2(0.041, -0.029) % (2 * math.pi),
            ),
        ),
        coherence="incoherent",
    )

    definition = beam_stack_to_launchplane(stack)
    rebuilt = beam_stack_definition_to_lcprop(definition)

    assert len(definition.beams) == 2
    assert all(beam.enabled for beam in definition.beams)
    for actual, expected in zip(rebuilt.channels, stack.channels):
        assert actual.name == expected.name
        assert actual.wavelength_um == expected.wavelength_um
        assert actual.power_mW == expected.power_mW
        assert actual.w1_um == expected.w1_um
        assert actual.w2_um == expected.w2_um
        assert actual.x0_um == expected.x0_um
        assert actual.y0_um == expected.y0_um
        assert actual.tilt_x_rad_per_um == expected.tilt_x_rad_per_um
        assert actual.tilt_y_rad_per_um == expected.tilt_y_rad_per_um
        assert actual.phase_rad == expected.phase_rad
        assert actual.coherence_group == expected.coherence_group


def test_apply_and_rebuild_saved_pr_request_is_lossless(app):
    controls = _controls(app)
    request = PRRunRequest(
        grid=GridSpec(
            Nx=96,
            Ny=80,
            dz_um=4.5,
            x_aperture_um=240.0,
            y_aperture_um=180.0,
            z_length_um=90.0,
        ),
        beams=BeamStack(
            channels=(
                BeamChannel(
                    name='pump',
                    wavelength_um=0.633,
                    power_mW=1.2,
                    x0_um=-8.0,
                    y0_um=4.0,
                    phase_rad=0.25,
                    coherence_group='shared-laser',
                    w1_um=24.0,
                    w2_um=23.0,
                    theta_ext_rad=math.asin(math.hypot(0.021, -0.017) * 0.633 / (2 * math.pi)),
                    phi_rad=math.atan2(-0.017, 0.021) % (2 * math.pi),
                ),
                BeamChannel(
                    name='signal',
                    wavelength_um=0.633,
                    power_mW=0.8,
                    x0_um=7.0,
                    y0_um=-3.0,
                    phase_rad=-0.35,
                    coherence_group='shared-laser',
                    w1_um=25.0,
                    w2_um=22.0,
                    theta_ext_rad=math.asin(math.hypot(-0.019, 0.013) * 0.633 / (2 * math.pi)),
                    phi_rad=math.atan2(0.013, -0.019) % (2 * math.pi),
                ),
            ),
            coherence="incoherent",
        ),
        material=PRMaterialSpec(
            dark_intensity=0.15,
            uniform_background_intensity=0.04,
            applied_field=-0.6,
            gain_length_product=-0.25,
            refractive_index=2.35,
            relative_permittivity=2200.0,
            mobile_charge_density_m3=5.5e20,
            temperature_K=301.0,
            characteristic_wavenumber_per_um_override=0.1,
        ),
        solver=PRSolverOptions(
            Nt=7,
            dt_normalized=1e-4,
            optical_substeps=3,
            integrator=PR_SEMI_IMPLICIT_INTEGRATOR,
        ),
        backend=BackendSpec(
            backend="auto",
            precision="float32",
            verbose=False,
        ),
    )

    _apply(request, controls)
    rebuilt = _build(controls)

    assert controls[-1].workflow_id() == PR_TIMEDEPENDENT_WORKFLOW
    assert rebuilt == request
    assert rebuilt.beams.channels[0].tilt_x_rad_per_um > 0.0
    assert rebuilt.beams.channels[0].tilt_y_rad_per_um < 0.0
    assert rebuilt.beams.channels[1].tilt_x_rad_per_um < 0.0
    assert rebuilt.beams.channels[1].tilt_y_rad_per_um > 0.0


def test_optional_characteristic_wavenumber_none_round_trips(app):
    controls = _controls(app)
    request = _build(controls)
    assert request.material.characteristic_wavenumber_per_um_override is None

    _apply(request, controls)

    assert _build(controls).material == request.material


def test_pr_gui_integrator_selection_round_trips(app):
    controls = _controls(app)
    semi_implicit = _build(controls)
    assert semi_implicit.solver.integrator == PR_SEMI_IMPLICIT_INTEGRATOR

    explicit_euler = replace(
        semi_implicit,
        solver=replace(
            semi_implicit.solver,
            integrator=PR_EULER_INTEGRATOR,
        ),
    )
    _apply(explicit_euler, controls)

    assert _build(controls).solver.integrator == PR_EULER_INTEGRATOR


def test_pr_gui_preflight_rejects_multiple_wavelengths(app):
    request = _build(_controls(app))
    second = replace(
        request.beams.channels[0],
        name="other wavelength",
        wavelength_um=0.532,
        coherence_group="other-laser",
    )
    incompatible = replace(
        request,
        beams=BeamStack(
            channels=(request.beams.channels[0], second),
            coherence="incoherent",
        ),
    )

    with pytest.raises(ValueError, match="one shared wavelength"):
        validate_pr_gui_request(incompatible)


def test_pr_gui_preflight_reports_sampling_boundary_and_grating_risks(app):
    request = _build(_controls(app))
    first = replace(
        request.beams.channels[0],
        name='first',
        w1_um=1.0,
        w2_um=1.0,
        x0_um=99.0,
        coherence_group='coherent',
        theta_ext_rad=math.asin(math.hypot(0.0, request.beams.channels[0].tilt_y_rad_per_um) * request.beams.channels[0].wavelength_um / (2 * math.pi * request.beams.channels[0].n_ext)),
        phi_rad=math.atan2(request.beams.channels[0].tilt_y_rad_per_um, 0.0) % (2 * math.pi),
    )
    second = replace(
        first,
        name='second',
        x0_um=0.0,
        theta_ext_rad=math.asin(math.hypot(1.5, first.tilt_y_rad_per_um) * first.wavelength_um / (2 * math.pi * first.n_ext)),
        phi_rad=math.atan2(first.tilt_y_rad_per_um, 1.5) % (2 * math.pi),
    )
    risky = replace(
        request,
        beams=BeamStack(
            channels=(first, second),
            coherence="incoherent",
        ),
    )

    warnings = validate_pr_gui_request(risky).warnings

    assert "beam waist is inadequately sampled" in warnings
    assert "coherent-beam grating period is inadequately sampled" in warnings
    assert any("periodic boundary" in warning for warning in warnings)


@pytest.mark.parametrize("profile", ["focused_gaussian", "uniform"])
def test_pr_gui_preflight_rejects_unsupported_physical_profiles(app, profile):
    request = _build(_controls(app))
    channel = replace(request.beams.channels[0], profile=profile)
    with pytest.raises(ValueError, match="collimated physical Gaussian"):
        validate_pr_gui_request(replace(request, beams=BeamStack(channels=(channel,))))


def test_pr_gui_preflight_uses_implemented_timestep_guard(app):
    request = _build(_controls(app))
    unstable = replace(
        request,
        solver=replace(request.solver, dt_normalized=1.0),
    )

    with pytest.raises(ValueError, match="exceeds conservative PR limit"):
        validate_pr_gui_request(unstable)


def test_pr_gui_controls_reject_request_owned_initial_arrays(app):
    controls = _controls(app)
    request = _build(controls)
    with_initial_state = replace(
        request,
        initial_A=np.zeros(
            (len(request.beams.channels), request.grid.Nx, request.grid.Ny),
            dtype=np.complex128,
        ),
    )

    with pytest.raises(ValueError, match="load a PR checkpoint"):
        _apply(with_initial_state, controls)


@pytest.mark.parametrize("algorithm", ["canonical_phase_slabs_v1", "canonical_phase_slabs_v2_cross_backend"])
def test_reduced_static_scattering_gui_experiment_transport_roundtrip(app, algorithm, tmp_path):
    from lcprop.pr.scattering import PRCanonicalScatteringSpec
    from lcprop.pr.experiment_codec import encode_pr_static_request, decode_pr_static_request
    from lcprop.pr.static_transport_codec import encode_pr_static_transport_request, decode_pr_static_transport_request
    controls = _controls(app)
    controls[-1].set_workflow_id(PR_STATIC_WORKFLOW)
    spec = PRCanonicalScatteringSpec(0.02, 0.4, 304001242, 1.0, algorithm)
    controls[-1].set_scattering_spec(spec)
    assert not controls[-1].scattering_enabled.isHidden()
    request = _build(controls)
    assert request.scattering == spec
    encoded = encode_pr_static_request(request)
    restored = decode_pr_static_request(encoded)
    from lcprop.persistence import save_experiment, load_experiment
    path = tmp_path / "scattering.json"
    from lcprop.pr.specs import PR_MATERIAL_ID
    save_experiment(restored, path, material_id=PR_MATERIAL_ID, workflow_id=PR_STATIC_WORKFLOW)
    restored = load_experiment(path).request
    transport = encode_pr_static_transport_request(restored)
    restored = decode_pr_static_transport_request(transport.payload.metadata, transport.payload.arrays)
    assert restored.scattering == spec
    _apply(restored, controls)
    assert _build(controls).scattering == spec
    # Existing saved experiments and transport requests decode as disabled.
    encoded.pop("scattering")
    assert decode_pr_static_request(encoded).scattering is None
    transport.payload.metadata.pop("scattering")
    assert decode_pr_static_transport_request(transport.payload.metadata, transport.payload.arrays).scattering is None


def test_reduced_static_gui_rejects_non_aligned_scattering(app):
    from lcprop.pr.scattering import PRCanonicalScatteringSpec
    controls = _controls(app)
    controls[-1].set_workflow_id(PR_STATIC_WORKFLOW)
    request = _build(controls)
    request = replace(request, scattering=PRCanonicalScatteringSpec(
        0.02, 0.4, 127, request.grid.z_length_um * 2,
    ))
    with pytest.raises(ValueError, match="align"):
        validate_pr_static_gui_request(request)
