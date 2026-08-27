from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.optics.launch import build_launch, channel_power_integrals
from lcprop.optics.screens import (
    ChannelLaunchElements,
    IntensityRasterScreen,
    RasterSource,
    ScreenPlacement,
    apply_passive_field_transmittance,
    intensity_transmission_to_field_transmittance,
    prepare_intensity_raster_screen,
)
from lcprop.pr.image_amplification import prepare_image_transmission
from lcprop.pr.image_sources import PRImageSource


def _grid():
    return make_grid(
        GridSpec(
            Nx=16,
            Ny=8,
            x_aperture_um=16.0,
            y_aperture_um=8.0,
            z_length_um=10.0,
            dz_um=1.0,
        ),
        real_dtype=np.float64,
    )


def _beams():
    return BeamStack(
        channels=(
            BeamChannel(name="one", power_mW=3.0, waist_x_um=3.0, waist_y_um=2.0),
            BeamChannel(name="two", power_mW=1.0, waist_x_um=3.0, waist_y_um=2.0),
        ),
        coherence="coherent",
    )


def _screen(samples, *, width_um=14.0, height_um=6.0, boundary="reject"):
    return IntensityRasterScreen(
        source=RasterSource.from_array(np.asarray(samples, dtype=float)),
        placement=ScreenPlacement(
            center_x_um=0.0,
            center_y_um=0.0,
            width_um=width_um,
            height_um=height_um,
            boundary_policy=boundary,
        ),
    )


def test_no_screen_build_launch_is_bitwise_identical_and_reports_identity_power():
    grid = _grid()
    default = build_launch(_beams(), grid, complex_dtype=np.complex128)
    explicit = build_launch(
        _beams(),
        grid,
        complex_dtype=np.complex128,
        launch_elements=(),
    )

    assert np.array_equal(default.A0, explicit.A0)
    assert np.array_equal(default.physical_powers_mW, [3.0, 1.0])
    assert np.allclose(default.post_element_physical_powers_mW, [3.0, 1.0])
    assert default.post_element_total_power_mW == pytest.approx(4.0)
    assert np.allclose(default.channel_throughput_fractions, [1.0, 1.0])
    assert default.summary()["physical_channel_powers_mW"] == [3.0, 1.0]
    assert default.summary()["post_element_total_power_mW"] == pytest.approx(4.0)


def test_passive_identity_attenuation_opaque_and_nonuniform_power_semantics():
    field = np.array(
        [[1.0 + 2.0j, -0.5j], [0.25 - 0.75j, -2.0 + 0.5j]],
        dtype=np.complex128,
    )
    incident = np.sum(np.abs(field) ** 2)

    identity = apply_passive_field_transmittance(field, np.ones(field.shape))
    assert np.array_equal(identity, field)

    quarter = intensity_transmission_to_field_transmittance(
        np.full(field.shape, 0.25)
    )
    attenuated = apply_passive_field_transmittance(field, quarter)
    assert np.array_equal(attenuated, 0.5 * field)
    assert np.sum(np.abs(attenuated) ** 2) == pytest.approx(0.25 * incident)

    opaque = apply_passive_field_transmittance(field, np.zeros(field.shape))
    assert np.count_nonzero(opaque) == 0

    transmission = np.array([[0.0, 0.25], [0.5, 1.0]])
    nonuniform = apply_passive_field_transmittance(
        field,
        intensity_transmission_to_field_transmittance(transmission),
    )
    assert np.sum(np.abs(nonuniform) ** 2) == pytest.approx(
        np.sum(transmission * np.abs(field) ** 2)
    )


def test_pure_phase_and_general_complex_screens_are_passive():
    field = np.array([[1.0 + 1.0j, 2.0], [-0.5j, -3.0]], dtype=np.complex128)
    phase = np.array([[0.0, 0.4], [-1.2, 2.1]])
    phase_screen = np.exp(1j * phase)
    phase_output = apply_passive_field_transmittance(field, phase_screen)

    assert not np.array_equal(phase_output, field)
    assert np.allclose(np.abs(phase_output) ** 2, np.abs(field) ** 2)
    assert np.sum(np.abs(phase_output) ** 2) == pytest.approx(
        np.sum(np.abs(field) ** 2), rel=2e-15
    )

    amplitude = np.array([[0.0, 0.25], [0.7, 1.0]])
    complex_screen = amplitude * phase_screen
    output = apply_passive_field_transmittance(field, complex_screen)
    assert np.array_equal(output, complex_screen * field)
    with pytest.raises(ValueError, match="magnitude must not exceed one"):
        apply_passive_field_transmittance(field, np.full(field.shape, 1.01))


def test_channel_assignment_changes_only_selected_channel_and_reports_throughput():
    grid = _grid()
    incident = build_launch(_beams(), grid, complex_dtype=np.complex128)
    screen = _screen([[0.0, 1.0], [1.0, 0.0]])
    transformed = build_launch(
        _beams(),
        grid,
        complex_dtype=np.complex128,
        launch_elements=(
            ChannelLaunchElements(channel_index=1, elements=(screen,)),
        ),
    )

    assert np.array_equal(transformed.A0[0], incident.A0[0])
    assert not np.array_equal(transformed.A0[1], incident.A0[1])
    assert np.array_equal(transformed.physical_powers_mW, incident.physical_powers_mW)
    expected = channel_power_integrals(transformed.A0, grid) * 4.0
    assert np.allclose(transformed.post_element_physical_powers_mW, expected)
    assert transformed.channel_throughput_fractions[0] == pytest.approx(1.0)
    assert 0.0 < transformed.channel_throughput_fractions[1] < 1.0
    assert transformed.post_element_total_power_mW < 4.0


def test_ordered_elements_multiply_in_declared_order():
    grid = _grid()
    incident = build_launch(_beams(), grid, complex_dtype=np.complex128)
    first = _screen([[0.0, 1.0], [1.0, 0.25]])
    second = _screen([[1.0, 0.5], [0.25, 0.0]])
    transformed = build_launch(
        _beams(),
        grid,
        complex_dtype=np.complex128,
        launch_elements=(
            ChannelLaunchElements(channel_index=1, elements=(first, second)),
        ),
    )
    first_t = intensity_transmission_to_field_transmittance(
        prepare_intensity_raster_screen(first, grid)
    )
    second_t = intensity_transmission_to_field_transmittance(
        prepare_intensity_raster_screen(second, grid)
    )

    assert np.array_equal(
        transformed.A0[1],
        incident.A0[1] * first_t * second_t,
    )


@pytest.mark.parametrize("channel_index", [-1, 2, 3])
def test_assignment_index_is_resolved_against_canonical_enabled_channels(
    channel_index,
):
    with pytest.raises(ValueError, match="canonical enabled beam channel"):
        build_launch(
            _beams(),
            _grid(),
            launch_elements=(ChannelLaunchElements(channel_index=channel_index),),
        )


def test_assignment_index_follows_enabled_channel_order_after_adapter_filtering():
    launchplane_model = pytest.importorskip("launchplane.model")
    from lcprop.adapters.launchplane import beam_stack_definition_to_lcprop

    definition = launchplane_model.BeamStackDefinition(
        beams=(
            launchplane_model.BeamDefinition(
                name="first enabled",
                power_mW=3.0,
                enabled=True,
            ),
            launchplane_model.BeamDefinition(
                name="disabled middle",
                power_mW=9.0,
                enabled=False,
            ),
            launchplane_model.BeamDefinition(
                name="second enabled",
                power_mW=1.0,
                enabled=True,
            ),
        )
    )
    beams = beam_stack_definition_to_lcprop(definition)
    grid = _grid()
    incident = build_launch(beams, grid, complex_dtype=np.complex128)
    transformed = build_launch(
        beams,
        grid,
        complex_dtype=np.complex128,
        launch_elements=(
            ChannelLaunchElements(
                channel_index=1,
                elements=(_screen([[0.0, 1.0], [1.0, 0.0]]),),
            ),
        ),
    )

    assert tuple(channel.name for channel in beams.channels) == (
        "first enabled",
        "second enabled",
    )
    assert np.array_equal(transformed.A0[0], incident.A0[0])
    assert not np.array_equal(transformed.A0[1], incident.A0[1])


def test_duplicate_channel_assignments_are_rejected():
    assignment = ChannelLaunchElements(channel_index=0)
    with pytest.raises(ValueError, match="only one ordered element assignment"):
        build_launch(
            _beams(),
            _grid(),
            launch_elements=(assignment, assignment),
        )


def test_rectangular_source_is_resampled_to_rectangular_grid_without_selecting_it():
    source = RasterSource.from_array(np.ones((40, 120)))
    grid = _grid()
    screen = IntensityRasterScreen(
        source=source,
        placement=ScreenPlacement(
            width_um=8.0,
            height_um=4.0,
            boundary_policy="reject",
        ),
    )
    transmission = prepare_intensity_raster_screen(screen, grid)

    assert source.grayscale.shape == (40, 120)
    assert transmission.shape == (16, 8)
    assert (grid.Nx, grid.Ny) == (16, 8)
    assert (grid.spec.x_aperture_um, grid.spec.y_aperture_um) == (16.0, 8.0)


def test_default_boundary_policy_rejects_footprint_instead_of_clipping():
    screen = IntensityRasterScreen(
        source=RasterSource.from_array(np.eye(2)),
        placement=ScreenPlacement(
            center_x_um=7.5,
            center_y_um=0.0,
            width_um=4.0,
            height_um=4.0,
        ),
    )
    with pytest.raises(ValueError, match="footprint extends outside"):
        prepare_intensity_raster_screen(screen, _grid())


def test_pr_source_is_a_shared_raster_with_compatibility_policy():
    source = PRImageSource.from_array(np.eye(3))
    assert isinstance(source, RasterSource)
    assert source.preprocessing_policy == "pr_intensity_transparency_v1"
    assert not source.grayscale.flags.writeable


def test_pr_historical_preprocessing_wrapper_is_bitwise_shared_policy():
    grid = _grid()
    image = np.array([[0.0, 0.2, 1.0], [0.7, 0.4, 0.1]])
    placement = ScreenPlacement(
        center_x_um=-1.0,
        center_y_um=0.0,
        width_um=6.0,
        height_um=6.0,
        boundary_policy="reject",
    )
    shared = prepare_intensity_raster_screen(
        IntensityRasterScreen(
            source=RasterSource.from_array(image),
            placement=placement,
            invert=True,
        ),
        grid,
    )
    compatibility = prepare_image_transmission(
        image,
        grid,
        center_x_um=-1.0,
        center_y_um=0.0,
        physical_size_um=6.0,
        invert=True,
        require_full_footprint=True,
    )
    assert np.array_equal(shared, compatibility)


def test_shared_screen_module_has_no_material_imports():
    source = Path("src/lcprop/optics/screens.py").read_text(encoding="utf-8")
    assert "lcprop.pr" not in source
    assert "lcprop.lc" not in source
