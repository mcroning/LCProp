import json

import numpy as np
import pytest

from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.optics.launch import (
    OpticalLaunchContext,
    build_launch,
    channel_power_integrals,
    reconstructed_physical_powers_mW,
    total_power,
)
from lcprop.optics.splitstep import hop_linear, linear_kernel, total_intensity


def test_beam_channel_old_positional_constructor_remains_stable():
    channel = BeamChannel(
        "old",
        0.532,
        2.0,
        4.0,
        5.0,
        1.0,
        -2.0,
        0.1,
        -0.2,
        0.3,
        "laser",
    )

    assert channel.coherence_group == "laser"
    assert channel.profile == "legacy_gaussian"
    assert channel.waist_x_at_focus_um is None


@pytest.mark.parametrize("resolution", [64, 128])
def test_single_gaussian_launch_power_normalization(resolution):
    grid = make_grid(GridSpec(Nx=resolution, Ny=resolution, z_length_um=50.0))
    beams = BeamStack(channels=(BeamChannel(power_mW=2.0),))
    launch = build_launch(beams, grid)

    assert launch.A0.shape == (1, resolution, resolution)
    assert np.isclose(total_power(launch.A0, grid), 1.0, rtol=1e-6)
    assert np.allclose(launch.power_fractions, [1.0])
    assert np.allclose(launch.physical_powers_mW, [2.0])


def test_two_channel_launch_power_normalization():
    grid = make_grid(GridSpec(Nx=128, Ny=128, z_length_um=50.0))

    beams = BeamStack(
        channels=(
            BeamChannel(name="a", power_mW=1.0, y0_um=-5.0),
            BeamChannel(name="b", power_mW=3.0, y0_um=5.0),
        ),
        coherence="coherent",
    )

    launch = build_launch(beams, grid)

    assert launch.A0.shape == (2, 128, 128)
    assert launch.coherence == "coherent"
    assert launch.coherence_groups == ("__lcprop_coherent__", "__lcprop_coherent__")
    assert np.allclose(channel_power_integrals(launch.A0, grid), [0.25, 0.75])
    assert np.isclose(total_power(launch.A0, grid), 1.0, rtol=1e-6)
    assert np.allclose(launch.power_fractions, [0.25, 0.75])
    assert np.allclose(
        reconstructed_physical_powers_mW(launch.A0, grid, launch),
        [1.0, 3.0],
    )
    assert launch.summary()["physical_channel_powers_mW"] == [1.0, 3.0]
    assert launch.summary()["physical_total_power_mW"] == 4.0
    assert launch.summary()["power_fractions"] == [0.25, 0.75]
    assert launch.summary()["field_normalization"] == (
        "sum_channel_integrals_equals_one"
    )
    assert np.allclose(np.asarray(launch.wavelengths_um), [0.633, 0.633])


def test_legacy_incoherent_stack_migrates_to_distinct_groups():
    grid = make_grid(GridSpec(Nx=32, Ny=32))
    beams = BeamStack(channels=(BeamChannel(name="a"), BeamChannel(name="b")))

    launch = build_launch(beams, grid)

    assert len(set(launch.coherence_groups)) == 2
    assert launch.summary()["coherence"] == "incoherent"
    assert launch.summary()["coherence_groups"] == list(launch.coherence_groups)


@pytest.mark.parametrize(
    ("coherence", "expected"),
    (("incoherent", "incoherent"), ("coherent", "coherent")),
)
def test_legacy_coherence_summary_remains_backward_compatible(coherence, expected):
    grid = make_grid(GridSpec(Nx=32, Ny=32))
    beams = BeamStack(
        channels=(BeamChannel(name="a"), BeamChannel(name="b")),
        coherence=coherence,
    )

    launch = build_launch(beams, grid)

    assert launch.summary()["coherence"] == expected


@pytest.mark.parametrize("coherence", ["incoherent", "coherent"])
def test_single_channel_summary_preserves_legacy_label(coherence):
    grid = make_grid(GridSpec(Nx=32, Ny=32))
    launch = build_launch(
        BeamStack(channels=(BeamChannel(),), coherence=coherence),
        grid,
    )

    assert launch.summary()["coherence"] == coherence


def test_partially_coherent_explicit_groups_are_preserved_and_reported():
    grid = make_grid(GridSpec(Nx=32, Ny=32))
    beams = BeamStack(
        channels=(
            BeamChannel(name="a", coherence_group="A"),
            BeamChannel(name="b", coherence_group="A"),
            BeamChannel(name="c", coherence_group="B"),
        )
    )

    launch = build_launch(beams, grid)

    assert launch.coherence_groups == ("A", "A", "B")
    assert launch.summary()["coherence"] == "coherent"


def test_launch_summary_reports_effective_explicit_group_coherence():
    grid = make_grid(GridSpec(Nx=32, Ny=32))
    beams = BeamStack(
        channels=(
            BeamChannel(name="a", coherence_group="laser"),
            BeamChannel(name="b", coherence_group="laser"),
        )
    )

    launch = build_launch(beams, grid)

    assert launch.coherence == "incoherent"
    assert launch.coherence_groups == ("laser", "laser")
    assert launch.summary()["coherence"] == "coherent"


def test_distinct_explicit_groups_override_legacy_coherent_summary():
    grid = make_grid(GridSpec(Nx=32, Ny=32))
    beams = BeamStack(
        channels=(
            BeamChannel(name="a", coherence_group="A"),
            BeamChannel(name="b", coherence_group="B"),
        ),
        coherence="coherent",
    )

    launch = build_launch(beams, grid)

    assert launch.coherence == "coherent"
    assert launch.coherence_groups == ("A", "B")
    assert launch.summary()["coherence"] == "incoherent"


def test_launch_summary_is_deterministic_json_safe_and_numerically_non_mutating():
    grid = make_grid(GridSpec(Nx=32, Ny=24))
    launch = build_launch(
        BeamStack(
            channels=(
                BeamChannel(name="a", power_mW=1.0, coherence_group="laser"),
                BeamChannel(name="b", power_mW=2.0, coherence_group="laser"),
            )
        ),
        grid,
    )
    numerical_before = {
        "A0": launch.A0.copy(),
        "physical_powers_mW": launch.physical_powers_mW.copy(),
        "power_fractions": launch.power_fractions.copy(),
        "wavelengths_um": launch.wavelengths_um.copy(),
        "post_element_physical_powers_mW": (
            launch.post_element_physical_powers_mW.copy()
        ),
        "channel_throughput_fractions": launch.channel_throughput_fractions.copy(),
    }
    scalar_before = (
        launch.physical_total_power_mW,
        launch.post_element_total_power_mW,
    )

    first = launch.summary()
    second = launch.summary()
    round_tripped = json.loads(json.dumps(first))

    assert first == second
    assert round_tripped == first
    for name, before in numerical_before.items():
        np.testing.assert_array_equal(getattr(launch, name), before)
    assert (
        launch.physical_total_power_mW,
        launch.post_element_total_power_mW,
    ) == scalar_before


def test_partially_migrated_coherence_groups_are_rejected():
    beams = BeamStack(
        channels=(
            BeamChannel(name="legacy"),
            BeamChannel(name="explicit", coherence_group="laser"),
        )
    )

    with pytest.raises(ValueError, match="coherence_group migration is incomplete"):
        beams.validate()


def test_zero_power_channel_is_zero():
    grid = make_grid(GridSpec(Nx=32, Ny=32))
    beams = BeamStack(
        channels=(BeamChannel(power_mW=0.0), BeamChannel(power_mW=1.0))
    )

    launch = build_launch(beams, grid)

    assert np.allclose(launch.A0[0], 0.0)
    assert np.isclose(total_power(launch.A0, grid), 1.0)


def test_all_zero_power_channels_are_rejected():
    grid = make_grid(GridSpec(Nx=32, Ny=32))
    beams = BeamStack(channels=(BeamChannel(power_mW=0.0),))

    with pytest.raises(ValueError, match="total physical power must be positive"):
        build_launch(beams, grid)


def test_tilt_adds_phase_variation():
    grid = make_grid(GridSpec(Nx=64, Ny=64))

    no_tilt = BeamStack(channels=(BeamChannel(tilt_x_rad_per_um=0.0),))
    tilted = BeamStack(channels=(BeamChannel(tilt_x_rad_per_um=0.1),))

    A0 = build_launch(no_tilt, grid).A0[0]
    A1 = build_launch(tilted, grid).A0[0]

    assert np.allclose(np.abs(A0), np.abs(A1), rtol=1e-5, atol=1e-7)
    assert not np.allclose(A0, A1)


@pytest.mark.parametrize(
    ("field_name", "requested_gradient", "axis"),
    (
        ("tilt_x_rad_per_um", 0.12, 0),
        ("tilt_y_rad_per_um", -0.09, 1),
    ),
)
def test_gaussian_channel_phase_gradient_matches_requested_tilt(
    field_name, requested_gradient, axis
):
    grid = make_grid(
        GridSpec(
            Nx=128,
            Ny=128,
            x_aperture_um=80.0,
            y_aperture_um=80.0,
        ),
        real_dtype=np.float64,
    )
    channel = BeamChannel(
        waist_x_um=10.0,
        waist_y_um=10.0,
        **{field_name: requested_gradient},
    )
    field = build_launch(
        BeamStack(channels=(channel,)), grid, complex_dtype=np.complex128
    ).A0[0]

    # Measure phase differences only where both adjacent samples have useful
    # amplitude.  The chosen gradients are comfortably below Nyquist, so the
    # principal phase difference is unambiguous.
    if axis == 0:
        product = field[1:, :] * np.conj(field[:-1, :])
        adjacent_amplitude = np.minimum(np.abs(field[1:, :]), np.abs(field[:-1, :]))
        spacing = grid.dx_um
    else:
        product = field[:, 1:] * np.conj(field[:, :-1])
        adjacent_amplitude = np.minimum(np.abs(field[:, 1:]), np.abs(field[:, :-1]))
        spacing = grid.dy_um
    mask = adjacent_amplitude > 0.05 * np.max(np.abs(field))
    measured_gradient = np.median(np.angle(product[mask]) / spacing)

    assert measured_gradient == pytest.approx(requested_gradient, abs=1e-12)


def test_coherent_group_interference_does_not_renormalize_channel_fractions():
    grid = make_grid(GridSpec(Nx=128, Ny=128))
    in_phase = BeamStack(
        channels=(
            BeamChannel(power_mW=1.0, coherence_group="A"),
            BeamChannel(power_mW=1.0, coherence_group="A"),
        )
    )
    out_of_phase = BeamStack(
        channels=(
            BeamChannel(power_mW=1.0, coherence_group="A"),
            BeamChannel(power_mW=1.0, phase_rad=np.pi, coherence_group="A"),
        )
    )

    launch_in = build_launch(in_phase, grid)
    launch_out = build_launch(out_of_phase, grid)
    area = grid.dx_um * grid.dy_um
    grouped_in = total_intensity(
        launch_in.A0,
        coherence_groups=launch_in.coherence_groups,
    )
    grouped_out = total_intensity(
        launch_out.A0,
        coherence_groups=launch_out.coherence_groups,
    )

    assert np.allclose(channel_power_integrals(launch_in.A0, grid), [0.5, 0.5])
    assert np.allclose(channel_power_integrals(launch_out.A0, grid), [0.5, 0.5])
    assert np.isclose(np.sum(grouped_in) * area, 2.0, rtol=1e-6)
    assert np.sum(grouped_out) * area < 1e-12


def _second_moment_radii(field, grid):
    intensity = np.abs(field) ** 2
    x_marginal = np.sum(intensity, axis=1)
    y_marginal = np.sum(intensity, axis=0)
    x_mean = np.sum(x_marginal * grid.x_um) / np.sum(x_marginal)
    y_mean = np.sum(y_marginal * grid.y_um) / np.sum(y_marginal)
    return (
        2.0
        * np.sqrt(np.sum(x_marginal * (grid.x_um - x_mean) ** 2) / np.sum(x_marginal)),
        2.0
        * np.sqrt(np.sum(y_marginal * (grid.y_um - y_mean) ** 2) / np.sum(y_marginal)),
    )


@pytest.mark.parametrize("focus_z_um", [45.0, -45.0])
def test_focus_defined_circular_gaussian_reaches_requested_signed_focus(focus_z_um):
    grid = make_grid(
        GridSpec(Nx=256, Ny=256, x_aperture_um=160.0, y_aperture_um=160.0),
        real_dtype=np.float64,
    )
    channel = BeamChannel(
        profile="focused_gaussian",
        waist_x_at_focus_um=6.0,
        waist_y_at_focus_um=6.0,
        focus_z_um=focus_z_um,
    )
    context = OpticalLaunchContext(grid=grid, n_ref=2.4, interaction_length_um=90.0)
    entrance = build_launch(
        BeamStack(channels=(channel,)),
        grid,
        complex_dtype=np.complex128,
        context=context,
    ).A0
    propagated = hop_linear(
        entrance,
        linear_kernel(
            grid.fxy2_um,
            dz=focus_z_um,
            wavelength=channel.wavelength_um,
            n_ref=context.n_ref,
        ),
    )

    assert _second_moment_radii(propagated[0], grid) == pytest.approx(
        (6.0, 6.0), abs=2e-9
    )


def test_elliptical_gaussian_and_midpoint_convenience_reach_axis_waists():
    grid = make_grid(
        GridSpec(Nx=384, Ny=384, x_aperture_um=192.0, y_aperture_um=192.0),
        real_dtype=np.float64,
    )
    channel = BeamChannel(
        profile="focused_gaussian",
        waist_x_at_focus_um=5.0,
        waist_y_at_focus_um=9.0,
        focus_at_interaction_midpoint=True,
    )
    context = OpticalLaunchContext(grid=grid, n_ref=1.7, interaction_length_um=80.0)
    entrance = build_launch(
        BeamStack(channels=(channel,)),
        grid,
        complex_dtype=np.complex128,
        context=context,
    ).A0
    propagated = hop_linear(
        entrance,
        linear_kernel(
            grid.fxy2_um,
            dz=40.0,
            wavelength=channel.wavelength_um,
            n_ref=context.n_ref,
        ),
    )

    assert context.resolved_focus_z_um(channel) == 40.0
    assert _second_moment_radii(propagated[0], grid) == pytest.approx(
        (5.0, 9.0), abs=3e-9
    )


def test_elliptical_gaussian_reaches_requested_negative_focus():
    grid = make_grid(
        GridSpec(Nx=384, Ny=384, x_aperture_um=192.0, y_aperture_um=192.0),
        real_dtype=np.float64,
    )
    channel = BeamChannel(
        profile="focused_gaussian",
        waist_x_at_focus_um=5.0,
        waist_y_at_focus_um=9.0,
        focus_z_um=-37.0,
    )
    context = OpticalLaunchContext(grid=grid, n_ref=1.7, interaction_length_um=80.0)
    entrance = build_launch(
        BeamStack(channels=(channel,)),
        grid,
        complex_dtype=np.complex128,
        context=context,
    ).A0
    propagated = hop_linear(
        entrance,
        linear_kernel(
            grid.fxy2_um,
            dz=-37.0,
            wavelength=channel.wavelength_um,
            n_ref=context.n_ref,
        ),
    )

    assert context.resolved_focus_z_um(channel) == -37.0
    assert _second_moment_radii(propagated[0], grid) == pytest.approx(
        (5.0, 9.0), abs=3e-9
    )


def test_focus_respects_negative_propagation_convention():
    grid = make_grid(
        GridSpec(Nx=256, Ny=256, x_aperture_um=160.0, y_aperture_um=160.0),
        real_dtype=np.float64,
    )
    channel = BeamChannel(
        profile="focused_gaussian",
        waist_x_at_focus_um=6.0,
        waist_y_at_focus_um=6.0,
        focus_z_um=45.0,
    )
    context = OpticalLaunchContext(
        grid=grid,
        n_ref=2.4,
        interaction_length_um=90.0,
        propagation_sign=-1,
    )
    entrance = build_launch(
        BeamStack(channels=(channel,)),
        grid,
        complex_dtype=np.complex128,
        context=context,
    ).A0
    propagated = hop_linear(
        entrance,
        linear_kernel(
            grid.fxy2_um,
            dz=-45.0,
            wavelength=channel.wavelength_um,
            n_ref=context.n_ref,
        ),
    )

    assert _second_moment_radii(propagated[0], grid) == pytest.approx(
        (6.0, 6.0), abs=2e-9
    )


@pytest.mark.parametrize(
    "channel",
    (
        BeamChannel(profile="collimated_gaussian"),
        BeamChannel(profile="uniform"),
        BeamChannel(
            profile="focused_gaussian",
            waist_x_at_focus_um=4.0,
            waist_y_at_focus_um=7.0,
            focus_z_um=20.0,
        ),
    ),
)
def test_explicit_profile_modes_preserve_normalized_power(channel):
    grid = make_grid(GridSpec(Nx=96, Ny=80), real_dtype=np.float64)
    context = OpticalLaunchContext(grid=grid, n_ref=2.4, interaction_length_um=40.0)

    launch = build_launch(
        BeamStack(channels=(channel,)),
        grid,
        complex_dtype=np.complex128,
        context=context,
    )

    assert total_power(launch.A0, grid) == pytest.approx(1.0, abs=2e-15)


@pytest.mark.parametrize("profile", ["collimated_gaussian", "uniform"])
def test_collimated_and_uniform_launches_have_no_quadratic_phase(profile):
    grid = make_grid(
        GridSpec(Nx=64, Ny=48, x_aperture_um=64.0, y_aperture_um=48.0),
        real_dtype=np.float64,
    )
    field = build_launch(
        BeamStack(
            channels=(
                BeamChannel(profile=profile, waist_x_um=8.0, waist_y_um=7.0),
            )
        ),
        grid,
        complex_dtype=np.complex128,
    ).A0[0]

    assert np.max(np.abs(np.imag(field))) == 0.0
    if profile == "uniform":
        np.testing.assert_array_equal(np.abs(field), np.abs(field[0, 0]))
        assert field[0, 0] == field[-1, -1]


def test_uniform_launch_requires_exact_periodic_tilt_modes():
    grid = make_grid(GridSpec(Nx=32, Ny=24), real_dtype=np.float64)
    exact_qx = 2.0 * np.pi * 3.0 / grid.spec.x_aperture_um
    build_launch(
        BeamStack(
            channels=(BeamChannel(profile="uniform", tilt_x_rad_per_um=exact_qx),)
        ),
        grid,
        complex_dtype=np.complex128,
    )

    with pytest.raises(ValueError, match="exact periodic Fourier mode"):
        build_launch(
            BeamStack(
                channels=(
                    BeamChannel(profile="uniform", tilt_x_rad_per_um=0.9 * exact_qx),
                )
            ),
            grid,
            complex_dtype=np.complex128,
        )


def test_legacy_launch_is_bitwise_unchanged_when_context_is_supplied():
    grid = make_grid(GridSpec(Nx=64, Ny=48), real_dtype=np.float64)
    beams = BeamStack(
        channels=(
            BeamChannel(
                power_mW=2.0,
                waist_x_um=7.0,
                waist_y_um=9.0,
                tilt_x_rad_per_um=0.03,
                phase_rad=0.2,
            ),
        )
    )
    legacy = build_launch(beams, grid, complex_dtype=np.complex128)
    contextual = build_launch(
        beams,
        grid,
        complex_dtype=np.complex128,
        context=OpticalLaunchContext(grid=grid, n_ref=1.5, interaction_length_um=50.0),
    )

    np.testing.assert_array_equal(contextual.A0, legacy.A0)
    np.testing.assert_array_equal(contextual.power_fractions, legacy.power_fractions)
