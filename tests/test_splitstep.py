import numpy as np
import pytest

from lcprop.core.context import GridSpec
from lcprop.core.grid import make_grid
from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.optics.launch import build_launch, total_power
from lcprop.optics.splitstep import (
    as_channel_stack,
    total_intensity,
    weighted_theta_intensity,
    linear_kernel,
    hop_linear,
    nonlinear_phase,
    advance_prepared_response,
    advance_slice,
    advance_slice_with_midintensity,
)


def test_as_channel_stack_promotes_2d():
    A = np.ones((8, 10), dtype=np.complex64)
    B = as_channel_stack(A)
    assert B.shape == (1, 8, 10)


def test_total_intensity_incoherent_vs_coherent():
    A = np.ones((2, 4, 5), dtype=np.complex64)

    I_incoh = total_intensity(A, coherent=False)
    I_coh = total_intensity(A, coherent=True)

    assert np.allclose(I_incoh, 2.0)
    assert np.allclose(I_coh, 4.0)


def test_one_coherence_group_reproduces_legacy_coherent_intensity():
    A = np.asarray(
        [
            [[1.0 + 2.0j, 0.5 - 1.0j]],
            [[-0.25 + 0.5j, 2.0 + 0.25j]],
        ]
    )

    grouped = total_intensity(A, coherence_groups=("laser", "laser"))
    legacy = total_intensity(A, coherent=True)

    assert np.allclose(grouped, legacy)


def test_explicit_identical_groups_override_legacy_incoherent_stack_flag():
    stack = BeamStack(
        channels=(
            BeamChannel(name="a", coherence_group="A"),
            BeamChannel(name="b", coherence_group="A"),
        ),
        coherence="incoherent",
    )
    A = np.ones((2, 2, 2), dtype=np.complex64)

    assert stack.coherence_groups == ("A", "A")
    assert np.allclose(
        total_intensity(A, coherence_groups=stack.coherence_groups),
        4.0,
    )


def test_distinct_coherence_groups_reproduce_legacy_incoherent_intensity():
    A = np.asarray(
        [
            [[1.0 + 2.0j, 0.5 - 1.0j]],
            [[-0.25 + 0.5j, 2.0 + 0.25j]],
        ]
    )

    grouped = total_intensity(A, coherence_groups=("laser_a", "laser_b"))
    legacy = total_intensity(A, coherent=False)

    assert np.allclose(grouped, legacy)


def test_mixed_grouped_intensity_matches_explicit_formula():
    A = np.asarray(
        [
            [[1.0 + 1.0j, 2.0 - 0.5j]],
            [[0.5 - 0.25j, -1.0 + 2.0j]],
            [[3.0 + 0.5j, 0.25 + 0.75j]],
        ]
    )

    actual = total_intensity(A, coherence_groups=("A", "A", "B"))
    expected = np.abs(A[0] + A[1]) ** 2 + np.abs(A[2]) ** 2

    assert np.allclose(actual, expected)


def test_grouped_intensity_is_independent_of_channel_order():
    A = np.asarray(
        [
            [[1.0 + 1.0j, 2.0 - 0.5j]],
            [[0.5 - 0.25j, -1.0 + 2.0j]],
            [[3.0 + 0.5j, 0.25 + 0.75j]],
        ]
    )
    groups = ("A", "A", "B")
    permutation = [2, 0, 1]

    original = total_intensity(A, coherence_groups=groups)
    reordered = total_intensity(
        A[permutation],
        coherence_groups=tuple(groups[index] for index in permutation),
    )

    assert np.allclose(reordered, original)


def test_single_channel_grouped_behavior_is_unchanged():
    A = np.asarray([[[1.0 + 2.0j, -0.5j]]])

    expected = np.abs(A[0]) ** 2

    assert np.allclose(total_intensity(A), expected)
    assert np.allclose(total_intensity(A, coherent=True), expected)
    assert np.allclose(total_intensity(A, coherence_groups=("laser",)), expected)


def test_weighted_theta_intensity():
    A = np.ones((2, 4, 5), dtype=np.complex64)
    weights = np.asarray([1.0, 3.0], dtype=np.float32)

    I = weighted_theta_intensity(A, weights)

    assert np.allclose(I, 4.0)


def test_weighted_grouped_theta_intensity_applies_one_weight_per_group():
    A = np.ones((3, 2, 2), dtype=np.complex64)
    weights = np.asarray([2.0, 2.0, 3.0], dtype=np.float32)

    actual = weighted_theta_intensity(
        A,
        weights,
        coherence_groups=("A", "A", "B"),
    )

    assert np.allclose(actual, 2.0 * np.abs(A[0] + A[1]) ** 2 + 3.0 * np.abs(A[2]) ** 2)


def test_weighted_grouped_theta_intensity_rejects_unequal_intragroup_weights():
    A = np.ones((2, 2, 2), dtype=np.complex64)
    weights = np.asarray([1.0, 2.0], dtype=np.float32)

    with pytest.raises(ValueError) as exc_info:
        weighted_theta_intensity(
            A,
            weights,
            coherence_groups=("laser", "laser"),
        )

    assert str(exc_info.value) == (
        "theta_weights must be equal within coherent group 'laser'; got [1.0, 2.0]"
    )


def test_linear_hop_preserves_power():
    grid = make_grid(GridSpec(Nx=64, Ny=64, z_length_um=50.0))
    beams = BeamStack(channels=(BeamChannel(power_mW=1.0),))
    launch = build_launch(beams, grid)

    A0 = launch.A0.copy()
    p0 = total_power(A0, grid)

    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um,
        wavelength=0.633,
        n_ref=1.5,
    )

    A1 = hop_linear(A0, kernel)
    p1 = total_power(A1, grid)

    assert np.isclose(p1, p0, rtol=1e-5)


@pytest.mark.parametrize(
    ("angle_x_rad", "angle_y_rad"),
    ((0.0, 0.0), (0.02, 0.0), (0.0, 0.02)),
)
def test_linear_propagation_centroid_follows_geometric_angle(
    angle_x_rad, angle_y_rad
):
    wavelength_um = 0.633
    n_medium = 1.5
    z_um = 100.0
    grid = make_grid(
        GridSpec(
            Nx=256,
            Ny=256,
            x_aperture_um=200.0,
            y_aperture_um=200.0,
            z_length_um=z_um,
            dz_um=z_um,
        ),
        real_dtype=np.float64,
    )
    k_medium_rad_per_um = 2.0 * np.pi * n_medium / wavelength_um
    channel = BeamChannel(
        wavelength_um=wavelength_um,
        waist_x_um=10.0,
        waist_y_um=10.0,
        tilt_x_rad_per_um=k_medium_rad_per_um * np.sin(angle_x_rad),
        tilt_y_rad_per_um=k_medium_rad_per_um * np.sin(angle_y_rad),
    )
    field = build_launch(
        BeamStack(channels=(channel,)), grid, complex_dtype=np.complex128
    ).A0
    kernel = linear_kernel(
        grid.fxy2_um,
        dz=z_um,
        wavelength=wavelength_um,
        n_ref=n_medium,
    )

    def centroid(A):
        intensity = np.abs(A[0]) ** 2
        total = intensity.sum()
        return (
            float((intensity.sum(axis=1) * grid.x_um).sum() / total),
            float((intensity.sum(axis=0) * grid.y_um).sum() / total),
        )

    x0, y0 = centroid(field)
    x1, y1 = centroid(hop_linear(field, kernel))

    # The propagator is paraxial (its exact result is z*sin(angle)); at these
    # small angles that agrees with the geometric z*tan(angle) expectation.
    assert x1 - x0 == pytest.approx(z_um * np.tan(angle_x_rad), abs=5e-4)
    assert y1 - y0 == pytest.approx(z_um * np.tan(angle_y_rad), abs=5e-4)


def test_nonlinear_phase_unit_magnitude():
    theta = np.ones((16, 16), dtype=np.float32) * 0.1

    phase = nonlinear_phase(
        theta,
        dz=5.0,
        wavelength=0.633,
        n_ref=1.5,
        ne=1.7,
        no=1.5,
    )

    assert np.allclose(np.abs(phase), 1.0)


def test_advance_slice_with_midintensity_shapes():
    grid = make_grid(GridSpec(Nx=32, Ny=32, z_length_um=50.0))
    beams = BeamStack(channels=(BeamChannel(power_mW=1.0),))
    launch = build_launch(beams, grid)

    theta = np.zeros((32, 32), dtype=np.float32)

    kernel = linear_kernel(
        grid.fxy2_um,
        dz=grid.dz_um,
        wavelength=0.633,
        n_ref=1.5,
    )

    A, I_before, I_after, I_mid = advance_slice_with_midintensity(
        launch.A0.copy(),
        theta,
        kernel=kernel,
        dz=grid.dz_um,
        wavelength=0.633,
        n_ref=1.5,
        ne=1.7,
        no=1.5,
    )

    assert A.shape == (1, 32, 32)
    assert I_before.shape == (32, 32)
    assert I_after.shape == (32, 32)
    assert I_mid.shape == (32, 32)


def test_lc_advance_wrapper_matches_prepared_response():
    rng = np.random.default_rng(23)
    A0 = (
        rng.normal(size=(2, 12, 10))
        + 1j * rng.normal(size=(2, 12, 10))
    ).astype(np.complex128)
    theta = np.linspace(0.05, 0.65, 120, dtype=np.float64).reshape(12, 10)
    fxy2 = rng.uniform(0.0, 0.2, size=(12, 10))
    dz = 7.5
    Nsub = 3
    wavelength = 0.633
    n_ref = 1.5
    kernel = linear_kernel(
        fxy2,
        dz=dz / Nsub,
        wavelength=wavelength,
        n_ref=n_ref,
    )
    half_step_response = nonlinear_phase(
        theta,
        dz=0.5 * dz / Nsub,
        wavelength=wavelength,
        n_ref=n_ref,
        ne=1.7,
        no=1.5,
    )

    wrapped = advance_slice(
        A0.copy(),
        theta,
        kernel=kernel,
        dz=dz,
        wavelength=wavelength,
        n_ref=n_ref,
        ne=1.7,
        no=1.5,
        Nsub=Nsub,
    )
    prepared = advance_prepared_response(
        A0.copy(),
        kernel=kernel,
        half_step_response=half_step_response,
        Nsub=Nsub,
    )

    np.testing.assert_allclose(wrapped, prepared, rtol=0.0, atol=0.0)


def test_prepared_shared_response_matches_explicit_channel_broadcast():
    rng = np.random.default_rng(7)
    A0 = (
        rng.normal(size=(3, 8, 6)) + 1j * rng.normal(size=(3, 8, 6))
    ).astype(np.complex128)
    kernel = np.exp(1j * rng.normal(size=(8, 6)))
    shared = np.exp(0.5j * rng.normal(size=(8, 6)))
    per_channel = np.broadcast_to(shared, A0.shape).copy()

    shared_result = advance_prepared_response(
        A0.copy(), kernel=kernel, half_step_response=shared
    )
    broadcast_result = advance_prepared_response(
        A0.copy(), kernel=kernel, half_step_response=per_channel
    )

    np.testing.assert_allclose(shared_result, broadcast_result, rtol=0.0, atol=0.0)


def test_prepared_per_channel_response_is_channel_local():
    A0 = np.ones((2, 4, 5), dtype=np.complex128)
    kernel = np.ones((4, 5), dtype=np.complex128)
    phases = np.asarray([0.3, -0.7])[:, None, None]
    response = np.broadcast_to(np.exp(0.5j * phases), A0.shape).copy()

    result = advance_prepared_response(
        A0.copy(), kernel=kernel, half_step_response=response
    )

    np.testing.assert_allclose(result[0], np.exp(0.3j))
    np.testing.assert_allclose(result[1], np.exp(-0.7j))


def test_prepared_identity_response_reproduces_pure_linear_propagation():
    rng = np.random.default_rng(11)
    A0 = (
        rng.normal(size=(2, 7, 9)) + 1j * rng.normal(size=(2, 7, 9))
    ).astype(np.complex128)
    kernel = np.exp(1j * rng.normal(size=(7, 9)))
    identity_response = np.ones((7, 9), dtype=np.complex128)

    expected = hop_linear(hop_linear(A0, kernel), kernel)
    actual = advance_prepared_response(
        A0.copy(),
        kernel=kernel,
        half_step_response=identity_response,
        Nsub=2,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)


def test_prepared_real_phase_response_preserves_power():
    rng = np.random.default_rng(19)
    A0 = (
        rng.normal(size=(2, 16, 18)) + 1j * rng.normal(size=(2, 16, 18))
    ).astype(np.complex128)
    kernel = np.exp(1j * rng.normal(size=(16, 18)))
    real_half_phase = rng.normal(size=(2, 16, 18))
    response = np.exp(1j * real_half_phase)

    result = advance_prepared_response(
        A0.copy(),
        kernel=kernel,
        half_step_response=response,
        Nsub=3,
    )

    assert np.sum(np.abs(result) ** 2) == pytest.approx(
        np.sum(np.abs(A0) ** 2), rel=2e-15
    )


def test_prepared_response_uses_symmetric_strang_splitting(monkeypatch):
    A = np.ones((1, 2, 2), dtype=np.complex128)
    kernel = np.ones((2, 2), dtype=np.complex128)
    response = np.ones((2, 2), dtype=np.complex128)
    calls = []

    def observed_apply(A, response_screen, *, xp=None):
        calls.append(("apply", None))
        return A

    def observed_hop(A, kernel, *, xp=None):
        calls.append(("linear", None))
        return A

    monkeypatch.setattr(
        "lcprop.optics.splitstep.apply_response_screen_inplace",
        observed_apply,
    )
    monkeypatch.setattr("lcprop.optics.splitstep.hop_linear_inplace", observed_hop)

    advance_prepared_response(
        A,
        kernel=kernel,
        half_step_response=response,
        Nsub=2,
    )

    assert calls == [
        ("apply", None),
        ("linear", None),
        ("apply", None),
        ("apply", None),
        ("linear", None),
        ("apply", None),
    ]
