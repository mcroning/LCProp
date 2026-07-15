import numpy as np

from lcprop.core.beams import BeamChannel, BeamStack
from lcprop.core.context import GridSpec, LCMaterial
from lcprop.core.grid import make_grid
from lcprop.lc.coupling import resolved_bi
from lcprop.optics.launch import build_launch
from lcprop.optics.splitstep import weighted_theta_intensity


def _normalization_metrics(power_mW: float, Nx: int) -> tuple[float, float]:
    grid_spec = GridSpec(
        Nx=Nx,
        Ny=Nx,
        x_aperture_um=75.0,
        y_aperture_um=100.0,
    )
    grid = make_grid(grid_spec)
    material = LCMaterial()
    beams = BeamStack(
        channels=(
            BeamChannel(
                power_mW=power_mW,
                waist_x_um=3.0,
                waist_y_um=3.0,
            ),
        )
    )
    launch = build_launch(beams, grid)
    weighted_intensity = weighted_theta_intensity(
        launch.A0,
        launch.theta_weights,
        coherence_groups=launch.coherence_groups,
        xp=grid.xp,
    )
    source = resolved_bi(grid_spec, material, beams) * weighted_intensity
    area = grid.dx_um * grid.dy_um
    return (
        float(np.sum(weighted_intensity) * area),
        float(np.sum(source) * area),
    )


def test_director_source_power_scaling_matches_trusted_linear_convention():
    integral_1_64, source_1_64 = _normalization_metrics(1.0, 64)
    integral_1_128, source_1_128 = _normalization_metrics(1.0, 128)
    integral_2_64, source_2_64 = _normalization_metrics(2.0, 64)
    integral_4_64, source_4_64 = _normalization_metrics(4.0, 64)

    assert np.allclose(
        [integral_1_64, integral_1_128, integral_2_64, integral_4_64],
        1.0,
        rtol=1e-6,
    )
    assert np.isclose(source_1_64, source_1_128, rtol=1e-6)

    assert np.allclose(
        [source_1_64, source_2_64, source_4_64],
        source_1_64 * np.asarray([1.0, 2.0, 4.0]),
        rtol=1e-6,
    )
