import numpy as np
import pytest

from lcprop.pr.readiness import (
    PRImageAmplificationReadinessSpec,
    make_image_amplification_readiness_request,
    run_image_amplification_readiness,
)
from lcprop.pr.specs import PR_SEMI_IMPLICIT_INTEGRATOR


def test_readiness_request_is_coherent_periodic_and_uses_production_integrator():
    request, kx, grating_samples = make_image_amplification_readiness_request()

    assert request.solver.integrator == PR_SEMI_IMPLICIT_INTEGRATOR
    assert request.beams.coherence == "coherent"
    assert request.beams.coherence_groups[0] == request.beams.coherence_groups[1]
    assert request.beams.channels[0].tilt_x_rad_per_um == pytest.approx(kx)
    assert request.beams.channels[1].tilt_x_rad_per_um == pytest.approx(-kx)
    assert kx * request.grid.x_aperture_um / (2.0 * np.pi) == pytest.approx(2.0)
    assert grating_samples == pytest.approx(16.0)


def test_readiness_case_clears_numerical_gate_before_image_benchmark():
    result = run_image_amplification_readiness()

    assert result.run_result.status == "completed"
    assert (
        result.request.solver.dt_normalized
        > 3.0 * result.explicit_euler_dt_limit
    )
    assert result.request.solver.dt_normalized < result.semi_implicit_dt_limit
    assert result.static_result.converged
    assert result.final_residual_rms < 1e-6
    assert result.final_residual_max < 1e-5
    assert result.static_relative_l2_error < 2e-5
    assert result.static_max_error < 2e-5
    assert abs(result.signal_matched_power_relative_change) > 0.1
    assert abs(result.normalized_power_relative_drift) < 2e-12
    assert np.all(np.isfinite(result.run_result.E_final))


@pytest.mark.parametrize("mode", (0, 16))
def test_readiness_request_rejects_nonresolving_grating_modes(mode):
    with pytest.raises(ValueError, match="grating below Nyquist"):
        make_image_amplification_readiness_request(
            PRImageAmplificationReadinessSpec(positive_mode_index=mode)
        )
