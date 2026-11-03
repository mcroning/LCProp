"""CPU reference implementation of transverse PR hopping transport.

This module is deliberately separate from the production reduced-x A7 solver.
It implements the periodic, constant-tensor potential formulation documented in
``docs/architecture/pr_transverse_reference_model.md``.  It is a validation
reference, not a production workflow or a tensor-calibrated BaTiO3 model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


BATIO3_CLAMPED_EPSILON_A = 2200.0
BATIO3_CLAMPED_EPSILON_C = 56.0


@dataclass(frozen=True)
class RotatedUniaxialDielectric:
    """Transverse dielectric tensor derived from an x-z crystal rotation.

    ``c_axis_xz_angle_deg=0`` places the uniaxial c-axis along simulation x;
    ``90`` places it along propagation z.  The c-axis unit vector is therefore
    ``(cos(gamma), 0, sin(gamma))``.  ``epsilon_a`` and ``epsilon_c`` are the
    crystal-frame relative dielectric constants perpendicular and parallel to
    the c-axis, respectively.
    """

    c_axis_xz_angle_deg: float
    epsilon_a: float = BATIO3_CLAMPED_EPSILON_A
    epsilon_c: float = BATIO3_CLAMPED_EPSILON_C

    def __post_init__(self) -> None:
        for name in ("c_axis_xz_angle_deg", "epsilon_a", "epsilon_c"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if float(self.epsilon_a) <= 0.0 or float(self.epsilon_c) <= 0.0:
            raise ValueError("epsilon_a and epsilon_c must be positive")

    @property
    def epsilon_xx(self) -> float:
        """Simulation-frame transverse epsilon_xx."""

        gamma = math.radians(float(self.c_axis_xz_angle_deg))
        return float(
            float(self.epsilon_a) * math.sin(gamma) ** 2
            + float(self.epsilon_c) * math.cos(gamma) ** 2
        )

    @property
    def epsilon_yy(self) -> float:
        """Simulation-frame transverse epsilon_yy."""

        return float(self.epsilon_a)

    @property
    def h_y(self) -> float:
        """Normalized transverse dielectric ratio epsilon_yy/epsilon_xx."""

        return self.epsilon_yy / self.epsilon_xx


@dataclass(frozen=True)
class TransverseReferenceOptions:
    """Numerical and material controls for the isotropic reference model.

    ``m_y`` and ``h_y`` are the y/x ratios of the diagonal transport and
    dielectric tensors.  Their unit defaults define an explicitly isotropic
    transverse reference, not calibrated BaTiO3 coefficients.
    """

    dx_normalized: float
    dy_normalized: float
    dt_normalized: float
    steps: int
    m_y: float = 1.0
    h_y: float = 1.0
    applied_field_x: float = 0.0
    dielectric: RotatedUniaxialDielectric | None = None

    @classmethod
    def from_barium_titanate_c_axis(
        cls,
        *,
        dx_normalized: float,
        dy_normalized: float,
        dt_normalized: float,
        steps: int,
        c_axis_xz_angle_deg: float,
        m_y: float = 1.0,
        applied_field_x: float = 0.0,
        epsilon_a: float = BATIO3_CLAMPED_EPSILON_A,
        epsilon_c: float = BATIO3_CLAMPED_EPSILON_C,
    ) -> "TransverseReferenceOptions":
        """Construct options with h_y derived from rotated BaTiO3 epsilon.

        The angle is the primary physical input.  Direct ``h_y`` construction
        remains available as the low-level isotropic/reference path.
        """

        dielectric = RotatedUniaxialDielectric(
            c_axis_xz_angle_deg=c_axis_xz_angle_deg,
            epsilon_a=epsilon_a,
            epsilon_c=epsilon_c,
        )
        return cls(
            dx_normalized=dx_normalized,
            dy_normalized=dy_normalized,
            dt_normalized=dt_normalized,
            steps=steps,
            m_y=m_y,
            h_y=dielectric.h_y,
            applied_field_x=applied_field_x,
            dielectric=dielectric,
        )

    def validate(self) -> None:
        for name in ("dx_normalized", "dy_normalized", "dt_normalized"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(self.steps) < 0:
            raise ValueError("steps must be nonnegative")
        for name in ("m_y", "h_y"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.applied_field_x)):
            raise ValueError("applied_field_x must be finite")
        if self.dielectric is not None and not math.isclose(
            float(self.h_y),
            self.dielectric.h_y,
            rel_tol=1.0e-14,
            abs_tol=0.0,
        ):
            raise ValueError("h_y must match the derived dielectric h_y")


@dataclass(frozen=True)
class TransverseReferenceState:
    """Potential state and fields derived from it on one transverse plane."""

    psi: np.ndarray
    carrier_density: np.ndarray
    E_x: np.ndarray
    E_y: np.ndarray


@dataclass(frozen=True)
class TransverseReferenceDiagnostics:
    """Conservation and electrostatic diagnostics for a reference evolution."""

    carrier_integrals: np.ndarray
    carrier_relative_drift: float
    curl_rms: float
    curl_max: float
    gauss_rms: float
    gauss_max: float


@dataclass(frozen=True)
class TransverseReferenceResult:
    """Final state and compact diagnostics from a reference evolution."""

    initial_state: TransverseReferenceState
    final_state: TransverseReferenceState
    diagnostics: TransverseReferenceDiagnostics
    completed_steps: int
    time_normalized: float


def _validate_plane(field, *, name: str) -> np.ndarray:
    array = np.asarray(field, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 3:
        raise ValueError(f"{name} must have shape (Nx, Ny) with both sizes at least 3")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _wavevectors(shape: tuple[int, int], *, dx: float, dy: float):
    kx = 2.0 * math.pi * np.fft.fftfreq(shape[0], d=dx)[:, None]
    ky = 2.0 * math.pi * np.fft.fftfreq(shape[1], d=dy)[None, :]
    # A real-valued first spectral derivative cannot represent the unpaired
    # Nyquist mode on an even grid.  Setting that derivative symbol to zero
    # makes grad, div, and the electrostatic inversion one consistent discrete
    # operator.  The corresponding checkerboard potential null modes are
    # fixed to zero below, just like the constant potential gauge mode.
    if shape[0] % 2 == 0:
        kx[shape[0] // 2, 0] = 0.0
    if shape[1] % 2 == 0:
        ky[0, shape[1] // 2] = 0.0
    return kx, ky


def _zero_mean_hat(field: np.ndarray) -> np.ndarray:
    transformed = np.fft.fft2(field)
    transformed[0, 0] = 0.0
    return transformed


def _spectral_derivatives(
    field: np.ndarray,
    *,
    kx: np.ndarray,
    ky: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    transformed = np.fft.fft2(field)
    derivative_x = np.fft.ifft2(1j * kx * transformed).real
    derivative_y = np.fft.ifft2(1j * ky * transformed).real
    return derivative_x, derivative_y


def state_from_potential(
    psi,
    *,
    options: TransverseReferenceOptions,
) -> TransverseReferenceState:
    """Return ``P``, ``E_x``, and ``E_y`` derived from periodic potential.

    The returned potential has zero spatial mean.  With
    ``H = diag(1, h_y)``, the spectral closure is

    ``P = 1 - div(H grad(psi))``.
    """

    options.validate()
    potential = _validate_plane(psi, name="psi")
    potential_hat = _zero_mean_hat(potential)
    kx, ky = _wavevectors(
        potential.shape,
        dx=float(options.dx_normalized),
        dy=float(options.dy_normalized),
    )
    denominator = kx * kx + float(options.h_y) * ky * ky
    potential_hat[denominator == 0.0] = 0.0
    potential = np.fft.ifft2(potential_hat).real
    psi_x = np.fft.ifft2(1j * kx * potential_hat).real
    psi_y = np.fft.ifft2(1j * ky * potential_hat).real
    carrier = 1.0 + np.fft.ifft2(denominator * potential_hat).real
    return TransverseReferenceState(
        psi=potential,
        carrier_density=carrier,
        E_x=float(options.applied_field_x) - psi_x,
        E_y=-psi_y,
    )


def potential_rhs(
    psi,
    intensity,
    *,
    options: TransverseReferenceOptions,
) -> np.ndarray:
    """Return ``psi_tau`` for the transverse hopping/potential equation.

    The implemented normalized equation is

    ``-div(H grad(psi_tau)) = div(M (grad(P*I) - P*I*e))``

    with ``M = diag(1, m_y)``, ``H = diag(1, h_y)``, and
    ``e = (E_app, 0) - grad(psi)``.  The zero Fourier mode is fixed to zero,
    selecting a zero-mean periodic space-charge potential.
    """

    options.validate()
    potential = _validate_plane(psi, name="psi")
    driving_intensity = _validate_plane(intensity, name="intensity")
    if potential.shape != driving_intensity.shape:
        raise ValueError("psi and intensity must have identical shapes")
    if np.any(driving_intensity < 0.0):
        raise ValueError("intensity must be nonnegative")

    state = state_from_potential(potential, options=options)
    kx, ky = _wavevectors(
        potential.shape,
        dx=float(options.dx_normalized),
        dy=float(options.dy_normalized),
    )
    carrier_intensity = state.carrier_density * driving_intensity
    gradient_x, gradient_y = _spectral_derivatives(
        carrier_intensity,
        kx=kx,
        ky=ky,
    )
    flux_x = gradient_x - carrier_intensity * state.E_x
    flux_y = float(options.m_y) * (
        gradient_y - carrier_intensity * state.E_y
    )
    carrier_rhs_hat = (
        1j * kx * np.fft.fft2(flux_x)
        + 1j * ky * np.fft.fft2(flux_y)
    )

    denominator = kx * kx + float(options.h_y) * ky * ky
    psi_rhs_hat = np.zeros_like(carrier_rhs_hat)
    nonzero = denominator > 0.0
    psi_rhs_hat[nonzero] = carrier_rhs_hat[nonzero] / denominator[nonzero]
    psi_rhs_hat[0, 0] = 0.0
    return np.fft.ifft2(psi_rhs_hat).real


def active_field_from_components(
    E_x,
    E_y,
    *,
    x_weight: float = 1.0,
    y_weight: float = 0.0,
) -> np.ndarray:
    """Return an explicit scalar projection of transverse electric field.

    The default ``(x_weight, y_weight) = (1, 0)`` selects ``E_active=E_x``.
    This is a reference geometry assumption, not a consequence of transport
    electrostatics and not a full BaTiO3 Pockels-tensor contraction.
    """

    field_x = _validate_plane(E_x, name="E_x")
    field_y = _validate_plane(E_y, name="E_y")
    if field_x.shape != field_y.shape:
        raise ValueError("E_x and E_y must have identical shapes")
    for name, value in (("x_weight", x_weight), ("y_weight", y_weight)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    return float(x_weight) * field_x + float(y_weight) * field_y


def electrostatic_residuals(
    state: TransverseReferenceState,
    *,
    options: TransverseReferenceOptions,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the spectral ``(curl, Gauss)`` residual fields."""

    options.validate()
    field_x = _validate_plane(state.E_x, name="E_x")
    field_y = _validate_plane(state.E_y, name="E_y")
    carrier = _validate_plane(state.carrier_density, name="carrier_density")
    if field_x.shape != field_y.shape or field_x.shape != carrier.shape:
        raise ValueError("state fields must have identical shapes")
    kx, ky = _wavevectors(
        field_x.shape,
        dx=float(options.dx_normalized),
        dy=float(options.dy_normalized),
    )
    E_y_x, _ = _spectral_derivatives(field_y, kx=kx, ky=ky)
    _, E_x_y = _spectral_derivatives(field_x, kx=kx, ky=ky)
    E_x_x, _ = _spectral_derivatives(field_x, kx=kx, ky=ky)
    _, E_y_y = _spectral_derivatives(field_y, kx=kx, ky=ky)
    curl = E_y_x - E_x_y
    gauss = carrier - 1.0 - E_x_x - float(options.h_y) * E_y_y
    return curl, gauss


def run_transverse_reference(
    psi_initial,
    intensity,
    *,
    options: TransverseReferenceOptions,
) -> TransverseReferenceResult:
    """Advance the periodic potential with transparent explicit Euler steps."""

    options.validate()
    potential = _validate_plane(psi_initial, name="psi_initial").copy()
    driving_intensity = _validate_plane(intensity, name="intensity")
    if potential.shape != driving_intensity.shape:
        raise ValueError("psi_initial and intensity must have identical shapes")
    initial_state = state_from_potential(potential, options=options)
    potential = initial_state.psi.copy()
    cell_area = float(options.dx_normalized) * float(options.dy_normalized)
    carrier_integrals = [
        float(np.sum(initial_state.carrier_density) * cell_area)
    ]

    for _ in range(int(options.steps)):
        potential += float(options.dt_normalized) * potential_rhs(
            potential,
            driving_intensity,
            options=options,
        )
        potential -= float(np.mean(potential))
        if not np.all(np.isfinite(potential)):
            raise FloatingPointError("nonfinite potential during reference evolution")
        state = state_from_potential(potential, options=options)
        carrier_integrals.append(
            float(np.sum(state.carrier_density) * cell_area)
        )

    final_state = state_from_potential(potential, options=options)
    curl, gauss = electrostatic_residuals(final_state, options=options)
    integrals = np.asarray(carrier_integrals, dtype=np.float64)
    initial_integral = float(integrals[0])
    relative_drift = float(
        np.max(np.abs(integrals - initial_integral)) / abs(initial_integral)
    )
    diagnostics = TransverseReferenceDiagnostics(
        carrier_integrals=integrals,
        carrier_relative_drift=relative_drift,
        curl_rms=float(np.sqrt(np.mean(curl * curl))),
        curl_max=float(np.max(np.abs(curl))),
        gauss_rms=float(np.sqrt(np.mean(gauss * gauss))),
        gauss_max=float(np.max(np.abs(gauss))),
    )
    return TransverseReferenceResult(
        initial_state=initial_state,
        final_state=final_state,
        diagnostics=diagnostics,
        completed_steps=int(options.steps),
        time_normalized=int(options.steps) * float(options.dt_normalized),
    )


__all__ = [
    "BATIO3_CLAMPED_EPSILON_A",
    "BATIO3_CLAMPED_EPSILON_C",
    "RotatedUniaxialDielectric",
    "TransverseReferenceDiagnostics",
    "TransverseReferenceOptions",
    "TransverseReferenceResult",
    "TransverseReferenceState",
    "active_field_from_components",
    "electrostatic_residuals",
    "potential_rhs",
    "run_transverse_reference",
    "state_from_potential",
]
