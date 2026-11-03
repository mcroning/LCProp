from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from lcprop.lc.optical_response import phase_screen_from_theta
from lcprop.lc.propagation import (
    advance_slice,
    neff_from_theta,
    nonlinear_phase,
)
from lcprop.optics.splitstep import advance_prepared_response, linear_kernel


def test_lc_propagation_functions_have_lc_module_provenance():
    assert advance_slice.__module__ == "lcprop.lc.propagation"
    assert nonlinear_phase.__module__ == "lcprop.lc.propagation"
    assert neff_from_theta.__module__ == "lcprop.lc.propagation"


def test_nonlinear_phase_is_exact_lc_optical_response():
    theta = np.linspace(0.1, 0.9, 120, dtype=np.float64).reshape(12, 10)
    kwargs = {
        "dz": 3.75,
        "wavelength": 0.633,
        "n_ref": 1.5,
        "ne": 1.7,
        "no": 1.5,
    }

    actual = nonlinear_phase(theta, **kwargs)
    expected = phase_screen_from_theta(theta, **kwargs)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(np.abs(actual), 1.0)


def test_lc_advance_wrapper_matches_prepared_response_exactly():
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

    np.testing.assert_array_equal(wrapped, prepared)


def test_shared_splitstep_has_no_material_specific_imports_or_exports():
    import lcprop.optics.splitstep as splitstep

    path = Path(splitstep.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any(
        name == "lcprop.lc" or name.startswith("lcprop.lc.")
        for name in imported
    )
    for removed in (
        "advance_slice",
        "apply_nonlinear_phase_inplace",
        "neff_from_theta",
        "nonlinear_phase",
    ):
        assert not hasattr(splitstep, removed)
