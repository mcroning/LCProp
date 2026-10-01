"""Bitwise derivative equivalence and operation-level unused-allocation guards."""
import numpy as np
import pytest

from lcprop.core.backend import BackendSpec
from lcprop.pr.evolution import periodic_derivatives_x, periodic_first_derivative_x
import lcprop.pr.reduced_field_linear as local


def previous_derivatives(field, *, dx_normalized, xp):
    # Frozen pre-refactor arithmetic/order; independent of the factored helper.
    dx = float(dx_normalized)
    forward = xp.roll(field, -1, axis=-2)
    backward = xp.roll(field, 1, axis=-2)
    first = (forward - backward) / (2.0 * dx)
    second = (forward - 2.0 * field + backward) / (dx * dx)
    return first, second


def fields(dtype, nx, batched):
    shape = (2, nx, 3) if batched else (nx, 3)
    x = np.arange(nx)[:, None]
    values = [np.full((nx, 3), 1.3),
              np.broadcast_to(1.3 + .4*np.cos(2*np.pi*3*x/nx), (nx, 3)),
              np.random.default_rng(28).uniform(.1, 2., (nx, 3))]
    if nx % 2 == 0:
        values.append(np.broadcast_to(1.3 + .4*(-1.)**x, (nx, 3)))
    return [np.broadcast_to(value, shape).astype(dtype).copy() for value in values]


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('nx', [15, 16])
@pytest.mark.parametrize('batched', [False, True])
def test_first_and_both_bitwise_previous(precision, nx, batched):
    for value in fields(precision, nx, batched):
        kw = dict(dx_normalized=.5228141586169444, xp=np)
        old = previous_derivatives(value, **kw)
        new = periodic_derivatives_x(value, **kw)
        first = periodic_first_derivative_x(value, **kw)
        assert first.dtype == value.dtype
        assert first.tobytes() == old[0].tobytes()
        assert all(a.tobytes() == b.tobytes() for a, b in zip(old, new))
    if nx % 2 == 0:
        assert not np.any(periodic_first_derivative_x(fields(precision, nx, batched)[-1], **kw))


@pytest.mark.parametrize('precision', ['float32', 'float64'])
@pytest.mark.parametrize('bias', [0., .7])
def test_local_solve_residual_bitwise_previous(monkeypatch, precision, bias):
    spec = local.PRReducedFieldLinearSpec(bias, .1, .5228141586169444)
    for nx in (15, 16):
        for intensity in fields(precision, nx, True):
            result = local.solve_pr_reduced_field_linear_intensity(
                intensity, spec=spec, backend=BackendSpec('numpy', precision, False))
            with monkeypatch.context() as patch:
                patch.setattr(local, 'periodic_first_derivative_x',
                              lambda f, **kw: previous_derivatives(f, **kw)[0])
                patch.setattr(local, 'periodic_derivatives_x', previous_derivatives)
                old = local.solve_pr_reduced_field_linear_intensity(
                    intensity, spec=spec, backend=BackendSpec('numpy', precision, False))
            for name in ('E', 'residual', 'denominator'):
                assert getattr(result, name).tobytes() == getattr(old, name).tobytes()


def test_first_primitive_constructs_only_first_difference():
    # Operator tracing catches construction even if an allocator caches memory.
    calls = []
    class Plane:
        ndim = 2
        def __sub__(self, other):
            calls.append('subtract')
            return Plane()
        def __truediv__(self, scalar):
            assert scalar == 1.04
            calls.append('divide')
            return Plane()
        def __rmul__(self, scalar):
            raise AssertionError('second-derivative product allocated')
        def __add__(self, other):
            raise AssertionError('second-derivative sum allocated')
    class Backend:
        @staticmethod
        def roll(field, shift, axis):
            calls.append(('roll', shift, axis))
            return Plane()
    periodic_first_derivative_x(Plane(), dx_normalized=.52, xp=Backend)
    assert calls == [('roll', -1, -2), ('roll', 1, -2), 'subtract', 'divide']


def test_local_intensity_never_uses_second_derivative_path(monkeypatch):
    intensity = fields('float64', 16, False)[2]
    both_inputs = []
    first_inputs = []
    def both(field, **kw):
        assert not np.shares_memory(field, intensity), 'Dxx intensity output constructed'
        both_inputs.append(field)
        return periodic_derivatives_x(field, **kw)
    def first(field, **kw):
        assert np.shares_memory(field, intensity)
        first_inputs.append(field)
        return periodic_first_derivative_x(field, **kw)
    monkeypatch.setattr(local, 'periodic_derivatives_x', both)
    monkeypatch.setattr(local, 'periodic_first_derivative_x', first)
    spec = local.PRReducedFieldLinearSpec(.3, .1, .52)
    result = local.solve_pr_reduced_field_linear_intensity(intensity, spec=spec)
    residual = local.reduced_field_linear_residual(result.E, intensity, spec=spec, xp=np)
    assert len(first_inputs) == 3 and len(both_inputs) == 2
    assert all(value is result.E for value in both_inputs)
    assert residual.tobytes() == result.residual.tobytes()


@pytest.mark.parametrize('precision', ['float32', 'float64'])
def test_cupy_bitwise_previous(precision, monkeypatch):
    try:
        import cupy as cp
        if not cp.cuda.runtime.getDeviceCount():
            pytest.skip('no CUDA device')
    except Exception:
        pytest.skip('CuPy/CUDA unavailable locally')
    spec = local.PRReducedFieldLinearSpec(.7, .1, .5228141586169444)
    for nx in (15, 16):
        for host in fields(precision, nx, True):
            intensity = cp.asarray(host)
            kw = dict(dx_normalized=spec.dx_normalized, xp=cp)
            first = periodic_first_derivative_x(intensity, **kw)
            previous = previous_derivatives(intensity, **kw)[0]
            assert cp.asnumpy(first).tobytes() == cp.asnumpy(previous).tobytes()
            result = local.solve_pr_reduced_field_linear_intensity(
                intensity, spec=spec, backend=BackendSpec('cupy', precision, False))
            with monkeypatch.context() as patch:
                patch.setattr(local, 'periodic_first_derivative_x',
                              lambda f, **kw: previous_derivatives(f, **kw)[0])
                patch.setattr(local, 'periodic_derivatives_x', previous_derivatives)
                old = local.solve_pr_reduced_field_linear_intensity(
                    intensity, spec=spec, backend=BackendSpec('cupy', precision, False))
            for name in ('E', 'residual', 'denominator'):
                assert cp.asnumpy(getattr(result, name)).tobytes() == cp.asnumpy(getattr(old, name)).tobytes()
