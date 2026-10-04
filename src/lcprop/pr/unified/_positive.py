"""V3 derived arithmetic; canonical state remains float32.

The q interval is an arithmetic representability contract, not a carrier floor.
No registration, state mutation, narrowing or scientific host conversion.
"""
from . import operators

Q_MIN = -103.972076416015625
Q_MAX = 88.72283172607421875


class CarrierDomainError(ValueError):
    pass


def domain_ok(xp, q):
    if q.dtype != xp.float32:
        raise TypeError('v3 domain predicate requires canonical float32 q')
    return bool((xp.all(xp.isfinite(q)) & xp.all(q >= Q_MIN) & xp.all(q <= Q_MAX)).item())


def carrier(xp, q):
    if q.dtype == xp.float64:
        return operators.carrier(xp, q)
    if q.dtype != xp.float32:
        raise TypeError('canonical q must be float32 or float64')
    if not domain_ok(xp, q):
        raise CarrierDomainError('q outside finite positive RN32 carrier domain; no clipping')
    n = xp.exp(q.astype(xp.float64))
    if n.dtype != xp.float64 or not bool((xp.all(xp.isfinite(n)) & xp.all(n > 0)).item()):
        raise CarrierDomainError('derived carrier is not positive finite float64')
    return n


def carrier_product(xp, q):
    """Detached quantitative product on the input backend, always float64.

    No dtype override: optional float32 casting was not natively qualified.
    This material API does not register a workflow product or a codec.
    """
    return carrier(xp, q).copy()


def normalized_log_carrier(g, p, intensity):
    if p.dtype == g.xp.float64:
        return operators.normalized_log_carrier(g, p, intensity)
    return operators.normalized_log_carrier(
        g, p.astype(g.xp.float64), intensity.astype(g.xp.float64)).astype(g.xp.float32)


class Geometry(operators.Geometry):
    def divergence(self, faces):
        if self.backend.precision == 'float64':
            return super().divergence(faces)
        # Primal field differences retain state precision; carrier fluxes do not.
        xp = self.xp
        out = xp.zeros(self.shape, dtype=xp.result_type(*[f.dtype for f in faces]))
        for j, (f, h) in enumerate(zip(faces, self.spacing)):
            out += (f-xp.roll(f, 1, axis=j))/h
        return out


class _Arithmetic:
    carrier = staticmethod(carrier)

    @staticmethod
    def weight(xp, intensity, q):
        n = carrier(xp, q)
        return intensity.astype(xp.float64)*n if q.dtype == xp.float32 else intensity*n

    @staticmethod
    def diagnostic_weight(xp, intensity, n):
        # Reuse the carrier already evaluated by the shared physical diagnostic.
        return intensity.astype(xp.float64)*n

    def flux(self, g, intensity, q, psi, b):
        return operators.flux(g, intensity, q, psi, b, arithmetic=self)


ARITHMETIC = _Arithmetic()
