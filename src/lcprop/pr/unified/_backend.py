"""Explicit backend, compact status transfers, and certified mixed linear solves.

There is no CPU fallback and no scientific-array export in this module.
"""
import numpy as np
from lcprop.core.backend import BackendSpec, get_backend


class MaterialBackend:
    def __init__(self, name, precision):
        if name not in ('numpy', 'cupy'):
            raise ValueError('Explicit material backend required')
        resolved = get_backend(BackendSpec(name, precision, verbose=False))
        if resolved.name != name:
            raise RuntimeError('Backend substitution prohibited')
        self.name, self.xp = name, resolved.xp
        self.precision, self.dtype = precision, resolved.real_dtype
        if name == 'numpy':
            import scipy.sparse as sparse
            import scipy.sparse.linalg as linalg
        else:
            import cupyx.scipy.sparse as sparse
            import cupyx.scipy.sparse.linalg as linalg
            import cupyx.cusolver
            if not cupyx.cusolver.check_availability('csrlsvqr'):
                raise RuntimeError('Native CSR QR required; no CPU fallback')
        self.sparse, self.linalg = sparse, linalg

    def array(self, value):
        # Callers have already validated scientific arrays. Small electrical
        # metadata is intentionally uploaded here, never scientific data exported.
        return self.xp.asarray(value, dtype=self.dtype)

    def status(self, values):
        # Only backend-reduced scalars may cross this boundary. Check each input
        # before stacking, so a caller cannot route a scientific plane through it.
        if not 1 <= len(values) <= 32 or any(self.xp.ndim(v) != 0 for v in values):
            raise ValueError('Status requires 1..32 already-reduced scalars')
        a = self.xp.stack([self.xp.asarray(v, dtype=self.xp.float64) for v in values])
        return np.array(a, copy=True) if self.name == 'numpy' else self.xp.asnumpy(a)

    def direct(self, matrix, rhs):
        matrix = matrix.tocsr().astype(self.xp.float64)
        return self.linalg.spsolve(matrix, rhs.astype(self.xp.float64)).astype(self.dtype, copy=False)

    def direct_dense(self, matrix, rhs):
        return self.xp.linalg.solve(matrix.astype(self.xp.float64),
                                    rhs.astype(self.xp.float64)).astype(self.dtype, copy=False)
