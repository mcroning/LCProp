"""One-dimensional compatible incidence operators and hopping face flux.

Ported from the frozen Stage-E1.1 core; no transport in batch axes."""
import math
from dataclasses import dataclass


def bernoulli(xp, value):
    # Preserve Stage-A branch thresholds and formulas. Float32 evaluates the
    # small subtraction in B' in float64 then rounds once; no altered flux law.
    v = value.astype(xp.float64)
    small, large, negative = abs(v) < 1e-3, v > 50, v < -50
    x = xp.where(small, v, 0.)
    bs = 1-x/2+x*x/12-x**4/720+x**6/30240
    ds = -.5+x/6-x**3/180+x**5/5040
    a = xp.where(large, v, 51.)
    bl, dl = a*xp.exp(-a), (1-a)*xp.exp(-a)
    m = xp.where(small | large | negative, 1., v)
    em = xp.expm1(m)
    bm, dm = m/em, (em-m*(em+1))/(em*em)
    b = xp.where(small, bs, xp.where(large, bl, xp.where(negative, -v, bm)))
    d = xp.where(small, ds, xp.where(large, dl, xp.where(negative, -1., dm)))
    return b.astype(value.dtype), d.astype(value.dtype)


@dataclass
class Geometry:
    shape: tuple
    lengths: tuple
    backend: object

    def __post_init__(self):
        if len(self.shape) != 1 or len(self.lengths) != 1:
            raise ValueError('M2 supports one active x axis only')
        if any(n < 2 or int(n) != n for n in self.shape) or any(not math.isfinite(v) or v <= 0 for v in self.lengths):
            raise ValueError('Invalid geometry')
        self.xp = self.backend.xp
        self.size = math.prod(self.shape)
        self.spacing = tuple(L/n for L, n in zip(self.lengths, self.shape))

    def neighbor(self, a, j):
        return self.xp.roll(a, -1, axis=j)

    def gradient(self, p):
        return tuple((self.neighbor(p, j)-p)/h for j, h in enumerate(self.spacing))

    def divergence(self, faces):
        out = self.xp.zeros(self.shape, dtype=self.backend.dtype)
        for j, (f, h) in enumerate(zip(faces, self.spacing)):
            out += (f-self.xp.roll(f, 1, axis=j))/h
        return out

    def poisson(self, p):
        return -self.divergence(self.gradient(p))

    def gauge(self, p):
        return p-self.xp.mean(p)

    def forward_pair_matrix(self, a, c, axis):
        xp, sp = self.xp, self.backend.sparse
        ids = xp.arange(self.size, dtype=xp.int32).reshape(self.shape)
        return sp.coo_matrix((xp.concatenate((a.ravel(), c.ravel())),
                              (xp.tile(ids.ravel(), 2), xp.concatenate((ids.ravel(), self.neighbor(ids, axis).ravel())))),
                             shape=(self.size, self.size)).tocsr()

    def sparse_poisson(self):
        xp, sp = self.xp, self.backend.sparse
        ids = xp.arange(self.size, dtype=xp.int32).reshape(self.shape)
        rows, cols, vals = [], [], []
        for j, h in enumerate(self.spacing):
            for shift, v in [(-1, -1.), (0, 2.), (1, -1.)]:
                rows.append(ids.ravel())
                cols.append(xp.roll(ids, shift, axis=j).ravel())
                vals.append(xp.full(self.size, v/h**2, dtype=self.backend.dtype))
        return sp.coo_matrix((xp.concatenate(vals), (xp.concatenate(rows), xp.concatenate(cols))), shape=(self.size, self.size)).tocsr()


def carrier(xp, q):
    return xp.exp(q)


def normalized_log_carrier(g, p, I):
    xp = g.xp
    v = -p-xp.log(I)
    mx = xp.max(v)
    return v-(mx+xp.log(xp.sum(xp.exp(v-mx))))+math.log(g.size)


def field(g, p, b):
    return tuple(b[j]-v for j, v in enumerate(g.gradient(p)))


def flux(g, I, q, p, b):
    xp = g.xp
    w = I*carrier(xp, q)
    result = []
    for j, h in enumerate(g.spacing):
        v = g.neighbor(p, j)-p-b[j]*h
        result.append((bernoulli(xp, v)[0]*w-bernoulli(xp, -v)[0]*g.neighbor(w, j))/h)
    return tuple(result)
