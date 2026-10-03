"""Private one-column Newton core, preserving the frozen Stage-E1.1 arithmetic."""
import numpy as np
from .operators import carrier, flux, bernoulli, normalized_log_carrier

def system(g, I, q, p, b, closure):
    """Full Gauss, independent divergence rows, gauge, electrical equations."""
    xp, sp, backend = g.xp, g.backend.sparse, g.backend
    csc_matrix = lambda a: sp.csr_matrix(a, dtype=backend.dtype)
    diags, bmat, vstack = sp.diags, sp.bmat, sp.vstack
    U, V, target = closure
    N, d = g.size, len(g.shape)
    n = carrier(xp, q)
    w = I * n
    faces = flux(g, I, q, p, b)
    jq, jp, jb, div = [], [], [], []
    for j, h in enumerate(g.spacing):
        v = g.neighbor(p, j) - p - b[j] * h
        B, D = bernoulli(xp, v)
        Bm, Dm = bernoulli(xp, -v)
        wr = g.neighbor(w, j)
        dv = (D * w + Dm * wr) / h
        jq.append(g.forward_pair_matrix(B * w / h, -Bm * wr / h, j))
        jp.append(g.forward_pair_matrix(-dv, dv, j))
        bj = xp.zeros((N, d), dtype=backend.dtype)
        bj[:, j] = (-h * dv).ravel()
        jb.append(csc_matrix(bj))
        grad = g.forward_pair_matrix(xp.full(g.shape, -1/h, dtype=backend.dtype), xp.full(g.shape, 1/h, dtype=backend.dtype), j)
        div.append(-grad.T)
    dq = sum(D @ Q for D, Q in zip(div, jq))
    dp = sum(D @ P for D, P in zip(div, jp))
    db = sum(D @ B for D, B in zip(div, jb))
    mq = vstack([csc_matrix(Q.mean(axis=0)) for Q in jq])
    mp = vstack([csc_matrix(P.mean(axis=0)) for P in jp])
    mb = xp.vstack([xp.asarray(B.mean(axis=0)) for B in jb])
    current = xp.stack([xp.mean(J) for J in faces])
    r = xp.concatenate(((g.poisson(p)-n+1).ravel(), g.divergence(faces).ravel()[:-1],
                        xp.mean(p).reshape(1), U @ b + V @ current - target))
    matrix = bmat([
        [-diags(n.ravel()), g.sparse_poisson(), csc_matrix((N, d))],
        [dq[:-1], dp[:-1], db[:-1]],
        [csc_matrix((1, N)), csc_matrix(xp.ones((1, N), dtype=backend.dtype)/N), csc_matrix((1, d))],
        [csc_matrix(V) @ mq, csc_matrix(V) @ mp, csc_matrix(U+V @ mb)]
    ], format='csr')
    return r, matrix



def residual(g, I, q, p, b, electrical):
    xp = g.xp
    U, V, target = electrical
    n = carrier(xp, q)
    J = flux(g, I, q, p, b)
    return xp.concatenate(((g.poisson(p)-n+1).ravel(), g.divergence(J).ravel()[:-1],
                           xp.mean(p).reshape(1), U @ b+V @ xp.stack([xp.mean(j) for j in J])-target))


def diagnose(g, I, q, p, b, electrical, zero):
    xp, B = g.xp, g.backend
    U, V, target = electrical
    n = carrier(xp, q)
    J = flux(g, I, q, p, b)
    G, D = g.poisson(p)-n+1, g.divergence(J)
    C = U @ b+V @ xp.stack([xp.mean(j) for j in J])-target
    vals = []
    for a in (G, D, C):
        vals += [xp.sqrt(xp.mean(a*a)), xp.max(abs(a))]
    # Floating-point stencil cancellation scales, reduced on backend.
    gs = 1+xp.max(n)+4*xp.max(abs(p))*sum(1/h**2 for h in g.spacing)
    js = 1+4*xp.max(abs(I*n))*sum(1/h**2 for h in g.spacing)+sum(2*xp.max(abs(j))/h for j, h in zip(J, g.spacing))
    vals += [abs(xp.mean(n)-1), abs(xp.mean(p)), xp.min(n), xp.max(n),
             xp.all(xp.isfinite(q)) & xp.all(xp.isfinite(p)) & xp.all(xp.isfinite(n)) & xp.all(xp.isfinite(b)),
             gs, js, xp.max(xp.stack([xp.max(abs(j)) for j in J]))]
    v = B.status(vals)
    eps = np.finfo(B.precision).eps if B.precision == 'float32' else 0.
    rms = np.array([max(1e-10, 8*eps*v[11]), max(1e-10, 8*eps*v[12]), max(1e-10, 8*eps)])
    maximum = np.array([max(1e-9, 32*eps*v[11]), max(1e-9, 32*eps*v[12]), max(1e-9, 32*eps)])
    passed = bool(np.all(np.isfinite(v)) and np.all(v[[0, 2, 4]] <= rms) and np.all(v[[1, 3, 5]] <= maximum)
                  and v[6] <= max(1e-11, 8*eps) and v[7] <= max(1e-11, 8*eps) and v[8] > 0 and v[10] == 1)
    if zero:
        passed = passed and v[13] <= max(1e-9, 32*eps*v[12])
    return dict(passed=bool(passed), values=v.tolist(), rms_limits=rms.tolist(), max_limits=maximum.tolist())


def solve_column(g, intensity, electrical, *, zero):
    xp, B = g.xp, g.backend
    I = B.array(intensity).copy()
    if I.shape != g.shape or electrical[2].size != 1:
        raise ValueError('Shape/closure mismatch')
    if not B.status([xp.all(xp.isfinite(I)) & xp.all(I > 0)])[0]:
        raise ValueError('Positive finite total intensity required')
    U, V, target = electrical
    b = B.direct_dense(U+xp.mean(I)*V, target)
    p = xp.zeros(g.shape, dtype=B.dtype)
    q = xp.zeros_like(p)
    L = g.sparse_poisson()
    trace = []
    for it in range(61):
        if zero:
            q = normalized_log_carrier(g, p, I)
        r = diagnose(g, I, q, p, b, electrical, zero)
        trace.append(dict(iteration=it, **r))
        if r['passed'] and (B.precision == 'float64' or it == 0):
            return dict(q=q, psi=p, b=b, trace=trace, precision=B.precision)
        if it == 60:
            raise RuntimeError('Newton bound: '+str(r)+'; corrections='+str([t.get('relative_potential_correction') for t in trace]))
        if zero:
            n = carrier(xp, q)
            merit = g.poisson(p)-n+1
            matrix = L+B.sparse.diags(n.ravel())
            dp = B.direct(matrix, -(merit-xp.mean(merit)).ravel()).reshape(g.shape)
            dp = g.gauge(dp)
            dq, db = None, None
        else:
            merit, matrix = system(g, I, q, p, b, electrical)
            step = B.direct(matrix, -merit)
            dq, dp, db = step[:g.size].reshape(g.shape), step[g.size:2*g.size].reshape(g.shape), step[2*g.size:]
        correction = B.status([xp.max(abs(dp))/(1+xp.max(abs(p)))])[0]
        trace[-1]['relative_potential_correction'] = float(correction)
        closure_correction = 0. if zero else B.status([xp.maximum(xp.max(abs(dq)), xp.max(abs(db)))])[0]
        trace[-1]['q_b_correction'] = float(closure_correction)
        if B.precision == 'float32' and r['passed'] and correction <= 2e-5 and closure_correction <= 2e-6:
            return dict(q=q, psi=p, b=b, trace=trace, precision=B.precision)
        if not B.status([xp.all(xp.isfinite(dp))])[0]:
            raise RuntimeError('Nonfinite linear correction')
        alpha = 1.
        for bt in range(30):
            pt = p+alpha*dp
            if zero:
                pt = g.gauge(pt)
                qt, bb = normalized_log_carrier(g, pt, I), b
            else:
                qt, bb = q+alpha*dq, b+alpha*db
            nt = carrier(xp, qt)
            valid = B.status([xp.all(xp.isfinite(nt)) & xp.all(nt > 0)])[0]
            if valid:
                rr = g.poisson(pt)-nt+1 if zero else residual(g, I, qt, pt, bb, electrical)
                good = B.status([xp.linalg.norm(rr.ravel()) <= (1-1e-4*alpha)*xp.linalg.norm(merit.ravel())])[0]
                if good:
                    p, q, b = pt, qt, bb
                    trace[-1].update(alpha=alpha, halvings=bt, matrix_nnz=int(matrix.nnz))
                    break
            alpha *= .5
        else:
            raise RuntimeError('Armijo bound: '+str(r))
    raise AssertionError('unreachable')
