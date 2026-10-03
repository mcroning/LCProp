"""Independent 1D edge-integral physical oracle; no Product operator imports."""
import numpy as np


def evaluate(intensity, q, psi, b, length, fixed, target, zero, precision):
    I, q, p = [np.array(a, dtype=np.longdouble) for a in (intensity, q, psi)]
    h = np.longdouble(length)/len(p)
    n = np.exp(q)
    E = np.empty_like(p)
    J = np.empty_like(p)
    for i in range(len(p)):
        j = (i+1) % len(p)
        drop = p[j]-p[i]-np.longdouble(b)*h
        E[i] = -drop/h
        integral = np.expm1(drop)/drop if drop else np.longdouble(1)
        log_left = q[i]+np.log(I[i])
        log_right = q[j]+np.log(I[j])
        J[i] = -np.exp(log_left)*np.expm1(log_right-log_left+drop)/(h*integral)
    gauss = np.array([(E[i]-E[i-1])/h-n[i]+1 for i in range(len(p))])
    divergence = np.array([(J[i]-J[i-1])/h for i in range(len(p))])
    closure = np.array([(np.mean(E) if fixed else np.mean(J))-target])
    eps = np.finfo(np.float32).eps if precision == 'float32' else 0.
    scales = (1+np.max(n)+4*np.max(abs(p))/h**2,
              1+4*np.max(n*I)/h**2+2*np.max(abs(J))/h, 1.)
    for residual, scale in zip((gauss, divergence, closure), scales):
        assert np.sqrt(np.mean(residual**2)) <= max(1e-10, 8*eps*scale)
        assert np.max(abs(residual)) <= max(1e-9, 32*eps*scale)
    assert abs(np.mean(n)-1) <= max(1e-11, 8*eps)
    assert abs(np.mean(p)) <= max(1e-11, 8*eps)
    assert abs(np.mean(E)-b) <= max(1e-11, 8*eps)
    assert np.isfinite(n).all() and n.min() > 0
    assert np.isfinite(E).all() and np.isfinite(J).all()
    if zero:
        assert abs(J).max() <= max(1e-9, 32*eps*scales[1])
    return E, J
