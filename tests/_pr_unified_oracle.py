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


def evaluate_plane(intensity, q, psi, b, lengths, fixed_components, target, zero, precision):
    """2D integrated-edge oracle with independent index pairing, no core imports."""
    I, q, p = [np.array(a, dtype=np.longdouble) for a in (intensity, q, psi)]
    n = np.exp(q)
    gauss = -(n-1)
    divergence = np.zeros_like(p)
    fields, currents = [], []
    logw = q+np.log(I)
    for axis in range(2):
        count = p.shape[axis]
        h = np.longdouble(lengths[axis])/count
        right, left = (np.arange(count)+1)%count, (np.arange(count)-1)%count
        drop = np.take(p, right, axis=axis)-p-np.longdouble(b[axis])*h
        E = -drop/h
        integral = np.ones_like(p)
        nonzero = drop != 0
        integral[nonzero] = np.expm1(drop[nonzero])/drop[nonzero]
        J = -np.exp(logw)*np.expm1(np.take(logw, right, axis=axis)-logw+drop)/(h*integral)
        gauss += (E-np.take(E, left, axis=axis))/h
        divergence += (J-np.take(J, left, axis=axis))/h
        fields.append(E)
        currents.append(J)
    closure = np.array([(np.mean(fields[j]) if fixed_components[j] else np.mean(currents[j]))-target[j] for j in range(2)])
    eps = np.finfo(np.float32).eps if precision == 'float32' else 0.
    invh2 = sum((np.longdouble(naxis)/L)**2 for naxis,L in zip(p.shape,lengths))
    scales = (1+np.max(n)+4*np.max(abs(p))*invh2,
              1+4*np.max(n*I)*invh2+sum(2*np.max(abs(j))*count/L for j,count,L in zip(currents,p.shape,lengths)),1.)
    for residual, scale in zip((gauss, divergence, closure), scales):
        assert np.sqrt(np.mean(residual**2)) <= max(1e-10,8*eps*scale)
        assert np.max(abs(residual)) <= max(1e-9,32*eps*scale)
    assert abs(np.mean(n)-1) <= max(1e-11,8*eps)
    assert abs(np.mean(p)) <= max(1e-11,8*eps)
    assert np.isfinite(n).all() and n.min()>0
    for j in range(2):
        assert np.isfinite(fields[j]).all() and np.isfinite(currents[j]).all()
        assert abs(np.mean(fields[j])-b[j]) <= max(1e-11,8*eps)
        if zero:
            assert abs(currents[j]).max() <= max(1e-9,32*eps*scales[1])
    return fields,currents
