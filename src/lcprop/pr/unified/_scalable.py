"""Certified matrix-free connected-2D core; not registered with workflows.

S2/v3 operation order and policy retained. Shared Product equations remain the
physical authority. Only explicit material calls may select this implementation.
"""
import math
from numpy.linalg import LinAlgError
from .operators import bernoulli
from ._positive import carrier, domain_ok, normalized_log_carrier, ARITHMETIC, CarrierDomainError
from ._newton import residual as _residual, diagnose as _diagnose
from ._krylov import Failure, scalar, norm, pcg, gmres


def residual(g, I, q, p, b, electrical):
    return _residual(g, I, q, p, b, electrical, arithmetic=ARITHMETIC if q.dtype == g.xp.float32 else None)


def diagnose(g, I, q, p, b, electrical, zero):
    return _diagnose(g, I, q, p, b, electrical, zero, arithmetic=ARITHMETIC if q.dtype == g.xp.float32 else None)


class Laplacian:
    def __init__(self,g):
        self.g=g;xp=g.xp;dt=g.backend.dtype
        self.off=[dt(-1/h**2) for h in g.spacing]
        center=dt(0)
        for h in g.spacing:center=dt(center+dt(2/h**2))
        self.center=center
    def __call__(self,a,diagonal=None):
        xp=self.g.xp
        out=(self.center if diagonal is None else diagonal)*a
        for j,c in enumerate(self.off):out=out+float(c)*(xp.roll(a,1,axis=j)+xp.roll(a,-1,axis=j))
        return out

def transport_stencil(g,left,right,axis):
    """Wide D@Q face assembly; no scientific carrier narrowing."""
    xp=g.xp;inv=xp.float64(1/g.spacing[axis]);prev=lambda a:xp.roll(a,1,axis=axis)
    center=inv*left-inv*prev(right)
    return center,inv*right,-inv*prev(left)

class WideJacobian:
    def __init__(self,g,I,q,p,b,electrical):
        xp=g.xp;dt=xp.float64;N=g.size
        self.g=g;self.n=carrier(xp,q).astype(xp.float64);self.L=Laplacian(g)
        n=carrier(xp,q);w=I*n if g.backend.precision=='float64' else I.astype(xp.float64)*n;U,V,_=electrical
        self.qcenter=xp.zeros(g.shape,dtype=dt);self.pcenter=xp.zeros(g.shape,dtype=dt)
        self.qoff=[];self.poff=[];self.hcols=[];mq=[];mp=[];mb=[];self.ranges=[]
        for j,h in enumerate(g.spacing):
            drop=g.neighbor(p,j)-p-b[j]*h
            B,D=bernoulli(xp,drop);Bm,Dm=bernoulli(xp,-drop);wr=g.neighbor(w,j)
            k=(D*w+Dm*wr)/h
            left=B*w/h;right=-Bm*wr/h
            qc,qr,ql=transport_stencil(g,left,right,j)
            pc,pr,pl=transport_stencil(g,-k,k,j)
            self.qcenter+=qc;self.pcenter+=pc
            self.qoff.append((qr.astype(xp.float64),ql.astype(xp.float64)))
            self.poff.append((pr.astype(xp.float64),pl.astype(xp.float64)))
            hb=-h*k;inv=dt(1/h)
            self.hcols.append((inv*hb-inv*xp.roll(hb,1,axis=j)).astype(xp.float64))
            scale=dt(1/N)
            mq.append(left*scale+xp.roll(right*scale,1,axis=j))
            mp.append((-k)*scale+xp.roll(k*scale,1,axis=j))
            mb.append(xp.cumsum((hb*scale).ravel(),dtype=dt)[-1])
            self.ranges.append({'k_min':scalar(k.min()),'k_max':scalar(k.max()),'left_min':scalar(left.min()),'left_max':scalar(left.max())})
        self.qcenter=self.qcenter.astype(xp.float64);self.pcenter=self.pcenter.astype(xp.float64)
        self.cq=(V@xp.stack([a.ravel() for a in mq])).astype(xp.float64)
        self.cp=(V@xp.stack([a.ravel() for a in mp])).astype(xp.float64)
        self.cb=(U+V@xp.diag(xp.stack(mb))).astype(xp.float64)
        self.gauge_coefficient=float(dt(1/N))
    def __call__(self,v):
        g=self.g;xp=g.xp;N=g.size
        a=v[:N].reshape(g.shape);p=v[N:2*N].reshape(g.shape);b=v[2*N:]
        d=self.qcenter*a+self.pcenter*p
        for j,((qr,ql),(pr,pl),hc) in enumerate(zip(self.qoff,self.poff,self.hcols)):
            d=d+qr*xp.roll(a,-1,axis=j)+ql*xp.roll(a,1,axis=j)+pr*xp.roll(p,-1,axis=j)+pl*xp.roll(p,1,axis=j)+hc*b[j]
        return xp.concatenate(((self.L(p)-self.n*a).ravel(),d.ravel()[:-1],(xp.sum(p)*self.gauge_coefficient).reshape(1),self.cq@a.ravel()+self.cp@p.ravel()+self.cb@b))

def symbol(g):
    xp=g.xp
    return sum((4*xp.sin(math.pi*xp.arange(n,dtype=xp.float64)/n)**2/h**2).reshape(tuple(n if k==j else 1 for k in range(len(g.shape)))) for j,(n,h) in enumerate(zip(g.shape,g.spacing)))

def preconditioner(g,I,q,electrical,zero):
    xp=g.xp;N=g.size;nbar=scalar(xp.mean(carrier(xp,q)));lam=symbol(g)
    if zero:
        den=lam+nbar
        return lambda r:xp.fft.ifftn(xp.fft.fftn(r.reshape(g.shape))/den).real.ravel()
    w=scalar(xp.mean(I*carrier(xp,q) if g.backend.precision=='float64' else I.astype(xp.float64)*carrier(xp,q)));U,V,_=electrical
    den=w*lam*(1+lam/nbar);den[(0,)*len(g.shape)]=1
    # This is the S1 uniform bordered inverse; floating stencil rounding is
    # intentionally not forced into its Fourier approximation.
    def apply(r):
        gg=r[:N].reshape(g.shape)
        dd=xp.concatenate((r[N:2*N-1],(-xp.sum(r[N:2*N-1])).reshape(1))).reshape(g.shape)
        ph=(xp.fft.fftn(dd)+(w/nbar)*lam*xp.fft.fftn(gg))/den;ph[(0,)*len(g.shape)]=0
        p=xp.fft.ifftn(ph).real+r[2*N-1]
        lp=xp.fft.ifftn(lam*xp.fft.fftn(p)).real
        a=(lp-gg)/nbar
        b=xp.linalg.solve(U.astype(xp.float64)+w*V.astype(xp.float64),r[2*N:])
        return xp.concatenate((a.ravel(),p.ravel(),b))
    return apply

def _numerical_errors(xp):
    # CuPy normally aliases NumPy's LinAlgError; also accept its explicit type.
    # API/programming/CUDA-runtime/resource errors are deliberately not caught.
    return (Failure, CarrierDomainError, LinAlgError,
            getattr(xp.linalg, 'LinAlgError', LinAlgError),
            FloatingPointError, OverflowError)


def _failure_evidence(original, *, stage, trace, state=None, iteration=None,
                      state_iteration=None, diagnostics=None,
                      diagnostics_iteration=None, linear_solver=None):
    """Attach bounded evidence only. Never evaluate, copy or repair an iterate."""
    failure=original if isinstance(original, Failure) else Failure(str(original))
    failure.state=state
    failure.trace=trace
    failure.stage='carrier_domain' if isinstance(original, CarrierDomainError) else stage
    failure.operation_stage=stage
    failure.iteration=iteration
    failure.state_iteration=state_iteration
    failure.last_diagnostics=diagnostics
    failure.diagnostics_iteration=diagnostics_iteration
    failure.linear_solver=linear_solver
    failure.inner_iterations=len(trace[-1]['linear']) if trace and 'linear' in trace[-1] else 0
    return failure


def solve(g,intensity,electrical,*,zero):
    xp=g.xp;B=g.backend
    trace=[];initial_norm=None
    stage='initialization';iteration=None;last_state=None;state_iteration=None
    last_diagnostics=None;diagnostics_iteration=None;linear_solver=None
    numerical_errors=_numerical_errors(xp)
    try:
        I=B.array(intensity).copy();U,V,target=electrical
        if I.shape!=g.shape or not bool(B.status([xp.all(xp.isfinite(I))&xp.all(I>0)])[0]):raise ValueError('invalid intensity')
        b=B.direct_dense(U+xp.mean(I)*V,target);p=xp.zeros(g.shape,dtype=B.dtype);q=xp.zeros_like(p)
        for it in range(61):
            iteration=it;linear_solver=None;stage='initialization' if it==0 else 'newton_correction'
            if zero:q=normalized_log_carrier(g,p,I)
            stage='diagnosis'
            check=diagnose(g,I,q,p,b,electrical,zero)
            last_diagnostics=check;diagnostics_iteration=it
            row={'iteration':it,**check};trace.append(row)
            if check['values'][10] == 1 and check['values'][8] > 0:
                last_state={'I':I,'q':q,'psi':p,'b':b};state_iteration=it
            if check['passed'] and (B.precision=='float64' or it==0):return {'q':q,'psi':p,'b':b,'trace':trace}
            stage='newton_correction'
            if it==60:raise Failure('Newton bound')
            stage='residual'
            if zero:
                n=carrier(xp,q);merit=g.poisson(p)-n+1;rhs=-(merit-xp.mean(merit)).ravel().astype(xp.float64)
                stage='coefficient_construction'
                L=Laplacian(g);diagonal=(B.dtype(L.center)+n).astype(xp.float64)
                action=lambda v:L(v.reshape(g.shape),diagonal).ravel()
            else:
                merit=residual(g,I,q,p,b,electrical);rhs=-merit.astype(xp.float64)
                stage='coefficient_construction'
                action=WideJacobian(g,I,q,p,b,electrical);row['coefficient_ranges']=action.ranges
            stage='residual'
            fnorm=norm(xp,merit)
            if initial_norm is None:initial_norm=max(fnorm,1e-300)
            eta=min(.1,max(1e-8,.1*math.sqrt(fnorm/initial_norm)))
            row.update(eta=eta,merit_norm=fnorm,linear=[])
            linear_solver='pcg' if zero else 'gmres'
            stage='preconditioner_setup'
            M=preconditioner(g,I,q,electrical,zero)
            def observed_preconditioner(r):
                # No extra evaluation/copy: locate failures inside Krylov calls.
                nonlocal stage
                previous_stage=stage;stage='preconditioner_application'
                result=M(r)
                stage=previous_stage
                return result
            stage=linear_solver
            step=(pcg if zero else gmres)(xp,action,rhs,observed_preconditioner,eta,row['linear'])
            stage='newton_correction'
            row['final_true_relative']=norm(xp,action(step)-rhs)/max(norm(xp,rhs),1e-300)
            if row['final_true_relative']>eta:raise Failure('true linear residual failed')
            step=step.astype(B.dtype)
            row['postcast_true_relative']=norm(xp,action(step.astype(xp.float64))-rhs)/max(norm(xp,rhs),1e-300)
            if row['postcast_true_relative']>eta:raise Failure('post-cast true linear residual failed')
            if zero:dp=g.gauge(step.reshape(g.shape));dq=db=None
            else:dq,dp,db=step[:g.size].reshape(g.shape),step[g.size:2*g.size].reshape(g.shape),step[2*g.size:]
            correction=B.status([xp.max(abs(dp))/(1+xp.max(abs(p)))])[0]
            cc=0. if zero else B.status([xp.maximum(xp.max(abs(dq)),xp.max(abs(db)))])[0]
            row.update(relative_potential_correction=float(correction),q_b_correction=float(cc))
            if B.precision=='float32' and check['passed'] and correction<=2e-5 and cc<=2e-6:return {'q':q,'psi':p,'b':b,'trace':trace}
            if not bool(B.status([xp.all(xp.isfinite(dp))])[0]):raise Failure('nonfinite correction')
            stage='globalization'
            alpha=1.;row['trials']=[]
            for bt in range(30):
                pt=p+alpha*dp
                if zero:pt=g.gauge(pt);qt=normalized_log_carrier(g,pt,I);bb=b
                else:qt=q+alpha*dq;bb=b+alpha*db
                valid=True if B.precision=='float64' else domain_ok(xp,qt)
                if valid:
                    nt=carrier(xp,qt);valid=bool(B.status([xp.all(xp.isfinite(nt))&xp.all(nt>0)])[0])
                trial={'alpha':alpha,'positive_finite':valid};row['trials'].append(trial)
                if valid:
                    rr=g.poisson(pt)-nt+1 if zero else residual(g,I,qt,pt,bb,electrical)
                    rn=norm(xp,rr);trial['norm']=rn
                    if rn<=(1-1e-4*alpha)*fnorm:
                        p,q,b=pt,qt,bb;row.update(alpha=alpha,halvings=bt)
                        if math.isfinite(rn):
                            last_state={'I':I,'q':q,'psi':p,'b':b};state_iteration=it+1
                        break
                alpha*=.5
            else:raise Failure('Armijo bound')
        raise AssertionError('unreachable')
    except numerical_errors as original:
        failure=_failure_evidence(original, stage=stage, trace=trace,
            state=last_state, iteration=iteration, state_iteration=state_iteration,
            diagnostics=last_diagnostics, diagnostics_iteration=diagnostics_iteration,
            linear_solver=linear_solver)
        if failure is original:
            raise
        raise failure from original


def solve_material(intensity, *, closure, solver):
    """Explicit standalone material API; unavailable to existing dispatch.

    Returns owned canonical state plus bounded iteration/physical evidence.
    Borrowed input must remain unchanged during this synchronous call. Failure
    retains the last solver iterate and trace, never an accepted workflow state.
    No product registration, resource planning or persistence is performed here.
    """
    from contextlib import nullcontext
    from .state import PRTransportIntensity, PRUnifiedMaterialState
    from ._backend import MaterialBackend
    from ._positive import Geometry
    from .solver_specs import PRUnifiedSolverSpec, SCALABLE
    from .specs import UNBIASED, FIXED_FIELD, PRESCRIBED_CURRENT
    if not isinstance(intensity, PRTransportIntensity):
        raise TypeError('PRTransportIntensity required')
    intensity.validate_structure()
    if not isinstance(solver, PRUnifiedSolverSpec) or solver.identity != SCALABLE:
        raise ValueError('explicit scalable solver specification required')
    solver.validate_material(intensity.spatial, closure, intensity.precision)
    context = intensity.values.device if intensity.backend == 'cupy' else nullcontext()
    with context:
        backend = MaterialBackend(intensity.backend, intensity.precision.state_dtype)
        if closure.identity in (UNBIASED, FIXED_FIELD):
            U, V = ((1., 0.), (0., 1.)), ((0., 0.), (0., 0.))
        elif closure.identity == PRESCRIBED_CURRENT:
            U, V = ((0., 0.), (0., 0.)), ((1., 0.), (0., 1.))
        else:
            U, V = ((1., 0.), (0., 0.)), ((0., 0.), (0., 1.))
        try:
            electrical = tuple(backend.array(a) for a in (U, V, closure.target))
            geometry = Geometry(intensity.spatial.active_shape, intensity.spatial.normalized_lengths, backend)
            solution = solve(geometry, intensity.values, electrical, zero=closure.identity == UNBIASED)
        except _numerical_errors(backend.xp) as original:
            failure=original if isinstance(original, Failure) and hasattr(original, 'stage') else _failure_evidence(
                original, stage='initialization', trace=[])
            # Frozen metadata only; no scientific-array export or new schema.
            failure.solver = solver
            failure.closure = closure
            failure.precision = intensity.precision
            failure.backend = intensity.backend
            if failure is original:
                raise
            raise failure from original
        state = PRUnifiedMaterialState(solution['q'], solution['psi'], solution['b'],
            intensity.spatial, closure, intensity.precision, intensity.backend)
        state.validate_structure()
        return state, solution['trace']
