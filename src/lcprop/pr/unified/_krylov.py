"""Frozen S2/v3 Krylov policy. Backend-resident vectors; scalar diagnostics only."""
import math

class Failure(RuntimeError):
    def __init__(self, reason, state=None, trace=None):
        super().__init__(reason)
        self.state, self.trace = state, trace

def scalar(x):
    return float(x.item())

def norm(xp, x):
    return scalar(xp.linalg.norm(x.ravel()))

def pcg(xp,A,b,M,eta,record):
    x=xp.zeros_like(b);r=b.copy();z=M(r);p=z.copy();rz=scalar(xp.dot(r,z));bn=norm(xp,b)
    if bn==0:return x
    history=[]
    for it in range(600):
        ap=A(p);curv=scalar(xp.dot(p,ap))
        if not math.isfinite(curv) or curv<=0:raise Failure('PCG curvature breakdown')
        alpha=rz/curv;x=x+alpha*p;r=r-alpha*ap
        true=norm(xp,b-A(x));pre=norm(xp,M(b-A(x)));history.append(true)
        record.append({'iteration':it+1,'true_relative':true/bn,'preconditioned_norm':pre})
        if true<=eta*bn:return x
        if it>=59 and true>=.99*history[-31]:raise Failure('PCG stagnation')
        z=M(r);nrz=scalar(xp.dot(r,z))
        if not math.isfinite(nrz) or rz==0:raise Failure('PCG scalar breakdown')
        p=z+(nrz/rz)*p;rz=nrz
    raise Failure('PCG iteration bound')

def gmres(xp,A,b,M,eta,record):
    """Restarted left-preconditioned GMRES; two-pass MGS, true checks each step."""
    x=xp.zeros_like(b);bn=norm(xp,b);stalled=0
    if bn==0:return x
    for cycle in range(20):
        true0=norm(xp,b-A(x));r=M(b-A(x));beta=norm(xp,r)
        if beta==0:raise Failure('Preconditioner annihilated nonzero residual')
        basis=xp.zeros((b.size,61),dtype=xp.float64);H=xp.zeros((61,60),dtype=xp.float64);basis[:,0]=r/beta
        rhs=xp.zeros(61,dtype=xp.float64);rhs[0]=beta
        for j in range(60):
            w=M(A(basis[:,j]));wn=norm(xp,w)
            for repeat in range(2):
                for i in range(j+1):
                    h=xp.dot(basis[:,i],w);H[i,j]+=h;w=w-h*basis[:,i]
            sub=norm(xp,w);H[j+1,j]=sub
            coef=xp.linalg.lstsq(H[:j+2,:j+1],rhs[:j+2],rcond=None)[0]
            trial=x+basis[:,:j+1]@coef;rr=b-A(trial);true=norm(xp,rr)
            record.append({'iteration':cycle*60+j+1,'restart':cycle,'true_relative':true/bn,'preconditioned_norm':norm(xp,M(rr)),'arnoldi_subdiagonal':sub})
            if true<=eta*bn:return trial
            if not math.isfinite(sub) or sub<=64*2.**-52*wn:raise Failure('Arnoldi breakdown without true convergence')
            basis[:,j+1]=w/sub
        x=trial;stalled=stalled+1 if true>=.99*true0 else 0
        if stalled>=2:raise Failure('GMRES restart stagnation')
    raise Failure('GMRES iteration bound')
