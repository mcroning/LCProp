"""Frozen v3 independently indexed face oracle. No Product/S2/operator import.
Carrier64, original state32 face-drop/Bernoulli return rounding, wide products.
"""
import math
import numpy as np
IDENTITY='state32_carrier64_coeff64_linear64_bernoulli64_v3'
F=np.float32

def bd(x):
    x=float(x)
    if abs(x)<.001:return F(1-x/2+x*x/12-x**4/720+x**6/30240),F(-.5+x/6-x**3/180+x**5/5040)
    if x>50:
        e=math.exp(-x);return F(x*e),F((1-x)*e)
    if x< -50:return F(-x),F(-1)
    em=math.expm1(x);return F(x/em),F((em-x*(em+1))/(em*em))

def evaluate(I,q,p,b,lengths,U,V,target,v):
    shape=I.shape;N=I.size;n=np.exp(q.astype(float));w=I.astype(float)*n
    dq=v[:N].reshape(shape);dp=v[N:2*N].reshape(shape);db=v[2*N:]
    flux=[];dflux=[];poisson=np.zeros(shape,np.float32);lp=np.zeros(shape,float);symbol=np.zeros(shape,float)
    center=F(0);offs=[]
    for axis in range(len(shape)):
        h=lengths[axis]/shape[axis];ff=np.empty(shape,float);df=np.empty(shape,float)
        for pos in np.ndindex(shape):
            nxt=list(pos);nxt[axis]=(nxt[axis]+1)%shape[axis];nxt=tuple(nxt)
            drop=F(F(p[nxt]-p[pos])-F(b[axis]*F(h)))
            B,D=bd(drop);Bm,Dm=bd(F(-drop));k=(float(D)*w[pos]+float(Dm)*w[nxt])/h
            ff[pos]=(float(B)*w[pos]-float(Bm)*w[nxt])/h
            df[pos]=float(B)*w[pos]/h*dq[pos]-float(Bm)*w[nxt]/h*dq[nxt]+k*(dp[nxt]-dp[pos]-h*db[axis])
        flux.append(ff);dflux.append(df)
        grad=(np.roll(p,-1,axis=axis)-p)/F(h)
        poisson+=(grad-np.roll(grad,1,axis=axis))/F(h)
        center=F(center+F(2/h**2));offs.append(F(-1/h**2))
        values=4*np.sin(math.pi*np.arange(shape[axis],dtype=float)/shape[axis])**2/h**2
        symbol+=values.reshape(tuple(shape[axis] if j==axis else 1 for j in range(len(shape))))
    poisson=-poisson
    div=np.zeros(shape,float);ddiv=np.zeros(shape,float)
    for axis,(f,df) in enumerate(zip(flux,dflux)):
        h=lengths[axis]/shape[axis];div+=(f-np.roll(f,1,axis=axis))/h;ddiv+=(df-np.roll(df,1,axis=axis))/h
    def L(x,diag):
        y=diag*x
        for axis,c in enumerate(offs):y+=float(c)*(np.roll(x,1,axis=axis)+np.roll(x,-1,axis=axis))
        return y
    closure=U.astype(float)@b.astype(float)+V.astype(float)@np.array([f.mean() for f in flux])-target
    Jv=np.r_[(L(dp,float(center))-n*dq).ravel(),ddiv.ravel()[:-1],[dp.mean()],U.astype(float)@db+V.astype(float)@np.array([f.mean() for f in dflux])]
    A0=L(dq,float(center)+n).ravel();nb=n.mean();wb=w.mean()
    den=wb*symbol*(1+symbol/nb);den[(0,)*len(shape)]=1
    dd=np.r_[v[N:2*N-1],-np.sum(v[N:2*N-1])].reshape(shape)
    ph=(np.fft.fftn(dd)+(wb/nb)*symbol*np.fft.fftn(dq))/den;ph[(0,)*len(shape)]=0
    ps=np.fft.ifftn(ph).real+v[2*N-1];ls=np.fft.ifftn(symbol*np.fft.fftn(ps)).real
    mat=U.astype(float)+wb*V.astype(float);z=v[2*N:]
    if len(shape)==2:
        det=mat[0,0]*mat[1,1]-mat[0,1]*mat[1,0]
        bs=np.array([(mat[1,1]*z[0]-mat[0,1]*z[1])/det,(-mat[1,0]*z[0]+mat[0,0]*z[1])/det])
    else:bs=z/mat[0,0]
    return dict(carrier=n,weight=w,Gauss=poisson-n+1,neutrality=np.asarray(n.mean()-1),flux=np.array(flux),divergence=div,closure=closure,Jv=Jv,A0=A0,Schur=np.r_[((ls-dq)/nb).ravel(),ps.ravel(),bs],FFT=np.fft.ifftn(np.fft.fftn(dq)/(symbol+nb)).real.ravel(),means=np.array([nb,wb]))
