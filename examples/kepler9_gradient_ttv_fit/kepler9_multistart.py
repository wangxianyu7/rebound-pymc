"""Parallel multistart for the real Kepler-9 TTV fit (rebound-pymc, JAC=bsjac).

Each random start runs the same whfast-forward + BS-analytic-Jacobian TRF fit as
kepler9_rebound_pymc_real_fit.py; starts are independent so we fan them out over CPU
cores with multiprocessing.Pool (each worker builds the compiled model once in an
initializer). Usage:  python kepler9_multistart.py [N]

Benchmark (N=40, 12-core Mac): serial 80.1s, Pool(8) 21.8s (3.7x), best chi2=532.6
(Borsato 2019), ~20/40 random starts land in the correct basin. Comparable to jnkepler's
40-start wall on CPU; jnkepler pulls ahead only on GPU / very large start counts where it
vectorizes starts into one JIT graph.  env: normal.
"""
import os, time, numpy as np
os.environ.setdefault("OMP_NUM_THREADS","1")  # avoid oversubscription in Pool workers
DPY=58.13244087623439; MEARTH=3.0027e-6
HERE="/Users/wangxianyu/Program/Github/rebound-pymc/examples/kepler9_gradient_ttv_fit"

# ---- data (module level, cheap) ----
def load(koi):
    d=np.genfromtxt(f"{HERE}/holczer2016_{koi}.csv",delimiter=",",names=True); m=d["flag_outlier"]==0
    return d["N"][m].astype(int), d["tn_BJD_off"][m]+d["OC_min"][m]/1440., np.maximum(d["eOC_min"][m],0.5)/1440.
Nb,tb,eb=load("377_01"); Nc,tc,ec=load("377_02")
t_all=np.concatenate([tb,tc]); sig_a=np.concatenate([eb,ec]); plan_a=np.array([0]*len(tb)+[1]*len(tc))
order=np.argsort(t_all); t_obs,sig,plan=t_all[order],sig_a[order],plan_a[order]; row=np.arange(len(t_obs))
Pb0=19.2389; t_start=min(tb.min(),tc.min())-1.5*Pb0
def lin_eph(N,t):
    A=np.vstack([np.ones_like(N),N]).T; c=np.linalg.lstsq(A,t,rcond=None)[0]; return c[1],c[0]+c[1]*N[0]
Pb_d,t0b=lin_eph(Nb,tb); Pc_d,t0c=lin_eph(Nc,tc)
lo=np.array([np.log(1e-6),np.log(1e-6),-.2,-.2,-.2,-.2,Pb_d*0.97,Pc_d*0.97,t0b-0.6,t0c-0.6])
hi=np.array([np.log(6e-4),np.log(6e-4),.2,.2,.2,.2,Pb_d*1.03,Pc_d*1.03,t0b+0.6,t0c+0.6])

G={}
def build():
    import pytensor, pytensor.tensor as pt
    from rebound_pymc.integrate import IntegrateOp
    from jacobi_ic import barycentric_coords0
    from scipy.optimize import least_squares
    th=pt.dvector("th"); tin=pt.dvector("tin")
    pm=[pt.exp(th[0]),pt.exp(th[1])]; ec=[th[2],th[4]]; es=[th[3],th[5]]; pe=[th[6],th[7]]; ti=[th[8],th[9]]
    ms,c0=barycentric_coords0(pe,ec,es,ti,pm,pt.constant(1.0),t_start)
    c0r=pt.concatenate([c0[:,:3],c0[:,3:]*DPY],axis=1)
    opw=IntegrateOp(t=0.0,dt=0.01,integrator="whfast")
    o,_=opw(ms,c0r,(tin-t_start)/DPY)
    fxv=pytensor.function([th,tin],[o[row,plan+1,0]-o[row,0,0],(o[row,plan+1,3]-o[row,0,3])/DPY])
    fz=pytensor.function([th],pt.concatenate([ms[:,None],c0r],axis=1))
    mv=pt.dvector("m"); cv=pt.dmatrix("c"); tv=pt.dvector("t")
    opb=IntegrateOp(t=0.0,dt=0.05,integrator="bs"); ob,jb=opb(mv,cv,tv)
    foj=pytensor.function([mv,cv,tv],[ob[row,plan+1,0]-ob[row,0,0],(ob[row,plan+1,3]-ob[row,0,3])/DPY,jb])
    G.update(fxv=fxv,fz=fz,foj=foj,ls=least_squares)

def run_start(seed_vec):
    fxv,fz,foj,ls=G["fxv"],G["fz"],G["foj"],G["ls"]
    def tmodel(theta,nit=3):
        t=t_obs.copy()
        for _ in range(nit): x,v=fxv(theta,t); t=t-x/v
        return t
    def res(theta): return (tmodel(theta)-t_obs)/sig
    def icjac(theta,h=1e-7):
        J=np.empty((21,10))
        for i in range(10):
            d=np.zeros(10); d[i]=h; J[:,i]=(fz(theta+d).reshape(-1)-fz(theta-d).reshape(-1))/(2*h)
        return J
    def rjac(theta):
        t=tmodel(theta); Z=fz(theta); m,c=Z[:,0].copy(),Z[:,1:].copy()
        x,v,Jj=foj(m,c,(t-t_start)/DPY)
        dxdz=np.stack([Jj[k,:,:,plan[k]+1,0]-Jj[k,:,:,0,0] for k in range(len(t))],0).reshape(len(t),21)
        return -((dxdz@icjac(theta))/v[:,None])/sig[:,None]
    try:
        sol=ls(res,seed_vec,jac=rjac,bounds=(lo,hi),method="trf",x_scale="jac",
               xtol=1e-10,ftol=1e-10,max_nfev=120)
        return float(np.sum(sol.fun**2)), sol.x
    except Exception as e:
        return 1e30, seed_vec

def make_seeds(n):
    rng=np.random.default_rng(0); S=[]
    for _ in range(n):
        s=np.empty(10)
        s[0]=np.log(rng.uniform(1e-6,6e-4)); s[1]=np.log(rng.uniform(1e-6,6e-4))
        s[2:6]=rng.uniform(-0.1,0.1,4)
        s[6]=Pb_d; s[7]=Pc_d; s[8]=t0b; s[9]=t0c
        S.append(s)
    return S

if __name__=="__main__":
    import sys, multiprocessing as mp
    N=int(sys.argv[1]) if len(sys.argv)>1 else 40
    seeds=make_seeds(N)
    # serial
    build()
    t0=time.time(); res_s=[run_start(s) for s in seeds]; ts=time.time()-t0
    best_s=min(res_s,key=lambda r:r[0]); ok_s=sum(1 for r in res_s if r[0]<600)
    print(f"SERIAL  N={N}: wall={ts:.1f}s  best chi2={best_s[0]:.1f}  converged={ok_s}/{N}",flush=True)
    # parallel
    ncpu=min(8,mp.cpu_count())
    t0=time.time()
    with mp.Pool(ncpu,initializer=build) as pool:
        res_p=pool.map(run_start,seeds)
    tp=time.time()-t0
    best_p=min(res_p,key=lambda r:r[0]); ok_p=sum(1 for r in res_p if r[0]<600)
    print(f"POOL({ncpu}) N={N}: wall={tp:.1f}s  best chi2={best_p[0]:.1f}  converged={ok_p}/{N}  speedup={ts/tp:.1f}x",flush=True)
    mm=np.exp(best_p[1][:2])/MEARTH
    print(f"best: m_b={mm[0]:.1f} m_c={mm[1]:.1f} Me",flush=True)
    print("DONE")
