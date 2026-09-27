"""Fit the REAL Kepler-9 TTVs (Holczer 2016) with rebound-pymc's differentiable
C N-body integrator (IntegrateOp) + analytic gradients.

Two fixes made real data work (see jacobi_ic.py and the README):
  1. Initial conditions must use jnkepler's JACOBI convention (outer planet's Kepler
     mass = star + all interior planets), NOT exoplanet's KeplerianOrbit (independent
     2-body per planet). That IC error is fatal for the 2:1 resonance. jacobi_ic.py
     ports jnkepler's construction to pytensor (verified to machine precision).
  2. Feed the integrator a GLOBALLY TIME-SORTED set of epochs so it never integrates
     backward (unsorted -> 1400-day back-jumps -> forward eval went from >400s to 0.3s).

Transit times solve x_sky(t;theta)=0. We find t_model by numeric Newton (pure forward,
cheap) and get the residual Jacobian analytically by the implicit function theorem:
    d t_model/d theta = -(d x_sky/d theta) / vx_sky      (evaluated at t_model)
The (d x_sky/d theta) term is one IntegrateOp Jacobian from rebound-pymc -- so this is a
genuine gradient-based N-body TTV fit, just without the graph blow-up of differentiating
through the unrolled Newton loop.  env: normal.
"""
import os, numpy as np, time, pytensor, pytensor.tensor as pt
from scipy.optimize import least_squares
from rebound_pymc.integrate import IntegrateOp
from jacobi_ic import barycentric_coords0

# Configs (all reproduce Borsato 2019 from a deliberately-off seed):
#   * bsjac (DEFAULT): INTEG=whfast forward + BS analytic Jacobian -> ~1.5 s. Fastest AND
#     fully-analytic/correct. whfast gives the fast residuals; BS (Gragg-Bulirsch-Stoer)
#     supplies the analytic N-body Jacobian for ALL columns -- BS integrates in inertial
#     coords (like ias15) so its mass variations are correct, unlike whfast, and it is ~5x
#     faster than ias15 (op-level jaccheck: BS mass&coord relerr ~1e-3 vs whfast's 117 on mass).
#   * hybrid: INTEG=whfast, JAC=hybrid -> ~2.4 s. Analytic for 8 non-mass cols + finite-diff
#     the 2 mass cols (avoids the whfast mass bug without BS).
#   * analytic: INTEG=bs or ias15, JAC=analytic -> ~5.4 s (bs) / ~58 s (ias15). Fully analytic.
#   * 2-point: JAC=2-point -> ~4.3 s (whfast). Fully numeric.
# WHFast caveat: its variational Jacobian is exact for coordinate derivatives but WRONG for
# d/d(mass) -- a REBOUND-internal limitation (whfast integrates in Jacobi coords; mass enters
# the transform). dt-independent. Hence BS or ias15 for a fully-analytic mass Jacobian.
JAC = os.environ.get("JAC", "bsjac")

DPY = 58.13244087623439           # day per (yr/2pi); makes G=1 in AU,Msun
MEARTH = 3.0027e-6; MJUP = 9.545942e-4
HERE = "/Users/wangxianyu/Program/Github/rebound-pymc/examples/kepler9_gradient_ttv_fit"

# ---- data ----------------------------------------------------------------
def load(koi):
    d = np.genfromtxt(f"{HERE}/holczer2016_{koi}.csv", delimiter=",", names=True)
    m = d["flag_outlier"] == 0
    return (d["N"][m].astype(int), d["tn_BJD_off"][m] + d["OC_min"][m] / 1440.,
            np.maximum(d["eOC_min"][m], 0.5) / 1440.)
Nb, tb, eb = load("377_01"); Nc, tc, ec = load("377_02")
t_all = np.concatenate([tb, tc]); sig_a = np.concatenate([eb, ec])
plan_a = np.array([0]*len(tb) + [1]*len(tc))
order = np.argsort(t_all)                              # GLOBAL sort -> monotonic integration
t_obs, sig, plan = t_all[order], sig_a[order], plan_a[order]
row = np.arange(len(t_obs))
Pb0, Pc0 = 19.2389, 38.985
t_start = min(tb.min(), tc.min()) - 1.5*Pb0           # same epoch as jnkepler run

# ---- symbolic model: x_sky and vx_sky at given input epochs --------------
# theta = [lnmb, lnmc, ecb, esb, ecc, esc, Pb, Pc, ticb, ticc]
th = pt.dvector("theta"); tin = pt.dvector("tin")
pmass = [pt.exp(th[0]), pt.exp(th[1])]
ecosw = [th[2], th[4]]; esinw = [th[3], th[5]]
period = [th[6], th[7]]; tic = [th[8], th[9]]
masses, c0 = barycentric_coords0(period, ecosw, esinw, tic, pmass, pt.constant(1.0), t_start)
c0_reb = pt.concatenate([c0[:, :3], c0[:, 3:] * DPY], axis=1)   # vel AU/day -> AU/(yr/2pi)
INTEG = os.environ.get("INTEG", "whfast")            # whfast (fixed-step, fast) or ias15
DT = float(os.environ.get("DT", "0.01"))             # step in yr/2pi (~0.58 d); whfast@0.01 matches ias15
                                                     # rms to <0.03 min at ~30x the speed (benchmarked)
op = IntegrateOp(t=0.0, dt=DT, integrator=INTEG)
out, _ = op(masses, c0_reb, (tin - t_start) / DPY)
xsky = out[row, plan + 1, 0] - out[row, 0, 0]        # (N,) AU
vxsky = (out[row, plan + 1, 3] - out[row, 0, 3]) / DPY  # (N,) AU/day

f_xv = pytensor.function([th, tin], [xsky, vxsky])          # forward: x_sky, vx_sky at epochs

# --- analytic residual Jacobian: rebound-pymc IntegrateOp's own jac OUTPUT (no pt.grad!) ---
# The IntegrateOp returns (coords, jac) where jac[t,i,p,j,b] = d coords[t,j,b]/d input[i,p]
# (input p: 0=mass, 1..6=coords0). We contract that analytic N-body Jacobian with a numeric
# Kepler-IC jacobian dz/dtheta (21x10, cheap; pytensor can't C-compile the reverse graph
# through the 15-iter unrolled Kepler solver). By the implicit function theorem,
#   d resid_k/d theta = -(d xsky_k/d theta) / (vx_k * sig_k)  at the model transit time.
# Jaccheck vs finite differences: ratio 1.0000.
f_z = pytensor.function([th], pt.concatenate([masses[:, None], c0_reb], axis=1))  # theta -> z(3,7)
mv = pt.dvector("m"); cv = pt.dmatrix("c"); tv = pt.dvector("t")
_out, _jac = op(mv, cv, tv)
_xs = _out[row, plan + 1, 0] - _out[row, 0, 0]
_vx = (_out[row, plan + 1, 3] - _out[row, 0, 3]) / DPY
f_oj = pytensor.function([mv, cv, tv], [_xs, _vx, _jac])    # x_sky, vx_sky, full N-body jac

# BS variational Jacobian: BS integrates in inertial coords (like ias15), so unlike whfast
# its mass variations are CORRECT (op-level jaccheck: mass & coord relerr ~1e-3 vs whfast's
# 117 on mass). ~5x faster than ias15. Used for a fully-analytic Jacobian while the fast
# whfast forward supplies the residuals (JAC=bsjac).
op_bs = IntegrateOp(t=0.0, dt=0.05, integrator="bs")
_ob, _jb = op_bs(mv, cv, tv)
f_oj_bs = pytensor.function([mv, cv, tv],
                            [_ob[row, plan+1, 0] - _ob[row, 0, 0],
                             (_ob[row, plan+1, 3] - _ob[row, 0, 3]) / DPY, _jb])

def split(theta): Z = f_z(theta); return Z[:, 0].copy(), Z[:, 1:].copy()

nfev = [0]
NIT = int(os.environ.get("NIT", "3"))                 # Newton steps for transit finding (3 = machine precision)
def t_model(theta, nit=NIT):                          # Newton uses jac-FREE f_xv (no wasted variational jac)
    t = t_obs.copy()
    for _ in range(nit):
        x, v = f_xv(theta, t); t = t - x / v
    return t
def res_fn(theta):
    nfev[0] += 1
    return (t_model(theta) - t_obs) / sig
def chi2_fn(theta):
    return float(np.sum(res_fn(theta)**2))
def ic_jac(theta, h=1e-7):                                  # numeric dz/dtheta (21,10)
    J = np.empty((21, 10))
    for i in range(10):
        d = np.zeros(10); d[i] = h
        J[:, i] = (f_z(theta + d).reshape(-1) - f_z(theta - d).reshape(-1)) / (2*h)
    return J
def res_jac(theta, oj=f_oj):
    t = t_model(theta); m, c = split(theta)
    x, v, J = oj(m, c, (t - t_start) / DPY)                # J: (N,3,7,3,6)
    dxdz = np.stack([J[k, :, :, plan[k]+1, 0] - J[k, :, :, 0, 0]
                     for k in range(len(t))], axis=0).reshape(len(t), 21)  # d xsky/d z
    dxdtheta = dxdz @ ic_jac(theta)                        # (N,10)
    return -(dxdtheta / v[:, None]) / sig[:, None]         # (N,10) d resid/d theta
def res_jac_bsjac(theta):                                  # whfast residuals + BS analytic jac (all cols)
    return res_jac(theta, oj=f_oj_bs)

# HYBRID: analytic for all columns except the 2 mass params (theta[0],theta[1]).
# WHFast's variational eqns are wrong ONLY for d/d(mass) (verified: coord-derivs exact,
# mass-derivs relerr>1 regardless of dt -- a REBOUND-internal limitation of the whfast
# mass variation, not fixable from rebound-pymc). Analytic columns 2-9 (e,P,tic) don't
# touch the mass variation (ic_jac mass-rows are 0 there), so they stay exact; only the
# 2 mass columns are recomputed by finite difference. ~3 integrations/jac vs ~11 numeric.
MASS_COLS = [0, 1]
def res_jac_hybrid(theta):
    Jm = res_jac(theta)                                    # analytic; cols 2-9 exact under whfast
    r0 = res_fn(theta)
    for i in MASS_COLS:
        h = 1e-6; d = np.zeros(10); d[i] = h
        Jm[:, i] = (res_fn(theta + d) - r0) / h            # forward diff for mass column
    return Jm

# ---- fit: seed P/tic from data linear ephemeris; masses & ecc OFF --------
def lin_eph(N, t):
    A = np.vstack([np.ones_like(N), N]).T; c = np.linalg.lstsq(A, t, rcond=None)[0]
    return c[1], c[0] + c[1]*N[0]
Pb_d, t0b = lin_eph(Nb, tb); Pc_d, t0c = lin_eph(Nc, tc)
seed = np.array([np.log(80*MEARTH), np.log(60*MEARTH),  # masses ~2x Borsato (OFF)
                 0.0, 0.0, 0.0, 0.0,                    # e = 0 (OFF)
                 Pb_d, Pc_d, t0b, t0c])
lo = np.array([np.log(1e-6), np.log(1e-6), -.3,-.3,-.3,-.3, Pb_d*0.97, Pc_d*0.97, t0b-0.6, t0c-0.6])
hi = np.array([np.log(6e-4), np.log(6e-4),  .3, .3, .3, .3, Pb_d*1.03, Pc_d*1.03, t0b+0.6, t0c+0.6])

JAC_ARG = {"analytic": res_jac, "hybrid": res_jac_hybrid,
           "bsjac": res_jac_bsjac}.get(JAC, "2-point")
print(f"# {len(t_obs)} transits (b={len(tb)},c={len(tc)}); JAC={JAC}; seed chi2={chi2_fn(seed):.1f}", flush=True)
t0 = time.time()
sol = least_squares(res_fn, seed, jac=JAC_ARG, bounds=(lo, hi), method="trf",
                    x_scale="jac", xtol=1e-10, ftol=1e-10,
                    max_nfev=int(os.environ.get("NFEV", "300")), verbose=2)
dt = time.time() - t0
p = sol.x; chi2 = chi2_fn(p)
tm = t_model(p)
def rms(mask): d = (tm[mask]-t_obs[mask])*1440; return np.sqrt(np.mean(d**2))
mb, mc = np.exp(p[0]), np.exp(p[1])
ebf, ecf = np.hypot(p[2], p[3]), np.hypot(p[4], p[5])
wbf = np.degrees(np.arctan2(p[3], p[2])) % 360; wcf = np.degrees(np.arctan2(p[5], p[4])) % 360
print(f"\n=== rebound-pymc gradient N-body fit to REAL Kepler-9 (Holczer 2016) ===", flush=True)
print(f"  wall={dt:.1f}s  nfev={nfev[0]}  chi2={chi2:.1f}  chi2/dof={chi2/(len(t_obs)-10):.2f}", flush=True)
print(f"  m_b={mb/MEARTH:.1f} Me ({mb/MJUP:.4f} MJ)  e_b={ebf:.3f} w_b={wbf:.0f}   (Borsato 43.4, 0.061, 357)", flush=True)
print(f"  m_c={mc/MEARTH:.1f} Me ({mc/MJUP:.4f} MJ)  e_c={ecf:.3f} w_c={wcf:.0f}   (Borsato 29.9, 0.067, 167)", flush=True)
print(f"  period b={p[6]:.5f} c={p[7]:.5f}", flush=True)
print(f"  O-C rms: b={rms(plan==0):.2f} min  c={rms(plan==1):.2f} min", flush=True)
np.savez(f"{HERE}/kepler9_rebound_pymc_fit.npz", theta=p, tmodel=tm, t_obs=t_obs, plan=plan, sig=sig)

# ---- plot ----------------------------------------------------------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 1, figsize=(9, 6))
for pl, N, a, nm, col in [(0, Nb, ax[0], "Kepler-9 b", "#1f77b4"),
                          (1, Nc, ax[1], "Kepler-9 c", "#d62728")]:
    mask = plan == pl; to = t_obs[mask]
    A = np.vstack([np.ones_like(N), N]).T; cf = np.linalg.lstsq(A, to, rcond=None)[0]
    a.errorbar(N, (to - A@cf)*1440, yerr=sig[mask]*1440, fmt="o", color=col, ms=3, label="Holczer 2016")
    a.plot(N, (tm[mask] - A@cf)*1440, "-k", lw=1.4, label="rebound-pymc fit")
    a.set_ylabel("O-C (min)"); a.set_title(nm); a.legend(fontsize=8); a.grid(alpha=.2)
ax[1].set_xlabel("transit number")
fig.suptitle(f"Kepler-9 real TTV — rebound-pymc gradient N-body fit "
             f"(b rms {rms(plan==0):.1f}, c rms {rms(plan==1):.1f} min)")
fig.tight_layout(); fig.savefig(f"{HERE}/kepler9_rebound_pymc_real_fit.png", dpi=150)
print("wrote kepler9_rebound_pymc_real_fit.png", flush=True)
