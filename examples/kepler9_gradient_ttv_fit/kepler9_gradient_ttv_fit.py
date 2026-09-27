"""Kepler-9 TTV fit with rebound-pymc gradients.

Differentiable transit-timing residual = x_sky / vx_sky evaluated at each observed
transit (a one-Newton-step linearised timing error), made differentiable in the
planet masses by rebound-pymc's variational-particle Jacobian. Fits (m_b, m_c) with
gradient L-BFGS vs gradient-free Nelder-Mead, compares forward-integration counts,
and writes an O-C (TTV) fit figure.  env: normal   ->  python kepler9_gradient_ttv_fit.py
"""
import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytensor
import pytensor.tensor as pt
from rebound_pymc.orbit import ReboundOrbit
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
MJ = 9.543e-4
TRUTH = np.array([0.137 * MJ, 0.094 * MJ])                       # m_b, m_c (Msun)
FIXED = dict(period=np.array([19.24, 38.99]), t0=np.array([5.0, 12.0]),
             ecc=np.array([0.06, 0.07]), omega=np.array([0.5, 1.2]),
             incl=np.array([np.pi / 2, np.pi / 2]), Omega=np.array([0.0, 0.0]))
TSPAN, SIG = 600.0, 1.0 / 1440.0                                 # 1-min timing error (days)


def orbit(mvec):
    return ReboundOrbit(m_planet=mvec, m_star=1.0, r_star=1.0, **FIXED)


# reusable: sky positions on a grid for given masses
_tg = np.linspace(0, TSPAN, 6000)
_m = pt.dvector("m")
_gridf = pytensor.function([_m], orbit(_m).get_relative_position(pt.as_tensor_variable(_tg)))


def transit_times(masses):
    """transit times per planet via sky-x zero-crossings (z>0), linear-interp."""
    X, _, Z = [np.asarray(a) for a in _gridf(masses)]
    out = []
    for pi in range(2):
        x, z = X[:, pi], Z[:, pi]
        s = np.sign(x)
        idx = np.where((s[:-1] * s[1:] < 0) & (z[:-1] > 0))[0]
        out.append(np.array([_tg[j] - x[j] * (_tg[j + 1] - _tg[j]) / (x[j + 1] - x[j]) for j in idx]))
    return out


# ---- 1. synthetic "observed" transit times at truth + 1-min noise ----
tb_true, tc_true = transit_times(TRUTH)
rng = np.random.default_rng(0)
tb = tb_true + rng.normal(0, SIG, len(tb_true))
tc = tc_true + rng.normal(0, SIG, len(tc_true))
all_t = np.concatenate([tb, tc])
plan = np.array([0] * len(tb) + [1] * len(tc))
print(f"# observed transits: b={len(tb)} c={len(tc)}")

# ---- 2. differentiable loss(m_b, m_c) ----
mb, mc = pt.dscalar("mb"), pt.dscalar("mc")
ob = orbit(pt.stack([mb, mc]))
tt = pt.as_tensor_variable(all_t)
Xo, _, _ = ob.get_relative_position(tt)
Vx, _, _ = ob.get_relative_velocity(tt)
row = pt.arange(all_t.shape[0])
resid = Xo[row, plan] / Vx[row, plan]                            # linearised timing error (days)
loss = pt.sum((resid / SIG) ** 2)
loss_fn = pytensor.function([mb, mc], loss)
grad_fn = pytensor.function([mb, mc], pt.grad(loss, [mb, mc]))

# ---- 3. fit from a perturbed start; count forward integrations ----
start = TRUTH * 1.4


class Counter:
    def __init__(self, f):
        self.f, self.n = f, 0

    def __call__(self, *a):
        self.n += 1
        return self.f(*a)


print(f"# start = 1.4x truth = {start / MJ} MJ ; truth = {TRUTH / MJ} MJ\n")
Lnm = Counter(lambda p: float(loss_fn(p[0], p[1])))
t0 = time.time()
rnm = minimize(Lnm, start, method="Nelder-Mead",
               options=dict(xatol=1e-9, fatol=1e-3, maxiter=2000))
tnm = time.time() - t0
Lg = Counter(lambda p: float(loss_fn(p[0], p[1])))
Gg = Counter(lambda p: np.array(grad_fn(p[0], p[1]), float))
t0 = time.time()
rg = minimize(Lg, start, jac=Gg, method="L-BFGS-B")
tg2 = time.time() - t0


def report(name, res, nfwd, wall):
    m = res.x
    e = np.abs(m - TRUTH) / TRUTH * 100
    print(f"{name:14s}: fwd-integr={nfwd:4d}  wall={wall:5.2f}s  loss={res.fun:9.3f}  "
          f"mb={m[0]/MJ:.4f} mc={m[1]/MJ:.4f} MJ  err=({e[0]:.2f}%,{e[1]:.2f}%)")


report("Nelder-Mead", rnm, Lnm.n, tnm)
report("L-BFGS+grad", rg, Lg.n + Gg.n, tg2)
print(f"\nforward-integration speedup: {Lnm.n/(Lg.n+Gg.n):.1f}x")

# ---- 4. O-C (TTV) fit figure: observed points vs best-fit model ----
best = rg.x
tb_fit, tc_fit = transit_times(best)
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for ax, name, t_obs, t_mod, col in [(axes[0], "Kepler-9 b", tb, tb_fit, "#1f77b4"),
                                    (axes[1], "Kepler-9 c", tc, tc_fit, "#d62728")]:
    n = np.arange(len(t_obs))
    A = np.vstack([np.ones_like(n), n]).T
    coef = np.linalg.lstsq(A, t_obs, rcond=None)[0]           # ephemeris from OBS
    oc_obs = (t_obs - A @ coef) * 24 * 60
    nmod = np.round((t_mod - coef[0]) / coef[1]).astype(int)
    oc_mod = (t_mod - (coef[0] + coef[1] * nmod)) * 24 * 60
    ax.errorbar(n, oc_obs, yerr=SIG * 24 * 60, fmt="o", color=col, ms=4, label="observed")
    ax.plot(nmod, oc_mod, "-", color="k", lw=1.5, label="best-fit model")
    ax.set_ylabel("O-C (min)")
    ax.set_title(f"{name}   (P={coef[1]:.4f} d)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
axes[1].set_xlabel("transit number")
fig.suptitle("Kepler-9 TTV — rebound-pymc gradient fit of (m$_b$, m$_c$)")
fig.tight_layout()
out = os.path.join(HERE, "kepler9_ttv_fit.png")
fig.savefig(out, dpi=150)
print("wrote", out)
