"""Kepler-9 TTV posterior with PyMC NUTS, powered by rebound-pymc gradients.

Same differentiable timing residual (x_sky/vx_sky at observed transits) as the
L-BFGS example, but here it drives Hamiltonian Monte Carlo (NUTS) to sample the
posterior of the planet masses (m_b, m_c). NUTS only works because rebound-pymc
supplies analytic gradients of the transit model through the N-body integration.
env: normal   ->   python kepler9_nuts.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytensor
import pytensor.tensor as pt
import pymc as pm
import arviz as az
from rebound_pymc.orbit import ReboundOrbit

HERE = os.path.dirname(os.path.abspath(__file__))
MJ = 9.543e-4
TRUTH = np.array([0.137, 0.094])                                  # m_b, m_c in M_Jup
FIXED = dict(period=np.array([19.24, 38.99]), t0=np.array([5.0, 12.0]),
             ecc=np.array([0.06, 0.07]), omega=np.array([0.5, 1.2]),
             incl=np.array([np.pi / 2, np.pi / 2]), Omega=np.array([0.0, 0.0]))
TSPAN, SIG = 600.0, 1.0 / 1440.0                                  # 1-min timing error (days)


def orbit(mvec_msun):
    return ReboundOrbit(m_planet=mvec_msun, m_star=1.0, r_star=1.0, **FIXED)


# ---- data: observed transit times at truth + 1-min noise (same as L-BFGS example) ----
_tg = np.linspace(0, TSPAN, 6000)
_m = pt.dvector("m")
_gridf = pytensor.function([_m], orbit(_m).get_relative_position(pt.as_tensor_variable(_tg)))


def transit_times(mjup):
    X, _, Z = [np.asarray(a) for a in _gridf(np.asarray(mjup) * MJ)]
    out = []
    for pi in range(2):
        x, z = X[:, pi], Z[:, pi]
        s = np.sign(x)
        idx = np.where((s[:-1] * s[1:] < 0) & (z[:-1] > 0))[0]
        out.append(np.array([_tg[j] - x[j] * (_tg[j + 1] - _tg[j]) / (x[j + 1] - x[j]) for j in idx]))
    return out


tb, tc = transit_times(TRUTH)
rng = np.random.default_rng(0)
tb = tb + rng.normal(0, SIG, len(tb))
tc = tc + rng.normal(0, SIG, len(tc))
all_t = np.concatenate([tb, tc])
plan = np.array([0] * len(tb) + [1] * len(tc))
print(f"# observed transits: b={len(tb)} c={len(tc)}")

# ---- PyMC model: masses -> differentiable timing residual -> Gaussian likelihood ----
with pm.Model() as model:
    mb = pm.Uniform("m_b", 0.05, 0.30)                           # M_Jup
    mc = pm.Uniform("m_c", 0.03, 0.20)
    ob = orbit(pt.stack([mb * MJ, mc * MJ]))
    tt = pt.as_tensor_variable(all_t)
    Xo, _, _ = ob.get_relative_position(tt)
    Vx, _, _ = ob.get_relative_velocity(tt)
    row = pt.arange(all_t.shape[0])
    resid = Xo[row, plan] / Vx[row, plan]                        # linearised timing error (days)
    pm.Normal("ttv", mu=resid, sigma=SIG, observed=np.zeros(all_t.shape[0]))

    idata = pm.sample(draws=500, tune=500, chains=2, cores=1,
                      target_accept=0.9, random_seed=0, progressbar=True)

print(az.summary(idata, var_names=["m_b", "m_c"], round_to=5).to_string())

# ---- corner / pair plot with truth markers ----
ax = az.plot_pair(idata, var_names=["m_b", "m_c"], kind="kde",
                  marginals=True, figsize=(6, 6),
                  reference_values={"m_b": TRUTH[0], "m_c": TRUTH[1]},
                  reference_values_kwargs=dict(color="red", markersize=10, marker="*"))
plt.suptitle("Kepler-9 mass posterior (NUTS via rebound-pymc gradients)", y=1.02)
out = os.path.join(HERE, "kepler9_nuts_posterior.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
