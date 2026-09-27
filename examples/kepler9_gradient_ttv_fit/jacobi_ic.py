"""jnkepler-style Jacobi initial conditions, ported to pytensor (differentiable).

This is the missing piece: rebound-pymc's ReboundOrbit builds ICs from exoplanet's
KeplerianOrbit (independent 2-body Kepler orbit per planet), which does NOT match the
Jacobi convention (outer planet's Kepler mass = star + all interior planets). For a
2:1 resonant system like Kepler-9 that IC error is fatal. Here we reproduce jnkepler's
utils.initialize_jacobi_xv -> conversion.jacobi_to_astrocentric -> astrocentric_to_cm
exactly, in pytensor, so it composes with the differentiable IntegrateOp.

Hardcoded for 2 planets (Kepler-9). Units: AU, Msun, day; G = jnkepler's G.
"""
import numpy as np
import pytensor.tensor as pt

G_JNK = 2.959122082855911e-4  # AU^3 / Msun / day^2  (jnkepler conversion.py:15)


def _kepler_solve(M, e, niter=15):
    """Solve Kepler's equation M = E - e sin E for E (unrolled Newton)."""
    E = M + e * pt.sin(M)
    for _ in range(niter):
        E = E - (E - e * pt.sin(E) - M) / (1.0 - e * pt.cos(E))
    return E


def _tic_to_M(tic, P, e, w, t_epoch):
    """time of inferior conjunction -> mean anomaly at t_epoch (conversion.py tic_to_m)."""
    tanw2 = pt.tan(0.5 * w)
    uic = 2.0 * pt.arctan(pt.sqrt((1.0 - e) / (1.0 + e)) * (1.0 - tanw2) / (1.0 + tanw2))
    return 2.0 * np.pi / P * (t_epoch - tic) + uic - e * pt.sin(uic)


def _elements_to_xv(P, e, inc, w, lnode, u, mass):
    """single-orbit Jacobi elements -> (x,y,z),(vx,vy,vz)  (conversion.py elements_to_xv)."""
    cosu, sinu = pt.cos(u), pt.sin(u)
    cosw, sinw = pt.cos(w), pt.sin(w)
    cosO, sinO = pt.cos(lnode), pt.sin(lnode)
    cosi, sini = pt.cos(inc), pt.sin(inc)
    n = 2.0 * np.pi / P
    na = (n * G_JNK * mass) ** (1.0 / 3.0)
    R = 1.0 - e * cosu
    P0 = cosw * cosO - sinw * sinO * cosi
    P1 = cosw * sinO + sinw * cosO * cosi
    P2 = sinw * sini
    Q0 = -sinw * cosO - cosw * sinO * cosi
    Q1 = -sinw * sinO + cosw * cosO * cosi
    Q2 = cosw * sini
    x, y = cosu - e, pt.sqrt(1.0 - e * e) * sinu
    vx, vy = -sinu, pt.sqrt(1.0 - e * e) * cosu
    xo = [(na / n) * (P0 * x + Q0 * y), (na / n) * (P1 * x + Q1 * y), (na / n) * (P2 * x + Q2 * y)]
    vo = [(na / R) * (P0 * vx + Q0 * vy), (na / R) * (P1 * vx + Q1 * vy), (na / R) * (P2 * vx + Q2 * vy)]
    return pt.stack(xo), pt.stack(vo)


def barycentric_coords0(period, ecosw, esinw, tic, pmass, smass, t_epoch,
                        cosi=(0.0, 0.0), lnode=(0.0, 0.0)):
    """params (pytensor scalars, 2 planets) -> (masses(3,), coords0(3,6)) in AU, AU/day.

    Row 0 = star, rows 1,2 = planets b,c.  Reproduces jnkepler exactly:
    Jacobi elements (enclosed mass) -> astrocentric -> center-of-mass.
    """
    m0, m1, m2 = smass, pmass[0], pmass[1]
    masses = pt.stack([m0, m1, m2])
    kmass = [m0 + m1, m0 + m1 + m2]  # Jacobi enclosed mass for planet 0, 1
    xj, vj = [], []
    for j in range(2):
        e = pt.sqrt(ecosw[j] ** 2 + esinw[j] ** 2)
        w = pt.arctan2(esinw[j], ecosw[j])
        inc = pt.arccos(cosi[j])
        M = _tic_to_M(tic[j], period[j], e, w, t_epoch)
        u = _kepler_solve(M, e)
        xjj, vjj = _elements_to_xv(period[j], e, inc, w, lnode[j], u, kmass[j])
        xj.append(xjj); vj.append(vjj)
    xj = pt.stack(xj)  # (2,3) Jacobi
    vj = pt.stack(vj)
    # Jacobi -> astrocentric: xast[0]=xj[0]; xast[1]=m1/(m0+m1)*xj[0]+xj[1]
    f = m1 / (m0 + m1)
    xast = pt.stack([xj[0], f * xj[0] + xj[1]])
    vast = pt.stack([vj[0], f * vj[0] + vj[1]])
    # astrocentric -> center of mass
    mtot = m0 + m1 + m2
    xcm_a = (m1 * xast[0] + m2 * xast[1]) / mtot
    vcm_a = (m1 * vast[0] + m2 * vast[1]) / mtot
    xstar = -xcm_a
    vstar = -vcm_a
    x = pt.stack([xstar, xast[0] - xcm_a, xast[1] - xcm_a])  # (3,3)
    v = pt.stack([vstar, vast[0] - vcm_a, vast[1] - vcm_a])
    coords0 = pt.concatenate([x, v], axis=1)  # (3,6) AU, AU/day
    return masses, coords0
