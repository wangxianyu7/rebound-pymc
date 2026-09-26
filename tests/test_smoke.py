"""Smoke test: does the ported rebound-pymc (pytensor + rebound 4.x) compile & run,
and does the C IntegrateOp match a direct rebound integration?  env: normal."""
import sys, os, numpy as np
import pytensor, pytensor.tensor as pt
import rebound

import rebound_pymc
print("rebound_pymc", rebound_pymc.__version__, "| pytensor", pytensor.__version__, "| rebound", rebound.__version__)
from rebound_pymc.integrate import IntegrateOp
from rebound_pymc.python_impl import ReboundOp

# --- a simple Sun + Jupiter-ish 2-body system in AU, Msun, yr/2pi units ---
masses = np.array([1.0, 1e-3])
# star at origin at rest; planet at a=1, circular, v=sqrt(G*Mtot/a), G=1 in these units
a = 1.0; vcirc = np.sqrt((masses.sum()) / a)
coords0 = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [a,   0.0, 0.0, 0.0, vcirc, 0.0],
], dtype=float)
times = np.linspace(0.0, 2*np.pi, 5)   # ~1 orbit

# --- C IntegrateOp via a compiled pytensor function ---
mvar = pt.dvector("m"); cvar = pt.dmatrix("c"); tvar = pt.dvector("t")
opC = IntegrateOp(t=0.0, dt=0.01, integrator="ias15")
outC, jacC = opC(mvar, cvar, tvar)
fC = pytensor.function([mvar, cvar, tvar], [outC, jacC])
cC, jC = fC(masses, coords0, times)
print("C op output shapes:", cC.shape, jC.shape)   # (ntime, nbody, 6), (ntime,nbody,7,nbody,6)

# --- pure-Python ReboundOp for cross-check ---
opP = ReboundOp()
op_c, op_j = opP(mvar, cvar, tvar)
fP = pytensor.function([mvar, cvar, tvar], [op_c, op_j])
cP, jP = fP(masses, coords0, times)

# --- direct rebound reference ---
sim = rebound.Simulation()
for i in range(2):
    sim.add(m=masses[i], x=coords0[i,0], y=coords0[i,1], z=coords0[i,2],
            vx=coords0[i,3], vy=coords0[i,4], vz=coords0[i,5])
ref = np.zeros((len(times), 2, 6))
for k,t in enumerate(times):
    sim.integrate(t)
    for i in range(2):
        p=sim.particles[i]; ref[k,i]=[p.x,p.y,p.z,p.vx,p.vy,p.vz]

dC = np.max(np.abs(cC - ref))
dP = np.max(np.abs(cP - ref))
print(f"max|C op   - rebound| = {dC:.3e}")
print(f"max|Py op  - rebound| = {dP:.3e}")
print(f"max|C - Py| coords      = {np.max(np.abs(cC-cP)):.3e}")

# --- gradient smoke test (the whole point: differentiable N-body) ---
loss = pt.sum(outC**2)
g = pytensor.grad(loss, [mvar, cvar])
fg = pytensor.function([mvar, cvar, tvar], g)
gm, gc = fg(masses, coords0, times)
print("grad shapes:", np.asarray(gm).shape, np.asarray(gc).shape, "| finite:", np.all(np.isfinite(gm)) and np.all(np.isfinite(gc)))
print("PASS" if (dC < 1e-6 and dP < 1e-6) else "MISMATCH")
