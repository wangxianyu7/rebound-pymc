"""Transit-timing speedups for the REBOUND TTV example.

Two orthogonal accelerations over the classic
https://rebound.hanno-rein.de/ipython_examples/TransitTimingVariations/ recipe
(IAS15 + bisection):

  1. replace the ~22-step bisection with a 2-3 step Newton refinement
     (root of relative-y; derivative = relative-vy, which is free from the state);
  2. swap the integrator (WHFast fixed-step, or BS adaptive-high-accuracy).

Both keep the transit times; BS+Newton matches IAS15 to ~1e-8, WHFast+Newton to
~1e-5 (period ~6.28), at 3-6x the speed. Run: python ttv_transit_finding_benchmark.py
"""
import time
import numpy as np
import rebound

N = 174  # number of inner-planet transits to collect


def make():
    sim = rebound.Simulation()
    sim.add(m=1.0)
    sim.add(m=1e-5, a=1.0, e=0.1)
    sim.add(m=1e-5, a=1.757)
    sim.move_to_com()
    return sim


def run(integrator=None, dt=None, refine="bisect", bs_eps=None):
    sim = make()
    if integrator:
        sim.integrator = integrator
        if dt:
            sim.dt = dt
        if integrator == "bs" and bs_eps:
            sim.ri_bs.eps_rel = bs_eps
            sim.ri_bs.eps_abs = bs_eps
    p = sim.particles
    tt = np.zeros(N)
    i = 0
    while i < N:
        y_old = p[1].y - p[0].y
        t_old = sim.t
        sim.integrate(sim.t + 0.5)
        t_new = sim.t
        if y_old * (p[1].y - p[0].y) < 0.0 and p[1].x - p[0].x > 0.0:
            if refine == "bisect":
                while t_new - t_old > 1e-7:
                    if y_old * (p[1].y - p[0].y) < 0.0:
                        t_new = sim.t
                    else:
                        t_old = sim.t
                    sim.integrate((t_new + t_old) / 2.0)
            else:  # Newton on relative-y; derivative = relative-vy
                for _ in range(6):
                    yr = p[1].y - p[0].y
                    if abs(yr) < 1e-10:
                        break
                    sim.integrate(sim.t - yr / (p[1].vy - p[0].vy))
            tt[i] = sim.t
            i += 1
            sim.integrate(sim.t + 0.05)  # advance past the transit
    return tt


if __name__ == "__main__":
    import os
    variants = [
        ("IAS15 + bisection\n(original)", dict(refine="bisect")),
        ("IAS15 + Newton", dict(refine="newton")),
        ("WHFast + Newton", dict(integrator="whfast", dt=2 * np.pi / 50, refine="newton")),
        ("BS(1e-9) + Newton", dict(integrator="bs", bs_eps=1e-9, refine="newton")),
        ("BS(1e-11) + Newton", dict(integrator="bs", bs_eps=1e-11, refine="newton")),
    ]
    ref, rows = None, []
    for name, kw in variants:
        t0 = time.time()
        tt = run(**kw)
        ms = (time.time() - t0) * 1000
        if ref is None:
            ref = tt
        acc = np.max(np.abs(tt - ref))
        rows.append((name, ms, acc))
        print(f"{name.replace(chr(10),' '):32s}: {ms:7.1f} ms   maxDelta vs original={acc:.2e}")

    # figure: wall time (bars) annotated with accuracy + speedup
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r[0] for r in rows]
    times = np.array([r[1] for r in rows])
    accs = [r[2] for r in rows]
    base = times[0]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["0.6", "#7fb3d5", "#e07b39", "#2e8b57", "#1f6f43"]
    y = np.arange(len(names))[::-1]
    ax.barh(y, times, color=colors)
    for yi, t, a in zip(y, times, accs):
        lab = "reference" if a == 0 else f"maxD={a:.0e},  {base/t:.1f}x"
        ax.text(t + base * 0.01, yi, lab, va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("wall time for 174 transits (ms)")
    ax.set_title("REBOUND TTV transit-finding: Newton + integrator speedups")
    ax.set_xlim(0, base * 1.35)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ttv_benchmark.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)
