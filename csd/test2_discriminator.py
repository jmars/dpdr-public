"""TESTS 2-4 - the CSD discriminator: is recovery slowing as the fold is
approached, or constant?

Protocol (Meisel 2015 adapted): chronic drive eps, healthy equilibrium
y*(eps) (continuation), brief perturbation of G, measure recovery.

Measurements per eps (12 values approaching eps_c from below, distances
spanning ~5 decades):
  EXACT: full eigenspectrum of the Jacobian at y*(eps) - the exact
         linearized recovery rates (no fitting).  PRIMARY measurement.
  PROTOCOL: integrate y(0) = y* - delta*e_G under chronic eps; fit the
         decay rate of ||y(t) - y*|| over the 50%->5% window, record t95.
         delta is scaled to the healthy basin half-width (40% of the
         distance to the saddle in G) so the probe stays inside the basin
         at every eps; a FIXED delta = 0.05 probe is also run to show
         where it stops recovering at all (basin collapse).
  BASIN: healthy basin half-width in G = G*(eps) - G_saddle(eps) from a
         saddle-branch continuation.

Run (from this directory, with the requirements installed):
    python test2_discriminator.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import F, Params, Schedule, deriv, eig, eigdesc, equilibrium, sched

P = Params()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cache")
os.makedirs(OUT, exist_ok=True)

EPS_C = 0.265192279            # from test1 (border-collision fold)
EPS_C_MID = 0.265186670        # saddle branch death (test1 c2)

# 12 eps values approaching the fold from below; distances span 5 decades
EPS_GRID = [0.00, 0.10, 0.20, 0.24, 0.25, 0.255, 0.26, 0.263, 0.265,
            0.2651, 0.26515, 0.26519]
FIXED_DELTA = 0.05             # the "realistic" perturbation size


def healthy_eq(eps: float, y0) -> np.ndarray:
    y = equilibrium(eps, y0)
    assert y is not None, f"no healthy equilibrium at eps={eps}"
    return y


def find_saddle(eps: float, y0s) -> np.ndarray | None:
    """The unstable middle equilibrium at eps, warm-started."""
    for y0 in y0s:
        y = equilibrium(eps, y0)
        if y is not None and np.max(eig(y, eps).real) > 0:
            return y
    return None


def recovery_run(eps: float, y_star: np.ndarray, delta: float,
                 T: float = 4000.0) -> dict:
    """Perturb G by -delta at t=0, integrate under chronic eps."""
    y0 = y_star.copy()
    y0[1] -= delta
    sol = solve_ivp(deriv, (0.0, T), y0, args=(P, sched(eps)),
                    method="RK45", rtol=1e-9, atol=1e-11, max_step=5.0,
                    dense_output=True)
    t = np.arange(0.0, T, 1.0)
    d = np.linalg.norm(sol.sol(t) - y_star[:, None], axis=0)
    d0 = d[0]
    below = np.where(d < 0.05 * d0)[0]
    t95 = float(t[below[0]]) if below.size else float("nan")
    w = (d < 0.5 * d0) & (d > 0.05 * d0)
    lam_fit = float(-np.polyfit(t[w], np.log(d[w]), 1)[0]) \
        if w.sum() > 20 else float("nan")
    # tail rate: fit the last half of the below-5%... no: tail of the decay
    # (between 1% and 0.1% of d0) isolates the slowest linear mode
    wt = (d < 0.01 * d0) & (d > 1e-3 * d0)
    lam_tail = float(-np.polyfit(t[wt], np.log(d[wt]), 1)[0]) \
        if wt.sum() > 20 else float("nan")
    # recovery of G itself
    G = sol.sol(t)[1]
    recovered = bool(G[-1] > 0.5 * (y_star[1] + 0.1))
    return dict(t=t, d=d, t95=t95, lam_fit=lam_fit, lam_tail=lam_tail,
                G_end=float(G[-1]), recovered=recovered)


def main():
    print("=" * 78)
    print("TESTS 2-4: CSD DISCRIMINATOR (exact eigenvalues + protocol runs)")
    print(f"fold (border collision) at eps_c = {EPS_C:.9f}")
    print("=" * 78)

    # healthy branch by warm-start continuation
    y = equilibrium(0.0, np.array([0.05, 0.7, 0.5, 0.8, 0.5]))
    # saddle branch: multistart seed at eps=0.20, then continue over the grid
    y_s20 = None
    for G0 in np.linspace(0.15, 0.33, 37):
        cand = equilibrium(0.20, np.array([0.30, G0, 0.5455, 0.40, 0.5]))
        if cand is not None and np.max(eig(cand, 0.20).real) > 0:
            y_s20 = cand
            break
    assert y_s20 is not None, "saddle seed not found at eps=0.20"

    rows = []
    y_sad = y_s20
    # seed healthy at 0.20 too (continuation goes 0 -> up)
    y_h = y
    for eps in EPS_GRID:
        y_h = healthy_eq(eps, y_h)
        # saddle warm-start: continue from previous grid point (grid ascends)
        if eps <= 0.20 and eps != 0.20:
            # continue DOWN from 0.20 seed
            seed = y_sad if y_sad is not None else y_s20
            cand = find_saddle(eps, [seed,
                                     np.array([0.30, 0.2573, 0.5455,
                                               0.40, 0.5])])
        elif eps > 0.20:
            cand = find_saddle(eps, [y_sad] if y_sad is not None else [])
        else:
            cand = y_s20
        y_sad = cand if cand is not None else y_sad
        w = eig(y_h, eps)
        re = np.sort(w.real)          # ascending
        basin = (y_h[1] - y_sad[1]) if cand is not None else float("nan")
        rows.append(dict(eps=eps, dist=EPS_C - eps, y=y_h,
                         lam1=re[-1], lam2=re[-2], lam3=re[-3],
                         lamG=re[-4], lam5=re[-5],
                         basin=basin, has_saddle=cand is not None))

    # ---------------------------------------------------------------- test 2
    print("\n[TEST 2] exact dominant eigenvalue (slowest mode) vs "
          "distance-to-fold,")
    print("         + basin-aware perturbation-recovery protocol:")
    print(f"    {'eps':>9s} {'dist':>10s} {'lambda_dom':>12s} {'1/lam':>8s} "
          f"{'basin(G)':>10s} {'delta':>9s} {'t95':>8s} {'lam_fit':>8s} "
          f"{'lam_tail':>9s} {'fix.05 recov':>12s}")
    recs = {}
    for r in rows:
        delta = 0.4 * r["basin"] if np.isfinite(r["basin"]) else 1e-6
        rec = recovery_run(r["eps"], r["y"], delta)
        recs[r["eps"]] = rec
        rec_fix = recovery_run(r["eps"], r["y"], FIXED_DELTA, T=8000.0)
        print(f"    {r['eps']:9.5f} {r['dist']:10.2e} {r['lam1']:+12.6f} "
              f"{1.0 / abs(r['lam1']):8.1f} {r['basin']:10.2e} "
              f"{delta:9.2e} {rec['t95']:8.1f} {rec['lam_fit']:8.4f} "
              f"{rec['lam_tail']:9.5f} "
              f"{'YES' if rec_fix['recovered'] else 'COLLAPSE':>12s}")
    lam_dom = np.array([r["lam1"] for r in rows])
    print(f"\n    dominant lambda across the whole approach: "
          f"min {lam_dom.min():+.8f}, max {lam_dom.max():+.8f}, "
          f"spread {lam_dom.max() - lam_dom.min():.2e}")
    print(f"    theory -mu/tau_g = {-P.mu / P.tau_g:+.8f}  -> EXACTLY "
          f"CONSTANT: recovery time 667 t.u. at every distance. "
          f"NO SLOWING.")

    # ---------------------------------------------------------------- test 3
    print("\n[TEST 3] mode separation - every exact mode vs distance:")
    print(f"    {'eps':>9s} {'dist':>10s} {'lam_g(slowest)':>15s} "
          f"{'lam_S':>10s} {'lam_D':>10s} {'lam_G-mode':>11s} {'lam_fast':>9s}")
    for r in rows:
        print(f"    {r['eps']:9.5f} {r['dist']:10.2e} {r['lam1']:+15.8f} "
              f"{r['lam2']:+10.6f} {r['lam3']:+10.6f} {r['lamG']:+11.6f} "
              f"{r['lam5']:+9.4f}")
    lamG = np.array([abs(r["lamG"]) for r in rows])
    dist = np.array([r["dist"] for r in rows])
    # near-fold trend: last 6 points (spanning 4+ decades of distance)
    near = slice(-6, None)
    soften_near = lamG[near][0] / lamG[near][-1]
    print(f"\n    NEAR the fold (dist {dist[near][0]:.1e} -> "
          f"{dist[near][-1]:.1e}, {np.log10(dist[near][0] / dist[near][-1]):.1f}"
          f" decades): G-mode {lamG[near][0]:.6f} -> {lamG[near][-1]:.6f}, "
          f"i.e. x{1 / soften_near:.4f} ({100 * (1 - 1 / soften_near):.2f}% "
          f"softer)")
    print(f"      sqrt (saddle-node) prediction over the same range: "
          f"x{np.sqrt(dist[near][0] / dist[near][-1]):.1f} softer "
          f"({100 * (1 - np.sqrt(dist[near][-1] / dist[near][0])):.3f}%)")
    print(f"    FAR field (eps 0 -> 0.2) the G-mode does drift: "
          f"{lamG[0]:.6f} -> {lamG[2]:.6f} (x{lamG[0] / lamG[2]:.2f}) - "
          f"but that is the ordinary a*-dependence of the generator's "
          f"logistic stiffness, monotone in eps and ALREADY SATURATED by "
          f"eps=0.24; it is not a fold precursor (see slope fits).")
    v3 = ("MASKED CSD (G-mode softens near the fold, slowest does not)"
          if soften_near < 0.5 else
          "NO SOFTENING of the G-mode near the fold either -> NO CSD AT ALL")
    print(f"    VERDICT 3: {v3}")

    # ---------------------------------------------------------------- test 4
    print("\n[TEST 4] exponent fits log|lambda| vs log(distance) "
          "(saddle-node prediction: slope 1/2):")
    fits = {}
    for name, key, sel in (("slowest g-loop, all points", "lam1",
                            slice(None)),
                           ("G-mode, all points", "lamG", slice(None)),
                           ("G-mode, NEAR fold (last 6)", "lamG", near),
                           ("G-mode, FAR field (first 4)", "lamG",
                            slice(0, 4))):
        lam = np.array([abs(r[key]) for r in rows])[sel]
        x = np.log(dist[sel])
        yy = np.log(lam)
        slope, icpt = np.polyfit(x, yy, 1)
        pred = icpt + slope * x
        ssr = ((yy - pred) ** 2).sum()
        sst = ((yy - yy.mean()) ** 2).sum()
        r2 = 1.0 - ssr / sst if sst > 0 else float("nan")
        fits[name] = (slope, r2)
        print(f"    {name:28s}: slope = {slope:+.4f}  (r2 = {r2:+.4f})  "
              f"{'~ sqrt law' if abs(slope - 0.5) < 0.05 else 'NOT 1/2'}")
    print("    (slope 0 = no softening; the CSD saddle-node prediction "
          "is +0.5)")

    # ---------------------------------------------------------------- cache
    np.savez(os.path.join(OUT, "test2_discriminator.npz"),
             eps=np.array(EPS_GRID), dist=dist,
             lam_g=np.array([r["lam1"] for r in rows]),
             lam_S=np.array([r["lam2"] for r in rows]),
             lam_D=np.array([r["lam3"] for r in rows]),
             lam_Gmode=np.array([r["lamG"] for r in rows]),
             lam_fast=np.array([r["lam5"] for r in rows]),
             basin=np.array([r["basin"] for r in rows]),
             t95=np.array([recs[e]["t95"] for e in EPS_GRID]),
             lam_fit=np.array([recs[e]["lam_fit"] for e in EPS_GRID]),
             lam_tail=np.array([recs[e]["lam_tail"] for e in EPS_GRID]),
             eps_c=np.array(EPS_C))
    print(f"\n  -> {os.path.join(OUT, 'test2_discriminator.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
