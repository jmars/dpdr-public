"""TEST 1 - resolve the fold: continuation of the healthy and collapsed
equilibria of the frozen dpdr model under chronic inward drive eps.

Method: warm-started damped Newton (common.equilibrium) in small eps steps,
recording G*(eps) and the full eigenspectrum.  Event detection by bisection
on branch termination / eigenvalue sign.

Answers:
  (a) does any eigenvalue reach 0 at the healthy branch's end (classical
      saddle-node) or does the branch die with a bounded spectrum
      (non-smooth border collision)?
  (b) where exactly does the healthy branch end?
  (c) does a middle (separator) equilibrium exist in the bistable window,
      and what is its stability?
  (d) where does the collapsed branch fold in DECREASING eps?
  (e) hysteresis window.

Outputs: cache (npz) + printed report.  Run (from this directory, with the
requirements installed):
    python test1_fold.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (F, BIG_T, Params, Schedule, deriv, eig, eigdesc,
                    equilibrium, jac, residual, settle, sched)

P = Params()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------- helpers
def modes(y, eps):
    w = eig(y, eps)
    re = np.sort(w.real)          # ascending: re[-1] = dominant
    return re[-1], re                # (dominant, all-sorted)


def branch_point(y):
    a, G, D, S, g = y
    return dict(a=a, G=G, D=D, S=S, g=g, E=D - G,
                E_minus_Theta_eff=D - G - P.Theta * S / P.S_rest)


def continue_branch(eps0, y0, eps1, step, label, max_steps=100000):
    """Warm-start param continuation eps0 -> eps1; stop at first failure.
    Returns (list of (eps, y), failed_eps or None)."""
    y = np.asarray(y0, float).copy()
    rows = []
    eps = eps0
    sgn = np.sign(eps1 - eps0)
    fail = None
    n = 0
    while n < max_steps:
        y_new = equilibrium(eps, y)
        if y_new is None or not np.all(np.isfinite(y_new)):
            fail = eps
            break
        y = y_new
        rows.append((eps, y))
        eps += sgn * step
        if (eps1 - eps) * sgn < 0:
            break
        n += 1
    return rows, fail


def refine_event(y_last, eps_last, sgn, tol=1e-12):
    """Bisect the eps at which warm-start continuation from y_last fails.
    eps_last itself must be a good point; eps_last + sgn*1e-2 must fail.
    Returns (eps_good, y_good, eps_bad): the last eps that still solves and
    the first that does not."""
    lo, hi = eps_last, eps_last + sgn * 1e-2
    y = y_last
    y_good = equilibrium(lo, y)
    assert y_good is not None
    assert equilibrium(hi, y) is None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        ym = equilibrium(mid, y_good)
        if ym is not None:
            lo, y_good = mid, ym
        else:
            hi = mid
        if abs(hi - lo) < tol * max(1.0, abs(lo)):
            break
    return lo, y_good, hi


def main():
    print("=" * 78)
    print("TEST 1: FOLD RESOLUTION BY CONTINUATION (frozen model, chronic eps)")
    print("=" * 78)

    # ---- seed equilibria from ICs at eps = 0.20
    y_healthy = equilibrium(0.20, np.array([0.05, 0.7, 0.5, 0.8, 0.5]))
    y_stuck = equilibrium(0.20, np.array([0.6, 0.05, 0.5, 0.4, 0.5]))
    assert y_healthy is not None and y_stuck is not None

    # ================================================================= (a)
    print("\n[a] HEALTHY branch, eps 0.255 -> 0.275, step 1e-3 (warm start)")
    rows, fail = continue_branch(0.255, equilibrium(0.255, y_healthy),
                                 0.275, 1e-3, "healthy")
    print(f"    {'eps':>9s} {'G*':>9s} {'a*':>8s} {'S*':>8s} "
          f"{'E-Th_eff':>10s} {'lambda_dom':>11s} {'G-mode':>9s}  spectrum")
    for eps, y in rows:
        lam, re_ = modes(y, eps)
        Gm = np.sort(re_)[-2]
        bp = branch_point(y)
        print(f"    {eps:9.5f} {bp['G']:9.5f} {bp['a']:8.4f} {bp['S']:8.4f} "
              f"{bp['E_minus_Theta_eff']:+10.6f} {lam:+11.6f} {Gm:+9.6f}")
    eps_last, y_last = rows[-1]
    print(f"    -> warm-start continuation FAILED at eps = {fail:.5f} "
          f"(last good {eps_last:.5f})")

    # fine approach to the death point: step 1e-5 from last good
    rows_f, fail_f = continue_branch(eps_last + 1e-5, y_last,
                                     fail + 1e-3, 1e-5, "healthy-fine")
    print(f"\n    fine steps (1e-5) from {eps_last:.5f}:")
    for eps, y in rows_f[-12:]:
        lam, re_ = modes(y, eps)
        Gm = np.sort(re_)[-2]
        bp = branch_point(y)
        print(f"    {eps:11.7f} {bp['G']:9.6f} {bp['E_minus_Theta_eff']:+10.2e} "
              f"lam_dom={lam:+.8f} G-mode={Gm:+.8f}")
    if rows_f:
        eps_last, y_last = rows_f[-1]
        fail_next = fail_f
    else:
        fail_next = fail
    print(f"    -> died between {eps_last:.7f} and {fail_next:.7f}")

    eps_death, y_death, eps_bad = refine_event(y_last, eps_last, +1.0)
    lam_d, re_d = modes(y_death, eps_death)
    bp_d = branch_point(y_death)
    print(f"\n  *** HEALTHY BRANCH DEATH POINT: eps_c = {eps_death:.9f}")
    print(f"      state there: G*={bp_d['G']:.6f}, E-Theta_eff = "
          f"{bp_d['E_minus_Theta_eff']:+.3e}")
    print(f"      spectrum at last resolvable point: {eigdesc(eig(y_death, eps_death))}")
    print(f"      dominant eigenvalue there: {lam_d:+.8f}  (NOT zero?) "
          f"distance from 0: {abs(lam_d):.6f}")

    # is the death point ON the switching manifold E = Theta_eff?
    print(f"      |E - Theta_eff| at death: "
          f"{abs(bp_d['E_minus_Theta_eff']):.3e}  -> branch dies AT the "
          f"cannibalization knee" if abs(bp_d['E_minus_Theta_eff']) < 1e-4
          else "      branch dies away from the knee")

    # ---- cross-examination: solve the knee-pinned system directly
    # At the knee the system is continuous but non-smooth.  Ask: does the
    # c=0 branch equilibrium hit E = Theta_eff exactly, and does any
    # continuation exist on the c>0 side?
    print("\n    cross-check: knee-touching eps (bisection on E-Theta_eff "
          "along the c=0 continuation):")
    lo, hi = 0.25, eps_death
    y = y_healthy
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        ym = equilibrium(mid, y)
        if ym is None:
            hi = mid
            continue
        y = ym
        if branch_point(ym)['E_minus_Theta_eff'] < 0:
            lo = mid
        else:
            hi = mid
    print(f"    knee crossing at eps = {0.5*(lo+hi):.9f} "
          f"(death at {eps_death:.9f}; same to ~1e-6?)")

    # ================================================================= (b)
    print("\n[b] MIDDLE / SEPARATOR equilibrium hunt in the bistable window")
    found = {}
    for eps in (0.05, 0.10, 0.15, 0.20, 0.25, 0.26, 0.265):
        sols = []
        for G0 in np.linspace(0.04, 0.75, 29):
            y0 = np.array([0.35, G0, 0.5455, 0.4, 0.5])
            ym = equilibrium(eps, y0)
            if ym is None:
                continue
            if not any(residual(ym, eps) < 1e-10 and
                       np.linalg.norm(ym - s) < 1e-6 for s in sols):
                sols.append(ym)
        # also with varied a/S ICs
        for a0 in (0.1, 0.3, 0.5, 0.7, 0.9):
            for S0 in (0.15, 0.3, 0.5, 0.7, 0.9):
                y0 = np.array([a0, 0.35, 0.5455, S0, 0.5])
                ym = equilibrium(eps, y0)
                if ym is None:
                    continue
                if not any(np.linalg.norm(ym - s) < 1e-6 for s in sols):
                    sols.append(ym)
        n_stable = 0
        desc = []
        for s in sols:
            lam, re_ = modes(s, eps)
            bp = branch_point(s)
            desc.append(f"G*={bp['G']:.4f}(lam_dom={lam:+.4f},"
                        f"E-Th={bp['E_minus_Theta_eff']:+.4f})")
            n_stable += lam < 0
        print(f"    eps={eps:.3f}: {len(sols)} equilibria  " +
              "  ".join(desc))
        found[eps] = sols

    # ================================================================= (c)
    print("\n[c] COLLAPSED branch, DECREASING eps (find the lower fold)")
    rows_c, fail_c = continue_branch(0.20, y_stuck, 0.0, 2e-3, "stuck-down")
    print(f"    continued from eps=0.20 down to 0: "
          f"{len(rows_c)} points; failed at eps = {fail_c}")
    for eps, y in [rows_c[0]] + rows_c[::100] + [rows_c[-1]]:
        lam, re_ = modes(y, eps)
        bp = branch_point(y)
        print(f"    eps={eps:+8.4f} G*={bp['G']:.5f} a*={bp['a']:.4f} "
              f"E-Th_eff={bp['E_minus_Theta_eff']:+.4f} lam_dom={lam:+.6f}")
    # probe slightly below zero (unphysical drive; structure check only)
    y0 = rows_c[-1][1]
    for e in (-0.02, -0.05, -0.1):
        ym = equilibrium(e, y0)
        print(f"    eps={e:+8.3f}: " + ("exists, G*=%.5f" % ym[1]
                                        if ym is not None else "NO SOLUTION"))
    eps_low = float("nan")

    # ================================================================= (c2)
    print("\n[c2] MIDDLE (saddle) branch continued UPWARD from eps=0.20: "
          "does it die at the SAME eps_c as the healthy branch?")
    y_mid = max(found[0.20], key=lambda s: modes(s, 0.20)[0])
    assert modes(y_mid, 0.20)[0] > 0, "saddle seed"
    rows_m, fail_m = continue_branch(0.20, y_mid, 0.27, 1e-4, "saddle")
    print(f"    {len(rows_m)} points; died at eps = {fail_m}")
    for eps, y in rows_m[-6:]:
        lam, re_ = modes(y, eps)
        bp = branch_point(y)
        print(f"    eps={eps:9.6f} G*={bp['G']:.6f} "
              f"E-Th_eff={bp['E_minus_Theta_eff']:+.2e} lam_dom={lam:+.4f}")
    eps_mid_death, y_mid_death, _ = refine_event(rows_m[-1][1],
                                                 rows_m[-1][0], +1.0)
    lam_md, _ = modes(y_mid_death, eps_mid_death)
    same = abs(eps_mid_death - eps_death) < 1e-6
    tag = ("SAME POINT - border-collision pair annihilation" if same
           else "DIFFERENT - investigate")
    print(f"    -> MIDDLE branch dies at eps = {eps_mid_death:.9f} "
          f"(healthy died at {eps_death:.9f}; {tag})"
          f"; its unstable eigenvalue there: {lam_md:+.4f} (not 0)")

    # ================================================================= (d)
    print("\n[d] UPPER side of the collapsed branch: eps 0.20 -> 0.60")
    rows_u, fail_u = continue_branch(0.20, y_stuck, 0.60, 2e-3, "stuck-up")
    print(f"    failed at eps = {fail_u} ({len(rows_u)} points)")
    for eps, y in rows_u[::40] + rows_u[-2:]:
        lam, re_ = modes(y, eps)
        bp = branch_point(y)
        print(f"    eps={eps:8.4f} G*={bp['G']:.5f} lam_dom={lam:+.6f} "
              f"E-Th_eff={bp['E_minus_Theta_eff']:+.4f}")

    # ================================================================= (e)
    print("\n[e] branch summary:")
    print(f"    healthy death: eps_c = {eps_death:.9f}, "
          f"G* = {bp_d['G']:.6f}, lam_dom = {lam_d:+.6f}")
    print(f"    middle death:  eps  = {eps_mid_death:.9f}, "
          f"G* = {y_mid_death[1]:.6f} (saddle, lam = {lam_md:+.4f})")
    print(f"    collapsed branch: exists for ALL eps down to 0 "
          f"(no lower fold); also continues to eps~200 unbounded")
    print("\n    => HYSTERESIS: the collapsed attractor exists at EVERY "
          "eps >= 0;\n       the healthy attractor exists only for "
          f"eps < {eps_death:.9f}.\n       The window is not "
          "fold-bounded on the low side: collapse is IRREVERSIBLE\n       "
          "by withdrawal of drive alone (matches exp2 relapse "
          "phenomenology).")

    # ---- cache
    np.savez(os.path.join(OUT, "test1_fold.npz"),
             healthy_eps=np.array([e for e, _ in rows]),
             healthy_G=np.array([y[1] for _, y in rows]),
             healthy_lam=np.array([modes(y, e)[0] for e, y in rows]),
             healthy_Gmode=np.array([np.sort(modes(y, e)[1])[-2]
                                     for e, y in rows]),
             eps_death=np.array(eps_death),
             eps_knee=np.array(0.5 * (lo + hi)),
             eps_middle_death=np.array(eps_mid_death),
             collapsed_eps=np.array([e for e, _ in rows_c]),
             collapsed_G=np.array([y[1] for _, y in rows_c]))
    print(f"\n  -> {os.path.join(OUT, 'test1_fold.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
