"""Sensitivity — plan §6b: one-at-a-time +/-20% and +/-30% on every model
parameter over the P1-P6 outcome measures.

Per perturbed Params, measured:
  P1  thr_dur   episode-duration failure threshold (log-bisection between
                dur=5 and dur=200; None = no collapse inside the bracket)
  P2  no_self   canonical failure run stays collapsed (G < 0.1 past
                episode end + 10*tau_G)
  P3  u_thr     min rescue strength u_ext achieving 'full' taxonomy
                (delay=200 past ep end, duration=60); None = unreachable
      peak_partial  max post-collapse G among the non-full scan runs
                (0.1 < value < 0.5 would be a partial lift / relapse band)
  P5  g_up      transient gain elevation on the canonical full-rescue run,
                measured at T=1800 and extrapolated to its exponential
                asymptote (None if the canonical rescue no longer yields
                'full'); the asymptote is g0 for every cell — see
                predictions.md P5 (FAIL)
  P6  lag_ratio t90(S)/t90(G) on the same run (None likewise)

P4 (relapse class presence) is absent from single-episode schedules (see
predictions.md P4): the scan measure peak_partial is its quantitative
stand-in.  Relapse DOES occur under recurring episodes (exp2 phase 3c).

Run: `.venv/bin/python -m experiments.sensitivity` (~15 min on 1 core; uses
a Pool).  Output: cache/sensitivity.npz + printed table.
"""
from __future__ import annotations

import os
from dataclasses import asdict, replace
from multiprocessing import Pool

import numpy as np

from dpdr.integrate import simulate
from dpdr.metrics import first_below
from dpdr.model import Params

from .common import episode_schedule, rescue_schedule, rescue_taxonomy, snap

T_FAIL = 800.0
T_RESC = 1800.0
U_SCAN = np.round(np.arange(0.20, 1.01, 0.05), 4)

# every model parameter (numerics + initial state excluded)
PERTURB = [k for k in asdict(Params())
           if k not in ("G_floor", "a0", "G0", "D0", "S0", "g_init")]
DELTAS = (-0.30, -0.20, 0.20, 0.30)


def log_bisect_dur(p: Params, lo=5.0, hi=200.0, iters=12):
    """P1: episode-duration threshold, log-bisected.  None if the bracket
    does not straddle (no collapse even at dur=200, or collapse at dur=5)."""
    def collapsed(d):
        sch = episode_schedule(t1=snap(100.0 + d))
        return simulate(p, sch, T_FAIL)["G"][-1] < 0.1
    if collapsed(lo) or not collapsed(hi):
        return None
    for _ in range(iters):
        mid = snap(np.sqrt(lo * hi))
        if collapsed(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def measure(p: Params) -> dict:
    out = dict(thr_dur=np.nan, no_self=np.nan, u_thr=np.nan,
               peak_partial=np.nan, g_up=np.nan, lag_ratio=np.nan)

    # P1 + P2 from the canonical failure run
    fail = simulate(p, episode_schedule(), T_FAIL)
    out["no_self"] = float(fail["G"][fail["t"] >= 300.0].max() < 0.1)
    out["thr_dur"] = log_bisect_dur(p)

    # P3 (+P4 stand-in) from the rescue-dose scan
    ep = episode_schedule()
    t_r = snap(400.0)
    peak = 0.0
    for u in U_SCAN:
        sol = simulate(p, rescue_schedule(ep, t_r, 60.0, float(u)), T_RESC)
        lab = rescue_taxonomy(sol, p, t_r)
        if lab == "full" and np.isnan(out["u_thr"]):
            out["u_thr"] = float(u)
        t_coll = first_below(sol["G"], sol["t"])
        if t_coll is not None and lab != "full":
            w = sol["t"] >= t_coll
            peak = max(peak, float(sol["G"][w].max()))
    out["peak_partial"] = peak
    if np.isnan(out["u_thr"]):
        out["u_thr"] = np.nan if peak < 0.5 else np.nan  # no full rescue

    # P5 + P6 from the canonical full-rescue run
    sol = simulate(p, rescue_schedule(ep, t_r, 60.0, 0.8), T_RESC)
    if rescue_taxonomy(sol, p, t_r) == "full":
        out["g_up"] = sol["g"][-1] / p.g0 - 1.0
        t = sol["t"]
        for key, x in (("G", sol["G"]), ("S", sol["S"])):
            w = t >= t_r
            tgt = 0.9 * x[-1]
            idx = np.where(x[w] >= tgt)[0]
            if idx.size == 0:
                out["lag_ratio"] = np.nan
                break
            out[key] = float(t[w][idx[0]] - t_r)
        if "G" in out and "S" in out and out["G"] > 0:
            out["lag_ratio"] = out["S"] / out["G"]
    return out


def _cell(job):
    name, frac = job
    val = getattr(Params(), name) * (1 + frac)
    p = replace(Params(), **{name: val})
    if name == "g0":
        # g0 is g's baseline AND (unperturbed) initial condition: propagate
        # the perturbation to g_init so g_up measures adaptation to the
        # episode, not relaxation from g_init=0.5 to the moved baseline
        # (mirrors exp1's g0 axis and exp3's g0 sweep).
        p = replace(p, g_init=val)
    m = measure(p)
    m.update(param=name, frac=frac)
    return m


def main() -> int:
    os.makedirs("cache", exist_ok=True)
    cells = [(k, f) for k in PERTURB for f in DELTAS]
    # baseline (unperturbed) row first
    base = measure(Params())
    base.update(param="(base)", frac=0.0)
    with Pool(16) as pool:
        rows = pool.map(_cell, cells)
    rows = [base] + rows

    cols = ["thr_dur", "no_self", "u_thr", "peak_partial", "g_up", "lag_ratio"]
    print(f"{'param':10s} {'frac':>5s} " + " ".join(f"{c:>13s}" for c in cols))
    for r in rows:
        vals = " ".join(f"{r[c]:13.4f}" if not np.isnan(r[c])
                        else f"{'--':>13s}" for c in cols)
        print(f"{r['param']:10s} {r['frac']:+5.2f} {vals}")

    np.savez(os.path.join("cache", "sensitivity.npz"),
             params=np.array([r["param"] for r in rows]),
             fracs=np.array([r["frac"] for r in rows]),
             **{c: np.array([r[c] for r in rows], float) for c in cols})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
