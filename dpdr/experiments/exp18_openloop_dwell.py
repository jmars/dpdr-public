"""exp18 — the open-loop control arm and the recall-protocol dwell anchor
(ALL FROZEN MODEL, deterministic; every number below is reproducible from
this driver + cache/exp18_*.npz).

Two small anchors the paper cites that no earlier committed driver covers
(formalized from the dev-time probes recorded in the project's development
notes, nodes handoff-selfreg-openloop / -weak-demand):

  A. THE OPEN-LOOP CONTROL ARM (§4.2).  The canonical failure schedule
     (inward episode a_hold 0.9 over [100, 200) + affect pulse A = 0.5
     over [100, 160)) with the control loop OPENED AT THE PARAMS LEVEL:
     g0 = 0 and pi = 0.  g0 is the adaptive-gain baseline and pi drives
     the gain response to |dE/dt|, so with both zero the gain equation
     dg = (pi*max(0, |dE/dt| - dEdt_ref) - mu*(g - g0)) / tau_g reduces
     to dg = -mu*g/tau_g: the gain decays away and the loop's corrective
     action is gone.  The loop is genuinely open — and no schedule term
     is removed (the other constructions that drop the capture term or
     the control-gain term probe different models, not this anchor).
     Anchors (T = 900): closed (g0 = 0.5, pi = 1.5) G_end = 0.0487;
     open (g0 = 0, pi = 0) G_end = 0.0121 — the open loop collapses
     DEEPER.  Rescued variant (u_ext 0.8 for 200 t.u. from t = 600,
     T = 1400): closed after_rescue G = 0.8854; open after_rescue
     G = 0.8984 — a closed loop rescues, an open one does not.
     Supporting decomposition at T = 1000 (why BOTH params are zeroed):
     g0 = 0 alone gives 0.0106; pi = 0 alone barely moves it (0.0485 vs
     closed 0.0487); the pair (g0 = 0, pi = 0) is the recorded form.

  B. THE RECALL-PROTOCOL DWELL THRESHOLD (§4.10).  Canonical ICs, recall
     = inward episode a_hold 0.9 over [100, 100+dur), NO affect pulse and
     NO engagement, T = 900, criterion is_stuck(sol, p) at the horizon.
     Bisected on the dt = 0.05 integration grid (schedule breakpoints
     must lie on the grid): 118.65 healthy (G_end 0.8854); 118.70 STUCK
     (G_end 0.0487) — the threshold is exactly 118.70 t.u. at the grid
     resolution the protocol uses.  (exp10 part B's dwell table is a
     DIFFERENT construction — settled ICs, S re-pinned, standing u_ext —
     and reproduces the u_ext gradient, not this anchor.)

Protocol notes (honest reporting):
  * Part A's closed control is Params() unchanged (g0 = 0.5, pi = 1.5,
    the canonical values); the open arm changes ONLY (g0, pi).
  * Part B's criterion is the plan's stuck criterion (metrics.is_stuck,
    sustained > 5*tau_G), judged at T = 900 — the dev-time probe's
    horizon.  The coarse [5, 700] bisection's transition midpoint
    (118.825 after 11 iterations) brackets the transition but does not
    pin it; the recorded threshold is the smallest STUCK grid cell of
    the fine scan (118.70, whose healthy neighbour is 118.65).
  * No RHS is reimplemented: dpdr.integrate.simulate on
    dpdr.model.deriv only, default dt = 0.05.

Usage:  .venv/bin/python experiments/exp18_openloop_dwell.py [probe|A|B|all]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from dpdr.events import affect_pulse, external_demand, inward_episode
from dpdr.integrate import simulate
from dpdr.metrics import is_stuck
from dpdr.model import Params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")

DT = 0.05                   # default integration grid
EP_T0, EP_T1 = 100.0, 200.0
A_PULSE = 0.5
PULSE_DUR = 60.0
T_OPEN = 900.0              # open-loop probe horizon (dev-time record)
RESCUE_T0, RESCUE_DUR, RESCUE_U = 600.0, 200.0, 0.8
T_RESCUE = 1400.0
T_DECOMP = 1000.0           # decomposition horizon (supporting detail)
D_T0 = 100.0                # recall-protocol episode start
D_GRID = 0.05               # dwell-bisection grid (threshold resolution)

# anchors (recorded values; compared, not forced)
ANCH_OPEN = {"closed": 0.0487, "open": 0.0121,
             "closed_rescue": 0.8854, "open_rescue": 0.8984}
ANCH_DECOMP = {"closed": 0.0487, "g0_only": 0.0106,
               "pi_only": 0.0485, "open": 0.0104}
ANCH_DWELL = 118.70


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def snap(x: float, dt: float = DT) -> float:
    return round(round(x / dt) * dt, 10)


def canonical_episode() -> object:
    """The canonical failure schedule: inward episode + affect pulse."""
    s = inward_episode(EP_T0, EP_T1, 0.9)
    return affect_pulse(s, EP_T0, PULSE_DUR, A_PULSE)


def openloop_run(g0: float, pi: float, rescue: bool = False,
                 T: float = T_OPEN):
    """The failure schedule with the loop opened via (g0, pi); the rescued
    variant adds the standard rescue after the episode (T = 1400)."""
    p = Params(g0=g0, pi=pi)
    s = canonical_episode()
    if rescue:
        s = external_demand(s, RESCUE_T0, RESCUE_DUR, RESCUE_U)
        T = T_RESCUE
    return p, simulate(p, s, T)


def recall_run(dur: float):
    """The recall protocol: canonical ICs, a_hold 0.9 over [100, 100+dur),
    no affect, no engagement, T = 900."""
    p = Params()
    s = inward_episode(D_T0, snap(D_T0 + dur), 0.9)
    return p, simulate(p, s, T_OPEN)


def bisect_bool(f, lo: float, hi: float, it: int = 11,
                grid: float = D_GRID):
    """Bool bisection with grid snapping; None when the bracket agrees."""
    lo, hi = snap(lo, grid), snap(hi, grid)
    flo, fhi = f(lo), f(hi)
    if flo == fhi:
        return None
    for _ in range(it):
        mid = snap(0.5 * (lo + hi), grid)
        if mid in (lo, hi):
            break
        if f(mid) == fhi:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------- part A
def part_a() -> None:
    log("partA START: open-loop control arm (g0=0, pi=0) vs closed, "
        f"T={T_OPEN:g} / rescue T={T_RESCUE:g}")
    arms = [("closed", 0.5, 1.5, False), ("open", 0.0, 0.0, False),
            ("closed_rescue", 0.5, 1.5, True),
            ("open_rescue", 0.0, 0.0, True)]
    names, G_end, g_end, stuck = [], [], [], []
    for name, g0, pi, resc in arms:
        t0 = time.time()
        p, sol = openloop_run(g0, pi, rescue=resc)
        names.append(name)
        G_end.append(float(sol["G"][-1]))
        g_end.append(float(sol["g"][-1]))
        stuck.append(is_stuck(sol, p))
        log(f"partA {name}: G_end={G_end[-1]:.4f} (anchor "
            f"{ANCH_OPEN[name]}) g_end={g_end[-1]:.4f} "
            f"stuck={stuck[-1]} ({time.time() - t0:.1f}s)")
    # decomposition: which of the two zeroed params carries the effect
    dec_names, dec_G = [], []
    for name, g0, pi in [("closed", 0.5, 1.5), ("g0_only", 0.0, 1.5),
                         ("pi_only", 0.5, 0.0), ("open", 0.0, 0.0)]:
        _, sol = openloop_run(g0, pi, T=T_DECOMP)
        dec_names.append(name)
        dec_G.append(float(sol["G"][-1]))
        log(f"partA decomp T={T_DECOMP:g} {name}: G_end={dec_G[-1]:.4f} "
            f"(anchor {ANCH_DECOMP[name]})")
    np.savez(os.path.join(CACHE, "exp18_openloop.npz"),
             arm=np.array(names), G_end=np.array(G_end),
             g_end=np.array(g_end), stuck=np.array(stuck),
             T_open=T_OPEN, rescue_t0=RESCUE_T0,
             rescue_dur=RESCUE_DUR, rescue_u=RESCUE_U,
             T_rescue=T_RESCUE,
             decomp_arm=np.array(dec_names),
             decomp_G_end=np.array(dec_G), T_decomp=T_DECOMP)
    log("partA DONE (exp18_openloop.npz)")


# --------------------------------------------------------------- part B
def part_b() -> None:
    log("partB START: recall-protocol dwell threshold (no affect, no "
        f"engagement, T={T_OPEN:g}, is_stuck, grid {D_GRID:g})")
    durs = np.round(np.arange(118.0, 119.0001, D_GRID), 4)
    stuck = np.zeros(durs.size, bool)
    G_end = np.zeros(durs.size, float)
    for i, dur in enumerate(durs):
        t0 = time.time()
        p, sol = recall_run(float(dur))
        stuck[i] = is_stuck(sol, p)
        G_end[i] = float(sol["G"][-1])
        log(f"partB scan dur={dur:.2f}: stuck={stuck[i]} "
            f"G_end={G_end[i]:.4f} ({time.time() - t0:.1f}s)")
    thr = float(durs[stuck][0]) if stuck.any() else float("nan")
    t0 = time.time()
    p = Params()
    bis = bisect_bool(lambda d: is_stuck(recall_run(d)[1], p), 5.0, 700.0)
    log(f"partB bisect [5, 700] -> transition midpoint {bis} "
        f"({time.time() - t0:.1f}s); recorded threshold = smallest stuck "
        f"grid cell = {thr} (anchor {ANCH_DWELL})")
    np.savez(os.path.join(CACHE, "exp18_dwell.npz"),
             dur=durs, stuck=stuck, G_end=G_end, threshold=thr,
             bisected=(np.nan if bis is None else bis),
             grid=D_GRID, T=T_OPEN, a_hold=0.9)
    log("partB DONE (exp18_dwell.npz)")


# --------------------------------------------------------------- probe
def probe() -> None:
    log("probe: the four open-loop anchors + the two boundary cells "
        "(no caches written)")
    for name, g0, pi, resc in [("closed", 0.5, 1.5, False),
                               ("open", 0.0, 0.0, False),
                               ("closed_rescue", 0.5, 1.5, True),
                               ("open_rescue", 0.0, 0.0, True)]:
        p, sol = openloop_run(g0, pi, rescue=resc)
        log(f"probe {name}: G_end={sol['G'][-1]:.4f} "
            f"(anchor {ANCH_OPEN[name]}) stuck={is_stuck(sol, p)}")
    for dur in (118.65, 118.70):
        p, sol = recall_run(dur)
        log(f"probe recall dur={dur:.2f}: stuck={is_stuck(sol, p)} "
            f"G_end={sol['G'][-1]:.4f}")
    log("probe DONE")


def main(argv: list[str]) -> None:
    os.makedirs(CACHE, exist_ok=True)
    parts = argv[1:] or ["probe", "A", "B"]
    if "all" in parts:
        parts = ["A", "B"]
    todo = {"probe": probe, "A": part_a, "B": part_b}
    for name in parts:
        todo[name]()


if __name__ == "__main__":
    main(sys.argv)
