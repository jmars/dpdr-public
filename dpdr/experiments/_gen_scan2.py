"""Landscape scan 2 (design-time, NOT the experiment): systematic search
for a LOCAL-VS-GLOBAL trap in the frozen model's own modification
landscape — a Params lever and evaluation state where the rollout-based
V-margin CHANGES SIGN with the horizon T (looks good short, bad long, or
vice versa).  Such a sign flip is the only structurally honest source of a
low-T threshold (adopting on partial lookahead); if none exists on any
natural lever at any stressed state, the pre-registration must predict
NO threshold (the falsifier side) rather than manufacture one.

Read-only reuse of window.py (rollout_self, _v_plus).  Nothing tuned.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from dpdr.events import Schedule
from dpdr.integrate import simulate
from dpdr.model import Params
from dpdr.window import WindowParams, _v_plus, rollout_self

P = Params()
WP = WindowParams()
T_GRID = (10.0, 20.0, 40.0, 80.0, 160.0, 320.0)
FRAC = 0.15

# every lever of the agent's OWN machinery (generator, switch, control
# loop); exogenous channel params (D_base, beta_D, delta_D, k_ext, u-side)
# are excluded in principle — the agent cannot modify the world.
LEVERS = ("alpha_G", "beta_G", "gam_G", "eta", "eps0",          # generator
          "Theta", "w", "sigma_c",                              # switch
          "g0", "Es", "pi", "mu", "dEdt_ref",                   # control loop
          "k_s", "lam_S",                                       # setpoint
          "k_in", "chi", "rho_a")                               # attention


def gen_schedule() -> Schedule:
    return Schedule({"a_hold": [(100.0, 200.0, 0.9)],
                     "A": [(100.0, 160.0, 0.5)],
                     "u_ext": [(800.0, 1000.0, 0.8)]})


def V_at(y, p, T, vals):
    _a, G_ol, D_ol = rollout_self(y, p, WP, T, vals[0], vals[1], vals[2],
                                  p.Theta * y[3] / p.S_rest)
    return _v_plus((D_ol - G_ol) - (y[2] - y[1]), WP.w_v)


def main() -> int:
    t0 = time.time()
    sch = gen_schedule()
    sol = simulate(P, sch, 1010.0, dt=0.5)
    # stressed arc (a_hold on): 105..195 step 10; exogenous held = current
    # (a_hold 0.9, A 0.5 while t<160; A 0 after 160)
    states = []
    for tq in range(105, 200, 10):
        i = int(np.searchsorted(sol["t"], tq))
        y = np.array([sol["a"][i], sol["G"][i], sol["D"][i], sol["S"][i],
                      sol["g"][i]])
        vals = (0.9, 0.5 if tq < 160 else 0.0, 0.0)
        states.append((float(tq), y, vals))
    # rescue arc 805..995 (u_ext 0.8 on until 1000)
    for tq in range(805, 1000, 10):
        i = int(np.searchsorted(sol["t"], tq))
        y = np.array([sol["a"][i], sol["G"][i], sol["D"][i], sol["S"][i],
                      sol["g"][i]])
        states.append((float(tq), y, (0.0, 0.0, 0.8)))
    # post-recovery 1005 (settling)
    i = int(np.searchsorted(sol["t"], 1005.0))
    states.append((1005.0, np.array([sol["a"][i], sol["G"][i], sol["D"][i],
                                     sol["S"][i], sol["g"][i]]),
                   (0.0, 0.0, 0.0)))

    flips = []
    adopt = {}          # (T -> list of (tq, lever, sgn, margin))
    for tq, y, vals in states:
        if V_at(y, P, T_GRID[-1], vals) <= 0.0 and \
           V_at(y, P, T_GRID[0], vals) <= 0.0:
            continue                      # V inert at this state
        for lever in LEVERS:
            for sgn in (-1.0, +1.0):
                pc = Params()
                setattr(pc, lever, getattr(P, lever) * (1.0 + sgn * FRAC))
                ms = np.array([V_at(y, pc, T, vals) for T in T_GRID]) \
                     - np.array([V_at(y, P, T, vals) for T in T_GRID])
                s = np.sign(ms)
                if (s > 0).any() and (s < 0).any():
                    flips.append((tq, lever, sgn, ms))
                for T, m in zip(T_GRID, ms):
                    if m < 0.0:
                        adopt.setdefault(T, []).append(
                            (tq, lever, sgn, float(m)))
    print(f"[{time.strftime('%H:%M:%S')}] scanned {len(states)} states x "
          f"{len(LEVERS)} levers x 2 signs; SIGN FLIPS ACROSS T: "
          f"{len(flips)}")
    for tq, lever, sgn, ms in flips:
        print(f"  FLIP t={tq:6.0f} {lever}{'+-'[sgn < 0]}: "
              + " ".join(f"{m:+.2e}" for m in ms))
    if not flips:
        print("  (none — no lever's V-margin changes sign with T at any "
              "stressed state)")
    print("\nadopt-signal census (margin < 0), by T:")
    for T in T_GRID:
        rows = adopt.get(T, [])
        # strongest 5 adopt signals at this T
        rows.sort(key=lambda r: r[3])
        print(f"  T={T:6.0f}: {len(rows):4d} adopt-signals; strongest: "
              + "; ".join(f"{lv}{'+-'[sg < 0]}@t{tq:.0f}={m:+.1e}"
                          for tq, lv, sg, m in rows[:5]))
    print(f"[{time.strftime('%H:%M:%S')}] scan2 done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
