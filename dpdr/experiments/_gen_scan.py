"""Landscape scan (design-time, NOT the experiment): measure the frozen
model's own modification landscape at the candidate evaluation states, on
the three Params levers, with window.py's rollout_self/_v_plus read-only.

Purpose: DERIVE, not tune — locate (i) which Params axes the V-based
evaluation can distinguish at which states/horizons, and (ii) whether any
axis has the local-vs-global property (short-horizon rollout prefers a
change whose LONG-horizon rollout / true inter-generational effect is
harmful).  Nothing here picks constants for the experiment; it decides
whether the pre-registered mechanism prediction is supportable at all.
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

from dpdr.events import Schedule
from dpdr.integrate import simulate
from dpdr.model import Params
from dpdr.window import WindowParams, _v_plus, rollout_self

P = Params()
WP = WindowParams()          # tau_sim=200, w_v=0.02, delta_roll=10 (defaults)
T_GRID = (10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0)
Deltas = {"eta": 0.15 * P.eta, "eps0": 0.15 * P.eps0, "g0": 0.15 * P.g0}


def gen_schedule() -> Schedule:
    """Canonical generation task: episode [100,200) a_hold 0.9 + pulse 0.5
    to 160, rescue u_ext 0.8 on [800,1000).  Same as metrics' rescue
    scenario geometry."""
    return Schedule({
        "a_hold": [(100.0, 200.0, 0.9)],
        "A": [(100.0, 160.0, 0.5)],
        "u_ext": [(800.0, 1000.0, 0.8)],
    })


def Teff_sw(S, p):
    return p.Theta * S / p.S_rest


def V_at(y, p, T, sch_vals):
    a_hold, A, u_ext = sch_vals
    _a, G_ol, D_ol = rollout_self(y, p, WP, T, a_hold, A, u_ext,
                                  Teff_sw(y[3], p))
    dE = (D_ol - G_ol) - (y[2] - y[1])
    return _v_plus(dE, WP.w_v)


def main() -> int:
    t0 = time.time()
    sch = gen_schedule()
    print(f"[{time.strftime('%H:%M:%S')}] frozen generation run to t=1010")
    sol = simulate(P, sch, 1010.0, dt=0.5)
    for tq in (160.0, 200.0, 250.0, 900.0, 1000.0, 1010.0):
        i = np.searchsorted(sol["t"], tq)
        y = sol["a"][i], sol["G"][i], sol["D"][i], sol["S"][i], sol["g"][i]
        print(f"  t={tq:6.0f}: a={y[0]:.3f} G={y[1]:.3f} D={y[2]:.3f} "
              f"S={y[3]:.3f} g={y[4]:.3f} E={y[2]-y[1]:+.3f} "
              f"c={sol['c'][i]:.3f}")

    # candidate evaluation states: transients where V is live
    for tq in (160.0, 900.0, 1000.0, 1010.0):
        i = int(np.searchsorted(sol["t"], tq))
        y = np.array([sol["a"][i], sol["G"][i], sol["D"][i], sol["S"][i],
                      sol["g"][i]])
        vals = (0.0, 0.0, 0.0)     # exogenous AFTER/AT these times (self-model:
        # the rollout holds current values; at t=160 a_hold is still ON)
        if tq < 200.0:
            vals = (0.9, 0.5, 0.0)
        print(f"\n=== eval state t={tq}: E={y[2]-y[1]:+.4f}, "
              f"G={y[1]:.4f}, g={y[4]:.4f} ===")
        base_V = {T: V_at(y, P, T, vals) for T in T_GRID}
        for name, d in Deltas.items():
            for sgn in (-1.0, +1.0):
                pc = Params()
                setattr(pc, name, getattr(P, name) + sgn * d)
                row = []
                for T in T_GRID:
                    v = V_at(y, pc, T, vals)
                    m = v - base_V[T]
                    row.append(f"{m:+.2e}")
                print(f"  {name:>4s}{'+-'[sgn < 0]:1s}: V-margin(T) = "
                      + " ".join(row))
        print("  base V(T) = " + " ".join(f"{base_V[T]:.2e}" for T in T_GRID))

    # true inter-generational effect of each single move: run the NEXT
    # generation with the modified params, frozen RHS, and compare
    print(f"\n=== TRUE next-generation effect of one move (frozen dynamics, "
          "full generation) ===")
    s0 = simulate(P, sch, 1010.0, dt=0.5)
    perf0 = (float(s0["G"].min()), float(s0["G"][-1]),
             float(s0["E"][-1]))
    print(f"  base: G_min={perf0[0]:.4f} G_end={perf0[1]:.4f} "
          f"E_end={perf0[2]:+.4f}")
    for name, d in Deltas.items():
        for sgn in (-1.0, +1.0):
            pc = Params()
            setattr(pc, name, getattr(P, name) + sgn * d)
            s = simulate(pc, sch, 1010.0, dt=0.5)
            print(f"  {name:>4s}{'+-'[sgn < 0]:1s}: G_min={s['G'].min():.4f} "
                  f"G_end={s['G'][-1]:.4f} E_end={s['E'][-1]:+.4f} "
                  f"dG_end={s['G'][-1]-perf0[1]:+.4f}")
    print(f"[{time.strftime('%H:%M:%S')}] scan done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
