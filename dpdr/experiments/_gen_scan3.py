"""Landscape scan 3 (design-time, NOT the experiment): DYNAMIC walks.

The static scans (1,2) found no trap-direction sign flip for single moves
at frozen-trajectory states.  Two things they cannot see:
  (a) ACCUMULATION — a modification applied repeatedly, with the
      evaluation state moving along the MODIFIED trajectory;
  (b) SLOW-MODE INVISIBILITY — tau_g = 200 (adaptive gain) and tau_S = 100
      (setpoint) are the slowest modes; a rollout at T << tau_g holds g
      ~frozen, so modifications of the control loop's STANDING structure
      (g0, mu, pi) are nearly invisible at low T and their long-run
      effect (the correction term is a LIFTER at E>0 but a BRAKE at E<0)
      only enters the rollout at T ~ tau_g.

This scan walks single levers multiplicatively (the candidate M-rule:
adopt the move the rollout certifies, repeat next generation), measuring
at each step (i) the V-margins of every CONTROL-LOOP lever at that
generation's own stressed state, at several T, and (ii) the TRUE
next-generation performance (G_min dip, G_end, E_end).  It answers: does
any walk reach a region that is TRULY harmful while still looking
ADOPTIVE at low T and REJECTABLE at high T?  That is the legitimate
local-vs-global trap; if no walk does, the pre-registration must predict
the falsifier.
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
STEP = 1.15                       # multiplicative adoption step (1+FRAC)
WALKS = ("g0", "mu", "pi", "Es", "dEdt_ref",       # control loop (slow-side)
         "alpha_G", "eta", "Theta", "eps0")        # stress machinery (fast)
NGEN = 12

CL_LEVERS = ("g0", "Es", "pi", "mu", "dEdt_ref")


def gen_schedule() -> Schedule:
    return Schedule({"a_hold": [(100.0, 200.0, 0.9)],
                     "A": [(100.0, 160.0, 0.5)],
                     "u_ext": [(800.0, 1000.0, 0.8)]})


def eval_state(sol, p, T):
    """The agent's own alarm point: the stressed sample (t in the episode)
    where V at horizon T is maximal.  Exogenous held at current values."""
    i0, i1 = int(np.searchsorted(sol["t"], 100.0)), \
        int(np.searchsorted(sol["t"], 200.0))
    best, bi = -1.0, i0
    for i in range(i0, i1):
        y = np.array([sol["a"][i], sol["G"][i], sol["D"][i], sol["S"][i],
                      sol["g"][i]])
        _a, G_ol, D_ol = rollout_self(y, p, WP, T, 0.9, 0.5, 0.0,
                                      p.Theta * y[3] / p.S_rest)
        v = _v_plus((D_ol - G_ol) - (y[2] - y[1]), WP.w_v)
        if v > best:
            best, bi = v, i
    return bi


def margins(sol, p, T, i, levers):
    y = np.array([sol["a"][i], sol["G"][i], sol["D"][i], sol["S"][i],
                  sol["g"][i]])
    vals = (0.9, 0.5 if sol["t"][i] < 160 else 0.0, 0.0)

    def V(pp):
        _a, G_ol, D_ol = rollout_self(y, pp, WP, T, *vals,
                                      pp.Theta * y[3] / pp.S_rest)
        return _v_plus((D_ol - G_ol) - (y[2] - y[1]), WP.w_v)

    v0 = V(p)
    out = {}
    for lv in levers:
        for sgn in (-1.0, 1.0):
            pc = Params()
            pc.__dict__.update(p.__dict__)
            setattr(pc, lv, getattr(p, lv) * (1.0 + sgn * FRAC))
            out[f"{lv}{'+' if sgn > 0 else '-'}"] = V(pc) - v0
    return out, v0


def main() -> int:
    t0 = time.time()
    sch = gen_schedule()
    for lever in WALKS:
        print(f"\n===== WALK {lever} x{STEP} per generation "
              f"({NGEN} generations) =====")
        p = Params()
        for k in range(NGEN + 1):
            sol = simulate(p, sch, 1010.0, dt=0.5)
            gmin, gend, eend = (float(sol["G"].min()), float(sol["G"][-1]),
                                float(sol["E"][-1]))
            # control-loop margins at the T=20 alarm state
            i20 = eval_state(sol, p, 20.0)
            m20, v20 = margins(sol, p, 20.0, i20, CL_LEVERS)
            i320 = eval_state(sol, p, 320.0)
            m320, v320 = margins(sol, p, 320.0, i320, CL_LEVERS)
            g0v = getattr(p, lever)
            s20 = " ".join(f"{k2}:{v:+.1e}" for k2, v in sorted(m20.items(),
                    key=lambda kv: kv[1])[:3])
            s320 = " ".join(f"{k2}:{v:+.1e}" for k2, v in sorted(
                m320.items(), key=lambda kv: kv[1])[:3])
            osc = float(sol["G"][sol["t"] >= 1000.0].std())
            print(f"  k={k:2d} {lever}={g0v:8.4f} G_min={gmin:.4f} "
                  f"G_end={gend:.4f} E_end={eend:+.4f} osc={osc:.4f}")
            print(f"      T=20  @{sol['t'][i20]:6.0f} V={v20:.3e} best: {s20}")
            print(f"      T=320 @{sol['t'][i320]:6.0f} V={v320:.3e} "
                  f"best: {s320}")
            p = Params()
            p.__dict__.update(p.__dict__)
            setattr(p, lever, g0v * STEP)
        if time.time() - t0 > 400:
            print("  (time guard)")
            break
    print(f"\n[{time.strftime('%H:%M:%S')}] scan3 done in "
          f"{time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
