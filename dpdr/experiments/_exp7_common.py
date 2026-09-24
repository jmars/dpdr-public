"""Shared helpers + part functions for the exp7 WINDOW battery.

This module holds the *measurement logic only*; each part is driven by its
own runnable script (exp70.py, exp7a.py .. exp7e.py) so that one expensive
part can never stall the others.  See exp7_window.py for the scientific
motivation; see dpdr/window.py for the variant's math.

NUMERICS NOTE (the stall diagnosis, preserved for the record): the window RHS
re-runs its bounded self-simulation (rollout_self, O(T/delta_roll) inner RK4
steps) at EVERY solve_ivp evaluation, and a t=3000 / dt=0.5 / rtol=1e-6
integration costs a fixed ~36014 RHS evaluations regardless of T.  The per-eval
cost scales linearly with T, so a single sim costs ~6 s (T=50) .. ~74 s
(T=800).  There is NO hang and NO stiffness (nfev is identical across T); the
two stalled runs simply accumulated ~40+ sims in part (b) with no per-eval
printing.  Fix: split parts, print every evaluation with a timestamp, and use
tight brackets (from the already-measured phase diagram) with 10 bisection
iterations instead of 12 — the grids stay "tens of points per axis".
"""
from __future__ import annotations

import os
import time

import numpy as np

from dpdr.events import baseline_schedule, failure_schedule, rescue_schedule
from dpdr.integrate import simulate
from dpdr.metrics import check_gates, is_stuck
from dpdr.model import Params, Schedule
from dpdr.window import WindowParams, rollout_self, simulate_window

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
FIGDIR = os.path.join(ROOT, "figs")

P = Params()
OFF = WindowParams(T_fixed=0.0, floor=None)      # = frozen RHS exactly
FLOOR = 0.7                                      # exp6's cheap floor
C_CAPS = (0.05, 0.1, 0.2)
T_ENGAGE = 600.0                                 # post-collapse assay geometry
T_ASSAY = 1500.0
EP_T0 = 100.0
STUCK_G = 0.0485                                 # frozen stuck level G*

# bisection resolution (range/2^it); 10 iters -> ~0.1-0.2 T-units on the
# tight brackets used, more than enough for a reported T_max to one decimal
BISECT_IT = 10
BISECT_IT_FAST = 8                               # cheap boolean axes


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def snap(x: float, dt: float = 0.05) -> float:
    return round(round(x / dt) * dt, 10)


def episode1(ah: float, dur: float = 100.0) -> Schedule:
    """Single canonical episode (a_hold hold + affect pulse), exp6 geometry."""
    return Schedule({"a_hold": [(EP_T0, snap(EP_T0 + dur), ah)],
                     "A": [(EP_T0, snap(EP_T0 + min(60.0, dur)), 0.5)]})


def episodes(n: int, gap: float, dur: float, ah: float) -> Schedule:
    """Repeated-episode schedule, exp6 part-(e) geometry (dt=0.5 grid)."""
    ch: dict = {"a_hold": [], "A": []}
    for k in range(n):
        t0 = snap(EP_T0 + k * gap, 0.5)
        ch["a_hold"].append((t0, snap(t0 + dur, 0.5), ah))
        ch["A"].append((t0, snap(t0 + min(60.0, dur), 0.5), 0.5))
    return Schedule(ch)


def G_end_w(wp: WindowParams, sch: Schedule, T: float, dt: float = 0.25
            ) -> float:
    return float(simulate_window(P, wp, sch, T, dt=dt)["G"][-1])


def bisect(f, lo, hi, it=BISECT_IT):
    """Bool bisection (exp6's); returns None when the endpoints agree."""
    flo, fhi = f(lo), f(hi)
    if flo == fhi:
        return None
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if f(mid) == fhi:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ parts
def part0() -> dict:
    """Fidelity at T=0 + gates + attractor exactness + stuck-survival +
    rescue-not-blocked."""
    log("part0: frozen gates")
    out: dict = {"gates": check_gates(P)}
    for name, sch, T in (("baseline", baseline_schedule(), 1000.0),
                         ("failure", failure_schedule(), 600.0),
                         ("rescue", rescue_schedule(), 1400.0)):
        log(f"part0: T=0 wrap fidelity vs frozen ({name})")
        sf = simulate(P, sch, T)
        sw = simulate_window(P, OFF, sch, T)
        out[f"fidelity_{name}"] = float(
            np.max(np.abs(sf["G"] - sw["G"])))
    # f_T exactness at both frozen attractors (V must be 0 -> benefit inert
    # at every equilibrium by construction)
    log("part0: frozen attractors for f_T exactness")
    sH = simulate(P, baseline_schedule(), 3000.0, dt=0.25)
    sS = simulate(P, failure_schedule(), 3000.0, dt=0.25)
    wp = WindowParams(T_fixed=100.0)
    for nm, s in (("healthy", sH), ("stuck", sS)):
        y = np.array([s["a"][-1], s["G"][-1], s["D"][-1], s["S"][-1],
                      s["g"][-1]])
        _a, Gol, Dol = rollout_self(y, P, wp, 100.0, 0.0, 0.0, 0.0,
                                    P.Theta * y[3] / P.S_rest)
        out[f"attractor_dE_{nm}"] = float((Dol - Gol) - (y[2] - y[1]))
    # stuck-survival: capacity on post-collapse, NO floor -> must stay stuck
    log("part0: stuck-survival (T=200 on, no floor)")
    s = simulate_window(P, WindowParams(T_fixed=200.0, t_engage=T_ENGAGE),
                        failure_schedule(), T_ASSAY, dt=0.25)
    out["stuck_survival_G_end"] = float(s["G"][-1])
    out["stuck_survival_is_stuck"] = bool(is_stuck(s, P))
    # ... and the floor still escapes WITH capacity on
    log("part0: floor escape with T on")
    s = simulate_window(P, WindowParams(T_fixed=200.0, floor=FLOOR,
                                        t_engage=T_ENGAGE),
                        failure_schedule(), T_ASSAY, dt=0.25)
    out["stuck_escape_with_T_G_end"] = float(s["G"][-1])
    # rescue-not-blocked: capacity on from t=0 does not block G2c
    log("part0: rescue-not-blocked")
    s = simulate_window(P, WindowParams(T_fixed=200.0), rescue_schedule(),
                        1400.0, dt=0.25)
    out["rescue_with_T_G_end"] = float(s["G"][-1])
    return out


def part_a() -> dict:
    """2D phase diagram: healthy G_end(T, c_cap) + stressed dip probe."""
    Ts = np.array([50.0, 100.0, 200.0, 300.0, 500.0, 800.0])
    healthy = np.zeros((len(C_CAPS), len(Ts)))
    for i, cc in enumerate(C_CAPS):
        for j, Tv in enumerate(Ts):
            log(f"part_a: healthy c_cap={cc:4.2f} T={Tv:5.0f} ...")
            healthy[i, j] = G_end_w(WindowParams(T_fixed=Tv, c_cap=cc),
                                    baseline_schedule(), 3000.0, dt=0.5)
            log(f"part_a:           -> G_end={healthy[i, j]:.4f}")
    # what capacity BUYS: dip depth on the canonical episode, floor on
    dipT = np.array([0.0, 50.0, 100.0, 200.0, 400.0])
    dips = np.zeros(len(dipT))
    Gends = np.zeros(len(dipT))
    for j, Tv in enumerate(dipT):
        log(f"part_a: canonical dip probe T={Tv:4.0f} (floor on)")
        s = simulate_window(P, WindowParams(T_fixed=Tv, floor=FLOOR),
                            episode1(0.9), 800.0, dt=0.25)
        dips[j], Gends[j] = float(s["G"].min()), float(s["G"][-1])
        log(f"part_a:           -> dip G_min={dips[j]:.4f} G_end={Gends[j]:.4f}")
    return dict(Ts=Ts, c_caps=np.array(C_CAPS), healthy=healthy,
                c_int=C_CAPS[1] * Ts / 200.0, dipT=dipT, dips=dips,
                dip_G_ends=Gends)


def _tmax_brackets():
    """Tight (lo, hi) brackets for the healthy T_max bisection, derived from
    the already-measured phase diagram (deterministic).  Falls back to the
    conservative original ranges if the cache is absent."""
    defaults = {0.05: (300.0, 500.0), 0.1: (200.0, 300.0), 0.2: (100.0, 200.0)}
    path = os.path.join(CACHE, "exp7_phase.npz")
    try:
        d = np.load(path)
        Ts, ccaps, healthy = d["Ts"], d["c_caps"], d["healthy"]
        out = {}
        for cc, lo0, hi0 in defaults.values() if False else []:
            pass
        for i, cc in enumerate(ccaps):
            row = healthy[i]
            above = Ts[row > 0.5]
            below = Ts[row < 0.5]
            if above.size and below.size and above.max() < below.min():
                out[float(cc)] = (float(above.max()), float(below.min()))
            else:
                out[float(cc)] = defaults[float(cc)]
        return out
    except (FileNotFoundError, KeyError, ValueError):
        return defaults


def part_b() -> dict:
    """Bisected edges + benefit argmax; T_set calibration for parts (d)/(e)."""
    out: dict = {}
    brackets = _tmax_brackets()
    # T_max per c_cap: healthy G_end > 0.5 (the MEASURED upper edge)
    Tmax = {}
    for cc in C_CAPS:
        lo, hi = brackets[cc]
        log(f"part_b: bisect T_max @ c_cap={cc:4.2f} in [{lo:.0f},{hi:.0f}]")

        def crit(Tv, _cc=cc):
            g = G_end_w(WindowParams(T_fixed=Tv, c_cap=_cc),
                        baseline_schedule(), 3000.0, dt=0.5)
            log(f"part_b:   c_cap={_cc:4.2f} eval T={Tv:7.2f} -> G_end={g:.4f}"
                f" {'(>0.5)' if g > 0.5 else '(<=0.5)'}")
            return g > 0.5

        Tmax[cc] = bisect(crit, lo, hi, it=BISECT_IT)
        log(f"part_b: T_max @ c_cap={cc:4.2f} = {Tmax[cc]}")
    out["T_max"] = Tmax
    out["T_max_c_int"] = {cc: (None if v is None else cc * v / 200.0)
                          for cc, v in Tmax.items()}
    # chronic-a_hold anchor on the frozen axis (same criterion)
    log("part_b: bisect frozen chronic a_hold_crit (healthy)")
    out["a_hold_crit_healthy"] = bisect(
        lambda ah: G_end_w(OFF, Schedule({"a_hold": [(0.0, 3000.0, ah)]}),
                           3000.0, dt=0.5) > 0.5,
        0.05, 0.30, it=BISECT_IT)
    log(f"part_b: a_hold_crit_healthy = {out['a_hold_crit_healthy']:.4f}")
    # benefit axis for T_set calibration: REGULATED (m2-gated) dip margin vs
    # T_set on the canonical episode, floor on.
    scanT = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0])
    dip_reg = np.zeros(len(scanT))
    for j, Tv in enumerate(scanT):
        if Tv == 0.0:
            log("part_b: regulated dip margin T_set=0 (floor only)")
            s0 = simulate_window(P, WindowParams(T_fixed=0.0, floor=FLOOR),
                                 episode1(0.9), 800.0, dt=0.25)
            dip_reg[j] = float(s0["G"].min())
        else:
            log(f"part_b: regulated dip margin T_set={Tv:4.0f}")
            s = simulate_window(P, WindowParams(regulate=True, T_set=Tv,
                                                floor=FLOOR),
                                episode1(0.9), 800.0, dt=0.25)
            dip_reg[j] = float(s["G"].min())
        log(f"part_b:   -> dip G_min={dip_reg[j]:.4f}")
    out["scanT"], out["dip_reg"] = scanT, dip_reg
    out["margin_reg"] = dip_reg - dip_reg[0]
    # T_set calibration: margin argmax clamped to T_max/2
    imax = int(np.argmax(out["margin_reg"]))
    Tmax10 = out["T_max"][0.1]
    out["T_argmax"] = float(scanT[imax])
    out["T_set"] = round(min(scanT[imax], 0.5 * Tmax10) / 10.0) * 10.0
    log(f"part_b: T_argmax={out['T_argmax']:.0f} -> T_set={out['T_set']:.0f}")
    return out


def part_c() -> dict:
    """CONJECTURE TEST: fine low-T grid of the dip-shallowing benefit +
    certified M; plus the SUSTAINED-stress tolerance axis."""
    fineT = np.concatenate(([0.0], np.arange(10.0, 130.0, 10.0)))
    dips, Ms = np.zeros(len(fineT)), np.zeros(len(fineT))
    log("part_c: canonical dip-shallowing fine grid T=0..120 (floor on)")
    s0 = simulate_window(P, WindowParams(T_fixed=0.0, floor=FLOOR),
                         episode1(0.9), 800.0, dt=0.25)
    dip0 = float(s0["G"].min())
    for j, Tv in enumerate(fineT):
        s = simulate_window(P, WindowParams(T_fixed=Tv, floor=FLOOR),
                            episode1(0.9), 800.0, dt=0.25)
        dips[j] = float(s["G"].min())
        Ms[j] = float(s["M"].max())
        log(f"part_c:   T={Tv:5.1f} -> dip={dips[j]:.4f} M_max={Ms[j]:.3f}")
    dGmins = dips - dip0
    # monotone after the tiny low-T cost transient (T >= 50)?
    mono_hi = bool(np.all(np.diff(dGmins[5:]) >= -1e-9))
    dip_deepen_lowT = float(dGmins[1:5].min())   # most-negative (deepening)
    # SUSTAINED-stress tolerance: dip crosses 0.1 during a dur=300 episode.
    # Bracket [0.5, 1.5] spans the crossing at every T (measured T=0 crossing
    # is 1.394; capacity lowers it monotonically).
    def ep_sustained(ah: float, dur: float = 300.0) -> Schedule:
        return Schedule({"a_hold": [(EP_T0, snap(EP_T0 + dur), ah)],
                         "A": [(EP_T0, 160.0, 0.5)]})

    def dip_ok(ah: float, Tv: float) -> bool:
        s = simulate_window(P, WindowParams(T_fixed=Tv, floor=FLOOR),
                            ep_sustained(ah), 800.0, dt=0.5)
        return float(s["G"].min()) > 0.1

    susT = np.array([0.0, 100.0, 200.0, 300.0])
    acrit = np.zeros(len(susT))
    for j, Tv in enumerate(susT):
        log(f"part_c: sustained-stress tolerance bisect T={Tv:4.0f}")

        def crit(ah, _Tv=Tv):
            ok = dip_ok(ah, _Tv)
            log(f"part_c:   T={_Tv:4.0f} eval a_hold={ah:.4f} -> "
                f"dip {'ok' if ok else 'deep'}")
            return ok

        acrit[j] = bisect(crit, 0.5, 1.5, it=BISECT_IT_FAST)
        log(f"part_c:   a_hold_crit(T={Tv:.0f}) = {acrit[j]:.4f}")
    return dict(fineT=fineT, dips=dips, dGmins=dGmins, Ms=Ms,
                a_hold_crit_sustained=acrit, susT=susT,
                monotone_hi=mono_hi, dip_deepen_lowT=dip_deepen_lowT,
                verdict=("GRADED (smooth onset; no lower threshold; "
                         "UNTESTABLE — the benefit term is monotone in T "
                         "by construction, so no lower threshold COULD "
                         "have appeared)"
                         if mono_hi else
                         "NON-MONOTONE (interior structure)"))


def part_d(T_set: float) -> dict:
    """Long horizon, 4 arms x 2 patterns; margins, not binary counts."""
    arms = {
        "unregulated": WindowParams(T_fixed=0.0, floor=None),
        "floor-only": WindowParams(T_fixed=0.0, floor=FLOOR),
        "window-only": WindowParams(regulate=True, T_set=T_set),
        "floor+window": WindowParams(regulate=True, T_set=T_set, floor=FLOOR),
    }
    patterns = [("canonical x60 gap300", 60, 300.0, 100.0, 0.9),
                ("dense-weak x100 gap200", 100, 200.0, 60.0, 0.9)]
    out: dict = {"patterns": [p[0] for p in patterns], "T_set": T_set}
    for label, n, gap, dur, ah in patterns:
        sch = episodes(n, gap, dur, ah)
        T = snap(EP_T0 + n * gap + 300.0, 0.5)
        log(f"part_d: {label} (horizon {T:.0f}, T_set={T_set:.0f})")
        out[label] = {}
        for name, wp in arms.items():
            log(f"part_d:   arm={name} ...")
            s = simulate_window(P, wp, sch, T, dt=0.5)
            t, G, Tc = s["t"], s["G"], s["T"]
            dips = []
            for k in range(n):
                t0 = EP_T0 + k * gap
                w = (t >= t0 - 1e-9) & (t < t0 + gap - 1e-9)
                dips.append(float(G[w].min()) if w.any() else np.nan)
            dips = np.array(dips)
            out[label][name] = dict(
                G_min=float(G.min()), G_end=float(G[-1]),
                dip_min=float(np.nanmin(dips)),
                dip_med=float(np.nanmedian(dips)),
                dips=dips, t=t, G=G, Tcap=Tc,
                frac_below_05=float((G < 0.5).mean()),
                T_max=float(Tc.max()) if Tc is not None else 0.0,
                T_mean_above_10=float(Tc[Tc > 10.0].mean())
                if np.any(Tc > 10.0) else 0.0,
                stuck=bool(is_stuck(s, P)))
            log(f"part_d:   {name:13s} -> G_min={out[label][name]['G_min']:.4f}"
                f" G_end={out[label][name]['G_end']:.4f}"
                f" dip_med={out[label][name]['dip_med']:.4f}")
        # dt sensitivity on the heaviest arm (compare on the coarse grid)
        log(f"part_d:   dt-sensitivity check (floor+window, dt=0.25)")
        s5 = simulate_window(P, arms["floor+window"], sch, T, dt=0.25)
        G_coarse = out[label]["floor+window"]["G"]
        t_coarse = out[label]["floor+window"]["t"]
        G_fine_on_coarse = np.interp(t_coarse, s5["t"], s5["G"])
        out[label]["dt_check_maxdG"] = float(
            np.max(np.abs(G_fine_on_coarse - G_coarse)))
        out[label]["dt_check_t_coarse"] = t_coarse
        out[label]["dt_check_G_coarse"] = G_coarse
        out[label]["dt_check_G_fine"] = G_fine_on_coarse
    return out


def part_e(T_set: float) -> dict:
    """Healthy-regime cost of the window regulator (must not degrade
    baseline), plus the exact c-int = a_hold equivalence check."""
    out: dict = {}
    log("part_e: frozen baseline reference")
    sb = simulate(P, baseline_schedule(), 2000.0, dt=0.25)
    out["frozen_G_end"] = float(sb["G"][-1])
    for name, wp in (("floor-only", WindowParams(T_fixed=0.0, floor=FLOOR)),
                     ("window-only", WindowParams(regulate=True,
                                                  T_set=T_set)),
                     ("floor+window", WindowParams(regulate=True, T_set=T_set,
                                                   floor=FLOOR))):
        log(f"part_e: {name} on baseline")
        s = simulate_window(P, wp, baseline_schedule(), 2000.0, dt=0.25)
        out[f"{name}_G_end"] = float(s["G"][-1])
        out[f"{name}_maxdG"] = float(np.max(np.abs(s["G"] - sb["G"])))
        out[f"{name}_T_max"] = float(s["T"].max()) if wp.regulate else 0.0
        out[f"{name}_T_end"] = float(s["T"][-1]) if wp.regulate else 0.0
        log(f"part_e:   {name:13s} G_end={out[name+'_G_end']:.4f}"
            f" maxdG={out[name+'_maxdG']:.2e}")
    # standing-capacity equivalence: window T=400 (c_int=0.2 at c_cap=.1)
    log("part_e: c_int = a_hold equivalence check (healthy axis)")
    sw = simulate_window(P, WindowParams(T_fixed=400.0), baseline_schedule(),
                         3000.0, dt=0.5)
    sf = simulate(P, Schedule({"a_hold": [(0.0, 3000.0, 0.2)]}), 3000.0,
                  dt=0.5)
    out["equiv_window_G_end"] = float(sw["G"][-1])
    out["equiv_frozen_G_end"] = float(sf["G"][-1])
    out["equiv_maxdG"] = float(np.max(np.abs(sw["G"] - sf["G"])))
    log(f"part_e:   window T=400 G_end={out['equiv_window_G_end']:.4f} vs "
        f"frozen a_hold=0.2 G_end={out['equiv_frozen_G_end']:.4f}, "
        f"max|dG|={out['equiv_maxdG']:.2e}")
    return out
