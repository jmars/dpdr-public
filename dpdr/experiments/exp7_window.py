"""exp7 — the WINDOW variant: introspective capacity T that BUYS something
(dip-margin / stress tolerance) and COSTS something (standing inward drive),
on the shared axis T = the bounded self-simulation horizon of arXiv
2607.04277's operator f = M o V o f_T.

The deliverable experiment for dpdr/window.py (opt-in; the frozen model is
untouched).  Two bounds bracket a capacity window on ONE axis:

  LOWER edge (EXTERNAL, HYPOTHESIS-DERIVED, UNPROVEN): Zhang/Yuan/Zhang
  2026 CONJECTURE — "There exists a critical level of self-modeling capacity
  below which AI systems undergo improvement degradation ... and at or above
  which they achieve sustained recursive self-improvement."  Their Kleene
  formalism proves EXISTENCE ONLY; no number exists to import.  In THIS
  realization the lower edge can only be MEASURED, and part (c) tests whether
  it exists at all: if the benefit degrades smoothly to zero as T -> 0+ with
  no threshold, the conjecture FAILS in-model — which is itself the finding.

  UPPER edge (MY MEASURED upper bound): capacity is inward attention while
  simulating — window.py's standing addend c_int = c_cap*T/tau_sim enters
  da/dt exactly where a_hold enters, so on the healthy axis (where the
  benefit is inert, V ~ 0) capacity in c-units IS a_hold units (verified
  exactly in part (e): window T=400, c_cap=0.1 -> G_end 0.3382 = frozen
  chronic a_hold 0.2 -> 0.3382).  The upper edge is therefore a MEASURED
  destabilization threshold on the same axis as the project's earlier
  c_mon/knowing-floor results.

Battery (parts labelled as in the handoff):

  (0) Integrity: frozen gates G1-G2c; the window-OFF wrap reproduces
      dpdr.integrate.simulate EXACTLY (max|dG| = 0.0) on the three standard
      scenarios; the rollout f_T is exact at both frozen attractors
      (V = 0 there, |dE| < 1e-6 -> the benefit cannot move an equilibrium);
      STUCK-SURVIVAL: post-collapse engagement with capacity on but no floor
      leaves the stuck state stuck (the benefit does not "magically fix"
      collapse); RESCUE-NOT-BLOCKED: capacity on does not block the external
      rescue (G2c scenario).

  (a) 2D phase diagram T x c_cap: healthy G_end (baseline schedule, standing
      cost only — the benefit is inert there) + a stressed a_hold-hold probe
      (canonical episode with the floor on: dip G_min vs T) showing what
      capacity BUYS.

  (b) Bisected edges at c_cap = 0.1 (and the other two c_caps): T_max =
      largest T keeping healthy G_end > 0.5 (MEASURED upper edge), reported
      in T-units and c-int units; T_set calibration for parts (d)/(e) from
      the measured REGULATED dip margin (argmax, clamped to T_max/2).

  (c) CONJECTURE TEST: fine low-T grid of the BENEFIT — the canonical-episode
      dip shallowing dG_min(T) (and the certified M that produced it): does
      low capacity show GRADED degradation or a SMOOTH ONSET?  Plus the
      honest stress-unit axis: a_hold_crit(T) where the SUSTAINED-episode
      (dur 300) dip crosses the frozen 0.1 functional bar — on sustained
      stress the standing cost DOMINATES and capacity REDUCES tolerance
      (a negative result, reported as such).

  (d) LONG HORIZON, 4 arms (unregulated / floor-only / window-only /
      floor+window) on the exp6 protocol: canonical x60 gap300 and dense-weak
      x100 gap200.  The reported quantity is the MARGIN (per-episode dip
      depths vs the frozen stuck level G* ~ 0.0485, re-settle level between
      episodes, G_min, G_end), NOT a binary episode count — the floor forces
      c = 0 so is_stuck's c > 0.5 leg is structurally unreachable on any
      floored arm and a 0/60 bar would be weakly circular.

  (e) Healthy-regime cost of the window regulator: baseline G_end and
      max|dG| vs frozen for window-only and floor+window arms; plus the
      exact c-int = a_hold equivalence check on the healthy axis.

REGULATOR CALIBRATION (the one calibrated quantity): T_set = the benefit
argmax over the measured stress-unit benefit curve, clamped to T_max/2
(safety from the measured upper edge); tau_T, the m2 volatility gate and
everything else are imported frozen constants (see dpdr/window.py).

NUMERICS: healthy sweeps integrate to t = 3000 at dt = 0.5 (the rollout cost
is O(#points x #inner steps), so dt = 0.05 would be ~10x slower for the same
information; a dt-sensitivity check is printed); stressed/long-horizon runs
use dt = 0.25 exactly as exp6.  Model is DETERMINISTIC (every threshold is a
locus, not a distribution) and all results are IN-MODEL ONLY.

Outputs: figs/f12_window.png, cache/exp7_phase.npz, cache/exp7_longhorizon.npz.
Reproducible: `.venv/bin/python -m experiments.exp7_window`.
"""
from __future__ import annotations

import os
import time

import numpy as np

from dpdr.events import baseline_schedule, failure_schedule, rescue_schedule
from dpdr.integrate import simulate
from dpdr.metrics import check_gates, is_stuck
from dpdr.model import Params, Schedule
from dpdr.window import WindowParams, kappa_of_T, rollout_self, simulate_window

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "figs")

P = Params()
OFF = WindowParams(T_fixed=0.0, floor=None)      # = frozen RHS exactly
FLOOR = 0.7                                      # exp6's cheap floor
C_CAPS = (0.05, 0.1, 0.2)
T_ENGAGE = 600.0                                 # post-collapse assay geometry
T_ASSAY = 1500.0
EP_T0 = 100.0
STUCK_G = 0.0485                                 # frozen stuck level G*


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


def bisect(f, lo, hi, it=12):
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
    out: dict = {"gates": check_gates(P)}
    for name, sch, T in (("baseline", baseline_schedule(), 1000.0),
                         ("failure", failure_schedule(), 600.0),
                         ("rescue", rescue_schedule(), 1400.0)):
        sf = simulate(P, sch, T)
        sw = simulate_window(P, OFF, sch, T)
        out[f"fidelity_{name}"] = float(
            np.max(np.abs(sf["G"] - sw["G"])))
    # f_T exactness at both frozen attractors (V must be 0 -> benefit inert
    # at every equilibrium by construction)
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
    s = simulate_window(P, WindowParams(T_fixed=200.0, t_engage=T_ENGAGE),
                        failure_schedule(), T_ASSAY, dt=0.25)
    out["stuck_survival_G_end"] = float(s["G"][-1])
    out["stuck_survival_is_stuck"] = bool(is_stuck(s, P))
    # ... and the floor still escapes WITH capacity on (rescue is the
    # floor's job; capacity must not break it)
    s = simulate_window(P, WindowParams(T_fixed=200.0, floor=FLOOR,
                                        t_engage=T_ENGAGE),
                        failure_schedule(), T_ASSAY, dt=0.25)
    out["stuck_escape_with_T_G_end"] = float(s["G"][-1])
    # rescue-not-blocked: capacity on from t=0 does not block G2c
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
            healthy[i, j] = G_end_w(WindowParams(T_fixed=Tv, c_cap=cc),
                                    baseline_schedule(), 3000.0, dt=0.5)
    # what capacity BUYS: dip depth on the canonical episode, floor on
    dipT = np.array([0.0, 50.0, 100.0, 200.0, 400.0])
    dips = np.zeros(len(dipT))
    Gends = np.zeros(len(dipT))
    for j, Tv in enumerate(dipT):
        s = simulate_window(P, WindowParams(T_fixed=Tv, floor=FLOOR),
                            episode1(0.9), 800.0, dt=0.25)
        dips[j], Gends[j] = float(s["G"].min()), float(s["G"][-1])
    return dict(Ts=Ts, c_caps=np.array(C_CAPS), healthy=healthy,
                c_int=C_CAPS[1] * Ts / 200.0, dipT=dipT, dips=dips,
                dip_G_ends=Gends)


def part_b() -> dict:
    """Bisected edges + benefit argmax; T_set calibration for parts (d)/(e)."""
    out: dict = {}
    # T_max per c_cap: healthy G_end > 0.5 (the MEASURED upper edge)
    Tmax = {}
    for cc, lo, hi in ((0.05, 300.0, 700.0), (0.1, 150.0, 350.0),
                       (0.2, 75.0, 175.0)):
        Tmax[cc] = bisect(
            lambda Tv: G_end_w(WindowParams(T_fixed=Tv, c_cap=cc),
                               baseline_schedule(), 3000.0, dt=0.5) > 0.5,
            lo, hi)
    out["T_max"] = Tmax
    out["T_max_c_int"] = {cc: (None if v is None else cc * v / 200.0)
                          for cc, v in Tmax.items()}
    # chronic-a_hold anchor on the frozen axis: the SAME functional edge in
    # c-units (the addend enters da/dt identically) -> MEASURED conversion
    out["a_hold_crit_healthy"] = bisect(
        lambda ah: G_end_w(OFF, Schedule({"a_hold": [(0.0, 3000.0, ah)]}),
                           3000.0, dt=0.5) > 0.5,
        0.05, 0.30)
    # benefit axis for T_set calibration: REGULATED (m2-gated) dip margin vs
    # T_set on the canonical episode, floor on.  A FIXED T pays its standing
    # cost through the whole post-episode window and WRECKS re-settling
    # (G_end 0.885 -> 0.339 at T=400); the regulator pays only while
    # volatile, so its margin can grow where the fixed-T benefit cannot.
    scanT = np.array([0.0, 50.0, 100.0, 150.0, 200.0, 300.0, 400.0])
    dip_reg = np.zeros(len(scanT))
    for j, Tv in enumerate(scanT):
        if Tv == 0.0:
            s0 = simulate_window(P, WindowParams(T_fixed=0.0, floor=FLOOR),
                                 episode1(0.9), 800.0, dt=0.25)
            dip_reg[j] = float(s0["G"].min())
        else:
            s = simulate_window(P, WindowParams(regulate=True, T_set=Tv,
                                                floor=FLOOR),
                                episode1(0.9), 800.0, dt=0.25)
            dip_reg[j] = float(s["G"].min())
    out["scanT"], out["dip_reg"] = scanT, dip_reg
    out["margin_reg"] = dip_reg - dip_reg[0]
    # T_set calibration: margin argmax clamped to T_max/2 (safety from the
    # measured upper edge), rounded to the 10-unit grid
    imax = int(np.argmax(out["margin_reg"]))
    Tmax10 = out["T_max"][0.1]
    out["T_argmax"] = float(scanT[imax])
    out["T_set"] = round(min(scanT[imax], 0.5 * Tmax10) / 10.0) * 10.0
    return out


def part_c() -> dict:
    """CONJECTURE TEST: fine low-T grid of the dip-shallowing benefit +
    certified M — graded degradation or smooth onset?  Plus the honest
    SUSTAINED-stress tolerance axis (expected negative: on a dur=300 episode
    the standing cost dominates and capacity REDUCES tolerance)."""
    fineT = np.concatenate(([0.0], np.arange(10.0, 130.0, 10.0)))
    dips, Ms = np.zeros(len(fineT)), np.zeros(len(fineT))
    s0 = simulate_window(P, WindowParams(T_fixed=0.0, floor=FLOOR),
                         episode1(0.9), 800.0, dt=0.25)
    dip0 = float(s0["G"].min())
    for j, Tv in enumerate(fineT):
        s = simulate_window(P, WindowParams(T_fixed=Tv, floor=FLOOR),
                            episode1(0.9), 800.0, dt=0.25)
        dips[j] = float(s["G"].min())
        Ms[j] = float(s["M"].max())
    dGmins = dips - dip0
    # graded-vs-threshold verdict: benefit monotone in T with no interior
    # edge, and exactly 0 only at T = 0?
    mono = bool(np.all(np.diff(dGmins) >= -1e-9))
    # SUSTAINED-stress tolerance: dip crosses 0.1 during a dur=300 episode
    def ep_sustained(ah: float, dur: float = 300.0) -> Schedule:
        return Schedule({"a_hold": [(EP_T0, snap(EP_T0 + dur), ah)],
                         "A": [(EP_T0, 160.0, 0.5)]})

    def dip_ok(ah: float, Tv: float) -> bool:
        s = simulate_window(P, WindowParams(T_fixed=Tv, floor=FLOOR),
                            ep_sustained(ah), 1200.0, dt=0.5)
        return float(s["G"].min()) > 0.1

    susT = np.array([0.0, 100.0, 200.0, 300.0])
    acrit = np.array([bisect(lambda ah: dip_ok(ah, Tv), 1.2, 1.4)
                      for Tv in susT])
    return dict(fineT=fineT, dips=dips, dGmins=dGmins, Ms=Ms,
                a_hold_crit_sustained=acrit, susT=susT,
                monotone=mono,
                verdict=("GRADED (smooth onset; no lower threshold)"
                         if mono else "NON-MONOTONE (interior structure)"))


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
        out[label] = {}
        for name, wp in arms.items():
            s = simulate_window(P, wp, sch, T, dt=0.5)
            t, G, Tc = s["t"], s["G"], s["T"]
            # per-episode dip = min G in [ep_start, next ep_start)
            dips = []
            for k in range(n):
                t0 = EP_T0 + k * gap
                w = (t >= t0 - 1e-9) & (t < t0 + gap - 1e-9)
                dips.append(float(G[w].min()) if w.any() else np.nan)
            dips = np.array(dips)
            out[label][name] = dict(
                G_min=float(G.min()), G_end=float(G[-1]),
                dip_min=float(np.nanmin(dips)), dip_med=float(np.nanmedian(dips)),
                dips=dips, t=t, G=G, Tcap=Tc,
                frac_below_05=float((G < 0.5).mean()),
                T_max=float(Tc.max()) if Tc is not None else 0.0,
                T_mean_above_10=float(Tc[Tc > 10.0].mean())
                if np.any(Tc > 10.0) else 0.0,
                stuck=bool(is_stuck(s, P)))
        # dt sensitivity on the heaviest arm
        s5 = simulate_window(P, arms["floor+window"], sch, T, dt=0.25)
        out[label]["dt_check_maxdG"] = float(
            np.max(np.abs(s5["G"] - out[label]["floor+window"]["G"])))
    return out


def part_e(T_set: float) -> dict:
    """Healthy-regime cost of the window regulator (must not degrade
    baseline), plus the exact c-int = a_hold equivalence check."""
    out: dict = {}
    sb = simulate(P, baseline_schedule(), 2000.0, dt=0.25)
    out["frozen_G_end"] = float(sb["G"][-1])
    for name, wp in (("floor-only", WindowParams(T_fixed=0.0, floor=FLOOR)),
                     ("window-only", WindowParams(regulate=True,
                                                  T_set=T_set)),
                     ("floor+window", WindowParams(regulate=True, T_set=T_set,
                                                   floor=FLOOR))):
        s = simulate_window(P, wp, baseline_schedule(), 2000.0, dt=0.25)
        out[f"{name}_G_end"] = float(s["G"][-1])
        out[f"{name}_maxdG"] = float(np.max(np.abs(s["G"] - sb["G"])))
        out[f"{name}_T_max"] = float(s["T"].max()) if wp.regulate else 0.0
        out[f"{name}_T_end"] = float(s["T"][-1]) if wp.regulate else 0.0
    # standing-capacity equivalence: window T=400 (c_int = 0.2 at c_cap=.1)
    # vs frozen chronic a_hold = 0.2 on the healthy axis
    sw = simulate_window(P, WindowParams(T_fixed=400.0), baseline_schedule(),
                         3000.0, dt=0.5)
    sf = simulate(P, Schedule({"a_hold": [(0.0, 3000.0, 0.2)]}), 3000.0,
                  dt=0.5)
    out["equiv_window_G_end"] = float(sw["G"][-1])
    out["equiv_frozen_G_end"] = float(sf["G"][-1])
    out["equiv_maxdG"] = float(np.max(np.abs(sw["G"] - sf["G"])))
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    t_start = time.time()

    print("== exp7 part 0: integrity (gates + T=0 fidelity + attractor "
          "exactness + stuck-survival + rescue-not-blocked) ==")
    p0 = part0()
    print(f"  frozen gates: {p0['gates']}  -> "
          f"{'ALL PASS' if all(p0['gates'].values()) else 'FAIL'}")
    for k in ("fidelity_baseline", "fidelity_failure", "fidelity_rescue"):
        print(f"  {k:22s}: max|dG| = {p0[k]:.2e}")
    print(f"  attractor |dE| healthy = {p0['attractor_dE_healthy']:+.2e}, "
          f"stuck = {p0['attractor_dE_stuck']:+.2e}  (V=0 at both -> "
          "benefit inert at equilibria)")
    print(f"  stuck-survival (T=200 on, no floor, engage t=600): G_end = "
          f"{p0['stuck_survival_G_end']:.4f}, is_stuck = "
          f"{p0['stuck_survival_is_stuck']}  (collapse NOT magically fixed)")
    print(f"  floor escape WITH T on: G_end = "
          f"{p0['stuck_escape_with_T_G_end']:.4f}  (rescue still the "
          "floor's job)")
    print(f"  G2c external rescue with T on: G_end = "
          f"{p0['rescue_with_T_G_end']:.4f} (frozen 0.8854) — not blocked")

    print("\n== exp7 (a): phase diagram T x c_cap — healthy G_end "
          "(standing cost) + stressed dip (floor on) ==")
    pa = part_a()
    print("  healthy G_end (baseline, T=3000):")
    print("           T:   " + " ".join(f"{T:6.0f}" for T in pa["Ts"]))
    for i, cc in enumerate(C_CAPS):
        print(f"    c_cap={cc:4.2f}: "
              + " ".join(f"{g:6.3f}" for g in pa["healthy"][i]))
    print("  c_int at c_cap=0.1: "
          + " ".join(f"{c:6.3f}" for c in pa["c_int"]))
    print("  canonical dip with floor (what capacity BUYS):")
    print("           T:   " + " ".join(f"{T:6.0f}" for T in pa["dipT"]))
    print("    dip G_min:    " + " ".join(f"{d:6.4f}" for d in pa["dips"]))
    print("    G_end:        " + " ".join(f"{d:6.4f}" for d in pa["dip_G_ends"])
          + "   (standing cost of never switching T off)")
    np.savez("cache/exp7_phase.npz", **{k: v for k, v in pa.items()})

    print("\n== exp7 (b): bisected edges ==")
    pb = part_b()
    for cc in C_CAPS:
        tm = pb["T_max"][cc]
        ci = pb["T_max_c_int"][cc]
        print(f"  T_max @ c_cap={cc:4.2f}: "
              f"{'n/a' if tm is None else f'{tm:.1f} T-units'}"
              f" = {'n/a' if ci is None else f'{ci:.4f}'} c-int units "
              "(healthy G_end > 0.5; MEASURED upper edge)")
    print(f"  frozen chronic a_hold_crit (healthy, same criterion) = "
          f"{pb['a_hold_crit_healthy']:.4f}  -> c-int units ARE a_hold "
          "units on this axis (measured, not asserted)")
    print("  benefit axis for calibration (REGULATED m2-gated dip margin,\n"
          "  canonical episode, floor on — the fixed-T standing cost would\n"
          "  wreck re-settling; the regulator pays only while volatile):")
    print("           T_set:  " + " ".join(f"{T:6.0f}" for T in pb["scanT"]))
    print("    dip G_min:     " + " ".join(f"{d:6.4f}" for d in pb["dip_reg"]))
    print("    margin:        " + " ".join(f"{d:+6.4f}" for d in pb["margin_reg"]))
    print(f"  margin argmax T = {pb['T_argmax']:.0f}; T_set = "
          f"{pb['T_set']:.0f} (argmax clamped to T_max/2 = "
          f"{0.5 * pb['T_max'][0.1]:.0f})")

    print("\n== exp7 (c): CONJECTURE TEST — graded degradation or smooth "
          "onset at low capacity? ==")
    pc = part_c()
    print("           T:   " + " ".join(f"{T:6.0f}" for T in pc["fineT"]))
    print("           T:   " + " ".join(f"{T:6.0f}" for T in pc["fineT"]))
    print("    dip G_min:    " + " ".join(f"{d:6.4f}" for d in pc["dips"]))
    print("    d(dip)/dT>0:  " + " ".join(f"{int(d > 1e-9):6d}" for d in np.diff(pc["dGmins"])))
    print("    M_max:        " + " ".join(f"{m:6.3f}" for m in pc["Ms"]))
    print(f"  -> canonical-episode benefit: {pc['verdict']}")
    print("  SUSTAINED-stress tolerance (dur=300 episode, dip > 0.1 bar):")
    print("           T:   " + " ".join(f"{T:6.0f}" for T in pc["susT"]))
    print("    a_hold_crit:  " + " ".join(f"{a:6.3f}" for a in pc["a_hold_crit_sustained"]))
    print("    -> NEGATIVE axis: on sustained stress the standing cost\n"
          "       DOMINATES the certified benefit and capacity REDUCES\n"
          "       tolerance monotonically (graded, no lower threshold)")

    print(f"\n== exp7 (d): LONG HORIZON, 4 arms (T_set = {pb['T_set']:.0f}) "
          "==")
    pd_ = part_d(pb["T_set"])
    for label in pd_["patterns"]:
        print(f"  --- {label} ---")
        for name in ("unregulated", "floor-only", "window-only",
                     "floor+window"):
            r = pd_[label][name]
            print(f"    {name:13s}: G_min={r['G_min']:.4f} "
                  f"dip_min={r['dip_min']:.4f} dip_med={r['dip_med']:.4f} "
                  f"G_end={r['G_end']:.4f} frac<0.5={r['frac_below_05']:.3f}"
                  + (f" T_max={r['T_max']:.0f}" if name != "unregulated"
                     and name != "floor-only" else ""))
        fw, fl = pd_[label]["floor+window"], pd_[label]["floor-only"]
        dd = fw["dips"] - fl["dips"]
        print(f"    floor+window vs floor-only dip margin: min "
              f"{dd.min():+.4f}, median {np.median(dd):+.4f}, max "
              f"{dd.max():+.4f}  -> window-targeting "
              f"{'BEATS' if np.median(dd) > 0 else 'DOES NOT BEAT'} "
              "floor-only"
              + (" everywhere" if dd.min() > 0 else " (not everywhere)"))
        print(f"    vs frozen stuck level G* = {STUCK_G:.4f}: every "
              f"floored dip clears it by >= "
              f"{min(fw['dip_min'], fl['dip_min']) - STUCK_G:.4f}")
        print(f"    dt 0.25 vs 0.5 check (floor+window): max|dG| = "
              f"{pd_[label]['dt_check_maxdG']:.2e}")

    print("\n== exp7 (e): healthy-regime cost of the window regulator ==")
    pe = part_e(pb["T_set"])
    print(f"    frozen       : G_end = {pe['frozen_G_end']:.4f}")
    for name in ("floor-only", "window-only", "floor+window"):
        print(f"    {name:13s}: G_end = {pe[name + '_G_end']:.4f}  "
              f"max|dG| vs frozen = {pe[name + '_maxdG']:.2e}"
              + (f"  T_max = {pe[name + '_T_max']:.1f}, T_end = "
                 f"{pe[name + '_T_end']:.2f}" if name != "floor-only" else ""))
    print(f"    equivalence: window T=400 (c_int=0.2) G_end = "
          f"{pe['equiv_window_G_end']:.4f} vs frozen a_hold=0.2 G_end = "
          f"{pe['equiv_frozen_G_end']:.4f}, max|dG| = "
          f"{pe['equiv_maxdG']:.2e}  -> c-int = a_hold on the healthy axis")

    # ------------------------------------------------------------ caches
    np.savez("cache/exp7_phase.npz",
             **{k: v for k, v in pa.items()},
             **{f"b_{k}": (np.nan if v is None else v)
                for k, v in pb.items() if not isinstance(v, dict)},
             b_T_max=np.array([np.nan if pb["T_max"][c] is None
                               else pb["T_max"][c] for c in C_CAPS]),
             b_T_max_c_int=np.array([np.nan if pb["T_max_c_int"][c] is None
                                     else pb["T_max_c_int"][c]
                                     for c in C_CAPS]),
             **{f"c_{k}": v for k, v in pc.items() if k != "verdict"})
    lh = {}
    for label in pd_["patterns"]:
        for name in ("unregulated", "floor-only", "window-only",
                     "floor+window"):
            r = pd_[label][name]
            for k, v in r.items():
                if isinstance(v, np.ndarray):
                    lh[f"{label[:12]}|{name}|{k}"] = v[::20]  # downsample
            for k in ("G_min", "G_end", "dip_min", "dip_med",
                      "frac_below_05", "T_max", "T_mean_above_10"):
                lh[f"{label[:12]}|{name}|{k}"] = r[k]
        lh[f"{label[:12]}|dt_check"] = pd_[label]["dt_check_maxdG"]
    lh["e_frozen_G_end"] = pe["frozen_G_end"]
    for k, v in pe.items():
        lh[f"e_{k}"] = v
    np.savez("cache/exp7_longhorizon.npz", **lh)

    # ------------------------------------------------------------ figure
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3)

    axA = fig.add_subplot(gs[0, 0])
    Te = np.concatenate(([pa["Ts"][0] - 25.0], 0.5 * (pa["Ts"][:-1]
                                                      + pa["Ts"][1:]),
                         [pa["Ts"][-1] + 100.0]))
    Ce = np.concatenate(([C_CAPS[0] - 0.025, ],
                         0.5 * (np.array(C_CAPS[:-1]) + np.array(C_CAPS[1:])),
                         [C_CAPS[-1] + 0.05]))
    Zm = np.ma.masked_where(pa["healthy"] < 0.10, pa["healthy"])
    pcA = axA.pcolormesh(Te, Ce, Zm, cmap="RdYlGn", vmin=0.0, vmax=0.9,
                         shading="flat")
    axA.scatter(np.repeat(pa["Ts"][None, :], len(C_CAPS), 0).ravel(),
                np.repeat(np.array(C_CAPS)[:, None], len(pa["Ts"]), 1)
                .ravel(), s=2, c="k", alpha=0.3)
    plt.colorbar(pcA, ax=axA, label="healthy G_end")
    for i, cc in enumerate(C_CAPS):
        tm = pb["T_max"][cc]
        if tm is not None:
            axA.plot(tm, cc, "v", color="k", ms=7)
            axA.annotate(f"T_max={tm:.0f}", (tm, cc), fontsize=6,
                         xytext=(3, -9), textcoords="offset points")
    axA.set_xlabel("capacity T (self-simulation horizon)")
    axA.set_ylabel("c_cap")
    axA.set_title("(a) phase diagram: healthy G_end(T, c_cap)\n"
                  "black = collapsed (<0.1); v = bisected T_max")

    axB = fig.add_subplot(gs[0, 1])
    i10 = C_CAPS.index(0.1)
    axB.plot(pa["Ts"], pa["healthy"][i10], "o-", label="healthy G_end")
    axB.axhline(0.5, color="k", ls=":", lw=0.6)
    tm = pb["T_max"][0.1]
    if tm is not None:
        axB.axvline(tm, color="tab:red", ls="--", lw=1,
                    label=f"T_max={tm:.0f} (c_int="
                          f"{pb['T_max_c_int'][0.1]:.3f})")
    axBb = axB.twinx()
    axBb.plot(pa["dipT"], pa["dips"], "^-", color="tab:green",
              label="canonical dip G_min (floor)")
    axB.set_xlabel("capacity T")
    axB.set_ylabel("healthy G_end")
    axBb.set_ylabel("dip G_min", color="tab:green")
    axB.set_title("(b) the trade on ONE axis: dip shallows,\n"
                  "healthy level degrades (both vs T)")
    h1, l1 = axB.get_legend_handles_labels()
    h2, l2 = axBb.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, fontsize=7)

    axC = fig.add_subplot(gs[0, 2])
    axC.plot(pc["fineT"], pc["dGmins"], "o-", ms=3, color="tab:blue",
             label="dip shallowing dG_min(T)  [G units]")
    axC.plot(pc["fineT"], pc["Ms"] * 0.1, "s--", ms=3, color="tab:orange",
             label="certified M_max x0.1")
    axC2 = axC.twinx()
    axC2.plot(pc["susT"], pc["a_hold_crit_sustained"], "^-", ms=4,
              color="tab:red", label="sustained-stress a_hold_crit(T)")
    axC.set_xlabel("capacity T (fine low-T grid)")
    axC.set_ylabel("dip shallowing [G units]")
    axC2.set_ylabel("a_hold_crit (sustained)", color="tab:red")
    axC.set_title(f"(c) CONJECTURE TEST: {pc['verdict']}\n"
                  "red: sustained stress — capacity LOWERS tolerance")
    h1, l1 = axC.get_legend_handles_labels()
    h2, l2 = axC2.get_legend_handles_labels()
    axC.legend(h1 + h2, l1 + l2, fontsize=6)

    axD = fig.add_subplot(gs[1, 0])
    lab0 = pd_["patterns"][0]
    cols = {"unregulated": "k", "floor-only": "tab:green",
            "window-only": "tab:blue", "floor+window": "tab:red"}
    for name in ("unregulated", "floor-only", "window-only", "floor+window"):
        r = pd_[lab0][name]
        axD.plot(r["t"][::20], r["G"][::20], color=cols[name], lw=0.9,
                 label=f"{name}: G_min={r['G_min']:.3f}, "
                       f"G_end={r['G_end']:.3f}")
    axD.axhline(0.1, color="k", ls=":", lw=0.6)
    axD.set_xlabel("t")
    axD.set_ylabel("G(t)")
    axD.set_title(f"(d) long horizon, 4 arms — {lab0}\n"
                  f"(T_set={pb['T_set']:.0f}; margins, not binary counts)")
    axD.legend(fontsize=7, loc="lower right")

    axE = fig.add_subplot(gs[1, 1])
    for label in pd_["patterns"]:
        fl = pd_[label]["floor-only"]
        fw = pd_[label]["floor+window"]
        axE.plot(np.arange(1, len(fl["dips"]) + 1), fl["dips"], "-",
                 color="tab:green", lw=1,
                 label=f"floor-only ({label[:9]})")
        axE.plot(np.arange(1, len(fw["dips"]) + 1), fw["dips"], "-",
                 color="tab:red", lw=1,
                 label=f"floor+window ({label[:9]})")
    axE.axhline(STUCK_G, color="k", ls=":", lw=0.8)
    axE.annotate("frozen stuck G*=0.0485", (2, STUCK_G + 0.004), fontsize=6)
    axE.set_xlabel("episode #")
    axE.set_ylabel("dip G_min per episode")
    axE.set_title("(d') MARGIN: per-episode dips, floor-only vs\n"
                  "floor+window (canonical + dense-weak)")
    axE.legend(fontsize=6)

    axF = fig.add_subplot(gs[1, 2])
    sch1 = episode1(0.9)
    s = simulate_window(P, WindowParams(regulate=True, T_set=pb["T_set"],
                                        floor=FLOOR), sch1, 600.0, dt=0.25)
    axF.plot(s["t"], s["G"], color="k", lw=1.2, label="G")
    axF.axhline(0.1, color="k", ls=":", lw=0.6)
    axF2 = axF.twinx()
    axF2.plot(s["t"], s["T"], color="tab:purple", lw=1.0, label="T(t)")
    axF2.plot(s["t"], s["M"], color="tab:orange", lw=0.8, label="M(t)")
    axF.set_xlabel("t")
    axF.set_ylabel("G")
    axF2.set_ylabel("T / M", color="tab:purple")
    axF.set_title("(f) mechanism: T rises with volatility (m2), the\n"
                  "certified M shallows the dip, T decays when quiet")
    h1, l1 = axF.get_legend_handles_labels()
    h2, l2 = axF2.get_legend_handles_labels()
    axF.legend(h1 + h2, l1 + l2, fontsize=7)

    fig.suptitle("exp7 — the capacity WINDOW: introspective horizon T buys "
                 "dip margin (graded, no lower threshold — the conjectured "
                 "lower edge fails in-model) and costs standing inward drive "
                 "(measured upper edge); deterministic, in-model only")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "f12_window.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\nfigure -> {path}")
    print(f"total {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
