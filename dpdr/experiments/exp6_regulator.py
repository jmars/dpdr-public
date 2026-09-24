"""exp6 — the self-rescue regulator: cheap floor vs elaborate introspection.

The deliverable experiment for dpdr/regulator.py (opt-in; the frozen model is
untouched).  The design principle under test (measured beforehand on the
frozen dynamics, handoff-selfreg-floor): ONE CHEAP CONSTANT — a floor
clamping the cannibalization switch's effective threshold from below —
severs the entire positive-feedback loop that sustains the stuck attractor
(floor -> Theta_eff >= E -> c = 0 -> attention no longer captured, a -> 0,
and G no longer eaten -> G regrows).  The floor need not track the real
internal state (post-collapse the true Theta_eff is ~0.12; any floor in
[0.49, 1.2+] works identically), so it is FALSE as a belief and CHEAP as a
mechanism.  The contrast: self-monitoring is itself inward attention, and
sustained expensive introspection sustains the collapse instead of fixing
it.  SUCCESS CRITERION THROUGHOUT IS FUNCTIONAL (does the agent stay
functional), not phenomenological match.

SCOPE OF THE CHEAP-VS-ELABORATE CONTRAST (review finding 1): the sharp
gated monitoring-cost threshold c_mon_crit ~ 0.5 exists ONLY in the
POST-COLLAPSE SETTLED assay (engage at t = 600, after the stuck attractor
has consolidated).  In the DEPLOYED configuration — always armed,
t_engage = 0 — the trigger gate fires early in the descent (G crosses
G_trig = 0.3 at t ~ 121, while a is still rising) so the actuator releases
attention before capture consolidates and the gated threshold VANISHES:
k_pull = 1.0 escapes at every gated c_mon in [0, 1.5] (G = 0.8855).  Part
(d) measures both geometries and reports this explicitly.  What survives
in deployment: the floor's structural zero cost, and the UNGATED
(always-on vigilance) axis, which is damaging in BOTH geometries.

Battery:

  (0) Integrity: frozen gates G1-G2c still pass; the regulator-OFF wrap
      reproduces dpdr.integrate.simulate exactly (max |dG|) on the three
      standard scenarios; the floor is INERT in the healthy baseline
      (max |dG| = 0 to tolerance) and does not interfere with the external
      rescue (G2c).

  (a) WORKING WINDOW: post-collapse escape over the (floor x k_pull x
      c_mon) grid.  Escape = functional at the horizon (G(T) > 0.5).

  (b) Thresholds (bisected): floor_crit (below it no escape; compare the
      stuck point's own error level E* ~ 0.497), k_crit(c_mon) in raw and
      u_ext-equivalent units (k_raw/k_ext; ASSERTED conversion, not
      derived — see part (b) note), c_mon_crit(k).

  (c) ANY FLOOR: G_end and escape TIME vs floor value — a sharp step at
      floor ~ 0.48 and flat 0.8855 everywhere above it (insensitive to the
      VALUE, sensitive to EXISTENCE), in both floor modes ('theta' and
      'S'), under three different collapse loads (canonical episode,
      sustained affect, chronic low-grade inward attention).

  (d) HEADLINE CONTRAST, both geometries.  POST-COLLAPSE (engage t=600,
      rescue of an ALREADY-STUCK agent): CHEAP (floor only, zero
      monitoring) escapes in ~22 t.u. while ELABORATE (actuator + gated
      c_mon >= 0.7) fails.  ALWAYS-ARMED (engage t=0, deployment): the
      gated threshold vanishes — an elaborate gated introspector escapes
      at every c_mon up to 1.5, so the floor's edge there is cost-free-
      ness, not beating a gated introspector; the UNGATED always-on
      vigilance cost still defeats even the FLOOR at c_mon ~ 0.2
      (post-collapse) — the knowing-floor result kc ~ 0.2, reproduced
      here as the 6th negative.  Plus the elaborateness axis in the
      HEALTHY regime: a GATED cost is free at any size (the trigger never
      fires), an UNGATED cost degrades G monotonically and c_mon >= ~0.8
      collapses even the healthy agent with the floor on.

  (e) LONG HORIZON: repeated stress episodes (N = 60-100).  Canonical
      intensity: the frozen agent collapses in episode 1 and stays stuck;
      the floor-regulated agent dips to G ~ 0.13 and re-settles to 0.885
      between episodes for the whole horizon.  The reported quantity is
      the MARGIN (dip depth vs stress intensity: 0.218 -> 0.103 over
      a_hold 0.4-2.0, always above the stuck attractor's 0.049), not the
      binary episode count — the floor forces c = 0 during every dip, so
      the c > 0.5 leg of metrics.is_stuck is structurally unreachable on
      a floor-regulated run and the criterion is G-based only.
      Generalization: (i) denser sub-threshold episodes (dur 60, gap 200;
      one episode alone does not collapse) still collapse the frozen agent
      (episode 2) while the regulated one stays functional; (ii) spaced
      sub-threshold episodes (dur 40, gap 300 x 100) collapse NEITHER — and
      the floor is exactly inert there (identical trajectory), i.e. no
      false-positive cost; (iii) intensity sweep: regulated vs frozen
      single-episode failure rate; (iv) permanent capture (a_hold held
      forever): the floor's protection ceiling — regulated G crosses
      below 0.1 at a_hold ~ 1.5 (cliff between episodic and permanent
      stress) but never reaches the frozen stuck level.

  (f) MECHANISM trace of the floor escape (a, c, G, Theta_eff frozen vs
      regulated) for the figure.

MODEL-LEVEL CAVEATS (stated, not hidden): deterministic model — every
threshold is a locus, not a distribution; demonstrated IN-MODEL only; the
"false belief" is a clamped constant, not a belief system; the monitoring
cost is a scalar inward-drive addend standing in for whatever real
introspection costs; k_pull/c_mon units are model units (the u_ext
equivalent k_pull/k_ext is an ASSERTED scale equivalence — a G-dependent
sigmoid-gated pull vs a constant external demand — not a derived
conversion).  Honest negatives found by this battery are printed with the
results and recorded in predictions.md.

Outputs: figs/f11_regulator.png, cache/exp6_*.npz.
Reproducible: `.venv/bin/python -m experiments.exp6_regulator`.
"""
from __future__ import annotations

import os
import time

import numpy as np

from dpdr.events import baseline_schedule, failure_schedule, rescue_schedule
from dpdr.integrate import simulate
from dpdr.metrics import check_gates, is_stuck
from dpdr.model import Params, Schedule
from dpdr.regulator import RegulatorParams, simulate_reg

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "figs")

P = Params()
OFF = RegulatorParams(floor=None, k_pull=0.0, c_mon=0.0)   # = frozen RHS
FLOOR = RegulatorParams(floor=0.7)                          # the cheap design
# post-collapse assay geometry: engage after the stuck attractor has settled
T_ENGAGE = 600.0
T_ASSAY = 1500.0

# long-horizon stress patterns: (label, n, gap, dur, a_hold)
PATTERNS = [
    ("canonical  x60  gap300", 60, 300.0, 100.0, 0.9),
    ("dense-weak x100 gap200", 100, 200.0, 60.0, 0.9),
    ("spaced-weak x100 gap300", 100, 300.0, 40.0, 0.9),
]
EP_T0 = 100.0


def snap(x: float) -> float:
    return round(round(x / 0.05) * 0.05, 10)


def episodes_schedule(n: int, gap: float, dur: float, ah: float,
                      pulse: float = 0.5) -> Schedule:
    ch: dict = {"a_hold": [], "A": []}
    for k in range(n):
        t0e = snap(EP_T0 + k * gap)
        ch["a_hold"].append((t0e, snap(t0e + dur), ah))
        ch["A"].append((t0e, snap(t0e + min(60.0, dur)), pulse))
    return Schedule(ch)


def G_end(r: RegulatorParams, sch: Schedule, T: float = T_ASSAY) -> float:
    return float(simulate_reg(P, r, sch, T)["G"][-1])


def escaped(r: RegulatorParams, sch: Schedule, T: float = T_ASSAY) -> bool:
    return G_end(r, sch, T) > 0.5


def bisect(f, lo, hi, it=12):
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
    """Frozen gates + wrap fidelity + healthy/inertness checks."""
    out: dict = {"gates": check_gates(P)}
    for name, sch, T in (("baseline", baseline_schedule(), 1000.0),
                         ("failure", failure_schedule(), 600.0),
                         ("rescue", rescue_schedule(), 1400.0)):
        sf = simulate(P, sch, T)
        sr = simulate_reg(P, OFF, sch, T)
        out[f"fidelity_{name}"] = float(
            np.max(np.abs(sf["G"] - sr["G"])))
    # floor inert in the healthy regime (no episode at all)
    sb = simulate(P, baseline_schedule(), 1000.0)
    sbr = simulate_reg(P, FLOOR, baseline_schedule(), 1000.0)
    out["healthy_inert_maxdG"] = float(
        np.max(np.abs(sb["G"] - sbr["G"])))
    out["healthy_c_max"] = float(sbr["c"].max())
    # floor does not block the external rescue (G2c scenario)
    sr = simulate_reg(P, FLOOR, rescue_schedule(), 1400.0)
    out["rescue_with_floor_G"] = float(sr["G"][-1])
    return out


def part_a() -> dict:
    """Working window: floor x k_pull x c_mon, post-collapse."""
    floors = [None, 0.0, 0.3, 0.48, 0.5, 0.7, 0.9, 1.2]
    ks = [0.0, 0.5, 1.25, 2.0, 4.0]
    cms = [0.0, 0.2, 0.5, 0.8, 1.0, 1.5]
    fail = failure_schedule()
    G = np.zeros((len(floors), len(ks), len(cms)))
    for i, fl in enumerate(floors):
        for j, k in enumerate(ks):
            for m, cm in enumerate(cms):
                r = RegulatorParams(floor=fl, k_pull=k, c_mon=cm,
                                    t_engage=T_ENGAGE)
                G[i, j, m] = G_end(r, fail)
    return dict(floors=[-1.0 if f is None else f for f in floors],
                floors_is_none=[f is None for f in floors],
                ks=np.array(ks), cms=np.array(cms), G=G,
                escape=G > 0.5)


def part_b() -> dict:
    """Bisected thresholds."""
    fail = failure_schedule()
    out = {}
    # floor_crit (theta mode and S mode)
    out["floor_crit_theta"] = bisect(
        lambda v: escaped(RegulatorParams(floor=v, t_engage=T_ENGAGE), fail),
        0.40, 0.50)
    out["floor_crit_S"] = bisect(
        lambda v: escaped(RegulatorParams(floor=v, floor_mode="S",
                                          t_engage=T_ENGAGE), fail),
        0.50, 0.65)
    # k_crit at several gated monitoring costs.  NOTE: the u_ext-equivalent
    # k_raw/k_ext is an ASSERTED scale equivalence (a G-dependent sigmoid-
    # gated pull vs the frozen channel's CONSTANT external demand), kept for
    # magnitude comparison only — not a derived conversion, and not a
    # reproduction of the earlier probe's k units (they differ ~2x).
    kc, kcu = {}, {}
    for cm in (0.0, 0.2, 0.5, 0.8):
        kc[cm] = bisect(
            lambda k: escaped(RegulatorParams(floor=None, k_pull=k,
                                              c_mon=cm, t_engage=T_ENGAGE),
                              fail),
            0.5, 2.0 if cm < 0.6 else 4.0)
        kcu[cm] = None if kc[cm] is None else kc[cm] / P.k_ext
    out["k_crit"], out["k_crit_uext"] = kc, kcu
    # c_mon_crit at several authorities (gated and ungated)
    for k in (2.0, 4.0):
        out[f"cmon_crit_gated_k{k:g}"] = bisect(
            lambda c: escaped(RegulatorParams(floor=None, k_pull=k,
                                              c_mon=c, t_engage=T_ENGAGE),
                              fail),
            0.5, 0.7)
        out[f"cmon_crit_ungated_k{k:g}"] = bisect(
            lambda c: escaped(RegulatorParams(floor=None, k_pull=k, c_mon=c,
                                              mon_gated=False,
                                              t_engage=T_ENGAGE), fail),
            0.0, 1.0)
    # stuck point's own error level, for the floor_crit comparison
    s = simulate_reg(P, OFF, fail, T_ASSAY)
    out["stuck_E"] = float(s["E"][-1])
    out["stuck_Theta_eff_frozen"] = float(s["Theta_eff_frozen"][-1])
    return out


def part_c() -> dict:
    """ANY floor: value sweep + escape time, both modes, three loads."""
    fail = failure_schedule()
    loads = {
        "canonical": (failure_schedule(), 1500.0),
        "sustained-A": (Schedule({"a_hold": [(100.0, 200.0, 0.9)],
                                  "A": [(100.0, 1100.0, 0.5)]}), 1600.0),
        "chronic-ah": (Schedule({"a_hold": [(100.0, 1500.0, 0.35)]}),
                       2600.0),
    }
    vals = np.round(np.arange(0.0, 1.31, 0.1), 2)
    out = dict(vals=vals)
    for mode in ("theta", "S"):
        Ge, ae = np.zeros(len(vals)), np.zeros(len(vals))
        for i, v in enumerate(vals):
            s = simulate_reg(P, RegulatorParams(floor=float(v),
                                                floor_mode=mode,
                                                t_engage=T_ENGAGE),
                             fail, T_ASSAY)
            Ge[i], ae[i] = s["G"][-1], s["a"][-1]
        out[f"G_end_{mode}"], out[f"a_end_{mode}"] = Ge, ae
    # escape time vs value (theta mode, canonical load)
    tesc = []
    for v in (0.44, 0.46, 0.48, 0.49, 0.50, 0.55, 0.60, 0.70, 0.90, 1.20):
        s = simulate_reg(P, RegulatorParams(floor=v, t_engage=T_ENGAGE),
                         fail, 4000.0)
        idx = np.where((s["t"] > T_ENGAGE) & (s["G"] > 0.5))[0]
        tesc.append((v, np.nan if idx.size == 0
                     else float(s["t"][idx[0]] - T_ENGAGE)))
    out["escape_times"] = np.array(tesc)
    # generalization across loads at the design floor 0.7 and at 0.5
    gen = {}
    for name, (sch, T) in loads.items():
        for fl in (0.5, 0.7):
            s = simulate_reg(P, RegulatorParams(floor=fl,
                                                t_engage=T_ENGAGE), sch, T)
            gen[(name, fl)] = float(s["G"][-1])
    out["loads"] = {f"{k[0]}|floor={k[1]}": v for k, v in gen.items()}
    return out


def part_d() -> dict:
    """Headline contrast: cheap floor vs elaborate introspection, in BOTH
    geometries — post-collapse rescue (t_engage=600, the settled-stuck
    assay) and always-armed deployment (t_engage=0, where the gated
    threshold vanishes)."""
    fail = failure_schedule()
    arms = {
        "frozen (no regulator)": (OFF, T_ASSAY),
        "CHEAP floor=0.7 (no monitoring)":
            (RegulatorParams(floor=0.7, t_engage=T_ENGAGE), T_ASSAY),
        "actuator only k=2, free check":
            (RegulatorParams(floor=None, k_pull=2.0, t_engage=T_ENGAGE),
             T_ASSAY),
        "ELABORATE k=2, c_mon=0.8 gated":
            (RegulatorParams(floor=None, k_pull=2.0, c_mon=0.8,
                             t_engage=T_ENGAGE), T_ASSAY),
        "ELABORATE k=2, c_mon=1.5 gated":
            (RegulatorParams(floor=None, k_pull=2.0, c_mon=1.5,
                             t_engage=T_ENGAGE), T_ASSAY),
    }
    trajs, summary = {}, {}
    for name, (r, T) in arms.items():
        s = simulate_reg(P, r, fail, T)
        trajs[name] = (s["t"].copy(), s["G"].copy())
        summary[name] = (float(s["G"][-1]), float(s["a"][-1]))
    # ALWAYS-ARMED (deployment) geometry: same arms, t_engage=0.  The gated
    # monitoring cost can never kill the rescue here — the gate fires during
    # the descent (G crosses G_trig at t~121) before capture consolidates.
    armed = {}
    for name, r in (
        ("frozen", OFF),
        ("floor=0.7", RegulatorParams(floor=0.7)),
        ("k=1 c=0.0", RegulatorParams(floor=None, k_pull=1.0)),
        ("k=1 c=0.5", RegulatorParams(floor=None, k_pull=1.0, c_mon=0.5)),
        ("k=1 c=1.5", RegulatorParams(floor=None, k_pull=1.0, c_mon=1.5)),
        ("k=2 c=0.8", RegulatorParams(floor=None, k_pull=2.0, c_mon=0.8)),
        ("k=2 c=1.5", RegulatorParams(floor=None, k_pull=2.0, c_mon=1.5)),
    ):
        s = simulate_reg(P, r, fail, T_ASSAY)
        armed[name] = (float(s["G"][-1]), float(s["G"].min()))
    # 6th negative: floor held + UNGATED always-on cost, post-collapse —
    # the knowing-floor configuration; the floor is defeated by continuous
    # re-examination at c_mon ~ 0.2.
    cms_u = np.array([0.0, 0.1, 0.2, 0.3, 0.5])
    post_ung = np.zeros(len(cms_u))
    for i, cm in enumerate(cms_u):
        post_ung[i] = G_end(RegulatorParams(floor=0.7, c_mon=cm,
                                            mon_gated=False,
                                            t_engage=T_ENGAGE), fail)
    # elaborateness axis in the HEALTHY regime (floor on throughout)
    cms = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2])
    h_ung, h_gated = np.zeros(len(cms)), np.zeros(len(cms))
    for i, cm in enumerate(cms):
        h_ung[i] = G_end(RegulatorParams(floor=0.7, c_mon=cm,
                                         mon_gated=False), baseline_schedule(),
                         3000.0)
        h_gated[i] = G_end(RegulatorParams(floor=0.7, c_mon=cm,
                                           mon_gated=True),
                           baseline_schedule(), 3000.0)
    return dict(trajs=trajs, summary=summary, armed=armed, cms=cms,
                cms_ungated_post=cms_u, post_ungated=post_ung,
                healthy_ungated=h_ung, healthy_gated=h_gated)


def part_e() -> dict:
    """Long horizon: repeated episodes, three patterns + intensity sweep
    (with dip MARGIN) + permanent-capture ceiling."""
    out: dict = {}
    for label, n, gap, dur, ah in PATTERNS:
        sch = episodes_schedule(n, gap, dur, ah)
        T = snap(EP_T0 + n * gap + 300.0)
        sF = simulate_reg(P, OFF, sch, T, dt=0.25)
        sR = simulate_reg(P, FLOOR, sch, T, dt=0.25)
        def first_fail(s):
            idx = np.where(s["G"] < 0.1)[0]
            return None if idx.size == 0 else (
                float(s["t"][idx[0]]),
                int((s["t"][idx[0]] - EP_T0) // gap) + 1)
        out[label] = dict(
            n=n, T=T,
            frozen=dict(G_min=float(sF["G"].min()), G_end=float(sF["G"][-1]),
                        first_fail=first_fail(sF),
                        frac_below_05=float((sF["G"] < 0.5).mean()),
                        stuck=is_stuck(sF, P), t=sF["t"], G=sF["G"]),
            reg=dict(G_min=float(sR["G"].min()), G_end=float(sR["G"][-1]),
                     first_fail=first_fail(sR),
                     frac_below_05=float((sR["G"] < 0.5).mean()),
                     stuck=is_stuck(sR, P),
                     max_dG_vs_frozen=float(
                         np.max(np.abs(sR["G"] - sF["G"]))),
                     n_below_01=int((sR["G"] < 0.1).sum()), t=sR["t"],
                     G=sR["G"]),
            schedule=sch)
    # single-episode intensity sweep: failure rate + dip MARGIN (review
    # finding 5) — the informative quantity is how deep each regulated dip
    # bottoms vs the frozen stuck level (0.049), not the binary count; the
    # sweep is extended past a_hold = 1.1 to locate the protection ceiling.
    ihs = np.array([0.4, 0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
                    1.5, 1.6, 1.8, 2.0])
    failF, failR, dipsF, dipsR = [], [], [], []
    for ah in ihs:
        sch = episodes_schedule(1, 400.0, 100.0, float(ah))
        T = snap(EP_T0 + 700.0)
        sF = simulate_reg(P, OFF, sch, T, dt=0.25)
        sR = simulate_reg(P, FLOOR, sch, T, dt=0.25)
        failF.append(bool(sF["G"].min() < 0.1 and sF["G"][-1] < 0.5))
        failR.append(bool(sR["G"].min() < 0.1 and sR["G"][-1] < 0.5))
        dipsF.append(float(sF["G"].min()))
        dipsR.append(float(sR["G"].min()))
    out["intensity"] = dict(ihs=ihs, failF=np.array(failF),
                            failR=np.array(failR), dipsF=np.array(dipsF),
                            dipsR=np.array(dipsR))
    # permanent capture (a_hold never released): the floor's ceiling — the
    # episodic-stress protection does not extend to permanent capture
    # (regulated G crosses below 0.1 at a_hold ~ 1.5 but never reaches the
    # frozen stuck level 0.049).
    pc_ah = np.array([0.35, 0.5, 0.9, 1.5, 2.0])
    pc_R, pc_F = [], []
    for ah in pc_ah:
        sch = Schedule({"a_hold": [(100.0, 4000.0, float(ah))]})
        pc_R.append(float(simulate_reg(P, FLOOR, sch, 4000.0,
                                       dt=0.25)["G"][-1]))
        pc_F.append(float(simulate_reg(P, OFF, sch, 4000.0,
                                       dt=0.25)["G"][-1]))
    out["permanent_capture"] = dict(ah=pc_ah, G_end_reg=np.array(pc_R),
                                    G_end_frozen=np.array(pc_F))
    return out


def part_f() -> dict:
    """Mechanism trace of the floor escape (for the figure)."""
    fail = failure_schedule()
    r = RegulatorParams(floor=0.7, t_engage=T_ENGAGE)
    s = simulate_reg(P, r, fail, 1200.0)
    return dict(t=s["t"], G=s["G"], a=s["a"], c=s["c"],
                Theta_eff=s["Theta_eff"],
                Theta_eff_frozen=s["Theta_eff_frozen"],
                E=s["E"], S=s["S"])


# ------------------------------------------------------------------ main
def main() -> int:
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    t_start = time.time()

    print("== exp6 part 0: integrity (frozen gates + wrap fidelity) ==")
    p0 = part0()
    print(f"  frozen gates: {p0['gates']}  -> "
          f"{'ALL PASS' if all(p0['gates'].values()) else 'FAIL'}")
    for k in ("fidelity_baseline", "fidelity_failure", "fidelity_rescue"):
        print(f"  {k:22s}: max|dG| = {p0[k]:.2e}")
    print(f"  healthy baseline, floor on: max|dG| = "
          f"{p0['healthy_inert_maxdG']:.2e}, c_max = "
          f"{p0['healthy_c_max']:.3f}  (floor structurally inert)")
    print(f"  G2c external rescue with floor on: G_end = "
          f"{p0['rescue_with_floor_G']:.4f} (frozen: 0.8854) — not blocked")

    print("\n== exp6 (a): working window (floor x k_pull x c_mon), "
          "post-collapse escape G(T) ==")
    pa = part_a()
    cms, ks = pa["cms"], pa["ks"]
    floors = pa["floors"]
    for j, k in enumerate(ks):
        print(f"  k_pull={k:5.2f}: " + " ".join(f"{cm:4.1f}" for cm in cms))
        for i, fl in enumerate(floors):
            lab = "none" if pa["floors_is_none"][i] else f"{fl:4.2f}"
            print(f"    floor={lab}: "
                  + " ".join(f"{pa['G'][i, j, m]:.2f}"
                             f"{'*' if pa['escape'][i, j, m] else ' '}"
                             for m in range(len(cms))))
    np.savez("cache/exp6_window.npz", **pa)

    print("\n== exp6 (b): bisected thresholds ==")
    pb = part_b()
    print(f"  stuck point: E* = {pb['stuck_E']:.4f}, "
          f"Theta_eff_frozen* = {pb['stuck_Theta_eff_frozen']:.4f}")
    print(f"  floor_crit (theta mode) = {pb['floor_crit_theta']:.4f}"
          f"   [floor must sit at/above the stuck point's own error level]")
    print(f"  floor_crit (S mode)     = "
          f"{pb['floor_crit_S']:.4f}"
          f"   (= theta {P.Theta * pb['floor_crit_S'] / P.S_rest:.4f})"
          if pb["floor_crit_S"] is not None else
          "  floor_crit (S mode)     = n/a")
    for cm, kc in pb["k_crit"].items():
        ku = pb["k_crit_uext"][cm]
        print(f"  k_crit @ c_mon={cm:.1f} = {kc:.3f} raw"
              f" = {ku:.3f} u_ext-units (ASSERTED k/k_ext scale,"
              " not a derived conversion)"
              + ("" if kc is not None else "  (no threshold in range)"))
    for k in (2.0, 4.0):
        for g in ("gated", "ungated"):
            v = pb[f"cmon_crit_{g}_k{k:g}"]
            print(f"  c_mon_crit ({g:7s}) @ k={k:g} = "
                  f"{'n/a' if v is None else f'{v:.3f}'}")
    np.savez("cache/exp6_bisect.npz",
             **{k: (np.nan if v is None else v) if not isinstance(v, dict)
                else np.array([np.nan if x is None else x
                               for x in v.values()])
                for k, v in pb.items() if not isinstance(v, dict)},
             k_crit_keys=np.array(list(pb["k_crit"].keys()), float),
             k_crit=np.array([np.nan if v is None else v
                              for v in pb["k_crit"].values()]))

    print("\n== exp6 (c): ANY floor — value insensitivity ==")
    pc = part_c()
    print("  theta mode: " + " ".join(f"{v:.1f}" for v in pc["vals"]))
    print("  G_end     : " + " ".join(f"{g:.2f}" for g in pc["G_end_theta"]))
    print("  a_end     : " + " ".join(f"{g:.2f}" for g in pc["a_end_theta"]))
    print("  S mode    : " + " ".join(f"{g:.2f}" for g in pc["G_end_S"]))
    print("  escape time after engagement (t.u.): "
          + ", ".join(f"{v:.2f}->{t:.0f}" for v, t in pc["escape_times"]))
    for k, v in pc["loads"].items():
        print(f"  load {k:22s}: G_end = {v:.4f}")

    print("\n== exp6 (d): HEADLINE — cheap floor vs elaborate introspection ==")
    pd_ = part_d()
    print("  POST-COLLAPSE assay (engage t=600, rescuing an ALREADY-STUCK "
          "agent):")
    for name, (ge, ae) in pd_["summary"].items():
        verdict = "ESCAPED (functional)" if ge > 0.5 else \
            "FAILED (still collapsed)" if ge < 0.15 else "degraded plateau"
        print(f"    {name:36s}: G_end = {ge:.4f}  a_end = {ae:.4f}"
              f"  -> {verdict}")
    print("  ALWAYS-ARMED deployment (t_engage=0, same failure schedule): "
          "the GATED threshold VANISHES —")
    print("    the gate fires during the descent (G<G_trig at t~121, "
          "before capture consolidates), so a gated cost")
    print("    of any size cannot kill the rescue; the floor's deployment "
          "edge is zero cost, not beating an")
    print("    elaborate gated introspector (scope of the headline: "
          "post-collapse rescue, NOT deployment):")
    for name, (ge, gmin) in pd_["armed"].items():
        print(f"    {name:12s}: G_end = {ge:.4f}  G_min = {gmin:.4f}"
              f"  -> {'escaped' if ge > 0.5 else 'FAILED'}")
    print("  6th NEGATIVE — floor held + UNGATED always-on cost, "
          "post-collapse (the knowing-floor cell):")
    print("    c_mon     : " + " ".join(f"{c:5.2f}" for c in
                                        pd_["cms_ungated_post"]))
    print("    G_end     : " + " ".join(f"{g:5.2f}" for g in
                                        pd_["post_ungated"]))
    print("    -> continuous re-examination DEFEATS the floor at "
          "c_mon ~ 0.2, matching the knowing-floor kc ~ 0.2:")
    print("       the floor must not be re-checked (all part-a c_mon runs "
          "were gated; this cell was missing).")
    print("  HEALTHY-regime elaborateness (floor on, baseline schedule):")
    print("    c_mon     : " + " ".join(f"{c:5.2f}" for c in pd_["cms"]))
    print("    G  gated  : " + " ".join(f"{g:5.2f}" for g in
                                        pd_["healthy_gated"]))
    print("    G  ungated: " + " ".join(f"{g:5.2f}" for g in
                                        pd_["healthy_ungated"]))
    np.savez("cache/exp6_contrast.npz", cms=pd_["cms"],
             healthy_ungated=pd_["healthy_ungated"],
             healthy_gated=pd_["healthy_gated"],
             cms_ungated_post=pd_["cms_ungated_post"],
             post_ungated=pd_["post_ungated"],
             armed_names=np.array(list(pd_["armed"].keys())),
             armed_G_end=np.array([v[0] for v in pd_["armed"].values()]),
             armed_G_min=np.array([v[1] for v in pd_["armed"].values()]),
             **{f"traj_{i}_{n[:12]}": np.vstack(t)
                for i, (n, t) in enumerate(pd_["trajs"].items())})

    print("\n== exp6 (e): LONG HORIZON — repeated stress episodes ==")
    pe = part_e()
    for label in [p[0] for p in PATTERNS]:
        r = pe[label]
        f, g = r["frozen"], r["reg"]
        ff = ("never below 0.1" if f["first_fail"] is None else
              f"G<0.1 at t={f['first_fail'][0]:.0f} "
              f"(episode {f['first_fail'][1]})")
        gf = ("never below 0.1" if g["first_fail"] is None else
              f"G<0.1 at t={g['first_fail'][0]:.0f} "
              f"(episode {g['first_fail'][1]})")
        print(f"  {label}  (N={r['n']}):")
        print(f"    frozen    : G_min={f['G_min']:.4f} G_end={f['G_end']:.4f}"
              f" stuck={f['stuck']}  {ff}")
        print(f"    regulated: G_min={g['G_min']:.4f} G_end={g['G_end']:.4f}"
              f" stuck={g['stuck']} episodes_below_0.1={g['n_below_01']}"
              f"  {gf}")
    ii = pe["intensity"]
    print("  single-episode intensity sweep (a_hold): "
          + " ".join(f"{a:.1f}" for a in ii["ihs"]))
    print("    frozen failure?    : "
          + " ".join(f"{int(b):d}    " for b in ii["failF"]))
    print("    regulated failure? : "
          + " ".join(f"{int(b):d}    " for b in ii["failR"]))
    print("    frozen dip    G_min: "
          + " ".join(f"{v:.2f}" for v in ii["dipsF"]))
    print("    regulated dip G_min: "
          + " ".join(f"{v:.2f}" for v in ii["dipsR"]))
    print("    MARGIN (review finding 5): the floor converts the frozen "
          "0.049 attractor into a\n"
          "    0.218->0.103 dip plateau over a_hold 0.4-2.0 — every "
          "regulated dip stays\n"
          "    above 0.1 until permanent-grade capture (a_hold>=1.5 held "
          "forever).  The floor\n"
          "    forces c=0 during dips, so is_stuck's c>0.5 leg is "
          "structurally unreachable and\n"
          "    the criterion is G-based only.")
    pcpc = pe["permanent_capture"]
    print("  permanent capture (a_hold held forever, T=4000): "
          + " ".join(f"{a:.2f}" for a in pcpc["ah"]))
    print("    regulated G_end : "
          + " ".join(f"{v:.3f}" for v in pcpc["G_end_reg"]))
    print("    frozen    G_end : "
          + " ".join(f"{v:.3f}" for v in pcpc["G_end_frozen"]))
    print("    -> the episodic-stress protection has a ceiling: permanent "
          "capture at a_hold ~ 1.5\n"
          "       drives regulated G below 0.1 (0.095/0.089 at 1.5/2.0), "
          "though never to the\n"
          "       frozen stuck level 0.049.")

    pf = part_f()
    np.savez("cache/exp6_longhorizon.npz",
             **{f"{lab}|{arm}|{key}": val
                for lab in [p[0] for p in PATTERNS]
                for arm in ("frozen", "reg")
                for key, val in pe[lab][arm].items()
                if isinstance(val, np.ndarray)},
             **{f"intensity_{k}": v for k, v in ii.items()},
             **{f"permcap_{k}": v
                for k, v in pe["permanent_capture"].items()})
    np.savez("cache/exp6_mechanism.npz", **pf)

    # ---------------------------------------------------------- figure
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3)

    axA = fig.add_subplot(gs[0, 0])
    for name, (t, G) in pd_["trajs"].items():
        col = ("k" if name.startswith("frozen") else
               "tab:green" if name.startswith("CHEAP") else
               "tab:blue" if name.startswith("actuator") else "tab:red")
        axA.plot(t, G, color=col, lw=1.2, label=f"{name} → G={G[-1]:.3f}")
    axA.axvline(T_ENGAGE, color="gray", ls=":", lw=0.8)
    axA.axhline(0.1, color="k", ls=":", lw=0.6)
    axA.set_xlabel("t"); axA.set_ylabel("G(t)")
    axA.set_title("(d) RESCUING AN ALREADY-STUCK AGENT (engage t=600):\n"
                  "CHEAP floor escapes, ELABORATE gated c_mon≥0.8 fails\n"
                  "(assay geometry — NOT deployment; see d'')")
    axA.legend(fontsize=7)

    axB = fig.add_subplot(gs[0, 1])
    i_none = pa["floors_is_none"].index(True)
    i_07 = pa["floors"].index(0.7)
    # true bin EDGES for pcolormesh: the grids (ks, cms) are non-uniform, so
    # imshow's uniform-cell assumption misplaces cells by up to ~0.2 (review
    # finding 6) — edges from the midpoint between neighbours.
    def edges_of(v):
        e = np.empty(len(v) + 1)
        e[1:-1] = 0.5 * (np.asarray(v[:-1], float)
                         + np.asarray(v[1:], float))
        e[0] = float(v[0]) - 0.5 * (float(v[1]) - float(v[0]))
        e[-1] = float(v[-1]) + 0.5 * (float(v[-1]) - float(v[-2]))
        return e
    cm_e, k_e = edges_of(cms), edges_of(ks)
    for i, cmap in ((i_none, "Greys"), (i_07, "Greens")):
        axB.pcolormesh(cm_e, k_e, (pa["G"][i] > 0.5).astype(float),
                       cmap=cmap, vmin=0, vmax=1, alpha=0.75,
                       shading="flat")
    axB.set_xticks(cms)
    axB.set_yticks(ks)
    axB.set_xlabel("monitoring cost c_mon (gated)")
    axB.set_ylabel("actuator authority k_pull")
    axB.set_title("(a) working window: escape mask\n"
                  "grey = no floor, green = floor 0.7 (dark = escaped)")

    axC = fig.add_subplot(gs[0, 2])
    axC.plot(pc["vals"], pc["G_end_theta"], "o-", ms=3,
             label="floor on Theta_eff")
    axC.plot(pc["vals"], pc["G_end_S"], "s--", ms=3, label="floor on S")
    axC.axvline(pb["floor_crit_theta"], color="tab:red", ls=":", lw=1)
    axC.axvline(pb["stuck_E"], color="k", ls=":", lw=0.8)
    axC.annotate(f"floor_crit={pb['floor_crit_theta']:.3f}\n"
                 f"stuck E*={pb['stuck_E']:.3f}",
                 xy=(pb["stuck_E"], 0.45), fontsize=7)
    axC.set_xlabel("floor value (even FALSE — real Theta_eff* = 0.12)")
    axC.set_ylabel("G(T)")
    axC.set_title("(c) ANY floor above the critical level\n"
                  "works identically; below it, none")
    axC.legend(fontsize=7)

    axD = fig.add_subplot(gs[1, 0])
    axD.plot(pd_["cms_ungated_post"], pd_["post_ungated"], "^--",
             color="tab:orange", ms=4,
             label="POST-COLLAPSE, floor held, ungated (knowing floor)")
    axD.plot(pd_["cms"], pd_["healthy_ungated"], "o-", color="tab:red",
             label="HEALTHY regime, ungated (always-on vigilance)")
    axD.plot(pd_["cms"], pd_["healthy_gated"], "s-", color="tab:green",
             label="HEALTHY regime, gated (check-then-act)")
    axD.axhline(0.5, color="k", ls=":", lw=0.6)
    axD.set_xlabel("c_mon")
    axD.set_ylabel("G(T), floor on")
    axD.set_title("(d') the threshold that survives deployment is the\n"
                  "UNGATED one: always-on re-examination defeats even the\n"
                  "floor at c_mon≈0.2; a GATED cost is free (assay-only)")
    axD.legend(fontsize=7)

    axE = fig.add_subplot(gs[1, 1])
    lab0 = PATTERNS[0][0]
    r = pe[lab0]
    axE.plot(r["frozen"]["t"], r["frozen"]["G"], color="k", lw=0.9,
             label=f"frozen (stuck from ep 1, G_end {r['frozen']['G_end']:.3f})")
    axE.plot(r["reg"]["t"], r["reg"]["G"], color="tab:green", lw=0.9,
             label=f"floor-regulated (G_min {r['reg']['G_min']:.3f}, "
                   f"G_end {r['reg']['G_end']:.3f})")
    axE.axhline(0.1, color="k", ls=":", lw=0.6)
    axE.set_xlabel("t"); axE.set_ylabel("G(t)")
    axE.set_title(f"(e) long horizon: {lab0}\n"
                  "regulated never enters the stuck attractor")
    axE.legend(fontsize=7)

    axF = fig.add_subplot(gs[1, 2])
    m = pf["t"] > 500.0
    axF.plot(pf["t"][m], pf["G"][m], color="tab:green", label="G")
    axF.plot(pf["t"][m], pf["a"][m], color="tab:red", label="a (captured)")
    axF.plot(pf["t"][m], pf["Theta_eff_frozen"][m], color="gray", ls="--",
             lw=0.8, label="Theta_eff (real, frozen)")
    axF.plot(pf["t"][m], pf["Theta_eff"][m], color="tab:blue", ls="--",
             lw=0.8, label="Theta_eff (floored belief)")
    axF.plot(pf["t"][m], pf["c"][m], color="orange", lw=0.8, label="c")
    axF.axvline(T_ENGAGE, color="gray", ls=":", lw=0.8)
    axF.set_xlabel("t"); axF.set_ylabel("state")
    axF.set_title("(f) mechanism: floor → Θ_eff ≥ E → c=0 → a→0 → G regrows")
    axF.legend(fontsize=7)

    fig.suptitle("exp6 — self-rescue regulator: a CHEAP FALSE FLOOR severs "
                 "the loop at zero cost; sustained inward introspection "
                 "cost defeats even the floor (assay: rescuing an "
                 "already-stuck agent; frozen model untouched; "
                 "deterministic, in-model only)")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "f11_regulator.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\nfigure -> {path}")
    np.savez("cache/exp6_floor.npz",
             vals=pc["vals"], G_end_theta=pc["G_end_theta"],
             a_end_theta=pc["a_end_theta"], G_end_S=pc["G_end_S"],
             escape_times=pc["escape_times"])
    print(f"total {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
