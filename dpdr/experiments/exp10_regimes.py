"""exp10 — the two-regime structure, the dwell-limit structure, and the
protection contrast (ALL FROZEN MODEL, deterministic; every number below is
reproducible from this driver + cache/exp10_*.npz).

Three measurements formalized from ad-hoc orchestrator runs (nodes
handoff-selfreg-liminal / -protection / -morning / -weak-demand):

  A. THE TWO REGIMES AND THE CAPTURE THRESHOLD (headline).  The frozen
     model is NOT binary in general: at low capture gain chi an inward
     episode (a_hold 0.9, dur 80 from t=100, canonical ICs) fires the
     cannibalization loop at FULL amplitude (c = 1.000) and the system
     RELEASES (verified anchor chi=0.3, eta=0.3: t=180 a=0.900 c=1.000
     G=0.1155; t=260 a=0.000 c=0.000 G=0.8834); at high chi it LOCKS
     (stuck attractor, is_stuck).  This driver maps the {no-fire /
     transient-release / locked} classes over (chi x eta) and bisects
     chi_crit(eta) — the release/lock boundary.  Anchor: chi_crit ~ 0.638
     at eta=0.3; canonical (chi=1.0, eta=1.2) sits deep in the locked
     regime, which is why duration sweeps AT canonical chi look binary.

  B. THE DWELL-LIMIT STRUCTURE.  Recall modelled as an inward-attention
     pulse (a_hold 0.9) applied from t=0 to settled ICs with S pinned at
     S0 and a standing u_ext; 'collapsed' = G(T) < 0.1 at T = D + 300
     (the exp9 P1 / weak-demand protocol).  Dwell limit = bisected
     smallest collapsing D on [5, 700].  Anchors: vs S0 (u=0):
     0.3->18.80, 0.5->35.30, 0.7->72.75, 0.8->98.90, 0.9->124.05,
     1.0->146.80; vs u_ext (S0=0.8): 0.00->98.90, 0.02->121.30,
     0.05->171.70, 0.08->262.60, 0.10->384.70, 0.15+ -> no limit to 700.

  C. THE PROTECTION CONTRAST.  Recall alone is harmful only if SUSTAINED:
     a 120 t.u. recall collapses, a 40 t.u. recall is safe (G=0.8855);
     duration is thresholded at canonical chi (anchor sweep dur=20/40/60
     -> c never fires, recovered; dur=80 -> c=1.000 locked, c>0.5 for
     1332.6 t.u.).  TWO INDEPENDENT PROTECTIONS, either sufficient:
     standing external work (u_ext = 0.3) and a threshold floor (0.7,
     dpdr.regulator) both make a 300 t.u. sustained recall safe.

  D. (bonus, kept separate) reduced control gain g0: dwell limit vs g0 at
     S0=0.8, no engagement.  Anchors: 0.50->98.90, 0.40->87.65,
     0.30->77.95, 0.20->69.75, 0.10->63.00, 0.05->60.05.

Protocol notes (honest reporting):
  * Parts A/C use CANONICAL ICs (Params defaults) and the canonical
    episode geometry (episode from t=100); part B/D use SETTLED ICs
    (baseline run to t=5000) with S0 re-pinned, episode from t=0 — this is
    what reproduces the 98.90 anchor family (exp9 precedent).
  * 'probe' re-checks the anchor cells cheaply before the sweeps and
    prints timings; the ad-hoc anchor numbers were produced with unknown
    dt/bisection grids, so small deviations are expected and are reported,
    not forced.
  * No RHS is reimplemented: parts A/B/C-alone use dpdr.integrate.simulate
    on dpdr.model.deriv with Params overrides; the floor arm of part C
    uses dpdr.regulator.simulate_reg (the shipped opt-in wrapper, floor
    only, inert-when-off by construction).

Usage:  .venv/bin/python experiments/exp10_regimes.py [probe|A|B|C|D|fig|all]
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace

import numpy as np

from dpdr.events import Schedule
from dpdr.integrate import simulate
from dpdr.metrics import is_stuck
from dpdr.model import Params
from dpdr.regulator import RegulatorParams, simulate_reg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
FIGDIR = os.path.join(ROOT, "figs")

P = Params()
DT = 0.5                    # integration output grid (exp8/9 precedent)
DT_DWELL = 0.05             # dwell-bisection grid (anchor resolution 0.05)
BISECT_GRID = 0.05          # schedule-time snapping inside bisections
EP_T0 = 100.0               # canonical episode start (parts A/C)
EP_DUR = 80.0               # part-A episode duration (the verified cell)

# anchor tables (ad-hoc orchestrator numbers; compared, not enforced)
ANCH_CHI_CRIT = {0.3: 0.638}
ANCH_DWELL_S0 = {0.3: 18.80, 0.5: 35.30, 0.7: 72.75, 0.8: 98.90,
                 0.9: 124.05, 1.0: 146.80}
ANCH_DWELL_U = {0.00: 98.90, 0.02: 121.30, 0.05: 171.70, 0.08: 262.60,
                0.10: 384.70}
ANCH_DWELL_U_NOLIMIT = (0.15, 0.20, 0.30)
ANCH_DWELL_G0 = {0.50: 98.90, 0.40: 87.65, 0.30: 77.95, 0.20: 69.75,
                 0.10: 63.00, 0.05: 60.05}
ANCH_LIMINAL = [(140.0, 0.871, 0.000, 0.1852), (180.0, 0.900, 1.000, 0.1155),
                (200.0, 0.692, 0.991, 0.1532), (260.0, 0.000, 0.000, 0.8834)]

CLS_NAMES = ["nofire", "release", "locked", "partial"]
NOFIRE, RELEASE, LOCKED, PARTIAL = 0, 1, 2, 3
# The canonical trigger includes the affect pulse A=0.5 (events'
# failure_schedule): probe-verified — WITHOUT it even chi=1.0 never fires;
# WITH it the liminal anchor cell (chi=0.3, eta=0.3) reproduces to ~2e-3.
A_PULSE = 0.5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def snap(x: float, dt: float = DT) -> float:
    return round(round(x / dt) * dt, 10)


# --------------------------------------------------------------- part 0
_SETTLED = None


def settled() -> dict:
    """Settled baseline state (canonical params, no schedule, t=5000)."""
    global _SETTLED
    if _SETTLED is None:
        f = os.path.join(CACHE, "exp10_part0.npz")
        if os.path.exists(f):
            z = np.load(f)
            _SETTLED = {k: float(z[k]) for k in z.files}
        else:
            t0 = time.time()
            sb = simulate(P, Schedule({}), 5000.0, dt=DT)
            _SETTLED = {k: float(sb[k][-1]) for k in ("a", "G", "D", "S", "g")}
            np.savez(f, **_SETTLED)
            log(f"part0: settled ICs {_SETTLED} "
                f"({time.time() - t0:.1f}s, cached exp10_part0.npz)")
    return _SETTLED


# --------------------------------------------------------------- part A
def classify_regime(sol, p: Params) -> int:
    if is_stuck(sol, p):
        return LOCKED
    if float(sol["c"].max()) < 0.5:
        return NOFIRE
    return RELEASE if float(sol["G"][-1]) > 0.5 else PARTIAL


def regime_run(chi: float, eta: float, dur: float = EP_DUR,
               T: float = 1500.0, pulse: float | None = None):
    """Canonical ICs + episode a_hold 0.9 at [100, 100+dur); affect pulse of
    the canonical amplitude 0.5 for the first min(60, dur) by default."""
    p = replace(P, chi=chi, eta=eta)
    t1 = snap(EP_T0 + dur)
    ch = {"a_hold": [(EP_T0, t1, 0.9)]}
    if pulse is None:
        pulse = A_PULSE
    if pulse:
        ch["A"] = [(EP_T0, snap(EP_T0 + min(60.0, dur)), pulse)]
    sol = simulate(p, Schedule(ch), snap(T), dt=DT)
    return sol, classify_regime(sol, p)


def partA_pulse() -> float | None:
    return float(os.environ.get("EXP10_A_PULSE", A_PULSE)) or None


def part_a() -> None:
    pulse = partA_pulse()
    log(f"partA START: regime map chi x eta, episode dur={EP_DUR:g} + pulse "
        f"{pulse}, T=1500, dt={DT:g}")
    chis = np.round(np.arange(0.0, 1.5001, 0.05), 4)
    etas = np.round(np.arange(0.0, 1.6001, 0.10), 4)
    M = np.zeros((len(etas), len(chis)), int)
    cmax = np.zeros_like(M, float)
    Gend = np.zeros_like(M, float)
    for i, eta in enumerate(etas):
        t0 = time.time()
        for j, chi in enumerate(chis):
            sol, k = regime_run(chi, eta, pulse=pulse)
            M[i, j], cmax[i, j], Gend[i, j] = k, sol["c"].max(), sol["G"][-1]
        n_l = int((M[i] == LOCKED).sum())
        log(f"partA map: eta={eta:.2f} row {i + 1}/{len(etas)} "
            f"done: locked={n_l}/{len(chis)} ({time.time() - t0:.1f}s)")
    np.savez(os.path.join(CACHE, "exp10_partA_map.npz"),
             chi=chis, eta=etas, cls=M, c_max=cmax, G_end=Gend,
             dur=EP_DUR, pulse=(pulse or 0.0), T=1500.0)
    log("partA map cached exp10_partA_map.npz")

    # bisect TWO boundaries per eta: chi_release (G_end > 0.5) and chi_lock
    # (is_stuck).  [0.02, 1.60], 11 iters, grid 0.005 -> cell ~0.0008
    bis_etas = [0.2, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5]
    chi_rel, chi_lock, cells = [], [], []
    for eta in bis_etas:
        t0 = time.time()

        def released(chi):
            return regime_run(chi, eta, pulse=pulse)[0]["G"][-1] > 0.5

        def locked(chi):
            return regime_run(chi, eta, pulse=pulse)[1] == LOCKED

        vr = bisect_bool(released, 0.02, 1.60, 11, grid=0.005)
        vl = bisect_bool(locked, 0.02, 1.60, 11, grid=0.005)
        chi_rel.append(np.nan if vr is None else vr)
        chi_lock.append(np.nan if vl is None else vl)
        cells.append(1.58 / 2 ** 11)
        log(f"partA bisect: eta={eta:.2f} chi_release={vr} chi_lock={vl} "
            f"(anchor chi_crit ~0.638 at eta=0.3) ({time.time() - t0:.1f}s)")
    np.savez(os.path.join(CACHE, "exp10_partA_bisect.npz"),
             eta=np.array(bis_etas),
             chi_release=np.array(chi_rel, float),
             chi_lock=np.array(chi_lock, float),
             cell=np.array(cells), T=1500.0, dur=EP_DUR,
             pulse=(pulse or 0.0))
    log("partA DONE (map + bisected boundaries cached)")


# ------------------------------------------------------ part A0 (no pulse)
def part_a_nopulse() -> None:
    """Document the NO-FIRE class: without the canonical affect pulse the
    loop never fires anywhere on the swept chi range at dur=80 (probe-
    verified at eta=0.3 and at the canonical cell).  Coarse map at the two
    anchor etas."""
    log("partA0 START: no-pulse map (eta=0.3 and 1.2), the no-fire class")
    chis = np.round(np.arange(0.0, 1.5001, 0.05), 4)
    etas = [0.3, 1.2]
    M = np.zeros((len(etas), len(chis)), int)
    for i, eta in enumerate(etas):
        for j, chi in enumerate(chis):
            M[i, j] = regime_run(chi, eta, pulse=0.0)[1]
        log(f"partA0: eta={eta} row: nofire={int((M[i] == NOFIRE).sum())}"
            f"/{len(chis)} release={int((M[i] == RELEASE).sum())} "
            f"locked={int((M[i] == LOCKED).sum())}")
    np.savez(os.path.join(CACHE, "exp10_partA_nopulse.npz"),
             chi=chis, eta=np.array(etas), cls=M, dur=EP_DUR, pulse=0.0,
             T=1500.0)
    log("partA0 DONE (exp10_partA_nopulse.npz)")


# --------------------------------------------------------------- bisect
def bisect_bool(f, lo: float, hi: float, it: int, grid: float = BISECT_GRID):
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


# --------------------------------------------------------------- part B
def dwell_collapsed(D: float, S0: float = 0.8, u: float = 0.0,
                    g0: float | None = None) -> bool:
    """exp9 P1 protocol: settled ICs, S pinned at S0, standing u_ext,
    a_hold 0.9 from t=0 for dwell D; collapsed = G(D+300) < 0.1."""
    st = settled()
    kw = dict(a0=st["a"], G0=st["G"], D0=st["D"], S0=S0, g_init=st["g"])
    if g0 is not None:
        kw["g0"] = g0
    p = replace(P, **kw)
    D = snap(D, BISECT_GRID)
    T = snap(D + 300.0)
    ch = {"a_hold": [(0.0, D, 0.9)]}
    if u > 0.0:
        ch["u_ext"] = [(0.0, T, u)]
    s = simulate(p, Schedule(ch), T, dt=DT_DWELL)
    return bool(s["G"][-1] < 0.1)


def part_b() -> None:
    log("partB START: dwell-limit structure (settled ICs, a_hold 0.9 from "
        "t=0, bisect [5,700] x 11, grid 0.05)")
    TMAX, NIT = 700.0, 11
    S0s = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    d_S0 = []
    for S0 in S0s:
        t0 = time.time()
        val = bisect_bool(lambda D: dwell_collapsed(D, S0=S0), 5.0, TMAX, NIT)
        d_S0.append(np.nan if val is None else val)
        log(f"partB vs S0: S0={S0:.2f} dwell={val} "
            f"(anchor {ANCH_DWELL_S0.get(S0)}) ({time.time() - t0:.1f}s)")
    us = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    d_u = []
    for u in us:
        t0 = time.time()
        val = bisect_bool(lambda D: dwell_collapsed(D, S0=0.8, u=u),
                          5.0, TMAX, NIT)
        d_u.append(np.nan if val is None else val)
        anch = ANCH_DWELL_U.get(u, "no-limit-to-700" if u > 0.1 else None)
        log(f"partB vs u_ext: u={u:.2f} dwell={val} (anchor {anch}) "
            f"({time.time() - t0:.1f}s)")
    np.savez(os.path.join(CACHE, "exp10_partB_dwell.npz"),
             S0=np.array(S0s), dwell_S0=np.array(d_S0, float),
             u=np.array(us), dwell_u=np.array(d_u, float),
             bracket=(5.0, TMAX), iters=NIT, grid=BISECT_GRID)
    log("partB DONE (exp10_partB_dwell.npz)")


# --------------------------------------------------------------- part C
def recall_run(dur: float, T: float = 1500.0, u: float | None = None,
               floor: float | None = None, pulse: float | None = None):
    """Canonical ICs, recall = a_hold 0.9 at [100, 100+dur); optional
    standing u_ext and/or threshold floor (dpdr.regulator)."""
    T = snap(T)
    t1 = snap(EP_T0 + dur)
    ch = {"a_hold": [(EP_T0, t1, 0.9)]}
    if pulse:
        ch["A"] = [(EP_T0, snap(EP_T0 + min(60.0, dur)), pulse)]
    if u is not None and u > 0.0:
        ch["u_ext"] = [(0.0, T, u)]
    if floor is None:
        return simulate(P, Schedule(ch), T, dt=DT)
    r = RegulatorParams(floor=floor, t_engage=0.0)   # k_pull=c_mon=0
    return simulate_reg(P, r, Schedule(ch), T, dt=DT)


def final_c_stuck(sol) -> float:
    """Duration of the final c>0.5 epoch (0 if the loop is off at the end)."""
    t, c = sol["t"], sol["c"]
    idx = np.where(c <= 0.5)[0]
    if idx.size == 0:
        return float(t[-1] - t[0])
    return float(t[-1] - t[idx[-1]]) if c[-1] > 0.5 else 0.0


def part_c() -> None:
    log("partC START: protection contrast (canonical chi/eta, T=1500)")
    pulse = float(os.environ.get("EXP10_C_PULSE", A_PULSE)) or None
    durs = [20.0, 40.0, 60.0, 80.0, 100.0, 120.0]
    rows = []
    for dur in durs:
        t0 = time.time()
        s = recall_run(dur, pulse=pulse)
        row = dict(dur=dur, c_max=float(s["c"].max()),
                   G_end=float(s["G"][-1]), c_stuck=final_c_stuck(s),
                   locked=is_stuck(s, P))
        rows.append(row)
        log(f"partC sweep: dur={dur:g} c_max={row['c_max']:.3f} "
            f"G_end={row['G_end']:.4f} c>0.5 for {row['c_stuck']:.1f} "
            f"locked={row['locked']} ({time.time() - t0:.1f}s)")
    # bisect the duration threshold of the recall-alone arm
    t0 = time.time()
    thr = bisect_bool(lambda d: is_stuck(recall_run(d, pulse=pulse), P),
                      20.0, 200.0, 11, grid=1.0)
    log(f"partC: recall-alone duration threshold = {thr} "
        f"(cell 180/2^11 = 0.09, grid 1) ({time.time() - t0:.1f}s)")
    # the two protections against a sustained (300 t.u.) recall
    arms = {}
    for name, kw in [("alone", {}),
                     ("u_ext0.3", dict(u=0.3)),
                     ("floor0.7", dict(floor=0.7))]:
        t0 = time.time()
        s = recall_run(300.0, **kw)
        arms[name] = dict(G_end=float(s["G"][-1]),
                          c_max=float(s["c"].max()),
                          locked=is_stuck(s, P))
        log(f"partC arm recall-300 {name}: G_end={arms[name]['G_end']:.4f} "
            f"c_max={arms[name]['c_max']:.3f} locked={arms[name]['locked']} "
            f"({time.time() - t0:.1f}s)")
    np.savez(os.path.join(CACHE, "exp10_partC_protection.npz"),
             dur=np.array([r["dur"] for r in rows]),
             c_max=np.array([r["c_max"] for r in rows]),
             G_end=np.array([r["G_end"] for r in rows]),
             c_stuck=np.array([r["c_stuck"] for r in rows]),
             locked=np.array([r["locked"] for r in rows]),
             dur_threshold=np.nan if thr is None else thr,
             arm=list(arms.keys()),
             arm_G_end=np.array([arms[k]["G_end"] for k in arms]),
             arm_locked=np.array([arms[k]["locked"] for k in arms]),
             pulse=(pulse or 0.0))
    log("partC DONE (exp10_partC_protection.npz)")


# --------------------------------------------------------------- part D
def dwell_collapsed_g0(D: float, st: dict, g0: float,
                       S0: float = 0.8) -> bool:
    """Dwell-collapse test from the SETTLED-AT-g0 state (the ad-hoc
    protocol: a chronically low control gain changes the settled G, g and
    a; reusing the g0=0.5 settled state under-reports the effect — probe-
    verified: settled-at-g0 reproduces the anchors 87.65/63.0/60.05,
    reused-settled gives 97.4/93.8/93.4)."""
    p = replace(P, a0=st["a"], G0=st["G"], D0=st["D"], S0=S0,
                g_init=st["g"], g0=g0)
    D = snap(D, BISECT_GRID)
    T = snap(D + 300.0)
    s = simulate(p, Schedule({"a_hold": [(0.0, D, 0.9)]}), T, dt=DT_DWELL)
    return bool(s["G"][-1] < 0.1)


def part_d() -> None:
    log("partD START: dwell vs control gain g0 (settled-AT-g0 ICs, S0=0.8, "
        "u=0; SEPARATE from the regime result — reduced executive function)")
    g0s = [0.50, 0.40, 0.30, 0.20, 0.10, 0.05]
    d, sts = [], []
    for g0 in g0s:
        t0 = time.time()
        sb = simulate(replace(P, g0=g0), Schedule({}), 5000.0, dt=DT)
        st = {k: float(sb[k][-1]) for k in ("a", "G", "D", "S", "g")}
        sts.append(st)
        val = bisect_bool(lambda D: dwell_collapsed_g0(D, st, g0),
                          5.0, 700.0, 11)
        d.append(np.nan if val is None else val)
        log(f"partD: g0={g0:.2f} settled G={st['G']:.4f} g={st['g']:.4f} "
            f"dwell={val} (anchor {ANCH_DWELL_G0.get(g0)}) "
            f"({time.time() - t0:.1f}s)")
    np.savez(os.path.join(CACHE, "exp10_partD_g0.npz"),
             g0=np.array(g0s), dwell=np.array(d, float),
             settled_G=np.array([s["G"] for s in sts]),
             settled_g=np.array([s["g"] for s in sts]),
             settled_a=np.array([s["a"] for s in sts]))
    log("partD DONE (exp10_partD_g0.npz)")


# --------------------------------------------------------------- probe
def probe() -> None:
    log("probe: timings + anchor checks (no caches written)")
    for dt, T in [(DT, 1000.0), (0.05, 1000.0), (DT, 1500.0)]:
        t0 = time.time()
        p = replace(P, chi=0.3, eta=0.3)
        ch = Schedule({"a_hold": [(EP_T0, snap(EP_T0 + EP_DUR), 0.9)]})
        s = simulate(p, ch, snap(T), dt=dt)
        log(f"probe timing: dt={dt:g} T={T:g} -> {time.time() - t0:.2f}s "
            f"(nfev={s['nfev']})")
    # liminal anchor, no-pulse vs canonical pulse
    for tag, pulse in [("nopulse", None), ("pulse0.5", 0.5)]:
        s, k = regime_run(0.3, 0.3, pulse=pulse)
        reads = []
        for (ta, aa, ca, Ga) in ANCH_LIMINAL:
            i = int(np.argmin(np.abs(s["t"] - ta)))
            reads.append(f"t={ta:g}: a={s['a'][i]:.3f} c={s['c'][i]:.3f} "
                         f"G={s['G'][i]:.4f} (anchor a={aa} c={ca} G={Ga})")
        log(f"probe A chi=0.3 eta=0.3 [{tag}] cls={CLS_NAMES[k]} G_end="
            f"{s['G'][-1]:.4f} | " + "; ".join(reads))
    s, k = regime_run(1.0, 1.2, pulse=None)
    log(f"probe A canonical chi=1.0 eta=1.2 nopulse: cls={CLS_NAMES[k]} "
        f"G_end={s['G'][-1]:.4f} c_end={s['c'][-1]:.3f}")
    for chi in (0.60, 0.63, 0.64, 0.65, 0.70):
        _, k = regime_run(chi, 0.3)
        log(f"probe A chi={chi:.2f} eta=0.3 -> {CLS_NAMES[k]}")
    # part B anchor checks (settled ICs)
    st = settled()
    log(f"probe B settled: {st}")
    for S0, near in [(0.8, (98.9, 99.4)), (0.3, (18.8, 19.3))]:
        for D in near:
            log(f"probe B S0={S0} D={D}: collapsed="
                f"{dwell_collapsed(D, S0=S0)}")
    # part C duration sweep, no-pulse vs pulse (which reproduces the
    # anchor pattern 20/40/60 safe, 80 stuck?)
    for tag, pulse in [("nopulse", None), ("pulse0.5", 0.5)]:
        for dur in (20.0, 40.0, 60.0, 80.0, 120.0):
            s = recall_run(dur, pulse=pulse)
            log(f"probe C [{tag}] dur={dur:g}: c_max={s['c'].max():.3f} "
                f"G_end={s['G'][-1]:.4f} c_stuck={final_c_stuck(s):.1f} "
                f"locked={is_stuck(s, P)}")
    log("probe DONE")


# --------------------------------------------------------------- figure
def part_fig() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log("fig START: f15_regimes.png")
    zA = np.load(os.path.join(CACHE, "exp10_partA_map.npz"))
    zB = np.load(os.path.join(CACHE, "exp10_partB_dwell.npz"))
    zC = np.load(os.path.join(CACHE, "exp10_partC_protection.npz"))
    zAb = np.load(os.path.join(CACHE, "exp10_partA_bisect.npz"))
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 9.0))

    ax = axs[0, 0]
    M = zA["cls"]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#74a9cf", "#1b7837", "#b2182b", "#e08214"])
    ax.pcolormesh(zA["chi"], zA["eta"], M, cmap=cmap,
                  vmin=-0.5, vmax=3.5, shading="nearest")
    ok = np.isfinite(zAb["chi_lock"])
    ax.plot(zAb["chi_release"][ok], zAb["eta"][ok], "k--", ms=4, lw=1.2,
            label="bisected $\\chi_{release}$")
    ax.plot(zAb["chi_lock"][ok], zAb["eta"][ok], "ko-", ms=4, lw=1.2,
            label="bisected $\\chi_{lock}$")
    ax.plot(P.chi, P.eta, "*", ms=16, mfc="yellow", mec="k",
            label="canonical (1.0, 1.2)")
    ax.set_xlabel("capture gain $\\chi$")
    ax.set_ylabel("cannibalization cost $\\eta$")
    ax.set_title("(a) two regimes: no-fire / release / locked"
                 " (white=partial)")
    ax.legend(fontsize=7, loc="lower left")

    ax = axs[0, 1]
    ax.plot(zB["S0"], zB["dwell_S0"], "o-", label="dwell limit")
    for s0, a in ANCH_DWELL_S0.items():
        ax.plot([s0], [a], "x", ms=6, label="ad-hoc anchor" if s0 == 0.3
                else None)
    ax.set_xlabel("pinned setpoint $S_0$")
    ax.set_ylabel("dwell limit [t.u.]")
    ax.set_title("(b) dwell limit vs setpoint ($u_{ext}=0$)")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    fin = np.isfinite(zB["dwell_u"])
    ax.plot(zB["u"][fin], zB["dwell_u"][fin], "o-", label="finite limit")
    nol = zB["u"][~fin]
    if nol.size:
        ax.plot(nol, np.full(nol.size, 650.0), "^", ms=8,
                label="no collapse $\\leq 700$")
    for u, a in ANCH_DWELL_U.items():
        ax.plot([u], [a], "x", ms=6, label="ad-hoc anchor" if u == 0.0
                else None)
    ax.set_xlabel("standing external demand $u_{ext}$")
    ax.set_ylabel("dwell limit [t.u.]")
    ax.set_title("(c) dwell limit vs engagement ($S_0=0.8$)")
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.plot(zC["dur"], zC["G_end"], "o-", label="$G(T)$, recall alone")
    ax.plot(zC["dur"], zC["c_max"], "s--", label="$c_{max}$, recall alone")
    arms = dict(zip([str(x) for x in zC["arm"]], zC["arm_G_end"]))
    ax.axhline(0.1, color="gray", lw=0.8, ls=":")
    ax.axvline(300, color="k", lw=0.8, ls=":")
    yl = ax.get_ylim()
    txt = [f"recall-300: {k} -> G={v:.3f}" for k, v in arms.items()]
    ax.text(0.03, 0.03, "\n".join(txt), transform=ax.transAxes, fontsize=8,
            va="bottom")
    ax.set_xlabel("recall duration [t.u.] ($T=1500$)")
    ax.set_ylabel("$G(T)$ / $c_{max}$")
    ax.set_title("(d) duration threshold + the two protections")
    ax.legend(fontsize=8, loc="center right")

    fig.suptitle("exp10 — frozen model: two regimes, dwell-limit structure, "
                 "protection contrast (deterministic)", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "f15_regimes.png")
    fig.savefig(out, dpi=140)
    log(f"fig DONE -> {out}")


def main(argv: list[str]) -> None:
    os.makedirs(CACHE, exist_ok=True)
    parts = argv[1:] or ["probe", "A", "B", "C", "D", "fig"]
    if "all" in parts:
        parts = ["A", "B", "C", "D", "fig"]
    todo = {"probe": probe, "A": part_a, "A0": part_a_nopulse, "B": part_b,
            "C": part_c, "D": part_d, "fig": part_fig}
    for name in parts:
        todo[name]()


if __name__ == "__main__":
    main(sys.argv)
