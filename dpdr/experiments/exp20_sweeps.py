"""exp20 — TWO SINGLE-PARAMETER SWEEPS that convert the three-phase trajectory
inference (handoff-selfreg-three-phases) into measurements, plus the chronic
comparison (handoff-selfreg-sweeps).

The frozen model (dpdr/model.py, READ-ONLY) has five states; the collapse
switch is c = sigma_c*tanh((E - Theta_eff)/w) with Theta_eff = Theta*S/S_rest,
so the switch fires exactly when E crosses Theta*S/S_rest.  The inference
under test says the canonical failure episode (events.failure_schedule:
a_hold 0.9 on [100,200), affect pulse 0.5 on [100,160)) collapses in three
phases — reduction-dominant runaway (c off, G 0.885 -> 0.145), then a crush
after E and Theta_eff MEET (c arms, G 0.145 -> 0.051) — with E saturating
near 0.526 while Theta_eff keeps falling: a TWO-TIMESCALE RACE between a
saturated deficit (fast) and a draining setpoint (slow), not a runaway
demand.

*** PRE-REGISTRATION — WRITTEN BEFORE ANY RUN OF THIS SCRIPT ***
(timestamp at writing: 2025-09-24; the sweeps below were first executed
only after this block was committed to the file; nothing here is tuned
after seeing output)

P1 — SWEEP tau_S (frozen value 100; grid {25, 50, 100, 150, 200, 400, 800}).
  MECHANISM PREDICTION (setpoint drain): with a -> its inward fixed point
  fast (tau_a = 1), S relaxes during the episode toward S_inf ~= 0.163 with
  rate (k_s + lam_S)/tau_S = 0.55/tau_S from its pre-onset level ~= 0.894.
  The switch can fire only once S < S_rest*E/Theta ~= 0.657 (E saturates
  ~= 0.52 on the tau_D/tau_G timescale, tau_S-independent).  Quasi-statically
  the crossing therefore sits at  t_c - 100 ~= max(t_E_rise, kappa*tau_S)
  with kappa = -ln(1 - x)/(k_s + lam_S), x = (0.894 - 0.657)/(0.894 - 0.163)
  ~= 0.32, i.e. kappa ~= 0.71 (all numbers from the frozen equations, not
  from a run).  SO: (a) the crossing time should scale APPROXIMATELY LINEARLY
  with tau_S where the crossing exists, with log-log slope below 1 at small
  tau_S because the E-rise floor (t_E ~= 25-35 t.u.) dominates there;
  (b) THE CROSSING SHOULD VANISH once kappa*tau_S exceeds the window in
  which E stays high (the episode ends at t = 200): no collapse at all for
  tau_S >= ~140, i.e. on this grid collapse at {25, 50, 100}, none at
  {200, 400, 800}; (c) the pre-crossing E plateau should be approximately
  tau_S-INDEPENDENT (E's dynamics contain no tau_S); (d) the closing of the
  last 15 t.u. before the crossing should be setpoint-dominated, with
  setpoint share INCREASING with tau_S (E flat, Theta_eff draining).
  FALSIFIER (any one refutes the setpoint-drain mechanism): the crossing
  time is INDEPENDENT of tau_S (=> E drives the crossing); OR the run still
  collapses at tau_S = 800 with the canonical crossing time (=> the drain
  is not necessary); OR the E plateau tracks tau_S strongly (=> E is not
  the saturated fast variable).
  ALSO MEASURED (expected, not load-bearing): the runaway/crush damage
  split is expected to be REGIME-DEPENDENT — the crush share should GROW as
  tau_S shrinks (crossing earlier in the episode, more of the loss pushed
  behind the switch) rather than stay at the canonical ~89/11.

P2 — SWEEP tau_g (frozen value 200; grid {6.25, 12.5, 25, 50, 100, 200, 400,
  800}), episodic canonical failure schedule.
  MECHANISM PREDICTION (slow gain): dg/dt = (pi*max(0, |dE/dt| - dEdt_ref)
  - mu*(g - g0))/tau_g has a tau_g-INDEPENDENT quasi-steady ceiling
  g - g0 -> pi*(|dE/dt| - dEdt_ref)/mu for sustained drive, and tau_g only
  sets how close g gets to that ceiling within an event: the excursion
  should scale ~ 1/tau_g (log-log slope ~ -1) until it saturates at the
  ceiling ~ pi*max|dE/dt|/mu ~= 0.05-0.09 for the canonical event.  SO the
  loop ENGAGES (excursion > ENGAGE_EXC = 0.05, ten times the canonical
  0.006) only for tau_g below some tau_g_engage ~ 25 t.u., while the EVENT
  ITSELF lasts ~65-100 t.u.: the architectural constraint "the loop must be
  faster than what it guards" acquires the number tau_g_engage.
  FALSIFIER: the episodic OUTCOME (G_end, stuck) changes materially
  (|dG_end| > 0.05) at some tau_g on the grid — that would REFUTE "the loop
  is inert episodically" and mean the canonical tau_g = 200 sits above a
  real outcome transition; OR the excursion never grows with 1/tau_g
  (refutes the mechanism reading of the g equation).

P3 — CHRONIC vs EPISODIC (same tau_g grid under sustained drive).
  Two chronic arms, both from canonical ICs: CHRONIC-HI (a_hold 0.9 held
  from t = 100 to the horizon T = 600 — the literal sustained version of the
  episodic trigger) and CHRONIC-LIMINAL (a_hold 0.30, just above the
  border-collision critical eps_c = 0.2652 of section 4.9 — the loop's BEST
  case, chosen because just above the fold the healthy branch is gone but
  the drift toward collapse is slow, keeping |dE/dt| above the 0.002
  deadband for long stretches so a slow g has time to accumulate).
  PREDICTION (inert episodically, load-bearing chronically): there is a
  chronic arm where the tau_g sweep moves the outcome — LOAD-BEARING
  pre-registered as |G_end(tau_g = 6.25) - G_end(tau_g = 800)| > 0.2 in a
  chronic arm while the same delta stays < 0.05 episodically.  That would
  be the cleaner statement replacing "the loop is inert", and would explain
  why the model carries a slow gain at all.
  FALSIFIER: no tau_g on the grid moves ANY chronic arm's G_end by more
  than 0.05 — then the loop is outcome-inert at EVERY timescale in this
  realization, the prediction is refuted, and the measured g-ceiling
  pi*max|dE/dt|/mu is reported as the mechanism (the excursion is capped by
  the mu/pi ratio, not by tau_g, so no timescale can make the loop matter).

METHOD — nothing is rebuilt: the frozen RHS is dpdr.model.deriv via
dpdr.integrate.simulate (READ-ONLY), the canonical scenario is
dpdr.events.failure_schedule, chronic arms are Schedule({"a_hold": ...}),
classification uses dpdr.metrics.{is_stuck, classify, summarize}.  Phase
decomposition (pre-registered, blind to output): onset = t = 100 exactly;
crossing t_c = first grid sample with c > c_star, for BOTH c_star = 0
(switch strictly on, i.e. E > Theta_eff) and c_star = 0.5 (half-height,
used as the runaway/crush PHASE BOUNDARY); runaway share = (G(onset) -
G(t_c50)) / (G(onset) - G_end); E plateau = mean E over [t_c50 - 10,
t_c50); setpoint share over the final 15 t.u. before crossing = mean|dTheta_eff/dt|
/ (mean|dE/dt| + mean|dTheta_eff/dt|) on the dt grid.  Damage-split
reference is the run's own G at t = 100.  Every anchor number quoted in the
handoff (0.885/0.145/0.051, t = 100/165/200, split 89/11, plateau 0.526)
is REPRODUCED in part 0 before any sweep runs; if it fails to reproduce the
script aborts.  Grid dt = 0.05 (crossing resolution), T = 1200 episodic,
600 chronic.  Caches cache/exp20_*.npz; figure figs/f21_sweeps.png verified
programmatically only.

Usage: .venv/bin/python experiments/exp20_sweeps.py [canon|tauS|tauG|chronic|fig|verify|all]
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
FIGDIR = os.path.join(ROOT, "figs")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dpdr.events import Schedule, failure_schedule
from dpdr.integrate import simulate
from dpdr.metrics import classify, is_stuck, summarize
from dpdr.model import Params

P = Params()
DT = 0.05                # integration output grid = crossing resolution
EP_T0, EP_T1 = 100.0, 200.0
T_EPIS = 1200.0
TAU_S_GRID = [25.0, 50.0, 100.0, 150.0, 200.0, 400.0, 800.0]
TAU_G_GRID = [6.25, 12.5, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]
CHRONIC_ARMS = [("hi", 0.90, 600.0), ("lim", 0.30, 600.0)]
ENGAGE_EXC = 0.05        # pre-registered engagement threshold (P2)
OUTCOME_TOL = 0.05       # pre-registered outcome-materiality tolerance (P2)
CHRONIC_TOL = 0.20       # pre-registered load-bearing threshold (P3)
SETPOINT_WIN = 15.0      # setpoint-share window before the crossing

# canonical anchors from handoff-selfreg-three-phases (compared, not trusted)
ANCH = dict(G_onset=0.885, G_b=0.145, G_end=0.051, t_b=165.0,
            runaway_share=0.89, E_plateau=0.526, g_excursion=0.006,
            t_cross=65.0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------ phase decomposition
def first_cross(t, c, c_star, t_from=EP_T0):
    """First grid time >= t_from with c > c_star (None if never)."""
    idx = np.where((t >= t_from - 1e-9) & (c > c_star))[0]
    return None if idx.size == 0 else float(t[idx[0]])


def crossing_time(t, c, c_star, t_from=EP_T0):
    tc = first_cross(t, c, c_star, t_from)
    return np.nan if tc is None else tc - t_from


def phase_row(sol, p) -> dict:
    """Pre-registered per-run measurements (see docstring METHOD)."""
    t, G, E, c, S, g, Teff = (sol["t"], sol["G"], sol["E"], sol["c"],
                              sol["S"], sol["g"], sol["Theta_eff"])
    i0 = int(np.argmin(np.abs(t - EP_T0)))
    G_onset = float(G[i0])
    t_c0 = crossing_time(t, c, 0.0)
    t_c50 = crossing_time(t, c, 0.5)
    G_end = float(G[-1])
    E_plateau = np.nan
    setpoint_share = np.nan
    G_b = np.nan
    if not np.isnan(t_c50):
        tb = EP_T0 + t_c50
        ib = int(np.argmin(np.abs(t - tb)))
        G_b = float(G[ib])
        w = (t >= tb - 10.0) & (t < tb)
        if w.sum() > 3:
            E_plateau = float(E[w].mean())
        w2 = (t >= tb - SETPOINT_WIN) & (t < tb)
        if w2.sum() > 3:
            dE = np.abs(np.gradient(E[w2], t[w2])).mean()
            dT = np.abs(np.gradient(Teff[w2], t[w2])).mean()
            if dE + dT > 0:
                setpoint_share = float(dT / (dE + dT))
    runaway_share = (np.nan if (np.isnan(t_c50) or G_onset == G_end)
                     else (G_onset - G_b) / (G_onset - G_end))
    g_exc = float(g.max() - g[0])
    dEdt = np.abs(np.gradient(E, t))
    return dict(
        t_c0=t_c0, t_c50=t_c50, G_onset=G_onset, G_b=G_b, G_end=G_end,
        G_min=float(G.min()), E_plateau=E_plateau,
        setpoint_share=setpoint_share, runaway_share=runaway_share,
        g_max=float(g.max()), g_end=float(g[-1]), g_excursion=g_exc,
        dEdt_max=float(dEdt.max()),
        deadband_frac=float((dEdt > P.dEdt_ref).mean()),
        stuck=float(is_stuck(sol, p)), regime=classify(sol, p),
        t_collapse=float(summarize(sol, p)["t_collapse"])
        if summarize(sol, p)["t_collapse"] is not None else np.nan,
    )


def fit_line(x, y):
    """OLS slope/intercept/R2 on finite points; NaN-safe."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan, np.nan, int(m.sum())
    b, a = np.polyfit(x[m], y[m], 1)
    r = float(np.corrcoef(x[m], y[m])[0, 1])
    return float(b), float(a), r**2, int(m.sum())


def fit_loglog(x, y):
    """OLS slope/R2 in log-log on finite positive points."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if m.sum() < 2:
        return np.nan, np.nan, int(m.sum())
    b, a = np.polyfit(np.log(x[m]), np.log(y[m]), 1)
    r = float(np.corrcoef(np.log(x[m]), np.log(y[m]))[0, 1])
    return float(b), float(a), r**2, int(m.sum())


# --------------------------------------------------------------- part 0
def part_canon() -> dict:
    """Reproduce the canonical trajectory and its three-phase decomposition
    from the frozen model; abort if an anchor disagrees materially."""
    log("part0 START: canonical run, T="
        f"{T_EPIS:g}, dt={DT:g}, failure_schedule (a_hold 0.9 [100,200) + "
        "pulse 0.5 [100,160))")
    t0 = time.time()
    sol = simulate(P, failure_schedule(), T_EPIS, dt=DT)
    log(f"part0 integrated in {time.time() - t0:.1f}s "
        f"(nfev={sol['nfev']}, methods={sol['methods']})")
    row = phase_row(sol, P)
    i0 = int(np.argmin(np.abs(sol["t"] - EP_T0)))
    i165 = int(np.argmin(np.abs(sol["t"] - 165.0)))
    log(f"part0 G(onset)={row['G_onset']:.3f} (anchor {ANCH['G_onset']}), "
        f"G(at c50 boundary)={row['G_b']:.3f} (anchor {ANCH['G_b']}), "
        f"G_end={row['G_end']:.3f} (anchor {ANCH['G_end']}), "
        f"G(165)={sol['G'][i165]:.3f}")
    log(f"part0 t_c0={row['t_c0']:.2f}, t_c50={row['t_c50']:.2f} "
        f"(anchor boundary t~{ANCH['t_b']:g}), "
        f"E_plateau={row['E_plateau']:.3f} (anchor {ANCH['E_plateau']}), "
        f"runaway_share={row['runaway_share']:.3f} "
        f"(anchor {ANCH['runaway_share']}), "
        f"g_excursion={row['g_excursion']:.4f} "
        f"(anchor {ANCH['g_excursion']}), stuck={row['stuck']:.0f}, "
        f"regime={row['regime']}")
    checks = [
        ("G_onset", row["G_onset"], ANCH["G_onset"], 0.02),
        ("G_end", row["G_end"], ANCH["G_end"], 0.02),
        ("E_plateau", row["E_plateau"], ANCH["E_plateau"], 0.02),
        ("runaway_share", row["runaway_share"], ANCH["runaway_share"], 0.05),
        ("g_excursion", row["g_excursion"], ANCH["g_excursion"], 0.005),
    ]
    bad = [(k, v, a) for k, v, a, tol in checks if abs(v - a) > tol]
    if bad:
        log(f"part0 ABORT: anchor reproduction failed for {bad}")
        sys.exit(1)
    log("part0 anchors reproduced within tolerance — proceeding is licensed")
    w = (sol["t"] >= 80.0) & (sol["t"] <= 300.0)
    np.savez(os.path.join(CACHE, "exp20_canon.npz"),
             t=sol["t"][w], a=sol["a"][w], G=sol["G"][w], D=sol["D"][w],
             S=sol["S"][w], g=sol["g"][w], c=sol["c"][w], E=sol["E"][w],
             Theta_eff=sol["Theta_eff"][w],
             **{k: np.array(v) for k, v in row.items()})
    log("part0 DONE (exp20_canon.npz)")
    return dict(sol=sol, row=row)


# --------------------------------------------------------------- part 1
def part_tauS() -> dict:
    """P1: sweep tau_S on the canonical episodic schedule."""
    log(f"part1 START: tau_S sweep {TAU_S_GRID}, T={T_EPIS:g} (P1: "
        "crossing ~ max(t_E, 0.71*tau_S); crossing LOST for tau_S >= ~140; "
        "falsifier = crossing tau_S-independent or collapse at 800)")
    rows, traces = [], []
    for v in TAU_S_GRID:
        t0 = time.time()
        p = replace(P, tau_S=v)
        sol = simulate(p, failure_schedule(), T_EPIS, dt=DT)
        row = phase_row(sol, p)
        row["tau_S"] = v
        rows.append(row)
        w = (sol["t"] >= 80.0) & (sol["t"] <= 300.0) & (
            (np.arange(sol["t"].size) % 4) == 0)   # ds to 0.2
        traces.append((sol["t"][w], sol["G"][w], sol["E"][w],
                       sol["Theta_eff"][w], sol["c"][w]))
        log(f"part1 tau_S={v:6.1f}: t_c0={row['t_c0']:7.2f} "
            f"t_c50={row['t_c50']:7.2f} G_end={row['G_end']:.4f} "
            f"stuck={row['stuck']:.0f} regime={row['regime']:>13s} "
            f"run_share={row['runaway_share']:.3f} "
            f"E_pl={row['E_plateau']:.4f} "
            f"sp_share={row['setpoint_share']:.3f} "
            f"({time.time() - t0:.1f}s)")
    ks = list(rows[0].keys())
    arr = {k: np.array([r[k] for r in rows]) for k in ks}
    tt = traces[0][0]
    np.savez(os.path.join(CACHE, "exp20_tauS.npz"),
             **arr, trace_t=tt,
             trace_G=np.array([tr[1] for tr in traces]),
             trace_E=np.array([tr[2] for tr in traces]),
             trace_Teff=np.array([tr[3] for tr in traces]),
             trace_c=np.array([tr[4] for tr in traces]))
    # fits: linear in tau_S where the crossing exists; log-log raw
    b, a, r2, n = fit_line(arr["tau_S"], arr["t_c50"])
    bg, ag, r2g, _ = fit_loglog(arr["tau_S"], arr["t_c50"])
    log(f"part1 FIT t_c50 = {b:.4f}*tau_S + {a:.2f}  (R2={r2:.4f}, n={n})")
    log(f"part1 FIT log-log slope = {bg:.3f} (R2={r2g:.4f}) — includes the "
        "on-top offset; slope<1 expected at small tau_S from the E-rise "
        "floor, NOT a refutation; refutation is independence (slope~0)")
    log("part1 DONE (exp20_tauS.npz)")
    return dict(rows=rows, arr=arr)


# --------------------------------------------------------------- part 2
def part_tauG() -> dict:
    """P2: sweep tau_g on the canonical episodic schedule."""
    log(f"part2 START: tau_g sweep {TAU_G_GRID}, T={T_EPIS:g} episodic "
        f"(P2: excursion ~ 1/tau_g up to the pi*max|dE/dt|/mu ceiling; "
        f"engage < {ENGAGE_EXC}; falsifier = |dG_end| > {OUTCOME_TOL})")
    rows = []
    for v in TAU_G_GRID:
        t0 = time.time()
        p = replace(P, tau_g=v)
        sol = simulate(p, failure_schedule(), T_EPIS, dt=DT)
        row = phase_row(sol, p)
        row["tau_g"] = v
        rows.append(row)
        log(f"part2 tau_g={v:6.2f}: G_end={row['G_end']:.4f} "
            f"G_min={row['G_min']:.4f} stuck={row['stuck']:.0f} "
            f"regime={row['regime']:>13s} t_c50={row['t_c50']:7.2f} "
            f"exc={row['g_excursion']:.4f} g_end={row['g_end']:.4f} "
            f"({time.time() - t0:.1f}s)")
    ks = list(rows[0].keys())
    arr = {k: np.array([r[k] for r in rows]) for k in ks}
    np.savez(os.path.join(CACHE, "exp20_tauG_epi.npz"), **arr)
    bg, ag, r2g, _ = fit_loglog(arr["tau_g"], arr["g_excursion"])
    g_ceiling = np.pi * (arr["dEdt_max"].max() - P.dEdt_ref) / P.mu
    log(f"part2 FIT log-log excursion slope = {bg:.3f} (R2={r2g:.4f}); "
        f"measured quasi-steady ceiling pi*max|dE/dt|/mu = {g_ceiling:.3f}")
    log("part2 DONE (exp20_tauG_epi.npz)")
    return dict(rows=rows, arr=arr)


# --------------------------------------------------------------- part 3
def part_chronic() -> dict:
    """P3: the same tau_g grid under sustained (chronic) inward drive."""
    log(f"part3 START: tau_g sweep under chronic arms {CHRONIC_ARMS} "
        f"(P3: load-bearing = |dG_end| > {CHRONIC_TOL} in a chronic arm "
        f"while < {OUTCOME_TOL} episodically; falsifier = nothing moves)")
    out = {}
    for name, ah, T in CHRONIC_ARMS:
        rows = []
        for v in TAU_G_GRID:
            t0 = time.time()
            p = replace(P, tau_g=v)
            sch = Schedule({"a_hold": [(EP_T0, T, ah)]})
            sol = simulate(p, sch, T, dt=DT)
            row = phase_row(sol, p)
            row["tau_g"] = v
            rows.append(row)
            log(f"part3 {name} tau_g={v:6.2f}: G_end={row['G_end']:.4f} "
                f"G_min={row['G_min']:.4f} stuck={row['stuck']:.0f} "
                f"regime={row['regime']:>13s} t_c50={row['t_c50']:7.2f} "
                f"exc={row['g_excursion']:.4f} "
                f"db_frac={row['deadband_frac']:.3f} "
                f"({time.time() - t0:.1f}s)")
        ks = list(rows[0].keys())
        arr = {k: np.array([r[k] for r in rows]) for k in ks}
        np.savez(os.path.join(CACHE, f"exp20_tauG_{name}.npz"), **arr)
        out[name] = arr
        d = abs(arr["G_end"][0] - arr["G_end"][-1])
        bg, _, r2g, _ = fit_loglog(arr["tau_g"], arr["g_excursion"])
        log(f"part3 {name} DONE: |G_end(6.25) - G_end(800)| = {d:.4f} "
            f"(load-bearing needs > {CHRONIC_TOL}); excursion log-log "
            f"slope {bg:.3f} (R2={r2g:.4f})")
    return out


# --------------------------------------------------------------- sharpen
def part_probe() -> None:
    """POST-GRID SHARPENING (both strengthen already-resolved outcomes; the
    registered grids are untouched): (i) bisect tau_S_crit, the boundary
    between collapse and no-collapse on the canonical schedule (the P1
    prediction 'crossing lost for tau_S >= ~140' put 150 on the safe side,
    100 on the collapse side); (ii) verify E's pre-crossing trajectory is
    tau_S-free (E's RHS has no tau_S once c = 0), which is the setpoint-
    drain mechanism in its cleanest form."""
    log("probe START: (i) bisect tau_S_crit in (100, 150] @0.05")
    lo, hi = 100.0, 150.0     # collapse(100)=True, collapse(150)=False

    def collapses(v: float) -> bool:
        sol = simulate(replace(P, tau_S=v), failure_schedule(), T_EPIS, dt=DT)
        return bool(sol["G"][-1] < 0.1)

    for _ in range(12):
        mid = round(round(0.5 * (lo + hi) / 0.05) * 0.05, 2)
        if collapses(mid):
            lo = mid
        else:
            hi = mid
        if hi - lo <= 0.051:
            break
    log(f"probe tau_S_crit: collapse at tau_S={lo:.2f}, healthy at "
        f"{hi:.2f} (boundary {0.5 * (lo + hi):.2f})")
    lo_tauS, hi_tauS = lo, hi
    # (ii) E tau_S-freedom pre-crossing: max |E(tauS=25) - E(tauS=100)|
    # over the window before the FIRST crossing of the faster run
    # (t_c0(tau_S=25) = 34.50; window ends strictly before t = 134.50)
    d = np.load(os.path.join(CACHE, "exp20_tauS.npz"))
    t, E25, E100 = d["trace_t"], d["trace_E"][0], d["trace_E"][2]
    tc_fast = float(d["t_c0"][0])
    w = (t >= EP_T0) & (t < EP_T0 + tc_fast - 0.10)
    dev = float(np.max(np.abs(E25[w] - E100[w])))
    log(f"probe E tau_S-freedom: max|E(tauS=25)-E(tauS=100)| before the "
        f"tau_S=25 crossing = {dev:.2e} (0 confirms E carries no tau_S "
        "pre-switch)")
    # (iii) THE CEILING COUNTERFACTUAL (why P3 failed): grant the loop its
    # full measured peak excursion (g = g0 + exc(tau_g=6.25) = 0.5845)
    # from the instant of ONSET and hold it there for the whole episode
    # (tau_g = 8000 freezes g to within mu*(g-g0)*T/tau_g ~ 0.003) — a
    # bound the actual slow dynamics cannot approach.  If the outcome is
    # still the canonical collapse, the loop is inert BY CEILING, not by
    # timescale.
    gpk = float(P.g0 + np.load(
        os.path.join(CACHE, "exp20_tauG_epi.npz"))["g_excursion"][0])
    sol = simulate(replace(P, tau_g=8000.0, g_init=gpk),
                   failure_schedule(), T_EPIS, dt=DT)
    dg = abs(float(sol["g"][-1]) - gpk)
    log(f"probe ceiling counterfactual: g held {gpk:.4f}->"
        f"{sol['g'][-1]:.4f} (drift {dg:.4f}) from onset -> G_end="
        f"{sol['G'][-1]:.4f} vs canonical {ANCH['G_end']:.3f}")
    # (iv) bisect the held-g escape boundary from the canonical ICs: the
    # gap between it and the reachable ceiling gpk IS the refutation of
    # P3, measured (compare the paper's static g0 critical 0.7029,
    # bisected from settled ICs).
    lo, hi = gpk, 0.80

    def held_stuck(gv: float) -> bool:
        s = simulate(replace(P, tau_g=8000.0, g_init=gv),
                     failure_schedule(), T_EPIS, dt=DT)
        return bool(s["G"][-1] < 0.1)

    for _ in range(10):
        mid = round(0.5 * (lo + hi), 4)
        if mid in (lo, hi):
            break
        if held_stuck(mid):
            lo = mid
        else:
            hi = mid
    log(f"probe held-g boundary: stuck at g={lo:.4f}, escape at "
        f"g={hi:.4f} -> {0.5 * (lo + hi):.4f} (reachable ceiling "
        f"{gpk:.4f} sits {0.5 * (lo + hi) - gpk:.4f} below it)")
    np.savez(os.path.join(CACHE, "exp20_probe.npz"),
             tau_S_crit=0.5 * (lo_tauS + hi_tauS), tau_S_lo=lo_tauS,
             tau_S_hi=hi_tauS, E_taufree_dev=dev, g_peak=gpk,
             G_end_ceiling=float(sol["G"][-1]), g_hold_crit=0.5 * (lo + hi),
             g_hold_lo=lo, g_hold_hi=hi)
    log("probe DONE (exp20_probe.npz)")


# --------------------------------------------------------------- figure
def part_fig(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log("fig START: figs/f21_sweeps.png (4 panels)")
    tauS = res["tauS"]["arr"]
    epi = res["tauG"]["arr"]
    hi, lim = res["hi"], res["lim"]
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.5))
    fs = 8.5

    ax = axs[0, 0]
    m = np.isfinite(tauS["t_c50"])
    ax.loglog(tauS["tau_S"][m], tauS["t_c50"][m], "o-", label="t_c(c=0.5)")
    m0 = np.isfinite(tauS["t_c0"])
    ax.loglog(tauS["tau_S"][m0], tauS["t_c0"][m0], "s--", label="t_c(c>0)")
    b, a, r2, _ = fit_line(tauS["tau_S"], tauS["t_c50"])
    xs = np.array([20.0, 160.0])
    ax.plot(xs, b * xs + a, "k:", alpha=0.6,
            label=f"fit {b:.3f}·τ$_S$+{a:.1f} (R²={r2:.3f})")
    ax.set_xlabel("τ$_S$ [t.u.]")
    ax.set_ylabel("onset → crossing [t.u.]")
    ax.set_title("(a) τ$_S$ sweep: crossing time", fontsize=10)
    ax.legend(fontsize=fs)
    ax.grid(True, which="both", alpha=0.25)

    ax = axs[0, 1]
    ax.semilogx(tauS["tau_S"], tauS["runaway_share"], "o-", color="C0",
                label="runaway share of G-loss")
    ax.semilogx(tauS["tau_S"], tauS["setpoint_share"], "s--", color="C1",
                label="setpoint share, last 15 t.u.")
    ax.axhline(0.5, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("τ$_S$ [t.u.]")
    ax.set_ylabel("share")
    ax.set_title("(b) damage split & setpoint dominance", fontsize=10)
    ax.legend(fontsize=fs)
    ax.grid(True, which="both", alpha=0.25)

    ax = axs[1, 0]
    ax.loglog(epi["tau_g"], epi["g_excursion"], "o-", color="C3",
              label="episodic")
    ax.loglog(hi["tau_g"], hi["g_excursion"], "s--", color="C2",
              label="chronic a=0.9")
    ax.loglog(lim["tau_g"], lim["g_excursion"], "^-", color="C4",
              label="chronic a=0.30")
    ax.axhline(ENGAGE_EXC, color="k", lw=0.8, ls=":",
               label=f"engage = {ENGAGE_EXC}")
    bg, _, _, _ = fit_loglog(epi["tau_g"], epi["g_excursion"])
    ax.set_xlabel("τ$_g$ [t.u.]")
    ax.set_ylabel("peak g excursion")
    ax.set_title(f"(c) loop engagement (episodic slope {bg:.2f})", fontsize=10)
    ax.legend(fontsize=fs)
    ax.grid(True, which="both", alpha=0.25)

    ax = axs[1, 1]
    ax.semilogx(epi["tau_g"], epi["G_end"], "o-", color="C3",
                label="episodic")
    ax.semilogx(hi["tau_g"], hi["G_end"], "s--", color="C2",
                label="chronic a=0.9")
    ax.semilogx(lim["tau_g"], lim["G_end"], "^-", color="C4",
                label="chronic a=0.30")
    ax.axhline(0.1, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("τ$_g$ [t.u.]")
    ax.set_ylabel("G(T)")
    ax.set_title("(d) outcome vs τ$_g$: chronic vs episodic", fontsize=10)
    ax.legend(fontsize=fs)
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("exp20 — setpoint-drain crossing (τ$_S$) and corrective-loop "
                 "engagement (τ$_g$): frozen model, canonical failure "
                 "schedule", fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "f21_sweeps.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    log(f"fig DONE ({out})")


def part_verify() -> None:
    """Programmatic figure verification only (never opened as an image)."""
    from PIL import Image
    out = os.path.join(FIGDIR, "f21_sweeps.png")
    im = Image.open(out).convert("L")
    w, h = im.size
    x = np.asarray(im, float)
    ink = (x < 0.95 * 255)
    frac = ink.mean()
    quads = [ink[: h // 2, : w // 2], ink[: h // 2, w // 2:],
             ink[h // 2:, : w // 2], ink[h // 2:, w // 2:]]
    qf = [q.mean() for q in quads]
    ok = (w == int(11.5 * 140)) and (h == int(8.5 * 140)) and 0.01 < frac < 0.9
    ok = ok and all(q > 0.003 for q in qf)
    log(f"verify FIGURE {out}: size {w}x{h}px, ink fraction {frac:.4f}, "
        f"quadrant ink {[f'{q:.4f}' for q in qf]} -> "
        f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        sys.exit(1)


# --------------------------------------------------------------- driver
def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    todo = (["canon", "tauS", "tauG", "chronic", "probe", "fig", "verify"]
            if which == "all" else [which])
    res: dict = {}
    for part in todo:
        if part == "canon":
            res["canon"] = part_canon()
        elif part == "tauS":
            res["tauS"] = part_tauS()
        elif part == "tauG":
            res["tauG"] = part_tauG()
        elif part == "chronic":
            res.update(part_chronic())
        elif part == "probe":
            part_probe()
        elif part == "fig":
            if not res:
                res["tauS"] = dict(arr=np.load(
                    os.path.join(CACHE, "exp20_tauS.npz")))
                res["tauG"] = dict(arr=np.load(
                    os.path.join(CACHE, "exp20_tauG_epi.npz")))
                res["hi"] = np.load(os.path.join(CACHE, "exp20_tauG_hi.npz"))
                res["lim"] = np.load(os.path.join(CACHE, "exp20_tauG_lim.npz"))
            part_fig(res)
        elif part == "verify":
            part_verify()
        else:
            print(f"unknown part {part}")
            sys.exit(2)
    log("exp20 ALL DONE")


if __name__ == "__main__":
    main()
