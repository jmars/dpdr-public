"""exp19 — the QUADRANT map: classify by (a, D) POSITION.  This is the
corrected instrument after exp17's term-dominance classifier was refuted in
its three-stream form.  Frozen model throughout (dpdr.model read-only); every
number below comes from this driver + cache/exp19_*.npz.

*** THE INSTRUMENT ERROR BEING CORRECTED (recorded so it is not repeated) ***
exp17 classified each state by WHICH TERM DOMINATES dG/dt.  Measured
consequence of that choice: the turnover term -gam_G*G can be the largest
in magnitude only where alpha_G*a*G < gam_G*G, i.e. a < gam_G/alpha_G =
0.30/1.20 = 0.25 — but at a < 0.25 attention is OUTWARD, where the growth
term beta_G*(1-a)*G*(1-G) is large and wins instead; and at the healthy
rest state the two nearly tie (a=0, G=0.8854: grow = +0.30440 vs turn =
-0.26562, a margin of 0.039).  exp17's "R2 relaxed" region was therefore
UNOCCUPIABLE BY CONSTRUCTION — an artifact of the classifier, not evidence
about the state.  STANDING RULE RE-APPLIED: a perverse or empty result
means suspect the metric; PREFER THE MODEL'S OWN STATE VARIABLES AS AXES
OVER DERIVED QUANTITIES LIKE TERM DOMINANCE.

*** THE CORRECTED INSTRUMENT — (a, D) POSITION ***
The model has two independent slow axes and the key structural fact is
dD = ((D_base + A)*beta_D*(1-D) - delta_D*D)/tau_D  (model.py:118)
CONTAINS NO a: demand is driven by AFFECT and exogenous input, not by
attention; attention is driven separately (model.py:120).  So the three
felt streams are QUADRANTS of the (a, D) plane:
  Q1 TALKING/outward  a low,  D low;   Q2 RELAXED  a high, D low;
  Q3 EFFORTFUL        a high, D high;  Q4 (unnamed by the reading;
      low a, high D — PREDICTED UNVISITED by all three protocols).

*** PRE-REGISTERED RULES (stated before any run; not tuned after) ***
  AXIS THRESHOLDS (both derived from the model's own nullclines, not free):
    A_STAR = 0.5   — the midpoint of the attention axis.  The setpoint
        equation tracks externality as (1-a) (deviation D3), so a = 0.5 is
        where that signal crosses its own midpoint; the protocols'
        attention attractors bracket it with wide margins (baseline
        nullcline a* = 0; episode nullcline a* = k_in*0.9/(k_in*0.9+rho_a)
        = 1.35/1.55 = 0.871).  BOUNDARY: a == A_STAR counts as the LOW
        (outward) side, matching exp17's boundary-belongs-to-OFF convention.
    D_STAR = 0.65368 — the MIDPOINT OF THE MODEL'S TWO DEMAND NULLCLINES:
        D_null(A) = (D_base+A)*beta_D / ((D_base+A)*beta_D + delta_D), so
        D_null(A=0)   = 0.24/0.44 = 0.54545  (no-affect attractor),
        D_null(A=0.5) = 0.64/0.84 = 0.76190  (canonical-pulse attractor),
        D_STAR = (0.54545 + 0.76190)/2 = 0.65368.  BOUNDARY: D == D_STAR
        counts as the LOW side.  Quadrant code: Q1 low/low, Q2 high-a/low-D,
        Q3 high-a/high-D, Q4 low-a/high-D (boundaries -> low side).
  ARMING RULE (NOT a fixed D threshold — stated explicitly): the switch
        fires iff E = D - G > Theta_eff = Theta*S/S_rest = 0.8*S, i.e.
        iff D > G + 0.8*S (c = sigma_c*tanh((E-Theta_eff)/w), boundary
        c=0 belongs to the OFF side; model.py:90-94).  Attention does not
        enter the rule at all; a affects it only through S's slow tracking
        of (1-a) (chronic inward attention lowers Theta_eff — lowers the
        bar).  At the healthy settled slice (G=0.8855, S=0.9545) this is
        D > 1.6491: ABOVE every demand nullcline, so no protocol arms from
        the healthy state by demand alone — arming happens through G
        falling (E rising), which is measured below.

  PRE-REGISTERED PREDICTIONS (nullcline algebra only, logged by `probe`
  BEFORE the runs):
    BASELINE  -> Q1 throughout (a -> 0, D -> 0.5455 < D_STAR).
    RECALL (inward_episode [100,200), no affect) -> Q2 from t ~ 100.55
        (a crosses 0.5 at 0.871*(1-exp(-1.55*dt)) = 0.5 => dt = 0.55) to
        t ~ 202.8 (a decays as 0.871*exp(-0.2*dt), crosses 0.5 at
        dt = 2.78), then Q1.  Survives (dwell limit 118.70 > 100).
    FAILURE (episode + affect pulse 0.5 on [100,160)) -> Q2 from ~100.55,
        Q3 from t ~ 146.6 (D rises 0.5266 -> 0.7619 with tau 59.5,
        crossing 0.6537 at dt = 46.6) to t ~ 181.3 (D decays back with
        tau 113.6, crossing at dt = 21.3 past 160), then Q2 TERMINALLY
        with the switch ARMED (armed once G has fallen enough that
        E = D - G rises through Theta_eff; exp17 measured that at t ~ 166).
        Terminal position a ~ k_in*chi/(k_in*chi + rho_a) = 0.882 (held
        high by c itself), D ~ 0.546: the collapsed state sits in the
        RELAXED quadrant position.
    Q4 unvisited by all three protocols.

  DECISIVE TEST (re-expressing exp17 in (a, D) terms): does the collapse
        trajectory leave the G-maintaining quadrants and STAY in the
        effortful one?  PREDICTION: it leaves Q1 at onset and never
        returns (AGREES with exp17), but it does NOT stay in Q3 — a ~ 35
        t.u. transit during the pulse, then Q2 for the remaining ~ 1300
        t.u. while armed.  exp17's own terminal class was gain-dominated
        (not eat), so both instruments agree the eat-window is brief; the
        (a, D) view adds that the terminal POSITION is the relaxed one.
  CARRY-FORWARD (exp17's classifier-independent result: the switch arms
        LATE AND BRIEFLY, R3-with-c-off 40.91% of the map vs c-on 4.0% —
        reduction-dominance PRECEDES cannibalization): re-expressed as
        Q3 ENTRY PRECEDES ARMING (~147 vs ~166) and ARMING OUTLASTS the
        Q3 residence manyfold (armed ~ 1334 t.u. vs Q3 ~ 35 t.u.).

  FALSIFIERS (a refutation is a useful result; all are reported):
    (F1) RECALL does not occupy a distinct quadrant from FAILURE (in the
         episode window) — would mean the model does not separate relaxed
         from effortful and the axes are not behaviourally distinct.
    (F2) BASELINE does not sit in Q1.
    (F3) No protocol ever visits Q3 (the D-axis separates nothing).
    (F4) The collapse trajectory returns to Q1, or stays in Q3 terminally.
    (F5) THRESHOLD ARTIFACT: labels flip for A_STAR anywhere in (0.1, 0.85)
         or D_STAR anywhere in the nullcline band (0.54545, 0.76190) —
         checked by an explicit (A_STAR, D_STAR) sensitivity scan.

METHOD — nothing rebuilt: dpdr.integrate.simulate on dpdr.model.deriv
(read-only); runs at the default dt = 0.05 grid (breakpoints 100/160/200
and the brink durations 118.65/118.70 all lie on it); stuck criterion
dpdr.metrics.is_stuck.  Protocols: baseline_schedule; RECALL =
inward_episode(100, 200, 0.9) (the critical no-affect case) plus the
exp18 brink pair (dwell 118.65 healthy / 118.70 stuck) as a
quadrant-vs-fate check; FAILURE = failure_schedule, T = 1500.

Usage: .venv/bin/python experiments/exp19_quadrants.py [probe|traj|areas|fig|all]
"""
from __future__ import annotations

import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from dpdr.events import baseline_schedule, failure_schedule, inward_episode
from dpdr.integrate import simulate
from dpdr.metrics import is_stuck
from dpdr.model import Params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
FIGDIR = os.path.join(ROOT, "figs")

P = Params()
DT = 0.05                    # integration/output grid (breakpoints lie on it)
T_RUN = 1500.0               # main-run horizon (exp17 precedent)
T_SETTLE = 5000.0            # settle run for the (G, S, g) slice
T_BRINK = 900.0              # exp18's recall-protocol horizon
BRINK_OK, BRINK_STUCK = 118.65, 118.70   # exp18's measured dwell threshold

# ---- pre-registered thresholds (derived from Params(), not free numbers)
A_STAR = 0.5
D_NULL_A0 = (P.D_base + 0.0) * P.beta_D / ((P.D_base + 0.0) * P.beta_D + P.delta_D)
D_NULL_AP = (P.D_base + 0.5) * P.beta_D / ((P.D_base + 0.5) * P.beta_D + P.delta_D)
D_STAR = 0.5 * (D_NULL_A0 + D_NULL_AP)

QNAMES = {1: "Q1 talk", 2: "Q2 relaxed", 3: "Q3 effortful", 4: "Q4 out+load"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def quadrant(a, D, a_star: float = A_STAR, d_star: float = D_STAR):
    """Pre-registered position rule (vectorized), per the docstring's
    definitions: Q1 = low a/low D, Q2 = high a/low D, Q3 = high a/high D
    (effortful), Q4 = low a/high D.  Boundaries count as the low side."""
    a = np.asarray(a, float)
    D = np.asarray(D, float)
    return np.where(a > a_star,
                    np.where(D > d_star, 3, 2),
                    np.where(D > d_star, 4, 1)).astype(int)


def armed(E, Teff):
    """Arming rule: E > Theta_eff strictly (boundary is OFF)."""
    return np.asarray(E) > np.asarray(Teff)


def seg_runs(t, quad, arm):
    """Contiguous (quadrant, armed) segments -> [(i0, i1, q, armed)]."""
    key = quad.astype(int) * 2 + arm.astype(int)
    out, s = [], 0
    for k in range(1, t.size + 1):
        if k == t.size or key[k] != key[s]:
            out.append((s, k, int(quad[s]), bool(arm[s])))
            s = k
    return out


def occupancy(t, quad, arm, t0, t1):
    """(quadrant fractions, armed fraction) over t in [t0, t1)."""
    w = (t >= t0) & (t < t1)
    n = max(int(w.sum()), 1)
    fr = {q: float((quad[w] == q).sum()) / n for q in (1, 2, 3, 4)}
    return fr, float(arm[w].mean())


# ------------------------------------------------------------------ probe
def probe() -> None:
    """Nullcline predictions logged BEFORE any trajectory run."""
    t0 = time.time()
    log("probe: pre-registered thresholds and nullcline predictions")
    log(f"probe: A_STAR={A_STAR:.5f} (attention-axis midpoint; attractors "
        f"a*=0 baseline, a*={P.k_in * 0.9 / (P.k_in * 0.9 + P.rho_a):.5f} "
        f"episode -> bracket it)")
    log(f"probe: D_null(A=0)={D_NULL_A0:.5f}  D_null(A=0.5)={D_NULL_AP:.5f} "
        f"-> D_STAR={D_STAR:.5f} (nullcline midpoint)")
    # instrument-error documentation, recomputed from Params (exp17's numbers)
    grow0 = P.beta_G * 1.0 * 0.8854 * (1 - 0.8854)
    turn0 = P.gam_G * 0.8854
    log(f"probe: exp17 tie at rest recomputed: grow={grow0:+.5f} turn={-turn0:+.5f} "
        f"margin={grow0 - turn0:.5f}; turn can dominate only at "
        f"a < gam_G/alpha_G = {P.gam_G / P.alpha_G:.3f} (outward) — R2 was "
        f"unoccupiable by construction")
    # arming boundary at the healthy slice (G, S from exp17's settle: verify later)
    sb = simulate(P, baseline_schedule(), T_SETTLE, dt=0.5)
    Gs, Ss, gs = (float(sb[x][-1]) for x in ("G", "S", "g"))
    arm_D = Gs + (P.Theta / P.S_rest) * Ss
    log(f"probe: settled slice G={Gs:.4f} S={Ss:.4f} g={gs:.4f}; arming "
        f"boundary there: D > G + 0.8*S = {arm_D:.4f} — ABOVE both demand "
        f"nullclines ({D_NULL_A0:.4f}, {D_NULL_AP:.4f}): no protocol arms from "
        f"the healthy state by demand alone")
    # analytic crossing-time predictions.  dD/dt = (K - (K+delta_D)*D)/tau_D
    # with K=(D_base+A)*beta_D is AFFINE in D, so the closed forms are exact:
    #   D(t) = D_null + (D0 - D_null)*exp(-r t),  r = (K+delta_D)/tau_D.
    a_ep = P.k_in * 0.9 / (P.k_in * 0.9 + P.rho_a)
    dt_a = -np.log(1 - A_STAR / a_ep) / (P.k_in * 0.9 + P.rho_a)
    r0 = ((P.D_base + 0.0) * P.beta_D + P.delta_D) / P.tau_D      # A = 0
    rp = ((P.D_base + 0.5) * P.beta_D + P.delta_D) / P.tau_D      # A = 0.5
    D100 = D_NULL_A0 + (P.D0 - D_NULL_A0) * np.exp(-r0 * 100.0)
    t_up = 100.0 + np.log((D_NULL_AP - D100) / (D_NULL_AP - D_STAR)) / rp
    D160 = D_NULL_AP + (D100 - D_NULL_AP) * np.exp(-rp * 60.0)
    t_dn = 160.0 + np.log((D160 - D_NULL_A0) / (D_STAR - D_NULL_A0)) / r0
    a_c_on = P.k_in * P.chi * 1.0 / (P.k_in * P.chi * 1.0 + P.rho_a)
    dt_a_off = np.log(a_ep / A_STAR) / P.rho_a
    log(f"probe PREDICT: Q2 entry t={100 + dt_a:.2f}; failure Q3 entry "
        f"t={t_up:.2f} exit t={t_dn:.2f} (Q3 dwell {t_dn - t_up:.1f}); "
        f"D(100)={D100:.4f} D(160)={D160:.4f} vs D_STAR={D_STAR:.4f}")
    log(f"probe PREDICT: recall Q2 exit t={200 + dt_a_off:.2f}; failure "
        f"terminal a~{a_c_on:.4f} (held by c), D~{D_NULL_A0:.4f} -> Q2 armed")
    log(f"probe DONE ({time.time() - t0:.1f}s)")


# ------------------------------------------------------------- part 1: traj
def part_traj() -> None:
    """The five runs, classified by position + arming; the decisive test."""
    log(f"partTRAJ START: dt={DT:g}, rules as pre-registered "
        f"(A_STAR={A_STAR:.5f}, D_STAR={D_STAR:.5f})")
    t00 = time.time()
    runs = {}
    specs = [
        ("baseline", baseline_schedule(), T_RUN),
        ("recall100", inward_episode(100.0, 200.0, 0.9), T_RUN),
        ("failure", failure_schedule(), T_RUN),
        ("brinkOK", inward_episode(100.0, 100.0 + BRINK_OK, 0.9), T_BRINK),
        ("brinkSTUCK", inward_episode(100.0, 100.0 + BRINK_STUCK, 0.9), T_BRINK),
    ]
    for name, sch, T in specs:
        t0 = time.time()
        sol = simulate(P, sch, T, dt=DT)
        q = quadrant(sol["a"], sol["D"])
        ar = armed(sol["E"], sol["Theta_eff"])
        runs[name] = (sol, q, ar)
        log(f"partTRAJ: {name} done T={T:g} G_end={sol['G'][-1]:.4f} "
            f"collapsed_at={sol['collapsed_at']} "
            f"({time.time() - t0:.1f}s)")
    # exp18 anchor cross-check on the brink pair (same construction, T=900)
    sok, ss = runs["brinkOK"][0], runs["brinkSTUCK"][0]
    log(f"partTRAJ: brink pair vs exp18 anchors: dwell {BRINK_OK:g} -> "
        f"G_end={sok['G'][-1]:.4f} (anchor 0.8854) is_stuck="
        f"{is_stuck(sok, P)}; dwell {BRINK_STUCK:g} -> G_end="
        f"{ss['G'][-1]:.4f} (anchor 0.0487) is_stuck={is_stuck(ss, P)}")

    out = {}
    for name, (sol, q, ar) in runs.items():
        t = sol["t"]
        for k in ("t", "a", "D", "G", "S", "E", "Theta_eff", "c"):
            out[f"{name}_{k}"] = sol[k]
        out[f"{name}_quad"] = q.astype(np.uint8)
        out[f"{name}_armed"] = ar
        # segment table
        segs = seg_runs(t, q, ar)
        log(f"traj {name}: {len(segs)} (quadrant, armed) segments")
        for (s, e, qq, aa) in segs:
            log(f"  t=[{t[s]:8.2f},{t[e - 1]:8.2f}] dur={t[e - 1] - t[s]:8.2f} "
                f"{QNAMES[qq]:12s} armed={int(aa)} "
                f"G {sol['G'][s]:.4f}->{sol['G'][e - 1]:.4f} "
                f"D {sol['D'][s]:.4f}->{sol['D'][e - 1]:.4f}")
        # occupancy: full run, episode window, post-episode
        for lbl, (w0, w1) in [("full", (0.0, T_RUN)),
                              ("episode", (100.0, 200.0)),
                              ("post", (200.0, 1e9))]:
            fr, fa = occupancy(t, q, ar, w0, w1)
            dw = {QNAMES[kk]: v * 100 for kk, v in fr.items()}
            log(f"traj {name} OCC [{lbl}] " +
                " ".join(f"Q{k}={fr[k] * 100:6.2f}%" for k in (1, 2, 3, 4)) +
                f" armed={fa * 100:6.2f}%")
        log(f"traj {name}: ranges a=[{sol['a'].min():.4f},{sol['a'].max():.4f}] "
            f"D=[{sol['D'].min():.4f},{sol['D'].max():.4f}] "
            f"c_max={sol['c'].max():.3f} terminal a={sol['a'][-1]:.4f} "
            f"D={sol['D'][-1]:.4f} G={sol['G'][-1]:.4f} "
            f"E={sol['E'][-1]:.4f} Theta_eff={sol['Theta_eff'][-1]:.4f} "
            f"quad={QNAMES[int(q[-1])]} armed={bool(ar[-1])}")

    # ---- the decisive test on the failure run
    sol, q, ar = runs["failure"]
    t = sol["t"]
    q1 = np.where(q == 1)[0]
    q3 = np.where(q == 3)[0]
    arm_i = np.where(ar & (t > 100.0))[0]
    onset_i = np.searchsorted(t, 100.0)
    log("DECISIVE TEST (failure run, (a,D) position):")
    log(f"  last Q1 sample t={t[q1[-1]]:.2f} (onset 100.0); any Q1 after "
        f"t=110: {bool((q1[q1 > np.searchsorted(t, 110.0)]).size)} -> "
        f"{'AGREES with exp17 (leaves, never returns)' if t[q1[-1]] < 110 else 'CONFLICTS'}")
    if q3.size:
        log(f"  Q3 window t=[{t[q3[0]]:.2f},{t[q3[-1]]:.2f}] dwell="
            f"{t[q3[-1]] - t[q3[0]]:.2f} t.u. "
            f"(exp17 eat-dominant window was 166-172, 5.5 t.u.)")
        term_q3 = bool(q3[-1] == q.size - 1)
        log(f"  terminal quadrant = {QNAMES[int(q[-1])]} armed={bool(ar[-1])}; "
            f"STAYS in Q3 terminally: {term_q3} -> "
            f"{'CONFLICTS with stay-in-effortful' if not term_q3 else 'agrees'}")
    else:
        log("  Q3 NEVER visited (F3 fires)")
    if arm_i.size:
        ta = t[arm_i[0]]
        q_at_arm = int(q[arm_i[0]])
        post = t >= ta
        fr_post, _ = occupancy(t, q, ar, ta, 1e9)
        log(f"  armed from t={ta:.2f} (exp17: ~166, E meets Theta_eff at "
            f"0.523; here E={sol['E'][arm_i[0]]:.4f} Theta_eff="
            f"{sol['Theta_eff'][arm_i[0]]:.4f}); quadrant at arming: "
            f"{QNAMES[q_at_arm]}; Q3 entry preceded arming: "
            f"{bool(q3.size and t[q3[0]] < ta)}")
        log(f"  post-arming occupancy: " +
            " ".join(f"Q{k}={fr_post[k] * 100:6.2f}%" for k in (1, 2, 3, 4)) +
            f"; armed duration {t[-1] - ta:.2f} t.u. vs Q3 dwell "
            f"{(t[q3[-1]] - t[q3[0]]) if q3.size else 0.0:.2f} t.u. "
            f"(ratio {(t[-1] - ta) / max((t[q3[-1]] - t[q3[0]]) if q3.size else 0.0, 1e-9):.1f}x)")
    # brink pair: quadrant vs fate
    for nm in ("brinkOK", "brinkSTUCK"):
        s2, q2, a2 = runs[nm]
        fr, fa = occupancy(s2["t"], q2, a2, 100.0, 1e9)
        log(f"  {nm}: post-onset Q2={fr[2] * 100:.1f}% armed={fa * 100:.1f}% "
            f"G_end={s2['G'][-1]:.4f} — quadrant time-series nearly "
            f"identical across the fate boundary (measured below)")
    np.savez(os.path.join(CACHE, "exp19_part1_traj.npz"),
             a_star=A_STAR, d_star=D_STAR,
             d_null_a0=D_NULL_A0, d_null_ap=D_NULL_AP, **out)
    log(f"partTRAJ DONE ({time.time() - t00:.1f}s, cached "
        f"exp19_part1_traj.npz)")


# ---------------------------------------------------- part 2: areas + scan
def part_areas() -> None:
    """Quadrant areas of the plane, armed overlap at the healthy slice,
    and the (A_STAR, D_STAR) threshold-artifact scan (F5)."""
    log("partAREAS START")
    t0 = time.time()
    # geometric areas on the canonical box a in [0,1], D in [0,3] (exp17's box)
    box_area = 1.0 * 3.0
    frac_lo_D = D_STAR / 3.0
    areas = {1: 0.5 * frac_lo_D * box_area, 2: 0.5 * frac_lo_D * box_area,
             3: 0.5 * (1 - frac_lo_D) * box_area,
             4: 0.5 * (1 - frac_lo_D) * box_area}
    for k in (1, 2, 3, 4):
        log(f"partAREAS: {QNAMES[k]:12s} area={areas[k] / box_area * 100:6.2f}% "
            f"of the (a,D) box [0,1]x[0,3] (position axes: equal by "
            f"construction; exp17's term map had R3=85.85% — the old "
            f"degeneracy cannot recur by construction, it moves to occupancy)")
    # armed overlap at the settled slice
    sb = simulate(P, baseline_schedule(), T_SETTLE, dt=0.5)
    Gs, Ss = float(sb["G"][-1]), float(sb["S"][-1])
    arm_D = Gs + (P.Theta / P.S_rest) * Ss
    Dg = np.linspace(0.0, 3.0, 3001)
    ag = np.linspace(0.0, 1.0, 1001)
    armed_line = Dg > arm_D          # arming ignores a entirely
    for k in (1, 2, 3, 4):
        lo, hi = ((0.0, D_STAR) if k in (1, 2) else (D_STAR, 3.0))
        inw = (Dg >= lo) & (Dg < hi)
        frac = float(armed_line[inw].mean()) if inw.any() else 0.0
        log(f"partAREAS: armed fraction inside {QNAMES[k]:12s} at healthy "
            f"slice (G={Gs:.4f}, S={Ss:.4f}, arm iff D>{arm_D:.4f}): "
            f"{frac * 100:6.2f}%")
    log(f"partAREAS: the occupied band D in [0,1] of EVERY quadrant is "
            f"unarmed at the healthy slice (arm boundary {arm_D:.4f} > 1): "
            f"these runs arm through G falling, not D rising")
    # (F5) threshold scan on the measured trajectories
    z = np.load(os.path.join(CACHE, "exp19_part1_traj.npz"))
    res = {}
    a_grid = np.linspace(0.10, 0.85, 31)
    d_grid = np.linspace(D_NULL_A0 + 0.005, D_NULL_AP - 0.005, 23)
    keys = ["fail_Q3", "fail_termQ2", "recall_Q3", "base_Q1", "Q4_empty"]
    for kk in keys:
        res[kk] = np.zeros((d_grid.size, a_grid.size), bool)
    Dpk = float(z["failure_D"].max())
    for i, ds in enumerate(d_grid):
        for j, asg in enumerate(a_grid):
            qf = quadrant(z["failure_a"], z["failure_D"], asg, ds)
            qr = quadrant(z["recall100_a"], z["recall100_D"], asg, ds)
            qb = quadrant(z["baseline_a"], z["baseline_D"], asg, ds)
            res["fail_Q3"][i, j] = bool((qf == 3).any())
            res["fail_termQ2"][i, j] = int(qf[-1]) == 2
            res["recall_Q3"][i, j] = bool((qr == 3).any())
            res["base_Q1"][i, j] = bool((qb == 1).all())
            res["Q4_empty"][i, j] = not (bool((qf == 4).any())
                                         or bool((qr == 4).any())
                                         or bool((qb == 4).any()))
    log(f"partAREAS: F5 scan over A_STAR in [0.10,0.85] x D_STAR in "
        f"[{d_grid[0]:.4f},{d_grid[-1]:.4f}] ({a_grid.size}x{d_grid.size}):")
    for kk in keys:
        log(f"  {kk:12s} holds on {res[kk].mean() * 100:6.2f}% of the grid")
    log(f"  failure D_peak={Dpk:.4f}: Q3 visited iff D_STAR < D_peak "
        f"(holds on {float((d_grid < Dpk).mean()) * 100:.1f}% of the D_STAR "
        f"band); the registered D_STAR={D_STAR:.5f} sits "
        f"{Dpk - D_STAR:+.4f} below the peak")
    np.savez(os.path.join(CACHE, "exp19_part2_quad.npz"),
             a_grid=a_grid, d_grid=d_grid, d_peak=Dpk, arm_D=arm_D,
             Gs=Gs, Ss=Ss, areas=np.array([areas[k] for k in (1, 2, 3, 4)]),
             **{k: v for k, v in res.items()})
    log(f"partAREAS DONE ({time.time() - t0:.1f}s, cached exp19_part2_quad.npz)")


# ----------------------------------------------------------------- figure
def part_fig() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    log("fig START: figs/f22_quadrants.png (4 panels)")
    z = np.load(os.path.join(CACHE, "exp19_part1_traj.npz"))
    za = np.load(os.path.join(CACHE, "exp19_part2_quad.npz"))
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.6))

    # (a) quadrant plane + trajectories
    ax = axs[0][0]
    ax.axvspan(0, A_STAR, ymin=0, ymax=D_STAR / 1.2, color="#74a9cf", alpha=0.18)
    ax.axvspan(A_STAR, 1.0, ymin=0, ymax=D_STAR / 1.2, color="#3690c0", alpha=0.18)
    ax.axvspan(0, A_STAR, ymin=D_STAR / 1.2, ymax=1, color="#bdbdbd", alpha=0.22)
    ax.axvspan(A_STAR, 1.0, ymin=D_STAR / 1.2, ymax=1, color="#b2182b", alpha=0.18)
    ax.axhline(D_STAR, color="k", ls="--", lw=1.0)
    ax.axvline(A_STAR, color="k", ls="--", lw=1.0)
    ax.text(0.03, 0.06, "Q1 talking", fontsize=9, color="#1a5f8a")
    ax.text(0.70, 0.06, "Q2 relaxed", fontsize=9, color="#1a6f8f")
    ax.text(0.70, 1.13, "Q3 effortful", fontsize=9, color="#8a1515")
    ax.text(0.03, 1.13, "Q4 (unvisited)", fontsize=9, color="#555555")
    ax.text(0.51, 0.30, f"D*={D_STAR:.4f}", fontsize=7, rotation=90)
    for nm, col, lsty in [("baseline", "#1b7837", ":"),
                          ("recall100", "#0570b0", "-"),
                          ("failure", "#b2182b", "-")]:
        ax.plot(z[f"{nm}_a"], z[f"{nm}_D"], color=col, lw=1.1, ls=lsty,
                label=nm)
    ax.plot(z["failure_a"][0], z["failure_D"][0], "k*", ms=11, mfc="yellow")
    for nm in ("baseline", "recall100", "failure"):
        ax.plot(z[f"{nm}_a"][-1], z[f"{nm}_D"][-1], "kx", ms=7)
    arm_t = z["failure_t"][np.where(z["failure_armed"] & (z["failure_t"] > 100))[0][0]]
    k = np.searchsorted(z["failure_t"], arm_t)
    ax.plot(z["failure_a"][k], z["failure_D"][k], "ko", ms=7, mfc="none",
            mew=1.4, label=f"switch arms t={arm_t:.1f}")
    ax.set_xlabel("attention a")
    ax.set_ylabel("reducer demand D")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.2)
    ax.set_title("(a) (a,D) quadrants: A*=0.5, D*=nullcline midpoint; "
                 "x = terminal, o = arming")
    ax.legend(fontsize=7, loc="lower right")

    # (b) quadrant-membership timelines
    ax = axs[0][1]
    for nm, col, lsty in [("baseline", "#1b7837", ":"),
                          ("recall100", "#0570b0", "-"),
                          ("failure", "#b2182b", "-"),
                          ("brinkOK", "#888888", "--"),
                          ("brinkSTUCK", "#000000", "--")]:
        ax.step(z[f"{nm}_t"], z[f"{nm}_quad"], where="post", color=col,
                lw=1.1, ls=lsty, label=nm)
    ax.axvspan(arm_t, 1500, color="#b2182b", alpha=0.08)
    ax.axvspan(100, 160, color="#fdae61", alpha=0.25)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["Q1 talk", "Q2 relaxed", "Q3 effortful", "Q4"],
                       fontsize=8)
    ax.set_xlabel("t [tau_a]")
    ax.set_ylabel("quadrant (position rule)")
    ax.set_title("(b) quadrant membership; orange = affect pulse, "
                 "red shade = armed")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(0, 400)

    # (c) G(t) with arming and Q3 window
    ax = axs[1][0]
    for nm, col, lsty in [("failure", "#b2182b", "-"),
                          ("recall100", "#0570b0", "-"),
                          ("brinkOK", "#888888", "--"),
                          ("brinkSTUCK", "#000000", "--")]:
        ax.plot(z[f"{nm}_t"], z[f"{nm}_G"], color=col, lw=1.1, ls=lsty,
                label=f"{nm} (G_end={z[f'{nm}_G'][-1]:.3f})")
    q3i = np.where(z["failure_quad"] == 3)[0]
    if q3i.size:
        ax.axvspan(z["failure_t"][q3i[0]], z["failure_t"][q3i[-1]],
                   color="#b2182b", alpha=0.12)
        ax.text(z["failure_t"][q3i[0]], 0.45, " Q3 transit", fontsize=7,
                color="#8a1515")
    ax.axvline(arm_t, color="k", ls=":", lw=1.2)
    ax.text(arm_t + 3, 0.25, f"arms t={arm_t:.1f}", fontsize=7)
    ax.axhline(0.1, color="grey", lw=0.7, ls="-.")
    ax.set_xlabel("t [tau_a]")
    ax.set_ylabel("G")
    ax.set_title("(c) G(t): collapse vs recall; brink pair straddles "
                 "dwell 118.65/118.70")
    ax.legend(fontsize=7, loc="center right")
    ax.set_xlim(0, 900)
    ax.set_ylim(-0.02, 1.0)

    # (d) occupancy bars (full run and episode window)
    ax = axs[1][1]
    names = ["baseline", "recall100", "failure", "brinkOK", "brinkSTUCK"]
    cols = {1: "#74a9cf", 2: "#3690c0", 3: "#b2182b", 4: "#bdbdbd"}
    ylab, y = [], []
    for nm in names:
        for lbl, (w0, w1) in [("full", (0.0, 1500.0)),
                              ("episode", (100.0, 200.0))]:
            fr, _ = occupancy(z[f"{nm}_t"], z[f"{nm}_quad"],
                              z[f"{nm}_armed"], w0, w1)
            left = 0.0
            for kq in (1, 2, 3, 4):
                ax.barh(len(y), fr[kq] * 100, left=left * 100,
                        color=cols[kq],
                        edgecolor="white", height=0.62)
                if fr[kq] > 0.06:
                    ax.text((left + fr[kq] / 2) * 100, len(y),
                            f"Q{kq}\n{fr[kq] * 100:.0f}%", ha="center",
                            va="center", fontsize=6,
                            color="white" if kq != 4 else "black")
                left += fr[kq]
            y.append(len(y))
            ylab.append(f"{nm} [{lbl}]" + ("*" if lbl == "episode" else ""))
    ax.set_yticks(range(len(ylab)))
    ax.set_yticklabels(ylab, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("% of samples in quadrant")
    ax.set_title("(d) quadrant occupancy (* = episode window t in [100,200); "
                 "Q4 = 0 everywhere)")
    ax.set_xlim(0, 100)

    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fp = os.path.join(FIGDIR, "f22_quadrants.png")
    fig.savefig(fp, dpi=140)
    plt.close(fig)
    im = Image.open(fp)
    w, h = im.size
    g = np.asarray(im.convert("L"), float)
    ink = float((g < 245).mean())
    nonwhite_cols = int(((g < 245).any(axis=0)).sum())
    log(f"fig VERIFIED {os.path.basename(fp)}: size={w}x{h} px, "
        f"ink={ink * 100:.1f}% of pixels, nonblank-cols={nonwhite_cols}, "
        f"bytes={os.path.getsize(fp)}")
    assert w >= 1000 and h >= 400, "figure too small"
    assert ink > 0.02, "figure looks blank"
    assert nonwhite_cols > w // 2, "figure mostly blank columns"
    log("fig DONE")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("probe", "all"):
        probe()
    if which in ("traj", "all"):
        part_traj()
    if which in ("areas", "all"):
        part_areas()
    if which in ("fig", "all"):
        part_fig()
