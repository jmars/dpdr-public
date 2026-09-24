"""exp15 — THE BLIND-LEVER POSITIVE CONTROL (the final ODE experiment,
handoff-selfreg-blindlever), carrying the BOUNDARY CORRESPONDENCE TEST.

THE PRECONDITION CLAIM UNDER TEST (ours, NOT a test of Zhang): degradation
from fallible self-modification requires a lever whose short-horizon
self-assessment misjudges the long-run true outcome — i.e. a lever the
agent's own evaluator V cannot see, or sees wrongly.  The two-arm result
(exp12) established the negative (no trap of either kind; arm B never made
an uncertified adoption) and the structural reason (all 18 DEFAULT_LEVERS
act on one inward-drive/weakening channel).  exp15 CONSTRUCTS the missing
object and asks whether the walk then degrades.

THE CONSTRUCTION — INVISIBILITY BY HORIZON ARITHMETIC, NOT BY TUNING (two
independent mechanisms, both verified in part 1, neither tuned):
  (i)  the alarm state is measured at t = 100.0 exactly (episode onset);
       certify_state rolls the agent's own source forward T from there,
       and at the grid max T = 640 the rollout reaches 100 + 640 = 740 <
       800 = RESCUE_T0, so a lever confined to the rescue window is
       unreachable at every registered T.  (If T > 800 - 100 = 700 the
       lever becomes VISIBLE and the construction is DESTROYED — the grid
       T <= 640 respects that boundary; T is never raised.)
  (ii) the standard self-model assumption holds exogenous channels at
       their CURRENT values and certify_state passes u_ext = 0.0; k_ext
       enters the frozen RHS only as -k_ext*u_ext*a (model.py:121), so its
       V-margin is EXACTLY 0.0 in floating point at every T.

THE REGISTERED BLIND LEVER: k_ext (the external-pull / rescue-efficacy
gain), OPT-IN via GenParams.blind_levers = ("k_ext",); the default lever
set is unchanged so every existing cache stays reproducible (part 0).
The rescue onset/offset times are NOT parameterized (generation_schedule
fixes them as module constants) — stated and dropped per the brief.

EITHER OUTCOME IS A FINDING: C3 (degradation — the precondition is
sufficient) or C5 (arm B never adopts a blind move — the agent's argmin +
tie rule ITSELF protects it; the precondition is necessary but NOT
sufficient, a stronger structural result than the two-arm verdict).

Parts:
  0  fidelity: blind-OFF reproduces the published two-arm arm-B cache
     (exp12_batteryB.npz) exactly; enabled=False bit-exact vs simulate;
     default blind_levers = (); plant margins unperturbed by the append;
     pytest count re-checked.
  1  THE HORIZON ARITHMETIC + THE MARGIN MATRIX: alarm-time verification
     per T; the full (lever, T) -> V-margin matrix at the frozen alarm
     state (19 levers x 7 capacities, both signs), invisible set
     (|margin| <= atol_tau) marked; the exact-zero check for k_ext.
  2  THE TWO-ARM BLIND BATTERY: T-grid x 24 generations x {A, B} with
     blind levers ON — adoption sequences, blind adoptions, blind-argmin
     events, per-decision margins (caches exp15_battery{A,B}_blind.npz).
  3  THE VERDICT: C1-C5 under the pre-registered criteria (DEG_TOL =
     1e-3, the two-arm run's own tolerance, reused) + THE COMMUTATIVITY
     CHECK (binding): Params-level (blind o plant == plant o blind) and
     WALK-level (forced blind adoption early in the walk; is the flip an
     exact order swap NO horizon can separate, or a genuine perturbation
     whose accumulation is measured?) + near-tie census.
  4  THE BOUNDARY CORRESPONDENCE TEST (failure side): B_failure =
     rollout reach t_a + T per T; the time-window each blind lever acts
     in; THE FULL (lever, T) -> V-margin matrix with the invisible set
     marked — the object queue step 4 (evidence-widening) must reproduce
     (B_fix == B_failure is the acceptance test; NOT a scalar summary).
  fig  f19_blindlever.png (verified programmatically — never opened).

Outputs: cache/exp15_part0.npz, exp15_part1.npz, exp15_batteryA_blind.npz,
exp15_batteryB_blind.npz, exp15_verdict.npz, exp15_Bfailure.npz,
figs/f19_blindlever.png.

Usage:  .venv/bin/python -m experiments.exp15_blindlever [0|1|2|3|4|fig|all]
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np

from dpdr.generational import (ALL_LEVERS, BLIND_LEVERS, DEFAULT_LEVERS,
                               EP_T0, EP_T1, PULSE_END, RESCUE_T0,
                               RESCUE_T1, RESCUE_U, GenParams,
                               generation_schedule, modify_once,
                               run_generational)
from dpdr.integrate import simulate
from dpdr.model import Params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
FIGDIR = os.path.join(ROOT, "figs")

P = Params()
NGEN = 24                    # generations per cell (the two-arm window)
T_GRID = (10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0)
ARMS = ("A", "B")
POLICY = {"A": "certified", "B": "always"}
DEG_TOL = 1e-3              # REUSED from the two-arm run (not re-invented)
# the construction's own boundary: above T_BOUNDARY the rollout reaches
# the rescue window and mechanism (i) is destroyed (grid max 640 < 700)
T_BOUNDARY = RESCUE_T0 - EP_T0          # 700.0
BLIND_SET = BLIND_LEVERS                # ("k_ext",)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _blind_on(T: float, policy: str = "certified") -> GenParams:
    return GenParams(T=T, adopt_policy=policy, blind_levers=BLIND_SET)


def _walk_local(gp: GenParams, n_gen: int, p0: Params | None = None,
                force: dict | None = None, keep_margins: bool = False):
    """A local replication of run_generational's loop (identical calls:
    simulate -> modify_once -> five-state carry), with one instrument:
    `force` = {'gen': k, 'lever': 'k_ext', 'sign': +1.0} replaces the
    agent's choice at generation k with a forced blind adoption (the
    commutativity / neutralization falsifier — perturb EARLY, never at
    the last generation: a perturbation there cannot propagate and the
    test passes trivially).  Replication is verified against
    run_generational in part 3 BEFORE any forced conclusion is drawn."""
    p = Params() if p0 is None else p0
    sch = generation_schedule()
    adopted, margins_hist = [], []
    G_min, G_end = [], []
    y_carry = None
    for k in range(n_gen):
        if y_carry is None:
            pk = p
        else:
            pk = replace(p, a0=float(np.clip(y_carry[0], 0.0, 1.0)),
                         G0=float(np.clip(y_carry[1], 0.0, 1.0)),
                         D0=float(np.clip(y_carry[2], 0.0, 1.5)),
                         S0=float(np.clip(y_carry[3], 0.0, 1.0)),
                         g_init=float(np.clip(y_carry[4], 0.0, 2.0)))
        sol = simulate(pk, sch, gp.horizon, dt=gp.dt)
        G_min.append(float(sol["G"].min()))
        G_end.append(float(sol["G"][-1]))
        if force is not None and k == force["gen"]:
            lever, sgn = force["lever"], force["sign"]
            adopted.append(f"{lever}{'+' if sgn > 0 else '-'}")
            if keep_margins:
                margins_hist.append({})
            p = replace(pk, **{lever: getattr(pk, lever)
                               * (gp.step ** sgn)})
        else:
            rec = modify_once(pk, sol, gp) if gp.enabled else dict(
                adopted="", lever="", p_new=pk, margins={})
            adopted.append(rec["adopted"])
            if keep_margins:
                margins_hist.append(rec["margins"])
            p = rec["p_new"]
        y_carry = np.array([sol["a"][-1], sol["G"][-1], sol["D"][-1],
                            sol["S"][-1], sol["g"][-1]])
    out = dict(adopted=np.array(adopted, dtype=object),
               G_min=np.array(G_min), G_end=np.array(G_end),
               params_final=p)
    if keep_margins:
        out["margins"] = np.array(margins_hist, dtype=object)
    return out


# --------------------------------------------------------------- part 0
def part0() -> None:
    """Fidelity: the opt-in construction leaves every existing number
    reproducible.  (i) blind-OFF (the default) reproduces the published
    two-arm arm-B cache exactly; (ii) enabled=False is bit-exact vs
    simulate; (iii) default blind_levers = (); (iv) appending the blind
    lever does not perturb the plant margins; (v) pytest count."""
    out: dict = {}
    log("part 0 START: fidelity (blind-OFF = the published two-arm "
        "system, exactly)")
    # (iii) the default is OFF
    out["default_blind_levers"] = repr(GenParams().blind_levers)
    out["default_effective_is_DEFAULT"] = bool(
        GenParams().effective_levers() == DEFAULT_LEVERS)
    out["ALL_LEVERS"] = np.array(ALL_LEVERS, dtype=object)
    log(f"part 0: default blind_levers = {out['default_blind_levers']}, "
        f"effective == DEFAULT_LEVERS: "
        f"{out['default_effective_is_DEFAULT']}")
    # (ii) enabled=False bit-exact vs the frozen driver
    gp = GenParams(enabled=False, T=40.0)
    res = run_generational(gp, 4, keep_sol=True)
    ref = simulate(P, generation_schedule(), gp.horizon, dt=gp.dt)
    out["gen1_bitexact_maxdG"] = float(
        np.max(np.abs(res["sols"][0]["G"] - ref["G"])))
    out["disabled_any_adopted"] = bool(
        any(a != "" for a in res["adopted"]))
    out["disabled_G_end"] = float(np.asarray(res["G_end"], float).mean())
    out["disabled_G_min"] = float(min(
        float(sol["G"].min()) for sol in res["sols"]))
    log(f"part 0: enabled=False gen-1 vs simulate max|dG| = "
        f"{out['gen1_bitexact_maxdG']:.3e}; any adoption = "
        f"{out['disabled_any_adopted']}; frozen G_end "
        f"{out['disabled_G_end']:.4f}, frozen G_min floor "
        f"{out['disabled_G_min']:.4f}")
    # (i) blind-OFF reproduces the PUBLISHED arm-B cache (exp12)
    zb = np.load(os.path.join(CACHE, "exp12_batteryB.npz"),
                 allow_pickle=True)
    T_ref = zb["T"]
    max_d_end = max_d_curve = 0.0
    for i, Ti in enumerate(T_ref):
        r = run_generational(GenParams(T=float(Ti),
                                       adopt_policy="always"), NGEN)
        d_end = abs(float(r["G_end"][-1]) - float(zb["G_end"][i]))
        d_curve = float(np.max(np.abs(np.asarray(r["G_end"], float)
                                       - zb[f"Gendcurve_T{Ti:g}"])))
        d_seq = bool(np.array_equal(
            np.array(r["adopted"], dtype=object), zb[f"seq_T{Ti:g}"]))
        max_d_end = max(max_d_end, d_end)
        max_d_curve = max(max_d_curve, d_curve)
        out[f"armB_T{Ti:g}_dGend"] = d_end
        out[f"armB_T{Ti:g}_dGendcurve"] = d_curve
        out[f"armB_T{Ti:g}_seq_equal"] = d_seq
        log(f"part 0: blind-OFF arm B T={Ti:6.1f}: dG_end "
            f"{d_end:.1e}, max|dG_end curve| {d_curve:.1e}, adopted "
            f"sequence identical {d_seq}")
    out["armB_reproduces_cache"] = bool(
        max_d_end == 0.0 and max_d_curve == 0.0
        and all(out[f"armB_T{Ti:g}_seq_equal"] for Ti in T_ref))
    log(f"part 0: blind-OFF reproduces the published arm-B cache "
        f"EXACTLY: {out['armB_reproduces_cache']} "
        f"(max dG_end {max_d_end:.1e}, max curve delta {max_d_curve:.1e})")
    # (iv) appending the blind lever does not perturb the plant margins
    sol = simulate(P, generation_schedule(), gp.horizon, dt=gp.dt)
    r_off = modify_once(P, sol, GenParams(T=40.0))
    r_on = modify_once(P, sol, _blind_on(40.0))
    plant_same = all(r_off["margins"][k] == r_on["margins"][k]
                     for k in r_off["margins"])
    out["plant_margins_unperturbed"] = bool(plant_same)
    out["blind_margin_plus_T40"] = float(r_on["margins"]["k_ext+"])
    out["blind_margin_minus_T40"] = float(r_on["margins"]["k_ext-"])
    out["n_levers_off"] = len(r_off["margins"])
    out["n_levers_on"] = len(r_on["margins"])
    log(f"part 0: plant margins identical with blind lever appended: "
        f"{plant_same}; k_ext margins "
        f"({r_on['margins']['k_ext+']:.1e}, "
        f"{r_on['margins']['k_ext-']:.1e}); candidates "
        f"{len(r_off['margins'])} -> {len(r_on['margins'])}")
    # (v) frozen-suite pytest count
    try:
        pr = subprocess.run(
            [sys.executable, "-m", "pytest", "-q",
             os.path.join(ROOT, "tests/")],
            capture_output=True, text=True, timeout=600, cwd=ROOT)
        tail = pr.stdout.strip().splitlines()[-1] if pr.stdout.strip() \
            else ""
        out["pytest_returncode"] = int(pr.returncode)
        out["pytest_tail"] = tail
        log(f"part 0: pytest tests/ -> rc={pr.returncode}: {tail}")
    except Exception as e:                                # pragma: no cover
        out["pytest_returncode"] = -1
        out["pytest_tail"] = f"subprocess failed: {e}"
    ok = (out["default_effective_is_DEFAULT"]
          and out["gen1_bitexact_maxdG"] == 0.0
          and not out["disabled_any_adopted"]
          and out["armB_reproduces_cache"]
          and out["plant_margins_unperturbed"]
          and out["pytest_returncode"] == 0)
    out["fidelity_ok"] = bool(ok)
    log(f"part 0 DONE: fidelity_ok = {ok}")
    np.savez(os.path.join(CACHE, "exp15_part0.npz"), **out)


# --------------------------------------------------------------- part 1
def part1() -> None:
    """The horizon arithmetic verified + THE (lever, T) -> V-margin
    matrix at the frozen alarm state.  This is B_failure object (iii)
    and the C1 evidence."""
    out: dict = {}
    log("part 1 START: horizon arithmetic + the margin matrix "
        f"(19 levers x {len(T_GRID)} capacities)")
    # ---- the arithmetic, printed and cached
    log(f"part 1: episode [{EP_T0:g}, {EP_T1:g}), pulse to {PULSE_END:g}, "
        f"rescue u_ext={RESCUE_U:g} on [{RESCUE_T0:g}, {RESCUE_T1:g})")
    log(f"part 1: HORIZON ARITHMETIC — alarm at t_a = {EP_T0:g} exactly "
        f"(episode onset, first sample of _alarm_state's argmax); "
        f"rollout reach = t_a + T; at T_max = {max(T_GRID):g} the reach "
        f"is {EP_T0 + max(T_GRID):g} < {RESCUE_T0:g} = RESCUE_T0, so any "
        f"lever confined to the rescue window is INVISIBLE to V at every "
        f"registered T — by arithmetic (100 + 640 = 740 < 800), not by "
        f"tuning")
    log(f"part 1: the construction's own boundary: T > {T_BOUNDARY:g} "
        f"(= 800 - 100) would reach the rescue window and DESTROY the "
        f"construction; the grid max {max(T_GRID):g} respects it with a "
        f"{T_BOUNDARY - max(T_GRID):g} t.u. margin; T is never raised")
    out["t_alarm"] = EP_T0
    out["T_grid"] = np.array(T_GRID)
    out["reach"] = np.array([EP_T0 + T for T in T_GRID])
    out["RESCUE_T0"] = RESCUE_T0
    out["T_BOUNDARY"] = T_BOUNDARY
    # ---- the frozen alarm state (generation 1 = Params defaults)
    gp0 = _blind_on(40.0)
    sol = simulate(P, generation_schedule(), gp0.horizon, dt=gp0.dt)
    # ---- the matrix: both signs, every lever, every T
    from dpdr.generational import _alarm_state, certify_state
    alarm = _alarm_state(sol, P, gp0)
    _i, t_a, y, V0, A = alarm
    out["alarm_t_gen1"] = float(t_a)
    out["alarm_V0_gen1"] = float(V0)
    log(f"part 1: gen-1 alarm state measured at t = {t_a:g} (episode "
        f"onset), V0 = {V0:.4e}")
    m_plus = np.zeros((len(ALL_LEVERS), len(T_GRID)))
    m_minus = np.zeros_like(m_plus)
    for j, T in enumerate(T_GRID):
        gpj = _blind_on(T)
        # the margin baseline MUST be recomputed at THIS horizon: the
        # alarm's V0 came from gp0 (T=40); a per-column baseline of a
        # different horizon would leak a horizon difference into every
        # row (found the hard way: it put a spurious 3.7e-01 on the
        # k_ext row, whose true margin is exactly 0 at every T)
        V0j = certify_state(y, P, gpj, gpj.T, A)
        for i, lev in enumerate(ALL_LEVERS):
            for sgn, arr in ((+1.0, m_plus), (-1.0, m_minus)):
                pc = replace(P, **{lev: getattr(P, lev)
                                   * (gpj.step ** sgn)})
                arr[i, j] = certify_state(y, pc, gpj, gpj.T, A) - V0j
    out["m_plus"], out["m_minus"] = m_plus, m_minus
    out["levers"] = np.array(ALL_LEVERS, dtype=object)
    invisible = (np.abs(m_plus) <= GenParams().atol_tau) & \
                (np.abs(m_minus) <= GenParams().atol_tau)
    out["invisible"] = invisible
    out["atol_tau"] = GenParams().atol_tau
    # ---- report: blind row, plant summary, exact-zero check
    bi = list(ALL_LEVERS).index("k_ext")
    blind_exact_zero = bool(np.all(m_plus[bi] == 0.0)
                            and np.all(m_minus[bi] == 0.0))
    out["blind_exact_zero_allT"] = blind_exact_zero
    out["blind_margin_max_abs"] = float(
        max(np.abs(m_plus[bi]).max(), np.abs(m_minus[bi]).max()))
    plant_rows = [i for i in range(len(ALL_LEVERS)) if i != bi]
    plant_min = float(min(m_plus[plant_rows].min(),
                          m_minus[plant_rows].min()))
    out["plant_margin_min"] = plant_min
    out["plant_any_certified_perT"] = np.array(
        [bool(min(m_plus[plant_rows, j].min(),
                  m_minus[plant_rows, j].min()) < -GenParams().atol_tau)
         for j in range(len(T_GRID))])
    log(f"part 1: BLIND ROW (k_ext): margin is EXACTLY 0.0 in floating "
        f"point at every T = {blind_exact_zero} (max |m| "
        f"{out['blind_margin_max_abs']:.1e}); BOTH mechanisms hold — "
        f"horizon arithmetic (reach 740 < 800) AND the u_ext = 0 "
        f"self-model identity")
    log(f"part 1: PLANT ROWS: best (most negative) margin {plant_min:.3e}"
        f"; some plant margin certified (< -atol) at every T: "
        f"{bool(out['plant_any_certified_perT'].all())}")
    n_inv_cells = int(invisible.sum())
    inv_names = [ALL_LEVERS[i] for i in range(len(ALL_LEVERS))
                 if invisible[i].all()]
    log(f"part 1: invisible CELLS (|m| <= atol) = {n_inv_cells} of "
        f"{invisible.size}; levers invisible at EVERY T = "
        f"{len(inv_names)} of {len(ALL_LEVERS)} -> {inv_names}")
    log("part 1: TWO TIERS of invisibility (stated so B_failure is "
        "honest): k_ext is invisible STRUCTURALLY (exactly 0.0 via the "
        "u_ext=0 identity — at EVERY state, every T); the plant levers "
        "in that list are invisible AT THIS (gen-1) alarm state only "
        "and MAY become visible at other walk states — part 4 measures "
        "the distinction across every battery decision")
    out["invisible_allT_names"] = np.array(inv_names, dtype=object)
    # ---- C1 verdict from the matrix
    c1 = bool(blind_exact_zero
              and out["plant_any_certified_perT"].all())
    out["C1_margin_table"] = c1
    log(f"part 1: C1 (blind margins <= atol at every T while plant "
        f"margins are certified at every T): {c1}")
    log("part 1 DONE: cached -> cache/exp15_part1.npz")
    np.savez(os.path.join(CACHE, "exp15_part1.npz"), **out)


# --------------------------------------------------------------- part 2
def _cell(T: float, policy: str) -> dict:
    gp = _blind_on(T, policy)
    r = run_generational(gp, NGEN)
    seq = np.array(r["adopted"], dtype=object)
    marg = np.asarray(r["margin"], float)
    # blind-lever adoptions and ARGMIN events (argmin even when the tie
    # gate then refuses to act — both recorded; C2 vs C5 hinge on them)
    blind_adopt = [k for k, a in enumerate(seq) if a.startswith("k_ext")]
    blind_argmin, blind_gap = [], []
    for k, md in enumerate(r["margins"]):
        if not md:
            continue
        bk = min(md, key=md.get)
        if bk.startswith("k_ext"):
            blind_argmin.append(k)
        gap = min(abs(md[bk] - md[kk]) for kk in ("k_ext+", "k_ext-")
                  if kk != bk)
        blind_gap.append(gap)
    uncert = int(np.sum([(a != "") and (m >= gp.atol_tau)
                         for a, m in zip(seq, marg)]))
    return dict(gp=gp, T=T, policy=policy, seq=seq, margin=marg,
                G_end=np.asarray(r["G_end"], float),
                G_min=np.asarray(r["G_min"], float),
                E_end=float(r["E_end"][-1]),
                t_eval=np.asarray(r["t_eval"], float),
                n_adopt=int(np.sum([a != "" for a in seq])),
                n_uncert=uncert,
                blind_adopt=blind_adopt, blind_argmin=blind_argmin,
                min_blind_gap=(min(blind_gap) if blind_gap
                               else float("nan")),
                params_final=r["params_final"],
                margins=np.array(r["margins"], dtype=object))


def part2() -> None:
    log(f"part 2 START: the two-arm BLIND battery — {len(T_GRID)} "
        f"capacities x {NGEN} generations x arms {ARMS}, blind levers "
        f"{BLIND_SET} ON")
    z0 = np.load(os.path.join(CACHE, "exp15_part0.npz"),
                 allow_pickle=True)
    frozen_end = float(z0["disabled_G_end"])
    frozen_min = float(z0["disabled_G_min"])
    for arm in ARMS:
        cells = []
        for T in T_GRID:
            c = _cell(T, POLICY[arm])
            cells.append(c)
            log(f"part 2 arm {arm}: T={T:6.1f}: G_min(worst) "
                f"{c['G_min'].min():.4f}  G_end(last) {c['G_end'][-1]:.4f}"
                f"  adoptions {c['n_adopt']:2d} (uncertified "
                f"{c['n_uncert']:2d})  blind adoptions "
                f"{len(c['blind_adopt'])}  blind argmin "
                f"{len(c['blind_argmin'])}  min blind gap "
                f"{c['min_blind_gap']:.2e}  t_eval range "
                f"[{c['t_eval'].min():.1f}, {c['t_eval'].max():.1f}]")
        np.savez(os.path.join(CACHE, f"exp15_battery{arm}_blind.npz"),
                 T=np.array(T_GRID), frozen_G_end=frozen_end,
                 frozen_G_min=frozen_min, n_gen=NGEN,
                 policy=POLICY[arm],
                 G_end_last=np.array([c["G_end"][-1] for c in cells]),
                 G_min_worst=np.array([c["G_min"].min() for c in cells]),
                 n_adopt=np.array([c["n_adopt"] for c in cells], int),
                 n_uncert=np.array([c["n_uncert"] for c in cells], int),
                 n_blind_adopt=np.array([len(c["blind_adopt"])
                                         for c in cells], int),
                 n_blind_argmin=np.array([len(c["blind_argmin"])
                                          for c in cells], int),
                 min_blind_gap=np.array([c["min_blind_gap"]
                                         for c in cells]),
                 t_eval_min=np.array([c["t_eval"].min() for c in cells]),
                 t_eval_max=np.array([c["t_eval"].max() for c in cells]),
                 **{f"seq_T{T:g}": c["seq"] for T, c in zip(T_GRID,
                                                            cells)},
                 **{f"Gendcurve_T{T:g}": c["G_end"]
                    for T, c in zip(T_GRID, cells)},
                 **{f"Gmincurve_T{T:g}": c["G_min"]
                    for T, c in zip(T_GRID, cells)},
                 **{f"margin_T{T:g}": c["margin"]
                    for T, c in zip(T_GRID, cells)},
                 **{f"blindadopt_T{T:g}":
                    np.array(c["blind_adopt"] + [-1], int)
                    for T, c in zip(T_GRID, cells)},
                 **{f"blindargmin_T{T:g}":
                    np.array(c["blind_argmin"] + [-1], int)
                    for T, c in zip(T_GRID, cells)})
        log(f"part 2 arm {arm} DONE: cached -> cache/exp15_battery"
            f"{arm}_blind.npz")


# --------------------------------------------------------------- part 3
def part3() -> None:
    """C1-C5 verdicts + THE COMMUTATIVITY CHECK (binding) + near-tie
    census."""
    log("part 3 START: the verdict (criteria pre-registered in "
        "exp15_preregistration.md BEFORE the battery)")
    z1 = np.load(os.path.join(CACHE, "exp15_part1.npz"),
                 allow_pickle=True)
    zA = np.load(os.path.join(CACHE, "exp15_batteryA_blind.npz"),
                 allow_pickle=True)
    zB = np.load(os.path.join(CACHE, "exp15_batteryB_blind.npz"),
                 allow_pickle=True)
    frozen_end = float(zA["frozen_G_end"])
    frozen_min = float(zA["frozen_G_min"])
    out: dict = {"DEG_TOL": DEG_TOL, "frozen_G_end": frozen_end,
                 "frozen_G_min": frozen_min}
    # ---- C1 (from part 1's matrix)
    out["C1"] = bool(z1["C1_margin_table"])
    log(f"part 3: C1 (blind margins <= atol at every T, plant margins "
        f"certified at every T): {out['C1']}")
    # ---- C2/C5 (blind adoptions per arm)
    bA_adopt = int(np.sum(zA["n_blind_adopt"]))
    bB_adopt = int(np.sum(zB["n_blind_adopt"]))
    bB_argmin = int(np.sum(zB["n_blind_argmin"]))
    bA_argmin = int(np.sum(zA["n_blind_argmin"]))
    out["blind_adoptions_A"], out["blind_adoptions_B"] = bA_adopt, bB_adopt
    out["blind_argmin_B"] = bB_argmin
    out["C2"] = bool(bB_adopt >= 1 and bA_adopt == 0)
    out["C5"] = bool(bB_adopt == 0)
    log(f"part 3: blind adoptions — arm A {bA_adopt}, arm B {bB_adopt} "
        f"(arm-B blind ARGMIN events {bB_argmin}, arm-A {bA_argmin}); "
        f"C2 (B adopts >=1, A none) = {out['C2']}; C5 (B never adopts "
        f"blind) = {out['C5']}")
    # ---- C3/C4 (degradation, the two-arm criteria, DEG_TOL reused)
    for arm, z in (("A", zA), ("B", zB)):
        c1e, c1m, c2 = [], [], []
        for i, Ti in enumerate(z["T"]):
            ge = z[f"Gendcurve_T{Ti:g}"]
            gm = z[f"Gmincurve_T{Ti:g}"]
            c1e.append(bool(np.any(ge < frozen_end - DEG_TOL)))
            c1m.append(bool(np.min(gm) < frozen_min - DEG_TOL))
            c2.append(bool(np.any(ge < ge[0] - DEG_TOL)))
        out[f"{arm}_c1_end"] = np.array(c1e)
        out[f"{arm}_c1_min"] = np.array(c1m)
        out[f"{arm}_c2"] = np.array(c2)
        out[f"{arm}_any_degrades"] = bool(
            np.any(c1e) or np.any(c1m) or np.any(c2))
        log(f"part 3 arm {arm}: any degradation (C1end/C1min/C2, "
            f"DEG_TOL={DEG_TOL:g}) = {out[f'{arm}_any_degrades']}")
    out["C3"] = bool(out["B_any_degrades"])
    out["C4"] = not out["A_any_degrades"]
    # where/when, if C3
    if out["C3"]:
        for i, Ti in enumerate(zB["T"]):
            if (out["B_c1_end"][i] or out["B_c1_min"][i]
                    or out["B_c2"][i]):
                ge = zB[f"Gendcurve_T{Ti:g}"]
                gm = zB[f"Gmincurve_T{Ti:g}"]
                k_end = int(np.argmax(ge < frozen_end - DEG_TOL)) \
                    if out["B_c1_end"][i] else -1
                k_min = int(np.argmax(gm < frozen_min - DEG_TOL)) \
                    if out["B_c1_min"][i] else -1
                log(f"part 3: C3 locus — T={Ti:g}: first C1end gen "
                    f"{k_end + 1 if k_end >= 0 else '-'}, first C1min "
                    f"gen {k_min + 1 if k_min >= 0 else '-'}")
    log(f"part 3: C3 (arm B degrades with blind levers present) = "
        f"{out['C3']}; C4 (arm A does not degrade) = {out['C4']}")
    # ---- near-tie census: any decision where the blind margin is
    # within atol of the argmin (the regime where a flip could occur)
    min_gap = float("inf")
    for arm, z in (("A", zA), ("B", zB)):
        for i, Ti in enumerate(z["T"]):
            gap = float(z["min_blind_gap"][i])
            if np.isfinite(gap):
                min_gap = min(min_gap, gap)
    out["min_blind_gap_any"] = min_gap
    log(f"part 3: smallest (argmin -> blind) margin gap anywhere in the "
        f"battery = {min_gap:.3e} vs atol_tau = {GenParams().atol_tau:g} "
        f"-> {'NEAR-TIE REGIME ENTERED' if min_gap <= GenParams().atol_tau else 'no blind near-tie anywhere'}")
    # ========================================================
    # THE COMMUTATIVITY CHECK (binding, from handoff-selfreg-gap-probe)
    # ========================================================
    log("part 3: COMMUTATIVITY CHECK (binding) — (a) Params level: "
        "apply blind o plant and plant o blind to the SAME Params in "
        "BOTH ORDERS, compare every field exactly")
    ok_params = True
    worst_field = None
    ref = Params()
    for lev in DEFAULT_LEVERS:
        for sgn in (+1.0, -1.0):
            for bsgn in (+1.0, -1.0):
                pA = replace(replace(ref, **{lev: getattr(ref, lev)
                                             * (1.15 ** sgn)}),
                             k_ext=ref.k_ext * (1.15 ** bsgn))
                pB = replace(replace(ref, k_ext=ref.k_ext
                                     * (1.15 ** bsgn)),
                             **{lev: getattr(ref, lev) * (1.15 ** sgn)})
                for f in ref.__dataclass_fields__:
                    if getattr(pA, f) != getattr(pB, f):
                        ok_params = False
                        worst_field = f"{lev}{sgn:+g}/k_ext{bsgn:+g}:{f}"
    out["commute_params_exact"] = bool(ok_params)
    out["commute_params_worst"] = str(worst_field)
    log(f"part 3: (a) blind o plant == plant o blind EXACTLY for all "
        f"18 plant levers x 2 signs x 2 blind signs: {ok_params} "
        f"(distinct multiplicative fields — the gap-probe's alpha_G/"
        f"gam_G mechanism)")
    # reversal: k_ext+ then k_ext- vs identity
    p_pm = replace(replace(ref, k_ext=ref.k_ext * 1.15),
                   k_ext=ref.k_ext * 1.15 * (1.15 ** -1.0))
    out["commute_blind_reversal_exact"] = bool(
        p_pm.k_ext == ref.k_ext)
    out["commute_blind_reversal_delta"] = float(p_pm.k_ext - ref.k_ext)
    log(f"part 3: (a') blind self-reversal (k_ext+ then k_ext-) exact: "
        f"{out['commute_blind_reversal_exact']} "
        f"(delta {out['commute_blind_reversal_delta']:.1e})")
    # (b) WALK level: replication FIRST (the probe's rule), then force
    log("part 3: (b) WALK level — replicate run_generational with the "
        "local loop FIRST (exact match required), then force a blind "
        "adoption EARLY (gen 5) and measure the accumulation")
    for T in (40.0, 640.0):
        gp = _blind_on(T, "always")
        r_ref = run_generational(gp, NGEN)
        w_loc = _walk_local(gp, NGEN, keep_margins=False)
        rep_ok = bool(np.array_equal(
            w_loc["adopted"], np.array(r_ref["adopted"], dtype=object))
            and np.max(np.abs(w_loc["G_end"]
                              - np.asarray(r_ref["G_end"], float)))
            == 0.0)
        out[f"T{T:g}_local_replicates"] = rep_ok
        log(f"part 3: T={T:g}: local loop replicates run_generational "
            f"exactly (sequence + G_end curve): {rep_ok}")
        # force k_ext+ at generation 5 (index 4) — EARLY, so the
        # perturbation can propagate (never at the last generation)
        w_f = _walk_local(gp, NGEN, force={"gen": 4, "lever": "k_ext",
                                           "sign": +1.0})
        d_end = np.abs(w_f["G_end"] - w_loc["G_end"])
        d_min = np.abs(w_f["G_min"] - w_loc["G_min"])
        div = [k for k in range(NGEN)
               if w_f["adopted"][k] != w_loc["adopted"][k]]
        # Params log-distance over ALL levers incl. blind
        ld = float(sum(abs(np.log(getattr(w_f["params_final"], lv)
                                  / getattr(ref, lv)))
                       for lv in ALL_LEVERS))
        ld0 = float(sum(abs(np.log(getattr(w_loc["params_final"], lv)
                                   / getattr(ref, lv)))
                        for lv in ALL_LEVERS))
        out[f"T{T:g}_forced_maxdGend"] = float(d_end.max())
        out[f"T{T:g}_forced_maxdGmin"] = float(d_min.max())
        out[f"T{T:g}_forced_final_dGend"] = float(abs(
            w_f["G_end"][-1] - w_loc["G_end"][-1]))
        out[f"T{T:g}_forced_n_diverge_gens"] = len(div)
        out[f"T{T:g}_forced_diverge_gens"] = repr(div[:8])
        out[f"T{T:g}_forced_logdist"] = ld
        out[f"T{T:g}_base_logdist"] = ld0
        out[f"T{T:g}_forced_kext_final"] = float(
            w_f["params_final"].k_ext)
        log(f"part 3: T={T:g}: FORCED k_ext+ at gen 5 -> adopted seq "
            f"diverges at {len(div)} generations {div[:8]}; max|dG_end| "
            f"{d_end.max():.2e}, max|dG_min| {d_min.max():.2e}; FINAL "
            f"|dG_end| {abs(w_f['G_end'][-1] - w_loc['G_end'][-1]):.2e};"
            f" final k_ext {w_f['params_final'].k_ext:.4f} (baseline "
            f"{w_loc['params_final'].k_ext:.4f}); log-dist "
            f"{ld:.6f} vs baseline {ld0:.6f}")
    # ---- the neutralization verdict (the false-negative hazard)
    forced_real = [bool(out[f"T{T:g}_forced_maxdGend"] > 0.0
                        or out[f"T{T:g}_forced_maxdGmin"] > 0.0)
                   for T in (40.0, 640.0)]
    out["blind_perturbation_is_real"] = bool(all(forced_real))
    log(f"part 3: NEUTRALIZATION HAZARD CHECK — a forced blind adoption "
        f"CHANGES the walk (not neutralized by commutativity): "
        f"{out['blind_perturbation_is_real']} — Params-level "
        f"commutativity holds (distinct fields) yet the WALK moves, so "
        f"'no blind degradation' is NOT a commutativity artifact: the "
        f"tie gate refuses the blind lever before any flip can occur")
    # ---- C5 semantics: WHAT refused the blind lever
    if out["C5"]:
        out["C5_mechanism"] = (
            "the agent's ARGMIN + TIE rule refused it: the blind margin "
            "is exactly 0.0 (never the strict argmin while any plant "
            "margin is negative), and even as argmin the tie gate "
            "|m| > atol_tau would refuse a no-basis move — the "
            "precondition is NECESSARY BUT NOT SUFFICIENT")
        log("part 3: C5 MECHANISM: " + out["C5_mechanism"])
    # ---- collapse-class discriminator columns (the queue's placement
    # decision): class-A protection (sound evaluator/refuses) vs
    # class-B degradation (blind evaluator/walks into it)
    out["collapse_class"] = (
        "class-A PROTECTION" if (out["C5"] and not out["C3"])
        else "class-B DEGRADATION" if out["C3"] else "INDETERMINATE")
    log(f"part 3: collapse-class discriminator: {out['collapse_class']}")
    log("part 3 DONE: cached -> cache/exp15_verdict.npz")
    np.savez(os.path.join(CACHE, "exp15_verdict.npz"), **out)


# --------------------------------------------------------------- part 4
def part4() -> None:
    """B_failure — the failure-side boundary specification, in the form
    queue step 4 (evidence-widening) must reproduce.  NOT a scalar."""
    log("part 4 START: B_failure (the Boundary Correspondence Test, "
        "failure side) — reach table, lever windows, the full matrix")
    z1 = np.load(os.path.join(CACHE, "exp15_part1.npz"),
                 allow_pickle=True)
    zB = np.load(os.path.join(CACHE, "exp15_batteryB_blind.npz"),
                 allow_pickle=True)
    T = np.asarray(z1["T_grid"], float)
    reach = np.asarray(z1["reach"], float)
    levers = [str(x) for x in z1["levers"]]
    m_plus = np.asarray(z1["m_plus"], float)
    m_minus = np.asarray(z1["m_minus"], float)
    invisible = np.asarray(z1["invisible"], bool)
    atol = float(z1["atol_tau"])
    # (i) the rollout reach per T
    log("part 4: (i) ROLLOUT REACH t_a + T (t_a = 100.0 measured):")
    for j, Ti in enumerate(T):
        log(f"part 4:   T={Ti:6.1f} -> reach {reach[j]:7.1f} "
            f"{'<' if reach[j] < RESCUE_T0 else '>='} "
            f"{RESCUE_T0:g} = RESCUE_T0 "
            f"{'(rescue window UNREACHED)' if reach[j] < RESCUE_T0 else '(REACHED — construction destroyed!)'}")
    # (ii) the time-window each blind lever acts in
    log("part 4: (ii) BLIND-LEVER ACTION WINDOWS vs {t > reach}:")
    windows = {}
    for lev in BLIND_SET:
        win = (RESCUE_T0, RESCUE_T1)
        windows[lev] = win
        log(f"part 4:   {lev}: acts on [{win[0]:g}, {win[1]:g}) "
            f"(u_ext = {RESCUE_U:g}); entirely inside the blind region "
            f"{{t > {reach[-1]:g}}} at every T "
            f"{'YES' if win[0] > reach.max() else 'NO'}")
    # (iii) THE MATRIX — printed in full with the invisible set marked
    log("part 4: (iii) THE FULL (lever, T) -> V-margin MATRIX "
        "(min-|margin| sign shown; * = INVISIBLE |m| <= atol at that T; "
        "row marked [INV-ALL] if invisible at every T):")
    hdr = "part 4:   lever        " + "".join(
        f"T={Ti:<8g}" for Ti in T)
    log(hdr)
    for i, lev in enumerate(levers):
        cells = []
        for j in range(len(T)):
            mp, mm = m_plus[i, j], m_minus[i, j]
            m, s = (mp, "+") if abs(mp) >= abs(mm) else (mm, "-")
            mark = "*" if invisible[i, j] else " "
            cells.append(f"{m:+.1e}{s}{mark:<4}")
        tag = " [INV-ALL]" if invisible[i].all() else ""
        log(f"part 4:   {lev:<12}" + "".join(cells) + tag)
    # the comparable object for step 4: B_fix must satisfy
    #   {(lever, T): margin > atol after widening} == the invisible set
    #   computed here, and visible-before stays visible.
    # (iii-b) THE WALK-WIDE MATRIX — the same (lever, T) object computed
    # at EVERY arm-B battery decision state (not only gen 1): the tier
    # distinction that matters to step 4.  A plant lever invisible at
    # gen 1 may become visible at a later walk state; k_ext cannot (the
    # u_ext = 0 identity holds at every alarm state).
    log("part 4: (iii-b) WALK-WIDE invisibility — the same matrix "
        "evaluated at EVERY arm-B decision state across the battery:")
    out_w = np.zeros((len(levers), len(T)), int)   # visible-cell counts
    n_dec = 0
    for j, Ti in enumerate(T):
        # margins per decision were not cached for space; recompute the
        # VISIBILITY (not the values) at each decision's alarm state by
        # replaying the walk with modify_once itself (the same call
        # run_generational makes — not a reimplementation of the rule)
        gpj = GenParams(T=float(Ti), adopt_policy="always",
                        blind_levers=BLIND_SET)
        from dpdr.generational import _alarm_state, certify_state
        from dpdr.integrate import simulate as _sim
        sch = generation_schedule()
        p_walk = Params()
        y_carry = None
        for k in range(NGEN):
            if y_carry is None:
                pk = p_walk
            else:
                pk = replace(p_walk,
                             a0=float(np.clip(y_carry[0], 0.0, 1.0)),
                             G0=float(np.clip(y_carry[1], 0.0, 1.0)),
                             D0=float(np.clip(y_carry[2], 0.0, 1.5)),
                             S0=float(np.clip(y_carry[3], 0.0, 1.0)),
                             g_init=float(np.clip(y_carry[4], 0.0, 2.0)))
            solk = _sim(pk, sch, gpj.horizon, dt=gpj.dt)
            alm = _alarm_state(solk, pk, gpj)
            if alm is not None:
                _ii, _tt, yk, V0k, Ak = alm
                n_dec += 1
                for i, lev in enumerate(levers):
                    for sgn in (+1.0, -1.0):
                        pc = replace(pk, **{lev: getattr(pk, lev)
                                            * (gpj.step ** sgn)})
                        m = certify_state(yk, pc, gpj, gpj.T, Ak) - V0k
                        if abs(m) > float(z1["atol_tau"]):
                            out_w[i, j] += 1
                            break
            rec = modify_once(pk, solk, gpj)
            p_walk = rec["p_new"]
            y_carry = np.array([solk["a"][-1], solk["G"][-1],
                                solk["D"][-1], solk["S"][-1],
                                solk["g"][-1]])
    walk_invisible = out_w == 0
    inv_walk = [levers[i] for i in range(len(levers))
                if walk_invisible[i].all()]
    log(f"part 4: evaluated {n_dec} decision states (arm B, 7 T x "
        f"{NGEN} gens); levers NEVER visible on any walk decision: "
        f"{inv_walk}")
    tier_struct = [l for l in inv_walk if l != "k_ext"]
    log(f"part 4: TIER STRUCTURE: k_ext invisible at every state "
        f"(structural, exact-0 identity) — the registered blind lever; "
        f"plant levers walk-invisible too: {tier_struct} — these are "
        f"invisible in PRACTICE on this task (their dynamics do not "
        f"move V at any visited alarm state), NOT structurally; step 4 "
        f"must widen assessment until the k_ext row moves — whether "
        f"the walk-invisible plant rows move is reported, not assumed")
    np.savez(os.path.join(CACHE, "exp15_Bfailure.npz"),
             walk_visible_counts=out_w,
             walk_n_decisions=n_dec,
             walk_invisible=walk_invisible,
             walk_invisible_names=np.array(inv_walk, dtype=object),
             T=T, reach=reach, t_a=float(z1["t_alarm"]),
             levers=np.array(levers, dtype=object),
             m_plus=m_plus, m_minus=m_minus,
             invisible=invisible, atol_tau=atol,
             blind_windows=np.array(
                 [windows[l] for l in BLIND_SET], float),
             blind_levers=np.array(BLIND_SET, dtype=object),
             RESCUE_T0=RESCUE_T0, RESCUE_T1=RESCUE_T1,
             T_BOUNDARY=T_BOUNDARY,
             blind_adoptions_B=int(np.sum(zB["n_blind_adopt"])),
             spec=("B_fix (step 4) must equal this invisible set: "
                   "levers invisible-before become visible, levers "
                   "visible-before stay visible — EXACT CLOSURE; "
                   "B_fix strictly containing B_failure = PARTIAL; "
                   "B_fix not containing B_failure = SUPERFICIAL"))
    n_inv_all = int(invisible.all(axis=1).sum())
    log(f"part 4: B_failure object: {len(levers)} levers x {len(T)} "
        f"capacities; invisible-at-every-T levers = {n_inv_all} "
        f"({', '.join(l for l, i in zip(levers, invisible.all(axis=1)) if i)});"
        f" cached -> cache/exp15_Bfailure.npz")
    log("part 4: STEP 4'S OBLIGATION (the pair designed as a unit): "
        "output B_fix = {(lever, T) whose margin moves beyond atol once "
        "the assessment is widened} + the region covered; the test is "
        "the SET comparison B_failure == B_fix; if C5 held the fix "
        "closes a boundary NOT CURRENTLY EXPLOITABLE — report AVAILABLE "
        "BUT NOT REQUIRED")
    log("part 4: TWO-BOUNDARY QUALIFIER: this is the HORIZON boundary "
        "B_h(T) (reach t_a + T) ONLY; exp16's CONTEXT boundary B_c "
        "(train/test evaluation representativeness) is DISTINCT, and "
        "whether one fix closes both is OPEN")
    log("part 4 DONE")


# --------------------------------------------------------------- figure
def part_fig() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    log("fig START: f19_blindlever.png")
    z1 = np.load(os.path.join(CACHE, "exp15_part1.npz"),
                 allow_pickle=True)
    zA = np.load(os.path.join(CACHE, "exp15_batteryA_blind.npz"),
                 allow_pickle=True)
    zB = np.load(os.path.join(CACHE, "exp15_batteryB_blind.npz"),
                 allow_pickle=True)
    T = np.asarray(z1["T_grid"], float)
    levers = [str(x) for x in z1["levers"]]
    m_plus = np.asarray(z1["m_plus"], float)
    m_minus = np.asarray(z1["m_minus"], float)
    invisible = np.asarray(z1["invisible"], bool)
    frozen = float(zA["frozen_G_end"])
    cA, cB = "#2166ac", "#b2182b"
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 9.0))

    # (a) the margin matrix: log10 max|m| with the invisible set masked
    ax = axs[0, 0]
    vis = np.maximum(np.abs(m_plus), np.abs(m_minus))
    data = np.where(vis > 0, np.log10(np.maximum(vis, 1e-300)),
                    np.nan)
    im = ax.imshow(data, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    ax.set_yticks(range(len(levers)),
                  [l + (" *" if invisible[i].all() else "")
                   for i, l in enumerate(levers)], fontsize=6.5)
    ax.set_xticks(range(len(T)), [f"{t:g}" for t in T])
    ax.set_xlabel("capacity $T$ [t.u.]")
    ax.set_title("(a) V-margin matrix $\\log_{10}\\max|m|$ at the alarm "
                 "state; * = invisible ($|m|\\leq$atol) at every $T$")
    fig.colorbar(im, ax=ax, shrink=0.8)
    bi = levers.index("k_ext")
    ax.add_patch(plt.Rectangle((-0.5, bi - 0.5), len(T), 1.0,
                               fill=False, edgecolor=cB, lw=1.8))

    # (b) G_end curves, both arms, grid extremes, vs the frozen line
    ax = axs[0, 1]
    gens = np.arange(1, NGEN + 1)
    for Ti, ls in ((T_GRID[0], "-"), (T_GRID[-1], "--")):
        ax.plot(gens, zA[f"Gendcurve_T{Ti:g}"], ls, color=cA, ms=3,
                label=f"A $T$={Ti:g}")
        ax.plot(gens, zB[f"Gendcurve_T{Ti:g}"], ls, color=cB, ms=3,
                label=f"B $T$={Ti:g}")
    ax.axhline(frozen, color="k", ls=":", lw=1.0, label="frozen")
    ax.set_xlabel("generation $k$")
    ax.set_ylabel("$G_{end}^{(k)}$")
    ax.set_title("(b) per-generation curves with blind levers present")
    ax.legend(fontsize=7)

    # (c) adoptions + blind argmin events per T
    ax = axs[1, 0]
    x = np.arange(len(T))
    w = 0.38
    ax.bar(x - w / 2, zA["n_adopt"], w, color=cA, label="A adoptions")
    ax.bar(x + w / 2, zB["n_adopt"], w, color=cB, label="B adoptions")
    ax.bar(x + w / 2, zB["n_blind_adopt"], w, facecolor="none",
           edgecolor="k", hatch="///", label="B blind adoptions")
    ax.plot(x, zB["n_blind_argmin"], "k^", ms=6,
            label="B blind argmin events")
    ax.set_xticks(x, [f"{t:g}" for t in T])
    ax.set_xlabel("capacity $T$")
    ax.set_ylabel(f"count in {NGEN} generations")
    ax.set_title("(c) adoption frequency; blind-lever events")
    ax.legend(fontsize=7)

    # (d) the boundary: rollout reach vs the rescue window + blind margin
    ax = axs[1, 1]
    ax.plot(T, np.asarray(z1["reach"], float), "o-", color=cA,
            label="rollout reach $t_a + T$")
    ax.axhline(RESCUE_T0, color=cB, ls="--",
               label=f"rescue onset $t$={RESCUE_T0:g}")
    ax.axvline(float(z1["T_BOUNDARY"]), color="k", ls=":",
               label=f"construction boundary $T$={float(z1['T_BOUNDARY']):g}")
    ax.fill_between(T, np.asarray(z1["reach"], float), RESCUE_T0,
                    color=cB, alpha=0.08)
    ax.set_xlabel("capacity $T$ [t.u.]")
    ax.set_ylabel("time reached [t.u.]")
    ax.set_title("(d) $B_h(T)$: the region no rollout reaches "
                 "(shaded) contains the rescue window")
    ax.legend(fontsize=7)

    zv = np.load(os.path.join(CACHE, "exp15_verdict.npz"),
                 allow_pickle=True)
    fig.suptitle("exp15 — blind-lever positive control: C1="
                 f"{bool(zv['C1'])} C2={bool(zv['C2'])} "
                 f"C3={bool(zv['C3'])} C4={bool(zv['C4'])} "
                 f"C5={bool(zv['C5'])}; blind adoptions A="
                 f"{int(zv['blind_adoptions_A'])} B="
                 f"{int(zv['blind_adoptions_B'])}; class "
                 f"{str(zv['collapse_class'])}", fontsize=10)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "f19_blindlever.png")
    fig.savefig(out, dpi=140)
    w_px, h_px = Image.open(out).size
    arr = np.asarray(Image.open(out).convert("L"))
    ink = float((arr < 250).mean())
    log(f"fig DONE -> {out} ({w_px}x{h_px}px, aspect "
        f"{w_px / h_px:.2f}, ink {ink:.3f})")
    assert w_px > 1000 and h_px > 800, f"bad size {w_px}x{h_px}"
    assert 0.005 < ink < 0.9, f"bad ink {ink}"
    assert abs(w_px / h_px - 11.5 / 9.0) < 0.05


PARTS = {"0": part0, "1": part1, "2": part2, "3": part3, "4": part4,
         "fig": part_fig}


def main(argv: list[str]) -> None:
    os.makedirs(CACHE, exist_ok=True)
    parts = argv[1:] or ["0", "1", "2", "3", "4", "fig"]
    if "all" in parts:
        parts = ["0", "1", "2", "3", "4", "fig"]
    t0 = time.time()
    for name in parts:
        if name not in PARTS:
            raise SystemExit(f"unknown part {name!r} "
                             f"(one of {sorted(PARTS)})")
        PARTS[name]()
    log(f"exp15 complete ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main(sys.argv)
