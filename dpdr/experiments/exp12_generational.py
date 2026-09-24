"""exp12 — the N-generation compounding realization of Zhang's operator
f = M o V o f_T, run as TWO ARMS (dpdr.generational, the opt-in module
that imports the frozen model read-only).

*** THE TWO-ARM DESIGN (handoff-selfreg-gen-twoarm; the single-arm test
had a design flaw the USER identified).  The original rule ended with
`if best_m < -atol_tau:` — the walk REFUSES TO MOVE unless the certifier
V approves, so its monotone improvement is GUARANTEED BY THE ACCEPTANCE
FILTER, not produced by the mechanism.  A self-improver that only
accepts certified improvements is optimizing against a trusted oracle.
The two arms from the same landscape:

  ARM A  adopt_policy="certified" — adopt argmin V only when the margin
         is certified (< -atol_tau).  INFALLIBLE BY CONSTRUCTION:
         monotone improvement guaranteed by the filter, degradation
         impossible.  The conservative baseline; CANNOT test the
         conjecture; NOT the agent design.
  ARM B  adopt_policy="always" — ALWAYS adopt argmin V whatever the
         margin's sign (atol_tau detects only an exact tie with the
         no-op, never gates on sign).  FALLIBLE: the agent acts on
         uncertain self-assessment; degradation is POSSIBLE.  At T=10
         the rollout sees ~10 of ~1010 t.u. (~1%) of the generation, so
         the agent acts on ~1% of the evidence — whether that harms is
         an OPEN QUESTION, and arm B is the only arm that can expose an
         ACCUMULATION trap (walk-level: every step looks no worse to V,
         the sequence lands worse; arm A halts at the first
         non-improving candidate and can never see it).

The pre-registered arm-B degradation criterion (stated BEFORE the
battery; see experiments/exp12_preregistration.md, ARM-B REGISTRATION):
arm B DEGRADES at capacity T if within the 24-generation window
  C1: G_end^(k) < frozen_G_end - DEG_TOL at any k, or
      min_k G_min^(k) < frozen_G_min - DEG_TOL;
  C2: G_end^(k) < arm B's own generation-1 G_end - DEG_TOL at any k.
DEG_TOL = 1e-3.  T_min exists iff some low capacities degrade and the
top of the grid does not (then bisect to ~2.5 t.u.); if NO T degrades,
T_min does not exist and NO bisect is performed.

Parts:
  0  fidelity: enabled=False is the frozen model exactly (generation 1
     bit-exact against a direct simulate call, no adoption anywhere);
     default adopt_policy is "certified"; frozen-suite pytest counts.
  1  atol sensitivity: certified adopt sets insensitive across atol_tau
     1e-6..1e-3 (the historical claim) + the arm-B tie gate at T=40.
  2  THE TWO-ARM BATTERY: T-grid x 24 generations x {A, B} — full
     per-generation G curves, adoption sequences, per-decision margins,
     uncertified-adoption counts (caches exp12_batteryA/B.npz).
  3  THE VERDICT: degradation per arm per T under C1/C2 above, the
     T_min bisect (only if the pattern demands it), same-Params
     comparison, and the accumulation-trap verdict.
  4  figure f16_generational.png (verified programmatically — size,
     aspect, non-trivial ink).

Outputs: cache/exp12_part0.npz, exp12_part1.npz, exp12_batteryA.npz,
exp12_batteryB.npz, exp12_verdict.npz, figs/f16_generational.png.

Usage:  .venv/bin/python -m experiments.exp12_generational [0|1|2|3|fig|all]
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

from dpdr.events import Schedule
from dpdr.generational import (DEFAULT_LEVERS, GenParams,
                               generation_schedule, run_generational)
from dpdr.integrate import simulate
from dpdr.model import Params

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
FIGDIR = os.path.join(ROOT, "figs")
PREREG_PATH = os.path.join(
    ROOT, "experiments", "exp12_preregistration.md")

P = Params()
NGEN = 24                    # generations per battery cell
# the battery T-grid: 10 (rollout floor delta_roll) .. 640 (>> tau_g=200;
# scan1's grid capped at 1280, where certification is already flat)
T_GRID = (10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0)
ATOLS = (1e-6, 1e-5, 1e-4, 1e-3)
ARMS = ("A", "B")
POLICY = {"A": "certified", "B": "always"}

# THE ARM-A REFERENCE (the orchestrator's published arm-A results, from
# handoff-selfreg-generational-finding, quoted verbatim).  Part 0 checks
# that adopt_policy="certified" still reproduces these.  Cells marked
# inferred were quoted as a group ("T=10/20/40", "T=160/640") in the
# finding; tolerance 2e-4 covers the 4-decimal rounding.
ARM_A_REF = {          # T: (final G_min, final G_end, quoted-cell?)
    10.0:  (0.5819, 0.9562, True),
    20.0:  (0.5819, 0.9562, False),
    40.0:  (0.5819, 0.9562, True),
    80.0:  (0.6660, 0.9774, True),
    160.0: (0.7000, 0.9806, True),
    320.0: (0.7000, 0.9806, False),
    640.0: (0.7000, 0.9806, True),
}
DEG_TOL = 1e-3          # pre-registered degradation tolerance (C1/C2)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _changed_fields(p: Params) -> dict:
    """Which levers differ from Params defaults, and by what factor."""
    ref = Params()
    return {lv: float(getattr(p, lv)) for lv in DEFAULT_LEVERS
            if getattr(p, lv) != getattr(ref, lv)}


def _log_distance(p: Params) -> float:
    """Total log-space wander from defaults over the lever set."""
    ref = Params()
    return float(sum(abs(np.log(getattr(p, lv) / getattr(ref, lv)))
                     for lv in DEFAULT_LEVERS))


# --------------------------------------------------------------- part 0
def part0() -> None:
    """Fidelity: M disabled = the frozen model, exactly, every
    generation; generation 1 bit-exact against a direct simulate call;
    default adopt_policy is certified; frozen-suite pytest counts."""
    out: dict = {}
    log("part 0 START: fidelity (enabled=False = frozen; default policy "
        "= certified)")
    out["default_adopt_policy"] = GenParams().adopt_policy
    gp = GenParams(enabled=False, T=40.0)
    res = run_generational(gp, 4, keep_sol=True)
    # generation 1 starts at Params defaults: must equal a direct frozen
    # simulate call bit-for-bit
    ref = simulate(P, generation_schedule(), gp.horizon, dt=gp.dt)
    out["gen1_bitexact_maxdG"] = float(
        np.max(np.abs(res["sols"][0]["G"] - ref["G"])))
    out["gen1_bitexact_maxda"] = float(
        np.max(np.abs(res["sols"][0]["a"] - ref["a"])))
    # disabled M: identical every generation (all adoptions empty)
    out["disabled_any_adopted"] = bool(
        any(a != "" for a in res["adopted"]))
    g_end = np.asarray(res["G_end"], float)
    # with the five-state carry, generations 2+ start at the carried
    # end-state (not Params defaults), so the frozen model's functional
    # end is equal to 4 decimals, not bit-exact — the claim reproduced
    out["disabled_G_end_spread"] = float(g_end.max() - g_end.min())
    out["disabled_G_end_all_equal"] = bool(
        np.allclose(g_end, g_end[0], rtol=0.0, atol=1e-4))
    out["disabled_G_end"] = float(g_end.mean())
    # the frozen G_min floor for criterion C1: the disabled control's
    # own worst episode dip (the unmodified agent's collapse-level dip)
    out["disabled_G_min"] = float(min(
        float(sol["G"].min()) for sol in res["sols"]))
    log(f"part 0: gen-1 vs simulate max|dG| = "
        f"{out['gen1_bitexact_maxdG']:.3e} (0 = bit-exact)")
    log(f"part 0: enabled=False over {len(res['adopted'])} generations: "
        f"any adoption = {out['disabled_any_adopted']}, G_end = "
        f"{out['disabled_G_end']:.4f} at every generation (spread "
        f"{out['disabled_G_end_spread']:.2e}, carry-IC effect), "
        f"G_min floor {out['disabled_G_min']:.4f}")
    # arm-A fidelity: the certified default must reproduce the published
    # arm-A numbers (handoff-selfreg-generational-finding)
    log("part 0: arm-A reproduction check (certified default vs the "
        "published arm-A reference)")
    rep_ok = True
    for T, (gmin_ref, gend_ref, quoted) in ARM_A_REF.items():
        r = run_generational(GenParams(T=T), NGEN)
        dmin = abs(float(r["G_min"][-1]) - gmin_ref)
        dend = abs(float(r["G_end"][-1]) - gend_ref)
        cell_ok = dend < 2e-4 and dmin < 2e-4
        if quoted:
            rep_ok &= cell_ok
        out[f"armA_T{T:g}_dGend"] = dend
        out[f"armA_T{T:g}_dGmin"] = dmin
        log(f"part 0:   T={T:6.1f}: G_end {float(r['G_end'][-1]):.4f} "
            f"(ref {gend_ref:.4f}, d={dend:.1e})  G_min "
            f"{float(r['G_min'][-1]):.4f} (ref {gmin_ref:.4f}, "
            f"d={dmin:.1e})  {'ok' if cell_ok else 'DIFFERS'}"
            f"{'' if quoted else ' (inferred cell, logged not asserted)'}")
    out["armA_reproduces"] = bool(rep_ok)
    log(f"part 0: certified default reproduces the published arm-A "
        f"numbers: {rep_ok}")
    # frozen-suite pytest counts (read-only subprocess)
    out["pytest_cmd"] = " ".join(
        [".venv/bin/python", "-m", "pytest", "-q", "tests/"])
    try:
        pr = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", os.path.join(ROOT,
                                                                "tests/")],
            capture_output=True, text=True, timeout=600, cwd=ROOT)
        tail = pr.stdout.strip().splitlines()[-1] if pr.stdout.strip() \
            else (pr.stderr.strip().splitlines()[-1]
                  if pr.stderr.strip() else "")
        out["pytest_returncode"] = int(pr.returncode)
        out["pytest_tail"] = tail
        log(f"part 0: pytest tests/ -> rc={pr.returncode}: {tail}")
    except Exception as e:                                # pragma: no cover
        out["pytest_returncode"] = -1
        out["pytest_tail"] = f"subprocess failed: {e}"
        log(f"part 0: pytest FAILED TO RUN: {e}")
    ok = (out["gen1_bitexact_maxdG"] == 0.0
          and not out["disabled_any_adopted"]
          and out["disabled_G_end_all_equal"]
          and out["default_adopt_policy"] == "certified"
          and rep_ok)
    out["fidelity_ok"] = bool(ok)
    log(f"part 0 DONE: fidelity_ok = {ok}")
    np.savez(os.path.join(CACHE, "exp12_part0.npz"), **out)


# --------------------------------------------------------------- part 1
def part1() -> None:
    """atol sensitivity: the certified adopt-set sequences across
    atol_tau 1e-6..1e-3 at the design T's must be identical (the
    historical claim), and the arm-B tie gate must not drive arm B's
    adopt sets either."""
    log("part 1 START: atol sensitivity "
        f"(atol_tau in {ATOLS[0]:g}..{ATOLS[-1]:g}, both arms)")
    seqs: dict = {}
    for T in (10.0, 40.0, 160.0):
        for at in ATOLS:
            gp = GenParams(T=T, atol_tau=at)
            r = run_generational(gp, NGEN)
            seqs[f"A_T{T:g}_atol{at:g}"] = np.array(r["adopted"],
                                                    dtype=object)
    for at in ATOLS:                       # arm B tie gate, one design T
        gp = GenParams(T=40.0, atol_tau=at, adopt_policy="always")
        r = run_generational(gp, NGEN)
        seqs[f"B_T40_atol{at:g}"] = np.array(r["adopted"], dtype=object)
    out: dict = {"atols": np.array(ATOLS), "Ts": np.array([10.0, 40.0,
                                                           160.0])}
    identical = True
    for T in (10.0, 40.0, 160.0):
        ref = seqs[f"A_T{T:g}_atol{ATOLS[0]:g}"]
        n_adopt = int(np.sum([a != "" for a in ref]))
        for at in ATOLS:
            same = bool(np.array_equal(seqs[f"A_T{T:g}_atol{at:g}"], ref))
            identical &= same
            log(f"part 1: A T={T:g} atol={at:g}: adopt-set sequence "
                f"{'identical' if same else 'DIFFERS'} ({n_adopt} "
                "adoptions)")
        out[f"n_adoptA_T{T:g}"] = n_adopt
    refB = seqs[f"B_T40_atol{ATOLS[0]:g}"]
    for at in ATOLS:
        same = bool(np.array_equal(seqs[f"B_T40_atol{at:g}"], refB))
        identical &= same
        log(f"part 1: B T=40 atol={at:g}: adopt-set sequence "
            f"{'identical' if same else 'DIFFERS'}")
    out["n_adoptB_T40"] = int(np.sum([a != "" for a in refB]))
    out["adopt_sets_atol_insensitive"] = bool(identical)
    log(f"part 1 DONE: adopt sets insensitive across the atol range "
        f"(both arms): {identical}")
    np.savez(os.path.join(CACHE, "exp12_part1.npz"), **out)


# --------------------------------------------------------------- part 2
def _battery_cell(T: float, policy: str):
    """One (T, arm) cell: run NGEN generations, return the record."""
    gp = GenParams(T=T, adopt_policy=policy)
    r = run_generational(gp, NGEN)
    seq = np.array(r["adopted"], dtype=object)
    marg = np.asarray(r["margin"], float)
    n_adopt = int(np.sum([a != "" for a in seq]))
    # uncertified adoptions: adopted while the margin was NOT certified
    # (>= +atol).  Impossible in arm A by construction; the count that
    # matters for arm B.
    n_uncert = int(np.sum([(a != "") and (m >= gp.atol_tau)
                           for a, m in zip(seq, marg)]))
    act = [m for a, m in zip(seq, marg) if a != ""]
    return dict(gp=gp, r=r, T=T, policy=policy,
                G_end=float(r["G_end"][-1]),
                G_min_final=float(r["G_min"][-1]),
                G_min_worst=float(np.min(r["G_min"])),
                E_end=float(r["E_end"][-1]),
                seq=seq, margin=marg, n_adopt=n_adopt,
                n_uncert=n_uncert,
                # closest approach to fallibility: the LEAST negative
                # margin the walk ever acted on (how close the best
                # guess came to being uncertified), and the most
                # positive best-margin ever seen at any decision
                min_act_margin=(float(np.min(act)) if act
                                else float("nan")),
                max_best_margin=float(np.max(marg)),
                G_end_curve=np.asarray(r["G_end"], float),
                G_min_curve=np.asarray(r["G_min"], float),
                changed=_changed_fields(r["params_final"]),
                logdist=_log_distance(r["params_final"]))


def part2() -> None:
    """THE TWO-ARM BATTERY: T-grid x 24 generations x {A, B}."""
    log(f"part 2 START: the two-arm battery — {len(T_GRID)} capacities x "
        f"{NGEN} generations x arms {ARMS}")
    z0 = np.load(os.path.join(CACHE, "exp12_part0.npz"),
                 allow_pickle=True)
    frozen_end = float(z0["disabled_G_end"])
    frozen_min = float(z0["disabled_G_min"])
    for arm in ARMS:
        Ts = np.array(T_GRID)
        cells = []
        for T in T_GRID:
            c = _battery_cell(T, POLICY[arm])
            cells.append(c)
            log(f"part 2 arm {arm}: T={T:6.1f}: G_min(final) "
                f"{c['G_min_final']:.4f}  G_end {c['G_end']:.4f}  "
                f"E_end {c['E_end']:+.4f}  adoptions {c['n_adopt']:2d} "
                f"(uncertified {c['n_uncert']:2d})  log-dist "
                f"{c['logdist']:.3f}  closest-margin "
                f"{c['min_act_margin']:+.2e}")
        np.savez(os.path.join(CACHE, f"exp12_battery{arm}.npz"),
                 T=Ts,
                 G_end=np.array([c["G_end"] for c in cells]),
                 G_min_final=np.array([c["G_min_final"] for c in cells]),
                 G_min_worst=np.array([c["G_min_worst"] for c in cells]),
                 E_end=np.array([c["E_end"] for c in cells]),
                 n_adopt=np.array([c["n_adopt"] for c in cells], int),
                 n_uncert=np.array([c["n_uncert"] for c in cells], int),
                 min_act_margin=np.array([c["min_act_margin"]
                                          for c in cells]),
                 max_best_margin=np.array([c["max_best_margin"]
                                           for c in cells]),
                 logdist=np.array([c["logdist"] for c in cells]),
                 frozen_G_end=frozen_end, frozen_G_min=frozen_min,
                 n_gen=NGEN, policy=POLICY[arm],
                 levers=np.array(DEFAULT_LEVERS, dtype=object),
                 changed=np.array([repr(c["changed"]) for c in cells],
                                  dtype=object),
                 **{f"seq_T{T:g}": c["seq"] for T, c in zip(T_GRID,
                                                            cells)},
                 **{f"Gendcurve_T{T:g}": c["G_end_curve"]
                    for T, c in zip(T_GRID, cells)},
                 **{f"Gmincurve_T{T:g}": c["G_min_curve"]
                    for T, c in zip(T_GRID, cells)},
                 **{f"margin_T{T:g}": c["margin"]
                    for T, c in zip(T_GRID, cells)})
        log(f"part 2 arm {arm} DONE: cached -> cache/exp12_battery"
            f"{arm}.npz")


# --------------------------------------------------------------- part 3
def _degrades(T: float, policy: str, frozen_end: float,
              frozen_min: float, n_gen: int = NGEN) -> dict:
    """The pre-registered degradation predicate C1/C2 for one cell
    (re-run live; used by the bisect if it is ever needed)."""
    c = _battery_cell(T, policy)
    ge = c["G_end_curve"]
    gm = c["G_min_curve"]
    return dict(c1_end=bool(np.any(ge < frozen_end - DEG_TOL)),
                c1_min=bool(np.min(gm) < frozen_min - DEG_TOL),
                c2=bool(np.any(ge < ge[0] - DEG_TOL)))


def part3() -> None:
    """THE VERDICT: degradation per arm per T under the pre-registered
    criterion (C1 frozen floor / C2 self floor, DEG_TOL = 1e-3), the
    T_min bisect (only if the degradation pattern demands it), the
    same-Params comparison, and the accumulation-trap verdict."""
    log("part 3 START: the two-arm verdict (criterion C1/C2, "
        f"DEG_TOL={DEG_TOL:g}, stated in the pre-registration BEFORE "
        "the battery)")
    zA = np.load(os.path.join(CACHE, "exp12_batteryA.npz"),
                 allow_pickle=True)
    zB = np.load(os.path.join(CACHE, "exp12_batteryB.npz"),
                 allow_pickle=True)
    T = zA["T"]
    frozen_end = float(zA["frozen_G_end"])
    frozen_min = float(zA["frozen_G_min"])
    out: dict = {"DEG_TOL": DEG_TOL, "frozen_G_end": frozen_end,
                 "frozen_G_min": frozen_min}
    # ---- per-cell verdicts, arm B (and A for honesty) under C1/C2
    for arm, z in (("A", zA), ("B", zB)):
        c1_end, c1_min, c2 = [], [], []
        for i, Ti in enumerate(T):
            ge = z[f"Gendcurve_T{Ti:g}"]
            gm = z[f"Gmincurve_T{Ti:g}"]
            c1_end.append(bool(np.any(ge < frozen_end - DEG_TOL)))
            c1_min.append(bool(np.min(gm) < frozen_min - DEG_TOL))
            c2.append(bool(np.any(ge < ge[0] - DEG_TOL)))
        out[f"{arm}_c1_end"] = np.array(c1_end)
        out[f"{arm}_c1_min"] = np.array(c1_min)
        out[f"{arm}_c2"] = np.array(c2)
        out[f"{arm}_any_degrades"] = bool(
            np.any(c1_end) or np.any(c1_min) or np.any(c2))
        for i, Ti in enumerate(T):
            log(f"part 3 arm {arm}: T={Ti:6.1f}: C1end={c1_end[i]} "
                f"C1min={c1_min[i]} C2={c2[i]}")
        log(f"part 3 arm {arm}: ANY degradation (C1 or C2, any T): "
            f"{out[f'{arm}_any_degrades']}")
    # ---- does a LOWER THRESHOLD exist?  (some low T degrades, top of
    # the grid does not -> bisect; no T degrades -> T_min does not
    # exist; all T degrade -> no threshold, fallibility unbounded)
    b_deg = out["B_c1_end"] | out["B_c1_min"] | out["B_c2"]
    if not np.any(b_deg):
        t_min, bisect_ran = None, False
        log("part 3: NO capacity degrades -> T_min DOES NOT EXIST; no "
            "bisect performed (a vacuous bisection is worse than a "
            "plain negative)")
    elif not np.any(b_deg[-2:]):
        t_min, bisect_ran = None, True
        log("part 3: low-T degradation with clean top-of-grid -> "
            "bisecting T_min on [10, 640]")
        lo, hi = 10.0, 640.0
        for _ in range(8):                      # resolution ~2.5 t.u.
            mid = 0.5 * (lo + hi)
            d = _degrades(mid, "always", frozen_end, frozen_min)
            if (d["c1_end"] or d["c1_min"] or d["c2"]):
                hi = mid
            else:
                lo = mid
            log(f"part 3: bisect [{lo:.2f}, {hi:.2f}] (mid {mid:.2f} "
                f"degrades={d['c1_end'] or d['c1_min'] or d['c2']})")
        t_min = hi
        log(f"part 3: T_min = {t_min:.2f}")
    else:
        t_min, bisect_ran = None, False
        log("part 3: degradation at the TOP of the grid too -> no "
            "threshold structure (fallibility unbounded on this grid); "
            "no bisect")
    out["B_T_min"] = np.nan if t_min is None else t_min
    out["B_bisect_ran"] = bisect_ran
    # ---- same-Params comparison + wander
    same = []
    for i, Ti in enumerate(T):
        chA = eval(zA["changed"][i])          # noqa: S307 (own cache)
        chB = eval(zB["changed"][i])          # noqa: S307
        same.append(chA == chB)
        if chA != chB:
            log(f"part 3: T={Ti:g}: arms END AT DIFFERENT PARAMS:\n"
                f"    A {chA}\n    B {chB}")
    out["same_params"] = np.array(same)
    out["all_same_params"] = bool(np.all(same))
    out["logdist_A"] = zA["logdist"]
    out["logdist_B"] = zB["logdist"]
    log(f"part 3: arms end at the same Params at every T: "
        f"{out['all_same_params']}; log-dist A "
        f"{np.array2string(out['logdist_A'], precision=2)} vs B "
        f"{np.array2string(out['logdist_B'], precision=2)}")
    # ---- the ACCUMULATION-trap verdict
    n_uncert_B = int(np.sum(zB["n_uncert"]))
    seqs_equal = []
    for Ti in T:
        seqs_equal.append(bool(np.array_equal(zA[f"seq_T{Ti:g}"],
                                              zB[f"seq_T{Ti:g}"])))
    out["seqs_equal_perT"] = np.array(seqs_equal)
    out["B_n_uncertified_total"] = n_uncert_B
    out["B_max_best_margin"] = float(np.max(zB["max_best_margin"]))
    if out["B_any_degrades"]:
        accum = ("DEGRADED — a fallible walk landed worse; inspect "
                 "whether per-step margins stayed negative (accumulation) "
                 "or flipped (per-step)")
    elif n_uncert_B == 0 and out["all_same_params"]:
        accum = ("NO TRAP OF EITHER KIND: arm B never had to act on an "
                 "uncertified judgement (every argmin V margin stayed "
                 "certified across all decisions), so no per-step trap "
                 "was ever met AND no accumulation could begin — the "
                 "arms coincide BY LANDSCAPE, not by construction of "
                 "arm B's rule")
    else:
        accum = ("NO ACCUMULATION TRAP: arm B acted on uncertified "
                 "judgements (or wandered past arm A) yet never "
                 "degraded — the landscape is benign beyond the "
                 "certified region")
    out["accumulation_verdict"] = accum
    log(f"part 3: arm-B uncertified adoptions (all T, all decisions): "
        f"{n_uncert_B}; most-positive best-margin ever seen "
        f"{out['B_max_best_margin']:+.2e}")
    log(f"part 3: ACCUMULATION-TRAP VERDICT: {accum}")
    log("part 3 DONE: cached -> cache/exp12_verdict.npz")
    np.savez(os.path.join(CACHE, "exp12_verdict.npz"), **out)


# --------------------------------------------------------------- figure
def part_fig() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    log("fig START: f16_generational.png (two arms)")
    zA = np.load(os.path.join(CACHE, "exp12_batteryA.npz"),
                 allow_pickle=True)
    zB = np.load(os.path.join(CACHE, "exp12_batteryB.npz"),
                 allow_pickle=True)
    z0 = np.load(os.path.join(CACHE, "exp12_part0.npz"),
                 allow_pickle=True)
    T = zA["T"]
    frozen = float(zA["frozen_G_end"])
    frozen_min = float(zA["frozen_G_min"])
    cA, cB = "#2166ac", "#b2182b"
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 9.0))

    # (a) G_end vs capacity — both arms against the frozen line
    ax = axs[0, 0]
    ax.plot(T, zA["G_end"], "o-", color=cA,
            label="arm A certified (baseline)")
    ax.plot(T, zB["G_end"], "s--", color=cB,
            label="arm B always-adopt (fallible)")
    ax.axhline(frozen, color="k", ls="--", lw=1.0,
               label=f"frozen (enabled=False) $G_{{end}}$={frozen:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("capacity $T$ [t.u.]")
    ax.set_ylabel("$G_{end}$ (gen 24)")
    ax.set_title("(a) final functional level: both arms, all capacities")
    ax.legend(fontsize=8)

    # (b) worst dip vs capacity — both arms
    ax = axs[0, 1]
    ax.plot(T, zA["G_min_worst"], "o-", color=cA, label="arm A")
    ax.plot(T, zB["G_min_worst"], "s--", color=cB, label="arm B")
    ax.axhline(frozen_min, color="k", ls="--", lw=1.0,
               label=f"frozen floor {frozen_min:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("capacity $T$ [t.u.]")
    ax.set_ylabel("$\\min_k G_{\\min}^{(k)}$ (worst dip)")
    ax.set_title("(b) worst episode dip vs the frozen floor (criterion "
                 "C1)")
    ax.legend(fontsize=8)

    # (c) per-generation G curves at low/high capacity, both arms
    ax = axs[1, 0]
    for Ti, ls in ((T_GRID[0], "-"), (T_GRID[-1], "--")):
        ax.plot(np.arange(1, NGEN + 1), zA[f"Gendcurve_T{Ti:g}"], ls,
                color=cA, ms=3, label=f"A $T$={Ti:g}")
        ax.plot(np.arange(1, NGEN + 1), zB[f"Gendcurve_T{Ti:g}"], ls,
                color=cB, ms=3, label=f"B $T$={Ti:g}")
    ax.axhline(frozen, color="k", ls=":", lw=1.0, label="frozen")
    ax.set_xlabel("generation $k$")
    ax.set_ylabel("$G_{end}^{(k)}$")
    ax.set_title("(c) per-generation curves: compounding in both arms")
    ax.legend(fontsize=7)

    # (d) adoption counts + uncertified (the fallibility meter)
    ax = axs[1, 1]
    x = np.arange(len(T))
    w = 0.38
    ax.bar(x - w / 2, zA["n_adopt"], w, color=cA, label="A adoptions")
    ax.bar(x + w / 2, zB["n_adopt"], w, color=cB, label="B adoptions")
    ax.bar(x + w / 2, zB["n_uncert"], w, facecolor="none",
           edgecolor="k", hatch="///",
           label="B uncertified ($m \\geq$ atol)")
    ax.set_xticks(x, [f"{t:g}" for t in T])
    ax.set_xlabel("capacity $T$")
    ax.set_ylabel(f"adoptions in {NGEN} generations")
    ax.set_title("(d) adoption frequency; hatched = fallible acts")
    ax.legend(fontsize=8)

    zv = np.load(os.path.join(CACHE, "exp12_verdict.npz"),
                 allow_pickle=True)
    fig.suptitle("exp12 — two arms: V-certified baseline (A) vs "
                 "always-adopt fallible agent (B); any degradation = "
                 f"{bool(zv['B_any_degrades'])}, uncertified adoptions "
                 f"= {int(zv['B_n_uncertified_total'])}", fontsize=10.5)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, "f16_generational.png")
    fig.savefig(out, dpi=140)
    # programmatic verification (never opened with read)
    w_px, h_px = Image.open(out).size
    arr = np.asarray(Image.open(out).convert("L"))
    ink = float((arr < 250).mean())
    log(f"fig DONE -> {out} ({w_px}x{h_px}px, aspect {w_px/h_px:.2f}, "
        f"ink {ink:.3f})")
    assert w_px > 1000 and h_px > 800, f"bad size {w_px}x{h_px}"
    assert 0.005 < ink < 0.9, f"bad ink {ink}"
    assert abs(w_px / h_px - 11.5 / 9.0) < 0.05


# ------------------------------------------------- unregistered extension
def part_ext() -> None:
    """UNREGISTERED HORIZON EXTENSION (labeled as such everywhere; run
    AFTER the battery + verdict, motivated by their outcome).  Within
    the registered 24-generation window the two arms coincide: every
    best-margin stayed certified (closest -2.9e-2, most positive ever
    -6.2e-3), so arm B never acted on an uncertified judgement.  But the
    per-decision margins DECAY as the walk proceeds, so the first
    fallible act may simply lie beyond the window.  This part runs the
    SAME rule (nothing tuned) for 96 generations at the grid extremes
    and records: the first generation where arm B acts on an uncertified
    margin (m >= +atol), where arm A halts (first m >= -atol), whether
    the arms diverge, and whether arm B degrades afterwards (same C1/C2
    criterion)."""
    log("part ext START: UNREGISTERED horizon extension — 96 "
        "generations at T in (10, 640), same rule, nothing tuned")
    z0 = np.load(os.path.join(CACHE, "exp12_part0.npz"),
                 allow_pickle=True)
    frozen_end = float(z0["disabled_G_end"])
    frozen_min = float(z0["disabled_G_min"])
    out: dict = {"n_gen": 96, "frozen_G_end": frozen_end,
                 "frozen_G_min": frozen_min,
                 "registered": False}
    for T in (10.0, 640.0):
        cells = {}
        for arm in ARMS:
            gp = GenParams(T=T, adopt_policy=POLICY[arm])
            r = run_generational(gp, 96)
            seq = np.array(r["adopted"], dtype=object)
            marg = np.asarray(r["margin"], float)
            cells[arm] = dict(
                seq=seq, marg=marg,
                G_end=np.asarray(r["G_end"], float),
                G_min=np.asarray(r["G_min"], float),
                n_adopt=int(np.sum([a != "" for a in seq])),
                n_uncert=int(np.sum([(a != "") and (m >= gp.atol_tau)
                                     for a, m in zip(seq, marg)])),
                first_uncert=int(next(
                    (k for k, (a, m) in enumerate(zip(seq, marg))
                     if a != "" and m >= gp.atol_tau), -1)),
                first_Ahalt=int(next(
                    (k for k, m in enumerate(marg) if m >= -gp.atol_tau),
                    -1)),
                changed=_changed_fields(r["params_final"]),
                logdist=_log_distance(r["params_final"]))
        A, B = cells["A"], cells["B"]
        div = int(next((k for k in range(96) if A["seq"][k] != B["seq"][k]),
                       -1))
        out[f"T{T:g}_A_nadopt"] = A["n_adopt"]
        out[f"T{T:g}_B_nadopt"] = B["n_adopt"]
        out[f"T{T:g}_B_nuncert"] = B["n_uncert"]
        out[f"T{T:g}_B_first_uncert"] = B["first_uncert"]
        out[f"T{T:g}_A_first_noncert"] = A["first_Ahalt"]
        out[f"T{T:g}_diverge_gen"] = div
        out[f"T{T:g}_B_Gend_min"] = float(B["G_end"].min())
        out[f"T{T:g}_B_Gmin_worst"] = float(B["G_min"].min())
        out[f"T{T:g}_B_c1"] = bool(
            np.any(B["G_end"] < frozen_end - DEG_TOL)
            or B["G_min"].min() < frozen_min - DEG_TOL)
        out[f"T{T:g}_B_c2"] = bool(
            np.any(B["G_end"] < B["G_end"][0] - DEG_TOL))
        out[f"T{T:g}_A_Gend_last"] = float(A["G_end"][-1])
        out[f"T{T:g}_B_Gend_last"] = float(B["G_end"][-1])
        out[f"T{T:g}_same_params"] = A["changed"] == B["changed"]
        out[f"T{T:g}_A_changed"] = repr(A["changed"])
        out[f"T{T:g}_B_changed"] = repr(B["changed"])
        out[f"T{T:g}_A_marg"] = A["marg"]
        out[f"T{T:g}_B_marg"] = B["marg"]
        out[f"T{T:g}_A_Gend"] = A["G_end"]
        out[f"T{T:g}_B_Gend"] = B["G_end"]
        out[f"T{T:g}_B_Gmin"] = B["G_min"]
        out[f"T{T:g}_B_seq"] = B["seq"]
        log(f"part ext T={T:g}: A adopts {A['n_adopt']}/96 (halts at "
            f"gen {A['first_Ahalt'] + 1 if A['first_Ahalt'] >= 0 else 'never'}"
            f"), B adopts {B['n_adopt']}/96, B uncertified "
            f"{B['n_uncert']} (first at gen "
            f"{B['first_uncert'] + 1 if B['first_uncert'] >= 0 else 'never'}"
            f"), arms diverge at gen "
            f"{div + 1 if div >= 0 else 'never'}")
        log(f"part ext T={T:g}: B G_end last {B['G_end'][-1]:.4f} "
            f"(min {B['G_end'].min():.4f}), worst dip "
            f"{B['G_min'].min():.4f} vs frozen floor {frozen_min:.4f}; "
            f"C1={out[f'T{T:g}_B_c1']} C2={out[f'T{T:g}_B_c2']}; "
            f"same final Params = {A['changed'] == B['changed']}")
        if A["changed"] != B["changed"]:
            log(f"part ext T={T:g}: A end {A['changed']}")
            log(f"part ext T={T:g}: B end {B['changed']}")
    log("part ext DONE: cached -> cache/exp12_extension.npz "
        "(UNREGISTERED — interpret accordingly)")
    np.savez(os.path.join(CACHE, "exp12_extension.npz"), **out)


PARTS = {"0": part0, "1": part1, "2": part2, "3": part3,
         "fig": part_fig, "ext": part_ext}


def main(argv: list[str]) -> None:
    os.makedirs(CACHE, exist_ok=True)
    parts = argv[1:] or ["0", "1", "2", "3", "fig"]
    if "all" in parts:
        parts = ["0", "1", "2", "3", "fig"]
    t0 = time.time()
    for name in parts:
        if name not in PARTS:
            raise SystemExit(f"unknown part {name!r} "
                             f"(one of {sorted(PARTS)})")
        PARTS[name]()
    log(f"exp12 complete ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main(sys.argv)
