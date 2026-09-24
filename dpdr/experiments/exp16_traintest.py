"""exp16 — THE TRAIN/TEST GAP (handoff-selfreg-exp16), the operative
post-exp15 experiment.

WHY THIS ONE, AFTER exp15: exp15's blind lever had a V-margin of
EXACTLY 0.0 at every decision — an exact tie with the no-op — so the
agent's argmin + tie rule REFUSED it structurally (0 blind adoptions).
For a maximising self-modifier that rejects ties, INVISIBILITY AND
ADOPTABILITY ARE MUTUALLY EXCLUSIVE; the mechanism the ODE could still
host is MISJUDGMENT — a NONZERO apparent improvement that is harmful in
truth.  A train/test gap supplies exactly that by construction: V is
evaluated in the TRAIN context (the agent lives and self-modifies
there), the outcome is realised in the TEST context (measured there).
This is class (C) — the sound-but-non-representative evaluator — the
one collapse class with replicated external evidence (Lin, arXiv
2606.21090: optimised metric rises while pass@1 falls).

THE REGISTERED PAIR (pre-registered in exp16_preregistration.md BEFORE
the first run; the anti-tuning rule is absolute):
  TRAIN = the canonical generation task (inward episode [100,200),
          a_hold 0.9 + affect pulse 0.5 to 160, rescue u_ext 0.8 on
          [800,1000)) — the published system, modify_once unchanged.
  TEST  = the SAME episode with the EXTERNAL RESCUE REMOVED.  Criterion:
          the test task is the train task minus EXOGENOUS HELP, agent
          machinery untouched — a genuine generalisation test (handle
          unaided what was tuned with aid on), the brief's option (b).
  OPT-IN: GenParams.schedule_override (default None = published system
          exactly; every existing cache stays reproducible — part 0).

Parts:
  0  fidelity: (i) train-evaluated walks reproduce the published exp12
     arm-A/arm-B caches EXACTLY (sequence + G_end curves); (ii) default
     schedule_override is None and exp15's blind arm-B cache stays
     reproducible; (iii) enabled=False bit-exact vs simulate on BOTH
     geometries; (iv) pytest count re-checked.
  1  THE TRAIN WALKS + P2: per (T, arm): adopted sequence, margin
     distribution of ADOPTED moves, adoption counts — and the explicit
     contrast with exp15 (margin exactly 0.0, 0 adoptions).  The train
     walks themselves ARE the exp12 caches (re-derived, not re-cached
     twice); what is cached is the P2 statistic block.
  2  THE TEST BATTERY (the headline): the frozen driver integrates the
     train-walk Params at every generation k on the TEST schedule;
     controls = the same Params walks under enabled=False, TEST- and
     TRAIN-evaluated; P3 criteria C1end/C1min/C2 with DEG_TOL = 1e-3
     (REUSED from exp12/exp15).
  3  THE VERDICT: P1-P5 + diagnostics (lever families under the split,
     final-Params log-distance, near-tie census) + the commutativity
     check.
  4  THE DIVERGENCE CURVE: per generation, optimised (V's certified
     margin on train; train G_end) against true (test G_min/G_end) —
     Lin's rise-then-collapse analogue.
  fig  f20_traintest.png (verified programmatically — never opened).

Usage:  .venv/bin/python -m experiments.exp16_traintest [0|1|2|3|4|fig|all]
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np

from dpdr.generational import (DEFAULT_LEVERS, EP_T0, EP_T1, PULSE_END,
                               GenParams, _alarm_state, generation_schedule,
                               modify_once, run_generational)
from dpdr.integrate import simulate
from dpdr.model import Params, Schedule

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
FIGDIR = os.path.join(ROOT, "figs")

P = Params()
NGEN = 24                    # generations per cell (the two-arm window)
T_GRID = (10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0)
ARMS = ("A", "B")
POLICY = {"A": "certified", "B": "always"}
DEG_TOL = 1e-3              # REUSED from exp12/exp15 (not re-invented)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def test_schedule() -> Schedule:
    """The registered TEST geometry: the canonical episode with the
    EXTERNAL RESCUE REMOVED (exogenous-help removal; the agent's own
    machinery is untouched — see exp16_preregistration.md)."""
    return Schedule({"a_hold": [(EP_T0, EP_T1, 0.9)],
                     "A": [(EP_T0, PULSE_END, 0.5)]})


def _gp(T: float, policy: str = "certified", **kw) -> GenParams:
    return GenParams(T=T, adopt_policy=policy, **kw)


# --------------------------------------------------------------- part 0
def part0() -> None:
    """Fidelity: the OPT-IN split leaves every published number
    reproducible."""
    out: dict = {}
    log("part 0 START: fidelity (schedule_override default None = the "
        "published two-arm system, exactly)")
    out["default_schedule_override"] = repr(
        GenParams().schedule_override)
    log(f"part 0: GenParams().schedule_override = "
        f"{out['default_schedule_override']} (None = canonical)")
    # (iii-a) enabled=False bit-exact vs the frozen driver, TRAIN geom
    gp = GenParams(enabled=False, T=40.0)
    res = run_generational(gp, 4, keep_sol=True)
    ref = simulate(P, generation_schedule(), gp.horizon, dt=gp.dt)
    out["train_gen1_bitexact_maxdG"] = float(np.max(np.abs(
        res["sols"][0]["G"] - ref["G"])))
    out["train_disabled_any_adopted"] = bool(
        any(a != "" for a in res["adopted"]))
    # (iii-b) the same on the TEST geometry
    gpT = GenParams(enabled=False, T=40.0,
                    schedule_override=test_schedule())
    resT = run_generational(gpT, 4, keep_sol=True)
    refT = simulate(P, test_schedule(), gpT.horizon, dt=gpT.dt)
    out["test_gen1_bitexact_maxdG"] = float(np.max(np.abs(
        resT["sols"][0]["G"] - refT["G"])))
    out["test_disabled_any_adopted"] = bool(
        any(a != "" for a in resT["adopted"]))
    out["test_disabled_G_end"] = float(
        np.asarray(resT["G_end"], float)[-1])
    out["test_disabled_G_min"] = float(min(
        float(sol["G"].min()) for sol in resT["sols"]))
    log(f"part 0: enabled=False bit-exact vs simulate — TRAIN max|dG| "
        f"{out['train_gen1_bitexact_maxdG']:.1e} (any adoption "
        f"{out['train_disabled_any_adopted']}), TEST max|dG| "
        f"{out['test_gen1_bitexact_maxdG']:.1e} (any adoption "
        f"{out['test_disabled_any_adopted']}); frozen ON TEST: G_end "
        f"{out['test_disabled_G_end']:.4f}, G_min "
        f"{out['test_disabled_G_min']:.4f}")
    # (i) the train walks reproduce the PUBLISHED exp12 caches EXACTLY
    for arm in ARMS:
        z = np.load(os.path.join(CACHE, f"exp12_battery{arm}.npz"),
                    allow_pickle=True)
        max_d_curve = 0.0
        seqs_ok = True
        for Ti in z["T"]:
            r = run_generational(_gp(float(Ti), POLICY[arm]), NGEN)
            d_curve = float(np.max(np.abs(
                np.asarray(r["G_end"], float)
                - z[f"Gendcurve_T{Ti:g}"])))
            seqs_ok &= bool(np.array_equal(
                np.array(r["adopted"], dtype=object),
                z[f"seq_T{Ti:g}"]))
            max_d_curve = max(max_d_curve, d_curve)
        out[f"exp12_arm{arm}_max_dGendcurve"] = max_d_curve
        out[f"exp12_arm{arm}_seqs_equal"] = bool(seqs_ok)
        log(f"part 0: train walk (override=None) reproduces exp12 "
            f"arm-{arm} cache: max|dG_end curve| {max_d_curve:.1e}, "
            f"sequences identical {seqs_ok}")
    # (ii) exp15's blind arm-B cache stays reproducible (the split does
    # not touch exp15's behaviour)
    zb = np.load(os.path.join(CACHE, "exp15_batteryB_blind.npz"),
                 allow_pickle=True)
    max_d_curve_b = 0.0
    seqs_ok_b = True
    for Ti in zb["T"]:
        r = run_generational(GenParams(T=float(Ti), adopt_policy="always",
                                       blind_levers=("k_ext",)), NGEN)
        d = float(np.max(np.abs(np.asarray(r["G_end"], float)
                                - zb[f"Gendcurve_T{Ti:g}"])))
        seqs_ok_b &= bool(np.array_equal(
            np.array(r["adopted"], dtype=object), zb[f"seq_T{Ti:g}"]))
        max_d_curve_b = max(max_d_curve_b, d)
    out["exp15_blindB_max_dGendcurve"] = max_d_curve_b
    out["exp15_blindB_seqs_equal"] = bool(seqs_ok_b)
    log(f"part 0: exp15 blind arm-B cache still reproducible: "
        f"max|dG_end curve| {max_d_curve_b:.1e}, sequences identical "
        f"{seqs_ok_b}")
    # (iv) pytest count re-checked
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
    ok = (GenParams().schedule_override is None
          and out["train_gen1_bitexact_maxdG"] == 0.0
          and out["test_gen1_bitexact_maxdG"] == 0.0
          and not out["train_disabled_any_adopted"]
          and not out["test_disabled_any_adopted"]
          and all(out[f"exp12_arm{a}_max_dGendcurve"] == 0.0
                  and out[f"exp12_arm{a}_seqs_equal"] for a in ARMS)
          and out["exp15_blindB_max_dGendcurve"] == 0.0
          and out["exp15_blindB_seqs_equal"]
          and out["pytest_returncode"] == 0)
    out["fidelity_ok"] = bool(ok)
    log(f"part 0 DONE: fidelity_ok = {ok}")
    np.savez(os.path.join(CACHE, "exp16_part0.npz"), **out)


# --------------------------------------------------------------- part 1
def _train_walk(T: float, policy: str) -> dict:
    """One TRAIN walk (override None = the published system); returns
    the walk plus the P2 statistics (margins of ADOPTED moves)."""
    gp = _gp(T, policy)
    r = run_generational(gp, NGEN)
    seq = np.array(r["adopted"], dtype=object)
    marg = np.asarray(r["margin"], float)
    adopted_marg = np.array([m for a, m in zip(seq, marg) if a != ""],
                            float)
    return dict(gp=gp, T=T, policy=policy, seq=seq, margin=marg,
                adopted_marg=adopted_marg,
                n_adopt=int(np.sum([a != "" for a in seq])),
                n_adopt_certified=int(np.sum(
                    [(a != "") and (m < -gp.atol_tau)
                     for a, m in zip(seq, marg)])),
                G_end=np.asarray(r["G_end"], float),
                G_min=np.asarray(r["G_min"], float),
                params_final=r["params_final"],
                params_per_gen=_params_per_gen(r),
                r=r)


def _params_per_gen(r: dict) -> list:
    """The Params the walk CARRIES INTO each generation k (k = 1..n):
    gen 1 runs the frozen Params; gen k+1 runs the Params after k
    adopted moves.  Rebuilt by replaying the recorded adoptions on the
    frozen Params (pure Params arithmetic — no integration)."""
    p = Params()
    out = [p]
    for a in r["adopted"]:
        if a != "":
            lever = a[:-1]
            sgn = 1.0 if a[-1] == "+" else -1.0
            p = replace(p, **{lever: getattr(p, lever) * (1.15 ** sgn)})
        out.append(p)
    return out


def part1() -> None:
    """P2: the precondition is met — V genuinely acts on TRAIN; the
    exp15 contrast, numerically."""
    log(f"part 1 START: the TRAIN walks + P2 (margins NONZERO, "
        f"adoptions > 0 — vs exp15's exact-0.0 / 0 adoptions)")
    out: dict = {"T": np.array(T_GRID), "n_gen": NGEN}
    all_m, all_n, all_ncert = [], [], []
    for arm in ARMS:
        for T in T_GRID:
            w = _train_walk(T, POLICY[arm])
            am = w["adopted_marg"]
            all_m.append(am)
            all_n.append(w["n_adopt"])
            all_ncert.append(w["n_adopt_certified"])
            out[f"{arm}_T{T:g}_n_adopt"] = w["n_adopt"]
            out[f"{arm}_T{T:g}_n_certified"] = w["n_adopt_certified"]
            out[f"{arm}_T{T:g}_margin_min"] = float(np.min(am)) \
                if am.size else float("nan")
            out[f"{arm}_T{T:g}_margin_med"] = float(np.median(am)) \
                if am.size else float("nan")
            out[f"{arm}_T{T:g}_margin_max"] = float(np.max(am)) \
                if am.size else float("nan")
            log(f"part 1 arm {arm}: T={T:6.1f}: adoptions "
                f"{w['n_adopt']:2d} (certified {w['n_adopt_certified']:2d})"
                f"  |margin| min {np.abs(am).min():.2e} med "
                f"{np.median(np.abs(am)):.2e} max {np.abs(am).max():.2e}"
                f"  first moves {list(w['seq'][:6])}")
    cat = np.concatenate(all_m)
    out["margin_min_all"] = float(cat.min())
    out["margin_median_all"] = float(np.median(cat))
    out["margin_max_all"] = float(cat.max())
    out["margin_min_abs_all"] = float(np.abs(cat).min())
    out["n_adopt_total"] = int(np.sum(all_n))
    out["n_certified_total"] = int(np.sum(all_ncert))
    out["atol_tau"] = GenParams().atol_tau
    # THE CONTRAST WITH EXP15 (cached numbers, quoted verbatim)
    z15 = np.load(os.path.join(CACHE, "exp15_verdict.npz"),
                  allow_pickle=True)
    z1_15 = np.load(os.path.join(CACHE, "exp15_part1.npz"),
                    allow_pickle=True)
    out["exp15_blind_margin_max_abs"] = float(
        z1_15["blind_margin_max_abs"])
    out["exp15_blind_adoptions_A"] = int(z15["blind_adoptions_A"])
    out["exp15_blind_adoptions_B"] = int(z15["blind_adoptions_B"])
    out["exp15_plant_margin_min"] = float(z1_15["plant_margin_min"])
    out["P2"] = bool(out["n_adopt_total"] > 0
                     and out["margin_min_abs_all"] > GenParams().atol_tau)
    log(f"part 1: POOLED over {len(T_GRID)} T x {NGEN} gens x "
        f"{len(ARMS)} arms: {out['n_adopt_total']} adoptions "
        f"({out['n_certified_total']} certified); adopted-move |margin| "
        f"min {out['margin_min_abs_all']:.2e}, median "
        f"{np.median(np.abs(cat)):.2e}, max {np.abs(cat).max():.2e}")
    log(f"part 1: THE CONTRAST — exp16 adopted moves have |margin| >= "
        f"{out['margin_min_abs_all']:.1e} (>> atol_tau="
        f"{GenParams().atol_tau:g}) and {out['n_adopt_total']} "
        f"ADOPTIONS; exp15's blind lever had margin EXACTLY "
        f"{out['exp15_blind_margin_max_abs']:.1e} at every T and "
        f"{out['exp15_blind_adoptions_B']} adoptions in arm B "
        f"({out['exp15_blind_adoptions_A']} in arm A).  Blindness is "
        f"inert; misjudgment candidates are not.")
    log(f"part 1: P2 (V genuinely acts on TRAIN) = {out['P2']}")
    log("part 1 DONE: cached -> cache/exp16_part1.npz")
    np.savez(os.path.join(CACHE, "exp16_part1.npz"), **out)


# --------------------------------------------------------------- part 2
def _test_eval(p: Params) -> dict:
    """Measure ONE Params on the registered TEST geometry (frozen
    driver, no self-modification): G_min, G_end, E_end."""
    sol = simulate(p, test_schedule(), GenParams().horizon,
                   dt=GenParams().dt)
    return dict(G_min=float(sol["G"].min()), G_end=float(sol["G"][-1]),
                E_end=float(sol["E"][-1]))


def _test_walk(T: float, policy: str) -> dict:
    """The TEST-schedule walk: the agent lives and self-modifies ON the
    TEST task (diagnostic (i)/(ii) — what the walk looks like when the
    rescue is absent from TRAINING too)."""
    gp = _gp(T, policy, schedule_override=test_schedule())
    return run_generational(gp, NGEN)


def part2() -> None:
    """THE HEADLINE: train-optimal Params measured on TEST; the P3
    criteria; the controls."""
    log("part 2 START: the TEST battery — train-walk Params on the "
        "rescue-removed task; controls; C1end/C1min/C2 at DEG_TOL="
        f"{DEG_TOL:g}")
    out: dict = {"T": np.array(T_GRID), "n_gen": NGEN,
                 "DEG_TOL": DEG_TOL}
    # controls measured ON TEST (enabled=False walk = frozen Params at
    # every generation; both geometries)
    ctrl_test = _test_eval(Params())
    out["ctrl_test_G_end"] = ctrl_test["G_end"]
    out["ctrl_test_G_min"] = ctrl_test["G_min"]
    # the frozen agent's TRAIN G_end is exp12's own frozen reference
    ctrl_train_G_end = float(np.load(
        os.path.join(CACHE, "exp12_batteryA.npz"),
        allow_pickle=True)["frozen_G_end"])
    out["ctrl_train_G_end"] = ctrl_train_G_end
    log(f"part 2: controls — frozen ON TEST: G_end "
        f"{ctrl_test['G_end']:.4f}, G_min {ctrl_test['G_min']:.4f}; "
        f"frozen ON TRAIN: G_end {out['ctrl_train_G_end']:.4f} (the gap "
        f"the exogenous help hides from V: "
        f"{out['ctrl_train_G_end'] - ctrl_test['G_end']:.4f})")
    for arm in ARMS:
        for T in T_GRID:
            w = _train_walk(T, POLICY[arm])
            Ges, Gms, Ees = [], [], []
            for p in w["params_per_gen"][:NGEN]:
                m = _test_eval(p)
                Ges.append(m["G_end"])
                Gms.append(m["G_min"])
                Ees.append(m["E_end"])
            Ges, Gms = np.array(Ges), np.array(Gms)
            c1e = bool(np.any(Ges < ctrl_test["G_end"] - DEG_TOL))
            c1m = bool(np.any(Gms < ctrl_test["G_min"] - DEG_TOL))
            c2 = bool(np.any(Ges < Ges[0] - DEG_TOL))
            out[f"{arm}_T{T:g}_testGend"] = Ges
            out[f"{arm}_T{T:g}_testGmin"] = Gms
            out[f"{arm}_T{T:g}_testEend"] = np.array(Ees)
            out[f"{arm}_T{T:g}_C1end"] = c1e
            out[f"{arm}_T{T:g}_C1min"] = c1m
            out[f"{arm}_T{T:g}_C2"] = c2
            out[f"{arm}_T{T:g}_testGend_last"] = float(Ges[-1])
            out[f"{arm}_T{T:g}_testGmin_worst"] = float(Gms.min())
            out[f"{arm}_T{T:g}_worst_drop_from_gen1"] = float(
                Ges[0] - Ges.min())
            log(f"part 2 arm {arm}: T={T:6.1f}: TEST G_end gen1 "
                f"{Ges[0]:.4f} -> last {Ges[-1]:.4f} (worst "
                f"{Ges.min():.4f}, worst drop {Ges[0] - Ges.min():+.2e});"
                f" TEST G_min worst {Gms.min():.4f}; C1end {c1e} C1min "
                f"{c1m} C2 {c2}")
        out[f"{arm}_any_degrades"] = bool(any(
            out[f"{arm}_T{T:g}_{c}"] for T in T_GRID
            for c in ("C1end", "C1min", "C2")))
        log(f"part 2 arm {arm}: ANY degradation on TEST (C1end/C1min/C2, "
            f"DEG_TOL={DEG_TOL:g}) = {out[f'{arm}_any_degrades']}")
    log("part 2 DONE: cached -> cache/exp16_testbattery.npz")
    np.savez(os.path.join(CACHE, "exp16_testbattery.npz"), **out)


# --------------------------------------------------------------- part 3
def part3() -> None:
    """The verdict: P1-P5 + diagnostics + commutativity."""
    log("part 3 START: the verdict (criteria pre-registered in "
        "exp16_preregistration.md BEFORE the first run)")
    z0 = np.load(os.path.join(CACHE, "exp16_part0.npz"),
                 allow_pickle=True)
    z1 = np.load(os.path.join(CACHE, "exp16_part1.npz"),
                 allow_pickle=True)
    z2 = np.load(os.path.join(CACHE, "exp16_testbattery.npz"),
                 allow_pickle=True)
    out: dict = {"DEG_TOL": DEG_TOL}
    # P1 fidelity
    out["P1_fidelity_ok"] = bool(z0["fidelity_ok"])
    out["P1_exp12_delta"] = float(max(
        z0["exp12_armA_max_dGendcurve"], z0["exp12_armB_max_dGendcurve"]))
    out["P1_exp15_delta"] = float(z0["exp15_blindB_max_dGendcurve"])
    out["P1_pytest"] = str(z0["pytest_tail"])
    log(f"part 3: P1 = {out['P1_fidelity_ok']} (exp12 delta "
        f"{out['P1_exp12_delta']:.1e}; exp15 delta "
        f"{out['P1_exp15_delta']:.1e}; pytest {out['P1_pytest']})")
    # P2 precondition
    out["P2"] = bool(z1["P2"])
    out["P2_n_adopt"] = int(z1["n_adopt_total"])
    out["P2_margin_min_abs"] = float(z1["margin_min_abs_all"])
    out["P2_exp15_margin"] = float(z1["exp15_blind_margin_max_abs"])
    out["P2_exp15_adoptions_B"] = int(z1["exp15_blind_adoptions_B"])
    log(f"part 3: P2 = {out['P2']} ({out['P2_n_adopt']} adoptions, "
        f"|margin| >= {out['P2_margin_min_abs']:.2e}; exp15 contrast: "
        f"margin exactly {out['P2_exp15_margin']:.1e}, "
        f"{out['P2_exp15_adoptions_B']} adoptions)")
    # P3/P4
    out["P3_armA"] = bool(z2["A_any_degrades"])
    out["P3_armB"] = bool(z2["B_any_degrades"])
    out["P3"] = out["P3_armA"] or out["P3_armB"]
    out["P4"] = not out["P3"]
    log(f"part 3: P3 (a train-optimal move harms TEST by > DEG_TOL) = "
        f"{out['P3']} (arm A {out['P3_armA']}, arm B "
        f"{out['P3_armB']}); P4 (the null) = {out['P4']}")
    # where the walks END on test, per arm/T
    for arm in ARMS:
        ends = [float(z2[f"{arm}_T{T:g}_testGend_last"]) for T in T_GRID]
        drops = [float(z2[f"{arm}_T{T:g}_worst_drop_from_gen1"])
                 for T in T_GRID]
        log(f"part 3 arm {arm}: TEST G_end last per T: "
            + " ".join(f"{e:.4f}" for e in ends)
            + "; worst within-walk drop: "
            + " ".join(f"{d:+.1e}" for d in drops))
    # diagnostics: (i) lever families under the split — the TEST-schedule
    # walks' adopted families vs the train walks'
    log("part 3: diagnostic (i)+(ii): the TEST-schedule walks (agent "
        "trains ON the rescue-removed task) — families and log-distance")
    fam_train = set()
    for arm in ARMS:
        for T in T_GRID:
            w = _train_walk(T, POLICY[arm])
            fam_train |= {a[:-1] for a in w["seq"] if a != ""}
    out["train_lever_families"] = np.array(sorted(fam_train), dtype=object)
    for arm in ARMS:
        fams, lds = [], []
        for T in T_GRID:
            r = _test_walk(T, POLICY[arm])
            seq = [a for a in r["adopted"] if a != ""]
            fams.append(sorted({a[:-1] for a in seq}))
            pf = r["params_final"]
            ld = float(sum(abs(np.log(getattr(pf, lv) / getattr(P, lv)))
                           for lv in DEFAULT_LEVERS))
            lds.append(ld)
            log(f"part 3 diag arm {arm}: TEST-schedule walk T={T:6.1f}: "
                f"{len(seq):2d} adoptions, families {fams[-1]}, final "
                f"log-dist {ld:.4f}, walk G_end "
                f"{float(np.asarray(r['G_end'], float)[-1]):.4f}")
        out[f"testwalk_{arm}_families"] = np.array(
            [",".join(f) for f in fams], dtype=object)
        out[f"testwalk_{arm}_logdist"] = np.array(lds)
    # train-walk log-distance, matched
    for arm in ARMS:
        lds = []
        for T in T_GRID:
            w = _train_walk(T, POLICY[arm])
            pf = w["params_final"]
            lds.append(float(sum(
                abs(np.log(getattr(pf, lv) / getattr(P, lv)))
                for lv in DEFAULT_LEVERS)))
        out[f"trainwalk_{arm}_logdist"] = np.array(lds)
        log(f"part 3 diag arm {arm}: TRAIN-walk final log-dist per T: "
            + " ".join(f"{d:.4f}" for d in lds))
    # (iv) near-tie census on the train walks (adopted vs runner-up)
    min_gap = float("inf")
    for arm in ARMS:
        for T in T_GRID:
            w = _train_walk(T, POLICY[arm])
            for md in w["r"]["margins"]:
                if not md:
                    continue
                vals = sorted(md.values())
                min_gap = min(min_gap, vals[1] - vals[0])
    out["near_tie_min_gap"] = min_gap
    log(f"part 3: diagnostic (iv): smallest (argmin -> runner-up) margin "
        f"gap anywhere on the train walks = {min_gap:.3e} vs atol_tau = "
        f"{GenParams().atol_tau:g} -> "
        f"{'NEAR-TIE PRESENT' if min_gap <= GenParams().atol_tau else 'no near-tie'}")
    # commutativity check (the gap-probe rule), Params level
    ok_params = True
    ref = Params()
    levs = list(DEFAULT_LEVERS)
    for i in range(len(levs)):
        for j in range(i + 1, len(levs)):
            l1, l2 = levs[i], levs[j]
            pA = replace(replace(ref, **{l1: getattr(ref, l1) * 1.15}),
                         **{l2: getattr(ref, l2) * 1.15})
            pB = replace(replace(ref, **{l2: getattr(ref, l2) * 1.15}),
                         **{l1: getattr(ref, l1) * 1.15})
            for f in ref.__dataclass_fields__:
                if getattr(pA, f) != getattr(pB, f):
                    ok_params = False
    out["commute_params_exact"] = bool(ok_params)
    log(f"part 3: commutativity (Params level, all lever pairs both "
        f"orders, exact): {ok_params}")
    # the closing statement pieces
    out["route_blindness"] = ("REFUSED STRUCTURALLY (exp15: margin "
                              "exactly 0.0, 0 adoptions — the tie rule)")
    out["route_misjudgment"] = (
        "DEGRADES (P3 fired: nonzero certified margins, harm on TEST "
        "> DEG_TOL)" if out["P3"] else
        "DOES NOT DEGRADE HERE (train-optimal moves are not harmful on "
        "the rescue-removed task at DEG_TOL; P4)")
    log(f"part 3: ROUTES — blindness: {out['route_blindness']}; "
        f"misjudgment: {out['route_misjudgment']}")
    log("part 3 DONE: cached -> cache/exp16_verdict.npz")
    np.savez(os.path.join(CACHE, "exp16_verdict.npz"), **out)


# --------------------------------------------------------------- part 4
def part4() -> None:
    """THE DIVERGENCE CURVE: optimised (train margin / train outcome)
    against true (test outcome), per generation — Lin's analogue."""
    log("part 4 START: the class-(C) divergence curve (optimised vs "
        "true), per generation")
    z2 = np.load(os.path.join(CACHE, "exp16_testbattery.npz"),
                 allow_pickle=True)
    out: dict = {"T": np.array(T_GRID), "n_gen": NGEN}
    # the frozen controls for the normalisation
    ctrl_test_G_end = float(z2["ctrl_test_G_end"])
    out["ctrl_test_G_end"] = ctrl_test_G_end
    for arm in ARMS:
        for T in T_GRID:
            w = _train_walk(T, POLICY[arm])
            marg = w["margin"]
            opt = np.cumsum(np.where(marg < 0, -marg, 0.0))  # optimised:
            # cumulative certified improvement V banked on TRAIN
            train_end = w["G_end"]
            test_end = z2[f"{arm}_T{T:g}_testGend"]
            test_min = z2[f"{arm}_T{T:g}_testGmin"]
            out[f"{arm}_T{T:g}_opt"] = opt
            out[f"{arm}_T{T:g}_trainGend"] = train_end
            out[f"{arm}_T{T:g}_testGend"] = test_end
            out[f"{arm}_T{T:g}_testGmin"] = test_min
            div = float(np.corrcoef(train_end, test_end)[0, 1]) \
                if np.std(test_end) > 0 else float("nan")
            out[f"{arm}_T{T:g}_corr_train_test"] = div
            log(f"part 4 arm {arm} T={T:6.1f}: optimised (cum "
                f"certified margin) {opt[-1]:.3e}; train G_end "
                f"{train_end[-1]:.4f}; test G_end {test_end[-1]:.4f} "
                f"(gen1 {test_end[0]:.4f}); corr(train_end, test_end) "
                f"= {div:+.3f}")
    # the signature verdict: optimised up while true down anywhere
    sig = False
    for arm in ARMS:
        for T in T_GRID:
            te = z2[f"{arm}_T{T:g}_testGend"]
            if np.any(np.diff(te) < -DEG_TOL):
                sig = True
    out["signature_rise_collapse"] = bool(sig)
    log(f"part 4: class-(C) signature (optimised up while true down by "
        f"> DEG_TOL anywhere): {sig}")
    log("part 4 DONE: cached -> cache/exp16_divergence.npz")
    np.savez(os.path.join(CACHE, "exp16_divergence.npz"), **out)


# --------------------------------------------------------------- figure
def part_fig() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    log("fig START: f20_traintest.png")
    z2 = np.load(os.path.join(CACHE, "exp16_testbattery.npz"),
                 allow_pickle=True)
    z4 = np.load(os.path.join(CACHE, "exp16_divergence.npz"),
                 allow_pickle=True)
    zv = np.load(os.path.join(CACHE, "exp16_verdict.npz"),
                 allow_pickle=True)
    gens = np.arange(1, NGEN + 1)
    ctrl = float(z2["ctrl_test_G_end"])
    cA, cB = "#2166ac", "#b2182b"
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 9.0))

    # (a) the divergence curve: optimised vs true, arm B, grid extremes
    ax = axs[0, 0]
    for Ti, ls in ((T_GRID[0], "-"), (T_GRID[-1], "--")):
        opt = z4[f"B_T{Ti:g}_opt"]
        ax.semilogy(gens, np.maximum(opt, 1e-12), ls, color=cA,
                    label=f"optimised $\\Sigma$cert-margin $T$={Ti:g}")
        te = z4[f"B_T{Ti:g}_testGend"]
        ax.plot(gens, np.maximum(te, 1e-3), ls, color=cB,
                label=f"true $G^{{TEST}}_{{end}}$ $T$={Ti:g}")
    ax.axhline(ctrl, color="k", ls=":", lw=1.0,
               label=f"frozen on TEST {ctrl:.3f}")
    ax.set_xlabel("generation $k$")
    ax.set_title("(a) optimised (train) vs true (test) — arm B")
    ax.legend(fontsize=7)

    # (b) TEST G_end curves, both arms, all T
    ax = axs[0, 1]
    for arm, c in (("A", cA), ("B", cB)):
        for Ti in T_GRID:
            ax.plot(gens, z2[f"{arm}_T{Ti:g}_testGend"], color=c,
                    lw=1.0, alpha=0.7,
                    label=f"{arm}" if Ti == T_GRID[0] else None)
    ax.axhline(ctrl, color="k", ls=":", lw=1.0, label="frozen on TEST")
    ax.set_xlabel("generation $k$")
    ax.set_ylabel("$G_{end}^{TEST}(k)$")
    ax.set_title("(b) train-optimal Params on TEST, all $T$")
    ax.legend(fontsize=7)

    # (c) the exp15/exp16 contrast: margins and adoptions
    ax = axs[1, 0]
    z1 = np.load(os.path.join(CACHE, "exp16_part1.npz"),
                 allow_pickle=True)
    xs = np.arange(len(T_GRID))
    w = 0.38
    ax.bar(xs - w / 2, [int(z1[f"A_T{T:g}_n_adopt"]) for T in T_GRID],
           w, color=cA, label="A adoptions (train)")
    ax.bar(xs + w / 2, [int(z1[f"B_T{T:g}_n_adopt"]) for T in T_GRID],
           w, color=cB, label="B adoptions (train)")
    ax.set_xticks(xs, [f"{t:g}" for t in T_GRID])
    ax.set_xlabel("capacity $T$")
    ax.set_ylabel(f"adoptions in {NGEN} generations")
    ax.set_title("(c) exp16: V ACTS (vs exp15: 0 adoptions, "
                 "margin exactly 0)")
    ax.legend(fontsize=7)

    # (d) worst within-walk drop on TEST per T
    ax = axs[1, 1]
    for arm, c in (("A", cA), ("B", cB)):
        drops = np.array([float(z2[f"{arm}_T{T:g}_worst_drop_from_gen1"])
                          for T in T_GRID])
        ax.plot(xs, drops, "o-", color=c, label=f"arm {arm}")
    ax.axhline(DEG_TOL, color="k", ls="--",
               label=f"DEG_TOL {DEG_TOL:g}")
    ax.set_xticks(xs, [f"{t:g}" for t in T_GRID])
    ax.set_xlabel("capacity $T$")
    ax.set_ylabel("worst $G_{end}^{TEST}$ drop from gen 1")
    ax.set_title("(d) the P3/C2 criterion per $T$")
    ax.legend(fontsize=7)

    fig.suptitle("exp16 — the train/test gap: P2="
                 f"{bool(zv['P2'])} P3={bool(zv['P3'])} "
                 f"P4={bool(zv['P4'])}; exp15 contrast: margin "
                 f"{float(zv['P2_exp15_margin']):.1e} / "
                 f"{int(zv['P2_exp15_adoptions_B'])} adoptions",
                 fontsize=10)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    outf = os.path.join(FIGDIR, "f20_traintest.png")
    fig.savefig(outf, dpi=140)
    w_px, h_px = Image.open(outf).size
    arr = np.asarray(Image.open(outf).convert("L"))
    ink = float((arr < 250).mean())
    log(f"fig DONE -> {outf} ({w_px}x{h_px}px, aspect "
        f"{w_px / h_px:.2f}, ink {ink:.3f})")
    assert w_px > 1000 and h_px > 800, f"bad size {w_px}x{h_px}"
    assert 0.005 < ink < 0.9, f"bad ink {ink}"
    assert abs(w_px / h_px - 11.5 / 9.0) < 0.05


PARTS = {"0": part0, "1": part1, "2": part2, "3": part3, "4": part4,
         "fig": part_fig}


def main(argv: list[str]) -> None:
    os.makedirs(CACHE, exist_ok=True)
    parts = argv[1:] or list(PARTS)
    if "all" in parts:
        parts = ["0", "1", "2", "3", "4", "fig"]
    t0 = time.time()
    for name in parts:
        if name not in PARTS:
            raise SystemExit(f"unknown part {name!r} "
                             f"(one of {sorted(PARTS)})")
        PARTS[name]()
    log(f"exp16 complete ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main(sys.argv)
