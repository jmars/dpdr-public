"""exp4 — vigilance discriminator: state-level vs parameter-level (P5 split).

The plan fused two hypotheses into P5.  This
driver runs the discriminating measurement battery on the FROZEN model and on
the opt-in parameter-level variant (dpdr/variants.py, "retune"):

  (a) steady-state g*/g0 — state-level predicts persistently > 1;
      parameter-level predicts = 1 (with kappa* > 1 instead);
  (b) post-rescue error-correction SPEED, measured two ways:
      - AT REST (a_hold probe pulse, decay rate lambda of E - E*_pre):
        an HONEST NEGATIVE.  At the settled healthy point the G-restoring
        stiffness is the generator's own logistic saturation
        (-beta_G*(1-a)*(1-2G)*G/tau_G, ~7x the control term's dG
        contribution), so lambda is INSENSITIVE to the loop gain: pinning
        kappa = 1.25 moves lambda by only ~1%.  No vigilance of either
        hypothesis is measurable by probing the recovered system at rest.
      - RECOVERY FROM COLLAPSE (rescue rise time): at low G the control
        term is near its actuator maximum ((G+eps0)(1-G) maximal, tanh(E/Es)
        large) while beta_G*G is small, so the rise DOES resolve kappa.
        Both hypotheses act on the same coefficient kappa*g and give the
        same per-unit speed change — they differ only in PERSISTENCE
        (state-level rides on g, which relaxes with tau_g/mu ~ 667 t.u.;
        parameter-level rides on kappa, which the ratchet holds) and in
        ACCUMULATION across episodes (state re-settles, parameter adds up).
  (c) second episode after rescue (P4 tie-in): second-episode duration
      threshold and second-rescue dose threshold, frozen vs variant.

Assay calibration: kappa pinned at K (kappa_b = kmax = K) is the frozen
model with one constant retuned — lambda(K)/lambda(1) or t50(K)/t50(1) gives
the assay's dose-response; a pinned g0-boost (g0 <- K*g0, g_init <- K*g0)
is the same coefficient change realized as the g-loop's own constant.
RUN in (b2)/(b3): at K = 1.0169 (the ratchet's kappa at rescue onset) the
g0-boost gives the SAME rise rate as the ratchet (lambda_rise 0.007143 vs
0.007143 — both act on the same coefficient kappa*g), but its EPISODE-BUILT
offset decays with tau_g/mu ~ 667 t.u. where the ratchet's kappa holds —
same speed, different persistence class.  A demand-side (A-channel) probe
is NOT used: its E decay is dominated by the D relaxation itself
(lambda_D ~ 0.0088), to which the loop coefficient contributes only ~0.1%
(measured; exp4_lambda.npz from the first pass) — it cannot resolve
vigilance.

Arms: frozen (dpdr.integrate.simulate), ratchet (mu_k = 0, persistent),
leak (mu_k = mu, negative control), pin<K> (kappa fixed at K),
g0boost<K> (frozen model, g0 <- K*g0 — the state-level realization).

Honest magnitude: the leak arm is PRESENT-BUT-SMALL, not absent (rise
x1.0069 vs the ratchet's x1.0206, ~3:1 in speed; kappa at N=2 +0.21% vs
+2.14%, ~10:1) — the two parameter-level arms are distinguishable by the
ACCUMULATION axis (b3: leak kappa decays with delay while the ratchet's
holds), not by speed alone.

Outputs: figs/f09_discriminator.png, cache/exp4_*.npz.
Reproducible: `.venv/bin/python -m experiments.exp4_discriminator`.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np

from dpdr.events import Schedule
from dpdr.integrate import simulate
from dpdr.model import Params
from dpdr.variants import (RetuneParams, check_gates_variant,
                           simulate_variant)

from .common import episode_schedule, rescue_schedule, snap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "figs")

# Canonical geometry (exp2/exp3): episode [100,200) a_hold 0.9 + pulse 0.5 on
# [100,160); rescue [400,460) u_ext 0.8 -> taxonomy 'full', G ~ 0.885.
T_RESCUE, RESC_DUR, RESC_U = 400.0, 60.0, 0.8
EP2_T0, EP2_PULSE_LEN = 800.0, 60.0       # second episode (exp2 Phase 3c)

# Probe: brief inward-attention hold — drives the control loop's own
# actuator (E rises, the correction term responds), touches no schedule
# channel that would confound the frozen-vs-variant comparison.
PROBE_AH, PROBE_DUR = 0.9, 8.0


def arm(name: str, K: float | None = None) -> RetuneParams | None:
    """None = frozen model; else the variant in one of its arms."""
    if name == "frozen":
        return None
    if name == "ratchet":
        return RetuneParams()                    # mu_k = 0 (persistent)
    if name == "leak":
        return RetuneParams(mu_k=0.3)            # symmetric negative control
    if name.startswith("pin"):
        return RetuneParams(kappa_b=K, kmax=K, kappa0=K)
    raise ValueError(name)


def run(name: str, sch: Schedule, T: float, K: float | None = None,
        p: Params | None = None) -> dict:
    p = p if p is not None else Params()
    if name == "g0boost":
        # State-level realization of the same coefficient change: the
        # frozen five-state model with the g-loop's own constant scaled
        # (kappa = 1 throughout; the offset decays with tau_g/mu).
        if K is None:
            raise ValueError("g0boost arm needs K")
        p = replace(p, g0=K * p.g0, g_init=K * p.g0)
        name = "frozen"
    vp = arm(name, K)
    if vp is None:
        sol = simulate(p, sch, T)
    else:
        sol = simulate_variant(p, vp, sch, T)
    sol.setdefault("kappa", np.ones_like(sol["t"]))
    return sol


def _with_probe(ch: dict, t_probe: float) -> dict:
    ch = dict(ch)
    ch["a_hold"] = ch.get("a_hold", []) + [
        (snap(t_probe), snap(t_probe + PROBE_DUR), PROBE_AH)]
    return ch


def rescued_schedule(t_probe: float | None = None, *, probe: bool = True,
                     ) -> Schedule:
    """Canonical rescued run, optionally + a probe pulse at t_probe.

    t_probe is positional-or-None ONLY — do not pass a bool: the historical
    signature made rescued_schedule(False) silently bind t_probe=False and
    snap to a probe at t=8.0.  Pass rescued_schedule(..., probe=False) to
    omit the probe."""
    if isinstance(t_probe, bool):
        raise TypeError("t_probe must be a float or None, not bool "
                        "(use probe=False to omit the probe)")
    ch = dict(rescue_schedule(episode_schedule(), snap(T_RESCUE),
                              RESC_DUR, RESC_U).channels)
    if probe and t_probe is not None:
        ch = _with_probe(ch, t_probe)
    return Schedule(ch)


def naive_schedule(t_probe: float) -> Schedule:
    return Schedule(_with_probe({}, t_probe))


def probe_response(sol: dict, t_probe: float) -> dict:
    """Decay-rate assay of the E response to the probe pulse.

    E_p(t) = E(t) - E*_pre with E*_pre the settled pre-pulse error (mean over
    the 50 t.u. before the pulse).  lambda = -slope of ln E_p from pulse end
    until E_p first falls below 5% of its peak; t95 = that crossing time;
    peak = max E_p.  r2 of the log-linear fit is reported (all fits r2 ~
    0.96 — near-exponential but not exact; the fitted lambda is used only
    for RATIO comparisons, never as an absolute rate).  The response is
    monotone (overdamped), so lambda is the error-correction rate of the
    closed loop.
    """
    t, E = sol["t"], sol["E"]
    pre = (t >= t_probe - 50.0) & (t < t_probe - 1.0)
    E0 = float(E[pre].mean())
    Ep = E - E0
    i0 = int(np.searchsorted(t, t_probe + PROBE_DUR))
    pk = float(Ep[i0 - 5: i0 + 60].max())
    if pk <= 0:
        return dict(lambda_=float("nan"), t95=float("nan"), peak=pk,
                    E0=E0, r2=float("nan"))
    w = t >= t_probe + PROBE_DUR
    above = np.where((Ep > 0.05 * pk) & w)[0]
    t95 = float(t[above[-1]] - (t_probe + PROBE_DUR)) if above.size else \
        float("nan")
    if above.size < 20:
        return dict(lambda_=float("nan"), t95=t95, peak=pk, E0=E0,
                    r2=float("nan"))
    x, y = t[above], np.log(Ep[above])
    fit = np.polyfit(x, y, 1)
    pred = np.polyval(fit, x)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(lambda_=float(-fit[0]), t95=t95, peak=pk, E0=E0,
                r2=1.0 - ss_res / ss_tot)


# ------------------------------------------------------------------ part (a)
def measure_a(T: float = 6000.0) -> dict:
    """Steady-state g*/g0 and kappa* on naive vs rescued runs."""
    out = {}
    for name in ("frozen", "ratchet", "leak"):
        out[(name, "naive")] = run(name, Schedule(), T)
        out[(name, "rescued")] = run(name, rescued_schedule(probe=False), T)
    # T=12000 confirmation that the offsets are not still drifting
    out[("frozen", "rescued12k")] = run("frozen", rescued_schedule(
        probe=False), 12000.0)
    out[("ratchet", "rescued12k")] = run("ratchet", rescued_schedule(
        probe=False), 12000.0)
    return out


# ------------------------------------------------------------------ part (b)
def measure_b_atrest(delays=(540.0, 1140.0, 2540.0)) -> dict:
    """At-rest probe assay (delay axis) — the honest-negative panel: lambda
    naive vs post-rescue per arm, plus the pinned-kappa insensitivity."""
    res = {}
    for name in ("frozen", "ratchet", "leak"):
        row = {}
        for d in delays:
            tp = 460.0 + d
            row[("naive", d)] = probe_response(
                run(name, naive_schedule(tp), tp + 500.0), tp)
            row[("rescued", d)] = probe_response(
                run(name, rescued_schedule(tp), tp + 500.0), tp)
        res[name] = row
    tp = 460.0 + 1140.0
    res["pin_kappa"] = {K: probe_response(
        run("pin", rescued_schedule(tp), tp + 500.0, K=K), tp)
        for K in (1.0, 1.0115, 1.05, 1.11, 1.25)}
    return res


def rescue_rise(sol: dict, t_r: float, frac: float = 0.5) -> dict:
    """Rise-time assay of a rescue: lambda_rise = -slope of ln(G_inf - G(t))
    from rescue onset until G crosses frac*(G_inf - G_start) + G_start;
    t_frac = that crossing time.  G_inf = mean G over the final 50 t.u."""
    t, G = sol["t"], sol["G"]
    i0 = int(np.searchsorted(t, t_r))
    G0r = float(G[i0 - 1])
    Ginf = float(G[-int(50.0 / 0.05):].mean())
    tgt = G0r + frac * (Ginf - G0r)
    above = np.where((t >= t_r) & (G >= tgt))[0]
    t_frac = float(t[above[0]] - t_r) if above.size else float("nan")
    w = (t >= t_r) & (t <= t[above[0]]) if above.size else None
    if w is None or above.size < 20:
        return dict(lambda_rise=float("nan"), t_frac=t_frac, G_inf=Ginf,
                    G_start=G0r)
    x, y = t[w], np.log(np.maximum(Ginf - G[w], 1e-12))
    return dict(lambda_rise=float(-np.polyfit(x, y, 1)[0]), t_frac=t_frac,
                G_inf=Ginf, G_start=G0r)


def measure_b_rise() -> dict:
    """Recovery-speed assay on the SECOND rescue (t=1150), whose rise the
    first episode's retuning (kappa ~ +1.7% for the ratchet) can speed up.
    Frozen vs ratchet vs leak + pin calibration on the same second rescue."""
    t_r2 = snap(EP2_T0 + 150.0 + 200.0)
    out = {"t_r2": t_r2}
    for name in ("frozen", "ratchet", "leak"):
        s = second_episode(name, 150.0, 0.5, rescue2=(t_r2, 60.0, 0.8),
                           T=t_r2 + 700.0)
        out[name] = rescue_rise(s, t_r2)
        out[f"sol_{name}"] = s                 # reused by the figure
    tp = 460.0 + 1140.0                     # at-rest reference for the panel
    out["pin"] = {}
    for K in (1.0, 1.0169, 1.05, 1.11, 1.25):
        s = second_episode("pin", 150.0, 0.5, rescue2=(t_r2, 60.0, 0.8),
                           T=t_r2 + 700.0, K=K)
        out["pin"][K] = rescue_rise(s, t_r2)
        out[f"sol_pin_{K:g}"] = s
    out["atrest_probe"] = tp
    # State-level counterpart: the same coefficient change realized as the
    # g-loop's own constant.  K=1.0169 is the ratchet's kappa at rescue
    # onset, so the boost multiplies kappa*g by the same factor — same
    # speed; K=1.0214 is the ratchet's kappa at rescue END, for reference.
    out["g0boost"] = {}
    for K in (1.0169, 1.0214):
        s = second_episode("g0boost", 150.0, 0.5, rescue2=(t_r2, 60.0, 0.8),
                           T=t_r2 + 700.0, K=K)
        out["g0boost"][K] = rescue_rise(s, t_r2)
        out[f"sol_g0boost_{K:g}"] = s
    return out


def measure_accumulation(delay: float = 1080.0,
                         long_delay: float = 4680.0) -> dict:
    """Probe assay along the EPISODE-COUNT axis, plus a persistence leg.

    N = 0: naive run.  N = 1: episode + canonical rescue (ends t=460).
    N = 2: + second episode [800,950) + second rescue [1150,1210).  In every
    run g and kappa are read at a FIXED delay past the last volatility (the
    rescue end): the state-level offset must be the same for N=1 and N=2
    (it depends only on time since the last kick), while the parameter-level
    offset accumulates.  A probe-free readout; the probe is not needed
    because g and kappa are directly observable here.  The LEAK arm runs
    alongside the ratchet: at this delay it is PRESENT-BUT-SMALL (~10:1 in
    kappa at N=2, ~3:1 in rise speed) — the persistence leg below, not
    speed, is where the two parameter-level arms separate.

    Persistence leg: the episode-built offset read at delay and long_delay
    past the last volatility.  The ratchet's kappa HOLDS (nothing in the
    variant erases it); the leak's kappa and the g0-boost's built g excess
    decay with tau/mu ~ 667 t.u.  The g0-boost's CONSTANT shift (K-1) does
    not decay — a retuned constant is not an accumulating state; only its
    episode-built excess decays, exactly as the frozen g-loop's does.
    """
    out: dict = {}
    T = 460.0 + delay
    ch2 = dict(rescued_schedule(probe=False).channels)
    ch2["a_hold"] = ch2["a_hold"] + [(EP2_T0, snap(EP2_T0 + 150.0), 0.9)]
    ch2["A"] = ch2["A"] + [(EP2_T0, snap(EP2_T0 + EP2_PULSE_LEN), 0.5)]
    ch2["u_ext"] = ch2["u_ext"] + [(1150.0, 1210.0, 0.8)]
    for name in ("ratchet", "leak"):
        runs = [run(name, Schedule(), T),
                run(name, rescued_schedule(probe=False), T),
                run(name, Schedule(ch2), 1210.0 + delay)]
        for N, s in enumerate(runs):
            out[(name, N, "vals")] = dict(
                g_off_pct=100.0 * (s["g"][-1] / Params().g0 - 1.0),
                kappa_off_pct=100.0 * (s["kappa"][-1] - 1.0),
                G_end=float(s["G"][-1]))
        out[(name, 2, "run")] = runs[2]      # reused by the persistence leg
    out["persist_delays"] = (delay, long_delay)
    K_G0B = 1.0169                           # the ratchet's kappa at onset
    out["persist_K"] = K_G0B
    out["persist"] = {}
    for name in ("ratchet", "leak"):
        out["persist"][name] = {}
        for d in (delay, long_delay):
            s = out[(name, 2, "run")] if d == delay else \
                run(name, Schedule(ch2), 1210.0 + d)
            out["persist"][name][d] = 100.0 * (s["kappa"][-1] - 1.0)
    out["persist"]["g0boost"] = {}
    for d in (delay, long_delay):
        s = run("g0boost", Schedule(ch2), 1210.0 + d, K=K_G0B)
        out["persist"]["g0boost"][d] = dict(
            built=100.0 * (s["g"][-1] / (K_G0B * Params().g0) - 1.0),
            total=100.0 * (s["g"][-1] / Params().g0 - 1.0))
    # frozen model at the same readout times (g only; no kappa to accumulate)
    for N, sch, T in ((0, Schedule(), 460.0 + delay),
                      (1, rescued_schedule(probe=False), 460.0 + delay)):
        s = run("frozen", sch, T)
        out[(N, "frozen_g_off_pct")] = 100.0 * (s["g"][-1] / Params().g0 - 1.0)
    return out


# ------------------------------------------------------------------ part (c)
def second_episode(name: str, dur2: float, pulse2: float = 0.5,
                   rescue2: tuple | None = None, T: float | None = None,
                   K: float | None = None):
    """Canonical rescued run + second inward episode at t=800 (exp2 Phase 3c
    geometry); optional second rescue (onset, dur, u)."""
    ch = dict(rescued_schedule(probe=False).channels)
    ch["a_hold"] = ch["a_hold"] + [(EP2_T0, snap(EP2_T0 + dur2), 0.9)]
    if pulse2 != 0.0:
        ch["A"] = ch["A"] + [(EP2_T0, snap(EP2_T0 + EP2_PULSE_LEN), pulse2)]
    if rescue2 is not None:
        t0, dur, u = rescue2
        ch["u_ext"] = ch["u_ext"] + [(snap(t0), snap(t0 + dur), u)]
    if T is None:
        T = max(1600.0, snap(EP2_T0 + dur2) + 400.0)
    return run(name, Schedule(ch), T, K=K)


def second_rescue_success(sol: dict, t_r2: float, dur2: float = 60.0) -> bool:
    """Second rescue judged at the horizon: collapsed (G < 0.5) at rescue
    onset, G(T) > 0.5 and G stays > 0.5 from rescue end onward.  (The
    shipped rescue_taxonomy cannot be reused here: it scans the whole
    post-first-collapse tail, so the first-rescue peak followed by episode
    2's collapse would label every second-rescued run 'transient'.)"""
    t, G = sol["t"], sol["G"]
    i = int(np.searchsorted(t, t_r2))
    if G[i] > 0.5:
        return False
    w = t >= t_r2 + dur2
    return bool(G[-1] > 0.5 and G[w].min() > 0.5)


def measure_c() -> dict:
    """(c) second-episode response, frozen vs ratchet."""
    out = {}
    p = Params()

    def collapsed(name, dur, pulse2=0.5):
        s = second_episode(name, dur, pulse2)
        return float(s["G"][-1]) < 0.1, s

    for name in ("frozen", "ratchet", "leak"):
        lo, hi = 5.0, 200.0
        if not collapsed(name, hi)[0] or collapsed(name, lo)[0]:
            out[(name, "thr2")] = None
            continue
        for _ in range(10):
            mid = snap(0.5 * (lo + hi))
            if collapsed(name, mid)[0]:
                hi = mid
            else:
                lo = mid
        out[(name, "thr2")] = 0.5 * (lo + hi)

    t_r2 = snap(EP2_T0 + 150.0 + 200.0)       # ep2 [800, 950), rescue at 1150
    us = np.round(np.arange(0.30, 1.01, 0.05), 4)
    for name in ("frozen", "ratchet"):
        full = []
        for u in us:
            s = second_episode(name, 150.0, 0.5,
                               rescue2=(t_r2, 60.0, float(u)),
                               T=t_r2 + 700.0)
            full.append(second_rescue_success(s, t_r2))
        full = np.array(full)
        out[(name, "u2_star")] = float(us[full][0]) if full.any() else None
        out[(name, "u2_grid")] = (us, full)
        # The grid step (0.05) QUANTIZES u2*: frozen 0.85 vs ratchet 0.80 is
        # one step.  Refine by bisection between the last failing and the
        # first succeeding grid strength (u values need no grid snapping —
        # only schedule TIMES do).
        if full.any() and not full.all():
            i0 = int(np.argmax(full))
            lo, hi = float(us[i0] - 0.05), float(us[i0])
            for _ in range(8):
                mid = 0.5 * (lo + hi)
                s = second_episode(name, 150.0, 0.5,
                                   rescue2=(t_r2, 60.0, mid), T=t_r2 + 700.0)
                if second_rescue_success(s, t_r2):
                    hi = mid
                else:
                    lo = mid
            out[(name, "u2_refined")] = 0.5 * (lo + hi)

    for name in ("frozen", "ratchet"):
        s = second_episode(name, 150.0)
        out[(name, "relapse_run")] = s
    return out


# ---------------------------------------------------------------------- main
def main() -> int:
    p = Params()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    # ---- part 0: does the variant preserve the frozen phenomenology?
    print("== exp4 part 0: gate check G1-G2c per arm (must match frozen) ==")
    from dpdr.metrics import check_gates
    gates = {"frozen": check_gates(p)}
    print(f"  frozen   : {' '.join(f'{k}={v}' for k, v in gates['frozen'].items())}"
          f"  -> {'ALL PASS' if all(gates['frozen'].values()) else 'FAIL'}")
    for name, K in [("pin1.25", 1.25), ("pin1.0115", 1.0115),
                    ("ratchet", None), ("leak", None)]:
        g = check_gates_variant(arm(name, K), p)
        gates[name] = g
        print(f"  {name:9s}: {' '.join(f'{k}={v}' for k, v in g.items())}"
              f"  -> {'ALL PASS' if all(g.values()) else 'FAIL'}")

    # ---- part (a): steady state
    print("\n== exp4 (a): steady-state g*/g0 and kappa* (T=6000; 12k check) ==")
    sols_a = measure_a()
    a_rows = {}
    for (name, kind), sol in sols_a.items():
        row = dict(g_ratio=float(sol["g"][-1] / p.g0),
                   kappa=float(sol["kappa"][-1]),
                   G_end=float(sol["G"][-1]))
        a_rows[(name, kind)] = row
        extra = ""
        if kind.startswith("rescued"):
            ipk = int(np.argmax(sol["g"]))
            extra = (f"  [peak g {sol['g'][ipk]:.4f} @ t={sol['t'][ipk]:.0f};"
                     f" peak kappa {sol['kappa'].max():.4f} @ t="
                     f"{sol['t'][int(np.argmax(sol['kappa']))]:.0f}]")
        print(f"  {name:8s} {kind:10s}: g*/g0 = {row['g_ratio']:.7f}  "
              f"kappa* = {row['kappa']:.6f}  G(T) = {row['G_end']:.4f}{extra}")

    # ---- part (b): at-rest probe (honest negative) + recovery-speed assay
    print("\n== exp4 (b1): at-rest probe decay-rate lambda (a_hold pulse) ==")
    delays = (540.0, 1140.0, 2540.0)
    res_b = measure_b_atrest(delays)
    lam_tbl = {}
    for name in ("frozen", "ratchet", "leak"):
        row = res_b[name]
        base = row[("naive", delays[0])]["lambda_"]
        print(f"  [{name}] naive lambda = {base:.5f} (t95="
              f"{row[('naive', delays[0])]['t95']:.1f})")
        for d in delays:
            for kind in ("naive", "rescued"):
                lam_tbl[f"{name}_{kind}_{d:g}"] = row[(kind, d)]["lambda_"]
            r = row[("rescued", d)]
            print(f"    delay {d:6.0f}: lambda = {r['lambda_']:.5f} "
                  f"(x{r['lambda_'] / base:.4f})  t95 = {r['t95']:.1f}  "
                  f"peak = {r['peak']:.4f}  r2 = {r['r2']:.4f}")
    print("  [at-rest pin-kappa insensitivity]")
    lam1 = res_b["pin_kappa"][1.0]["lambda_"]
    print(f"    lambda(K=1) = {lam1:.5f}")
    for K in sorted(k for k in res_b["pin_kappa"] if k != 1.0):
        lam_tbl[f"pin_{K:g}"] = res_b["pin_kappa"][K]["lambda_"]
        print(f"    pin kappa={K:.4f}: lambda = "
              f"{res_b['pin_kappa'][K]['lambda_']:.5f} "
              f"(x{res_b['pin_kappa'][K]['lambda_'] / lam1:.4f})")

    print("\n== exp4 (b2): recovery-speed assay (second-rescue rise) ==")
    res_rise = measure_b_rise()
    print(f"  second episode [800,950), second rescue at t="
          f"{res_rise['t_r2']:.0f} (u=0.8, 60 t.u.)")
    for name in ("frozen", "ratchet", "leak"):
        r = res_rise[name]
        print(f"  [{name:7s}] lambda_rise = {r['lambda_rise']:.5f}  "
              f"t50 = {r['t_frac']:.1f} t.u.  G_inf = {r['G_inf']:.4f}")
    lam_r1 = res_rise["pin"][1.0]["lambda_rise"]
    print(f"  [pin calibration on the same second rescue]")
    print(f"    K=1: lambda_rise = {lam_r1:.5f}")
    for K in sorted(k for k in res_rise["pin"] if k != 1.0):
        r = res_rise["pin"][K]
        print(f"    K={K:.4f}: lambda_rise = {r['lambda_rise']:.5f} "
              f"(x{r['lambda_rise'] / lam_r1:.4f})  t50 = {r['t_frac']:.1f}"
              f"  (t50 ratio {r['t_frac'] / res_rise['pin'][1.0]['t_frac']:.4f})")
    print(f"  [state-level counterpart: g0 <- K*g0 (frozen model, kappa = 1)]")
    lam_r = res_rise["ratchet"]["lambda_rise"]
    for K in sorted(res_rise["g0boost"]):
        r = res_rise["g0boost"][K]
        note = ("SAME speed as the ratchet — the two hypotheses differ "
                "ONLY in persistence, see (b3)" if K == 1.0169 else
                "reference: K = the ratchet's kappa at rescue END")
        print(f"    g0boost K={K:.4f}: lambda_rise = {r['lambda_rise']:.5f} "
              f"(x{r['lambda_rise'] / lam_r1:.4f}; ratchet "
              f"x{lam_r / lam_r1:.4f})  t50 = {r['t_frac']:.1f}  -> {note}")

    # ---- part (b3): episode-count axis (accumulation vs state relaxation)
    print("\n== exp4 (b3): g / kappa offset at fixed delay past last "
          "volatility ==")
    acc = measure_accumulation()
    for name in ("ratchet", "leak"):
        for N in (0, 1, 2):
            v = acc[(name, N, "vals")]
            print(f"  [{name:7s}] N={N}: g offset = {v['g_off_pct']:+.4f}%  "
                  f"kappa offset = {v['kappa_off_pct']:+.4f}%  "
                  f"G(T) = {v['G_end']:.3f}")
        k2 = acc[(name, 2, "vals")]["kappa_off_pct"]
        k1 = acc[(name, 1, "vals")]["kappa_off_pct"]
        print(f"    -> kappa N=2 vs N=1: {k2:+.4f}% vs {k1:+.4f}% "
              f"({'ACCUMULATES across episodes' if name == 'ratchet' else
                'present-but-small, near-flat across episodes'})")
    print("    (ratchet vs leak at N=2: ~10:1 in kappa offset, ~3:1 in "
          "rise speed — distinguishable by the accumulation axis below, "
          "not by speed alone)")
    print(f"  (frozen model g offset at the same readout: N=0 "
          f"{acc[(0, 'frozen_g_off_pct')]:+.4f}%, N=1 "
          f"{acc[(1, 'frozen_g_off_pct')]:+.4f}% — N=2 has no frozen "
          f"counterpart because kappa does not exist there)")
    d1, d2 = acc["persist_delays"]
    K = acc["persist_K"]
    print(f"  [persistence: episode-built offset at delay {d1:g} vs "
          f"{d2:g} t.u. past the last volatility (tau/mu ~ 667)]")
    print(f"    ratchet  kappa: {acc['persist']['ratchet'][d1]:+.4f}% -> "
          f"{acc['persist']['ratchet'][d2]:+.4f}%   HOLDS")
    print(f"    leak     kappa: {acc['persist']['leak'][d1]:+.4f}% -> "
          f"{acc['persist']['leak'][d2]:+.4f}%   DECAYS")
    gb1, gb2 = acc["persist"]["g0boost"][d1], acc["persist"]["g0boost"][d2]
    print(f"    g0boost built g excess (g - K*g0)/(K*g0), K={K}: "
          f"{gb1['built']:+.4f}% -> {gb2['built']:+.4f}%   DECAYS "
          f"(total vs frozen g0 {gb1['total']:+.4f}% -> {gb2['total']:+.4f}%:"
          f" the CONSTANT shift does not decay, the built excess does — "
          f"same speed as the ratchet at onset, but not a persistent state)")

    # ---- part (c): second episode
    print("\n== exp4 (c): second episode after rescue (P4 tie-in) ==")
    res_c = measure_c()
    for name in ("frozen", "ratchet", "leak"):
        print(f"  [{name}] second-episode duration threshold (pulse 0.5): "
              f"{res_c[(name, 'thr2')]:.2f} t.u.")
    for name in ("frozen", "ratchet"):
        us, full = res_c[(name, "u2_grid")]
        print(f"  [{name}] second-rescue u* (delay 200, dur 60): "
              f"{res_c[(name, 'u2_star')]}   (full at "
              f"{int(full.sum())}/{len(us)} grid strengths; grid step 0.05 "
              f"QUANTIZES u* — refined {res_c[(name, 'u2_refined')]:.4f})")
    for name in ("frozen", "ratchet"):
        s = res_c[(name, "relapse_run")]
        print(f"  [{name}] canonical relapse (ep2 dur 150, no second rescue): "
              f"G(T) = {s['G'][-1]:.3f}, g(T)/g0 = {s['g'][-1] / p.g0:.5f}, "
              f"kappa(T) = {s['kappa'][-1]:.4f}")

    # ---- figure
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2)

    # (A) g(t) and kappa(t) on the rescued runs (part a)
    axA = fig.add_subplot(gs[0, 0])
    sf = sols_a[("frozen", "rescued")]
    sr = sols_a[("ratchet", "rescued")]
    w = sf["t"] <= 3000.0
    axA.plot(sf["t"][w], sf["g"][w], lw=0.9, label="frozen g(t)")
    axA.plot(sr["t"][w], sr["g"][w], lw=0.9, ls="--", label="ratchet g(t)")
    axA.axhline(p.g0, color="gray", ls=":", lw=0.8, label="g0")
    axA.set_ylabel("g(t)")
    axA.set_xlabel("t (t.u.)")
    axA.set_title("(a) both hypotheses give g* = g0\n"
                  "(state-level > 1 falsified; kappa persists instead)")
    axA.legend(fontsize=8, loc="upper right")
    axA2 = axA.twinx()
    axA2.plot(sr["t"][w], sr["kappa"][w], lw=0.9, color="tab:red")
    axA2.set_ylabel("kappa(t) (ratchet)", color="tab:red")
    axA2.tick_params(axis="y", labelcolor="tab:red")

    # (B) second-rescue rise: frozen vs ratchet, pin calibration (part b2)
    axB = fig.add_subplot(gs[0, 1])
    t_r2 = res_rise["t_r2"]
    for name, col, ls in (("frozen", "tab:blue", "-"),
                          ("ratchet", "tab:red", "-"),
                          ("leak", "tab:green", "--")):
        s = res_rise[f"sol_{name}"]
        w = (s["t"] >= t_r2 - 20.0) & (s["t"] <= t_r2 + 250.0)
        axB.plot(s["t"][w] - t_r2, s["G"][w], ls, color=col, lw=1.2,
                 label=f"{name} (t50={res_rise[name]['t_frac']:.1f})")
    # the state-level counterpart: SAME speed at the onset coefficient,
    # different persistence class (see the (b3) persistence leg)
    KGB = 1.0169                            # the ratchet's kappa at onset
    s = res_rise[f"sol_g0boost_{KGB:g}"]
    w = (s["t"] >= t_r2 - 20.0) & (s["t"] <= t_r2 + 250.0)
    axB.plot(s["t"][w] - t_r2, s["G"][w], "-.", color="tab:orange", lw=1.0,
             label=(f"g0 x{KGB:g} (t50={res_rise['g0boost'][KGB]['t_frac']:.1f},"
                    f" same speed — decays)"))
    # pin reference curves (gray dotted): only K=1.05 from the b2 pin set
    K = 1.05
    if f"sol_pin_{K:g}" in res_rise:
        s = res_rise[f"sol_pin_{K:g}"]
        w = (s["t"] >= t_r2 - 20.0) & (s["t"] <= t_r2 + 250.0)
        axB.plot(s["t"][w] - t_r2, s["G"][w], ":", color="gray", lw=1.0,
                 label=f"pin kappa={K:g} (t50={res_rise['pin'][K]['t_frac']:.1f})")
    axB.set_xlabel("t - t_rescue2 (t.u.)")
    axB.set_ylabel("G(t)")
    axB.set_title("(b2) second-rescue rise: ratchet and g0-boost are EQUALLY\n"
                  "FASTER with the same onset coefficient; only the ratchet's\n"
                  "gain persists (at rest unmeasurable — see (b1))")
    axB.legend(fontsize=7)

    # (C) at-rest probe: the honest negative (part b1)
    axC = fig.add_subplot(gs[1, 0])
    for name, col, mk in (("frozen", "tab:blue", "o"),
                          ("ratchet", "tab:red", "s"),
                          ("leak", "tab:green", "^")):
        row = res_b[name]
        ratios = [row[("rescued", d)]["lambda_"]
                  / row[("naive", delays[0])]["lambda_"] for d in delays]
        axC.plot(delays, ratios, mk + "-", color=col, label=name)
    for K, mk in ((1.05, "x"), (1.25, "X")):
        axC.axhline(res_b["pin_kappa"][K]["lambda_"] / lam1, color="gray",
                    ls=":", lw=0.8)
        axC.annotate(f"pin kappa={K:g}", xy=(delays[-1], res_b["pin_kappa"][K]
                      ["lambda_"] / lam1), fontsize=7, color="gray",
                     xytext=(-60, 4), textcoords="offset points")
    axC.axhline(1.0, color="k", ls=":", lw=0.8)
    axC.set_xlabel("probe delay past rescue end (t.u.)")
    axC.set_ylabel("lambda_rescued / lambda_naive")
    axC.set_title("(b1) HONEST NEGATIVE: at-rest correction speed does not\n"
                  "resolve the loop gain (logistic stiffness dominates)")
    axC.legend(fontsize=8)

    # (D) accumulation axis (b3): ratchet vs leak + persistence leg
    axD = fig.add_subplot(gs[1, 1])
    Ns = (0, 1, 2)
    g_offs = [acc[("ratchet", N, "vals")]["g_off_pct"] for N in Ns]
    k_offs_r = [acc[("ratchet", N, "vals")]["kappa_off_pct"] for N in Ns]
    k_offs_l = [acc[("leak", N, "vals")]["kappa_off_pct"] for N in Ns]
    xb = np.arange(3)
    axD.bar(xb - 0.22, g_offs, 0.2, label="g offset (%)", color="tab:blue")
    axD.bar(xb, k_offs_r, 0.2, label="kappa offset, ratchet (%)",
            color="tab:red")
    axD.bar(xb + 0.22, k_offs_l, 0.2, label="kappa offset, leak (%)",
            color="tab:green")
    d1, d2 = acc["persist_delays"]
    axD.annotate(
        f"persistence (delay {d1:g} -> {d2:g}):\n"
        f"  ratchet kappa {acc['persist']['ratchet'][d2]:+.2f}% HOLDS\n"
        f"  leak kappa {acc['persist']['leak'][d2]:+.2f}% DECAYS\n"
        f"  g0-boost built g {acc['persist']['g0boost'][d2]['built']:+.2f}%"
        " DECAYS",
        xy=(0.02, 0.98), xycoords="axes fraction", va="top",
        ha="left", fontsize=7,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    axD.set_xticks(xb)
    axD.set_xticklabels([f"N={N}" for N in Ns])
    axD.set_ylabel("offset at fixed delay past last episode (%)")
    axD.set_title("(b3) state re-settles, parameter accumulates;\n"
                  "leak is PRESENT-BUT-SMALL (~10:1 kappa) and decays —\n"
                  f"the accumulation axis is where the arms separate "
                  f"(c) thr2: frozen {res_c[('frozen', 'thr2')]:.1f} / "
                  f"ratchet {res_c[('ratchet', 'thr2')]:.1f} t.u.")
    axD.legend(fontsize=7, loc="upper center")

    fig.suptitle("exp4 vigilance discriminator: state-level (g; falsified) "
                 "vs parameter-level (kappa; opt-in variant) — g* = g0 in "
                 "both; recovery-speed and accumulation signatures differ")
    fig.tight_layout()
    f09 = os.path.join(FIGDIR, "f09_discriminator.png")
    fig.savefig(f09, dpi=110)
    plt.close(fig)
    print(f"\n  -> {f09}")

    # ---- cache
    np.savez(os.path.join("cache", "exp4_lambda.npz"),
             **{k: v for k, v in lam_tbl.items()},
             **{f"rise_{name}": res_rise[name]["lambda_rise"]
                for name in ("frozen", "ratchet", "leak")},
             **{f"rise_t50_{name}": res_rise[name]["t_frac"]
                for name in ("frozen", "ratchet", "leak")},
             **{f"rise_pin_{K:g}": res_rise["pin"][K]["lambda_rise"]
                for K in res_rise["pin"]},
             **{f"rise_t50_pin_{K:g}": res_rise["pin"][K]["t_frac"]
                for K in res_rise["pin"]},
             **{f"rise_g0boost_{K:g}": res_rise["g0boost"][K]["lambda_rise"]
                for K in res_rise["g0boost"]},
             **{f"rise_t50_g0boost_{K:g}": res_rise["g0boost"][K]["t_frac"]
                for K in res_rise["g0boost"]})
    np.savez(os.path.join("cache", "exp4_summary.npz"),
             **{f"a_{n}_{k}_g": r["g_ratio"] for (n, k), r in a_rows.items()},
             **{f"a_{n}_{k}_kappa": r["kappa"]
                for (n, k), r in a_rows.items()},
             acc_g=np.array(g_offs),
             acc_kappa=np.array(k_offs_r),
             acc_kappa_leak=np.array(k_offs_l),
             persist_delays=np.array(acc["persist_delays"]),
             persist_kappa_ratchet=np.array(
                 [acc["persist"]["ratchet"][d] for d in acc["persist_delays"]]),
             persist_kappa_leak=np.array(
                 [acc["persist"]["leak"][d] for d in acc["persist_delays"]]),
             persist_g0boost_built=np.array(
                 [acc["persist"]["g0boost"][d]["built"]
                  for d in acc["persist_delays"]]),
             persist_g0boost_total=np.array(
                 [acc["persist"]["g0boost"][d]["total"]
                  for d in acc["persist_delays"]]),
             thr2=np.array([res_c[("frozen", "thr2")],
                            res_c[("ratchet", "thr2")],
                            res_c[("leak", "thr2")]], float),
             u2=np.array([res_c[("frozen", "u2_star")] or np.nan,
                          res_c[("ratchet", "u2_star")] or np.nan]),
             u2_refined=np.array([res_c[("frozen", "u2_refined")],
                                  res_c[("ratchet", "u2_refined")]]),
             gates_ok=np.array([all(g.values()) for g in gates.values()]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
