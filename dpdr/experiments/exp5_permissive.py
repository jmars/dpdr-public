"""exp5 — two-axis permissive AND-gate: the conjunction made testable.

The FROZEN model is single-factor-sufficient (OR-gate): sustained inward
attention ALONE collapses it (a_hold 0.35 x 1400 t.u. -> G stuck 0.0485;
affect alone never collapses).  The reported etiology was a CONJUNCTION of
four factors.  This driver runs the measurement battery on the opt-in
variant dpdr/permissive.py, whose destabilization term is multiplicative in
two thresholded axes — fast/serotonergic (nightly window, channel 'ser')
and slow/neurotrophic (accumulating N, channel 'ngf'):

    dG weakening:   - Phi(t) * alpha_G * a * G,
    Phi = Phi_max * g_fast(A_ser(t)) * g_slow(N(t)).

NO constant was fitted to the reported event; every choice is a documented
hypothesis-space choice (see permissive.py's docstring): Phi_max = 2.0
("fully expressed conjunction doubles the weakening rate"; the frozen model
is the Phi = 1 locus), Hill-1 g_fast with K_ser = 0.25 (half-max at a
quarter-full nightly load), sigmoid g_slow centered N50 = 14 days / width
Nw = 4 (weeks-scale NGF accumulation), mu_N = 0 (G2b analog) with a
mu_N = 1/42-per-day washout control, and the timescale mapping
1 t.u. = 5 min (tau_a = 5 min physiological) -> T_day = 288 t.u., nightly
window W = 8 h = 96 t.u.

Protocol (all drive values are the FROZEN model's own canonical trigger,
not tuned here): each night the subject applies a_hold 0.9 for the full
window + affect pulse A 0.5 for the first 60 t.u. (the canonical episode's
intensity and pulse; duration = one nightly window).  The conjunction arm
adds ser 1.0 nightly (fast axis) and ngf 1.0/day (slow axis).

  (a) AND-gate: the IDENTICAL nightly drive under four arms — conjunction,
      fast-only, slow-only, drive-only (60 nights for the single-axis arms,
      25 for the conjunction).  Plus the frozen-model reference: ONE such
      night with Phi pinned at 1 (the frozen default) already collapses —
      the OR-gate the variant replaces.
  (b) Phase boundary in (fast-axis amplitude x slow-axis accumulated level):
      a single nightly window at T0 with the slow axis pre-set (N0) and the
      fast axis at amplitude ser — stuck criterion (G(T) < 0.1 AND
      is_stuck), NOT first-crossing (G dips below 0.1 transiently near the
      boundary and recovers; first_below would misclassify those).  Both
      boundary legs are bisected; the pinned-Phi constant dose-response
      thr_dur(P) calibrates the map against the frozen model's own
      duration-threshold assay.
  (c) Timing: 24-cell ensemble (a_hold x dose scatter) on the nightly
      protocol; phase-of-onset distribution vs the window [0, W)/T_day.
      A uniform-onset null puts 2/3 of onsets OUTSIDE the window.
  (d) Persistence: priming (15 nights, both axes, NO drive) -> 28-night
      gap (both axes withdrawn) -> re-exposure (the canonical nightly drive
      again).  mu_N = 0: N holds (14.25 at re-exposure) and re-exposure
      collapses; mu_N = 1/42: N washes out (6.18) and the same re-exposure
      does not.  Also read: after a conjunction collapse, withdrawing BOTH
      axes leaves the system stuck — INHERITED from the frozen G2b
      (attractor dynamics), not an emergent property of the gate; the
      variant-specific persistence claim is about N (vulnerability), not G.

Honest negatives / caveats (also in predictions.md): (i) the conjunction
arm's collapse is marginal, not dramatic — Phi ~ 1.5 in-window sits just
above the boundary Phi_c ~ 0.84, so onset takes ~13 nights of nightly
drive and onset NIGHT is sensitive to a_hold/dose; (ii) the late-at-night
subcluster (phase -> window END) appears only for marginal drives —
stronger drives collapse at window START; the model does not resolve which
regime the reported event was in; (iii) the ser = 0 column of the boundary
map is clean by CONSTRUCTION (Phi = 0 removes the weakening term), so only
the N0-leg is a dynamical finding; (iv) the mechanism linking serotonergic
tone / NGF induction to this G pathway is INFERRED from composition +
timing, not established; (v) sleep architecture is an unresolvable confound
in the single case (5-HTP -> melatonin shifts, caffeine + endurance load =
sleep pressure; sleep deprivation is an independent DPDR trigger).

Outputs: figs/f10_permissive.png, cache/exp5_*.npz.
Reproducible: `.venv/bin/python -m experiments.exp5_permissive`.
"""
from __future__ import annotations

import os

import numpy as np

from dpdr.metrics import check_gates, first_below, is_stuck
from dpdr.model import Params, Schedule
from dpdr.permissive import (NIGHT_W, T_DAY, PermissiveParams,
                             check_gates_perm, simulate_perm)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "figs")

T0 = 192.0            # first nightly window starts (after 16 h settling)
P = Params()
QP = PermissiveParams()                 # the gate (mu_N = 0)
QW = PermissiveParams(mu_N=1.0 / 42.0)  # washout control


def snap(x: float) -> float:
    return round(round(x / 0.05) * 0.05, 10)


# ------------------------------------------------------------- schedules
def nightly_channels(nights: int, ah: float = 0.9, dose: float = 1.0,
                     ser: float = 1.0, fast: bool = True, slow: bool = True,
                     drive: bool = True, t_start: float = T0) -> dict:
    """One protocol phase: `nights` nights of the canonical trigger, with
    the fast axis (ser), the slow axis (ngf), and/or the inward drive
    (a_hold + A) each switchable — the AND-gate's four arms share the
    IDENTICAL drive when drive=True."""
    ch: dict = {}
    def add(k, spans):
        ch[k] = ch.get(k, []) + spans
    ks = range(nights)
    if fast:
        add("ser", [(snap(t_start + k * T_DAY),
                     snap(t_start + k * T_DAY + NIGHT_W), ser) for k in ks])
    if slow:
        add("ngf", [(snap(t_start + k * T_DAY),
                     snap(t_start + k * T_DAY + 1.0), dose) for k in ks])
    if drive:
        add("a_hold", [(snap(t_start + k * T_DAY),
                        snap(t_start + k * T_DAY + NIGHT_W), ah) for k in ks])
        add("A", [(snap(t_start + k * T_DAY),
                   snap(t_start + k * T_DAY + 60.0), 0.5) for k in ks])
    return ch


def run_phase(ch: dict, extra: float = 600.0) -> dict:
    """Integrate a schedule ending at (last window end + extra)."""
    ends = [t1 for spans in ch.values() for (_t0, t1, _v) in spans]
    T = snap(max(ends) + extra) if ends else extra
    return simulate_perm(P, QP, Schedule(dict(ch)), T)


# ------------------------------------------------------------------ parts
def part0() -> dict:
    """Frozen gates + the wrap's fidelity at Phi pinned to 1."""
    gates = {"frozen": check_gates(P),
             "perm pin_phi=1": check_gates_perm(QP),
             "perm pin_phi=1 (washout consts)": check_gates_perm(QW)}
    return gates


def part_a() -> dict:
    """Four arms under the identical nightly canonical drive + frozen ref."""
    out = {}
    arms = [("conjunction", dict(fast=True, slow=True), 25),
            ("fast-only", dict(fast=True, slow=False), 60),
            ("slow-only", dict(fast=False, slow=True), 60),
            ("drive-only", dict(fast=False, slow=False), 60)]
    for name, kw, nights in arms:
        s = run_phase(nightly_channels(nights, **kw))
        tc = first_below(s["G"], s["t"])
        out[name] = dict(sol=s, G_min=float(s["G"].min()),
                         G_end=float(s["G"][-1]), t_collapse=tc,
                         stuck=bool(s["G"][-1] < 0.1 and is_stuck(s, P)))
    # frozen reference: ONE night of the same drive with Phi = 1 (frozen
    # default) — the OR-gate: a single factor suffices.
    ch = nightly_channels(1)
    ends = [t1 for spans in ch.values() for (_t0, t1, _v) in spans]
    T = snap(max(ends) + 600.0)
    s = simulate_perm(P, PermissiveParams(pin_phi=1.0), Schedule(ch), T)
    out["frozen-one-night"] = dict(
        sol=s, G_min=float(s["G"].min()), G_end=float(s["G"][-1]),
        t_collapse=first_below(s["G"], s["t"]),
        stuck=bool(s["G"][-1] < 0.1 and is_stuck(s, P)))
    return out


def _single_window(ser: float, N0: float, q: PermissiveParams = QP) -> dict:
    ch = {"ser": [(snap(T0), snap(T0 + NIGHT_W), ser)],
          "a_hold": [(snap(T0), snap(T0 + NIGHT_W), 0.9)],
          "A": [(snap(T0), snap(T0 + 60.0), 0.5)]}
    q0 = PermissiveParams(N0=N0, mu_N=q.mu_N, pin_phi=q.pin_phi)
    return simulate_perm(P, q0, Schedule(ch), snap(T0 + NIGHT_W + 500.0))


def _stuck(ser: float, N0: float, q: PermissiveParams = QP) -> bool:
    s = _single_window(ser, N0, q)
    return bool(s["G"][-1] < 0.1 and is_stuck(s, P))


def part_b() -> dict:
    """Single-window boundary map (stuck criterion) + bisected legs."""
    sers = np.round(np.arange(0.0, 1.01, 0.125), 4)
    N0s = np.round(np.arange(0.0, 28.01, 3.5), 4)
    grid = np.zeros((len(sers), len(N0s)), bool)
    for i, ser in enumerate(sers):
        for j, N0 in enumerate(N0s):
            grid[i, j] = _stuck(float(ser), float(N0))
    legs = {}
    for ser in (0.25, 1.0):                       # N0 leg at two amplitudes
        lo, hi = 14.0, 28.0
        if _stuck(ser, hi) and not _stuck(ser, lo):
            for _ in range(7):
                mid = 0.5 * (lo + hi)
                if _stuck(ser, mid):
                    hi = mid
                else:
                    lo = mid
        legs[f"N0_crit@ser={ser:g}"] = 0.5 * (lo + hi)
    for N0 in (17.5, 28.0):                       # ser leg at two levels
        lo, hi = 0.125, 1.0
        if _stuck(hi, N0) and not _stuck(lo, N0):
            for _ in range(6):
                mid = 0.5 * (lo + hi)
                if _stuck(mid, N0):
                    hi = mid
                else:
                    lo = mid
        legs[f"ser_crit@N0={N0:g}"] = 0.5 * (lo + hi)
    # pinned-Phi constant dose-response: the frozen model's own duration
    # threshold assay at alpha_G <- P*alpha_G (calibration, not a fit)
    dose = {}
    for Phip in (0.5, 0.8, 1.0, 1.2, 1.6, 2.0):
        q = PermissiveParams(pin_phi=Phip)
        lo, hi = 5.0, 600.0
        def col(dur: float) -> bool:
            t1 = snap(100.0 + dur)
            ch = {"a_hold": [(100.0, t1, 0.9)],
                  "A": [(100.0, min(snap(160.0), t1), 0.5)]}
            s = simulate_perm(P, q, Schedule(ch), snap(t1 + 300.0))
            return bool(s["G"][-1] < 0.1 and is_stuck(s, P))
        if col(hi) and not col(lo):
            for _ in range(8):
                mid = snap(0.5 * (lo + hi))
                if col(mid):
                    hi = mid
                else:
                    lo = mid
            dose[Phip] = 0.5 * (lo + hi)
        else:
            dose[Phip] = None
    return dict(sers=sers, N0s=N0s, grid=grid, legs=legs, thr_dur=dose)


def part_c() -> dict:
    """24-cell a_hold x dose ensemble; phase-of-onset distribution."""
    rows = []
    for ah in (0.85, 0.88, 0.90, 0.92, 0.95, 1.0, 1.05, 1.1):
        for dose in (0.9, 1.0, 1.1):
            s = run_phase(nightly_channels(30, ah=ah, dose=dose))
            tc = first_below(s["G"], s["t"])
            row = dict(ah=ah, dose=dose, t_collapse=tc,
                       phase=(None if tc is None
                              else ((tc - T0) % T_DAY) / T_DAY),
                       night=(None if tc is None else (tc - T0) / T_DAY),
                       stuck=bool(s["G"][-1] < 0.1 and is_stuck(s, P)))
            rows.append(row)
    return dict(rows=rows)


def part_d() -> dict:
    """Priming -> gap -> re-exposure; plus post-collapse withdrawal read."""
    out = {}
    n_prim, n_gap = 15, 28
    prim = nightly_channels(n_prim, drive=False)     # both axes, no drive
    t_re = T0 + (n_prim - 1) * T_DAY + NIGHT_W + n_gap * T_DAY
    for n_re in (1, 2):
        re = dict(prim)
        for k, v in nightly_channels(n_re, t_start=snap(t_re)).items():
            re[k] = re.get(k, []) + v
        ends = [t1 for spans in re.values() for (_t0, t1, _v) in spans]
        T = snap(max(ends) + 500.0)
        for name, q in (("ratchet", QP), ("washout", QW)):
            s = simulate_perm(P, q, Schedule(dict(re)), T)
            i_re = int(np.searchsorted(s["t"], t_re))
            tc = first_below(s["G"], s["t"])
            out[(n_re, name)] = dict(
                N_at_re=float(s["N"][i_re]), t_collapse=tc,
                G_end=float(s["G"][-1]),
                stuck=bool(s["G"][-1] < 0.1 and is_stuck(s, P)),
                sol=s, t_re=t_re)
    # post-collapse withdrawal: conjunction until stuck, then both axes off
    n_c, n_post = 15, 28
    ch = nightly_channels(n_c)
    ends = [t1 for spans in ch.values() for (_t0, t1, _v) in spans]
    T = snap(max(ends) + n_post * T_DAY + 400.0)
    for name, q in (("ratchet", QP), ("washout", QW)):
        s = simulate_perm(P, q, Schedule(dict(ch)), T)
        out[("withdrawal", name)] = dict(
            t_collapse=first_below(s["G"], s["t"]),
            G_end=float(s["G"][-1]), N_end=float(s["N"][-1]),
            stuck_at_end=is_stuck(s, P), sol=s)
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    print("== exp5 part 0: gates (frozen untouched; wrap fidelity at Phi=1) ==")
    gates = part0()
    for name, g in gates.items():
        print(f"  {name:28s}: {' '.join(f'{k}={v}' for k, v in g.items())}"
              f"  -> {'ALL PASS' if all(g.values()) else 'FAIL'}")

    print("\n== exp5 (a): AND-gate — identical nightly canonical drive ==")
    res_a = part_a()
    for name, r in res_a.items():
        tc = r["t_collapse"]
        night = "" if tc is None else f"  onset night {(tc - T0) / T_DAY:.2f}"
        print(f"  {name:16s}: G_min = {r['G_min']:.4f}  G(T) = {r['G_end']:.4f}"
              f"  stuck = {r['stuck']}{night}")
    conj = res_a["conjunction"]
    singles = [res_a[n] for n in ("fast-only", "slow-only", "drive-only")]
    ok_and = conj["stuck"] and not any(s["stuck"] for s in singles)
    print(f"  -> AND-gate {'DEMONSTRATED' if ok_and else 'NOT demonstrated'}:"
          f" conjunction collapses (onset night "
          f"{(conj['t_collapse'] - T0) / T_DAY:.2f}); every single axis"
          f" stays healthy (G_min >= {min(s['G_min'] for s in singles):.3f})"
          f" over 60 nights")

    print("\n== exp5 (b): single-window phase boundary (stuck criterion) ==")
    res_b = part_b()
    sers, N0s, grid = res_b["sers"], res_b["N0s"], res_b["grid"]
    print("        N0: " + " ".join(f"{n:5.1f}" for n in N0s))
    for i, ser in enumerate(sers):
        print(f"  ser={ser:5.3f}: " + " ".join(
            "  X  " if grid[i, j] else "  .  " for j in range(len(N0s))))
    for k, v in res_b["legs"].items():
        print(f"  {k} = {v:.3f}")
    print("  pinned-Phi constant dose-response thr_dur(P): "
          + ", ".join(f"P={Phip:g}->{('n/a' if v is None else f'{v:.1f}')}"
                      for Phip, v in res_b["thr_dur"].items()))

    print("\n== exp5 (c): phase-of-onset distribution (24 cells) ==")
    res_c = part_c()
    phases = [r["phase"] for r in res_c["rows"] if r["phase"] is not None]
    n_stuck = sum(r["stuck"] for r in res_c["rows"])
    in_win = sum(1 for ph in phases if ph < NIGHT_W / T_DAY)
    at_end = sum(1 for ph in phases if 0.30 <= ph < NIGHT_W / T_DAY)
    at_start = sum(1 for ph in phases if ph < 0.05)
    print(f"  collapsed+stuck: {n_stuck}/24; onsets {len(phases)}/24")
    print(f"  in-window (phase < {NIGHT_W / T_DAY:.3f}): {in_win}/{len(phases)}"
          f"   [uniform null: ~{len(phases) / 3:.0f}]")
    print(f"  subclusters: window-start (phase<0.05) {at_start}, "
          f"window-end (phase>=0.30, 'late at night') {at_end}")
    for r in res_c["rows"]:
        if r["phase"] is not None:
            print(f"    ah={r['ah']:.2f} dose={r['dose']:.1f}: night "
                  f"{r['night']:5.2f}  phase {r['phase']:.3f} "
                  f"({r['phase'] * 24:.1f} h into the 24 h cycle)")

    print("\n== exp5 (d): persistence of the slow axis (and its limits) ==")
    res_d = part_d()
    for n_re in (1, 2):
        for name in ("ratchet", "washout"):
            r = res_d[(n_re, name)]
            tc = r["t_collapse"]
            when = ("no collapse" if tc is None else
                    f"collapse {tc - r['t_re']:.1f} t.u. into re-exposure")
            print(f"  re-exposure x{n_re} [{name:7s}]: N at re-exposure = "
                  f"{r['N_at_re']:5.2f}  -> {when}  stuck = {r['stuck']}")
    for name in ("ratchet", "washout"):
        r = res_d[("withdrawal", name)]
        tc = r["t_collapse"]
        night = "" if tc is None else f" onset night {(tc - T0) / T_DAY:.2f}"
        print(f"  post-collapse withdrawal [{name:7s}]: collapse {tc}{night};"
              f" after 28 nights off: G = {r['G_end']:.4f}, N = "
              f"{r['N_end']:.2f}, stuck_at_end = {r['stuck_at_end']}"
              + ("  [G2b INHERITED from the frozen model]"
                 if r["stuck_at_end"] else ""))

    # ---------------------------------------------------------- figure
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2)

    axA = fig.add_subplot(gs[0, 0])
    for name, col in (("conjunction", "tab:red"), ("fast-only", "tab:blue"),
                      ("slow-only", "tab:green"), ("drive-only", "tab:gray")):
        s = res_a[name]["sol"]
        axA.plot((s["t"] - T0) / T_DAY, s["G"], color=col, lw=1.0,
                 label=f"{name} (G_min {res_a[name]['G_min']:.2f})")
    axA.axhline(0.1, color="k", ls=":", lw=0.8)
    axA.set_xlabel("nights since first dose")
    axA.set_ylabel("G(t)")
    axA.set_title("(a) AND-gate: identical nightly canonical drive —\n"
                  "only the conjunction collapses; frozen model: ONE night"
                  " at Phi=1\nsuffices (G_end "
                  f"{res_a['frozen-one-night']['G_end']:.3f}, stuck="
                  f"{res_a['frozen-one-night']['stuck']})")
    axA.legend(fontsize=8)

    axB = fig.add_subplot(gs[0, 1])
    axB.imshow(grid, origin="lower", aspect="auto", cmap="Greys",
               extent=[N0s[0] - 1.75, N0s[-1] + 1.75,
                       sers[0] - 0.0625, sers[-1] + 0.0625])
    for k, v in res_b["legs"].items():
        if k.startswith("N0"):
            axB.axvline(v, color="tab:red", ls="--", lw=1.0)
            axB.annotate(f"{k}={v:.1f}", xy=(v, 1.02),
                         xytext=(0, 4), textcoords="offset points",
                         fontsize=7, color="tab:red",
                         ha="left", rotation=45)
        else:
            axB.axhline(v, color="tab:blue", ls="--", lw=1.0)
            axB.annotate(f"{k}={v:.2f}", xy=(28.2, v), fontsize=7,
                         color="tab:blue", ha="right", va="bottom")
    axB.set_xlabel("slow axis: accumulated N0 (full-dose days)")
    axB.set_ylabel("fast axis: ser amplitude in window")
    axB.set_title("(b) single-window boundary (stuck criterion):\n"
                  "corner, not a band — both axes must clear threshold;\n"
                  "ser=0 column clean by construction (Phi=0)")

    axC = fig.add_subplot(gs[1, 0])
    axC.hist(phases, bins=np.arange(0.0, 1.0001, 1.0 / 24), color="tab:red")
    axC.axvspan(0.0, NIGHT_W / T_DAY, color="tab:orange", alpha=0.2,
                label=f"permissive window [0, {NIGHT_W / T_DAY:.3f})")
    axC.axhline(len(phases) / 24, color="k", ls=":", lw=0.8,
                label="uniform-onset null")
    axC.set_xlabel("phase of collapse onset in the 24 h cycle")
    axC.set_ylabel("#cells (a_hold x dose ensemble, n=24)")
    axC.set_title(f"(c) onset clusters IN-WINDOW: {in_win}/{len(phases)}; "
                  f"window-start {at_start} vs\nwindow-end "
                  f"('late at night') {at_end} — the marginal-drive "
                  "subcluster")
    axC.legend(fontsize=8)

    axD = fig.add_subplot(gs[1, 1])
    for name, col in (("ratchet", "tab:red"), ("washout", "tab:green")):
        r = res_d[(2, name)]
        s, t_re = r["sol"], r["t_re"]
        axD.plot((s["t"] - T0) / T_DAY, s["N"], color=col, lw=1.1,
                 label=f"N, {name} (mu_N="
                       f"{'0' if name == 'ratchet' else '1/42 per day'})")
        axD.plot((s["t"] - T0) / T_DAY, s["G"], color=col, lw=0.8, ls="--",
                 alpha=0.7)
    axD.axvline((t_re - T0) / T_DAY, color="k", ls=":", lw=0.8)
    axD.annotate("re-exposure\n(canonical drive)", xy=((t_re - T0) / T_DAY,
                 0.05), fontsize=7, ha="left")
    axD.set_xlabel("nights since first dose (priming 15, gap 28)")
    axD.set_ylabel("N(t) solid, G(t) dashed")
    axD.set_title("(d) slow-axis persistence: N holds (14.25 at re-exposure,"
                  "\ncollapse) vs washes out (6.18, no collapse);"
                  " post-collapse\nG-stuckness is INHERITED G2b, not the "
                  "gate's doing")
    axD.legend(fontsize=8, loc="center left")

    fig.suptitle("exp5 two-axis permissive AND-gate (opt-in variant): "
                 "Phi = g_fast(ser) x g_slow(N) x Phi_max on alpha_G — "
                 "neither axis alone collapses; conjunction does")
    fig.tight_layout()
    f10 = os.path.join(FIGDIR, "f10_permissive.png")
    fig.savefig(f10, dpi=110)
    plt.close(fig)
    print(f"\n  -> {f10}")

    # ---------------------------------------------------------- caches
    np.savez(os.path.join("cache", "exp5_and.npz"),
             **{f"{k}_Gmin": v["G_min"] for k, v in res_a.items()},
             **{f"{k}_Gend": v["G_end"] for k, v in res_a.items()},
             **{f"{k}_stuck": float(v["stuck"]) for k, v in res_a.items()})
    np.savez(os.path.join("cache", "exp5_boundary.npz"),
             sers=sers, N0s=N0s, grid=grid,
             **{f"leg_{k.replace('=', '_').replace('@', '_at_')}": v
                for k, v in res_b["legs"].items()},
             **{f"thrdur_P{Phip:g}": (np.nan if v is None else v)
                for Phip, v in res_b["thr_dur"].items()})
    np.savez(os.path.join("cache", "exp5_phase.npz"),
             ah=np.array([r["ah"] for r in res_c["rows"]]),
             dose=np.array([r["dose"] for r in res_c["rows"]]),
             phase=np.array([np.nan if r["phase"] is None else r["phase"]
                             for r in res_c["rows"]]),
             night=np.array([np.nan if r["night"] is None else r["night"]
                             for r in res_c["rows"]]),
             stuck=np.array([float(r["stuck"]) for r in res_c["rows"]]))
    np.savez(os.path.join("cache", "exp5_persist.npz"),
             **{f"reexp{n_re}_{name}_N": res_d[(n_re, name)]["N_at_re"]
                for n_re in (1, 2) for name in ("ratchet", "washout")},
             **{f"reexp{n_re}_{name}_stuck":
                float(res_d[(n_re, name)]["stuck"])
                for n_re in (1, 2) for name in ("ratchet", "washout")},
             **{f"wd_{name}_G": res_d[("withdrawal", name)]["G_end"]
                for name in ("ratchet", "washout")},
             **{f"wd_{name}_N": res_d[("withdrawal", name)]["N_end"]
                for name in ("ratchet", "washout")},
             **{f"wd_{name}_stuck":
                float(res_d[("withdrawal", name)]["stuck_at_end"])
                for name in ("ratchet", "washout")},
             gates_ok=np.array([all(g.values()) for g in gates.values()]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
