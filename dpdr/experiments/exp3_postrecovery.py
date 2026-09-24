"""exp3 — Phase 4: post-recovery characterization on rescued runs (plan §4).

On FULL-rescue runs only (taxonomy 'full' from exp2's canonical setting):
  1. Welch PSD of E(t) in a post-rescue window vs a baseline (never-collapsed)
     window — frequency shift.  G4's "PSD peak shifts" component.
  2. Log-decrement damping of successive E peaks — needs >= 2 resolvable
     peaks; reports 'none' when the response is overdamped (no peaks).
  3. Gain upregulation g*/g0 vs eta (P5 vigilance compensation).
  4. G-vs-S recovery timescale lag (P6): t90 ratio vs tau_S/tau_G.

The probes behind this driver showed the E-loop is OVERDAMPED (D is purely
exogenous — no oscillator), so (1) and (2) are expected to report no peak /
no ringing; they are computed and reported honestly rather than skipped.

Outputs: figs/f07_postrecovery.png, f07b_vigilance.png, f07b2_freq.png,
cache/exp3-*.npz.  Reproducible: `.venv/bin/python -m experiments.exp3`.
"""
from __future__ import annotations

import os
from dataclasses import replace

import numpy as np

from dpdr.events import baseline_schedule
from dpdr.integrate import simulate
from dpdr.model import Params
from dpdr.plots import FIGDIR, postrecovery

from .common import episode_schedule, rescue_schedule, rescue_taxonomy, snap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.optimize import curve_fit  # noqa: E402
from scipy.signal import find_peaks, welch  # noqa: E402

FS = 20.0            # 1/dt sampling rate of the solution grid

# Canonical rescued run (full re-couple, from exp2 probes)
CANON = dict(delay=200.0, dur=60.0, u=0.8)   # t_rescue = 400, end 460
T_END = 1800.0


def rescued_solution(p: Params, delay: float = CANON["delay"],
                     dur: float = CANON["dur"], u: float = CANON["u"],
                     T: float = T_END):
    ep = episode_schedule()
    t_r = snap(200.0 + delay)
    sch = rescue_schedule(ep, t_r, dur, u)
    sol = simulate(p, sch, T)
    return sol, t_r


def psd_peak(x: np.ndarray, fs: float = FS):
    """(freq, power) of the dominant Welch peak, or (None, total power)."""
    x = x - x.mean()
    if x.std() < 1e-12:
        return None, 0.0
    f, P = welch(x, fs=fs, nperseg=min(4096, len(x)))
    i = int(np.argmax(P[1:])) + 1            # skip DC bin
    return float(f[i]), float(P[i])


def e_peak_logdecrement(E: np.ndarray, t: np.ndarray, t0: float,
                        prominence: float = 0.002):
    """Log-decrement of successive E peaks after t0: delta_n =
    ln(A_n / A_{n+1}).  Returns (peaks_t, peaks_E, deltas) — empty when the
    response has fewer than 2 resolvable peaks (overdamped)."""
    w = t >= t0
    tw, Ew = t[w], E[w]
    pk, _ = find_peaks(Ew, prominence=prominence)
    if len(pk) < 2:
        return tw[pk], Ew[pk], np.array([])
    A = Ew[pk] - Ew[-1]                       # amplitudes about the settle
    A = np.abs(A)
    deltas = np.log(A[:-1] / A[1:])
    return tw[pk], Ew[pk], deltas


def gain_asymptote(p: Params, delay: float = CANON["delay"],
                   dur: float = CANON["dur"], u: float = CANON["u"],
                   T: float = 6000.0):
    """P5 steady-state gain: is g* > g0 REAL or finite-horizon residue?

    dg/dt = (pi*max(0,|dE/dt|-ref) - mu*(g-g0))/tau_g relaxes toward g0 with
    time constant tau_g/mu ~ 667 t.u. once E settles, so g(T) at any practical
    horizon is a transient.  Integrate to T, then fit
        g(t) - g0 = C + A*exp(-(t-tpk)/tau)
    over the post-peak window; C is the PERSISTENT offset (the t->inf
    asymptote), A*exp(...) the decaying relaxation residue.  C ~ 0 means the
    falsifier 'rescued == naive gains' is triggered (P5 FAIL).
    Returns (sol, fit) with fit = dict(peak_t, peak_g, peak_up_pct, C, A,
    tau, asym, asym_up_pct, ratio_T).
    """
    sol, t_r = rescued_solution(p, delay, dur, u, T=T)
    t, g = sol["t"], sol["g"]
    dev = g - p.g0
    ipk = int(np.argmax(g))
    m = slice(ipk, None, 20)             # post-peak window, subsampled

    def _f(x, C, A, tau):
        return C + A * np.exp(-x / tau)

    (C, A, tau), _cov = curve_fit(_f, t[m] - t[ipk], dev[m],
                                  p0=(0.0, dev[ipk], p.tau_g / p.mu))
    fit_d = dict(peak_t=float(t[ipk]), peak_g=float(g[ipk]),
                 peak_up_pct=float(100 * dev[ipk] / p.g0),
                 C=float(C), A=float(A), tau=float(tau),
                 asym=float(p.g0 + C),
                 asym_up_pct=float(100 * C / p.g0),
                 ratio_T=float(100 * (g[-1] / p.g0 - 1.0)))
    return sol, fit_d


def main() -> int:
    p = Params()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs("cache", exist_ok=True)
    sol, t_r = rescued_solution(p)
    t = sol["t"]

    # ---------------- 1+2: PSD + damping on the canonical rescued run
    lab = rescue_taxonomy(sol, p, t_r)
    print(f"== exp3: canonical rescued run (t_r={t_r:.0f}) taxonomy={lab} ==")
    t_end_resc = t_r + CANON["dur"]
    base = simulate(p, baseline_schedule(), T_END)

    f_b, P_b = psd_peak(base["E"][base["t"] >= 600.0])
    f_r, P_r = psd_peak(sol["E"][t >= t_end_resc + 200.0])
    print(f"  PSD peak: baseline f={f_b} P={P_b:.3e} | "
          f"post-rescue f={f_r} P={P_r:.3e} | shift={f_r - f_b if f_r and f_b else 'n/a'}")

    pt, pE, deltas = e_peak_logdecrement(sol["E"], t, t_end_resc)
    if len(deltas):
        print(f"  E peaks at t={np.round(pt, 1)}; log-decrements="
              f"{np.round(deltas, 3)}")
    else:
        print("  E peaks after rescue end: NONE (response overdamped; "
              "log-decrement undefined) — G4 ringing component FAILS")

    # ---------------- 3: g*/g0 vs eta
    print("== exp3: g*/g0 vs eta (vigilance compensation, P5) ==")
    etas = np.round(np.linspace(0.2, 1.4, 13), 4)
    gstar, gok = np.zeros(len(etas)), np.zeros(len(etas), bool)
    for i, eta in enumerate(etas):
        q = replace(p, eta=float(eta))
        s, tr = rescued_solution(q)
        lab_i = rescue_taxonomy(s, p, tr)
        gok[i] = lab_i == "full"
        gstar[i] = s["g"][-1] / q.g0
        print(f"  eta={eta:.2f}: taxonomy={lab_i:9s} g*/g0={gstar[i]:.5f}")

    # ---------------- 3b: g asymptote (P5 steady state)
    # g(T)/g0 at the finite horizon is relaxation residue: the g equation's
    # equilibrium once E settles is g0, with time constant tau_g/mu ~ 667.
    print("== exp3: gain asymptote (P5 steady state) ==")
    print("  horizon scan g(T)/g0-1 on the canonical rescued run:")
    for T in (1000.0, 1800.0, 3000.0, 6000.0, 12000.0):
        s, _ = rescued_solution(p, T=T)
        print(f"    T={T:6.0f}: {100 * (s['g'][-1] / p.g0 - 1):+.5f}%")
    gsl, gf = gain_asymptote(p)
    print(f"  peak g = {gf['peak_g']:.4f} ({gf['peak_up_pct']:+.2f}%) at "
          f"t = {gf['peak_t']:.0f}")
    print(f"  fit C + A*exp(-t/tau): tau = {gf['tau']:.0f} t.u. "
          f"(tau_g/mu = {p.tau_g / p.mu:.0f}), A = {gf['A']:.5f}")
    print(f"  fitted persistent offset C = {gf['C']:.2e} "
          f"({gf['asym_up_pct']:+.5f}% of g0) -> P5 gain component: "
          f"{'PASS' if gf['C'] > 0 else 'FAIL'} "
          f"(rescued == naive gains {'not ' if gf['C'] > 0 else ''}triggered)")

    # ---------------- 4: P6 lag (t90 S / t90 G) — reused measurement
    from .exp2_rescue import t90
    tG = t90(sol["G"], t, t_r)
    tS = t90(sol["S"], t, t_r)
    ratio = tS / tG if (tG and tS) else float("nan")
    print(f"== exp3: P6 lag: t90(G)={tG:.1f} t90(S)={tS:.1f} "
          f"ratio={ratio:.2f} (tau_S/tau_G={p.tau_S / p.tau_G:.1f}) ==")

    # ---------------- figures
    # f07: E(t) post-rescue + PSD inset + g(t)
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(3, 2, width_ratios=[2, 1])
    axE = fig.add_subplot(gs[0, 0])
    axP = fig.add_subplot(gs[0, 1])
    axg = fig.add_subplot(gs[1, 0], sharex=axE)
    axb = fig.add_subplot(gs[1, 1])
    axS = fig.add_subplot(gs[2, :], sharex=axE)
    w = t >= t_r
    axE.plot(t[w], sol["E"][w], lw=0.8)
    axE.axhline(0, color="k", lw=0.5)
    axE.set_ylabel("E(t)")
    axE.set_title("rescued run: E(t) after rescue onset")
    for f0, P0, name, ax_ in [(f_b, P_b, "baseline", axP),
                              (f_r, P_r, "post-rescue", axP)]:
        x_ = (base["E"][base["t"] >= 600.0] if name == "baseline"
              else sol["E"][t >= t_end_resc + 200.0])
        f_, P_ = welch(x_ - x_.mean(), fs=FS, nperseg=min(4096, len(x_)))
        ax_.semilogy(f_[1:], P_[1:], lw=0.8,
                     label=f"{name} (peak f={f0:.3f})" if f0 else name)
    axP.legend(fontsize=7); axP.set_xlabel("freq (1/t.u.)")
    axP.set_title("Welch PSD of E")
    axg.plot(t[w], sol["g"][w], lw=0.8, color="tab:green",
             label=f"g (peak +{(sol['g'][np.argmax(sol['g'])] / p.g0 - 1) * 100:.1f}% "
                   f"at t={t[np.argmax(sol['g'])]:.0f}, -> g0)")
    axg.axhline(p.g0, color="gray", ls=":", lw=0.8, label="g0 (asymptote)")
    axg.legend(fontsize=7); axg.set_ylabel("g(t)")
    if len(pt):
        axb.plot(pt, pE, "o-")
        axb.set_title("successive E peaks")
    else:
        axb.text(0.5, 0.5, "no resolvable E peaks\n(overdamped)",
                 ha="center", va="center", transform=axb.transAxes)
    axS.plot(t[w], sol["G"][w] / sol["G"][-1], label="G/x_end", lw=0.8)
    axS.plot(t[w], sol["S"][w] / sol["S"][-1], label="S/x_end", lw=0.8)
    axS.axhline(0.9, color="gray", ls=":", lw=0.8)
    axS.legend(fontsize=7); axS.set_ylabel("normalized"); axS.set_xlabel("t")
    axS.set_title(f"P6: t90 S/G = {ratio:.2f}")
    fig.suptitle("exp3 post-recovery characterization")
    fig.tight_layout()
    f07 = os.path.join(FIGDIR, "f07_postrecovery.png")
    fig.savefig(f07, dpi=110)
    plt.close(fig)
    print(f"  -> {f07}")

    # f07b: g*/g0 vs eta
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(etas[gok], gstar[gok], "o-", label="full-rescue runs")
    if (~gok).any():
        ax.plot(etas[~gok], gstar[~gok], "x", color="gray",
                label="not full (excluded)")
    ax.axhline(1.0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("eta (cannibalization cost)")
    ax.set_ylabel("g*/g0")
    dg = (gstar[gok].max() - gstar[gok].min()) if gok.any() else float("nan")
    ax.set_title(f"transient g(T)/g0 vs eta (T={T_END:.0f}; decays to g0 at "
                 f"steady state — P5 gain FAIL; max-min = {dg:.4f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    f07b = os.path.join(FIGDIR, "f07b_vigilance.png")
    fig.savefig(f07b, dpi=110)
    plt.close(fig)
    print(f"  -> {f07b}")

    # f07b2: ringing frequency vs g0 (P5 'faster corrections')
    print("== exp3: PSD peak freq vs g0 (rescued vs baseline) ==")
    g0s = np.round(np.linspace(0.2, 3.0, 15), 4)
    fr, fb = np.full(len(g0s), np.nan), np.full(len(g0s), np.nan)
    sr, sb = np.full(len(g0s), np.nan), np.full(len(g0s), np.nan)
    for i, g0 in enumerate(g0s):
        q = replace(p, g0=float(g0), g_init=float(g0))
        s, tr = rescued_solution(q)
        lab_i = rescue_taxonomy(s, q, tr)
        if lab_i == "full":
            f_, std_ = psd_peak(s["E"][s["t"] >= t_end_resc + 200.0])
            fr[i], sr[i] = f_, s["E"][s["t"] >= t_end_resc + 200.0].std()
        b = simulate(q, baseline_schedule(), 800.0)
        f_, std_ = psd_peak(b["E"][b["t"] >= 400.0])
        fb[i], sb[i] = f_, std_
        print(f"  g0={g0:.2f}: {lab_i:11s} rescued f={fr[i]} std={sr[i]:.2e} | "
              f"baseline f={fb[i]} std={sb[i]:.2e}")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    m = ~np.isnan(fr)
    axs[0].plot(g0s[m], fr[m], "o-", label="rescued peak freq")
    axs[0].plot(g0s, fb, "s--", label="baseline peak freq", alpha=0.6)
    axs[0].set_xlabel("g0"); axs[0].set_ylabel("PSD peak freq (1/t.u.)")
    axs[0].legend(fontsize=8)
    axs[1].semilogy(g0s[m], sr[m], "o-", label="rescued E std")
    axs[1].semilogy(g0s, sb, "s--", label="baseline E std", alpha=0.6)
    axs[1].set_xlabel("g0"); axs[1].set_ylabel("E std in window")
    axs[1].legend(fontsize=8)
    fig.suptitle("exp3: post-recovery E dynamics vs g0 (P5)")
    fig.tight_layout()
    f07b2 = os.path.join(FIGDIR, "f07b2_freq.png")
    fig.savefig(f07b2, dpi=110)
    plt.close(fig)
    print(f"  -> {f07b2}")

    np.savez(os.path.join("cache", "exp3_summary.npz"),
             etas=etas, gstar=gstar, gok=gok, g0s=g0s, fr=fr, fb=fb,
             sr=sr, sb=sb, t90_G=tG, t90_S=tS, ratio=ratio,
             f_base=f_b, f_resc=f_r, P_base=P_b, P_resc=P_r,
             n_peaks=len(pt), deltas=deltas,
             gain_peak_t=gf["peak_t"], gain_peak_g=gf["peak_g"],
             gain_peak_up_pct=gf["peak_up_pct"], gain_tau=gf["tau"],
             gain_C=gf["C"], gain_A=gf["A"],
             gain_asym=gf["asym"], gain_asym_up_pct=gf["asym_up_pct"],
             gain_ratio_T=gf["ratio_T"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
