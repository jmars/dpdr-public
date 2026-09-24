"""exp2 — Phase 3/4: rescue taxonomy and hysteresis (plan §4, figs f06/f08).

Phase 3: rescue outcome taxonomy over (rescue timing x strength x duration),
three classes per plan §4:
    full        G(T) > 0.5, E(T) < 0.8*Theta, and no re-collapse after the
                post-rescue peak (plan: "full re-couple")
    transient   G rose past 0.5 after collapse, then re-crossed < 0.1
                ("transient lift + relapse"; relapse = SECOND G<0.1 crossing)
    failure     G never exceeded 0.5 after collapsing
The rescue always starts AFTER episode end (t >= 200), and the episode offset
is swept explicitly (rescue delay 30..600 past episode end) — the plan's
relapse-confound control (risk #5).

Phase 3c (recurring episodes, P4): the single-episode grids above cannot
relapse because after a full rescue E <= ~0.15 << Theta_eff ~ 0.76 — but that
argument holds only while NO new trigger arrives.  Here the canonical rescued
run (rescue [400,460), full re-couple to G ~ 0.885) is hit by a SECOND inward
episode (a_hold 0.9 + the same affect pulse) starting at t=800, sweeping the
second-episode duration.  No new mechanism, no retuning — schedule only.
Relapse here is exactly plan §4's class 2 ('transient lift + relapse'), and
the shipped dpdr.metrics.classify()/detect_relapse() are checked against it.

Phase 4 (hysteresis, f08): one canonical rescued run, G(t) and S(t)
recovery trajectories; recovery timescales measured as t90 (time from rescue
onset to reach 90% of the final settled value).  Prediction P6: tau ratio
t90(S)/t90(G) ~ tau_S/tau_G = 5.

Outputs: figs/f06_rescue_map.png, figs/f06b_relapse.png,
figs/f08_hysteresis.png, cache/exp2-*.npz.  Reproducible:
`.venv/bin/python -m experiments.exp2`.
"""
from __future__ import annotations

import os

import numpy as np

from dpdr.integrate import simulate
from dpdr.metrics import classify, detect_relapse, first_below
from dpdr.model import Params, Schedule
from dpdr.plots import FIGDIR

from .common import (TAXONOMY_CODES, episode_schedule, rescue_schedule,
                     rescue_taxonomy, snap)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Rescue grid: timing = delay past episode end (200); strength; duration.
DELAYS = np.array([30, 60, 100, 200, 400, 600], float)     # t_rescue = 230..800
STRENGTHS = np.round(np.linspace(0.2, 1.0, 17), 4)
DURATIONS = np.array([10, 20, 30, 40, 60, 80, 120, 200], float)

# Confound demonstration (plan risk #5): rescue OVERLAPPING the episode
# (delay < 0) — the design the controlled grid excludes.  Shows relapse is
# reachable only while the trigger is still active.
CONF_ONSET = np.array([120.0, 140.0, 160.0, 180.0])        # during episode
CONF_STRENGTHS = np.round(np.linspace(0.4, 1.0, 7), 4)
CONF_DURATIONS = np.array([20.0, 60.0, 100.0, 200.0])

T_END = 1800.0       # >= rescue_end + 800 (2x the longest recovery + stay)

# Phase 3c (P4 recurring episodes): a second inward episode of the same shape
# as the first (a_hold 0.9 + affect pulse 0.5 lasting 60 t.u.) hits the
# canonical rescued run (rescue [400,460) -> full re-couple) at t=800,
# sweeping duration and pulse amplitude.
EP2_T0 = 800.0
EP2_PULSE_LEN = 60.0
EP2_DURATIONS = np.array([50, 75, 100, 125, 150, 175, 200, 250, 300], float)
EP2_PULSES = np.array([0.0, 0.5])
EP2_T_END = 1600.0    # >= ep2_end + 300 (second collapse is FASTER than the first)


def taxonomy_grid(p: Params):
    """3-D grid (delay x strength x duration) -> taxonomy code + G_end."""
    grid = np.zeros((len(DELAYS), len(STRENGTHS), len(DURATIONS)), int)
    Gend = np.zeros_like(grid, float)
    for i, delay in enumerate(DELAYS):
        ep = episode_schedule()
        for j, u in enumerate(STRENGTHS):
            for k, dur in enumerate(DURATIONS):
                t_r = snap(200.0 + delay)
                sch = rescue_schedule(ep, t_r, dur, u)
                sol = simulate(p, sch, T_END)
                lab = rescue_taxonomy(sol, p, t_r)
                grid[i, j, k] = TAXONOMY_CODES[lab]
                Gend[i, j, k] = sol["G"][-1]
    return grid, Gend


def confound_grid(p: Params):
    """Rescue starting DURING the episode: taxonomy over (onset x strength x
    duration).  Relapse here is the trivially-confounded kind."""
    grid = np.zeros((len(CONF_ONSET), len(CONF_STRENGTHS),
                     len(CONF_DURATIONS)), int)
    for i, t_r in enumerate(CONF_ONSET):
        ep = episode_schedule()
        for j, u in enumerate(CONF_STRENGTHS):
            for k, dur in enumerate(CONF_DURATIONS):
                sch = rescue_schedule(ep, snap(t_r), dur, u)
                sol = simulate(p, sch, T_END)
                grid[i, j, k] = TAXONOMY_CODES[
                    rescue_taxonomy(sol, p, t_r)]
    return grid


def second_episode_grid(p: Params):
    """Phase 3c (P4): recurring episodes.  The canonical rescued run (episode
    [100,200), rescue [400,460) -> 'full', G ~ 0.885) is hit by a SECOND
    inward episode at t=800 with the same a_hold=0.9 and 60-t.u. affect pulse
    as the first, sweeping second-episode duration x pulse amplitude.
    Returns the taxonomy grid, final-G grid, and for each cell whether the
    shipped classify() agrees ('relapsed' <-> taxonomy 'transient')."""
    n_d, n_p = len(EP2_DURATIONS), len(EP2_PULSES)
    grid = np.zeros((n_d, n_p), int)
    Gend = np.zeros_like(grid, float)
    agree = np.zeros_like(grid, bool)
    base = rescue_schedule(episode_schedule(), snap(400.0), 60.0, 0.8)
    for i, dur in enumerate(EP2_DURATIONS):
        for j, pulse in enumerate(EP2_PULSES):
            ch = dict(base.channels)
            ch["a_hold"] = ch["a_hold"] + [(EP2_T0, snap(EP2_T0 + dur), 0.9)]
            if pulse != 0.0:
                ch["A"] = ch["A"] + [(EP2_T0, snap(EP2_T0 + EP2_PULSE_LEN),
                                      pulse)]
            sol = simulate(p, Schedule(ch), EP2_T_END)
            t_r = snap(400.0)
            lab = rescue_taxonomy(sol, p, t_r)
            grid[i, j] = TAXONOMY_CODES[lab]
            Gend[i, j] = sol["G"][-1]
            agree[i, j] = (classify(sol, p) == "relapsed") == (lab == "transient")
    return grid, Gend, agree


def second_episode_threshold(p: Params, pulse: float = 0.5,
                             lo: float = 5.0, hi: float = 200.0,
                             iters: int = 10) -> float | None:
    """Bisect the SECOND-episode duration threshold (P4 like-for-like): the
    same bisection exp1 runs for the first episode, but starting from the
    canonical rescued state (G ~ 0.885) instead of the healthy initial state
    (G0 = 0.7).  Episode 1's threshold at these parameters is 66.3 t.u.; if
    episode 2's is comparable or lower, G's head start does not protect."""
    def collapsed(dur):
        ch = dict(rescue_schedule(episode_schedule(), snap(400.0), 60.0,
                                  0.8).channels)
        ch["a_hold"] = ch["a_hold"] + [(EP2_T0, snap(EP2_T0 + dur), 0.9)]
        if pulse != 0.0:
            ch["A"] = ch["A"] + [(EP2_T0, snap(EP2_T0 + EP2_PULSE_LEN),
                                  pulse)]
        T = max(EP2_T_END, snap(EP2_T0 + dur) + 300.0)
        return simulate(p, Schedule(ch), T)["G"][-1] < 0.1
    if collapsed(lo) or not collapsed(hi):
        return None
    for _ in range(iters):
        mid = snap(0.5 * (lo + hi))
        if collapsed(mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def hysteresis_run(p: Params, delay: float = 200.0, dur: float = 60.0,
                   u: float = 0.8):
    """Canonical rescued run for the P6 lag measurement."""
    ep = episode_schedule()
    t_r = snap(200.0 + delay)
    sch = rescue_schedule(ep, t_r, dur, u)
    sol = simulate(p, sch, T_END)
    return sol, t_r


def t90(x: np.ndarray, t: np.ndarray, t0: float, direction: str = "up"):
    """Time from t0 until x reaches 90% of its final settled change (None if
    it never does).  For 'down' variables (S falls then recovers upward too —
    S_rest < S_recovery here both rise), we measure the upward recovery to 90%
    of the final value."""
    xf = x[-1]
    w = t >= t0
    tw, xw = t[w], x[w]
    tgt = 0.9 * xf
    idx = np.where(xw >= tgt)[0]
    return None if idx.size == 0 else float(tw[idx[0]] - t0)


def main() -> int:
    p = Params()
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs("cache", exist_ok=True)

    print("== exp2 Phase 3: rescue taxonomy (delay x strength x duration) ==")
    grid, Gend = taxonomy_grid(p)
    total = grid.size
    frac = {lab: float((grid == code).sum()) / total
            for lab, code in TAXONOMY_CODES.items()}
    for lab in ("failure", "transient", "full", "no-collapse"):
        print(f"  {lab:10s}: {frac.get(lab, 0):.3f}")
    np.savez(os.path.join("cache", "exp2_taxonomy.npz"), grid=grid, Gend=Gend,
             delays=DELAYS, strengths=STRENGTHS, durations=DURATIONS)

    print("== exp2 Phase 3b: confound demonstration (rescue DURING episode) ==")
    cgrid = confound_grid(p)
    cfrac = {lab: float((cgrid == code).sum()) / cgrid.size
             for lab, code in TAXONOMY_CODES.items()}
    for lab in ("failure", "transient", "full", "no-collapse"):
        print(f"  {lab:10s}: {cfrac.get(lab, 0):.3f}")
    np.savez(os.path.join("cache", "exp2_confound.npz"), grid=cgrid,
             onsets=CONF_ONSET, strengths=CONF_STRENGTHS,
             durations=CONF_DURATIONS)

    print("== exp2 Phase 3c: recurring episodes (P4 relapse) ==")
    print("  canonical rescued run (ep [100,200), rescue [400,460)) hit by a")
    print("  second inward episode at t=800: same a_hold 0.9, swept duration/pulse")
    rgrid, rGend, ragree = second_episode_grid(p)
    rfrac = {lab: float((rgrid == code).sum()) / rgrid.size
             for lab, code in TAXONOMY_CODES.items()}
    for lab in ("failure", "transient", "full", "no-collapse"):
        print(f"  {lab:10s}: {rfrac.get(lab, 0):.3f}")
    for j, pulse in enumerate(EP2_PULSES):
        trans = EP2_DURATIONS[rgrid[:, j] == 1]
        full = EP2_DURATIONS[rgrid[:, j] == 2]
        if len(trans) and len(full):
            print(f"  pulse={pulse:.1f}: full -> relapse threshold between "
                  f"dur {full.max():.0f} and {trans.min():.0f} t.u. "
                  f"(one grid step)")
    print(f"  dpdr.metrics.classify() == 'relapsed' agrees with the taxonomy "
          f"in {int(ragree.sum())}/{ragree.size} cells")
    np.savez(os.path.join("cache", "exp2_relapse.npz"), grid=rgrid,
             Gend=rGend, durations=EP2_DURATIONS, pulses=EP2_PULSES,
             ep2_t0=EP2_T0, t_end=EP2_T_END, classify_agree=ragree)

    # The canonical relapse counterexample, in full detail (predictions.md P4)
    i0 = int(np.argmin(np.abs(EP2_DURATIONS - 150.0)))
    j0 = int(np.argmin(np.abs(EP2_PULSES - 0.5)))
    ch = dict(rescue_schedule(episode_schedule(), snap(400.0), 60.0,
                              0.8).channels)
    d2 = float(EP2_DURATIONS[i0])
    ch["a_hold"] = ch["a_hold"] + [(EP2_T0, snap(EP2_T0 + d2), 0.9)]
    ch["A"] = ch["A"] + [(EP2_T0, snap(EP2_T0 + EP2_PULSE_LEN), 0.5)]
    rel = simulate(p, Schedule(ch), EP2_T_END)
    rel_lab = rescue_taxonomy(rel, p, snap(400.0))
    relapsed, t_rel = detect_relapse(rel, p)
    t_coll1 = first_below(rel["G"], rel["t"])
    w2 = rel["t"] >= EP2_T0
    t_coll2 = first_below(rel["G"][w2], rel["t"][w2])   # absolute time
    G_pre = float(rel["G"][np.searchsorted(rel["t"], EP2_T0)])
    print(f"  counterexample (dur2={d2:.0f}, pulse=0.5): "
          f"G {G_pre:.3f} -> {rel['G'][-1]:.3f}, c(T)={rel['c'][-1]:.2f}, "
          f"Theta_eff(T)={rel['Theta_eff'][-1]:.3f}")
    print(f"  taxonomy={rel_lab}  classify()={classify(rel, p)}  "
          f"detect_relapse t={t_rel:.1f}")
    print(f"  collapse speed: first G<0.1 at t={t_coll1:.0f} "
          f"({t_coll1 - 100:.0f} after ep1 start), second at t={t_coll2:.0f} "
          f"({t_coll2 - EP2_T0:.0f} after ep2 start) — identical onset "
          f"despite the second starting from G=0.885 vs 0.7: the recovered "
          f"head start buys no delay")
    # P4 like-for-like: second-episode duration threshold vs the first's
    # (66.3 t.u. from exp1).  A comparable threshold means the recovered
    # state's G ~ 0.885 head start does NOT protect against a fresh episode.
    thr2_p = second_episode_threshold(p, pulse=0.5)
    thr2_np = second_episode_threshold(p, pulse=0.0)
    print(f"  episode-duration collapse threshold: first (healthy start) "
          f"66.3 t.u.; second (recovered start, pulse) "
          f"{thr2_p:.1f}; second (no pulse) {thr2_np:.1f}")

    np.savez(os.path.join("cache", "exp2_relapse_run.npz"),
             t=rel["t"], G=rel["G"], S=rel["S"], c=rel["c"],
             E=rel["E"], Theta_eff=rel["Theta_eff"], dur2=d2,
             t_relapse=t_rel, t_coll2=t_coll2, G_pre=G_pre,
             thr2_pulse=thr2_p, thr2_nopulse=thr2_np)

    # f06: one panel per delay: strength x duration -> taxonomy (3 colors)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#d62728", "#ff7f0e", "#2ca02c", "#7f7f7f"])
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, i in zip(axs.flat, range(len(DELAYS))):
        sub = grid[i]     # (strength, duration)
        ax.imshow(sub.T, origin="lower", aspect="auto", vmin=0, vmax=3,
                  cmap=cmap, interpolation="nearest",
                  extent=[STRENGTHS.min(), STRENGTHS.max(),
                          DURATIONS.min(), DURATIONS.max()])
        ax.set_title(f"rescue delay {DELAYS[i]:.0f} past ep end")
        ax.set_xlabel("u_ext")
        if ax is axs.flat[0]:
            ax.set_ylabel("duration")
    import matplotlib.patches as mpatches
    handles = [mpatches.Patch(color=cmap(i), label=lab)
               for lab, i in [("failure", 0), ("transient(relapse)", 1),
                              ("full re-couple", 2), ("no-collapse", 3)]]
    fig.legend(handles=handles, loc="upper center", ncol=4)
    fig.suptitle("exp2 Phase 3: rescue taxonomy map (u_ext x duration x delay)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    f06 = os.path.join(FIGDIR, "f06_rescue_map.png")
    fig.savefig(f06, dpi=110)
    plt.close(fig)
    print(f"  -> {f06}")

    # f06b: recurring-episode relapse — taxonomy vs second-episode duration
    # (two panels: pulse 0 / 0.5) + the counterexample trajectory
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5),
                            gridspec_kw={"width_ratios": [1, 1, 2]})
    for ax, j in zip(axs[:2], range(len(EP2_PULSES))):
        sub = rgrid[:, j]
        ax.imshow(sub[None, :], aspect="auto", vmin=0, vmax=3, cmap=cmap,
                  interpolation="nearest",
                  extent=[EP2_DURATIONS.min() - 12.5,
                          EP2_DURATIONS.max() + 12.5, 0, 1])
        trans = EP2_DURATIONS[sub == 1]
        full = EP2_DURATIONS[sub == 2]
        if len(trans) and len(full):
            ax.axvline(0.5 * (full.max() + trans.min()), color="k",
                       ls="--", lw=1)
        ax.set_yticks([])
        ax.set_xlabel("second-episode duration (t.u.)")
        ax.set_title(f"pulse {EP2_PULSES[j]:.1f}")
    axs[0].set_ylabel("outcome")
    relz = np.load(os.path.join("cache", "exp2_relapse_run.npz"))
    axs[2].axvspan(100, 200, color="tab:red", alpha=0.12, label="episodes")
    axs[2].axvspan(800, 800 + float(relz["dur2"]), color="tab:red", alpha=0.12)
    axs[2].axvspan(400, 460, color="tab:green", alpha=0.15, label="rescue")
    axs[2].plot(relz["t"], relz["G"], color="tab:blue", label="G")
    axs[2].axhline(0.1, color="gray", ls=":", lw=0.8)
    axs[2].axhline(0.5, color="gray", ls=":", lw=0.8)
    ax2 = axs[2].twinx()
    ax2.plot(relz["t"], relz["S"], color="tab:orange", lw=0.8, alpha=0.8)
    ax2.plot(relz["t"], relz["Theta_eff"], color="tab:purple", lw=0.8,
             ls="--", alpha=0.8)
    ax2.set_ylabel("S, Theta_eff", color="tab:orange")
    axs[2].set_xlabel("t")
    axs[2].set_ylabel("G", color="tab:blue")
    axs[2].legend(loc="lower left", fontsize=7)
    axs[2].set_title(
        f"relapse counterexample (dur2={float(relz['dur2']):.0f}, pulse 0.5): "
        f"G {float(relz['G_pre']):.3f} -> {float(relz['G'][-1]):.3f}; "
        f"classify()=relapsed, detect_relapse t={float(relz['t_relapse']):.0f}")
    fig.suptitle("exp2 Phase 3c: recurring episodes — P4 relapse class EXISTS "
                 "(single-episode schedules cannot relapse; these can)")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    f06b = os.path.join(FIGDIR, "f06b_relapse.png")
    fig.savefig(f06b, dpi=110)
    plt.close(fig)
    print(f"  -> {f06b}")

    print("== exp2 Phase 4: hysteresis (P6 G-vs-S recovery lag) ==")
    sol, t_r = hysteresis_run(p)
    t, G, S = sol["t"], sol["G"], sol["S"]
    tG = t90(G, t, t_r)
    tS = t90(S, t, t_r)
    ratio = (tS / tG) if (tG and tS) else None
    print(f"  rescue at t={t_r:.0f}: t90(G)={tG:.1f}  t90(S)={tS:.1f}  "
          f"ratio={ratio:.2f} (predicted tau_S/tau_G = "
          f"{p.tau_S / p.tau_G:.1f})")

    # f08: G and S (normalized to their final values) vs t, from rescue on
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    w = t >= 100.0
    ax = axs[0]
    ax.plot(t[w], G[w], label="G", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(t[w], S[w], label="S", color="tab:orange")
    ax.set_xlabel("t"); ax.set_ylabel("G", color="tab:blue")
    ax2.set_ylabel("S", color="tab:orange")
    ax.axvspan(t_r, t_r + 60, color="tab:green", alpha=0.15)
    ax.set_title("raw G(t), S(t)")
    ax = axs[1]
    w2 = t >= t_r
    Gn = G[w2] / G[-1]
    Sn = S[w2] / S[-1]
    ax.plot(t[w2] - t_r, Gn, label=f"G (t90={tG:.0f})", color="tab:blue")
    ax.plot(t[w2] - t_r, Sn, label=f"S (t90={tS:.0f})", color="tab:orange")
    ax.axhline(0.9, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("t - t_rescue"); ax.set_ylabel("x / x_final")
    ax.set_title(f"P6 recovery lag: t90(S)/t90(G) = {ratio:.2f} "
                 f"(tau_S/tau_G = {p.tau_S / p.tau_G:.1f})")
    ax.legend()
    fig.suptitle("exp2 Phase 4: temporal-depth hysteresis")
    fig.tight_layout()
    f08 = os.path.join(FIGDIR, "f08_hysteresis.png")
    fig.savefig(f08, dpi=110)
    plt.close(fig)
    print(f"  -> {f08}")
    np.savez(os.path.join("cache", "exp2_hysteresis.npz"),
             t=t, G=G, S=S, t_rescue=t_r, t90_G=tG, t90_S=tS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
