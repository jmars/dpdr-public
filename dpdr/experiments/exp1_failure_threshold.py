"""exp1 — Phase 1/2: failure thresholds and regime maps (plan §4, figs f04/f05).

Phase 1: 1-D bifurcation curves over seven axes (alpha_G, eta, Theta, g0,
episode duration, episode intensity, affect amplitude A), each axis bracketing
the healthy/stuck transition seen in probes.  The saddle-node boundary on each
axis is located by bisection on the discontinuity of final G (threshold
resolution 1e-3, stopping when the bracket is that wide or after 24 steps).

Phase 2: 2-D regime heatmap (alpha_G x episode duration) -> {healthy,
recovered(escaped), stuck}, classified with dpdr.metrics.classify.

Outputs: figs/f04_sweep_heatmap.png, figs/f05_bifurcation.png,
cache/exp1-*.npz.  Reproducible: `.venv/bin/python -m experiments.exp1`.
"""
from __future__ import annotations

import os

import numpy as np

from dpdr.metrics import classify, summarize
from dpdr.model import Params, Schedule
from dpdr.plots import FIGDIR, bifurcation_diagram, sweep_heatmap
from dpdr.sweep import sweep

from .common import EP_T0, episode_schedule, one_run

# Axes bracketing the transition (probe-verified ranges).
AXES = {
    "alpha_G": (np.linspace(0.4, 1.6, 25), "params"),
    "eta": (np.linspace(0.0, 2.4, 25), "params"),
    "Theta": (np.linspace(0.25, 0.55, 25), "params"),
    "g0": (np.linspace(0.1, 1.5, 25), "params"),
    "ep_duration": (np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120,
                              150, 200]), "sched"),
    "ep_intensity": (np.linspace(0.2, 1.0, 25), "sched"),
    "pulse_amp": (np.linspace(0.0, 1.0, 25), "sched"),
}

T_1D = 800.0          # episode ends <= 300; +10*tau_G margin
COLLAPSED = 0.1


def final_state(p: Params, axis: str, val: float) -> dict:
    """One Phase-1 cell: (possibly replaced param) + schedule variant."""
    if axis in ("alpha_G", "eta", "Theta", "g0"):
        from dataclasses import replace
        q = replace(p, **{axis: float(val)})
        if axis == "g0":
            q = replace(q, g_init=float(val))
        sch = episode_schedule()
    else:
        q = p
        if axis == "ep_duration":
            sch = episode_schedule(t1=EP_T0 + val)
        elif axis == "ep_intensity":
            sch = episode_schedule(intensity=val)
        else:                                  # pulse_amp
            sch = episode_schedule(pulse=val)
    return one_run(q, sch, T_1D)


def bisect_threshold(p: Params, axis: str, lo: float, hi: float,
                     tol: float = 1e-3, max_iter: int = 24) -> float | None:
    """Bisection on the final-G discontinuity between lo and hi (which must
    straddle it: collapsed on one side, healthy G > 0.5 on the other).
    Returns None if the pair turns out not to straddle."""
    def collapsed(x):
        return final_state(p, axis, x)["G_end"] < COLLAPSED

    if collapsed(lo) == collapsed(hi):
        return None
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        if collapsed(mid) == collapsed(lo):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def find_threshold(p: Params, axis: str, vals: np.ndarray, G_end: np.ndarray):
    """Locate the saddle-node boundary: the first adjacent pair with a final-G
    jump > 0.4, bisected to 1e-3.  Returns (threshold, healthy_side) where
    healthy_side is 'below' or 'above' (or (None, None) if no jump)."""
    for i in range(len(vals) - 1):
        if abs(G_end[i + 1] - G_end[i]) > 0.4:
            lo, hi = float(vals[i]), float(vals[i + 1])
            thr = bisect_threshold(p, axis, lo, hi)
            if thr is not None:
                side = "below" if G_end[i] > 0.5 else "above"
                return thr, side
    return None, None


def phase1(p: Params) -> dict:
    curves, thresholds = {}, {}
    for axis, (vals, _kind) in AXES.items():
        rows = [final_state(p, axis, float(v)) for v in vals]
        curves[axis] = dict(
            vals=vals.astype(float),
            G_end=np.array([r["G_end"] for r in rows]),
            S_end=np.array([r["S_end"] for r in rows]),
            T_stuck=np.array([r["T_stuck"] for r in rows]),
            regime=np.array([r["regime"] for r in rows]),
        )
        thr, side = find_threshold(p, axis, vals, curves[axis]["G_end"])
        thresholds[axis] = None if thr is None else (thr, side)
    return dict(curves=curves, thresholds=thresholds)


def phase2(p: Params):
    """(alpha_G x episode duration) regime map."""
    alphas = np.round(np.linspace(0.4, 1.6, 24), 4)
    durs = np.array([5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 150, 200],
                    float)
    grid = np.zeros((len(durs), len(alphas)), int)      # imshow row=y=durs
    Gmap = np.zeros_like(grid, float)
    from dataclasses import replace
    for j, ag in enumerate(alphas):
        for i, d in enumerate(durs):
            r = one_run(replace(p, alpha_G=float(ag)),
                        episode_schedule(t1=EP_T0 + d), T_1D)
            grid[i, j] = r["regime"]
            Gmap[i, j] = r["G_end"]
    return alphas, durs, grid, Gmap


REGIME_NAMES = {0: "healthy(baseline)", 1: "stuck(depersonalized)",
                2: "escaped(recovering)", 3: "relapsed"}


def main() -> int:
    p = Params()
    os.makedirs("cache", exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)
    print("== exp1 Phase 1: 1-D bifurcation curves ==")
    ph1 = phase1(p)
    for axis, c in ph1["curves"].items():
        thr = ph1["thresholds"][axis]
        print(f"  {axis:12s} G_end range [{c['G_end'].min():.3f},"
              f" {c['G_end'].max():.3f}]  threshold="
              f"{('%.4f (healthy %s)' % thr) if thr else 'none (graded/no sharp jump)'}")

    # f05: multi-panel bifurcation (one panel per axis, shared log-y)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (axis, c) in zip(axs.flat, ph1["curves"].items()):
        ax.plot(c["vals"], c["G_end"], "o-", ms=3)
        thr = ph1["thresholds"][axis]
        if thr is not None:
            ax.axvline(thr[0], color="r", ls="--", lw=1,
                       label=f"thr={thr[0]:.3f}")
            ax.legend(fontsize=7)
        ax.set_title(axis)
        ax.set_ylabel("G(T)")
        ax.axhline(COLLAPSED, color="gray", lw=0.5, ls=":")
    axs.flat[-1].axis("off")
    fig.suptitle("exp1 Phase 1: final G vs parameter (bifurcation curves)")
    fig.tight_layout()
    f05 = os.path.join(FIGDIR, "f05_bifurcation.png")
    fig.savefig(f05, dpi=110)
    plt.close(fig)
    print(f"  -> {f05}")

    print("== exp1 Phase 2: (alpha_G x duration) regime heatmap ==")
    alphas, durs, grid, Gmap = phase2(p)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid, origin="lower", aspect="auto", vmin=0, vmax=3,
                   extent=[alphas.min(), alphas.max(), durs.min(), durs.max()],
                   interpolation="nearest", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([REGIME_NAMES[i] for i in range(4)])
    ax.set_xlabel("alpha_G")
    ax.set_ylabel("episode duration (t.u.)")
    ax.set_title("exp1 Phase 2: regime map alpha_G x episode duration")
    # regime-countour at 0.5 between healthy(0) and stuck(1)
    cs = ax.contour(alphas, durs, (grid >= 1).astype(float), levels=[0.5],
                    colors="w", linewidths=1.5)
    ax.clabel(cs, fmt="healthy|stuck boundary", fontsize=7)
    fig.tight_layout()
    f04 = os.path.join(FIGDIR, "f04_sweep_heatmap.png")
    fig.savefig(f04, dpi=110)
    plt.close(fig)
    n = {REGIME_NAMES[k]: int((grid == k).sum()) for k in np.unique(grid)}
    print(f"  regime counts: {n}")
    print(f"  -> {f04}")

    np.savez(os.path.join("cache", "exp1_summary.npz"),
             thresholds=np.array([f"{k}: {('%0.4f' % v[0]) + ' (healthy ' + v[1] + ')' if v else 'none'}"
                                  for k, v in ph1["thresholds"].items()]),
             **{f"{ax}_vals": c["vals"] for ax, c in ph1["curves"].items()},
             **{f"{ax}_G_end": c["G_end"] for ax, c in ph1["curves"].items()},
             alphas=alphas, durs=durs, grid=grid, Gmap=Gmap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
