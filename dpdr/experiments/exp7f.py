"""exp7 figure assembly — builds figs/f12_window.png from the per-part caches.
Run standalone: `.venv/bin/python -m experiments.exp7f`.  Reads
cache/exp7_phase.npz, exp7_partb.npz, exp7_partc.npz, exp7_partd.npz,
exp7_parte.npz (all produced by the exp70/exp7a-e scripts)."""
from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments._exp7_common import (CACHE, C_CAPS, EP_T0, FLOOR, FIGDIR,
                                      P, STUCK_G, episode1)
from dpdr.window import WindowParams, simulate_window

P_B = dict(np.load(os.path.join(CACHE, "exp7_partb.npz")))
T_SET = float(P_B["T_set"])


def _canvas():
    fig = plt.figure(figsize=(14, 9))
    return fig, fig.add_gridspec(2, 3)


def main() -> int:
    os.makedirs(FIGDIR, exist_ok=True)
    pa = dict(np.load(os.path.join(CACHE, "exp7_phase.npz")))
    pb = dict(np.load(os.path.join(CACHE, "exp7_partb.npz")))
    pc = dict(np.load(os.path.join(CACHE, "exp7_partc.npz")))
    pd_ = dict(np.load(os.path.join(CACHE, "exp7_partd.npz"), allow_pickle=True))
    pe = dict(np.load(os.path.join(CACHE, "exp7_parte.npz")))

    fig, gs = _canvas()

    # --- (a) phase diagram
    axA = fig.add_subplot(gs[0, 0])
    Ts = pa["Ts"]
    Ce = np.concatenate(([C_CAPS[0] - 0.025],
                         0.5 * (np.array(C_CAPS[:-1]) + np.array(C_CAPS[1:])),
                         [C_CAPS[-1] + 0.05]))
    Te = np.concatenate(([Ts[0] - 25.0], 0.5 * (Ts[:-1] + Ts[1:]),
                         [Ts[-1] + 100.0]))
    Zm = np.ma.masked_where(pa["healthy"] < 0.10, pa["healthy"])
    pcA = axA.pcolormesh(Te, Ce, Zm, cmap="RdYlGn", vmin=0.0, vmax=0.9,
                         shading="flat")
    plt.colorbar(pcA, ax=axA, label="healthy G_end")
    for i, cc in enumerate(C_CAPS):
        tm = pb["T_max"][i]
        if not np.isnan(tm):
            axA.plot(tm, cc, "v", color="k", ms=7)
            axA.annotate(f"T_max={tm:.0f}", (tm, cc), fontsize=6,
                         xytext=(3, -9), textcoords="offset points")
    axA.set_xlabel("capacity T (self-simulation horizon)")
    axA.set_ylabel("c_cap")
    axA.set_title("(a) phase diagram: healthy G_end(T, c_cap)\n"
                  "black = collapsed (<0.1); v = bisected T_max")

    # --- (b) trade on one axis
    axB = fig.add_subplot(gs[0, 1])
    axB.plot(Ts, pa["healthy"][1], "o-", label="healthy G_end (c_cap=0.1)")
    axB.axhline(0.5, color="k", ls=":", lw=0.6)
    tm = pb["T_max"][1]
    axB.axvline(tm, color="tab:red", ls="--", lw=1,
                label=f"T_max={tm:.0f} (c_int={pb['T_max_c_int'][1]:.3f})")
    axBb = axB.twinx()
    axBb.plot(pa["dipT"], pa["dips"], "^-", color="tab:green",
              label="canonical dip G_min (floor)")
    axB.set_xlabel("capacity T")
    axB.set_ylabel("healthy G_end")
    axBb.set_ylabel("dip G_min", color="tab:green")
    axB.set_title("(b) the trade on ONE axis: dip shallows,\n"
                  "healthy level degrades (both vs T)")
    h1, l1 = axB.get_legend_handles_labels()
    h2, l2 = axBb.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, fontsize=7)

    # --- (c) conjecture test
    axC = fig.add_subplot(gs[0, 2])
    axC.plot(pc["fineT"], pc["dGmins"], "o-", ms=3, color="tab:blue",
             label="dip shallowing dG_min(T)  [G units]")
    axC.plot(pc["fineT"], pc["Ms"] * 0.1, "s--", ms=3, color="tab:orange",
             label="certified M_max x0.1")
    axC2 = axC.twinx()
    axC2.plot(pc["susT"], pc["a_hold_crit_sustained"], "^-", ms=4,
              color="tab:red", label="sustained-stress a_hold_crit(T)")
    axC.set_xlabel("capacity T (fine low-T grid)")
    axC.set_ylabel("dip shallowing [G units]")
    axC2.set_ylabel("a_hold_crit (sustained)", color="tab:red")
    verdict = ("GRADED onset (no lower threshold; UNTESTABLE — benefit "
               "monotone in T by construction)" if bool(pc["monotone_hi"])
               else "NON-MONOTONE")
    axC.set_title(f"(c) CONJECTURE TEST: {verdict}\n"
                  "red: sustained stress — capacity LOWERS tolerance")
    h1, l1 = axC.get_legend_handles_labels()
    h2, l2 = axC2.get_legend_handles_labels()
    axC.legend(h1 + h2, l1 + l2, fontsize=6)

    # --- (d) long horizon, 4 arms (canonical)
    axD = fig.add_subplot(gs[1, 0])
    pat = pd_["patterns"][0]
    cols = {"unregulated": "k", "floor-only": "tab:green",
            "window-only": "tab:blue", "floor+window": "tab:red"}
    for name in ("unregulated", "floor-only", "window-only", "floor+window"):
        key = f"{pat[:12]}|{name}"
        if key + "|t" not in pd_:
            continue
        t = pd_[key + "|t"]
        G = pd_[key + "|G"]
        axD.plot(t, G, color=cols[name], lw=0.9,
                 label=f"{name}: G_end={pd_[key + '|G_end']:.3f}")
    axD.axhline(0.1, color="k", ls=":", lw=0.6)
    axD.set_xlabel("t")
    axD.set_ylabel("G(t)")
    axD.set_title(f"(d) long horizon, 4 arms — {pat}\n"
                  f"(T_set={T_SET:.0f}; margins, not binary counts)")
    axD.legend(fontsize=7, loc="lower right")

    # --- (d') per-episode dip margins
    axE = fig.add_subplot(gs[1, 1])
    for i, pat in enumerate(pd_["patterns"]):
        fl = pd_[f"{pat[:12]}|floor-only|dips"]
        fw = pd_[f"{pat[:12]}|floor+window|dips"]
        axE.plot(np.arange(1, len(fl) + 1), fl, "-",
                 color="tab:green", lw=1, label=f"floor-only ({pat[:9]})")
        axE.plot(np.arange(1, len(fw) + 1), fw, "-",
                 color="tab:red", lw=1, label=f"floor+window ({pat[:9]})")
    axE.axhline(STUCK_G, color="k", ls=":", lw=0.8)
    axE.annotate("frozen stuck G*=0.0485", (2, STUCK_G + 0.004), fontsize=6)
    axE.set_xlabel("episode #")
    axE.set_ylabel("dip G_min per episode")
    axE.set_title("(d') MARGIN: per-episode dips, floor-only vs\n"
                  "floor+window (canonical + dense-weak)")
    axE.legend(fontsize=6)

    # --- (f) mechanism
    axF = fig.add_subplot(gs[1, 2])
    s = simulate_window(P, WindowParams(regulate=True, T_set=T_SET,
                                        floor=FLOOR), episode1(0.9),
                        600.0, dt=0.25)
    axF.plot(s["t"], s["G"], color="k", lw=1.2, label="G")
    axF.axhline(0.1, color="k", ls=":", lw=0.6)
    axF2 = axF.twinx()
    axF2.plot(s["t"], s["T"], color="tab:purple", lw=1.0, label="T(t)")
    axF2.plot(s["t"], s["M"], color="tab:orange", lw=0.8, label="M(t)")
    axF.set_xlabel("t")
    axF.set_ylabel("G")
    axF2.set_ylabel("T / M", color="tab:purple")
    axF.set_title("(f) mechanism: T rises with volatility (m2), the\n"
                  "certified M shallows the dip, T decays when quiet")
    h1, l1 = axF.get_legend_handles_labels()
    h2, l2 = axF2.get_legend_handles_labels()
    axF.legend(h1 + h2, l1 + l2, fontsize=7)

    fig.suptitle("exp7 — the capacity WINDOW: introspective horizon T buys "
                 "dip margin (graded onset; NO lower threshold COULD appear "
                 "— the benefit term is monotone in T by construction, so "
                 "the external lower-bound conjecture is UNTESTABLE here, "
                 "not refuted) and costs standing inward drive (upper edge "
                 "= 0.112 c-units, a consistency check of the designed "
                 "cost channel); deterministic, in-model only")
    fig.tight_layout()
    path = os.path.join(FIGDIR, "f12_window.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"figure -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
