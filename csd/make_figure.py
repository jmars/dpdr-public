"""Figure f13: the CSD discriminator results (tests 1-4).

Four panels:
  (A) bifurcation diagram G*(eps): healthy (stable), middle (saddle),
      collapsed branches; the border-collision fold at eps_c; bistable
      region shaded.
  (B) the discriminator: exact eigenvalues vs distance-to-fold (log-log),
      with the saddle-node sqrt law for contrast.
  (C) protocol measurements: recovery t95 and fitted decay rate vs
      distance; fixed-delta=0.05 outcome (recover vs collapse).
  (D) the warning signal the model DOES have: healthy basin half-width
      vs distance-to-fold (closes linearly).

Run (from this directory, with the requirements installed):
    python make_figure.py
Output: figs/f13_csd.png (verified programmatically; do not open with
image viewers inside the agent loop).
"""
from __future__ import annotations

import os
import struct
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import F, Params, equilibrium, eig, jac  # noqa: E402

P = Params()
EPS_C = 0.265192279
OUTDIR = os.path.join(HERE, "figs")
os.makedirs(OUTDIR, exist_ok=True)

d1 = np.load(os.path.join(HERE, "cache", "test1_fold.npz"))
d2 = np.load(os.path.join(HERE, "cache", "test2_discriminator.npz"))

# ---- branch data for panel A (fresh dense continuation, cached npz is coarse)
def dense_branches():
    yh = equilibrium(0.10, np.array([0.05, 0.7, 0.5, 0.8, 0.5]))
    ys = None
    for G0 in np.linspace(0.15, 0.33, 37):
        cand = equilibrium(0.10, np.array([0.30, G0, 0.5455, 0.40, 0.5]))
        if cand is not None and np.max(eig(cand, 0.10).real) > 0:
            ys = cand
            break
    H = [yh.copy()]
    M = [ys.copy()]
    eps = 0.10
    while eps < 0.265:
        eps = min(eps + 2e-3, 0.265)
        hn = equilibrium(eps, H[-1])
        if hn is not None:
            H.append(hn)
        sn = equilibrium(eps, M[-1])
        if sn is not None and np.max(eig(sn, eps).real) > 0:
            M.append(sn)
        else:
            break
    # also down to eps=0
    yhd = H[0]
    epsd = 0.10
    Hlow = [yhd.copy()]
    while epsd > 0.0:
        epsd = max(epsd - 2e-3, 0.0)
        hn = equilibrium(epsd, Hlow[-1])
        if hn is not None:
            Hlow.append(hn)
    H = Hlow[::-1] + H[1:]
    # collapsed branch
    yc = equilibrium(0.10, np.array([0.6, 0.05, 0.5, 0.4, 0.5]))
    C = [yc.copy()]
    epsc = 0.10
    while epsc < 0.40:
        epsc = min(epsc + 4e-3, 0.40)
        cn = equilibrium(epsc, C[-1])
        if cn is not None:
            C.append(cn)
    return (np.array(H), np.array(M), np.array(C))


H, M, C = dense_branches()
eps_H = np.linspace(0.0, 0.265, len(H))
eps_M = np.linspace(0.10, eps_H[-1], len(M))
eps_C = np.linspace(0.10, 0.40, len(C))

fig = plt.figure(figsize=(13.5, 10))
gs = fig.add_gridspec(2, 2)

# ---------------------------------------------------------------- (A)
axA = fig.add_subplot(gs[0, 0])
axA.axvspan(0.0, EPS_C, color="tab:green", alpha=0.06,
            label="bistable (healthy + collapsed)")
axA.plot(eps_H, H[:, 1], "-", color="tab:blue", lw=1.8,
         label="healthy (stable)")
axA.plot(eps_M, M[:, 1], "--", color="tab:red", lw=1.4,
         label="middle (saddle, unstable)")
axA.plot(eps_C, C[:, 1], "-", color="k", lw=1.4,
         label="collapsed (stable, exists at ALL eps ≥ 0)")
axA.axvline(EPS_C, color="tab:purple", lw=1.2, ls=":")
axA.annotate(f"border-collision fold\neps_c = {EPS_C:.6f}\n"
             "(on switching manifold\nE = Theta_eff;\nno eigenvalue → 0)",
             xy=(EPS_C, 0.2657), xytext=(0.16, 0.13), fontsize=8.5,
             arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.0),
             bbox=dict(boxstyle="round", fc="white", ec="tab:purple",
                       alpha=0.9))
axA.set_xlabel("chronic drive eps (a_hold)")
axA.set_ylabel("G* at equilibrium")
axA.set_title("(A) TEST 1 fold structure: healthy+saddle annihilate AT the\n"
              "cannibalization knee; collapsed branch never folds (no lower\n"
              "fold: collapse irreversible by drive withdrawal)")
axA.legend(fontsize=8, loc="center left")
axA.set_xlim(-0.005, 0.40)
axA.set_ylim(0.0, 0.75)

# ---------------------------------------------------------------- (B)
axB = fig.add_subplot(gs[0, 1])
dist = d2["dist"]
for key, lbl, col, mk in (("lam_g", "g-loop (slowest)", "tab:blue", "o"),
                          ("lam_S", "S (setpoint)", "tab:cyan", "s"),
                          ("lam_D", "D (demand)", "tab:green", "^"),
                          ("lam_Gmode", "G (generator) — the fold's mode",
                           "tab:red", "D"),
                          ("lam_fast", "a (attention, fast)", "gray", "v")):
    axB.loglog(dist, np.abs(d2[key]), mk + "-", color=col, ms=4, lw=1.1,
               label=lbl)
ref = np.array([dist[3], dist[-2]])
axB.loglog(ref, np.abs(d2["lam_Gmode"][3]) * np.sqrt(ref / ref[0]), ":",
           color="k", lw=1.4)
axB.annotate("saddle-node sqrt law\n(slope 1/2) — NOT followed",
             xy=(ref[1], np.abs(d2["lam_Gmode"][3]) * np.sqrt(ref[1] / ref[0])),
             xytext=(2e-4, 1e-1), fontsize=8,
             arrowprops=dict(arrowstyle="->", lw=0.8))
axB.set_xlabel("distance to fold: eps_c − eps")
axB.set_ylabel("|lambda| (exact Jacobian eigenvalue)")
axB.invert_xaxis()
axB.set_title("(B) TESTS 2-4 discriminator: slowest mode EXACTLY constant\n"
              "(−μ/τ_g = 0.0015, spread 0.0e+00); the G-mode is flat near\n"
              "the fold (0.0429 → 0.0427 over 3.4 decades; slope +0.0006)")
axB.legend(fontsize=7.5, loc="lower left")

# ---------------------------------------------------------------- (C)
axC = fig.add_subplot(gs[1, 0])
axC.semilogx(dist, d2["t95"], "o-", color="tab:blue", lw=1.4,
             label="t95 recovery (basin-scaled probe)")
axC.set_xlabel("distance to fold (log scale)")
axC.set_ylabel("t95 (t.u.)", color="tab:blue")
axC.invert_xaxis()
axC2 = axC.twinx()
axC2.semilogx(dist, d2["lam_fit"], "s--", color="tab:red", lw=1.2,
              label="fitted decay rate (50%→5%)")
axC2.set_ylabel("fitted decay rate", color="tab:red")
axC.axhspan(65, 71, color="tab:blue", alpha=0.08)
axC.annotate("t95 = 65–71 t.u. at EVERY distance\n(no slowing; 667 t.u. "
             "tail constant set by the g-loop)",
             xy=(1e-3, 70), fontsize=8.5,
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
# fixed-delta outcomes
recov = np.array([True, True, True, False, False, False, False, False,
                  False, False, False, False])
first_c = dist[~recov][0]
axC.axvline(first_c, color="k", ls=":", lw=1.2)
axC.annotate(f"fixed δ=0.05 probe starts COLLAPSING\nhere (dist "
             f"{first_c:.1e}): basin retreat,\nnot slowing, is what fails",
             xy=(first_c, 50), xytext=(3e-5, 40), fontsize=8,
             arrowprops=dict(arrowstyle="->", lw=0.8),
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
axC.set_title("(C) TEST 2 perturbation-recovery protocol: recovery time\n"
              "flat to within ±5% over 5 decades of distance")
h1, l1 = axC.get_legend_handles_labels()
h2, l2 = axC2.get_legend_handles_labels()
axC.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

# ---------------------------------------------------------------- (D)
axD = fig.add_subplot(gs[1, 1])
basin = d2["basin"]
ok = np.isfinite(basin)
axD.loglog(dist[ok], basin[ok], "o-", color="tab:purple", lw=1.4,
           label="healthy basin half-width (G* − G_saddle)")
axD.loglog(dist[ok], dist[ok], ":", color="k", lw=1.2,
           label="y = distance-to-fold")
axD.set_xlabel("distance to fold (log scale)")
axD.set_ylabel("basin half-width in G")
axD.invert_xaxis()
sl = np.polyfit(np.log(dist[ok][-6:]), np.log(basin[ok][-6:]), 1)[0]
axD.annotate(f"basin closes LINEARLY in distance\n(log-log slope {sl:.3f} "
             f"near the fold):\nthe model's real warning signal is\n"
             "BASIN RETREAT, not critical slowing",
             xy=(dist[ok][-2], basin[ok][-2]), xytext=(1e-3, 3e-2),
             fontsize=8.5, arrowprops=dict(arrowstyle="->", lw=0.8),
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
axD.legend(fontsize=8, loc="upper left")
axD.set_title("(D) what replaces CSD here: the stable basin shrinks in\n"
              "proportion to the distance to the fold (slope ≈ 1), so a\n"
              "fixed-size kick fails at a sharp, predictable threshold")

fig.suptitle("CSD discriminator on the frozen dpdr model: NO critical "
             "slowing down — constant recovery (667 t.u.), a "
             "border-collision (non-smooth) fold at eps_c = 0.265192, and "
             "linear basin retreat instead of critical slowing",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.965))
out = os.path.join(OUTDIR, "f13_csd.png")
fig.savefig(out, dpi=115)
plt.close(fig)


# ------------------------------------------------- programmatic verification
def png_size(path):
    with open(path, "rb") as f:
        head = f.read(33)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    w, h = struct.unpack(">II", head[16:24])
    return w, h


w, h = png_size(out)
size = os.path.getsize(out)
print(f"figure written: {out}")
print(f"  PNG {w}x{h} px, {size} bytes")
assert w > 1000 and h > 700 and size > 50000, "figure looks wrong"
# spot-check the numbers baked into the annotations against the cache
assert abs(float(d1["eps_death"]) - EPS_C) < 1e-9
assert float(np.max(np.abs(d2["lam_g"] - d2["lam_g"][0]))) == 0.0, \
    "slowest mode not constant"
print("  verification: eps_death matches test1; lam_g spread is exactly 0; "
      "PNG header OK")
