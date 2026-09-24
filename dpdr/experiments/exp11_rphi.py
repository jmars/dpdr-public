"""exp11 — the capability/recoverability tradeoff, made reproducible.

The r*phi ~ 0.2 boundary of [TR-16] (handoff-selfreg-capability, the
capability experiment) was originally measured with an ad-hoc patched-RHS
script and recorded only in the memory record, which violated the paper's
own traceability rule (paper.md section 3.2: every section-4 number is
printed by a committed driver or loadable from a cache).  This driver
reproduces that boundary from the committed, frozen artifacts only:

  * the SYSTEM is the frozen model wrapped by dpdr.regulator (the opt-in
    regulator of exp6), not a re-written RHS: the protective floor is the
    regulator's floor (mode 'theta', floor = 0.6, engaged post-collapse at
    t_engage = 600 exactly as in exp6's settled-stuck assay), and the
    cost term r*phi is the regulator's UNGATED monitoring cost c_mon
    (mon_gated = False) — an always-on inward-drive addend in the same
    slot of da/dt that a_hold occupies in the frozen model.  This is the
    same functional form the ad-hoc patch used, built from committed
    parts: patched inward drive = a_hold + r*phi, switch floored at 0.6.
  * the PROTOCOL is exp6's post-collapse assay: the canonical failure
    schedule (a_hold 0.9 on [100, 200] + affect pulse 0.5 to t = 160)
    collapses the frozen system; the regulator engages at t = 600 on the
    settled stuck state (G = 0.049); escape is functional recovery,
    G_end > 0.5 at the T = 1500 horizon.

Recorded boundary [TR-16]: r = 0.0/0.1 escape at all phi; r = 0.2 escapes
phi <= 0.5 and fails phi = 1.0; r = 0.4 escapes phi <= 0.25 and fails
phi >= 0.5; r = 0.8 and r = 1.6 escape only at phi = 0 — a rectangular
hyperbola r*phi ~ 0.2 in (r, phi).

Verification carried by this driver: the printed escape grid must match
the recorded boundary cell-for-cell before the paper cites it.

Outputs: cache/exp11_rphi.npz (escape grid + bisected phi_crit(r)).
Reproducible: `.venv/bin/python -m experiments.exp11_rphi`.
"""
from __future__ import annotations

import os
import time

import numpy as np

from dpdr.events import failure_schedule
from dpdr.model import Params
from dpdr.regulator import RegulatorParams, simulate_reg

CACHEDIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "cache")

P = Params()
FAIL = failure_schedule()          # canonical episode: a_hold 0.9 + affect 0.5
T_ENGAGE = 600.0                   # exp6's post-collapse assay geometry
T_ASSAY = 1500.0
FLOOR = 0.6                        # [TR-16]: protective floor fixed at 0.6

# the recorded [TR-16] axes and expected outcomes.  [TR-16]'s phi grid is
# {0, 0.25, 0.5, 1.0}; phi = 0.1 is added here to bracket the bisection and
# is EXCLUDED from the match check (unrecorded).
RS = np.array([0.0, 0.1, 0.2, 0.4, 0.8, 1.6])
PHIS = np.array([0.0, 0.1, 0.25, 0.5, 1.0])
RECORDED = np.array([True, False, True, True, True])   # cells [TR-16] measured
EXPECTED = np.array([
    # phi:      0.0   0.1   0.25  0.5   1.0     r:
    [1, 1, 1, 1, 1],     # 0.0  — no cost, floor escapes
    [1, 1, 1, 1, 1],     # 0.1
    [1, 1, 1, 1, 0],     # 0.2  — escapes phi<=0.5, fails phi=1.0
    [1, 1, 1, 0, 0],     # 0.4  — escapes phi<=0.25, fails phi>=0.5
    [1, 0, 0, 0, 0],     # 0.8  — escapes phi=0 only
    [1, 0, 0, 0, 0],     # 1.6  — escapes phi=0 only
], dtype=bool)


def G_end(r: float, phi: float) -> float:
    """Escape assay at (r, phi): floor 0.6 + ungated cost r*phi, engaged
    post-collapse at t = 600; G at the T = 1500 horizon."""
    reg = RegulatorParams(floor=FLOOR, c_mon=r * phi, mon_gated=False,
                          t_engage=T_ENGAGE)
    return float(simulate_reg(P, reg, FAIL, T_ASSAY)["G"][-1])


def escaped(r: float, phi: float) -> bool:
    return G_end(r, phi) > 0.5


def main() -> None:
    t0 = time.time()
    # integrity of the assay geometry itself: with no cost the floor must
    # rescue the settled stuck state (the exp6 'any floor' result)
    g_floor_only = G_end(0.0, 0.0)
    g_frozen = float(simulate_reg(
        P, RegulatorParams(floor=None, t_engage=T_ENGAGE), FAIL,
        T_ASSAY)["G"][-1])
    print(f"assay geometry: frozen G_end = {g_frozen:.4f} (stuck), "
          f"floor 0.6 alone G_end = {g_floor_only:.4f} (escape)")

    grid = np.zeros((len(RS), len(PHIS)))
    escape = np.zeros_like(grid, dtype=bool)
    for i, r in enumerate(RS):
        for j, phi in enumerate(PHIS):
            grid[i, j] = G_end(r, phi)
            escape[i, j] = grid[i, j] > 0.5
        print(f"  r = {r:4.1f}  G_end = "
              + " ".join(f"{g:6.4f}" for g in grid[i])
              + ("   ESCAPE" if escape[i, 0] else ""))

    match = bool(np.array_equal(escape[:, RECORDED],
                                EXPECTED[:, RECORDED]))
    print(f"escape grid matches the recorded [TR-16] boundary "
          f"(on the {RECORDED.sum()} recorded cells): {match}")

    # bisect the phi-axis boundary phi_crit(r) on [0, 1] (12 halvings,
    # resolution ~2.4e-4), for the r values where a boundary exists
    phi_crit = {}
    for i, r in enumerate(RS):
        if r == 0.0:
            continue                     # escapes everywhere: no boundary
        if not escape[i, 0]:
            continue                     # nowhere escapes: not the regime
        lo, hi = 0.0, 1.0                # escape(lo), fail(hi) established
        if escape[i, len(PHIS) - 1]:
            continue                     # escapes at phi = 1: no boundary
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if escaped(r, mid):
                lo = mid
            else:
                hi = mid
        phi_crit[r] = 0.5 * (lo + hi)
        print(f"  phi_crit(r = {r:g}) = {phi_crit[r]:.4f}   "
              f"(r*phi_crit = {r * phi_crit[r]:.4f})")

    os.makedirs(CACHEDIR, exist_ok=True)
    np.savez(os.path.join(CACHEDIR, "exp11_rphi.npz"),
             r=RS, phi=PHIS, G_end=grid, escape=escape,
             phi_crit_r=np.array(list(phi_crit.keys()), float),
             phi_crit=np.array(list(phi_crit.values()), float),
             frozen_G_end=g_frozen, floor_only_G_end=g_floor_only,
             matches_record=bool(match))
    print(f"cached -> cache/exp11_rphi.npz   ({time.time() - t0:.1f}s)")
    if not match:
        raise SystemExit("REPRODUCED GRID DOES NOT MATCH [TR-16] — do not "
                         "cite; investigate before any paper edit.")


if __name__ == "__main__":
    main()
