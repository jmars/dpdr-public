"""exp7 part (c) — CONJECTURE TEST: canonical-episode dip-shallowing fine grid
T=0..120 PLUS sustained-stress (dur=300) tolerance bisect.  Standalone:
`.venv/bin/python -m experiments.exp7c`.  Saves cache/exp7_partc.npz."""
from __future__ import annotations

import os

import numpy as np

from experiments._exp7_common import CACHE, log, part_c

if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    pc = part_c()
    print("\n== exp7 part (c) RESULTS — CONJECTURE TEST ==")
    print("  canonical-episode dip-shallowing (floor on, T=0..120):")
    print("           T:   " + " ".join(f"{T:6.0f}" for T in pc["fineT"]))
    print("    dip G_min:    " + " ".join(f"{d:6.4f}" for d in pc["dips"]))
    print("    dG_min:       " + " ".join(f"{d:+6.4f}" for d in pc["dGmins"]))
    print("    M_max:        " + " ".join(f"{m:6.3f}" for m in pc["Ms"]))
    print(f"  -> canonical-episode benefit: {pc['verdict']}"
          f"  (monotone for T>=50: {pc['monotone_hi']};"
          f" max low-T dip deepening = {pc['dip_deepen_lowT']:+.4f} G-units)")
    print("  SUSTAINED-stress tolerance (dur=300, dip>0.1 bar):")
    print("           T:   " + " ".join(f"{T:6.0f}" for T in pc["susT"]))
    print("    a_hold_crit:  " + " ".join(f"{a:6.3f}"
                                          for a in pc["a_hold_crit_sustained"]))
    print("    -> NEGATIVE axis: on sustained stress the standing cost "
          "DOMINATES\n       the certified benefit and capacity REDUCES "
          "tolerance monotonically (graded; NB the benefit term is "
          "monotone in T BY CONSTRUCTION, so this assay cannot exhibit "
          "a lower threshold either way — untestable, not falsifying)")
    np.savez(os.path.join(CACHE, "exp7_partc.npz"),
             fineT=pc["fineT"], dips=pc["dips"], dGmins=pc["dGmins"],
             Ms=pc["Ms"], susT=pc["susT"],
             a_hold_crit_sustained=pc["a_hold_crit_sustained"],
             monotone_hi=np.array(pc["monotone_hi"]),
             dip_deepen_lowT=np.array(pc["dip_deepen_lowT"]))
    log("part_c cache written")
