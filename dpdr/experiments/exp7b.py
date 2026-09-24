"""exp7 part (b) — bisected T_max per c_cap + regulated-dip-margin T_set
calibration.  Run standalone: `.venv/bin/python -m experiments.exp7b`.
Tight bisection brackets are read from cache/exp7_phase.npz (deterministic);
saves cache/exp7_partb.npz.  This is the part that previously stalled (a
~36-sim silent batch); it now prints every evaluation with a timestamp."""
from __future__ import annotations

import os

import numpy as np

from experiments._exp7_common import CACHE, C_CAPS, log, part_b

if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    pb = part_b()
    print("\n== exp7 part (b) RESULTS ==")
    for cc in C_CAPS:
        tm = pb["T_max"][cc]
        ci = pb["T_max_c_int"][cc]
        print(f"  T_max @ c_cap={cc:4.2f}: "
              f"{'n/a' if tm is None else f'{tm:.2f} T-units'}"
              f" = {'n/a' if ci is None else f'{ci:.4f}'} c-int units "
              "(healthy G_end > 0.5; MEASURED upper edge)")
    print(f"  frozen chronic a_hold_crit (healthy, same criterion) = "
          f"{pb['a_hold_crit_healthy']:.4f}")
    print("  regulated (m2-gated) dip margin vs T_set, canonical ep, floor on:")
    print("           T_set:  " + " ".join(f"{T:6.0f}" for T in pb["scanT"]))
    print("    dip G_min:     " + " ".join(f"{d:6.4f}" for d in pb["dip_reg"]))
    print("    margin:        " + " ".join(f"{d:+6.4f}"
                                          for d in pb["margin_reg"]))
    print(f"  margin argmax T = {pb['T_argmax']:.0f}; T_set = {pb['T_set']:.0f}"
          f" (argmax clamped to T_max/2 = {0.5 * pb['T_max'][0.1]:.0f})")
    flat = {
        "scanT": pb["scanT"], "dip_reg": pb["dip_reg"],
        "margin_reg": pb["margin_reg"],
        "T_argmax": pb["T_argmax"], "T_set": pb["T_set"],
        "a_hold_crit_healthy": pb["a_hold_crit_healthy"],
        "T_max": np.array([pb["T_max"][c] for c in C_CAPS]),
        "T_max_c_int": np.array([pb["T_max_c_int"][c] for c in C_CAPS]),
        "c_caps": np.array(C_CAPS),
    }
    np.savez(os.path.join(CACHE, "exp7_partb.npz"), **flat)
    log("part_b cache written")
