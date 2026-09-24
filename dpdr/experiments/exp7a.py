"""exp7 part (a) — healthy phase diagram T x c_cap + canonical dip probe.
Run standalone: `.venv/bin/python -m experiments.exp7a`.
Reuses cache/exp7_phase.npz when present and complete (the prior run's part (a)
finished); pass --force to recompute.  Saves cache/exp7_phase.npz."""
from __future__ import annotations

import os
import sys

import numpy as np

from experiments._exp7_common import CACHE, log, part_a

PATH = os.path.join(CACHE, "exp7_phase.npz")


def _complete(d: dict) -> bool:
    return (d.get("healthy") is not None
            and d["healthy"].shape == (3, 6)
            and d.get("dips") is not None
            and d["dips"].shape == (5,))


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(PATH) and "--force" not in sys.argv:
        d = dict(np.load(PATH))
        if _complete(d):
            log(f"part_a: cache {PATH} already complete; reusing "
                "(pass --force to recompute)")
            for k, v in d.items():
                print(f"  {k} = {v}")
            raise SystemExit(0)
        log("part_a: cache present but incomplete; recomputing")
    pa = part_a()
    np.savez(PATH, **{k: v for k, v in pa.items()})
    print("\n== exp7 part (a) RESULTS ==")
    print("  healthy G_end (baseline, T=3000):")
    print("           T:   " + " ".join(f"{T:6.0f}" for T in pa["Ts"]))
    for i, cc in enumerate(pa["c_caps"]):
        print(f"    c_cap={cc:4.2f}: "
              + " ".join(f"{g:6.3f}" for g in pa["healthy"][i]))
    print("  c_int at c_cap=0.1: " + " ".join(f"{c:6.3f}" for c in pa["c_int"]))
    print("  canonical dip with floor (what capacity BUYS):")
    print("           T:   " + " ".join(f"{T:6.0f}" for T in pa["dipT"]))
    print("    dip G_min:    " + " ".join(f"{d:6.4f}" for d in pa["dips"]))
    print("    G_end:        " + " ".join(f"{d:6.4f}" for d in pa["dip_G_ends"]))
    log("part_a cache written")
