"""exp7 part (e) — healthy-regime cost of the window regulator + the
c_int = a_hold equivalence check.  Standalone:
`.venv/bin/python -m experiments.exp7e`.  T_set read from cache/exp7_partb.npz
(or --T-set).  Saves cache/exp7_parte.npz."""
from __future__ import annotations

import os
import sys

import numpy as np

from experiments._exp7_common import CACHE, log, part_e

if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    T_set = None
    for a in sys.argv[1:]:
        if a.startswith("--T-set="):
            T_set = float(a.split("=", 1)[1])
    if T_set is None:
        b = dict(np.load(os.path.join(CACHE, "exp7_partb.npz")))
        T_set = float(b["T_set"])
    log(f"part_e: using T_set = {T_set:.0f}")
    pe = part_e(T_set)
    print("\n== exp7 part (e) RESULTS — healthy-regime cost ==")
    print(f"    frozen       : G_end = {pe['frozen_G_end']:.4f}")
    for name in ("floor-only", "window-only", "floor+window"):
        print(f"    {name:13s}: G_end = {pe[name + '_G_end']:.4f}  "
              f"max|dG| vs frozen = {pe[name + '_maxdG']:.2e}"
              + (f"  T_max = {pe[name + '_T_max']:.1f}, T_end = "
                 f"{pe[name + '_T_end']:.2f}" if name != "floor-only"
                 else ""))
    print(f"    equivalence: window T=400 (c_int=0.2) G_end = "
          f"{pe['equiv_window_G_end']:.4f} vs frozen a_hold=0.2 G_end = "
          f"{pe['equiv_frozen_G_end']:.4f}, max|dG| = {pe['equiv_maxdG']:.2e}"
          f"  -> c-int = a_hold on the healthy axis")
    np.savez(os.path.join(CACHE, "exp7_parte.npz"),
             T_set=np.array(T_set), **{k: np.array(v)
                                       for k, v in pe.items()})
    log("part_e cache written")
