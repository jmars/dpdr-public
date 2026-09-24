"""exp7 part 0 — integrity (fidelity/gates/attractor-exactness/stuck-survival/
rescue-not-blocked).  Run standalone: `.venv/bin/python -m experiments.exp70`.
Saves cache/exp7_part0.npz and prints per-evaluation progress."""
from __future__ import annotations

import os

import numpy as np

from experiments._exp7_common import CACHE, log, part0

if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    p0 = part0()
    print("\n== exp7 part 0 RESULTS ==")
    print(f"  frozen gates: {p0['gates']} -> "
          f"{'ALL PASS' if all(p0['gates'].values()) else 'FAIL'}")
    for k in ("fidelity_baseline", "fidelity_failure", "fidelity_rescue"):
        print(f"  {k:22s}: max|dG| = {p0[k]:.2e}")
    print(f"  attractor |dE| healthy = {p0['attractor_dE_healthy']:+.2e}, "
          f"stuck = {p0['attractor_dE_stuck']:+.2e}")
    print(f"  stuck-survival (T=200 on, no floor): G_end = "
          f"{p0['stuck_survival_G_end']:.4f}, is_stuck = "
          f"{p0['stuck_survival_is_stuck']}")
    print(f"  floor escape WITH T on: G_end = {p0['stuck_escape_with_T_G_end']:.4f}")
    print(f"  rescue with T on: G_end = {p0['rescue_with_T_G_end']:.4f}")
    np.savez(os.path.join(CACHE, "exp7_part0.npz"), **p0)
    log("part0 cache written")
