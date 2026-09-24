"""exp7 part (d) — long horizon, 4 arms x 2 patterns, per-episode dip margins
(not binary counts).  Standalone: `.venv/bin/python -m experiments.exp7d`.
T_set is read from cache/exp7_partb.npz (or passed via --T-set).  Saves
cache/exp7_partd.npz."""
from __future__ import annotations

import os
import sys

import numpy as np

from experiments._exp7_common import CACHE, STUCK_G, log, part_d

if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    T_set = None
    for a in sys.argv[1:]:
        if a.startswith("--T-set="):
            T_set = float(a.split("=", 1)[1])
    if T_set is None:
        b = dict(np.load(os.path.join(CACHE, "exp7_partb.npz")))
        T_set = float(b["T_set"])
    log(f"part_d: using T_set = {T_set:.0f}")
    pd_ = part_d(T_set)
    print("\n== exp7 part (d) RESULTS — LONG HORIZON, 4 arms ==")
    for label in pd_["patterns"]:
        print(f"  --- {label} ---")
        for name in ("unregulated", "floor-only", "window-only",
                     "floor+window"):
            r = pd_[label][name]
            print(f"    {name:13s}: G_min={r['G_min']:.4f} "
                  f"dip_min={r['dip_min']:.4f} dip_med={r['dip_med']:.4f} "
                  f"G_end={r['G_end']:.4f} frac<0.5={r['frac_below_05']:.3f}"
                  + (f" T_max={r['T_max']:.0f}"
                     if name in ("window-only", "floor+window") else ""))
        fw, fl = pd_[label]["floor+window"], pd_[label]["floor-only"]
        dd = fw["dips"] - fl["dips"]
        print(f"    floor+window vs floor-only dip margin: min "
              f"{dd.min():+.4f}, median {np.median(dd):+.4f}, max "
              f"{dd.max():+.4f}  -> window-targeting "
              f"{'BEATS' if np.median(dd) > 0 else 'DOES NOT BEAT'} "
              "floor-only"
              + (" everywhere (PER-DIP metric only; between episodes "
                 "floor+window sits BELOW floor-only — see the "
                 "inter-episode deficit printed below)" if dd.min() > 0
                 else " (not everywhere)"))
        fw, fl = pd_[label]["floor+window"], pd_[label]["floor-only"]
        m_int = fw["t"] > 500.0
        d_int = (np.interp(fl["t"][m_int], fw["t"], fw["G"])
                 - fl["G"][m_int])
        print(f"    inter-episode standing cost (t>500, G(floor+window)"
              f" - G(floor-only)): mean {d_int.mean():+.4f}, "
              f"min {d_int.min():+.4f}  -> the dip gain is bought at a "
              f"standing level cost ~{abs(d_int.mean() / max(np.median(dd), 1e-9)):.0f}x "
              "its size")
        print(f"    vs frozen stuck level G* = {STUCK_G:.4f}: every "
              f"floored dip clears it by >= "
              f"{min(fw['dip_min'], fl['dip_min']) - STUCK_G:.4f}")
        print(f"    dt 0.25 vs 0.5 check (floor+window): max|dG| = "
              f"{pd_[label]['dt_check_maxdG']:.2e}")
    # flatten for cache
    lh = {"T_set": T_set, "patterns": np.array(pd_["patterns"])}
    for label in pd_["patterns"]:
        for name in ("unregulated", "floor-only", "window-only",
                     "floor+window"):
            r = pd_[label][name]
            for k in ("G_min", "G_end", "dip_min", "dip_med",
                      "frac_below_05", "T_max", "T_mean_above_10", "stuck"):
                lh[f"{label[:12]}|{name}|{k}"] = r[k]
            lh[f"{label[:12]}|{name}|dips"] = r["dips"]
            lh[f"{label[:12]}|{name}|t"] = r["t"][::20]
            lh[f"{label[:12]}|{name}|G"] = r["G"][::20]
            lh[f"{label[:12]}|{name}|Tcap"] = r["Tcap"][::20]
        lh[f"{label[:12]}|dt_check"] = pd_[label]["dt_check_maxdG"]
    np.savez(os.path.join(CACHE, "exp7_partd.npz"), **lh)
    log("part_d cache written")
