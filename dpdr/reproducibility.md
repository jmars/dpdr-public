# Reproducibility file — every headline number's exact invocation

This file exists to fix one specific failure mode: **the record pinned
VALUES but not INVOCATIONS**, and an almost-right construction yields a
confident wrong number. Two numbers were flagged unreproducible during the
deposit audit (TR-14's open-loop G = 0.0121 and TR-31's dwell threshold
118.70); both reproduced exactly once the *original command* was recovered —
the first only after finding that the horizon was T = 900 (a T = 1000 rerun
gives a different fourth decimal and reads exactly like a discrepancy), the
second only after using the recall protocol rather than exp10 part B's
settled-ICs protocol. A row below therefore states the INVOCATION (module,
parameters, schedule/protocol, horizon T, criterion), not just the value and
a cache key. A row with only a value has failed at the thing this file
exists for.

The companion script `dpdr/reproduce.py` executes every row: Tier A
recomputes from the frozen model, Tier B reads the recorded cache with the
parameters echoed. Run:

```
cd dpdr && .venv/bin/python reproduce.py          # Tier A: 90 checks, ~66 s
.venv/bin/python reproduce.py --all               # + Tier B: 128 checks, ~66 s
```

Exit status 0 iff every executed check passes (usable as a gate). If the
paper's numbers can be re-derived by running one command, the appendix's
provenance apparatus ([TR-n], Appendix A) is supporting rather than
load-bearing.

**Coverage at a glance** (paper: 482 lines, sections 4.1–4.11, Table 1,
[TR-1]…[TR-36], 14 figures): 128 checks — Abstract's headline numbers;
P1–P6 verdict-bearing values; every results section 4.1–4.11 including the
new §4.11 (exp19/exp20, TR-35/TR-36, Table 1); the substrate legs (TR-10,
TR-11, TR-32); the CSD audit (TR-29); and the TR entries whose invocation is
non-obvious. The explicit remainder list is at the end.

---

## Tolerance discipline — read before quoting any check

There are four kinds of number in the paper, and **using the wrong tolerance
is how a TRUE number gets flagged as a mismatch**:

| class | meaning | check |
|---|---|---|
| EXACT / BIT-EXACT | identity or derived check (e.g. `max\|dG\| = 0.00e+00`, cache equality, grid counts) | equality |
| INTEGRATOR | trajectory quantity where adaptive-stepper divergence is expected | ~1e-3 (rtol 1e-6 integrator; none of the current rows needs looser than REPORTED, so 0 rows carry this class today — the class is retained because any future row comparing a raw trajectory at full double precision belongs in it) |
| GRID-QUANTISED | a **bisected threshold** on the dt = 0.05 grid | the transition lies in **(v − dt, v]** — the largest healthy cell is v − dt and the smallest collapsed cell is v — NOT "a run equals v" |
| REPORTED-PRECISION | the paper quotes k significant figures | match to that precision (4 s.f. ⇒ abs diff ≤ 5e-4 for O(1) quantities) |

Two conventions that have already bitten once each, recorded so they are not
re-bitten:

* **Bisected thresholds quote the grid cell, not the midpoint.** exp18's
  part-B bisection prints `midpoint 118.825` alongside
  `threshold = smallest stuck grid cell = 118.70`; **the paper uses 118.70**,
  and the dwell check below checks the bracket (118.65 healthy, 118.70
  stuck), never "equals 118.70". Likewise τ_S_crit = 141.47 is the reported
  *midpoint* 141.475 of the bracket (141.45 collapse / 141.50 healthy) —
  exp20's own cache records the midpoint convention for that one.
* **dt and T are part of every claim.** The default grid is dt = 0.05
  (breakpoints must lie on it); horizons vary per protocol (T = 900 for the
  recall dwell, 1400 for the rescued open loop, 1500 for assays, 1200 for
  exp20, 3000 at dt = 0.5 for the chronic-hold critical). The TR-14
  near-miss was exactly a T = 1000 reproduction against a T = 900 value.
  Each row below states dt and T.

**Recompute vs cache-only.** A RECOMPUTE row validates the pipeline (the
number regenerates from the shipped code); a CACHE row validates only that
the stored number equals the cited number. Both are worth having; they are
not the same claim and every row says which it is.

---

## Abstract and gates

| claim | exact invocation | expected | artifact | tol | mode |
|---|---|---|---|---|---|
| frozen gates G1–G2c pass | `dpdr.metrics.check_gates(Params())` | all True | tests/ | EXACT | recompute |
| healthy equilibrium G\* = 0.886 | `simulate(Params(), baseline_schedule(), T=5000)['G'][-1]`, dt 0.05 | 0.886 | §2.3 | REPORTED | recompute |
| canonical collapse G = 0.049 | `simulate(Params(), failure_schedule(), T=600)['G'][-1]` | 0.049 | §2.3 | REPORTED | recompute |
| floor critical 0.4795 | bisect `escaped(RegulatorParams(floor=v, t_engage=600), failure_schedule(), T=1500)` (criterion `G_end > 0.5`) on [0.40, 0.50], 12 it, dt 0.05 — exp6 part (b)'s bracket; midpoint convention | 0.4795 | cache/exp6_bisect.npz `floor_crit_theta` | REPORTED | recompute |
| collapsed state's own error E\* = 0.4969 | `simulate(Params(), failure_schedule(), T=1500)['E'][-1]` (settled stuck state) | 0.4969 | cache/exp6_bisect.npz `stuck_E` | REPORTED | recompute |
| c_mon_crit gated 0.511 (k_pull = 2) | bisect escape on [0.5, 0.7], `RegulatorParams(floor=None, k_pull=2, c_mon=v, mon_gated=True, t_engage=600)`, T=1500 — **floor=None explicit** (dataclass default is floor=0.7); escape falls with c_mon | 0.511 | cache/exp6_bisect.npz `cmon_crit_gated_k2` | REPORTED | recompute |
| c_mon_crit ungated 0.112 (k_pull = 2) | same construction, `mon_gated=False`, bracket [0.0, 1.0] | 0.112 | cache/exp6_bisect.npz `cmon_crit_ungated_k2` | REPORTED | recompute |
| capability hyperbola r·φ ≈ 0.1122 | `exp11_rphi.py`: grid r∈{0.1,…,1.6} × φ, floor 0.6 engaged t=600, cost = ungated c_mon = r·φ; boundary bisected per r; product `phi_crit × r` constant | 0.1122 (mean), spread ≤ 1e-3 | cache/exp11_rphi.npz | REPORTED | cache+recompute of product |
| Datalog formula 2^(n−1) | `datalog-leg.py` ladder fixpoint: derivations of (0,0)→(n,0); n = 2…14 → 2/8/…/8192 | 2^(n−1) | §4.6, [TR-32] | EXACT | recompute |

**The budget's three-way agreement on 0.112 is a consistency check, not a
unification** (all three are standing inward-drive addends in the same slot
of da/dt — §4.5, limitation 4). The rows above therefore verify each label
lands on the one number; they do not claim three independent measurements.

## §4.1 — P1–P6

| claim | exact invocation | expected | artifact | tol | mode |
|---|---|---|---|---|---|
| P1 thresholds: αG 0.8871 / Θ 0.4418 / g0 0.7029 / duration 66.32 / intensity 0.5992 / A 0.1859 | `exp1_failure_threshold.py`: final-G bisection per dose axis, res 1e-3, T=800 (duration axis has NO affect pulse) | as listed | cache/exp1_summary.npz `thresholds` | REPORTED | cache |
| 214 healthy / 98 stuck, zero intermediate | same driver's αG×duration grid (13×24), T=800 | 214/98 | cache/exp1_summary.npz `Gmap` | EXACT | cache |
| duration 60 (no pulse) healthy; 70 healthy; +pulse collapses; A=0.15 does not | `simulate(Params(), inward_episode(100, 100+dur, 0.9), T=800)`; pulse via `affect_pulse(..., 100, 60, A)` | 0.885 / not stuck / stuck / not stuck | §4.1 | REPORTED | recompute |
| P3 battery 816 runs | `exp2_rescue.taxonomy_grid` (6 delays × 17 strengths × 8 durations, rescue always AFTER episode end) | 816 | cache/exp2_taxonomy.npz | EXACT | cache |
| P4 second-episode threshold 66.0 (pulse) / 117.7 (no pulse) vs first 66.3 | `exp2_rescue.second_episode_threshold`: canonical rescued run = episode [100,200)+pulse + rescue u=0.8 **[400,460)**, second episode a_hold 0.9 at t=800 + pulse [800,860), bisected | 66.0 / 117.7 | cache/exp2_relapse_run.npz | REPORTED | cache |
| P4 counterexample G 0.885 → 0.049, relapse t=872.8 | same schedule, dur2=150, pulse 0.5, T=1600 | 0.885 / 0.049 / 872.8 | cache/exp2_relapse_run.npz | REPORTED | recompute (P4-Gpre/P4-relapse) + cache (P4-trel) |
| P5 transient peak +1.7% | exp3's rescued run (rescue [400,460)), T=1800: `100·(max(g)−g0)/g0` (peak at t=478.2) | 1.72% | cache/exp3_summary.npz `gain_peak_up_pct` | REPORTED | recompute |
| P5 fitted asymptote C ≈ −2.5e-8, τ = 667 = τg/μ; PSD shift 0 | `exp3_postrecovery` fit + Welch PSD | −2.5e-8 / 667 / 0 | cache/exp3_summary.npz | REPORTED | cache |
| P6 t90(S)/t90(G) = 5.41 | `exp2_rescue.hysteresis_run` (rescue at t=400), T=1800; t90 = first t past rescue with x ≥ 0.9·x_final | 5.41 | cache/exp2_hysteresis.npz | REPORTED | recompute |

## §4.2 — terminator gate + TR-14 open loop

| claim | exact invocation | expected | artifact | tol | mode |
|---|---|---|---|---|---|
| truth table (no content/no floor → 0.0486, c=1.00; other three cells → 0.8855, c≈0) | exp10 part C's `recall_run(300, pulse=0.5)` arms: a_hold 0.9 **[100,400)** + A 0.5 [100,160), T=1500; content = `u_ext 0.3 standing [0,1500)`; floor = `RegulatorParams(floor=0.7, t_engage=0)` (k_pull=c_mon=0) | as listed | [TR-13] | REPORTED | recompute (all 4 cells) |
| TR-14 closed G_end 0.0487 / open 0.0121 | `simulate(Params(g0, pi), failure_schedule(), T=900)` with (g0,pi) = (0.5,1.5) / **(0,0)** — BOTH zeroed; **T=900, not 1000**; removing other loop terms probes different models (0.108 / ≈0) | 0.0487 / 0.0121 | cache/exp18_openloop.npz | REPORTED | recompute |
| TR-14 after rescue: closed 0.8854 / open 0.8984 | same + `external_demand(600, 200, 0.8)`, **T=1400** | 0.8854 / 0.8984 | cache/exp18_openloop.npz | REPORTED | recompute |
| decomposition: g0=0 alone → 0.0106, pi=0 alone → 0.0485 (T=1000) | same at T=1000 | 0.0106 / 0.0485 | cache/exp18_openloop.npz `decomp_G_end` | REPORTED | cache (remainder-listed) |

## §4.3 — regulator

| claim | exact invocation | expected | artifact | tol | mode |
|---|---|---|---|---|---|
| floors 0.0–0.4 fail / 0.5–1.2 escape identically | `simulate_reg(Params(), RegulatorParams(floor=v, t_engage=600), failure_schedule(), T=1500)`; endpoints 0.4/0.5 recomputed | 0.049 / 0.8855 | cache/exp6_floor.npz | REPORTED | recompute (endpoints) |
| knowing floor: floor 0.7 + ungated c_mon 0.1/0.2/0.3/0.5 → 0.53/0.34/0.24/0.16 | `RegulatorParams(floor=0.7, c_mon=v, mon_gated=False, t_engage=600)`, T=1500 | as listed | cache/exp6_contrast.npz `post_ungated` | REPORTED | recompute |
| elaborate (k=2 + gated 0.8/1.5) fails: 0.116 / 0.090 | `RegulatorParams(floor=None, k_pull=2, c_mon=v, mon_gated=True, t_engage=600)` | 0.116 / 0.090 | cache/exp6_contrast.npz | REPORTED | recompute |
| deployed gated threshold vanishes (k=1 escapes at c_mon ∈ {0,0.5,1.5}) | same with `t_engage=0` | all escape | [TR-7] | EXACT | recompute |
| healthy-regime cost of floor 0.7: max\|ΔG\| = 0.00e+00 | `max|G(simulate) − G(simulate_reg(floor=0.7))|` on baseline, T=1000 | 0.0 | [TR-7] | EXACT | recompute |

## §4.4 — r·φ (see Abstract rows)

The escape grid reproduces **on the recorded cells** (φ ∈ {0, 0.25, 0.5,
1.0}). Note for anyone re-deriving: the committed driver added a φ = 0.1
column to bracket the bisection and **excluded it from the record-match
check**; on that unrecorded cell (r = 0.8, φ = 0.1) the grid reads ESCAPE
with G_end = 0.587 — a 0.087 margin over the 0.5 bar. This is not a
contradiction of the paper (the paper never cites that cell) but it is
exactly the kind of cell a naive full-grid reproduction will trip on.

## §4.5 — standing-cost budget

| claim | exact invocation | expected | artifact | tol | mode |
|---|---|---|---|---|---|
| a_hold_crit = 0.11213 (frozen channel) | bisect `simulate(Params(), Schedule(a_hold 0→3000 at v), T=3000, dt=0.5)['G'][-1] > 0.5` on [0.05, 0.30], 12 it — **dt=0.5, T=3000** (exp7 part (b)) | 0.11213 | cache/exp7_partb.npz | REPORTED | recompute |
| T_max = 448.93 / 224.46 / 112.26 at c_cap = 0.05/0.1/0.2; c_int = 0.11223/0.11223/0.11226 | exp7 part (b) bisect over T (window module, baseline schedule, T=3000, dt=0.5); product forced by construction | as listed | cache/exp7_partb.npz `T_max_c_int` | EXACT (cache) | cache |
| sustained-stress sign reversal 1.393→1.326→1.271→1.221 | exp7 part (c): dur-300 episode, bisect dip > 0.1 over a_hold, T=1200, dt=0.5 | as listed | cache/exp7_partc.npz | REPORTED | cache |
| per-dip margins +0.0028 (n=60) / +0.0339 (n=100), no negative episodes; inter-episode cost mean −0.058/−0.075, min −0.164/−0.180; G_end 0.8755 vs 0.8851 | exp7 part (d) 4-arm battery (dt=0.5, T_set=110); margins from cached per-episode dip arrays | as listed | cache/exp7_partd.npz | REPORTED | cache |
| window-only cannot self-rescue (0.0486/0.0487) | same battery's window-only arms | as listed | cache/exp7_partd.npz | REPORTED | cache |

## §4.6 — substrate legs

| claim | exact invocation | expected | artifact | tol | mode |
|---|---|---|---|---|---|
| K=0 stuck at every N=2…128; K=1 resolves at every N | `sref2.run(N, K)` (artifact-root `sref2.py`) | stuck / resolve | [TR-10] | EXACT | recompute |
| steps scale ~ linearly: 4/6/10/18/34/66/130 = 2N | `sref2.run(N, 10^6)` (the paper's "~2N + 2" is the same line within its "~") | 2N | [TR-10] | EXACT | recompute |
| cut-elimination: compact 1+k·d; expanded 31/511/8191 (k=2, d=4/8/12), 797161 (k=3, d=12) | `cutred2.compact_size / expanded_size`; recursion c(d)=1+k·c(d−1) closes as (k^(d+1)−1)/(k−1) — so 2^(d+1)−1 and (3^13−1)/2 | as listed | [TR-11] | EXACT | recompute |
| Datalog: ladder circuit 12/40/84/144/220/312/420; chain 3/10/21/36/55/78/105; ladder formula 2^(n−1) | `datalog-leg.derivable_pairs(ladder(n))` / `(chain(n))`; `path_count(ladder(14))` = 2^13 | as listed | [TR-32] | EXACT | recompute |

## §4.7 — permissive AND-gate (Tier B: the 60-night battery is expensive)

| claim | exact invocation | expected | artifact | tol | mode |
|---|---|---|---|---|---|
| only the conjunction collapses: G_min 0.0397 vs 0.483/0.518/0.518 (fast/slow/drive-only) | `exp5_permissive.py` "and" battery (60 nightly canonical drives, four arms) | as listed | cache/exp5_and.npz | REPORTED | cache |
| corner boundary: ser_crit(N0=28) = 0.187; N0_crit(ser=1) = 14.4 | exp5 boundary bisections | as listed | cache/exp5_boundary.npz | REPORTED | cache |

## §4.8 — discriminator (Tier B)

| claim | exact invocation | expected | artifact | tol | mode |
|---|---|---|---|---|---|
| λ boost 0.0071425 = λ ratchet 0.0071435 (rel. 1.4e-4) | `exp4_discriminator.py` second-rescue rise-speed battery; the pinned g0-boost at K=1.0169 vs the ratchet | 1.4e-4 | cache/exp4_lambda.npz | REPORTED | cache |
| frozen g\*/g0 = 1.00000000 at T=12000 (neither signature) | exp4 long-horizon arm | 1.0 | cache/exp4_summary.npz | REPORTED | cache |

## §4.9 — CSD audit (Tier B: the audit's committed outputs are the artifact)

| claim | exact invocation | expected | artifact | tol | mode |
|---|---|---|---|---|---|
| border-collision fold at ε_c = 0.265192279, on the manifold (\|E−Θ_eff\| = 1.03e-13 at death) | `csd/test1_fold.py` (warm-started damped Newton on the exact frozen RHS, continuation 0.255→0.275 + fine 1e-5) | as listed | csd/cache/test1_out.txt | EXACT (text) | cache |
| dominant eigenvalue −0.001500 exactly constant (spread 0.00e+00); G-mode 0.53% vs 97.9% saddle-node; δ=0.05 recovers ≤0.20 / collapses ≥0.24 | `csd/test2_discriminator.py` (exact Jacobians + basin-aware perturbation, δ = 40% basin half-width) | as listed | csd/cache/test2_out.txt | EXACT (text) | cache |

## §4.10 — regime map, dwell, protections

| claim | exact invocation | expected | artifact | tol | mode |
|---|---|---|---|---|---|
| release anchor (χ=0.3, η=0.3): c_max 1.000, releases; G=0.8834 by t=260 | `simulate(replace(Params(), chi=0.3, eta=0.3), a_hold 0.9 [100,180) + A 0.5 [100,160), T=1500)` [= exp10 `regime_run(0.3, 0.3)`] | as listed | [TR-30] | REPORTED | recompute |
| partial band η=0.3: χ=0.35 → G_end 0.1472 | same at χ=0.35 | 0.1472 | [TR-30] | REPORTED | recompute |
| χ_release(η) / χ_lock(η) at 7 η values (14 numbers) | exp10 part A bisections: canonical episode **dur 80 + pulse 0.5**, T=1500, bisect [0.02,1.60]×11, grid 0.005 — **the affect pulse is part of the canonical trigger** (without it the loop never fires on the swept range) | table in §4.10 | cache/exp10_partA_bisect.npz | REPORTED | cache |
| **TR-31 dwell threshold 118.70** | **recall protocol**: `simulate(Params(), inward_episode(100, 100+dur, 0.9), T=900)` — canonical ICs, **NO affect pulse, NO engagement**; criterion `is_stuck(sol, p)`. Check: 118.65 healthy (G_end 0.8854), 118.70 stuck (G_end 0.0487) — GRID-QUANTISED, transition in (118.65, 118.70]; the paper quotes the smallest stuck grid cell, **not** the bisection midpoint 118.825 | 118.65/0.8854 healthy; 118.70/0.0487 stuck | cache/exp18_dwell.npz | GRID-QUANT | recompute (both cells) |
| exp10 part-B dwell vs S0: 18.73/35.43/72.73/98.80/123.95/146.98 | **different protocol** (settled ICs from t=5000 baseline, S re-pinned at S0, a_hold 0.9 from t=0, standing u_ext, collapse = G < 0.1 at D+300) — gives 98.80 at the default S0=0.8, NOT 118.70 | as listed | cache/exp10_partB_dwell.npz | REPORTED | cache |
| dwell vs u_ext (S0=0.8): 98.80/121.18/171.78/262.73/384.88; none to 700 at ≥0.15 | same construction, standing u_ext | as listed | cache/exp10_partB_dwell.npz | REPORTED | cache |
| two protections: recall-300 alone 0.0486; +u_ext 0.3 or +floor 0.7 → 0.8855 | §4.2's truth-table construction (recall_run(300, pulse=0.5) arms) | as listed | cache/exp10_partC_protection.npz | REPORTED | recompute (all 3 arms) |
| bisected duration threshold 67.5 (vs P1's 66.32) | exp10 part C bisection, grid 1.0 | 67.5 | cache/exp10_partC_protection.npz | REPORTED | cache |

## §4.11 — exp20 sweeps (TR-35) and exp19 quadrants (TR-36, Table 1)

**exp20 PART 0 — the canonical reproduction (Tier A's anchor set; it pins
the three-phase reading):**

| claim | exact invocation | expected | artifact | tol |
|---|---|---|---|---|
| G(onset) = 0.885 | `simulate(Params(), failure_schedule(), T=1200)`; G at t=100; dt 0.05 | 0.885 | cache/exp20_canon.npz | REPORTED |
| G at the c=0.5 boundary = 0.140 | same run; G at onset + t_c50 (first grid sample with c > 0.5) | 0.140 | " | REPORTED |
| G_end = 0.0486 | same run | 0.0486 | " | REPORTED |
| t_c0 = 65.25 | same run; first t ≥ 100 with c > 0, minus 100 | 65.25 | " | REPORTED |
| E_plateau = 0.525 | mean E over the 10 t.u. before t_c50 | 0.525 | " | REPORTED |
| runaway_share = 0.891 | (G(onset) − G(t_c50)) / (G(onset) − G_end) | 0.891 | " | REPORTED |
| setpoint share = 0.569 (0.525→see row) | mean\|dΘ_eff/dt\| / (mean\|dE/dt\| + mean\|dΘ_eff/dt\|) over the final 15 t.u. before t_c50 | 0.569 | " | REPORTED |
| g_excursion = 0.0061; stuck | max(g) − g(0); `is_stuck` | 0.0061; stuck | " | REPORTED |

**exp20 sweeps:**

| claim | exact invocation | expected | artifact | tol | mode |
|---|---|---|---|---|---|
| τ_S crossings 35.0/47.7/67.5 at τ_S = 25/50/100 | `simulate(replace(Params(), tau_S=v), failure_schedule(), T=1200)`; crossing = first c > **0.5** (t_c50: 35.00/47.65/67.50). The c > 0 times are 34.5/46.8/65.25 — **the paper's sweep numbers are the t_c50 convention** | as listed | cache/exp20_tauS.npz | REPORTED | recompute |
| fit 0.428·τ_S + 25.1, R² = 0.996 | polyfit(tau_S, t_c50) over the collapse cells | as listed | cache/exp20_tauS.npz | REPORTED | cache |
| τ_S_crit = 141.47 (collapse 141.45, healthy 141.50) | `simulate(replace(Params(), tau_S=v), failure_schedule(), T=1200)['G'][-1] < 0.1` for the bracket — GRID-QUANTISED; note this one's reported value is exp20's **midpoint** 141.475 | bracket holds | cache/exp20_probe.npz | GRID-QUANT | recompute |
| E pre-crossing τ_S-freedom 9.3e-12 | max\|E(τ_S=25) − E(τ_S=100)\| before the faster run's crossing (cached traces) | 9.3e-12 | cache/exp20_probe.npz | REPORTED | cache |
| setpoint share 0.435/0.443/0.569; runaway share 0.817/0.862/0.891 at τ_S = 25/50/100 | exp20 part_tauS phase decomposition | as listed | cache/exp20_tauS.npz | REPORTED | cache |
| τ_g episodic: G_end 0.0485–0.0486 (spread 1.5e-4); excursion 0.0845 (6.25) → 0.0017 (800) | `simulate(replace(Params(), tau_g=v), failure_schedule(), T=1200)`; endpoints recomputed | as listed | cache/exp20_tauG_epi.npz | REPORTED | recompute (endpoints) |
| reachable gain ceiling 0.5845 (g0 + excursion at τ_g=6.25); held-g escape boundary 0.7096; held-at-ceiling still collapses 0.0568 | exp20 probe (iii)/(iv): τ_g = 8000 with g_init = 0.5845 holds g; bisection over held g | as listed | cache/exp20_probe.npz | REPORTED | cache |

**exp19 quadrants (Table 1 is verified cell-for-cell by `reproduce.py`
rows E19-table-\*, from `cache/exp19_part1_traj.npz` with the registered
rule a > A\* = 0.5, D > D\* = 0.65368, boundaries → low side, armed = E >
Θ_eff):**

| claim | exact invocation | expected | artifact | tol | mode |
|---|---|---|---|---|---|
| D\* = 0.65368 = midpoint of the demand nullclines 0.54545/0.76190 | `0.5·(D_null(A=0) + D_null(A=0.5))` with `D_null(A) = (D_base+A)·β_D/((D_base+A)·β_D+δ_D)`; A\* = 0.5 is the registered attention midpoint | 0.65368 | [TR-36] | REPORTED | recompute |
| Table 1 (all rows, full run and episode window) | exp19 part_traj five runs at dt=0.05 (baseline/recall100/failure T=1500; brink pair T=900); occupancy over the stated window | table in §4.11 | cache/exp19_part1_traj.npz | REPORTED (0.01%) | cache |
| Q3 transit [146.25, 181.25], 35.00 t.u.; entry 146.25 | `simulate(Params(), failure_schedule(), T=1500)` + quadrant rule | as listed | " | REPORTED | recompute |
| switch arms t = 165.25 (19 t.u. after Q3 entry); armed 1334.75 t.u. | same run; first t > 100 with E > Θ_eff; T − t_arm | as listed | " | REPORTED | recompute |
| terminal a = 0.8824, D = 0.5455 (relaxed position) | same run, a(T), D(T) | as listed | " | REPORTED | recompute |
| threshold-free scan: recall visits Q3 on 0.00% of the 31×23 (A\*, D\*) grid, failure on 60.87%; base-Q1/terminal-Q2/Q4-empty on 100% | exp19 part_areas F5 scan (reclassify cached trajectories at every grid point) | as listed | cache/exp19_part2_quad.npz | REPORTED | cache |
| recall enters Q2 at t=100.60 (predicted 100.55), exits 202.75; survives G_end 0.8855 | `simulate(Params(), inward_episode(100, 200, 0.9), T=1500)` | as listed | " | REPORTED | recompute |
| boundary pair: brinkOK 6.15 t.u. armed flicker (t = 217.50→223.65; 124 samples × dt = 6.20 under the count convention), G_end 0.8854; brinkSTUCK armed from 217.50, G_end 0.0487 | the two dwell runs (T=900) | as listed | " | REPORTED | recompute |

The paper's "armed for 1318.70 t.u." (§4.11 decisive-result paragraph) is
1500 − 181.30, the horizon minus the first armed sample **after** the Q3
transit — an endpoint convention the checker records but does not
independently re-derive (see remainder list).

---

## Explicit remainder (not covered above — stated, not silently sampled)

1. **exp12/exp13/exp14/exp15/exp16 batteries** ([TR-33], [TR-34]): the
   generational realization (7×24 registered decisions, 0 uncertified
   adoptions, no degradation criterion fired), the blind-lever construction
   (0 blind adoptions), and the train/test-lift numbers (336 adoptions, test
   G_end 0.9562–0.9806, correlation +0.46). Multi-hour recompute; caches
   `cache/exp12_*.npz` … `exp16_*.npz` hold every number; not row-verified
   here.
2. **§4.3's long-horizon patterns** (dip minima 0.218/0.184/0.163/0.139/
   0.132 at a_hold 0.4–0.9; dense-weak failure at episode 2; permanent
   capture G = 0.21/0.16/11 at a_hold 0.35/0.5/0.9): cache
   `exp6_longhorizon.npz`; dt = 0.25, T up to 4000.
3. **P2's ~2500-run no-self-recovery sweep** and the 106/112 sensitivity
   cells: `cache/sensitivity.npz`, predictions.md P2.
4. **exp10 part A's full 17×31 regime map** (the bisected boundaries are
   verified; the map itself is `cache/exp10_partA_map.npz`).
5. **§4.1's rescue-map panels** (u\* 0.95→0.45 falling with duration,
   0.40→0.95 rising with delay): derivable from `cache/exp2_taxonomy.npz`,
   not row-verified.
6. **Figures 1–14**: verified programmatically at generation time by their
   drivers (`*_verify` parts); no image is opened by this file.
7. **The CSD audit's recomputation** (as opposed to its committed text
   outputs): `csd/test1_fold.py` / `test2_discriminator.py` run in ~minutes
   from the bundle; the rows above read the audit's recorded outputs, which
   is the artifact the paper cites.
8. **exp18's T=1000 decomposition arm** (0.0106/0.0485/0.0104) and the
   coarse [5, 700] bisection midpoint 118.825: cached
   (`cache/exp18_openloop.npz`), not recomputed (Tier A already recomputes
   the four T=900/1400 anchors).
9. **TR entries that are decision records or prose** ([TR-3], [TR-12],
   [TR-15], [TR-18], [TR-20]–[TR-28]): no number to check beyond what the
   rows above already carry.
10. **The 1318.70 t.u. armed-residence duration** (§4.11): recomputed
    during construction (1500 − 181.30 under the after-Q3-transit
    convention) but not kept as a row because the convention is the paper
    prose's own; the 1334.75 row (T − t_arm) is the unambiguous form.

## Negative results (things checked that could have failed and did not)

* Every GRID-QUANTISED bracket holds on both sides (dwell 118.65/118.70;
  τ_S_crit 141.45/141.50) — recompute, not just cache.
* The frozen model files `dpdr/dpdr/{model,integrate,metrics,events}.py`
  were not modified: this file and `reproduce.py` are the only additions.
* No new claim or number is introduced anywhere above; every expected value
  is quoted from the paper or its caches. The one place the record and a
  naive reproduction diverge — exp11's unrecorded (r=0.8, φ=0.1) cell — is
  documented in §4.4 above rather than "fixed".
