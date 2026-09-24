# Testable predictions — dpdr model (plan §6)

Each prediction: statement, the figure that demonstrates it, phenomenological
correlate, **falsifier**, and the measured result at the frozen Step 0–B
parameters (no retuning was done in Steps C–D).  Scenario geometry:
episode `[100, 200)` with `a_hold=0.9`, affect pulse 0.5 on `[100, 160)`;
rescues always start ≥ 200 (after episode end — the relapse-confound control).

**Verdict summary: P1 PASS · P2 PASS · P3 PASS · P4 PASS under recurring
episodes (absent under single-episode schedules) · P5 FAIL ·
P6 PASS.**  Details and numbers below.

---

## P1 — Threshold dose-response (PASS)

**Statement.** Failure risk is a function of episode duration × intensity ×
affect amplitude (separable interaction): below a threshold dose the system
returns to healthy; above it, G collapses to the stuck attractor — a sharp,
not graded, transition.

**Figure.** `figs/f05_bifurcation.png` (1-D curves per axis),
`figs/f04_sweep_heatmap.png` (α_G × duration regime map).

**Measured.** Saddle-node thresholds (bisection on the final-G discontinuity,
resolution 1e-3), healthy side in parentheses:

| axis | threshold | healthy side |
|---|---|---|
| `alpha_G` | 0.8871 | below |
| `Theta` | 0.4418 | above |
| `g0` | 0.7029 | above |
| episode duration | 66.32 t.u. | below |
| episode intensity | 0.5992 | below |
| affect amplitude A | 0.1859 | below |
| `eta` | **none** — graded (G(T) 0.108→0.031 over η∈[0,2.4], no jump) | — |

α_G×duration interaction (Phase-2 grid): the duration threshold falls
monotonically 200→60 t.u. as α_G rises 0.50→1.34; α_G ≤ 0.45 never collapses
at any duration ≤ 200.  Across the full grid: 214 healthy / 98 stuck cells,
**zero intermediate** (no cell ends with 0.1 < G(T) < 0.5 after an episode —
the transition is all-or-nothing at the horizon).

**Correlate.** Episode severity (duration × absorption × affect load)
determines DPDR onset with an effective threshold, not a linear gradient.

**Falsifier.** Failure risk independent of any of the three dose axes, or a
graded final-G continuum across dose.  **Not triggered** — duration,
intensity and A each show a sharp threshold; the exception is η (see note).

**Note.** At the frozen parameters the collapse is driven by the allostatic
term α_G·a·G plus the affect-raised E crossing Θ_eff; the cannibalization
cost η·c·G modulates the collapse depth but does not carry its own saddle-node
in [0, 2.4].  The three dose axes trade off: at the canonical duration (100
t.u.) affect is required (A-threshold 0.186; A=0 stays healthy), but a longer
episode collapses even with A=0 (duration threshold ≈ 150 t.u. at A=0).
The graded η curve is **not** smooth protection at low η: at η ≈ 0 the run
ends at G = 0.108 with c = 1.0, a = 0.88, Θ_eff = 0.12 — full
cannibalization, attention captured, i.e. the same mechanism-stuck attractor
as at every other η (the collapse machinery is fully on across the whole
range); η only sets the stuck floor's depth, which slides under the G < 0.1
classifier bar by 0.008 at η ≈ 0.  "Graded" therefore means "the floor
crosses the 0.1 threshold", not "η is protective at low values".

---

## P2 — No self-recovery (PASS)

**Statement.** Once collapsed (G < 0.1, E > Θ_eff, c > 0.5), the system does
not recover without external demand: the no-pull attractor is stable.

**Figure.** `figs/failure.png` (Step A), grid columns coded
"stuck(depersonalized)" in `figs/f04_sweep_heatmap.png`.

**Measured.** Canonical failure run: G stays at 0.049 for ≥ 600 t.u. (≥ 30·τ_G)
after episode end; max post-episode G = 0.050 (gate G2b).  In the sensitivity
sweep the canonical episode still ends stuck in **106/112** perturbation
cells; the 6 exceptions are cells where the episode no longer collapses the
system at all (α_G −30%, β_D −30%, S_rest −20/−30%, Θ +20/+30%) — i.e. P2's
premise (deep collapse) is never met there, rather than self-recovery
occurring.

**Correlate.** Chronicity; recovery requiring changed circumstances
(job, crisis, relationship) rather than time alone.

**Falsifier.** Spontaneous recovery from deep collapse.  **Never observed**
in any of the ~2500+ runs of Steps C–D (including the second-episode and
long-horizon scans of the fix pass).

---

## P3 — Rescue dose-response with sharp threshold (PASS)

**Statement.** Rescue success increases monotonically with external-demand
strength and duration along a sharp bifurcation line (minimal effective
"dose" of engagement); delaying the rescue raises the required dose.

**Figure.** `figs/f06_rescue_map.png` (strength × duration × delay panels).

**Measured.** Outcome fractions over the controlled grid
(6 delays × 17 strengths × 8 durations = 816 runs):
**failure 0.630 · transient 0.000 · full 0.370 · no-collapse 0.000.**

- Transition width in u_ext at fixed (delay=200, duration): one scan step
  (≤ 0.05) for every duration — the success boundary is sharp to grid
  resolution (gate G3's monotone-region condition holds; no patchwork).
- Dose threshold `u*` falls with duration and rises with delay, e.g.
  at delay 200: dur 40→200 gives u* = 0.95→0.45; at duration 60:
  delay 30→600 gives u* = 0.40→0.95 (time makes rescue harder — consistent
  with D relaxing only toward 0.61 and S habituating low).
- No rescue succeeds with duration ≤ 30 at delay ≥ 200 (the longest waits
  tested; likewise at delays 400 and 600) at any strength ≤ 1.0 after the
  episode ends: past a long wait there is a hard minimum engagement duration.
  (At the shortest waits the boundary is slightly lower — duration 30
  succeeds from delay 30 at u* ≥ 0.6, duration 20 from delay 30 at
  u* ≥ 0.9 — but never duration 10 anywhere in the grid.)

**Correlate.** Minimal effective "dose" of engagement needed to break
DPDR; longer waited, harder to break.

**Falsifier.** Smooth/gradual success everywhere, or a non-monotone success
region.  **Not triggered.**

---

## P4 — Relapse class exists (PASS under recurring episodes)

**Statement.** Parameter regimes where rescue lifts G past 0.5 and then
relapses to G < 0.1 after the external demand ends; relapse risk higher when
inward pull reasserts after external demand ends.

**Figure.** `figs/f06b_relapse.png` (recurring-episode grid + the canonical
relapse trajectory), `figs/f06_rescue_map.png` (single-episode grid — no
orange cells there).

**Measured.**

- **Single-episode schedules: relapse fraction 0.000.**  In the controlled
  grid (816 runs: 6 delays × 17 strengths × 8 durations) and in a
  deliberately confounded grid (112 runs with the rescue overlapping the
  still-active episode: 0.705 no-collapse, 0.205 full, 0.089 failure), no run
  lifts past 0.5 and then re-crosses below 0.1.  The largest post-collapse G
  reached by a non-rescued run anywhere is 0.298 (sensitivity
  `peak_partial` max over all 112 cells).  With no new trigger, a rescued
  state is stable for the right micro-structural reason: E ≤ 0.15 «
  Θ_eff ≈ 0.76 after a full rescue (G ≈ 0.88, S ≥ S_rest), so c cannot
  reactivate.
- **Recurring episodes: relapse exists and is threshold-like.**  Drive the
  *canonical rescued run* (episode [100,200), rescue [400,460) → 'full',
  G = 0.885) with a **second** inward episode at t = 800 — the same
  a_hold 0.9 and 60-t.u. affect pulse 0.5 the canonical scenario already
  uses; no new mechanism, no retuning, schedule only (`experiments/exp2`
  Phase 3c):
  - Counterexample (second-episode duration 150, pulse 0.5):
    **G 0.885 → 0.049, c = 1.0, Θ_eff = 0.129**; the shipped
    `dpdr.metrics.classify()` labels it **'relapsed'** and
    `detect_relapse()` fires at **t = 872.8**.
  - Threshold in second-episode duration (grid): relapse appears between
    duration 50 and 75 with the pulse, between 100 and 125 without it — a
    one-grid-step boundary, like every other threshold in this model.
  - Like-for-like: the second-episode duration threshold (bisected) is
    **66.0 t.u. with pulse / 117.7 without**, versus the first episode's
    **66.3 / ≈150** from a healthy start.  The recovered state's G ≈ 0.885
    head start buys **no protection** — the S-loop that slowly restored
    Θ_eff (0.13 → 0.76) after the rescue is exactly what re-collapses it
    (0.76 → 0.13) during the second episode, and there G's head start does
    not help.  Collapse onset is identical too: G < 0.1 is reached 73 t.u.
    into episode 2 versus 73 t.u. into episode 1, despite starting from
    G = 0.885 instead of 0.7.

**Why the first P4 write-up said FAIL.**  The original verdict generalized
the single-episode E ≤ 0.15 < Θ_eff argument to "a rescued state is
unconditionally stable".  That argument is correct only while no new trigger
arrives; it does not survive a recurring episode, which was listed as
out-of-scope but required no new mechanism and no retuning to test.

**Correlate.** Clinical relapse subpopulation — relapse under renewed
trigger exposure (stress, renewed absorption), with a threshold-like dose,
not random drift.

**Falsifier.** No relapse regime anywhere in sweep space.  **Not triggered**
— relapse is reachable at the frozen parameters via recurring episodes, with
a sharp dose threshold in second-episode duration.

---

## P5 — Post-recovery vigilance (FAIL)

**Statement.** Rescued systems show elevated gain g\* > g0, faster
E-corrections (higher PSD frequency), higher damping.

**Figure.** `figs/f07_postrecovery.png` (g(t) annotated with its transient
peak and g0 asymptote), `figs/f07b_vigilance.png` (transient g(T)/g0 vs η),
`figs/f07b2_freq.png`.

**Measured.**

- **g\* > g0 at steady state: NO — the model produces no persistent
  vigilance.**  The equilibrium of
  `dg/dt = (π·max(0,|dE/dt|−ref) − μ·(g−g0))/τg` (deadband
  `ref = dEdt_ref = 0.002`) is g = g0 once E settles
  (dE/dt → 0), with relaxation time τg/μ ≈ 667 t.u.  The rescue's last
  |dE/dt| kick is at t ≈ 478, so g(T) is still relaxing at every practical
  horizon.  Measured on the canonical rescued run (T → 12000):
  **T=1000 +0.79 %, T=1800 +0.24 %, T=3000 +0.039 %, T=6000 +0.00044 %,
  T=12000 +0.00000 %** — a clean exponential decay to g0.  Fitting
  `g(t)−g0 = C + A·exp(−t/τ)` over the post-peak window gives
  **τ = 667 t.u. (= τg/μ exactly), A = 0.0086, C = −2.5·10⁻⁸** — the
  persistent offset C is zero to numerical precision; everything reported
  earlier as "g\* > g0" was finite-horizon relaxation residue.  (The
  originally reported +0.24 % was the T=1800 value.)
  The honest statement: **transient upregulation of ~+1.7 % peaking at
  t = 478 (g = 0.5086), decaying back to g0 with time constant ≈ 667 t.u.**
- **g\* vs η (vigilance compensates capture risk): NO.**  g(T)/g0 is flat at
  1.0023–1.0024 for η ∈ [0.2, 1.3] (all full-rescue runs, transient values);
  at η = 1.4 rescue fails.  The capture risk does not modulate the
  vigilance signature.
- **Ringing / PSD frequency shift: NO.**  The E-loop is overdamped: zero
  resolvable E peaks after rescue end at any g0 ∈ [0.2, 3.0] (log-decrement
  undefined), and the Welch PSD peak sits at the same lowest resolvable
  frequency bin (0.0049 t.u.⁻¹) for baseline and rescued runs — **shift = 0**.
  Post-rescue E variance is ~30–150× the baseline level (transient
  settling, not oscillation).  Gate **G4's spectral-shift component fails.**
- Structural cause: D is purely exogenous (plan §7 judgment call), so the
  G–D error loop has no restoring–inertia pair and cannot oscillate; g's
  actuator (G-modulation only) cannot change that — and g itself is a pure
  leaky integrator toward g0, so it cannot hold an offset either.

**Correlate.** Post-DPDR hypervigilance.  **Not produced by the frozen
model** beyond a ~1.7 % transient during the rescue itself.

**Falsifier.** Rescued == naive gains.  **Triggered at steady state**: the
fitted asymptote is g0 exactly (C ≈ 0), so rescued and naive systems carry
identical gain once the rescue transient has relaxed; the earlier "g\* > g0
in 98 % of cells" counted transients.

---

## P6 — Temporal-depth hysteresis (PASS — most distinctive)

**Statement.** Rescue restores G quickly; the allostatic setpoint S lags by
~τ_S/τ_G = 5× — subjective "no temporal depth" outlasts objective functional
recovery.

**Figure.** `figs/f08_hysteresis.png` (raw G(t), S(t) and normalized
recovery curves + t90 markers; vision-verified to show the lag), inset panel
of `figs/f07_postrecovery.png`.

**Measured.** Canonical rescued run (rescue at t=400, u=0.8, 60 t.u.):
**t90(G) = 69.2, t90(S) = 373.8, ratio = 5.41** vs predicted τ_S/τ_G = 5.0.
Sensitivity: the ratio tracks τ_S almost exactly
(τ_S ±30 % → ratio 3.87...7.35) and inversely tracks τ_G
(τ_G ±30 % → 6.87...4.51); across all 87 cells where the canonical rescue
yields 'full' the ratio spans **3.87–8.30** (median ≈ 5.4), i.e. the lag is
a timescale-separation mechanism, robust to every other parameter's ±30 %.

**Correlate.** Retrospective report of shallow-time persisting during
functional recovery.

**Falsifier.** S and G recover at the same rate.  **Not triggered** — the
ratio never drops below 3.9 in any perturbation cell.

---

## Sensitivity summary (§6b; `experiments/sensitivity.py`, 28 parameters × ±20/±30 % = 112 cells)

Outcome measures: `thr_dur` (P1 duration threshold), `no_self` (P2),
`u_thr` (P3 minimal rescue strength), `peak_partial` (P4 stand-in),
`g_up` (P5), `lag_ratio` (P6).  Full table: `cache/sensitivity.npz`.

| Prediction | Robustness | Detail |
|---|---|---|
| P1 threshold exists | **robust** | thr_dur defined in all 112 cells, ∈ [33.6, 184.8]; in the 6 cells where the canonical 100-t.u. episode no longer collapses (α_G −30 %, β_D −30 %, S_rest −20/−30 %, Θ +20/+30 %) the threshold simply moves past the canonical episode length (130–185 t.u.) rather than disappearing |
| P2 no self-recovery | **robust** | 106/112; the 6 failures are "never collapsed", not "recovered" |
| P3 rescue threshold | **robust, delicate in level** | u_thr ∈ [0.45, 1.0]; undefined in 8 cells — 6 where the episode never collapses (α_G −30 %, β_D −30 %, S_rest −20/−30 %, Θ +20/+30 %) and 2 where the system collapses but no u ≤ 1.0 rescues at the canonical dose (β_G −30 %, Θ −30 %); the *existence* of a sharp dose threshold persists everywhere it can be measured |
| P4 relapse class | **present under recurring episodes** | single-episode peak_partial ≤ 0.298 « 0.5 in every cell; under a second episode after rescue, relapse occurs with a sharp duration threshold (see P4) |
| P5 g\* > g0 | **FAILS at steady state** | g(T)/g0−1 > 0 at T=1800 in 87/87 defined cells (+0.12…+0.47 %) but these are finite-horizon transients: the fitted persistent offset C ≈ 0 and the asymptote is g0 exactly (see P5); flat vs η; no ringing at any g0 |
| P6 lag ratio | **robust** | 3.87–8.30, set by τ_S/τ_G as predicted |

**Note on the g_up column.**  The g0 rows were contaminated before this pass:
perturbing g0 without propagating it to g_init made g relax from 0.5 toward
the moved baseline, producing spurious −0.9/−1.4 % (g0 +20/+30 %) and +3.2 %
(g0 −30 %) entries.  `experiments/sensitivity.py` now propagates g0 → g_init
(mirroring exp1 and exp3); the corrected g0 cells read +0.18…+0.30 % at
T=1800 (g0 −30 % is undefined — with a consistent initial gain the canonical
rescue no longer yields 'full' there).  All g_up values are horizon-bound
transients; none survives at steady state.

**Delicate band.** The two canonical gates live in tension (plan risk #6).
Across the 112 OAT perturbation cells, BOTH gates hold simultaneously in 87;
the canonical episode fails to collapse in 6 (α_G −30 %, β_D −30 %,
S_rest −20/−30 %, Θ +20/+30 %) and the canonical rescue dose (u=0.8, 60 t.u.)
fails in the remaining 19.  Most
parameters tolerate ±30 % in both directions, but the
Θ_eff = Θ·S/S_rest coupling (plan risk #4's "strongest assumption") is the
tightest: **no ±20/±30 % perturbation of Θ or S_rest preserves both gates** —
Θ +20/+30 % and S_rest −20/−30 % prevent the collapse, while the opposite
directions leave the system collapsed but unrescuable at the canonical dose —
and β_G, k_ext, δ_D hold only on one side (+) while η, χ, k_in hold only on
the other (−).  One-sided tolerance of this kind means the
defaults sit near a ridge of the joint feasible set, not at its center.

Sobol variance decomposition was not run (the OAT grid already answers the
robust-vs-delicate question the plan poses; ~2600 runs, including the
second-episode grid and the long-horizon gain scan of this fix pass).

---

## Opt-in variant: two-axis permissive AND-gate (`dpdr/permissive.py`, `experiments/exp5_permissive.py`) — ADDED, not a verdict change

**Context.** The frozen model is single-factor-sufficient (OR-gate): inward
attention alone collapses it (a_hold 0.35 × 1400 t.u. → G stuck 0.0485),
while the reported etiology was a conjunction of four factors. This opt-in
variant makes the AND-gate structure explicit and testable. It touches
nothing frozen: `dpdr/permissive.py` is a new module, `pin_phi = 1`
reproduces the frozen `simulate()` to integrator tolerance (max |ΔG| ≤
1.4·10⁻⁵, wrap not duplicate), and the frozen G1–G2c gates and all 34
tests still pass.

**Structure.** The allostatic weakening term becomes multiplicative in two
thresholded axes, −Φ(t)·α_G·a·G with Φ = Φ_max·g_fast(A_ser(t))·g_slow(N(t)):
a fast/serotonergic axis (nightly capsule window; channel `ser`, Hill n=1,
K=0.25) and a slow/neurotrophic axis (daily NGF induction; channel `ngf`
accumulating N in full-dose-day units, sigmoid N50=14 d / Nw=4 d, μ_N=0).
Φ = 1 is the frozen model; Φ = 0 removes the weakening term entirely —
under the gate, no amount of inward attention alone can collapse the
system. All constants are hypothesis-space choices documented in
`permissive.py` (none fitted to the event); timescale 1 t.u. = 5 min ⇒
T_day = 288, window = 96 t.u. = 8 h. Drive protocol = the frozen model's
own canonical trigger (a_hold 0.9 × full window + A = 0.5 × 60 t.u.),
repeated nightly.

**Measured (cache/exp5_*.npz, figs/f10_permissive.png).**

- **(a) AND-gate demonstrated.** Under the identical nightly canonical
  drive: conjunction collapses (stuck, onset night 13.3, G_min 0.040);
  fast-only, slow-only and drive-only arms stay healthy over **60 nights**
  (G_min 0.483 / 0.518 / 0.518). Frozen reference: ONE such night at Φ = 1
  (frozen default) already collapses (G 0.049) — the OR-gate the variant
  replaces.
- **(b) Corner, not a band.** Single-window boundary (stuck criterion;
  slow axis pre-set via N₀): no collapse anywhere at N₀ ≤ 14; both legs
  finite — ser_crit(N₀=28) = 0.187, N₀_crit(ser=1) = 14.4 full-dose days
  (critical in-window Φ ≈ 0.83–0.85, consistent with the pinned-Φ constant
  assay thr_dur(P): Φ=0.8 → 89.8 t.u. ≈ window length 96, after the ~6%
  duty-cycle discount). The ser = 0 row is clean by construction (Φ = 0);
  only the N₀-leg is a dynamical finding.
- **(c) Timing.** 24-cell (a_hold × dose) ensemble on the nightly
  protocol: **24/24 onsets inside the permissive window** (phase ∈ [0,
  0.333); uniform-onset null ≈ 8). Subclusters: 15 at window start, 9 at
  window end (phase 0.31–0.33 ≈ 7.5–8 h, i.e. “late at night”); the
  window-end subcluster is associated with *marginal* conjunctions — the
  reported event’s late-at-night timing corresponds to the marginal branch,
  which the model does not independently confirm.
- **(d) Persistence.** Priming (15 nights, both axes, no drive) → 28-night
  gap → re-exposure (canonical drive): μ_N = 0 holds N = 14.25 at
  re-exposure and re-exposure collapses (×2 nights); μ_N = 1/42/day washes
  N to 6.18 and the same re-exposure does not collapse. Post-collapse
  withdrawal of both axes leaves G stuck (0.095) — but that G2b
  stuck-ness is INHERITED from the frozen model’s attractor, not an
  emergent property of the gate; the variant-specific persistence claim is
  about N (vulnerability), not G.

**Honest negatives / caveats.**

1. The conjunction arm’s collapse is *marginal*, not dramatic: Φ ≈ 1.5
   in-window sits just above the boundary Φ_c ≈ 0.84, onset takes ~13
   nights, and the onset night shifts 11–15 under a_hold/dose scatter.
   The conjunction is *possible*, not *dramatically enabled*, at these
   principled constants.
2. The “late at night” subcluster is a minority branch (9/24) and appears
   only for marginal drives; stronger drives collapse at window start.
3. ser_crit at N₀ = 28 ≈ 0.19 > 0 and N₀_crit at ser = 1 ≈ 14.4 d > 0, so
   the corner is genuine — but its N₀-leg depends on the N50/Nw choices;
   it is a hypothesis-space boundary, not a measured pharmacological one.
4. Mechanism linking serotonergic tone / NGF induction to this G pathway
   is INFERRED from composition + timing, not established.
5. Sleep architecture is an unresolvable confound in this single case
   (5-HTP → melatonin shifts; caffeine + endurance load = sleep pressure;
   sleep deprivation is an independent DPDR trigger).

**Falsifier.** A case collapsing with inward drive alone (no serotonergic
axis, no weeks-scale accumulation) would kill the AND-gate reading; a case
with both axes but no sustained inward drive collapsing would kill the
drive requirement.

## Self-rescue regulator (`dpdr/regulator.py`, `experiments/exp6_regulator.py`) — ADDED, not a verdict change

The deliverable built for the reframed goal (a WORKING self-rescue
regulator for long-horizon agents; functional criterion, not
phenomenological).  Opt-in module; the frozen model is untouched and the
regulator-OFF wrap reproduces `dpdr.integrate.simulate` EXACTLY
(max |dG| = 0.00e+00 on all three standard scenarios).  The regulator has
three orthogonal pieces: a CHEAP FIXED FLOOR clamping the cannibalization
switch's effective threshold from below (Θ_eff_reg = max(Θ·S/S_rest,
floor); a floor on S entering the switch is the same clamp in different
units) — a constant belief that does NOT track the real state; an
ATTENTION-redirection actuator (threshold-triggered outward pull on
da/dt, the frozen external-rescue functional form); and an explicit
monitoring cost c_mon (inward-drive addend standing in for whatever real
introspection costs), GATED by the trigger (the scalar check is free;
acting on a positive check is paid introspection).

**Design principle — CONFIRMED in all five assays.**

1. **ANY cheap floor, even false.** Post-collapse (engaged at t=600 with
   the stuck attractor settled, G=0.049, a=0.882, true Θ_eff*=0.122):
   floor 0.0–0.4 → still stuck; floor 0.5/0.6/0.7/0.9/1.0/1.2 → ALL
   escape IDENTICALLY to G=0.8855, a→0, escape in ~22 t.u. after
   engagement.  Bisected critical floor = **0.4795** ≈ the stuck point's
   own error level E* = 0.4969 (the floor must sit at/above what the
   switch would otherwise see; above that, the value is irrelevant).
   S-mode floor_crit = 0.5994 (same clamp: Θ·0.5994/S_rest = 0.4795).
   Escapes under all three collapse loads (canonical episode; sustained
   affect A=0.5×1000; chronic low-grade a_hold 0.35×1400).
2. **Mechanism** (fig f11 panel f): floor → Θ_eff ≥ E → c=0 → attention
   no longer captured (a→0 via ρ_a) AND G no longer eaten (η·c·G=0) → G
   regrows.  One constant severs the whole positive-feedback loop.
3. **Cheap vs elaborate — RESCUE OF AN ALREADY-STUCK AGENT** (assay
   geometry, engage t=600 after the stuck state has settled; k_pull=2.0):
   CHEAP floor only → ESCAPED 0.8855 (cost structurally zero);
   ELABORATE (actuator + gated c_mon=0.8) → **FAILED 0.116**;
   c_mon=1.5 → FAILED 0.090.  Monitoring-cost threshold (bisected,
   k=2): **c_mon_crit = 0.511**.
   **SCOPE (review finding 1): this contrast is a property of the
   POST-COLLAPSE SETTLED assay, not of deployment.**  In the deployed
   configuration — always armed, t_engage=0 — the gated threshold
   VANISHES: the trigger gate fires early in the descent (G crosses
   G_trig=0.3 at t~121, while a is still rising), the actuator releases
   attention before capture consolidates, and k_pull=1.0 escapes at every
   gated c_mon in [0, 1.5] (G_end=0.8855; measured at c_mon =
   0/0.5/1.5 and k=2 at 0.8/1.5 — all escape, G_min ≥ 0.21).  An
   elaborate GATED introspector is therefore NOT fatal in deployment;
   the floor's deployment advantage is COST-FREE-NESS (no observation
   channel at all), not beating a gated introspector.  What IS fatal in
   deployment is the UNGATED always-on vigilance axis (negative 6
   below).  The c_mon ~ 0.5-0.8 magnitude from the earlier probe is
   CONSISTENT IN MAGNITUDE with c_mon_crit = 0.511 after unit
   conversion — it is not a reproduction: the two actuator
   normalizations differ (~2×), and the earlier probe's arm (k_pull=1.0
   raw, c_mon=0) does NOT rescue in this regulator (G=0.114, a genuine
   equilibrium at T=1500/4000/12000).  Note k_crit rises smoothly with
   c_mon (1.256/1.547/1.984/2.421 raw = 0.314/0.387/0.496/0.605
   u_ext-units at c_mon=0/0.2/0.5/0.8; the k/k_ext = k/4 conversion is
   ASSERTED — a G-dependent sigmoid-gated pull equated to a constant
   external demand — not derived): authority can BUY BACK some
   monitoring cost, but never all of it.
4. **Long horizon** (functional criterion): canonical episodes ×60
   (a_hold 0.9×100 + pulse, gap 300): frozen agent collapses IN EPISODE
   1 (t=173) and stays stuck 60 episodes (G_end 0.0486); floor-regulated
   agent NEVER enters the stuck attractor (G_min 0.1318, 0/60 episodes
   below 0.1, G_end 0.885).  **MARGIN, not the binary count (review
   finding 5): the regulated dips bottom at 0.218/0.184/0.163/0.139/
   0.132 for a_hold 0.4/0.5/0.6/0.8/0.9 and never below 0.103 up to
   a_hold=2.0 — the floor converts the frozen 0.049 attractor into a
   ~0.10-0.22 dip plateau that re-settles to 0.885 between episodes.**
   The floor forces c=0 during every dip (c_max = 0.000 while G<0.2),
   so the c>0.5 leg of the stuck criterion is structurally unreachable
   on a floor-regulated run and the G<0.1 bar is the only falsifiable
   leg — and it is falsifiable: permanent capture (a_hold held forever)
   drives regulated G below 0.1 at a_hold ~ 1.5 (0.095 at 1.5, 0.089 at
   2.0; frozen 0.043-0.046 at all intensities), never reaching the
   frozen stuck level.  Denser sub-threshold pattern (dur 60, gap
   200 ×100; a single such episode does not collapse): frozen fails at
   EPISODE 2; regulated functional throughout.  Spaced sub-threshold
   (dur 40, gap 300 ×100): NEITHER fails — and the floor is exactly
   inert there (max|dG| = 0.00e+00): no false-positive cost.
5. **Healthy-regime cost (false positives)**: with no episode, floor 0.7
   changes nothing at all (max|dG| = 0.00e+00, c_max = 0.000 — E ≤ 1−G
   ≈ 0.12 « floor, so the clamp is never active); external rescue (G2c)
   is not blocked (0.8855 vs 0.8854).  Sub-collapse-intensity episodes
   (a_hold 0.4/0.5/0.6/0.8/0.9/1.0/1.1 single): regulated dips are
   SHALLOWER than frozen (0.22/0.18/0.16/0.14/0.13/0.13/0.12 vs frozen
   0.22/0.18/0.16/0.05/0.05/0.05/0.05) — the floor never suppresses a
   legitimate response; it only caps the runaway.  The elaborateness
   axis in the HEALTHY regime: GATED cost is free at ANY size
   (G=0.886 for c_mon up to 1.2 — trigger never fires); UNGATED
   (always-on vigilance) degrades G monotonically (0.53 at c_mon=0.1,
   0.12 at 0.8) — elaborate introspection harms even a healthy agent.

**Honest negatives / boundaries (do not tune these away).**

1. A GATED cost with NO actuator authority defeats even the floor
   (k=0, floor=0.7, c_mon≥0.5 → fails): the cost only self-extinguishes
   if the actuator actually lifts G above the trigger.  A regulator that
   checks and worries but never acts is worse than the floor alone.
2. At k_pull = 4.0 (≈ u_ext 1.0) gated cost never kills the
   post-collapse rescue up to c_mon = 1.5 — the monitoring-cost failure
   is a finite-authority phenomenon, not absolute.  The threshold
   c_mon_crit ≈ 0.5 is quoted at k = 2.0.
3. The contrast's sharpness is FORMULATION-DEPENDENT: gated → sharp
   threshold in the post-collapse assay (0.511); ungated constant
   vigilance → smooth degradation (c_mon_crit = 0.112 at both k=2 and
   k=4; and in the HEALTHY regime ungated c_mon ≥ 0.8 collapses the
   agent even with the floor on).  "Elaborate introspection is fatal"
   is demonstrated for sustained, inward-directed cost — and only in
   the post-collapse assay geometry for the gated form (see the scope
   note above: always armed, the gated threshold vanishes).
4. The floor does not protect against ALL failure modes: it severs the
   endogenous cannibalization loop but cannot release EXOGENOUSLY
   captured attention.  With a_hold applied and never released
   (T=4000), the floor-regulated system settles degraded at
   G = 0.21/0.16/0.11 at a_hold = 0.35/0.5/0.9 (frozen: stuck
   0.042-0.046) — better than collapse, far from functional; at
   a_hold ≥ 1.5 held forever even the regulated G crosses below 0.1
   (0.095/0.089 at 1.5/2.0).  The long-horizon claim holds for
   EPISODIC stress (holds that end), not permanent capture.  (Now a
   committed arm in exp6 part (e): `permanent_capture`.)
5. Floor 0.48 (just below critical) escapes only after 224 t.u. vs 22
   t.u. at ≥0.49 — near the boundary escape is slow, and a floor at
   exactly the wrong value buys a long limbo rather than rescue.
6. **Holding the floor while continuously re-checking it (UNGATED
   always-on cost, post-collapse) DEFEATS the floor** — the knowing-
   floor configuration: floor=0.7 + ungated c_mon =
   0.1/0.2/0.3/0.5 → G_end = 0.53/0.34/0.24/0.16 (vs 0.8855 at
   c_mon=0).  Continuous re-examination at c_mon ~ 0.2 costs the
   rescue — exactly matching the knowing-floor result kc ~ 0.2
   (handoff-selfreg-knowingfloor), measured here independently in the
   regulator battery (all part-a c_mon runs were gated; this cell was
   missing until review).  This STRENGTHENS the cheap-floor principle:
   the floor must not be re-examined — its value is precisely that it
   is held WITHOUT observation.

**Model-level caveats.** Deterministic model — every threshold is a
locus, not a distribution; demonstrated IN-MODEL only; the "false
belief" is a clamped constant, not a belief system; the monitoring cost
is a scalar inward-drive addend, a stand-in for whatever real
introspection costs; k_pull/c_mon are model units (the u_ext equivalent
k_pull/k_ext is an ASSERTED scale equivalence — a G-dependent
sigmoid-gated pull vs a constant external demand — not a derived
conversion, and cross-run number agreement with the earlier probe is
consistency in magnitude, not reproduction).  Success criterion is
functional (G stays above the floor / agent stays out of the stuck
attractor), not phenomenological.

**Falsifier.** In a stochastic or empirically grounded variant: a floor
set above the typical error level that nonetheless fails to prevent
entry into the stuck attractor under episodic stress; or an elaborate
high-bandwidth introspection mechanism that rescues at LOW cost — either
would break the cheap-floor design principle as stated.
