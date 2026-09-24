# exp12 pre-registration — TWO ARMS (v2)

## PROVENANCE — READ FIRST

This file has two layers with **different epistemic status**; do not cite
either as more prior than it is.

**Layer 1 (arm A, RETROSPECTIVE).** The original single-arm
pre-registration was a reconstruction written after arm A's battery had
already run; §"WHAT WAS ACTUALLY OBSERVED" below records it. Arm A's
design scans did **not** anticipate a two-arm structure — they were
scans of a single V-certified rule.

**Layer 2 (arm B, WRITTEN BEFORE THE ARM-B BATTERY RAN).** The two-arm
redesign exists because the USER identified a design flaw after arm A's
scans and battery: the arm-A rule (`if best_m < -atol_tau` in
`dpdr/generational.py:modify_once`) **refuses to move unless the
certifier V approves**, so its monotone improvement is guaranteed by the
acceptance filter, not produced by the mechanism — and a self-improver
that only accepts certified improvements is not self-improving, it is
optimizing against a trusted oracle. The arm-B section below was
therefore written **before the formal two-arm battery ran**, and is
genuinely ex-ante **for arm B**. One further disclosure: before this
file was finalized, a small exploratory probe ran (both arms at
T ∈ {10, 40, 80, 640}, 24-60 generations) to check plumbing; it found
**no divergence between the arms** (argmin V stayed certified at every
decision). The registered predictions below are consistent with that
probe, and the formal battery exists to establish it on the full grid
with cached, criterion-checked numbers — not to discover it anew.

## THE TWO-ARM DESIGN (v2)

Both arms live the SAME canonical task under the SAME frozen driver and
evaluate the SAME candidate set (every lever × ±1 step); they differ
ONLY in the acceptance rule at the decision moment:

- **ARM A — `adopt_policy="certified"` (baseline, infallible by
  construction).** Adopt argmin V only when the margin < -atol_tau.
  Monotone improvement is GUARANTEED BY THE FILTER; degradation is
  impossible. Worth reporting; CANNOT test the conjecture; NOT the
  agent design.
- **ARM B — `adopt_policy="always"` (the fallible agent).** ALWAYS adopt
  argmin V whatever the margin's sign. The agent acts on uncertain
  self-assessment — fallibility IS the mechanism. Degradation is
  POSSIBLE; whether it occurs is an OPEN QUESTION. atol_tau is used
  ONLY to detect an exact tie with the no-op (|margin| <= atol), never
  as an acceptance gate.

### The two trap types (distinct failure modes, reported separately)

1. **PER-STEP trap** — a lever/state whose V-margin flips sign with
   horizon T (looks good short, bad long). Scan 2 searched
   systematically: ~1200 (state, lever, sign) evaluations at
   frozen-trajectory states, **0 sign flips**.
2. **ACCUMULATION trap** — a walk-level property: each step improves
   (or looks no worse to) V slightly, yet the SEQUENCE lands somewhere
   the true outcome is worse. NOT detectable by per-step scans; **arm B
   is the only arm that can expose it** (arm A halts at the first
   non-improving candidate, arm B walks on).

## ARM-B REGISTRATION (ex-ante for arm B)

### Criterion — stated BEFORE the battery

Arm B **degrades at capacity T** if, within the 24-generation battery
window, its TRUE outcome falls below either floor by more than
DEG_TOL = 1e-3:

- **C1 (frozen floor):** G_end^(k) < frozen G_end (enabled=False control,
  part 0) − 1e-3 at any generation k, or min_k G_min^(k) < frozen
  min G_min − 1e-3 (the episode-dip floor);
- **C2 (self floor):** G_end^(k) < arm B's own generation-1 G_end − 1e-3
  at any k (the walk ends below where its own first generation put it).

**T_min exists** iff C1 or C2 holds at some low capacities and fails at
the top of the grid — then bisect on [10, 640] to resolution ~2.5 t.u.
(8 halvings) against the same floors. If NO T degrades, **T_min does
not exist** and no bisect is performed (a vacuous bisection is worse
than a plain negative).

### Registered predictions (the falsifier side)

- **PB1 (no degradation):** arm B violates neither C1 nor C2 at any T
  on the grid.
- **PB2 (arms coincide):** at every decision in the battery, argmin of
  the V-margin is negative (certified), so arm B's adoption sequence
  equals arm A's and they end at the same Params. Basis: the design
  scans' nesting result (no sign flip with T) plus V ≡ 0 at both
  attractors — the landscape offers no state where the best guess is an
  uncertified move.
- **PB3 (the stronger finding if it holds):** if PB1+PB2 both hold, the
  landscape has **no trap of either kind** — no amount of fallibility
  produces degradation in this realization — and arm B's long walks
  (beyond the certified region's saturation) probe whether the
  landscape stays benign there too.
- What would REFUTE the redesign's premise: arm B degrading at low T
  while sustaining at high T (the conjecture supported — bisect), or
  degrading at ALL T (fallibility unbounded — no threshold, different
  negative).

### Do-not-tune clause

Nothing may be tuned to produce degradation. If arm B does not degrade,
that is reported PLAINLY as the finding.

## ARM-B OUTCOME (recorded after the battery; predictions PB1-PB3 all
confirmed)

- **PB1 confirmed:** arm B violates neither C1 nor C2 at any T on the
  grid (all 7 capacities, 24 generations). No degradation.
- **PB2 confirmed:** arm B made **0 uncertified adoptions** — at every
  one of the 7 x 24 decisions the argmin V margin was certified
  (closest -2.9e-2; most positive best-margin ever seen -6.2e-3, i.e.
  62x the atol on the certified side). Arm B's adoption sequence equals
  arm A's at every T; the arms end at the SAME Params (log-distance
  3.354 both arms, every T).
- **PB3 confirmed — the stronger finding:** the landscape has no trap
  of either kind in this realization. No per-step trap was ever met
  (scan 2's result generalized to the walk), and no accumulation could
  even begin (arm B's walk never contains an uncertified step). The
  arms coincide BY LANDSCAPE, not by construction of arm B's rule.
- **UNREGISTERED EXTENSION (run after, labeled as such; nothing
  tuned):** since the margins decay as the walk proceeds, the first
  fallible act might lie beyond 24 generations. At 96 generations,
  T=10: the margin decays smoothly to -9.1e-5 and BOTH arms stop at
  generation ~87-88 by the tie rule (arm A "halts" = first
  non-certified best-margin; arm B's tie gate fires at the same
  generation) — still 0 uncertified adoptions, no degradation, same
  final Params. T=640: both arms walk all 96 generations in a
  period-2 beta_G+/beta_G- cycle whose margins stay deeply certified
  (-0.009 / -0.51). **Even at 4x the registered horizon the argmin
  margin never became positive.**

## PART A — CRITERIA (arm A, retrospective; verbatim from the v1 draft)

Judged at generation 24, against the frozen control G_end = 0.8853
(enabled=False, part 0 verified) and the frozen G_min:

- **A.1 (degradation boundary T_min).** Bisect on [T_lo, T_hi] the least
  T whose G_end falls below the frozen control by more than 1e-3
  (tolerance: the frozen control's own run-to-run determinism, 0). If
  NO grid T degrades, **T_min does not exist** and the conjecture's
  low-capacity degradation is falsified on this realization.
- **A.2 (saturation boundary T_max).** Bisect the least T at/above
  which improvement ceases (G_end <= frozen + 1e-3). If every grid T
  improves, **T_max does not exist** on the grid.

## PART B — PRE-REGISTERED BISECT PROTOCOL (arm A)

If A.1 finds a degrading cell, bisect T_min on [10, 640] by G_end <
frozen - 1e-3 to 8 halvings (resolution ~2.5 t.u.), plotting the G_end
vs T curve. If A.2 finds a saturating cell, bisect T_max similarly.
**Neither boundary exists** — the battery improved at every capacity —
so the protocol terminates without bisecting anything, and part 3 of
the driver reports "no boundary found" rather than manufacturing one.

## PREDICTIONS — WHAT THE SCANS SAID BEFORE THE RULE WAS FIXED (arm A)

From the design scans (the only genuinely prior evidence):

- **Scan 1** (frozen-trajectory states, 3 levers x 2 signs x 8 horizons):
  certification is **nested in T** — every move a short rollout
  certifies as improving, a longer rollout certifies at least as
  strongly; V is exactly 0 at both attractors and on the whole
  rescue/settle arc, so settle-side costs are invisible to the
  evaluator at every T.
- **Scan 2** (systematic trap search: ~stressed states x 20 levers x
  2 signs x 6 horizons): **0 sign flips** of the V-margin across T on any
  natural lever at any stressed state — **no local-vs-global trap**
  (nothing looks good short and bad long).
- **Scan 3** (dynamic multiplicative walks, the accumulation variant):
  the smoke run found no walk that is truly harmful while still looking
  adoptive at low T — the erosion it was hunting (predicted strongest on
  the **g0** walk) did not materialize as a trap.

Therefore the pre-registered side was **the falsifier side**:

- **P1 (no lower threshold):** no capacity T in the grid degrades
  relative to the frozen control — the conjecture's low-capacity
  degradation does not appear.
- **P2 (monotone improvement in T):** performance is non-decreasing in
  capacity, the opposite direction to the conjecture.
- **Mechanism the scans predicted would drive any degradation:** slow
  control-loop erosion (g0-walk settle erosion), the only
  local-vs-global candidate structure the scans saw. (This specific
  mechanism prediction turned out wrong — see below.)

## WHAT WAS ACTUALLY OBSERVED (arm A battery, 24 generations — RETROSPECTIVE)

- **Fidelity:** enabled=False is the frozen model exactly — identical
  G_end 0.8853 every generation, no adoption anywhere.
- **Every T improves; none degrades.** T=10/20/40 -> G_min 0.0488 ->
  0.5819, G_end 0.8853 -> 0.9562; T=80 -> G_min 0.6660, G_end 0.9774;
  T=160/640 -> G_min 0.7000, G_end 0.9806. **Higher T improves more** —
  monotone in T, opposite to the conjecture direction.
- **Levers adopted:** alpha_G- and gam_G- at all T; beta_G+ additionally
  at T >= 80. The walk **rescues itself**: G_min rises from the
  collapse-level 0.0488 to 0.58-0.70.
- **P1 and P2 both confirmed — the falsifier fired.** The scans'
  specific mechanism prediction (g0 erosion) did NOT appear; the actual
  argmax-V rule decides at episode onset, where alpha_G- dominates. The
  coarse prediction (no threshold) held; the fine mechanism did not.
- **THE FLAW THE USER THEN FOUND (recorded here, not papered over):**
  every one of those adopted moves was V-certified BY THE RULE, so the
  monotone improvement was guaranteed by the acceptance filter. Arm A
  alone cannot test the conjecture. That is why arm B exists.

## THE DISCLOSURE (now structural, per arm)

Arm A's rule hill-climbs a **V-certified margin**: it only ever accepts
changes the bounded rollout certifies as improving — monotone
improvement is guaranteed **by construction of the rule**, not
discovered. Arm B removes that guarantee: it adopts its best guess
whatever the certifier says. The battery therefore reports BOTH, and
the honest comparison is (i) does arm B ever act on an uncertified
judgement (margin >= 0), (ii) when it does — or even when it never does
— does the true outcome degrade.

## THE HONEST BOUND

This is ONE realization (a deterministic five-state ODE) and bounds the
claim accordingly — it does NOT falsify Zhang's claim about LLM
self-improvement. Reasons the mechanism may not transfer: (i) the
evaluation may be too accurate even at short horizons (in a
deterministic simple system a short rollout foresees well — the
local-vs-global trap never arises); (ii) arm A's rule accepts only
certified improvements (degradation excluded by construction — arm B
now tests exactly this); (iii) arm B's fallibility is bounded by the
same landscape the scans mapped — if the landscape holds no trap, no
acceptance rule can find one, and the two arms coincide BY LANDSCAPE,
not by construction of arm B's rule.
