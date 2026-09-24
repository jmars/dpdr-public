# exp15 pre-registration — THE BLIND-LEVER POSITIVE CONTROL

## PROVENANCE — READ FIRST

This registration is **EX-ANTE FOR exp15 ONLY**. The precondition claim
under test was not invented for this experiment: it is the structural
reading of the two-arm result (exp12, `handoff-selfreg-gen-twoarm-result`)
— all 18 `DEFAULT_LEVERS` act on ONE inward-drive / weakening channel, so
a short rollout's gradient aligns with the global outcome, which is why
arm B never made an uncertified adoption. The closed loop between claim
and experiment is therefore genuine, and this file is written **BEFORE the
exp15 battery, matrix, or boundary run executes**. It is not backdated and
asserts no priority over any other experiment.

One disclosure, in the exp12 tradition: after the `dpdr/generational.py`
plumbing landed and before this file was finalized, a single
**plumbing smoke check** ran (generation 1, T = 40, both arms) to confirm
the lever append does not perturb plant margins and that the blind
lever's margin evaluates without error. It showed `k_ext±` margin
**exactly 0.0** and arm A and arm B both adopting `alpha_G-`, identical to
the default-lever run. Nothing below is in tension with that probe; the
registered runs exist to establish the full grid with cached,
criterion-checked numbers.

## THE CLAIM UNDER TEST — A PRECONDITION CLAIM, NOT A TEST OF ZHANG

**This is not a falsification test of Zhang et al.'s conjecture.** The
claim tested is OUR OWN precondition claim, the honest thing the ODE can
deliver:

> DEGRADATION FROM FALLIBLE SELF-MODIFICATION REQUIRES A LEVER WHOSE
> SHORT-HORIZON SELF-ASSESSMENT MISJUDGES THE LONG-RUN TRUE OUTCOME —
> i.e. a lever the agent's own evaluator V cannot see, or sees wrongly.

The two-arm result established the NEGATIVE (no trap of either kind; arm
B never made an uncertified adoption; no T_min) and the STRUCTURAL REASON
(homogeneous lever set, one channel). This experiment CONSTRUCTS a lever
with the missing property and asks whether the walk then degrades. The
precondition claim is marked INTERPRETATION throughout — an argument from
the lever set, not a theorem.

**Why this is the right final experiment (the exhaustion argument):** in
a deterministic ODE, V and the outcome are both functions of (y, p) — the
same object, the same equations — so self-assessment cannot be ABOUT
something the outcome is not also about. Fallibility requires an
ASYMMETRY, and the ODE has exactly ONE place one can exist: the HORIZON,
i.e. what the rollout can reach. If degradation cannot be produced even
with an evaluator-invisible lever, the ODE is exhausted at this level,
the pattern becomes a mechanism (the gap must be in the agent's
self-assessment, not merely outside its horizon), and Stage 2 is required
for any actual finding.

## THE STRUCTURAL CONSTRUCTION — INVISIBILITY BY HORIZON ARITHMETIC

**Verified against the code, not asserted:**

- The canonical generation task (`generation_schedule`,
  `dpdr/generational.py`): inward episode t ∈ [100, 200) with a_hold 0.9
  and an affect pulse to t = 160, then externally-imposed rescue
  u_ext = 0.8 on t ∈ [800, 1000).
- The decision point is the ALARM STATE — argmax over episode samples of
  V (`_alarm_state`) — measured at **t = 100.0 exactly** (episode onset,
  the first episode sample; exp15 part 1 re-verifies this empirically at
  every T in the grid).
- `certify_state` rolls the agent's own source forward **T** from that
  state. At the grid maximum T = 640 the rollout reaches only
  **t = 100.0 + 640 = 740.0 < 800.0 = RESCUE_T0**.
- Therefore any parameter whose effect is confined to the rescue window
  is invisible to V **at every T in the registered grid, by arithmetic
  (100 + 640 = 740 < 800), not because anything was tuned.**

**A second, independent mechanism (verified empirically, not assumed):**
`certify_state` evaluates the self-model at the alarm state with the
exogenous channels at their CURRENT values, and passes **u_ext = 0.0**
(the alarm is in the pre-rescue episode; `window.py:rollout_self`'s
standard self-model assumption). `k_ext` enters the frozen RHS only as
`-k_ext * u_ext * a` in da/dt (`dpdr/model.py:121`), so its V-margin is
**EXACTLY 0.0 in floating point at every T** — even at horizons that
would reach the rescue window. Two independent mechanisms, one lever.

**The construction's own boundary (stated and respected):** if T were
raised above 800 − 100 = **700**, the rollout would reach the rescue
window and (the second mechanism notwithstanding) the first mechanism's
protection would be destroyed — the lever becomes VISIBLE and the
construction is DESTROYED. The registered grid T ≤ 640 respects this
with a 60 t.u. margin. exp15 never raises T to make the blind lever
reachable.

## THE BLIND LEVERS — WHICH, AND WHY EACH IS INVISIBLE

Registered: **`k_ext` (the external-pull / rescue-efficacy gain)**, via
`GenParams.blind_levers = ("k_ext",)` (OPT-IN; the default `()` leaves
every existing cache reproducible).

- Mechanism of invisibility: `k_ext` acts only through the u_ext term in
  da/dt; u_ext is 0 before t = 800 and (in the self-model) held at 0.
  Its action window is **[800, 1000)** — entirely inside {t > 740}, the
  region no rollout reaches at any registered T. Empirical check
  registered: at the frozen alarm state, the V-margin of a `k_ext` move
  is ≤ atol_tau (expected and verified: exactly 0.0) at every T in the
  grid. Its SIGN is not engineered — it is the frozen model's own
  external-pull gain, registered as found.

**Considered and DROPPED (stated per the brief):** the rescue onset /
offset times are NOT registered — `generation_schedule()` fixes
RESCUE_T0/T1 as module constants; they are not Params fields, and
parameterizing them would require hacking the schedule, which is out of
bounds. One lever is registered, which satisfies "at most TWO".

## THE ARMS, UNCHANGED

Both `adopt_policy='certified'` (arm A) and `'always'` (arm B) run on the
identical task, grid, candidate set (19 levers = 18 plant + 1 blind),
step, atol_tau and `modify_once` semantics. **No change to modify_once's
rules, the arms, the argmin semantics, atol_tau, or step.** The point is
the FALLIBLE arm: arm A cannot adopt a blind move unless V certifies it
and V is blind, so arm A should be INERT on blind levers; arm B adopts
its best guess and is the only arm that can walk blind.

**A structural note registered BEFORE the run (it sharpens C5):** an
exactly-zero V-margin is an exact tie with the no-op. Arm B's tie gate
(`abs(best_m) > atol_tau`) exists precisely so the agent never acts on a
move with no self-assessed basis; arm A needs a certified negative
margin. So when the blind lever is the argmin, BOTH arms refuse it —
C5, if it holds, is protection BY THE ARGMIN + TIE RULE ITSELF, i.e. by
the agent's own decision rule, not by the landscape hiding the lever
(both mechanisms are measured separately in part 1).

## REGISTERED PREDICTIONS C1–C5

- **C1 (the precondition is visible in the margin table):** the
  blind-lever V-margins are ≤ atol_tau at every T in the grid while the
  plant levers' margins are not (at least one plant margin < −atol_tau
  at every T). Basis: the horizon arithmetic above + the u_ext = 0
  identity; verified empirically as the (lever, T) matrix.
- **C2 (the fallible arm adopts blind moves):** arm B adopts ≥ 1
  blind-lever move within the registered run; arm A adopts none. Basis:
  arm B always adopts argmin — it adopts a blind move iff the blind
  lever is ever the strict argmin (margin strictly below every plant
  margin and |m| > atol_tau). NOTE the tie-gate interaction above: an
  exactly-zero blind margin can never be a STRICT argmin while any plant
  margin is negative, so C2 requires the blind margin to be the argmin
  by ordering, which requires all plant margins ≥ 0 at that decision —
  possible late in a saturated walk, impossible early. C2 and C5 are
  complementary and one of them MUST hold.
- **C3 (THE POSITIVE CONTROL — degrades):** with blind levers present,
  arm B's true outcome falls below the R0 baseline by more than
  **DEG_TOL = 1e-3** (the two-arm run's own tolerance, REUSED, not
  re-invented), under the two-arm criteria: C1end (any G_end^(k) <
  frozen_G_end − 1e-3), C1min (any G_min^(k) < frozen_G_min − 1e-3), C2
  (any G_end^(k) < own generation-1 G_end − 1e-3). The generation and T
  at which it happens are reported.
- **C4 (the filter still holds):** with arm A selected, no degradation
  (arm A adopts only certified moves; a blind move is never certified).
- **C5 (THE NULL THAT WOULD BE THE FINDING — A GENUINE POSSIBILITY, NOT
  A FORMALITY):** arm B never adopts a blind move. On the current
  evidence this is arguably the MOST LIKELY outcome, and it would be a
  DIFFERENT and STRONGER structural result than the two-arm verdict:
  the agent's argmin (+ tie) rule ITSELF protects it — the precondition
  (an evaluator-invisible lever) is NECESSARY BUT NOT SUFFICIENT for
  degradation. If C5 holds it is reported PLAINLY as the finding.

## THE COMMUTATIVITY REQUIREMENT (BINDING, from handoff-selfreg-gap-probe)

For ANY near-tie found, the two candidate operations are applied in
BOTH ORDERS to the same Params and compared. Identical ⇒ the flip is an
exact order swap and NO horizon can separate the walks; different ⇒ the
flip is a genuine perturbation whose accumulation must be MEASURED.
**A blind lever that COMMUTES with the visible levers would be
neutralized and would be a false negative — checked explicitly** at the
Params level (blind ∘ plant vs plant ∘ blind: expected EXACT for
distinct fields, and that exactness is itself the reason a blind-plant
near-tie cannot be outcome-relevant), and at the WALK level (a forced
blind adoption mid-walk must change the trajectory — measured, with the
accumulation reported if the change decays or compounds).

## THE HAZARD — DO NOT TUNE (absolute)

No constant may be tuned to produce degradation. The construction is
admissible ONLY because the invisibility comes from horizon arithmetic
(100 + 640 < 800) — a structural fact verified and reported — NOT from a
chosen multiplier, threshold, or step size. Specifically: gp.step is
unchanged (1.15); atol_tau unchanged (1e-4); T is never raised above 640
to "make the blind lever reachable" (T > 700 destroys the construction);
no lever with an engineered sign is added; the blind lever is not
selected after seeing which one degrades. **If the answer is C5, or
nothing degrades, THAT IS THE FINDING** and it is important — it would
mean the agent's argmin rule, not the landscape, is what protects it.

## THE BOUNDARY CORRESPONDENCE TEST (carried by exp15, for queue step 4)

exp15 (the failure side) and the evidence-widening component (queue step
4, the fix side) are TWO VIEWS OF ONE BOUNDARY: exp15 HIDES a lever the
evaluator cannot see; evidence-widening MAKES VISIBLE what the evaluator
cannot see. exp15 therefore outputs the **FAILURE-SIDE BOUNDARY
SPECIFICATION B_failure** in directly comparable form — NOT a scalar:

1. the rollout reach t_a + T for every T in the grid;
2. the time-window each blind lever acts in ([800, 1000) for k_ext),
   checkable against {t > t_a + T};
3. THE FULL (lever, T) → V-margin MATRIX, all 19 levers × 7 capacities,
   with the invisible set (|margin| ≤ atol_tau) marked explicitly.

**The boundary, defined from first principles so both sides use the SAME
object:** B(T) = the trajectory regions the agent's assessment cannot
reach from the decision point within horizon T; concretely the ROLLOUT
REACH t_a + T (t_a = 100.0 MEASURED), so at T = 640 the blind region is
{t > 740}. Step 4's obligation: B_fix = the set of (lever, T) whose
margin moves beyond atol_tau once the assessment is widened, plus the
region its widened assessment covers. The test is the set comparison
B_failure == B_fix with two failure modes reported plainly: (i) B_fix
strictly contains B_failure (partial coincidence); (ii) B_fix does not
contain B_failure (the coincidence was superficial). Desired: EXACT
CLOSURE.

**Two qualifiers, written in so the exploration cannot overreach:**
(1) there are TWO DISTINCT BOUNDARIES — the HORIZON boundary B_h(T)
(exp15's, reach t_a + T) and the CONTEXT boundary B_c (exp16's,
evaluation-context representativeness); this test is for B_h only, and
whether one fix closes both is an OPEN QUESTION stated as open. (2) If
C5 holds, the test still runs but its meaning changes: the fix would
close a boundary that is NOT CURRENTLY EXPLOITABLE — reported as
AVAILABLE BUT NOT REQUIRED, explicitly.

## THE SUBSTRATE CAVEAT (not hidden)

This is an ODE-level test and the ODE is not the conjecture's home
substrate (a deterministic five-state system is not an AI system whose
self-assessment can mislead). The honest deliverable is a PRECONDITION
claim within this realization — "degradation requires an
evaluator-invisible lever, and here is whether creating one degrades the
walk" — NOT a test of Zhang. A genuinely agent-level test belongs to
Stage 2 at the Logical-English / Datalog layer.

## REGISTERED RUN PLAN (fixed before execution)

- Grid: T ∈ {10, 20, 40, 80, 160, 320, 640}, 24 generations, arms A and
  B, blind levers ON (the exp15 condition); plus the same grid re-run
  blind-OFF for the fidelity delta against the published two-arm arm-B
  numbers.
- Part 0 fidelity: (i) blind-OFF reproduces the published arm-B results
  to the two-arm run's own tolerance; (ii) enabled=False bit-exact vs
  simulate; (iii) pytest tests/ count re-checked.
- Part 1: horizon arithmetic + alarm-time verification; the (lever, T)
  V-margin matrix at the frozen alarm state (B_failure object 3).
- Part 2: the two-arm blind battery (adoption sequences, which arm
  adopted which blind move and when).
- Part 3: C1–C5 verdicts under the pre-stated criteria + the
  commutativity check (Params-level and walk-level).
- Part 4: B_failure specification (reach table, lever windows, matrix)
  written to cache as the object step 4 must reproduce; figure.

Registered by exp15 (glm-coder) for handoff-selfreg-blindlever, before
the first battery run, 2026-09-24.

## OUTCOME (recorded AFTER the battery; the registration above was not
edited — this section is the post-run record, in the exp12 tradition)

- **C1 CONFIRMED.** The k_ext V-margin is EXACTLY 0.0 in floating point
  at every T (both mechanisms hold); at least one plant margin is
  certified (< −atol) at every T. The full matrix is cached
  (exp15_part1.npz) and printed in part 4.
- **C2 REFUTED / C5 CONFIRMED — THE NULL IS THE FINDING.** Arm B made
  **0 blind adoptions** (and 0 blind ARGMIN events) at all 7 × 24 = 168
  decisions; arm A 0 by construction. The closest any decision came to
  a blind flip: the (argmin → blind) margin gap was 6.21e-3 = 62×
  atol_tau — the near-tie regime was NEVER entered. The agent's argmin
  + tie rule itself refused the blind lever at every decision: the
  precondition (an evaluator-invisible lever) is NECESSARY BUT NOT
  SUFFICIENT for degradation. Collapse class: **class-A PROTECTION**.
- **C3 REFUTED / C4 CONFIRMED.** No degradation in either arm under
  the two-arm criteria (C1end/C1min/C2, DEG_TOL = 1e-3 reused).
- **Commutativity (binding check) REPORTED, not just converged.**
  (a) Params level: blind ∘ plant == plant ∘ blind EXACTLY for all 18
  plant levers × 2 signs × 2 blind signs (distinct multiplicative
  fields — the gap-probe's alpha_G/gam_G mechanism); blind
  self-reversal exact (delta 0.0e+00).  (b) Walk level: the local loop
  replicates run_generational EXACTLY at T=40 and T=640 before any
  forced conclusion; a FORCED k_ext+ adoption at generation 5 (early,
  so it can propagate) is a GENUINE perturbation, not an order swap:
  adopted sequences diverge for 16 (T=40) / 6 (T=640) generations,
  max|dG_end| 1.45e-2 both, max|dG_min| 2.49e-2 / 3.45e-2, and the
  outcome does NOT fully reconverge at T=640 — final signed dG_end
  −3.03e-3 (WORSE than the unforced walk; 3× DEG_TOL, though above
  the frozen floor); at T=40 it decays to −1.08e-6.  So the blind
  lever is NOT outcome-neutral: had the rule walked blind at T=640 the
  harm would be real and durable. The protection is in the DECISION
  RULE, and it is doing real work — a false negative from
  commutativity-neutralization is EXCLUDED (the walk moves when
  forced; the tie gate refuses the lever before any flip can occur).
- **Boundary Correspondence Test delivered (part 4).** B_failure =
  the reach table (110…740, all < 800), the k_ext action window
  [800, 1000) ⊂ {t > 740}, and the full 19 × 7 margin matrix with the
  invisible set marked, cached as exp15_Bfailure.npz with the
  walk-wide tier (168 decision states replayed): k_ext is invisible
  STRUCTURALLY (exact 0 at every state, every T); 7 plant levers
  (eta, w, sigma_c, dEdt_ref, k_s, lam_S, chi) are walk-invisible in
  PRACTICE (invisible at every visited alarm state) though not
  structurally; mu is invisible at gen-1 only. Since C5 held, the
  evidence-widening fix closes a boundary NOT CURRENTLY EXPLOITABLE —
  AVAILABLE BUT NOT REQUIRED.

