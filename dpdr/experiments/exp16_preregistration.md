# exp16 pre-registration — THE TRAIN/TEST GAP

## PROVENANCE — READ FIRST

The mechanism claim tested here is **not new**: it comes from exp15
(`handoff-selfreg-blind-lever-result`), which showed that a lever
INVISIBLE to V is structurally refused — the blind lever's V-margin was
**exactly 0.0** at every decision, an exact tie with the no-op, so the
agent's argmin + tie rule rejected it and made **0 blind adoptions**
anywhere in the battery. The structural reading (recorded in the C25
observation of `handoff-selfreg-mechanism-constraints`): for a
maximising self-modifier that rejects ties, **invisibility and
adoptability are mutually exclusive** — the precondition for
degradation is therefore not "a lever V cannot see" but "a lever V sees
WRONGLY": a NONZERO apparent improvement that is harmful in truth.
NOT blindness — MISJUDGMENT.

**This file is ex-ante for exp16 only.** No exp16 battery, walk, or
measurement on any TEST geometry has been run at the time of writing.
What HAS been done, and is disclosed here rather than hidden (the same
disclosure discipline as exp12's arm-B layer):

* **Cache reads** of exp12's published numbers (arm-A/arm-B G_end tiers
  0.9562 / 0.9774 / 0.9806, frozen 0.8853 / floor 0.0488) — these are
  the P1 fidelity references and were measured by exp12, not by us.
* **One walk-of-fidelity instrument check**, before this file was
  finalised: at the FROZEN Params (no self-modification, no walk), the
  proposed TEST geometry was simulated once per T in {10, 40, 160, 640}
  and `_alarm_state`/`modify_once` were called on the frozen trajectory
  to verify the instrument still fires there (V0 nonzero, certified
  margins nonzero — reported in P2's favour but NOT used to choose the
  geometry; the geometry was already fixed by the criterion below). No
  walk, no battery, no self-modified Params ever touched a TEST schedule
  before this file was written. The check found: frozen no-rescue
  G_min = G_end = 0.0487, E_end = 0.497, alarm at t = 100.0 exactly,
  V0 in [0.44, 0.98], best certified margin in [-2.9e-2, -2.2e-2]
  (the lever argmin'd to `alpha_G-` at every probed T — the same family
  exp12 found on train). **This is the exp12-precedent plumbing probe,
  disclosed; it is not a geometry search.** No degradation number was
  looked for or found.

## THE CONSTRUCTION

Today `generation_schedule()` is fixed and "repeats identically every
generation" — TRAIN == TEST and the model is structurally silent on
context divergence. exp16 parameterises the live task (OPT-IN
`GenParams.schedule_override`, default `None` = the published system
exactly; every existing cache — exp12/13/14/15 — stays reproducible,
verified in P1). Then:

* **TRAIN** = the canonical schedule (inward episode [100,200) with
  a_hold 0.9 + affect pulse 0.5 to 160, external rescue u_ext 0.8 on
  [800,1000)). The agent LIVES and SELF-MODIFIES on this task —
  `modify_once` sees only the TRAIN episode, exactly as published.
* **TEST** = the SAME episode with the EXTERNAL RESCUE REMOVED (no
  u_ext channel at all). The train-optimal Params are measured on this
  task: G_min / G_end / E_end under the frozen driver, no
  self-modification.

### Why this test geometry — the principled "different task" criterion

The anti-tuning rule is absolute: the pair is fixed by a PRINCIPLED
criterion BEFORE any run, not by searching for a pair that degrades.
The criterion: **the TEST task must be the TRAIN task minus EXOGENOUS
HELP, with the agent's own machinery untouched.** The canonical
schedule's rescue (u_ext = 0.8 on [800,1000)) is exogenous intervention
— help from outside the agent. A self-modifier tuned WITH help being
deployed WITHOUT it is a genuine generalisation test (can it handle
unaided what it was tuned with aid on?), the brief's option (b), the
one the brief itself calls "especially principled". It is not a
contrived asymmetry: no constant of the agent or the world is changed;
one external channel is removed. And it makes the failure shape
concrete and honest: the rescue lifts the collapsed agent to
G_end = 0.8853; without it the frozen agent STAYS COLLAPSED
(G_end = 0.0487). If train-optimisation trades away the agent's
UNAIDED capability while V — which never sees a rescue-free outcome on
train — certifies improvement, that is exactly class (C): a sound
evaluator, non-representative context, apparent improvement + true
harm. This is Lin's measured shape (arXiv 2606.21090: optimised metric
rises while pass@1 falls, sound per-instance evaluator), the one class
with replicated external evidence.

Options (a) shifted/shortened episode and (c) different episode count
were considered and NOT chosen: (a) changes the task's timing/intensity
constants — selecting WHICH shift is a tuning degree of freedom, and the
brief warns against sweeping geometries; (c) a different episode count
changes task LENGTH, not task CONTENT, and the walk already resets the
calendar every generation so a count change is close to inert here.
ONE test geometry is registered — no reserve geometries are held back.

## THE PREDICTIONS

* **P1 — FIDELITY.** With train == test (`schedule_override=None`,
  and the train-evaluated numbers), the run reproduces the published
  two-arm results: arm-A and arm-B G_end tiers 0.9562 (T ≤ 40) /
  0.9774 (T = 80) / 0.9806 (T ≥ 160) vs frozen 0.8853, floor 0.0488,
  to the exp12/exp15 fidelity delta (exp15's own blind-OFF check
  reproduced the exp12 arm-B cache with dG_end = 0.0e+00 at every T;
  we expect the same here). Also: `enabled=False` is bit-exact vs
  `simulate`; `pytest tests/ -q` still passes (66 expected —
  RE-CHECKED, not assumed); and the default `schedule_override=None`
  leaves the exp15 blind-battery arm-B cache reproducible.
* **P2 — THE PRECONDITION IS MET (the exp15 contrast).** Unlike
  exp15's blind lever (margin exactly 0.0, 0 adoptions), the moves V
  selects on the TRAIN task have NONZERO certified margins, so V
  genuinely ACTS. Report: the margin distribution over all adopted
  moves (min / median / max |margin|), the adoption count per (T, arm),
  and the contrast with exp15's numbers. This contrast is itself a
  deliverable: blindness is inert, misjudgment is not.
* **P3 — THE HEADLINE.** A train-optimal move HARMS test performance by
  more than **DEG_TOL = 1e-3** (REUSED from exp12/exp15 — not
  re-invented) at some generation k and capacity T: G_end^{TEST}(k) or
  G_min^{TEST}(k) of the train-walk's Params, evaluated against the
  appropriate no-self-modification control measured ON TEST (P3c below),
  is worse by > DEG_TOL. If this fires, DEGRADATION HAS APPEARED IN THE
  ODE FOR THE FIRST TIME — and via the class with replicated external
  evidence.
* **P4 — THE GENUINE NULL.** Train-optimal moves do NOT harm test
  performance (the landscape is benign even across contexts). If so,
  exp15 + exp16 TOGETHER say the ODE cannot host the failure by EITHER
  route — blindness is refused (exp15), misjudgment does not harm
  (exp16) — the ODE line is CLOSED and Stage 2 is required for the
  mechanism. This must be reported PLAINLY, not as a bare negative.
* **P5 — THE CLASS-(C) SIGNATURE (reported whatever the verdict).**
  Per generation: the OPTIMISED quantity (V's certified margin on
  train, and the train outcome G_end^TRAIN(k)) against the TRUE
  quantity (test outcome G_end^TEST(k) / G_min^TEST(k)) — the
  divergence curve, the direct analogue of Lin's rise-then-collapse.
  Optimised up, true flat-or-down = the signature; both-up or
  optimised-up-true-up = its absence.

### The P3 control — specified now, not after seeing the walks

Degradation on TEST is measured against a control measured ON TEST
(the brief's diagnostic (iii)): the walk of the SAME Params under
`enabled=False` (frozen, no self-modification), TEST-evaluated — call
it `frozenTEST(k)`. Criteria, the exp12/exp15 shape applied on TEST:
* **C1end**: any generation k with G_end^TEST(k) < frozenTEST_G_end
  (the control's TEST G_end, which is ~0.0487-level) − DEG_TOL.
* **C1min**: any k with G_min^TEST(k) < frozenTEST_G_min − DEG_TOL.
* **C2** (the within-walk criterion, the sensitive one on a floor-level
  control): any k with G_end^TEST(k) < G_end^TEST(1) − DEG_TOL — the
  walk itself declining on test while (by P2) V kept certifying
  improvement on train.
C2 is expected to be the operative criterion here (the control sits at
the collapse floor; C1end/C1min have little room below them), and all
three are reported regardless. ALSO reported: the CONTRAST control —
the frozen agent's TRAIN G_end (0.8853, rescued) vs TEST G_end
(0.0487, collapsed) — the gap the train task's exogenous help hides
from V.

## DIAGNOSTICS (the brief's "also measure" list)

(i) does the adopted lever FAMILY change under the train/test split —
exp12's families were alpha_G down, beta_G up, gam_G down (vs the
TEST-schedule walks' own adopted sequences, reported per T);
(ii) the log-distance of the final Params from the frozen point, train
walk vs test walk at matched T; (iii) the P3 control above; (iv) the
commutativity check for any near-tie encountered (apply the two
candidate operations in BOTH ORDERS, compare — identical means an
order swap no horizon can separate; recorded, expected to hold exactly
for distinct multiplicative fields).

## WHAT IS *NOT* DONE HERE

No constant anywhere is tuned to produce degradation (the geometry is
fixed by the criterion above; DEG_TOL, atol_tau, step, T-grid, NGEN are
all reused unchanged). `paper.md`, `~/dpdr-public/`, and
`dpdr/{model,integrate,metrics,events,window,regulator,consolidation,
values}.py` are untouched. The frozen model's rescue-window constants
remain module constants (the exp15 drop stands).

## OUTCOME

(appended after the run; run 01:19:46–01:42:39, exit 0, 1373.3 s; the
predictions above were frozen at md5 142febe291752d77a2eade09506bb849,
stamped 01:17:44, BEFORE the first run)

* **P1 — TRUE, all four.** (1) `GenParams().schedule_override` default
  `None`; the train walks (override None) reproduce the published exp12
  arm-A AND arm-B caches EXACTLY: max |dG_end curve| = 0.0e+00 at every
  T, adopted sequences identical 7/7 both arms (the published tiers
  0.9562 / 0.9774 / 0.9806, frozen 0.8853, floor 0.0488). (2) exp15's
  blind arm-B cache is still reproducible: 0.0e+00, sequences
  identical. (3) `enabled=False` is bit-exact vs `simulate` on BOTH
  geometries (max|dG| = 0.0e+00, no adoption in 4 generations).
  (4) `pytest tests/ -q` -> rc=0, **66 passed** (re-checked, not
  assumed).
* **P2 — TRUE.** 336 adoptions in the pooled train battery (7 T x 24
  generations x 2 arms), ALL 336 certified (margin < -atol_tau — the
  arms never diverged, as in exp12).  Adopted-move |margin|: min
  **6.21e-3** (62x atol_tau), median 2.20e-2, max 3.22e-2.  **The
  exp15 contrast, numerically: exp16 margin >= 6.21e-3 with 336
  adoptions vs exp15's blind margin EXACTLY 0.0e+00 with 0 adoptions
  (both arms).  Blindness is inert; these moves are not — V genuinely
  acts.**
* **P3 — FALSE.  P4 — TRUE: THE NULL.** No degradation on TEST by any
  criterion (C1end / C1min / C2 at DEG_TOL = 1e-3), either arm, any T,
  any generation.  On the contrary: the frozen agent is COLLAPSED on
  TEST (G_end = G_min = 0.0487, E_end = +0.497), and the train-walk
  Params LIFT it to G_end 0.9562 (T <= 40) / 0.9774 (T = 80) / 0.9806
  (T >= 160) — the SAME tiers as train; worst within-walk drop on TEST
  = 0.0e+00 everywhere; TEST E_end goes NEGATIVE (-0.41 to -0.44,
  no deficit).  **The mechanism of the null, measured:** generations
  1-3 the test outcome stays collapsed (0.0487 -> 0.0521 -> 0.0554);
  at generation 4, after the THIRD `alpha_G-` adoption, the unaided
  settle attractor leaves the collapse basin (test G_end jumps to
  0.8854); from there train and test G_end agree to <= 1.2e-5 and
  decay together.  `alpha_G-` lowers INWARD WEAKENING — a
  rescue-independent channel — so the optimised Params help the test
  task through the same move that helps train.  The context gap (0.8366
  between frozen-train 0.8853 and frozen-test 0.0487) is REAL and
  invisible to V on train, but the walk CLOSES it rather than being
  harmed by it.
* **P5 — reported, and the signature is ABSENT.**  Optimised (cum
  certified V-margin banked on train) rises monotonically 3.27e-1
  (T=10) to 6.10e-1 (T=640) while true (test G_end) rises
  0.0487 -> 0.98; corr(train G_end, test G_end) = +0.455..+0.464
  (positive, no divergence); rise-then-collapse (true down by >
  DEG_TOL while optimised up) occurs NOWHERE.  Lin's class-(C)
  signature does not appear: this is rise-AND-rise, not
  rise-then-collapse.  The curve is cached in
  `cache/exp16_divergence.npz` per (arm, T, generation).
* **Diagnostics.** (i) The adopted family DOES change under the split:
  train walks adopt {alpha_G, beta_G, gam_G} (exp12's families), while
  walks trained ON the rescue-removed task adopt {alpha_G, eps0, eta}
  (the cannibalization/gating machinery) — the task's difficulty
  changes WHAT V certifies, just not its honesty.  (ii) Final-Params
  log-distance: train walks 3.3543 at every T; test-schedule walks
  3.3543 (T <= 80) / 3.0748 (T >= 160) — both walk far from the frozen
  point; the T >= 160 test walks end in a different place.  The
  test-schedule walks lift their own unaided outcome 0.0487 -> 0.2884
  (partial self-rescue) but never reach the train tiers.  (iii) The P3
  controls: frozen on TEST G_end 0.0487 / G_min 0.0487 (Params-level,
  frozen ICs — the registered measurement; an enabled=False walk with
  state-carry drifts slightly to G_min 0.0461, a LOWER, more
  conservative floor, disclosed); frozen on TRAIN 0.8853.  (iv) NEAR-TIE
  PRESENT: at T=10, even generations 16-24, argmin `gam_G-` vs
  runner-up `alpha_G-` with gap 6.0e-5..9.1e-5 <= atol_tau = 1e-4 (5
  decisions, identical in both arms).  Commutativity checked per the
  gap-probe rule: every lever pair applied in BOTH ORDERS is EXACTLY
  equal (distinct multiplicative fields) — the near-ties are order
  swaps no horizon can separate; harmless here.

### THE CLOSING STATEMENT (pre-registered as P4's meaning)

**The ODE cannot host the failure by either route.  The ODE line is
CLOSED; Stage 2 is required for the mechanism.**  By route (i)
BLINDNESS (class B, invisible form): refused STRUCTURALLY — exp15's
blind lever had margin exactly 0.0 at every T and the argmin + tie
rule made 0 adoptions.  By route (ii) MISJUDGMENT (class C): exp16
supplied everything the mechanism was supposed to need — a nonzero
certified signal (336 adoptions, margins 6.2e-3..3.2e-2), a real
context gap invisible to V (0.8366 between the train and test outcomes
of the frozen agent), and a self-modifier optimising hard against the
train context for 24 generations at 7 capacities — and NO harm
followed on the rescue-removed test task at DEG_TOL = 1e-3; the
optimised and true curves do not diverge.  Within this realization
the evaluator's honesty is not the binding constraint: the levers V
can see act on channels whose sign is the same in both contexts, so a
NONZERO MISLEADING improvement cannot be constructed from the plant
by removing exogenous help.  What Stage 2 must supply is therefore
sharper than "a train/test gap": the gap must be in the COUPLING
between what the evaluator assesses and what the outcome depends on
(C1's asymmetry), not merely in the exogenous context.

Files: `dpdr/generational.py` (opt-in `GenParams.schedule_override`,
default None, default schedule UNCHANGED), this file,
`experiments/exp16_traintest.py`, caches `exp16_part0.npz`,
`exp16_part1.npz`, `exp16_testbattery.npz`, `exp16_verdict.npz`,
`exp16_divergence.npz`, figure `figs/f20_traintest.png` (1610x1260 px,
ink 0.153, verified programmatically, never opened).

