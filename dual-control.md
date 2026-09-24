# Dual Control Theory: Primary-Source Read, and What It Anticipates

**Task.** The paper (self-regulating agents; DPDR) rests on three claims that a search pass flagged as
possibly anticipated by dual control theory (Feldbaum 1960–65): (1) the sensing channel is the failure
channel; (2) a capability–recoverability tradeoff, r·φ ≈ 0.2 (viable-assumption set shrinks
hyperbolically with reasoning strength); (3) the "floor" — a cheap fixed, never-re-derived constant as
the escape. This document states what dual control actually formalizes (from primary sources, not
search summaries) and gives an honest verdict per claim.

**Method.** Two full survey treatments were read end-to-end (Mesbah 2018, 11 pp.; Meijer & Rantzer
2026, Annu. Rev. Control Robot. Auton. Syst. 10 — the canonical modern review), plus primary/
secondary sources as listed in the Appendix. Feldbaum's originals are paywalled/archival; his
formulation is recovered via the two surveys that quote it directly, the English-translation
citation record, and Rantzer's own lecture notes (Rantzer is a direct intellectual heir in this line).
No PNG was opened; no API keys echoed; no filesystem-wide scans.

---

## 1. What exactly does dual control formalize? (Q1)

**The setting.** A stochastic system with *unknown parameters* is controlled under feedback while the
controller simultaneously *estimates* those parameters from the same closed-loop data. Mesbah (2018):
x_{k+1} = f(x_k, u_k, w_k, θ), y_k = h(x_k, v_k), with θ ∈ R^{nθ} unknown. The controller acts on a
belief, not a state: the **hyperstate** ξ_{k|k} := P[x_k | I_k] — "the conditional probability of
states x_k given" the information vector I_k = [y_k,…,y_0, u_{k−1},…,u_0] — propagated by recursive
Bayesian estimation. Meijer & Rantzer (2026) note Feldbaum "addressed the dual control problem using
Bellman's work on dynamic programming, by combining the physical state with an information state."
(Their minimax reformulation literalizes this: the empirical covariance Z(t) of past data is appended
to the state — "With Feldbaum's nomenclature, Z(t) can be viewed as the information state,
complementing the physical state x(t).")

**The objective.** Minimize expected additive stage cost over a horizon under causal feedback:

  J_N(ξ_{k|k}, π) := E_{x_k}[ l_N(x_{k+N}) + Σ_{i=0}^{N−1} l(x_{k+i}, π_{k+i}) ],

solved (in principle) by the Bellman equation for stochastic dynamic programming. There is **no
separate exploration budget and no explicit information-gathering term in the cost**. The
explore/exploit tension is *intrinsic*: because today's input shapes tomorrow's measurement, which
reshapes the hyperstate, the optimal cost-to-go depends on the future hyperstate, so the optimal
controller automatically weighs degrading present performance (probing) against improving future
performance (learning). Mesbah: "Dual control maintains an optimal balance (in the sense of the
principle of optimality) between the probing activity and control activity of control inputs, which
are naturally in conflict. This arises from systematically accounting for the possibility of poor
transient control performance due to probing in order to achieve improved control performance in
future because of reduced system uncertainty."

**The dual effect (the formal core).** Feldbaum's insight: one and the same input has two effects.
Mesbah (quoting the standard gloss): control inputs "must have a probing effect for active learning of
system uncertainty and a directing effect for controlling the system dynamics." The formal definition
is due to Bar-Shalom & Tse (1974), via Mesbah, Definition 2:

> "A control input is said to have dual control effect if it can affect, with nonzero probability, at
> least one rth-order central moment of a state variable (r ≥ 2)."

i.e. the input can change not just the expected state but the *uncertainty about* the state. Inputs
that cannot are "neutral" (Feldbaum's term): "The lack of dual control effect is called neutrality"
— and LQG is the classic neutral case, which is why the separation principle holds there and
exploration is worthless. Meijer & Rantzer's one-line gloss: dual control is "Feedback control that
simultaneously regulates the plant and probes it to reduce uncertainty," and "Aggressive probing may
deteriorate performance but improve parameter estimates. Conversely, cautious control may give better
short-term performance but fail to gather information."

**How the tradeoff is traded off in practice** (four canonical directions, per Meijer & Rantzer):
(a) multi-armed bandits — Gittins index (Bayesian optimum), UCB/"optimism in the face of uncertainty,"
regret minimization (the *regret* — performance gap vs. the known-parameter optimal controller — is
the standard currency for exploration cost); (b) self-tuning regulators (Åström & Wittenmark) —
certainty-equivalence with recursive estimation, where exploration issues (persistent excitation,
closed-loop identifiability) emerged as *pathologies* rather than design choices; (c) regret-rate
minimization for LQR (log T vs √T regimes, set by closed-loop identifiability); (d) minimax optimal
dual control (Rantzer, Vinnicombe) — worst-case cost over a model class M, where exploration appears
because the adversary "also selects the unknown parameters." In the three-step worked example, the
exploration motive literally appears as a *concave parabola* in the Bellman minimization: "the concave
parabola shifts the maximizing u away from zero, causing the controller to probe the system by
applying non-zero actuation even when no prior knowledge is available (Z=0)."

When DP is intractable (always, beyond toy cases), **explicit dual control** re-injects probing by
hand: an additive probing-reward term in the cost (Goodwin & Payne 1977; Wittenmark 1975a; Milito et
al. 1982), a constraint (Alster & Bélanger 1974), a persistent-excitation constraint (Nikolaou and
coworkers), or an optimal-experiment-design term (Fisher-information objectives; "self-reflective" MPC,
Houska et al.). Mesbah: "The reward term for system probing reflects the quality of the to-be-identified
parameters, and is intended to prevent the controller from yielding zero control inputs."

**Two failure modes named in the classical literature — both from *too little* probing, not too much:**
1. *Turn-off / zero input*: cautious control (uncertainty-penalizing, one-step) can die: "cautious
   control can yield exceedingly small control inputs when the uncertainty grows. Small control inputs
   will in turn generate less information about the uncertain system and, consequently, the uncertainty
   will be increased further, eventually yielding zero control inputs" (Mesbah, citing Åström &
   Wittenmark 1971). Note the *feedback loop through the information channel*: low input → low
   information → high uncertainty → lower input. This is the closest classical analog to a
   self-reinforcing collapse driven by the sensing/learning channel — but it is a collapse of
   *actuation*, not of the sensor, and it is cured *by more probing*.
2. *Bursting / instability under adaptation*: certainty-equivalence adaptive control on a plant with
   unmodeled dynamics or lacking persistent excitation can go unstable (Anderson 1985 "bursting";
   Rohrs et al. 1985 counterexamples — see §2).

---

## 2. Does dual control already contain "monitoring cost destabilizes" — sensing channel = failure channel? (Q2)

**What is in dual control.** The information-gathering action (the probe) does perturb the plant and
does degrade short-run performance — deliberately, transiently, and by design: probing "may detract
from short-term performance but will improve control in the future" (Wikipedia's summary of
Feldbaum's car analogy, consistent with the primary glosses). The cost of probing is represented
exactly as in §1: poorer transient stage cost, paid for reduced future uncertainty; in regret terms,
a growing regret component; in explicit dual control, an added cost/constraint term. Probing is
*chosen, bounded, and instrumental*: "uncertainty reduction is sought to the extent dictated by the
closed-loop control performance improvement… closed-loop control inputs intrinsically seek
control-oriented model adaptation" (Mesbah). Nothing in the corpus lets the probe become *self-
sustaining* damage; the optimizer stops probing exactly when probing stops paying.

**What is not in dual control.**
- *The observation channel is never the actuator of collapse.* In the standard formulation the
  sensing equation y = h(x, v) carries measurement noise only; the controller cannot perturb the
  plant through it. The thing that "changes the state adversely" is the probing *control input*, not
  a sensing/self-monitoring action. Our claim inverts the topology: the *sensor is the actuator*.
- *No result where monitoring sustains the failure it monitors.* The searches and both full surveys
  surface no theorem, example, or folk result in which the act of observing/estimating/introspecting
  drives the observed system into a worse *state* (as opposed to a worse *score*). The classical
  destabilizers run the other way: too little excitation (turn-off) or structural mismatch (bursting).
- *Adjacent branch that exists — and what it covers.* There is a classical branch treating
  *observation itself as a decision with a cost* (partially observed stochastic control, "measurable
  controls"; Runggaldier & Stettner 1994, *Approximations of Discrete Time Partially Observed Control
  Problems*; modern surveys by Yüksel). Searches surfaced there only *resource* costs of observing
  (pay to observe, budget to observe), not state-degradation caused by observing. No quote asserting
  monitoring-driven state collapse was found.
- *The closest genuine precedent inside our own paper's scope is adaptive-control fragility*: Rohrs,
  Valavani, Athans & Stein (1985) — "sinusoidal reference inputs at specific frequencies and/or …
  sinusoidal output disturbances at any frequency (including dc), can cause the loop gain to increase
  without bound, thereby exciting the unmodeled high-frequency dynamics, and yielding an unstable
  control system… existing adaptive control algorithms … cannot be used with confidence in practical
  designs where the plant contains unmodeled dynamics because instability is likely to result."
  This establishes "the adaptive machinery itself can destabilize the plant" — but the destabilizer
  is the *adaptation/update*, triggered by external signals, not the sensing channel, and there is no
  sensor-death mechanism. Anderson's bursting (cited by Mesbah) is the same family.
- Meijer & Rantzer also record the *information-channel feedback* behind turn-off-like phenomena on
  the estimation side: "It has long been recognized that system parameters can be hard to identify in
  closed-loop operation [Söderström, Gustavsson & Ljung 1975]… if the optimal controller makes
  essential parameters non-identifiable … exploration needs to drive the system away from optimality."
  Again: sensing interacts with control, but never as a destructive self-monitoring loop.

**Verdict on claim 1 (sensing-channel-is-failure-channel): (b) adjacent/variant — the *cost* half is
classical, the *failure-channel* half is absent.** Dual control fully contains "information gathering
has a performance cost that must be traded off" (and even a positive feedback through the information
state that can produce turn-off). It does *not* contain "the monitoring action drives the monitored
system into a worse state / sustains the collapse it detects," and it does not model a sensor that is
destroyed by the failure it must detect. Our contribution must be stated as: *we move the cost of
monitoring from the objective into the state dynamics* — monitoring is not expensive, it is
*pathogenic* — plus the design rule (sensor must live in the surviving subsystem). Do not cite dual
control as if it anticipated this; cite it as the formalization of the tradeoff we modify.

---

## 3. Does dual control contain the capability–recoverability tradeoff r·φ ≈ const? (Q3 — highest value)

**The claim under test:** as controller capability grows, the set of survivable assumptions/floors
*shrinks* (rectangular hyperbola r·φ ≈ const): more capable agents can disprove more of their own
floors, so self-rescue requires an ever-more-undecidable floor.

**What dual control says about capability growth — the opposite monotonicity.** Dual control is a
theory of uncertainty *reduction*: data accumulates, posterior variance shrinks, probing decays, and
the controller converges toward certainty equivalence, which is *good* (regret is sublinear in every
formulation: log T or √T). In Jedra & Proutiere's CE-with-probing controller the excitation budget
ν(t) ~ N(0, σ²√(d_x/t)) shrinks over time and "the expected number of times that K_0 is used is
finite" — exploration is front-loaded and dies out as competence grows. There is no mechanism by
which learning *enables* new destruction of the controller's own supports; the hyperstate only gets
better. Meijer & Rantzer's surveys — which explicitly organize the field around "when is exploration
fundamentally necessary" — contain no result of the form "more capability ⇒ less stability" or
"capability shrinks the viable parameter/assumption set."

**The nearest formal neighbors (checked, none anticipates the claim):**
1. *Worst case over a larger model class is worse.* Meijer & Rantzer §7.1: Megretski's lower bound
   implies "enlarging the model class M generally negatively impacts the ℓ₂-gain from w to x even
   when the true system parameters are small"; Vinnicombe: "There exists, at least in discrete time,
   no ℓ₂-gain stabilizing adaptive controller that is universal in the sense that it works for all
   a ∈ ℝ." This is structurally the closest statement — more capacity to *entertain possibilities*
   worsens the guaranteed bound — but the "enlargement" is of the *uncertainty set*, not of the
   controller's reasoning strength; it says nothing about self-derived floors, falsifiability, or a
   hyperbolic boundary, and the worsening is linear-ish in the set size, not a product law.
2. *Performance–robustness tradeoff* (classical loop shaping / small-gain: high performance ⇒ tight
   loop ⇒ small margins). A static design tradeoff at design time; not capability-indexed, no
   shrinking set over the controller's lifetime, no learning. (Checked via targeted search; standard
   textbook material.)
3. *Adaptive-control fragility* (Rohrs; Anderson's bursting): adaptation destabilizes under unmodeled
   dynamics — but richer models/estimators are the *cure* in that literature, and instability is
   signal-condition-dependent, not capability-monotone.
4. *Robustness of the regret guarantees themselves*: "these rate bounds rely on idealized conditions
   and even small mismatch between the model structure and the true system will typically lead to
   growth rate T instead of √T" (Meijer & Rantzer, citing Lee, Rantzer & Matni 2024). Fragility of
   guarantees to mismatch — again not capability-indexed.
5. *Outside control theory:* the AI-safety "scalable oversight" literature records the analogous
   monotone trend ("as AI systems become more capable, humans become less able to evaluate" them),
   but qualitatively, with a different mechanism (evaluator asymmetry, not self-reference), and no
   quantitative law. (Also already noted in handoff-selfreg-novelty-app: 2607.04277's introspection
   threshold is a *lower* bound — complementary direction to ours.)

**Verdict on claim 2 (r·φ ≈ const, capability narrows the viable set): (c) not present in dual
control or its adjacent literatures, to the standard of evidence available here** (two full modern
surveys read end-to-end — neither mentions anything of the sort; targeted searches into robust
adaptive control, performance–robustness tradeoffs, minimax dual control, and scalable oversight).
The honest caveats: (i) sixty years of IEEE TAC cannot be exhaustively excluded; the claim of absence
rests on the surveys' silence plus the framing mismatch (dual control's monotonicity runs the other
way); (ii) the *constant* 0.2 is a property of the specific model (saturating tanh switch), not of
dual control, and the handoff notes already flag that it needs analytic derivation to be a law; the
*defensible* novelty is the hyperbolic *shape* of the boundary — r·φ ≈ const, viable set → {φ = 0}
as r → ∞ — i.e., "escape requires an undecidable floor, and undecidability is relative to the
agent's own power." That inversion (capability *consumes* recoverability) is not in this literature
and is the paper's most defensible quantitative content. This is good news for the paper; do not
soft-pedal it.

---

## 4. Is there a counterpart to the "floor" — a cheap fixed, never-re-derived constant? (Q4)

**Analogues that exist (functional, not conceptual):**
1. *The known-stabilizing controller assumption K₀* in regret-minimizing LQR: every √T-regret result
   (Abbasi-Yadkori & Szepesvári 2011; Jedra & Proutiere; Campi & Kumar 1998) *assumes* a stabilizing
   feedback known in advance, used while data is poor. Meijer & Rantzer: "the assumption that a
   stabilizing controller K₀ is known in advance is more restrictive than it might seem at first
   glance" (they give the parameter-space wedge it carves out, and note minimax dual control is the
   way to remove it). This is a *fixed prior support that is not re-derived from data* — functionally
   the closest thing to a floor — but it is an assumption on the *plant*, never an epistemic floor
   for the agent, and nothing in the literature prevents re-estimating it; its status as
   "must-not-re-derive" is ours.
2. *Fallback controllers in robust adaptive control / safety control*: switch to a fixed, conservative
   baseline when adaptation misbehaves (standard robust-adaptive practice; adaptive/backup control
   barrier functions use "a pre-certified safe controller" as the safety fallback). Same family: a
   fixed, cheap, non-adaptive object that bounds the damage an adaptive layer can do.
3. In the minimax formulation, prior knowledge enters as Z₀ ≠ 0 ("the initial condition can be
   exploited to incorporate prior knowledge about the system") — again a given, not re-derived.

**What does not exist anywhere in this literature:** the three properties that make *our* floor what
it is — (a) it is an *assumption of the agent*, not of the designer; (b) its essential property is
*undecidability by the agent itself* (φ = 0), i.e., a derivability property, not a numerical
constant; (c) the failure mode of a *knowingly-held, decidable* floor (it generates a monitored
discrepancy and fails — the r·φ result). Classical control has fixed baselines; it has no notion of a
belief the controller must hold without adjudicating, and no mechanism by which re-deriving a
baseline destroys it.

**Verdict on claim 3 (the floor): (c) not present.** Adjacent artifacts exist (K₀, fallback
controllers, prior-injected Z₀) and should be cited as engineering kin — especially K₀, which
lets the paper say "regret-optimal learning controllers already rely on a given, un-re-derived
support; we show that for self-referential agents such a support must additionally be
non-adjudicable, and quantify for how long that holds as capability grows." But the floor as
*epistemic undecidable anchor*, and the knowing-floor failure, are outside dual control.

---

## 5. Standard citation set (Q5)

**Founding:**
- Feldbaum, A.A. (1960a,b). "Dual control theory, I–II." *Avtomatika i Telemekhanika* 21(9): 1240–1249;
  21(11): 1453–1464 (Russian). English translation: *Automation and Remote Control* 21(9): 874–880
  (Part I, April 1961); 21(11): 1033–1039 (Part II, May 1961).
- Feldbaum, A.A. (1961a,b). "Dual control theory, III–IV." *Automation and Remote Control* 22: 1–12 ff.
- Feldbaum, A.A. (1961). "Dual control theory problems." *IFAC Proceedings Volumes* 1(2): 541–550
  (2nd IFAC Congress, Basle).
- Feldbaum, A.A. (1965). *Optimal Control Systems.* Academic Press, New York. ← the standard
  book-length citation; this is what "Feldbaum (1965)" should point to.
- Åström, K.J. (1965). "Optimal control of Markov processes with incomplete state information."
  *J. Math. Anal. Appl.* 10(1): 174–205. (POMDP formalization — per Meijer & Rantzer the rigorous
  framework for dual control.)

**The formal definitions:**
- Bar-Shalom, Y. & Tse, E. (1974). "Dual effect, certainty equivalence, and separation in stochastic
  control." *IEEE Trans. Automatic Control* 19(5): 494–500. ← cite for "dual effect"/Definition 2.
- Tse, E., Bar-Shalom, Y. & Meier, L. (1973). "Wide-sense adaptive dual control of stochastic
  nonlinear systems." *IEEE Trans. Automatic Control* 18(2): 98–108. (implicit dual control line.)

**Surveys (choose 2–3; do not cite off footnotes):**
- Wittenmark, B. (1995). "Adaptive dual control methods: An overview." *IFAC Proceedings Volumes*
  28(13): 67–72 (5th IFAC Symp. Adaptive Systems in Control and Signal Processing). [Abstract
  verified: "The optimal solution can quite straightforwardly be characterized. The solution is,
  however, numerically demanding."]
- Filatov, N.M. & Unbehauen, H. (2000). "Survey of adaptive dual control methods." *IEE Proceedings —
  Control Theory and Applications* 147(1): 118–128. [Abstract verified.]
- Mesbah, A. (2018). "Stochastic model predictive control with active uncertainty learning: A survey
  on dual control." *Annual Reviews in Control* 45: 107–117. ← read in full; best single modern
  formulation of the OCP/hyperstate/Bellman apparatus.
- Meijer, T.J. & Rantzer, A. (2026). "Dual Control: On Exploration–Exploitation in Linear Systems."
  *Annual Review of Control, Robotics, and Autonomous Systems* 10 (2027); arXiv:2608.20073. ← read
  in full; the canonical current review (bandits / self-tuning / regret / minimax).

**Context primaries the paper should know:**
- Åström, K.J. & Wittenmark, B. (1971). "Problems of identification and control." *J. Math. Anal.
  Appl.* (turn-off/zero-input phenomenon). Åström & Wittenmark (1973). "On self-tuning regulators."
  *Automatica* 9(2): 185–199.
- Sternby, J. (1976). "A simple dual control problem with an analytical solution." *IEEE TAC*
  21(6): 840–844. (closed-form dual solution).
- Söderström, T., Gustavsson, I. & Ljung, L. (1975). "Identifiability conditions for linear systems
  operating in closed loop." *Int. J. Control* 21(2): 243–255.
- Anderson, B.D.O. (1985). "Adaptive systems, lack of persistency of excitation and bursting
  phenomena" (per Mesbah's reference list). Rohrs, Valavani, Athans & Stein (1985). "Robustness of
  continuous-time adaptive control algorithms in the presence of unmodeled dynamics." *IEEE TAC*
  AC-30(9): 881 ff. [read]
- Vinnicombe (2004), Megretski & Rantzer (2003), Rantzer (2020/2025/2026) minimax series, Lee,
  Rantzer & Matni (2024) — via Meijer & Rantzer §7.

---

## 6. Explicit verdicts on the paper's three claims

| Claim | (a) already in dual control | (b) adjacent/variant | (c) not present |
|---|---|---|---|
| 1. Sensing channel IS the failure channel (monitoring destabilizes the state; self-defeating sensor) | — | **YES — (b)**: probing-cost tradeoff and information-loop pathologies (turn-off) are classical; monitoring-as-collapse-cause and sensor-death are absent | half |
| 2. Capability–recoverability tradeoff, r·φ ≈ 0.2, viable set shrinks with capability | no | nearest: larger model class ⇒ worse minimax gain; performance–robustness tradeoff; scalable-oversight trend (qualitative) | **YES — (c)** in dual control proper |
| 3. The floor (cheap fixed never-re-derived undecidable constant as escape) | no | nearest: assumed stabilizing K₀; fixed fallback controllers; prior injection Z₀ | **YES — (c)** as epistemic/undecidable floor |

---

## 7. Repositioning required (explicit statements)

1. **Claim 1 must be narrowed, not abandoned.** Say: "dual control (Feldbaum 1960–65; Bar-Shalom &
   Tse 1974) already prices information gathering as a performance cost traded against control
   performance. We change its place in the dynamics: in self-referential agents the monitoring action
   is not merely costly but *pathogenic* — it adds inward drive (c_mon > c_mon_crit ⇒ collapse it was
   meant to detect) and the sensor itself dies with the failure (G collapses while D is untouched).
   Dual control contains no result in which observing drives the observed system into a worse state;
   its pathologies (turn-off, bursting) run in the opposite direction." Do NOT cite dual control as
   anticipating the self-defeating sensor; cite it as the formalized tradeoff being modified.
2. **Claim 2 is the paper's most defensible quantitative content and is NOT anticipated.** State
   plainly: dual control and its modern reviews contain no result in which capability growth shrinks
   a survivable set; their monotonicity is the reverse (learning reduces uncertainty, probing decays,
   certainty equivalence is approached). The r·φ hyperbola — and especially its limit "only φ = 0
   floors survive as r → ∞" — is novel in this literature; guard it by (i) deriving the constant
   analytically or labeling it model-specific, (ii) citing the nearest neighbors (minimax model-class
   enlargement; performance–robustness tradeoff) as kin, not anticipation.
3. **Claim 3 stands, with new citations.** Anchor the floor to the K₀ assumption of regret-optimal
   learning controllers (it is already "a given, not re-derived" — but for the plant, not the agent)
   and to fixed fallback controllers; the novelty to assert is the *undecidability requirement*
   (φ = 0) and the knowingly-held-floor failure, neither of which exists in this literature.
4. **Mechanism credit.** Anywhere the paper says "we discover the explore/exploit tension in
   self-monitoring" — delete and cite Feldbaum/Bar-Shalom–Tse/Mesbah/Meijer–Rantzer instead. The
   phrase to keep is the one from the earlier novelty note: we *rederive* the monitoring-vs-control
   tradeoff for self-referential agents and then break it (monitoring is not traded off against
   control; it *is* the failure channel).

---

## Appendix: sources actually read (how, and at what depth)

| Source | Depth | How obtained |
|---|---|---|
| Mesbah (2018), *Annu. Rev. Control* 45:107–117 | **Full text, end-to-end** (11 pp.) | jina read of author-hosted PDF (static1.squarespace.com/…/ARC_2018.pdf) |
| Meijer & Rantzer (2026), "Dual Control: On Exploration–Exploitation in Linear Systems," arXiv:2608.20073 | **Full text, end-to-end** | jina read of arXiv HTML; raw HTML fetched with curl to recover full passages; reference list extracted |
| Rohrs, Valavani, Athans & Stein (1985), IEEE TAC AC-30(9) | Abstract + intro/abstract text (excerpt read) | jina read of publicly hosted PDF (maxim.ece.illinois.edu) |
| Rantzer (n.d.), "Dual Control: Optimization Based Exploration/Exploitation" (Lund lecture slides) | Full slide text | jina read of nikolaimatni.github.io PDF |
| Venkatasubramanian, Köhler, Berberich, Allgöwer (2020), "Robust Dual Control based on Gain Scheduling," arXiv:2004.04563 | Introduction read (full paper on disk) | jina read of arXiv PDF |
| Wikipedia, "Dual control theory" | Full (short) | jina read |
| Wittenmark (1995), IFAC 28(13):67–72 | **Abstract only** (paywalled; CiteSeerX copy dead — verified) | ScienceDirect abstract page |
| Filatov & Unbehauen (2000), IEE Proc-CTA 147(1) | **Abstract only** (paywalled; ResearchGate blocked by CAPTCHA) | IET digital library abstract |
| Feldbaum (1960–65) originals | **Not read** (archival/paywalled); formulation recovered via the two full surveys + translation citation record + Wikipedia | — |
| Sternby (1976); Åström & Wittenmark (1971); Anderson (1985); Jedra & Proutiere; Lee/Rantzer/Matni (2024) | Bibliographically verified via the surveys' reference lists; content known via survey descriptions | — |
| Adjacent-field checks (performance–robustness tradeoff; costly observations/partial observation; scalable oversight; fallback-controller practice) | Search-level only (hit titles/abstracts) | jina search |

**Negative-evidence caveat.** The "(c) not present" verdicts rest on: two full modern surveys read
end-to-end (their combined coverage is the field's self-understanding, and neither hints at any
capability-destabilization or monitoring-state-degradation result), plus targeted searches into the
adjacent literatures named above. That is strong but not exhaustive; the handoff's own standard
("be rigorous and honest") is met by stating that the surveys' silence is the load-bearing evidence.
