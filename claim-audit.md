# Claim audit: dynamical fact vs agent interpretation

Companion to `paper.md` (revision of the claim-audit pass, [TR] indexing unchanged). Every substantive claim in the paper was sorted into three classes:

- **A — DYNAMICAL FACT.** A computed result of the specified system, reproducible from the code (`dpdr/dpdr/*.py`), the caches (`dpdr/cache/*.npz`), or the versioned documents. These are the paper's substance and were **not weakened** — every headline number was re-verified from cache during this audit (see "Verification" at the end).
- **B — AGENT INTERPRETATION.** A reading of a dynamical fact in agent vocabulary (values, belief, care, adjudication, vigilance, vocation, worry). Legitimate **if marked**; this revision marks every surviving one *interpretation* in the text.
- **C — UNSUPPORTED PROJECTION.** An agent claim with no dynamical fact behind it, or an agent attribution presented as measurement. These were removed or demoted to the dynamical fact underneath.

The ground for the sorting (verified in `dpdr/dpdr/model.py:105-123`): the model is a control system with one setpoint — the `g·tanh(E/Es)` loop driving E → 0 — and contains **no** value, preference, goal, want, survival, representational, or self-referential term beyond the loop input E = D − G. (A naive grep for "value" hits only `Schedule.value()`, a method name.) Agent vocabulary can therefore never be a *measurement* of this system; at most a marked reading, and only ever of the agent the model is a specification toward (§8), not of the model.

## 1. Results (§2–§4): essentially all class A

| Claim | Class | Source | Action taken |
|---|---|---|---|
| Gates G1–G3: healthy G* = 0.886; collapse G = 0.049; no self-recovery 0.050; rescue map monotone | A | `dpdr/README.md:108-117`; `dpdr/predictions.md` | Kept verbatim |
| P1 thresholds (αG 0.8871, Θ 0.4418, g0 0.7029, 66.32 t.u., 0.5992, A 0.1859); 214/98 grid, zero intermediate | A | `dpdr/predictions.md:25-63`; figs f04, f05 | Kept verbatim |
| P2 no recovery in ~2500 runs, 106/112 sensitivity cells (ε0 caveat) | A | `dpdr/predictions.md:67-89` | Kept verbatim |
| P3 rescue monotone in dose/duration; u* 0.95→0.45; delay 0.40→0.95; hard minimum engagement duration | A | `dpdr/predictions.md:92-123`; fig f06 | Kept verbatim |
| P4 relapse: second threshold 66.0 vs first 66.3; onset 73 t.u. either episode; single-episode relapse fraction 0 | A | `dpdr/predictions.md:126-183`; fig f06b | Kept verbatim |
| ~~"the recovered state's head start buys no protection"~~ | C | (gloss on P4) | **Demoted**: "leaves the threshold unchanged" — same dynamical fact, no transactional agent verb |
| P5 FAIL: g → g0 exactly (C ≈ −2.5e-8, τ = 667 = τg/μ); PSD same bin; E-loop overdamped | A | `dpdr/predictions.md:187-236` | Kept verbatim |
| P6 t90 ratio 5.41 vs built-in 5.0; ordering residue | A | `dpdr/predictions.md:239-262`; fig f08 | Kept verbatim |
| Terminator truth table: content OR floor prevents collapse (4 cells, G_end 0.0486/0.8855) | A | [TR-13] | Kept verbatim |
| Open loop collapses deeper: 0.0121 vs 0.0487 | A | [TR-14] | Kept verbatim |
| Terminator/sink unification of four mechanisms | B | [TR-13, *synthesis*] | Was already marked interpretation; kept |
| ~~"the reducer normalizes the self"~~ (§4.2 cut-reduction gloss) | C | (gloss) | **Demoted**: "normalizes the proof object it is part of" — dynamical referent |
| §4.3 heading "rescue of an already-stuck **agent**" | C | (register error) | **Demoted**: "already-stuck **system**" |
| Any cheap floor: floors ≥ 0.5 escape identically to 0.8855 in ~22 t.u. under all three loads; floor_crit 0.4795 ≈ E* 0.4969 | A | [TR-5, TR-7]; `cache/exp6_floor.npz`, `exp6_bisect.npz` (re-verified in this audit) | Kept verbatim |
| Mechanism chain floor → Θeff ≥ E → c = 0 → G regrows | A | `dpdr/dpdr/regulator.py` | Kept verbatim |
| ~~"The floor is false by construction"~~ | C | (gloss) | **Demoted**: "a clamped constant that ignores the true S"; the agent reading ("held belief not re-examined") retained but **marked** *interpretation* |
| Cheap-vs-elaborate: elaborate fails (0.116/0.090), c_mon_crit = 0.511 gated; deployed gated threshold vanishes; ungated cost degrades 0.53→0.12 | A | [TR-7, TR-15] | Kept verbatim |
| ~~"the frozen agent collapses… the regulated agent never"~~ (long horizon) | C | (register error) | **Demoted**: "the frozen system / the floor-regulated system" |
| Long horizon: 60 episodes, dip minima 0.218…0.132, never below ~0.103 to a_hold 2.0; permanent-capture degraded states 0.21/0.16/0.11 | A | [TR-7] | Kept verbatim |
| Healthy-regime cost zero: max \|ΔG\| = 0.00e+00; dips shallower | A | [TR-7] | Kept verbatim |
| Knowing floor: G_end 0.53/0.34/0.24/0.16; kc ≈ 0.2; gated-no-authority fails; k = 4 removes the failure | A | [TR-4, TR-7, TR-15] | Kept verbatim |
| "checks and worries but never acts" (no-authority configuration) | B | (gloss) | **Marked** *interpretation*; "knowing floor" flagged as project assay name, not a knowledge attribution |
| r·φ ≈ 0.2 hyperbola (all cells); viable φ-set shrinks with r; φ = 0 escapes at any r | A | [TR-16] | Kept verbatim |
| ~~"escape requires an **undecidable** floor… only genuinely undecidable floors survive" (asserted flat)~~ | C | (agent claim) | **Demoted**: dynamical statement primary ("no *representable* derivable disproof"); the agent reading retained but **marked** *interpretation*, with "in the model, 'undecidable' means only φ = 0" |
| Window: T_max = 448.93/224.46/112.26; product 0.11223/0.11223/0.11226; invariance forced by construction; consistency check not unification; identity 1.1e-16; healthy collapse 0.781→0.045 | A | [TR-9]; `cache/exp7_partb.npz` (re-verified in this audit) | Kept verbatim |
| Lower edge untestable (monotone benefit; sign bug; one-generation mismatch); conjecture neither supported nor refuted | A | [TR-9]; window-correction node | Kept verbatim |
| ~~"the agent's own dynamics"~~ (self-simulation object, §4.5) | C | (register error) | **Demoted**: "the system's own dynamics" |
| ~~"more capacity reduces tolerance"~~ | C | (gloss) | **Demoted**: "lowers the sustained-drive threshold" (same measurement: 1.393→1.221) |
| Per-dip margins +0.0028/+0.0339; inter-episode cost −0.058/−0.075, min −0.164/−0.180; window-only cannot self-rescue | A | [TR-9]; `cache/exp7_partc.npz` | Kept verbatim |
| ~~"buys intra-episode dip protection"~~ / ~~"escape remains the floor's job"~~ | C | (gloss) | **Demoted**: "provides…"; "only the floor achieves escape in this assay" |
| Logic engine: K = 0 stuck at all N; K = 1 resolves to N = 128 in ~2N + 2; no depth threshold | A | [TR-10] | Kept verbatim |
| Cut-elimination mapping: 1 + k·d linear vs ~k^d blowup; sharp cliff derived; floor = keep cuts | A/B | [TR-11] | Kept; analytic status already flagged |
| Permissive AND-gate: four signatures + honest negatives | A | [TR-17]; `cache/exp5_*.npz` | Kept verbatim |
| Discriminator: neither signature in frozen model; ratchet vs boost λ equality; leak arm; μ_k family | A | [TR-18, TR-19]; `cache/exp4_*.npz` | Kept verbatim |
| "vigilance" as the signature name | B | (gloss) | **Marked**: agent-register gloss; dynamical candidates named |
| Fidelity checks (0.00e+00 / 1.4e-5 / bit-identical); 53 tests; freezing discipline | A | §3.1–3.2, `dpdr/tests/` | Kept verbatim |

## 2. Abstract, intro, discussion, conclusion

| Claim | Class | Source | Action taken |
|---|---|---|---|
| Abstract: all four quantified boundaries (0.4795 ≈ 0.4969; 0.511/0.112; r·φ ≈ 0.2; AND-gate) | A | caches above | Kept verbatim |
| Abstract: substrate-independence legs; conjecture untestable; 0.112 consistency check; falsifiers; limitations enumeration | A | [TR-9, TR-10, TR-11] | Kept verbatim |
| Abstract: the unifying reading stated as the paper's synthesis | B | [TR-3] | **Marked**: now "a dynamical pattern first and the paper's synthesis over its measurements second, offered as an interpretation in agent terms for the agent this line of work is a specification toward (§8), not as a measured property of an agent" — plus a new framing sentence declaring the fact/interpretation register |
| Intro item 1: escape taxonomy (terminates or routes outward) | A | [TR-13] | Kept |
| Intro item 1: ~~"an external vocation"~~ as a rescue mechanism | B/C | (gloss) | **Demoted out of the mechanism list**; retained as a marked *interpretation* gloss ("a standing external demand is an outward 'vocation' for the reducer") |
| Intro item 1: knowing-floor exception reproduces kc ≈ 0.2 | A | [TR-4, TR-5] | Kept |
| Intro item 2: budget / product bound / consistency check | A | [TR-9] | Kept |
| Intro item 3: ~~"the stronger the reasoner, the fewer floors it can hold… a floor the agent cannot itself adjudicate — undecidable" (unmarked)~~ | C | [TR-16] | **Demoted**: dynamical statement primary; agent reading **marked** *interpretation*; "undecidable" scoped to φ = 0 |
| §5.1 synthesis: six operations, same source of benefit and failure | A (as pattern over A-facts) | [TR-3] | Kept |
| §5.1: ~~"self-examination of the floor verifies and cancels the protection"~~ (agent operation) | C | (gloss) | **Demoted**: "re-deriving the floor while holding it cancels its protection" |
| §5.1: the synthesis stated as agent-property result | B | [TR-3] | **Rewritten as required**: "a dynamical result, interpretable in agent terms" — agent reading explicitly marked *interpretation*, tied to §2.1's inventory, and declared unmeasurable in this substrate |
| §5.1: honest status (coherence not confirmation; external legs; measured edges; cheap-and-deep corollary) | A/B | [TR-3] | Kept (corollary is a prediction, already labeled testable) |
| §5.2: budget exists/sharp/measured per channel; agreement is not shared-budget evidence; sign reversal | A | [TR-9] | Kept |
| §5.3: D/G dissociation (D 0.545 unchanged, G 0.886→0.049); sensor-placement rule | A | [TR-20] | Kept |
| §5.3: "the sensing channel is converted into the state being watched for" | B | [TR-20] | **Demoted to dynamical**: "converted into the failure variable: monitoring adds inward drive, and inward drive is the collapse axis"; agent gloss retained **marked** |
| §5.4: model has no values term | A | [TR-22]; verified in `model.py` | Kept (now also stated in §2.1) |
| §5.4: "a self is expensive… justifiable only if…" argument; floor-as-value reading | B | [TR-22] | **Marked** *interpretation, not a measurement — the model cannot adjudicate it*; readings explicitly tied to the underlying dynamical facts |
| §5.5: channel-coincidence constraint vs Chialvo's free-monitoring | A | [TR-20] | Kept |
| §5.5: "self-defeating sensor" | B | (gloss) | **Marked**: agent-register gloss for the channel coincidence; measured content named (monitoring-cost ceiling) |
| §6 items 1–14 (all limitations as previously written) | A | various | Kept verbatim |
| §6 new item 15: the category distinction itself | A (meta) | `handoff-selfreg-category` | **Added**: states the one-setpoint inventory, the fact/interpretation rule, and the six earlier projections as dynamical facts misread |
| §7 related work: positioning statements | A (attribution) | lit-sweep notes | Kept |
| §8 preamble | A (meta) | this audit | **Added**: registers the object change — §8's agent vocabulary refers to an agent *to be built*, its claims are design hypotheses |
| §8.1: slot constraints (τa = 1, c_mon ceiling disqualify an LLM salience module); all-LLM failure thesis | A-derived design claims | [TR-27, TR-28] | Kept (falsifiable design thesis, correctly labeled) |
| §8.1: ~~"the analysis here vindicates it as a choice of substrate"~~ | C | (gloss) | **Demoted**: "consistent with it as a substrate choice (a consistency, not a vindication)" |
| §8.3: value-consistency metric; pinned-vs-maintained value prediction | B (design hypothesis on marked reading) | [TR-22] | **Mark**: now explicitly "on the floor-as-value reading of §5.4 (*interpretation*)" |
| §8.4: values dimension on G | future work | [TR-22] | Reading now marked *interpretation* |
| §9 conclusion: duality/budget/escape facts; untestable conjecture; design corollary; mind-changers | A | above | Kept verbatim |
| §9: ~~agent-level readings implied as findings~~ | B | this audit | **Marked**: new sentences — every headline is a dynamical fact; the agent-level readings are interpretations, tabulated in `claim-audit.md`, and "cannot become facts in this substrate"; the build is where they could become true or false |

## 3. The six earlier headline failures, resorted

The project's six walked-back headlines (P4 framing, P5 framing, P6, unit-coincidence, engagement-timing, window convergence) were, on this audit, **projections**: dynamical facts read as agent properties. Current state in the paper:

| Earlier agent claim | The dynamical fact underneath | Class | Current paper state |
|---|---|---|---|
| "the agent cannot self-recover" | the collapsed fixed point is stable (G2b: max post-episode G = 0.050) | A + B gloss | Stated as the gate/P2 measurement; agent phrasing absent from results |
| "the agent's values protect it" | a clamped constant ≥ E* shifts the attractor (floor_crit 0.4795) | A + B gloss | Dynamical statement primary; "held belief not re-examined" marked *interpretation* |
| "the agent must hold a floor it cannot adjudicate" | escape boundary r·φ ≈ 0.2; only φ = 0 floors survive at high r | A + B gloss | Dynamical statement primary; agent reading marked; "undecidable" scoped to φ = 0 |
| "escape requires attention not be total" | viable escapes terminate the self-application or route it outside the reduction loop | A | §5.1 dynamical reading is now primary |
| "the unit coincidence unifies three tolerances" | one inward-drive slot ⇒ three labels for one number (consistency check, 1.1e-16) | A | Already corrected by the window revision; unchanged here |
| "the agent's post-recovery vigilance" | no persistent signature exists (g → g0; PSD unchanged) | A | P5 FAIL kept; "vigilance" flagged as project naming |

## 4. Counts

- **Class A (dynamical facts, kept verbatim): 62** entries in the tables above — the paper's measurements are strong and were not weakened. Headline numbers re-verified from cache during this audit: floor_crit 0.47950 ≈ E* 0.49688; c_mon_crit 0.51091 gated / 0.11218 ungated; T_max products 0.11223/0.11223/0.11226; a_hold_crit 0.11213; floors ≥ 0.5 escape to G = 0.8855, escape times ≈ 22 t.u.
- **Class B (agent interpretations retained, now explicitly marked): 14** — external vocation; undecidable-floor reading; held-belief-not-re-examined; checks-and-worries; vigilance naming; synthesis agent reading; values-gap argument; floor-as-value (three sites); sensor self-watching gloss; self-defeating-sensor gloss; intentional-stance role of all of the above.
- **Class C (projections removed or demoted): 15** — three "agent"→"system" register demotions (heading, long-horizon, window object); "false by construction"; "buys no protection"; "reduces tolerance"; "buys dip protection"; "the floor's job"; "the reducer normalizes the self"; "vindicates the substrate choice"; flat "undecidable floor" assertion; unmarked "stronger reasoner" phrasing; "vocation" as a mechanism; "self-examination" as an operation; synthesis-as-agent-property framing (rewritten per the brief).

## 5. Verification

- No measurement was weakened, rounded, hedged, or deleted. `diff` of headline constants against the pre-audit draft: all present, all unchanged (0.4795 ×3, 0.4969 ×2, 0.511 ×4, 0.11223 ×4, 0.11213 ×3, r·φ ≈ 0.2 ×6, 0.8855 ×4, 1.1e-16 ×2).
- Class-A numbers spot-checked directly against `dpdr/cache/exp6_bisect.npz`, `exp6_floor.npz`, `exp7_partb.npz` (loaded with the project venv) — all match the text.
- The dpdr project was not modified. Only `paper.md` edited and this file created.
- Word count: `paper.md` 11,164 → 12,421 words (net +1,257: the register convention, §2.1 inventory, §5.1 rewrite, §6 item 15, §8 preamble, and the marking parentheticals; against small deletions of demoted glosses).
