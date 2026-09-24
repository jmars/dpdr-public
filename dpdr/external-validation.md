# External-validation / pre-registration dossier — dpdr model

Status: **pre-registration protocol, not results.**  Nothing in this document
validates the model.  It specifies which of the model's predictions make
contact with *independent* (non-self-report-derived) data, what signature each
contact predicts, what would falsify it, and whether usable data already
exists.  Model numbers below are quoted from `predictions.md` (frozen Step 0–D
parameters); no model code, parameters, or verdicts were changed.

---

## 0. Why this dossier exists — the circularity problem

The model's architecture was **derived from a first-person report**;
the experiencer is the model's source, not an independent
instrument.  Scoring the model against that same report (P1–P6 in
`predictions.md`) is therefore a **self-consistency check, not confirmation** —
a model fit to its own source data cannot be evidence for itself.

The escape is to test the model's *quantitative structure* against data that
was neither collected from the experiencer nor shaped by this project.

**The anchor.**  Independently published work targets the same construct and
the same modeling program:

- Tolchinsky et al. 2025, *Temporal depth in a coherent self and in
  depersonalization: theoretical model*
  (https://pmc.ncbi.nlm.nih.gov/articles/PMC12444765/,
  doi:10.3389/fpsyg.2025.1585315) — proposes **temporal-depth collapse** as
  the mechanism of dissociation, framed in **nonlinear dynamical systems /
  attractor-landscape** terms, with recovery described as "de-stabilization of
  the maladaptive regime."  The authors state that a direct human experiment
  is currently infeasible ("we have not identified non-invasive methods of
  temporarily and harmlessly reducing temporal depth in humans") and that they
  **intend a computer simulation** ("should the computer simulation results
  support our hypothesis").  Simulation is thus the published methodology of
  record for this construct — not a convenience of this project.
- Madden & Serper 2026, *The time collapse–entrapment model of depersonalization
  severity* (https://www.sciencedirect.com/science/article/pii/S016517812600332X,
  PMID 42288070) — 433 adults (79 with clinically significant depersonalization):
  **temporal blurring** and reduced vividness of the future self predict
  depersonalization severity **via perceived entrapment** (mediation);
  discriminant function analysis classifies 75.6% of cases with distress,
  temporal blurring, absorption, and entrapment as the strongest contributors.
  Temporal-continuity disruption is here an *empirically loaded* predictor,
  not a metaphor.

The model's **S variable (allostatic setpoint, "temporal-depth proxy") maps
onto this published temporal-depth construct** — the identification that makes
external instruments (TII, GDSA) relevant to S at all (§7, loose mappings, for
how loose that is).

### What transfers across the model–data gap

The model is dimensionless (time in units of τa = 1; states in [0,1]).  Its
**absolute constants cannot be compared to human data** — no calibration
exists between t.u. and minutes, or between G and any test score.  Only three
kinds of prediction transfer:

1. **Ordering / dissociations** (X recovers slower than Y; reports dissociate
   from mechanism);
2. **Dimensionless ratios** (t90(S)/t90(G) = 5.41, robust range 3.87–8.30 —
   partly set by construction, see §3.6);
3. **Shape** (threshold-like vs graded transitions; sharp minimum effective
   dose; all-or-nothing outcomes).

Everything below is stated in one of those three currencies.

---

## 1. Master table — prediction → instrument → signature → falsifier → data status

| # | Model prediction (measured) | External instrument / dataset | Concrete predicted signature | Falsifier | Data status |
|---|---|---|---|---|---|
| **P1** | Threshold dose-response: sharp saddle-node on each of duration (66.32 t.u.), intensity (0.5992), affect (A ≥ 0.1859); zero intermediate outcomes (214 healthy / 98 stuck, none between) | Laboratory THC challenge (Melges 1970; Mathew 1992/93; D'Souza 2004; Colizzi 2019) with **dense within-subject dose ladder**, TII/CADSS-DP + GDSA at fixed timepoints; optionally THC × stress-induction cross (affect axis: mCPP challenge, Simeon 1995) | Within-subject: below a personal dose threshold, DP/temporal-disintegration returns to baseline after clearance; above it, disproportionately severe response — response-vs-dose curve is a **step, not a line**, within-subject. Axes trade off (higher affect ⇒ lower dose threshold; longer duration ⇒ lower intensity threshold) | Within-subject response is a smooth graded function of dose with no steep segment; or dose, duration, affect act purely additively with no interaction | **Partial.** Dose–response exists (2–3 dose levels, group means: Mathew high>low>placebo, 86% ≥ some temporal disintegration at high potency; D'Souza 0/2.5/5 mg dose-related psychotomimetics) but **group means cannot distinguish "graded" from "threshold + between-subject heterogeneity"** — only dense within-subject ladders can. New collection needed for the sharpness claim |
| **P1-a** (a-axis variant) | Inward-attention dose: sustained a_hold collapses the system; duration threshold 66 t.u. at a_hold 0.9 | Meditation-adverse-event data (Pons 2026, N=121: n=60 meditation-triggered, 61 non-meditation-triggered; Deane 2020 theory) | Onset risk as a function of cumulative practice hours is threshold/step-like rather than linear once susceptibility is controlled; triggers are dominated by sustained-inward-attention practices | Onset probability rises smoothly and proportionally with hours at all scales | **Partial / unminted.** Pons reports hours-before-first-episode (median 175 h; 36.5% ≤ 100 h) but no denominator (matched non-onset meditators) — threshold not identifiable from their published tables. Would need recontact/new collection |
| **P3** | Sharp minimum effective rescue dose; u* rises with delay (0.40→0.95 as delay 30→600); hard minimum engagement duration past long waits | CBT-for-DDD session-dose data (Hunter 2005/2023/2025 audit + feasibility RCT); treatment-intensity studies | A minimum effective engagement "dose" below which outcome ≈ untreated; required dose higher for longer-ill patients; no patchwork (monotone success region in dose × duration) | Symptom change proportional to sessions from session 1 (pure dose-linearity); or non-monotone success region | **Partial.** Hunter 2023 treats ≥ 8 sessions as the minimum "dose" and finds medium effect sizes after mean 17.3 months of therapy; the 10.6-month pre-treatment wait shows no significant CDS change (consistent with P2, not informative for P3's threshold). Dose-threshold sharpness untested — new collection (session-level outcome tracking) needed |
| **P4** | Recovered head start buys **no protection**: second-episode duration threshold **66.0** vs first-episode **66.3** t.u.; collapse onset identical (G<0.1 at 73 t.u. into both episodes); relapse counterexample G 0.885→0.049 | Clinical relapse data: episodic-course series (Simeon 1997, 30 cases; Simeon 2003a 117 cases; Baker 2003, 204 cases), Psychedelic/THC-triggered case series (Medford 2003; Masah 2025) | Relapse under renewed trigger exposure is **as easy as (or easier than) first onset** — threshold dose for episode 2 ≤ threshold for episode 1; episodic courses **escalate** (episodes lengthen/severity increases over time), not attenuate | Recovered patients need systematically *larger* triggers to relapse than to index onset (a protective margin that grows with recovery time); episodes attenuate over time | **Qualitative support exists** (episodic-course series — Simeon 2003a, Baker 2003; the "initially episodic, episodes becoming longer and more severe until pervasive" phrasing is from the Medford, Sierra, Baker & David 2005 review; ~⅓ episodic course). **The quantitative prediction (episode-2 threshold ≈ episode-1 threshold, within-patient) has no dataset anywhere.** New collection needed |
| **P2** | No self-recovery from deep collapse (max post-episode G = 0.050 over ≥ 600 t.u.; 106/112 sensitivity cells) | Natural-history/longitudinal data (Michal 2024 Gutenberg 5-year follow-up; Baker 2003 mean duration ~12–14 y at first specialist contact; Simeon 1997 "highly treatment refractory") | Among persistently depersonalized individuals, symptom levels do not drift back to norms without changed circumstances (treatment, life change); remission is rare and circumstance-linked, not time-linked | Documented spontaneous full recovery cohorts at rates comparable to treated remission | **Consistent existing data, not decisive.** Michal 2024: only 6.9% of depressed-with-DP/DR reach remission at 5 y (vs 15.9% without DP/DR — depression remission in a sub-clinical CDS-2 ≥ 1 sample, see §3.4); Baker/Simeon series show year-decade chronicity. But "absence of spontaneous recovery" cannot be cleanly estimated from treatment-seeking samples (selection bias: those who spontaneously recovered never present). Untested as a sharp claim |
| **P5-corollary** | Architecture denies a watcher ⇒ **vigilance/dissociation REPORTS dissociate from MECHANISM**: reports without measurable dynamic change; false-positive reports in healthy/non-impaired subjects. (Frozen model: g* → g0 exactly; zero persistent post-rescue compensation of any kind) | (a) DES/DES-T psychometrics in non-clinical samples (Leavitt 1999); (b) meditation-triggered DPDR-like states (Pons 2026); (c) subjective–objective dissociation in DDD (Sierra 2002 SCR; Guralnik 2000 — boundary case, §3.5; Millman 2024 systematic review) | Scale-endorsed "pathological dissociation" reports at high rate in subjects with no disorder/impairment; symptom-report severity decoupled from physiological/behavioral responsivity; post-recovery patients reporting vigilance show **rescued == naive** dynamic measures | Reports and mechanism always co-occur: endorsement of dissociation/vigilance items reliably predicts measurable dynamic abnormality (no false positives at meaningful rates); recovered patients show persistently faster error-correction / elevated gain-like measures | **Existing data already supports the report≠mechanism direction:** Leavitt 1999 — DES-T false-positive rate **54%** (vs 13% for DES) with only 35% of the sample classified correctly; Pons 2026 — 61.7% of the meditation-triggered group (n=60) scores above the CDS-70 clinical cutoff yet only 4 participants in the whole study (N=121) carry any DPDR diagnosis and cutoff-positive MEDT subjects show current mental disorder at 10.8%; Sierra 2002 — DPD patients' skin-conductance responses to unpleasant pictures markedly reduced; Guralnik 2000 — DPD patients show measurable attention/short-term-memory and spatial-reasoning deficits against comparable intellectual abilities (i.e. cognitive complaints **track** objective impairment there, not dissociate from it — included as the report/objective boundary case); Millman 2024 — reduced subjective emotional responses with mixed autonomic findings. **Caveat:** none of these measured *vigilance* dynamics per se; the g-channel is the model's own construct (see §7) |
| **P6** | S recovers ~5× slower than G; t90(S)/t90(G) = **5.41** (robust 3.87–8.30). **Largely built-in:** τ_S/τ_G = 100/20 = 5 is a chosen structural constant (`dpdr/model.py`), and the ratio "tracks τ_S almost exactly" (`predictions.md`) — see §3.6 | TII (Melges 1970) + GDSA as the temporal-depth channel vs a functional/cognitive battery (Guralnik-type) or DPDR symptom scale (CDS/DES-DP), measured **longitudinally** during recovery (treatment cohort or post-THC timecourse) | Non-built-in residue: (i) the measured ratio 5.41 exceeds the 5.0 set by construction; (ii) **ordering** — temporal-depth measures normalize **after** functional/symptom measures, i.e. patients functionally improved while still scoring high-TII | Temporal and functional measures recover at the same rate (ratio ≈ 1) — or temporal measures recover *faster* | **No usable existing dataset.** Requires new longitudinal cohort (or reanalysis of within-subject THC timecourses, §3.6). Simeon 2007 is cross-sectional and cannot test it |
| **P5 (original)** | FAIL: no persistent g* > g0; transient +1.7% peak decaying with τ=667; PSD shift = 0 | Post-recovery longitudinal measurement of error-correction speed / gain-like indices (reaction-time variability, ERP latencies) in recovered DPD vs matched naive | Rescued and naive systems statistically indistinguishable on dynamic gain measures once transient settling is past | Persistently faster post-recovery error correction (elevated gain-like signature) in recovered patients | **No existing dataset** (post-recovery psychophysiology cohorts are essentially absent). Would need new collection |

---

## 2. The anchor instruments — what each actually measures

- **TII (Temporal Integration Inventory)**, Melges, Tinklenberg, Hollister &
  Gillespie 1970 (*Temporal disintegration and depersonalization during
  marihuana intoxication*, Arch. Gen. Psychiatry 23:204–210,
  https://pubmed.ncbi.nlm.nih.gov/4916452/) — self-report temporal
  disintegration.  Used by Simeon 2007 in DPD; the closest published
  instrument for the model's S.
- **GDSA (Goal-Directed Serial Alternation)**, same group (see also Melges
  1971, *Marihuana and the temporal span of awareness*, Arch. Gen. Psychiatry
  24:564; Casswell & Marks 1973, Science 179:803,
  https://www.science.org/doi/10.1126/science.179.4075.803) — *behavioral*
  serial-coordination test; high oral doses of marihuana extract impaired
  serial coordination of cognitive operations, related to impaired immediate
  memory.  The only **objective** temporal-depth-channel instrument.
- **DES / DES-II** (Bernstein & Putnam 1986) — dissociation self-report;
  DES-T taxon (8 items) intended to isolate pathological dissociation.  The
  instrument of record for the report-side of the P5-corollary.
- **THC-exposure laboratory studies** — Melges 1970 (oral extract, dose
  variation); Mathew, Wilson & Melges 1992 (*Changes in the experience of time
  after marijuana smoking*, Ann. Clin. Psychiatry,
  https://scholars.duke.edu/display/pub806397: 35 volunteers, high/low potency
  vs placebo, 86% ≥ some temporal disintegration at high potency, peak 30 min)
  and Mathew et al. 1993 (*Depersonalization after marijuana smoking*, Biol.
  Psychiatry, https://pubmed.ncbi.nlm.nih.gov/8490070/: depersonalization
  maximal 30 min after high-potency cigarettes, not after placebo); D'Souza et
  al. 2004 (IV Δ9-THC 0/2.5/5 mg, n=22, double-blind crossover, transient
  psychotomimetic including dissociative symptoms,
  https://www.nature.com/articles/1300496); Colizzi 2019 (1.19 mg IV THC,
  depersonalization/derealization and time slowing in healthy volunteers,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6523579).  Escelsior 2025
  (https://link.springer.com/article/10.1007/s11469-023-01125-8) reviews:
  cannabis produces dose-dependent time overestimation.
- **DPDR neuroimaging** — Simeon et al. 2000 (*Feeling Unreal: a PET Study of
  Depersonalization Disorder*, Am. J. Psychiatry 157:1782,
  https://psychiatryonline.org/doi/10.1176/appi.ajp.157.11.1782): 8 DPD vs 24
  controls — **hypometabolism right superior/middle temporal gyri (BA 22/21),
  hypermetabolism parietal BA 7B/39 and occipital 19**; dissociation and
  depersonalization scores **positively correlated with area 7B metabolism**.
  Relevant to any future S-proxy neuroimaging arm (see loose mappings, §7).
- **Simeon, Hwu & Knutelska 2007** (*Temporal Disintegration in
  Depersonalization Disorder*, J. Trauma Dissociation 8:11–24,
  https://www.tandfonline.com/doi/abs/10.1300/J229v08n01_02): 52 DPD vs 30
  controls — TII significantly higher in DPD; TII correlates with DES total;
  **of the three DES domains (absorption, amnesia, depersonalization/
  derealization) only absorption predicted TII**; TII **not** associated with
  age of onset or duration of illness.

Two structural notes drawn from these instruments (both used in §3):

- Simeon 2007's absorption-mediation result is **directionally consistent with
  the model's S-dynamics**: in the model S tracks externality (1−a), i.e.
  sustained inward attention (absorption-like) is what drags S down — and in
  the data it is absorption, not DP/DR severity itself, that predicts
  temporal-disintegration scores.  This is a consistency between the model's
  causal path (a → S) and an independent regression — *not* a test of the
  dynamics (cross-sectional, and the a ↔ absorption bridge is itself loose).
- Simeon 2007's null TII-by-illness-duration association is **not** evidence
  against P6: that sample is cross-sectional and (mostly) still ill — P6
  predicts a *within-patient recovery-time* separation, which no published
  study has measured.

---

## 3. Per-prediction test specs

### 3.1 P1 — threshold dose-response (external: THC challenge)

**Claim transferred:** DPDR/temporal-disintegration onset is threshold-like in
a separable duration × intensity × affect dose space, with axes trading off
(model: at duration 100 affect is required, A-threshold 0.186; at A = 0 a
longer episode still collapses, duration threshold ≈ 150; affect is required
at short durations — equivalently affect lowers the duration/intensity
thresholds).

**Design.**  Within-subject, double-blind, dose ladder denser than published
(e.g. 5+ active levels across sessions), each session: TII-state + GDSA +
CADSS-DP at fixed post-dose timepoints.  Optional second arm: fixed THC dose ×
stress induction (or mCPP, Simeon 1995, which induced depersonalization
significantly more than placebo — in a mixed sample of 67 normal volunteers
and patients with OCD, social phobia, and borderline personality disorder,
not healthy volunteers only — via serotonergic anxiety challenge) to load
the affect axis.

**Concrete signature.**  Within-subject peak-response vs dose shows a steep
segment (step) rather than a proportionate line; the step location moves
predictably between arms (lower dose threshold under affect load).

**Falsifier.**  Smooth within-subject dose–response with no steep segment;
pure additivity (no cross-axis interaction).

**Data status.**  Existing studies have 2–3 dose levels and report **group
means** — group means cannot distinguish "graded mechanism" from "threshold
mechanism + between-subject threshold heterogeneity".  The published data are
*compatible with* P1 but do not select it.  New collection (or pooled
individual-level data reanalysis) is required.

### 3.2 P3 — minimum effective rescue dose (external: treatment dose)

**Claim transferred:** a sharp minimum effective engagement dose exists; the
required dose rises the longer the system has waited (u*: 0.40→0.95 as delay
30→600; no rescue at duration ≤ 30 past long waits at any u ≤ 1.0).

**Design.**  Session-level outcome tracking in DDD treatment (session-by-
session CDS/DES; dose = cumulative structured-engagement sessions or
ecological engagement measures), analyzed for (a) a minimum-dose boundary,
(b) dose requirement as a function of illness duration at treatment start.

**Existing data.**  Hunter 2023 (n = 36 completers,
https://www.tandfonline.com/doi/full/10.1080/16506073.2023.2255744) *assumes*
a ≥ 8-session minimum dose (so cannot test it), finds medium effect sizes
after mean 17.3 months of therapy, and shows no significant CDS change across
a mean 10.6-month pre-treatment wait (that null wait period is evidence
relevant to **P2**, not P3).  Hunter 2025 feasibility RCT
(https://pmc.ncbi.nlm.nih.gov/articles/PMC12801544) reports CDS decrease 16.88
(CBT) vs 5.5 (TAU).  Neither design resolves a threshold.  New collection
needed.

**Falsifier.**  Proportional session-dose response from the first session, or
success that does not rise monotonically with dose.

### 3.3 P4 — relapse as easy as first onset (external: relapse data)

**Claim transferred:** a recovered state (G ≈ 0.885, Θ_eff ≈ 0.76) buys **no
protection** against a renewed trigger: second-episode duration threshold
66.0 t.u. vs first-episode 66.3; collapse speed identical.

**Existing data (qualitative).**  The episodic-course literature matches the
model's escalation pattern: Simeon 1997 (30 cases, mean onset 16.1 y, "chronic
course... usually continuous but sometimes episodic", ~30% with episodes
"from minutes to a few years",
https://www.ovid.com/journals/ajps2/fulltext/00000465-199708000-00013);
Simeon 2003a (117 cases) and Baker 2003 (204 cases,
https://www.semanticscholar.org/paper/ded3513cabab109c6e101b44c2b33b9c0686d413)
— "typically symptoms are initially episodic, with episodes becoming longer
and more severe until pervasive and unremitting" (Medford, Sierra, Baker &
David 2005 review,
https://www.cambridge.org/core/journals/advances-in-psychiatric-treatment/article/6216AE06994D1094873145C016CC1F57);
~⅓ of patients run an episodic course.  Episodes **lengthening/severing** over
time is the direction P4 predicts (no protective margin accrues).  **Caveat:**
these are qualitative course descriptions, not dose measurements —
lengthening episodes are also consistent with kindling/sensitization
accounts, so this literature corroborates the escalation *direction* only
(see "What's missing" below for the dose comparison that would actually
discriminate).

**What's missing.**  No study anywhere measures, within patients, the trigger
dose required for episode 2 vs episode 1.  That comparison is the actual
quantitative content of P4.  New collection (or retrospective structured
relapse-history interview in episodic-course patients: trigger severity
ratings per episode) is the cheapest route.

**Falsifier.**  A growing protective margin — systematically larger triggers
needed for later episodes, or episode severity attenuating over successive
episodes.

### 3.4 P2 — no self-recovery (external: natural history)

Existing longitudinal data are **consistent**: Michal 2024 (Gutenberg cohort,
n = 522 depressed-with-DP/DR, 5-year follow-up,
https://pmc.ncbi.nlm.nih.gov/articles/PMC10924423) — 6.9% remission vs 15.9%
for depression without DP/DR.  **Caveat on Michal:** the remission is of
*depression* (PHQ-9 < 5), the sample is defined by CDS-2 ≥ 1 (a sub-clinical
threshold — clinical cutoff is 3), and the DP/DR group's median CDS-2 at
5 years is 1.0, below the clinical cutoff; it is therefore evidence about
sub-clinical DP/DR symptom persistence, weaker for P2's "no self-recovery
from *deep collapse*" than the raw 6.9%-vs-15.9% contrast suggests.  Baker
2003 — mean symptom duration over 12 years before first specialist contact;
Simeon 1997 — "highly treatment refractory".

But treatment-seeking samples cannot estimate *spontaneous* recovery rates
(selection bias), and an ethically feasible untreated prospective cohort does
not exist.  **P2 remains externally supported only weakly and cannot be
sharply falsified with existing or plausibly collectible data.**  (The model's
own 106/112 sensitivity result is the stronger evidence and is internal.)

### 3.5 P5-corollary — vigilance reports ≠ vigilance mechanism (external: existing data)

This is the corollary derived in `handoff-selfreg-vigilance`: a state variable
with a restoring force (g) cannot hold a persistent offset without an agent;
the architecture denies the agent; therefore reports of vigilance should
dissociate from measurable dynamic change.

**Existing data already point the way the corollary predicts:**

- **Scale-endorsed pathological dissociation without pathology:** Leavitt
  1999 (*Dissociative Experiences Scale Taxon and Measurement of Dissociative
  Pathology*, J. Clin. Psychol. Med. Settings,
  https://link.springer.com/article/10.1023/A:1026275916184) — DES-T
  false-positive rate 54% (DES 13%), only 35% of the sample classified
  correctly.
- **DPDR-like reports without impairment:** Pons et al. 2026
  (https://www.nature.com/articles/s41598-026-51014-y, N = 121 total:
  n = 60 meditation-triggered, 61 non-meditation-triggered) — 61.7% of the
  meditation-triggered group score above the CDS-70 clinical cutoff, yet
  only 4 participants in the whole study carry any DPDR diagnosis and
  cutoff-positive meditation-triggered subjects show current mental disorder
  at 10.8%; the same symptom profile as the non-meditation-triggered group
  with far less pathology.
- **Report/physiology dissociation in DDD:** Sierra, Senior, Dalton et al.
  2002 (reduced skin-conductance responses to unpleasant pictures in DPD,
  https://pubmed.ncbi.nlm.nih.gov/12215083); Millman et al. 2024
  systematic review (reduced subjective emotional responses with mixed
  autonomic findings,
  https://kclpure.kcl.ac.uk/portal/en/publications/behavioural-autonomic-and-neural-responsivity-in-depersonalisatio).
- **The boundary case, counted against the corollary:** Guralnik et al.
  2000 (https://pubmed.ncbi.nlm.nih.gov/10618020) — DPD patients were
  measurably worse on attention, short-term memory, and spatial reasoning
  despite comparable intellectual abilities.  Cognitive complaints there
  **track** objective impairment rather than dissociating from it; only the
  preserved-IQ channel dissociates.  This is *not* "reports without
  impairment" evidence and is retained only to mark the boundary of the
  report-vs-objective dissociation claim.

**Limit of the existing support.**  With the Guralnik boundary case
excepted, these show *symptom reports* dissociating from *impairment or
emotional responsivity*.  None measures the **post-recovery vigilance
channel** the model actually denies (persistent
elevated error-correction gain).  The decisive test — recovered-DPD vs matched
naive on gain-like dynamic measures (RT variability, error-correction
latencies, ERP components) at settled state — requires new collection.  The
model's exact commitment: **rescued == naive, zero persistent compensation of
any kind**; any persistently faster post-recovery correction falsifies it.

### 3.6 P6 — temporal-depth lags functional recovery (external: longitudinal TII/GDSA) — **weak / partly built-in**

**The 5× headline is largely an input, not an output.**  The model's
t90(S)/t90(G) ratio of ~5 is set by construction: the time constants
τ_S = 100 and τ_G = 20 (`dpdr/model.py`) were *chosen* with exactly this
5× separation, and `predictions.md`'s own sensitivity analysis concedes the
measured ratio "tracks τ_S almost exactly".  Presenting "~5× slower temporal
recovery" as the model's *prediction* would be circular — the model cannot
help but produce it.  What survives as genuine, non-built-in content is
only:

1. the **deviation from construction**: the measured ratio 5.41 exceeds the
   5.0 the constants alone dictate (t90 is not exactly τ), and
2. the **ordering claim**: temporal-depth measures normalize *after*
   functional recovery — direction, not magnitude.

Everything quantitative beyond that (including the [3.9, 8.3]
pre-registration interval below) inherits its width from the model's own
τ_S/τ_G perturbations — the band is "where the model cannot help but land",
not a mapping to human variation.  A rejection region that coincides with
the model's own sensitivity range is falsifiable only in the trivial sense
that a human ratio of ~1 or ~12 would fall outside it; it carries almost no
predictive content about the mechanism, and this dossier does not present
it as an independent test of the architecture.  P6 is therefore demoted to
the weak tier, and the remainder of this section registers only the residue.

**Design A (clinical cohort).**  DDD patients entering treatment, measured at
fixed intervals on: TII + GDSA (temporal-depth channel) and a functional
battery + CDS/DES-DP (functional/symptom channel).  Pre-registered statistic:
t90(temporal)/t90(functional), computed within patient.

**What is actually committed:** temporal-depth normalization **later than**
functional/symptom normalization (the ordering — the only component not set
by construction); secondarily, that the human ratio exceeds ~3.9, i.e. that
temporal-depth recovery is not merely detectably slower but several-fold
slower, as the constructed timescale separation implies.  The [3.9, 8.3]
band is recorded for completeness, flagged as the model's own
construction-derived envelope rather than an independent point prediction.

**Falsifier.**  Ratio ≈ 1 (synchronous normalization), or temporal measures
recovering *faster* than functional ones — either rejects the ordering, the
only non-built-in claim.

**Design B (pharmacological, cheaper).**  Within-subject THC challenge with
TII-state + GDSA + functional tasks at dense post-clearance timepoints.
D'Souza 2004-style protocols already measure symptoms and cognition at
multiple timepoints; **if individual-level timecourses from such studies are
shareable, the recovery-time ratio is computable from existing raw data
without any new drug administration.**  This is the single highest-value data
request available: reanalysis of existing challenge datasets.  (Caveat: those
studies measure ~3-hour intoxication windows, and GDSA was not administered —
a 5× t90 separation may exceed the observable window, so this route can
plausibly test the *ordering* only.)

**Why no existing publication answers it.**  Simeon 2007 is cross-sectional
(and its TII-duration null is uninformative for recovery dynamics, §2);
Mathew 1992/93 track only the intoxication phase; no study reports temporal-
depth and functional measures tracked together across recovery in the same
subjects.

### 3.7 Predictions with no external instrument at all

- **η axis (P1 note):** the cannibalization cost η has no proposed external
  correlate; its graded threshold behavior is a model-internal structure.
- **The numeric constants** (66.32 t.u., u* = 0.45–0.95, A = 0.1859): see §0 —
  dimensionless, uncalibrated, untestable as values.  Any claim that "66 t.u.
  equals N hours" would be numerology, not validation.
- **The like-for-like equality 66.0 vs 66.3** (P4): testable only in the
  ordering currency (episode-2 threshold ≤ episode-1), not as a ratio, since
  it compares two doses of the same axis — actually a ratio IS computable if
  within-patient trigger doses are measured (design in §3.3); the 0.5%
  difference itself is far below any thinkable measurement resolution, so the
  pre-registered human-scale claim is "no protective margin", i.e. ratio ≤ 1
  within confidence intervals.
- **P5's PSD/ringing component:** the model predicts its own failure here
  (D exogenous ⇒ no oscillator); the human-side analogue (post-recovery EEG
  spectral shift) is measurable but the model makes a *null* prediction, so
  it can only be confirmed trivially — weak evidence either way.

---

## 4. What this dossier does and does not buy

**Buys:**

1. **S is not idiosyncratic.**  The variable the whole P6 hysteresis result
   rests on maps onto a published, instrumented construct (temporal depth /
   temporal disintegration: Tolchinsky 2025; Madden & Serper 2026; TII/GDSA).
   Simeon 2007's absorption-mediation is directionally consistent with the
   model's a → S causal path.
2. **Two predictions have existing independent data pointing at them** —
   P5-corollary (report≠mechanism: Leavitt's DES-T false positives, Pons
   meditation cohort, DDD psychophysiology, with the framing caveats in §3.5)
   and, more loosely, P4's escalation *direction* (episodic-course series:
   episodes lengthen over time).  P4's support is qualitative course
   descriptions, not dose comparisons — no study measures episode-2 vs
   episode-1 trigger dose (§3.3), and lengthening episodes are equally
   consistent with kindling/sensitization accounts — so it is directional
   corroboration only, not "genuinely independent" in the strong sense.  The
   point stands that the model is not hermetically sealed inside its source
   report.
3. **A concrete data request exists** (reanalysis of within-subject THC
   challenge timecourses for the P6 ratio) that would not require new drug
   administration — though, per §3.6, that route can plausibly test the
   *ordering* only.
4. **The methodology is the published one.**  The anchor literature explicitly
   designates computer simulation as the testing instrument for this construct
   because direct human experimentation is infeasible — this project occupies
   the slot their program calls for.

**Does not buy:**

1. **Nothing here validates anything yet.**  This is a protocol.  Every
   genuinely quantitative external test except the P5-corollary's existing
   psychometrics requires data that does not exist.
2. **The mapping layer is untested.**  S ↔ temporal depth, G ↔ "functional
   measures", u_ext ↔ "engagement dose" are identifications made *here*, not
   bridges anyone has measured (§7).
3. **Absolute numbers are untransferrable** (dimensionless model, no
   calibration).  Only orderings, ratios, and shapes transfer — and of the
   model's headline numbers, only the **5.41 lag ratio** is a dimensionless
   quantity of the kind that could eventually be compared to a human
   statistic.  That comparison is weaker than it looks: the ratio's ~5×
   magnitude is set by the chosen τ_S/τ_G = 100/20 construction (§3.6), so
   the human statistic would test the residue (the 5.41-vs-5.0 deviation and
   the ordering), not an independent quantitative prediction.
4. **The model's most load-bearing internal caveat survives untouched:**
   ε0 = 0.25 is tuned, not derived, and both the stuck state and its rescue
   hinge on it (eps0 = 0 loses rescue entirely, eps0 ≥ 0.5 loses collapse).
   External validation of the architecture would not license that constant.
5. **Externally untestable now and for the foreseeable future:** P2's sharp
   claim (no untreated prospective cohort is ethically obtainable); the η
   axis; P5's spectral null.  These remain internal-only results.

---

## 5. Data-status summary

| Prediction | Existing data | Needs new collection |
|---|---|---|
| P6 lag ratio (**weak / partly built-in**, §3.6) | **None usable** (cross-sectional only) | Longitudinal cohort **or reanalysis of existing THC-challenge raw timecourses** |
| P1 threshold/sharpness | Dose–response at 2–3 levels, group means (compatible, non-selective) | Dense within-subject dose ladder; THC × affect cross |
| P1-a attention-dose | Onset-hours distribution published, no denominator | Recontact/new meditation cohort |
| P3 minimum rescue dose | Indirect (CBT session-dose assumptions) | Session-level outcome tracking |
| P4 no-protection relapse | Qualitative escalation (episodic→continuous course) | Within-patient episode-1 vs episode-2 trigger dose |
| P2 no self-recovery | Consistent chronicity/remission data (biased samples) | Not ethically obtainable sharply |
| P5-corollary report≠mechanism | **Yes — multiple independent literatures (DES-T false positives; meditation DPDR; DDD psychophysiology)** | Vigilance-channel-specific measurement (optional strengthening) |
| P5 original (no persistent gain) | None | Post-recovery psychophysiology cohort |
| η axis; numeric constants | None | None proposed — untestable externally |

---

## 6. Pre-registration commitments (frozen before any new data)

For any future externally collected or reanalyzed dataset, the model — at its
frozen parameters, with no refitting — commits to:

1. **(P6)** t90(temporal-depth channel) / t90(functional channel) ∈ [3.9, 8.3]
   in recovering DPDR, within-patient.  Rejection region: ratio < 2 or
   temporal faster than functional.  **Status of this commitment (§3.6): the
   interval's magnitude is set by construction (τ_S/τ_G = 100/20 = 5, a
   chosen constant pair) and its width comes from the model's own
   τ_S/τ_G sensitivity band, so the region is falsifiable only in the
   trivial sense that a human ratio far outside it (≈ 1, or ≫ 10) would
   reject; it carries little independent predictive content and is recorded
   as the model's construction-derived envelope.  The substantive
   pre-registered claim is the *ordering* — temporal-depth normalization
   strictly later than functional/symptom normalization — plus the deviation
   detail that the realized ratio (5.41) exceeds the constructed 5.0.**
2. **(P1)** Within-subject onset dose–response has a steep segment (step
   location moving under affect load); rejection: proportionate response with
   no steep segment, or no cross-axis interaction.
3. **(P3)** A minimum effective engagement dose with monotone success above
   it; rejection: proportionality from the first unit of dose.
4. **(P4)** Within-patient episode-2 trigger dose ≤ episode-1 trigger dose
   (no protective margin); rejection: episode-2 dose reliably and
   substantially greater, growing with recovery duration.
5. **(P5)** Recovered-DPD == naive on settled-state gain-like dynamic
   measures; rejection: any persistent faster-correction signature.
6. **(P5-corollary)** Dissociation/vigilance-report endorsement rates
   substantially exceed measurable-dynamic-abnormality rates in non-impaired
   samples; rejection: concordant report–mechanism contingency.

Any refitting of the model in response to such data must be logged as a **new
model version**, not as validation of this one.

---

## 7. Loose mappings — stated plainly, not papered over

1. **S ↔ temporal depth (TII/GDSA).**  The tightest of the mappings on paper
   and still only a construct-level identification.  The model's S is an
   *allostatic context tracker* that gates the collapse threshold (Θ_eff =
   Θ·S/S_rest); "temporal depth" in the literature is a phenomenological/
   experiential construct measured by self-report (TII) and serial task
   (GDSA).  Nobody has ever shown that TII/GDSA track an allostatic-gating
   variable.  The identification is *motivated* (both collapse under
   sustained inward attention; both outlast the trigger; Simeon 2007's
   absorption-mediation fits the a→S direction) but it is **assumed**.
2. **G ↔ "functional/reasoning measures."**  G is self-content production
   capacity in the model's internal currency; Guralnik-style batteries measure
   attention/memory/reasoning.  The bridge is plausible (the model's own P6
   correlate says functional recovery is the fast channel — though that
   ordering's *magnitude* is itself a constructed timescale separation, §3.6)
   but no instrument measures G.  P6's human test therefore *presupposes* the
   G-mapping to define its "functional channel" — the test is only as strong
   as this identification.
3. **a_hold ↔ absorption / focused attention.**  Needed for both the Simeon
   2007 consistency claim (§2) and the meditation data (P1-a).  Absorption as
   measured (Tellegen-style) is a trait/object engagement construct; a_hold is
   a controlled inward-attention drive.  Directionally aligned, quantitatively
   unlinked.
4. **u_ext ↔ "engagement dose" (therapy sessions, jobs, crisis).**  The
   model's u_ext is a dimensionless external-demand drive on attention.  CBT
   sessions are not units of u_ext.  P3's external test is shape-only
   (threshold existence, monotonicity), never level.
5. **THC challenge ↔ inward episode.**  The model's episode is attentional
   (a_hold) and endogenous; THC is pharmacological and has no representation
   in the model at all.  Using THC studies to test P1/P6 tests the *shared
   downstream construct* (temporal disintegration + depersonalization), not
   the model's a-mechanism.  If THC data supported a prediction, it would
   support the threshold/lag *structure*, not the attentional etiology.  This
   is the dossier's largest scope caveat.
6. **Entrapment (Madden & Serper) ↔ anything in the model.**  The time
   collapse–entrapment model's mediator — perceived entrapment — has **no
   state variable in this model**.  The convergence is at the level of "time
   collapse matters", not mechanism.  A future model version could put
   entrapment on the D (demand) axis, but that would be a new hypothesis, not
   a mapping.
7. **t.u. ↔ human time.**  No calibration exists or is claimed.  Every
   time-denominated number in `predictions.md` (66.32 t.u., τg/μ = 667, delay
   30–600) is internal.  Only ratios and orderings were carried into §6.

---

## 8. Sources

- Tolchinsky et al. 2025, Front. Psychol. 16:1585315, doi:10.3389/fpsyg.2025.1585315 — https://pmc.ncbi.nlm.nih.gov/articles/PMC12444765/
- Madden & Serper 2026, Psychiatry Research, S016517812600332X — https://www.sciencedirect.com/science/article/abs/pii/S016517812600332X (PMID 42288070)
- Melges, Tinklenberg, Hollister & Gillespie 1970, Arch. Gen. Psychiatry 23:204–210 — https://pubmed.ncbi.nlm.nih.gov/4916452/
- Melges 1971, Arch. Gen. Psychiatry 24:564; Casswell & Marks 1973, Science 179:803 — https://www.science.org/doi/10.1126/science.179.4075.803
- Mathew, Wilson & Melges 1992, Ann. Clin. Psychiatry — https://scholars.duke.edu/display/pub806397
- Mathew et al. 1993, Biol. Psychiatry — https://pubmed.ncbi.nlm.nih.gov/8490070/
- D'Souza et al. 2004, Neuropsychopharmacology 29:1558–1572 — https://www.nature.com/articles/1300496
- Colizzi et al. 2019 — https://pmc.ncbi.nlm.nih.gov/articles/PMC6523579/
- Escelsior et al. 2025 (cannabis–timing review) — https://link.springer.com/article/10.1007/s11469-023-01125-8
- Simeon, Guralnik et al. 1997, Am. J. Psychiatry 154:1107 — https://www.ovid.com/journals/ajps2/fulltext/00000465-199708000-00013
- Simeon et al. 2000, Am. J. Psychiatry 157:1782 — https://psychiatryonline.org/doi/10.1176/appi.ajp.157.11.1782
- Simeon, Hwu & Knutelska 2007, J. Trauma Dissociation 8:11–24 — https://www.tandfonline.com/doi/abs/10.1300/J229v08n01_02 (PMID 17409052)
- Baker et al. 2003, Br. J. Psychiatry 182:428–433 — https://www.semanticscholar.org/paper/ded3513cabab109c6e101b44c2b33b9c0686d413
- Medford, Sierra, Baker & David 2005 (review; course/duration data), Adv. Psychiatr. Treat. 11:92–100 — https://www.cambridge.org/core/journals/advances-in-psychiatric-treatment/article/6216AE06994D1094873145C016CC1F57
- Medford et al. 2003, Addiction 98:1731–1736 — https://pubmed.ncbi.nlm.nih.gov/14651505/
- Michal et al. 2024 (Gutenberg) — https://pmc.ncbi.nlm.nih.gov/articles/PMC10924423/
- Hunter et al. 2023 (CBT audit) — https://www.tandfonline.com/doi/full/10.1080/16506073.2023.2255744
- Hunter et al. 2025 (feasibility RCT) — https://pmc.ncbi.nlm.nih.gov/articles/PMC12801544/
- Pons et al. 2026 — https://www.nature.com/articles/s41598-026-51014-y
- Leavitt 1999, J. Clin. Psychol. Med. Settings — https://link.springer.com/article/10.1023/A:1026275916184
- Guralnik et al. 2000, Am. J. Psychiatry 157:103 — https://pubmed.ncbi.nlm.nih.gov/10618020/
- Sierra, Senior, Dalton, McDonough, Bond, Phillips, O'Dwyer & David 2002 (autonomic response), Arch. Gen. Psychiatry 59:833–838 — https://pubmed.ncbi.nlm.nih.gov/12215083/
- Millman et al. 2024 (systematic review) — https://kclpure.kcl.ac.uk/portal/en/publications/behavioural-autonomic-and-neural-responsivity-in-depersonalisatio (doi:10.1016/j.neubiorev.2024.105783)
- Deane, Miller & Wilkinson 2020, Front. Psychol. 11:539726 — https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2020.539726/full
- Masah et al. 2025 (psychedelic-associated DDD cases) — https://econtent.hogrefe.com/doi/10.1024/0939-5911/a000959

Model-side numbers: `predictions.md` (P1–P6 verdicts and measured values),
`README.md` (equations, parameters, documented deviations).  Nothing in either
file was modified by this dossier.
