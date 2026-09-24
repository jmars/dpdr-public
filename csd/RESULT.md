# CSD discriminator test — result

**Date:** frozen-model audit · **Scripts:** `~/thing/csd/` (`common.py`, `test1_fold.py`, `test2_discriminator.py`, `make_figure.py`) · **Figure:** `figs/f13_csd.png` (1552×1150, verified programmatically; panel ink fractions 0.33/0.09/0.13/0.08) · **Model:** `dpdr/model.py` untouched (`pytest tests/ -q` → **53 passed** after all runs; model.py md5 ba4691df15efede12d11314936c95d2e)

Setup: chronic inward drive = constant `a_hold = eps` (the same channel the model's own experiments use), equilibrium analysis by warm-started damped-Newton continuation on the exact RHS, Jacobians by finite differences at the equilibrium, eigenvalues exact (no log-fitting) — the perturbation-recovery protocol (perturb G, integrate, fit) is reported as the empirical cross-check.

---

## TEST 1 — the fold: settled

**The healthy branch's disappearance is a BORDER-COLLISION FOLD, not a saddle-node.**

- **Where:** the healthy equilibrium dies at **eps_c = 0.265192279**, and it dies **exactly on the cannibalization switching manifold** `E = Theta_eff` (residual |E − Theta_eff| = 1.0e-13 at the last resolvable point). This resolves the orchestrator's anomaly: the branch was never "stopping being findable" — it exists up to the knee and the knee is where the tanh-regime change pinches it off. Warm-start continuation at 1e-5 steps approaches it cleanly (E − Theta_eff: −3.4e-6 at eps 0.26519 → 0).
- **No eigenvalue reaches 0.** At the last resolvable point the healthy spectrum is **[−0.0015, −0.0055, −0.0088, −0.042702, −0.5978]** — every mode bounded away from zero by ≥ 0.0015. There is no smooth saddle-node: the equilibrium hits the non-smooth switching manifold (dc/dE jumps 0 → ~60 per unit at the knee) while still linearly stable. This is why the earlier grid search saw "findable at 0.265, gone at 0.266" — the death is a hard boundary in eps, not a soft approach to criticality.
- **The annihilating partner is the middle (saddle) equilibrium.** A third equilibrium exists at every eps in the bistable window (e.g. eps=0.20: G* = 0.0469 stable-collapsed / 0.2573 saddle with λ_dom = +1.577 / 0.3382 stable-healthy), sitting slightly on the c>0 side of the same manifold (E − Theta_eff = +0.0009 at eps 0.20, → +1.2e-6 as eps → eps_c). Healthy and saddle **coalesce at eps_c on the manifold** — healthy last solves at 0.265192279, saddle last solves at 0.265186670; the 5.6e-6 gap is the numerical shadow of the crossing, not a region where only one branch exists (an 81-point multistart hunt inside the gap finds only the healthy point). The saddle's unstable eigenvalue is still **+1.29** where it dies — it is not softening into the healthy branch either. **Pair annihilation with both spectra bounded away from zero = border-collision bifurcation** (Leine & Nijmeijer-class), the non-smooth analogue of a fold.
- **The other side: there is NO second fold.** Continuing the collapsed equilibrium in **decreasing** eps: it exists at eps = 0.20 → 0.0020 without event, and still exists at eps = −0.02, −0.05, −0.10 (where the "drive" is unphysical; structure check only). **The hysteresis window is therefore not fold-bounded on the low side: it is [0, eps_c) — unbounded below.** The collapsed attractor is present at every physical eps; the healthy attractor exists only below eps_c. Collapse is **irreversible by withdrawal of the drive alone** (consistent with the project's exp2 relapse phenomenology, where rescue requires external demand, not the passage of time). The orchestrator's earlier "bistability at eps 0.20–0.26" is confirmed and extended: bistability holds at *every* eps ≥ 0 where the healthy branch exists.

## TEST 2 — the discriminator: CONSTANT recovery, no slowing

Exact dominant eigenvalue of the Jacobian at the healthy equilibrium, across 12 eps values spanning 5 decades of distance-to-fold (2.65e-1 → 2.28e-6):

| eps | dist = eps_c − eps | λ_slowest (exact) | 1/λ (t.u.) | basin half-width (G) | protocol t95 | protocol λ_fit |
|-----|------|------|------|------|------|------|
| 0.00000 | 2.65e-1 | −0.001500 | 666.7 | 6.5e-1 | 27 | (n/a) |
| 0.10000 | 1.65e-1 | −0.001500 | 666.7 | 2.9e-1 | 49 | 0.063 |
| 0.20000 | 6.5e-2 | −0.001500 | 666.7 | 8.1e-2 | 65 | 0.047 |
| 0.24000 | 2.5e-2 | −0.001500 | 666.7 | 2.8e-2 | 69 | 0.044 |
| 0.25000 | 1.5e-2 | −0.001500 | 666.7 | 1.6e-2 | 70 | 0.043 |
| 0.25500 | 1.0e-2 | −0.001500 | 666.7 | 1.1e-2 | 70 | 0.043 |
| 0.26000 | 5.2e-3 | −0.001500 | 666.7 | 5.4e-3 | 70 | 0.043 |
| 0.26300 | 2.2e-3 | −0.001500 | 666.7 | 2.3e-3 | 71 | 0.043 |
| 0.26500 | 1.9e-4 | −0.001500 | 666.7 | 2.0e-4 | 71 | 0.043 |
| 0.26510 | 9.2e-5 | −0.001500 | 666.7 | 9.4e-5 | 71 | 0.043 |
| 0.26515 | 4.2e-5 | −0.001500 | 666.7 | 4.3e-5 | 71 | 0.043 |
| 0.26519 | 2.3e-6 | −0.001500 | 666.7 | — | 71 | 0.043 |

**Verdict: recovery is CONSTANT.** The dominant eigenvalue is **exactly −μ/τ_g = −0.0015** at every distance — measured spread across all 12 points: **0.0e+00** (bit-identical, because at any equilibrium dĠ = −μ(g−g0)/τ_g with |dE/dt| = 0 below the upregulation deadband, decoupling the g-row exactly; see "mechanism" below). The model's counter-prediction is confirmed against the CSD prediction: no slowing, at any proximity, down to 2e-6 of the fold.

Protocol cross-check: the perturbation-recovery protocol was run basin-aware (probe δ = 40% of the healthy-basin half-width) — t95 is **65–71 t.u. at every distance from 6.5e-2 to 4.2e-5** (flat to ±5% over 3+ decades), and the tail decay rate locks onto the constant g-loop mode (λ_tail → 0.0015). A fixed δ = 0.05 probe recovers for eps ≤ 0.20 and **collapses for eps ≥ 0.24** — the failure near the threshold is basin retreat, not slowing.

## TEST 3 — mode separation: NO CSD at all (not even masked)

| mode | eigenvector | eps = 0 | eps = 0.20 | eps = 0.26519 |
|------|------|------|------|------|
| g (adaptive gain) | pure g | −0.001500 | −0.001500 | −0.001500 |
| S (setpoint) | pure S | −0.005500 | −0.005500 | −0.005500 |
| D (demand) | D+G mix | −0.008800 | −0.008800 | −0.008800 |
| **G (generator) — the fold's mode** | pure G | −0.119781 | −0.047326 | **−0.042702** |
| a (attention) | pure a | −0.200000 | −0.500000 | −0.597800 |

The G-mode — the mode whose equilibrium the fold annihilates — is **flat near the fold: 0.042932 → 0.042702 over 3.4 decades of distance (×1.005, 0.53% total softening)**, where a saddle-node would predict ×47.7 (97.9%) softening over the same range. The far-field drift (0.1198 at eps=0 → 0.0473 at eps=0.20) is the ordinary monotone dependence of the generator's logistic stiffness on a*, already saturated by eps = 0.24; it is not a fold precursor (see the slope fits: +0.39 with the four far points, +0.0006 with the six near points — softening stops where the fold physics starts).

**Verdict: NO CSD — neither displayed nor masked.** The model has no critical mode approaching zero in any variable. The timescale-separation story ("the slowest mode buries the critical one") is thereby *sharpened but also demoted*: the slowest mode is constant, and there is no softening G-mode for it to bury.

## TEST 4 — exponent: NOT FITTABLE to a scaling law (honestly)

| fit | slope | r² |
|-----|------|------|
| slowest (g) mode, all points | −0.0000 | — |
| G-mode, all points | +0.0474 | 0.31 |
| **G-mode, near fold (last 6 points, 3.4 decades)** | **+0.0006** | 0.64 |
| G-mode, far field (first 4 points) | +0.3857 | 0.78 |

The saddle-node prediction is slope = +0.5. The near-fold slope is **+0.0006 — three orders of magnitude from the prediction**, with the near-fold data a better fit to *flat* than to any power law. No exponent is fittable because there is no scaling regime. (Nothing was tuned; these are the frozen parameters.)

## Mechanism (why the model behaves this way)

At any equilibrium: D* = D_base·β_D/(β_D·D_base+δ_D) = 0.5455 with the D-mode exactly −(β_D·D_base+δ_D)/τ_D = −0.0088; S-mode exactly −(k_s+λ_S)/τ_S = −0.0055; a-mode exactly −(k_in·(eps+χ·c*)·(1−a*)/a* + ρ_a)/τ_a (tracked in the table as a* rises with eps); and the g-row of the Jacobian is exactly [0,0,0,0,−μ/τ_g] because dE/dt = Ḋ − Ġ = 0 at equilibrium puts the upregulation term under its deadband (dEdt_ref = 0.002), decoupling the gain loop. Four of five modes are dead-exact constants or trivial functions of eps; the only mode that carries the fold (G) is a logistic-stiffness mode that stays bounded (its equilibrium is pinched off by the switching manifold, not softened to zero). The structure is what forbids CSD: **a non-smooth fold plus an exactly-constant slowest mode**.

## What replaces CSD in this model

The genuinely shrinking quantity near the fold is the **healthy basin half-width** (G* − G_saddle): 6.5e-1 → 2.0e-4, tracking the distance-to-fold almost exactly (log-log slope ≈ 1 near the fold: basin ≈ distance). The model's real early-warning signal is **basin retreat** — a fixed-size perturbation starts collapsing at a sharp threshold (δ = 0.05 fails between eps 0.20 and 0.24) — with **no warning in recovery rates**. For the empirical protocol this is a strong, falsifiable counter-prediction: if the real meditation-approach/DPDR transition shows recovery-time slowing (CSD), the model is wrong about the transition type; the model instead predicts a sharp basin boundary with constant settling times on either side of it.

## Summary verdicts

1. **Fold:** border-collision fold at **eps_c = 0.265192279** on the switching manifold E = Theta_eff; no eigenvalue → 0; annihilates healthy + saddle (saddle's λ = +1.29 at death). **No lower fold of the collapsed branch** (exists at all eps ≥ 0, and beyond) ⇒ hysteresis window is [0, eps_c) — collapse irreversible by drive withdrawal alone.
2. **Discriminator:** recovery time **constant** (667 t.u.; spread exactly 0). No slowing. Protocol confirms (t95 flat 65–71; fixed-δ probe fails by basin retreat, not slowing).
3. **CSD:** **NO CSD AT ALL** — the G-mode does not soften near the fold (+0.0006 vs the required +0.5 slope), so CSD is neither displayed nor masked.
4. **Exponent:** not fittable (no scaling regime); near-fold slope +0.0006 ≪ 0.5.

Negative results stated plainly, as required: no eigenvalue reaches zero anywhere; no mode softens; no scaling law exists to fit. The model cannot display CSD because it does not possess it.
