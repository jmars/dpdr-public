# dpdr — control-theoretic self-regulation model with a depersonalization failure mode

Five-state ODE model of a self-regulation loop (generator / reducer / control
loop) whose failure mode — collapse of the generator into a self-maintaining
"no-pull" state — reproduces the depersonalization cascade: sustained inward
attention weakens the generator, error rises past a threshold, the reducer
cannibalizes self-content, capture pulls attention further inward, and the
system sticks.  External demand (a job, a crisis, a conversation) pulls
attention outward and lets the generator regrow.

Full plan and source brief: held separately from this artifact.

## Model

States (all dimensionless, time in units of τa = 1):

| Var | Meaning |
|---|---|
| a(t) | attentional focus: 0 = fully external, 1 = fully inward |
| G(t) | generator strength (self-content production capacity) |
| D(t) | reducer demand |
| S(t) | allostatic setpoint (context tracker; temporal-depth proxy) |
| g(t) | adaptive control gain |

Algebraic: error `E = D − G`; cannibalization switch
`c = σc·max(0, tanh((E − Θeff)/w))` with `Θeff = Θ·S/S_rest`;
disconnection `u_loop = G/(D + ε)`.

```
dG/dt = (1/τG)·[ βG·(1−a)·G·(1−G) − αG·a·G − η·c·G − γG·G
                 + g·tanh(E/Es)·(G + ε0)·(1−G) ]
dD/dt = (1/τD)·[ (D_base + A(t))·βD·(1−D) − δD·D ]
dS/dt = (1/τS)·[ k_s·((1−a) − S) − λS·(S − S_rest) ]
da/dt = (1/τa)·[ κin·(a_hold(t) + χ·c)·(1−a) − κext·u_ext(t)·a − ρa·a ]
dg/dt = (1/τg)·[ π·max(0, |dE/dt| − ref) − μ·(g − g0) ]
```

Time constants (four-timescale separation): τa=1, τG=20, τD=50, τS=100, τg=200.

`ref` = `dEdt_ref` = 0.002 is a deadband on the gain-upregulation term: g
rises only when |dE/dt| exceeds 0.002 (so a settled loop does not accumulate
gain from numerical drift), and relaxes toward g0 with time constant
τg/μ ≈ 667 otherwise — which is why g carries no persistent offset after a
rescue (predictions.md P5).

## Documented deviations from the plan's literal equations

The plan (§3) marked its default parameters "starting values; tuned against
gates in Step A", and §7 risk #6 anticipated that healthy stability and
collapse vulnerability live in tension.  Five structural changes were needed
for gates G1–G2c to pass simultaneously; each is documented in `sim0.py`:

1. **Generator turnover `−γG·G`** — without a maintenance cost the healthy
   attractor is exactly G=1, which violates G1's strict `0 < G* < 1`.
2. **Control gated by `(G + ε0)`** — the plan's `g·tanh(E/Es)·(1−G)` keeps a
   finite upward pull at G=0, so the "stuck / no self-recovery" attractor of
   G2b does not exist.  ε0 = 0.25 sets the stuck floor (~0.3·ε0 < 0.1).
3. **S tracks externality `(1−a)`, not `a`** — the plan's literal sign
   contradicts its own prose ("chronic inward context drags S down").  With
   externality tracking, inward episodes lower S → lower Θeff → the collapsed
   state is self-consistent.
4. **`|dE/dt|_filt` → instantaneous `|dE/dt|`** — τg = 200 already filters;
   keeps the state vector at the plan's five.
5. **LSODA fallback threshold per segment** (`nfev > max(5000, 60·len)`) — a
   bare `nfev > 5000` fires on every long segment by construction under
   `max_step = 0.5` (~12 nfev/t.u. is the healthy RK45 rate).

Tuned defaults (from §3 starting values): βG 1.0→3.0, αG 0.6→1.2, ρa 1.0→0.2,
κext 2.0→4.0, k_s 0.3→0.5.  A ±20% one-at-a-time probe on the eight
load-bearing parameters leaves all gates passing.

## Layout

```
dpdr/
├── sim0.py            Step A single-file milestone (scenarios + gates + PNGs)
├── dpdr/              Step B package
│   ├── model.py       Params dataclass, Schedule, deriv (RHS)
│   ├── events.py      schedule builders (inward episode, affect pulse, rescue)
│   ├── integrate.py   segmented solve_ivp (RK45, LSODA fallback, collapse event)
│   ├── metrics.py     classify / relapse / rescue-success / gate checks
│   ├── sweep.py       grid sweeps + .npz cache
│   ├── permissive.py  opt-in: two-axis permissive AND-gate (exp5)
│   ├── regulator.py   opt-in: threshold floor + actuator + monitoring cost (exp6)
│   ├── window.py      opt-in: bounded self-simulation horizon (exp7)
│   ├── generational.py opt-in: persistent compounding M over generations (exp12)
│   └── plots.py       figure builders (f01–f08)
├── experiments/       Step C drivers (plus pre-registrations exp12/15/16 and the
│   │                  _gen_scan{,2,3}.py design scans behind the generational line;
│   │                  drivers exp4–exp16 for the opt-in variants and later assays)
│   ├── common.py      schedule helpers, plan-text rescue taxonomy
│   ├── exp1_failure_threshold.py   Phase 1/2: bifurcation + regime map (f04/f05)
│   ├── exp2_rescue.py              Phase 3/4: rescue taxonomy + recurring-episode
│   │                               relapse + hysteresis (f06/f06b/f08)
│   ├── exp3_postrecovery.py        Phase 4: PSD/damping/gain-asymptote/lag (f07 series)
│   └── sensitivity.py              §6b: OAT ±20/±30 % over P1–P6 measures
├── tests/             gate regressions (G1–G2c) + classifier unit tests
├── predictions.md     Step D: P1–P6 verdicts + falsifiers + sensitivity, plus the
│                      opt-in-variant sections (permissive AND-gate, regulator battery)
├── figs/              generated PNGs (paper figures and the exp12/15/16 experiment
│                      figures f16/f19/f20 cited in Appendix A)
└── cache/             sweep caches
```

## Environment / run

```
python3 -m venv .venv
.venv/bin/pip install numpy scipy matplotlib
.venv/bin/python sim0.py                     # Step A: scenarios + gates + PNGs
.venv/bin/python -m pytest tests/ -q         # Step B: gate regressions
.venv/bin/python -m experiments.exp1         # Step C (also exp2, exp3,
                                             # sensitivity; ~2–10 min each)
```

## Gates (plan §4c) — current status

| Gate | Statement | Status |
|---|---|---|
| G1 | baseline settles 0 < G* < 1, E < Θ, c ≈ 0 | PASS (G*=0.886, E*=−0.340, c*=0) |
| G2 | inward episode → stuck (G<0.1, E>Θ, c>0.5) | PASS (G=0.049, E=+0.499, c=1.0) |
| G2b | no self-recovery ≥ 10·τG after episode | PASS (max G = 0.050) |
| G2c | rescue at t=800 recovers G > 0.5, stays | PASS (G(T)=0.886, min post-rescue 0.885) |
| G3 | rescue map monotone success region | PASS (sharp monotone boundary; no patchwork) |
| G4 | post-rescue ringing (PSD shift) and g* > g0 | **FAIL** — no ringing (E-loop overdamped: D exogenous ⇒ no oscillator; PSD shift = 0) AND no persistent g\* > g0: g decays to g0 exactly (fitted offset C ≈ 0, τ = τg/μ = 667); the falsifier "rescued == naive gains" is triggered at steady state.  See predictions.md P5. |

Prediction verdicts (Step D, `predictions.md`): **P1 PASS · P2 PASS ·
P3 PASS · P4 PASS under recurring episodes (relapse reachable at frozen
parameters via a second inward episode after rescue — `figs/f06b_relapse.png`,
exp2 Phase 3c; absent under single-episode schedules) · P5 FAIL ·
P6 PASS (t90(S)/t90(G) = 5.41 ≈ τ_S/τ_G)**.
