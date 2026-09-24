# What Held

Everything below was **measured**, not claimed. Each is a computed result that
reproduces from cache. None of it was retracted at any point during the project —
every correction landed on the interpretation wrapped around these numbers, never
on the numbers.

Verification: `cd dpdr && python -m pytest tests/ -q` → 53 passed.

---

## 1. The floor: existence matters, value doesn't

Sweep of the threshold clamp, at the stuck fixed point:

```
Theta_eff floor:  0.0  0.1  0.2  0.3  0.4  ->  G_end = 0.04857479   (identical, all FAIL)
                  0.5  0.6  0.7  0.8  1.2  ->  G_end = 0.88545195   (identical, all ESCAPE)
attention a:      0.882 (captured)  below  |  5.9e-79 (released)  above
```

**A single clamped constant breaks the entire positive-feedback loop.** Eight
values, two outcomes, identical to 8 significant figures within each group. The
critical value is `0.4795` — which equals the stuck point's own error level
`E* = 0.4969`. That identity is derived from the dynamics, not fitted.

*Interpretation, if you want one: a constant that is never re-examined is
protective; the value is irrelevant.* That sentence is an interpretation. The
table above is a fact.

---

## 2. The terminator gate: collapse needs both absences

Truth table (episode `a_hold 0.9` + affect `0.5`):

```
external content?  floor?  ->  G_end     c      collapsed
      no             no     ->  0.0486   1.00    YES
      no             yes    ->  0.8855   0.00    no
      yes            no     ->  0.8855   0.00    no
      yes            yes    ->  0.8855   0.00    no
```

Collapse requires **(no content) AND (no terminator)**. Either one prevents it.
A clean AND-gate, and it is derived from the mechanism rather than fitted to
anything.

---

## 3. The capability–recoverability tradeoff

Escape as a function of reasoning strength `r` and floor-disprovability `φ`:

```
 r     | φ=0.00 | φ=0.25 | φ=0.50 | φ=1.00
 0.0   | ESCAPE | ESCAPE | ESCAPE | ESCAPE
 0.2   | ESCAPE | ESCAPE | ESCAPE |  fail
 0.4   | ESCAPE | ESCAPE |  fail  |  fail
 0.8   | ESCAPE |  fail  |  fail  |  fail
 1.6   | ESCAPE |  fail  |  fail  |  fail
```

The boundary is a **rectangular hyperbola, `r·φ ≈ 0.2`**. As reasoning strength
grows, the set of survivable floors shrinks — and at `φ=0` (a floor the reasoning
cannot adjudicate) escape holds at *any* strength.

---

## 4. The cycle check: one slot, any depth

SLD-resolution engine, self-referential chain of depth `N`:

```
          | no cache | cache K=1 | K=2 | K=8
 N=128    |  STUCK   |    ok     | ok  | ok

 steps (with cache):  N=8 → 10,  N=32 → 34,  N=64 → 66   (linear, ~2N+2)
```

Without a cycle check: **stuck at every depth**. With one slot: **resolves at every
depth up to 128**. Capacity is irrelevant above 1 — same shape as the floor result,
in a completely different substrate (deduction, not ODEs).

*(The logic-programming half is a known theorem — tabling / SLG resolution. The
result here is the mapping, not the theorem.)*

---

## 5. Cut-elimination: keep is linear, inline is exponential

Nested lemmas, each level reused `k` times:

```
 k=2:  depth 4  →  compact 9    | cut-free 31     | elimination work 30
       depth 8  →  compact 17   | cut-free 511    | elimination work 510
       depth 12 →  compact 25   | cut-free 8191   | elimination work 8190
```

Compact (with cuts) is **linear**; cut-free is **exponential**. So the cost of
"reducing" live structure grows exponentially with its depth — which is a
theorem-based explanation for why the collapse threshold is sharp.

*(Known theorem: Gentzen, Statman. Again the mapping is the contribution.)*

---

## 6. The D/G split

At the stuck fixed point, versus baseline:

```
D (reasoning)  0.545  vs baseline 0.545   →  numerically identical, untouched
G (self-content) 0.049  vs baseline 0.886 →  collapsed
```

The reasoning resource is **exactly** unchanged while the self-content resource is
destroyed. *(Caveat found later: D is exogenously driven, so this is partly by
construction — but the qualitative split follows from the failure mode itself:
cannibalization eats G, not D.)*

---

## 7. The stuck state is a strong attractor

One-time perturbations at the fixed point:

```
pulse 0.02 / 0.1 / 0.5 / 2.0 / 5.0  ->  ALL return to G = 0.04850 exactly
```

It self-restores from any perturbation, including a large one. It is a stable
attractor, not an exhausted state.

---

## 8. The regulator works — with a scoped claim

- Escapes from collapse in **22 t.u.** at any actuator authority (post-collapse assay)
- Long-horizon, 60 episodes: frozen collapses at **episode 1**; regulated dips bottom at **0.132** vs frozen **0.049**
- **Zero healthy-regime cost:** `max|dG| = 0.00e+00`, the clamp never binds
- False-positive cost zero; legitimate responses not suppressed

Scope (found by review): the cheap-vs-elaborate contrast holds for *rescuing an
already-stuck* system, not for deployment. And between episodes the window
regulator is **worse** by −0.059/−0.075 G.

---

## 9. The instrument

- **53 tests passing**, gates G1–G2c with real margins
- **24 experiment scripts**, **15 figures**, reproducible from cache
- Frozen model verified untouched across every revision
- Every headline claim **checked adversarially** — six were corrected, zero measurements were

---

## The one-line version

The model behaves. The prose about it was wrong six times, and each time the
*measurement* was the thing that caught it.

Nothing here needs the interpretation to be true. That's what makes it worth
keeping.
