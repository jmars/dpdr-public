#!/usr/bin/env python3
"""Substrate leg test: does the reuse-vs-expansion blowup appear in a THIRD
formalism (Datalog provenance), or is it specific to proof theory / lambda?

Model of the phenomenon (the thing being tested for substrate-independence):
  REUSE BY REFERENCE  vs  FULL EXPANSION (re-derivation at every use).

Measured in each formalism as  compact size  vs  expanded size  for a target,
as a function of nesting depth d (or a graph parameter), where a sub-result is
reused k times per level.

  (1) CUT-ELIMINATION:  compact = 1+k*d        expanded = k^d
  (2) LAMBDA INLINING:  same arithmetic (beta-reduction of a let-chain)
  (3) DATALOG PROVENANCE: circuit (shared sub-derivations) vs formula
      (every derivation written out) -- measured here on a real Datalog program

Control: a LINEAR program, where no sub-result is reused. If it also blows up,
the effect is a measurement artifact rather than a property of reuse.
"""
import sys
from functools import lru_cache
sys.setrecursionlimit(100000)


# ------------------------------------------------------------------ graphs
def ladder(n):
    """Levels 0..n, two nodes per level; from any node you reach EITHER node of
    the next level. Heavy reuse: paths (0,s)->(n,t) number 2^n."""
    E = set()
    for i in range(n):
        for s in (0, 1):
            for t in (0, 1):
                E.add(((i, s), (i + 1, t)))
    return E


def chain(n):
    """Linear control: exactly one edge per level. No reuse, one path."""
    return {((i, 0), (i + 1, 0)) for i in range(n)}


# ------------------------------------------- Datalog: transitive closure
def derivable_pairs(E):
    """CIRCUIT: the set of distinct facts (pairs) with a derivation.

    Semi-naive fixpoint of  path(X,Y) :- edge(X,Y).
                            path(X,Z) :- path(X,Y), path(Y,Z).
    Each distinct fact is ONE node, however many ways it can be derived."""
    facts = set(E)
    changed = True
    while changed:
        changed = False
        new = set()
        for (a, b) in facts:
            for (c, d) in facts:
                if b == c and (a, d) not in facts:
                    new.add((a, d))
        if new:
            facts |= new
            changed = True
    return facts


def path_count(E, s, t):
    """FORMULA side: number of distinct derivations of path(s,t) after full
    expansion -- i.e. paths in the graph, counted by DP over levels (well-founded).

    Uses the level ordering so the recursion terminates."""
    lvl = lambda x: x[0]
    adj = {}
    for (a, b) in E:
        adj.setdefault(a, set()).add(b)

    @lru_cache(maxsize=None)
    def P(a, b):
        if lvl(a) >= lvl(b):
            return 1 if b in adj.get(a, ()) else 0
        total = 1 if b in adj.get(a, ()) else 0
        for c in mids[a[0]:]:
            if c in adj.get(a, ()):
                total += P(c, b)
        return total

    mids = sorted({x for e in E for x in e})
    return P(s, t)


def path_count_simple(E, s, t):
    """Number of walks s->t, DP by level (adjacency-matrix style)."""
    nodes = sorted({x for e in E for x in e})
    adj = {}
    for (a, b) in E:
        adj.setdefault(a, []).append(b)
    # count paths by DP from s
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def P(x):
        if x == t:
            return 1
        return sum(P(y) for y in adj.get(x, ()))
    return P(s)


# ----------------------------------------------------- other two formalisms
def cut_elim(d, k):
    """Cut-elimination: compact (lemmas kept) vs cut-free (all inlined)."""
    return 1 + k * d, k ** d


def lambda_inline(d, k):
    """Lambda: let-chain kept vs fully beta-reduced. Same reuse arithmetic."""
    return 1 + k * d, k ** d


if __name__ == "__main__":
    print("=" * 72)
    print("(3) DATALOG PROVENANCE  --  the candidate NEW leg")
    print("=" * 72)
    print()
    print(" NON-LINEAR (ladder): heavy reuse of sub-derivations")
    print("   n | circuit: distinct facts | formula: derivations of (0,0)->(n,0)")
    for n in [2, 4, 6, 8, 10, 12, 14]:
        E = ladder(n)
        circuit = len(derivable_pairs(E))
        formula = path_count_simple(E, (0, 0), (n, 0))
        print(f"  {n:3d} | {circuit:24d} | {formula}")
    print()
    print(" LINEAR control (chain): NO reuse")
    print("   n | circuit: distinct facts | formula: derivations of (0,0)->(n,0)")
    for n in [2, 4, 6, 8, 10, 12, 14]:
        E = chain(n)
        circuit = len(derivable_pairs(E))
        formula = path_count_simple(E, (0, 0), (n, 0))
        print(f"  {n:3d} | {circuit:24d} | {formula}")
    print()
    print("=" * 72)
    print("(1)+(2) CUT-ELIMINATION and LAMBDA INLINING (k=2), same sweep")
    print("=" * 72)
    print("   d | cut compact | cut expanded | lam compact | lam expanded")
    for d in [2, 4, 6, 8, 10, 12, 14]:
        c1, x1 = cut_elim(d, 2)
        c2, x2 = lambda_inline(d, 2)
        print(f"  {d:3d} | {c1:11d} | {x1:12d} | {c2:11d} | {x2:12d}")
