#!/usr/bin/env python3
"""Cognition as generation + cut reduction: the WINDOW has a provable basis.

User's pre-event design:  GENERATOR = cut introduction (produce lemmas),
REDUCER = cut reduction (eliminate cuts, inline lemmas).

Two classical facts about cut-elimination (Gentzen):
  (A) a cut/lemma ABBREVIATES a repeated subproof   -> compactness BENEFIT
  (B) eliminating it INLINES those copies           -> size/compute COST

If generation buys (A) and reduction pays (B), there is an OPTIMUM in how much
intermediate (lemma/self-content) structure to keep. That is the window, and it
is a THEOREM-BASED result, not a fitted curve.

Model: build a nested-lemma proof where level d's lemma is used k times, so a
cut-free expansion multiplies by k at every level.
"""
import sys
sys.setrecursionlimit(100000)


def compact_size(depth, k):
    """Size counting lemma REFERENCES as 1 (the with-cuts proof).

    Each level contributes: 1 lemma definition + k use-sites.
    """
    return 1 + k * depth


def expanded_size(depth, k, memo=None):
    """Size after FULL cut-elimination (every lemma inlined).

    Level d uses level d-1 at k sites; eliminating multiplies.
    """
    # cost(d) = 1 + k * cost(d-1),  cost(0) = 1
    c = 1
    for _ in range(depth):
        c = 1 + k * c
    return c


def eliminate_work(depth, k):
    """Reduction WORK = number of substitution steps to reach cut-free form.

    Each of the k sites must receive a copy of the previous level's proof:
    work(d) = k * work(d-1) + k   (copy + splice per site)
    """
    w = 0
    for _ in range(depth):
        w = k * w + k
    return w


def retention_value(depth, k, leak=0.0):
    """A crude utility: modularity won from sharing, minus inlining paid.

    benefit ~ log of the compression achieved (how much is being abbreviated)
    cost    ~ the elimination work that must be performed
    """
    comp = compact_size(depth, k)
    exp = expanded_size(depth, k)
    sharing_benefit = exp / comp            # how much the lemmas abbreviate
    return sharing_benefit


if __name__ == "__main__":
    print("GENERATION (lemmas) buys compactness; REDUCTION (cut-elimination) pays.")
    print("depth d = how much intermediate (incl. self-referential) structure is kept")
    print()
    for k in [2, 3]:
        print(f"--- branching factor k = {k} (how often a lemma is reused) ---")
        print("   d |   compact (with cuts) |  expanded (cut-free) |  elim. work")
        for d in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
            c = compact_size(d, k)
            e = expanded_size(d, k)
            w = eliminate_work(d, k)
            print(f"{d:4d} | {c:21d} | {e:20d} | {w:11d}")
        print()
    print("READING: compact size is LINEAR in depth; expanded size and elimination")
    print("work are EXPONENTIAL in depth. So KEEPING lemmas is cheap and INLINING")
    print("them is where the cost lives -- the reducer's cost grows exponentially")
    print("with the depth of intermediate structure it must eliminate.")
