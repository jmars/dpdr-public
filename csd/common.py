"""Shared numerics for the CSD discriminator test (frozen dpdr model).

All equilibrium analysis is done on the EXACT autonomous system defined by
dpdr.model.deriv under a constant chronic inward drive: the schedule has a
single channel 'a_hold' = eps on [0, inf).  Nothing in the dpdr project is
modified; this package only imports it.

Conventions
-----------
state y = [a, G, D, S, g] (as in dpdr.model.deriv)
chronic drive eps: Schedule({'a_hold': [(0.0, T, eps)]}) with T -> inf
equilibrium: F(y; eps) = deriv(inf, y, Params(), sched(eps)) = 0

Analytic structure used for validation (see RESULT.md):
  at ANY equilibrium, dg/dt row of the Jacobian = [0, 0, 0, 0, -mu/tau_g]
  because dE/dt = dD/dt - dG/dt = 0 at equilibrium, so the upregulation
  term pi*max(0, |dE|-ref) is inactive iff the equilibrium has |dE/dt|=0,
  and d(dg/dt)/dg = -mu/tau_g.  One eigenvalue is thus ALWAYS exactly
  -mu/tau_g = -0.0015 (timescale 667 t.u.).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

# Derived from this file's own location, NOT hardcoded, so the artifact runs
# wherever it is unpacked: <bundle>/csd/common.py -> bundle root, and
# <bundle>/dpdr is the directory that makes "import dpdr.model" resolve.
_ARTIFACT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ARTIFACT_ROOT, "dpdr"))

from dpdr.model import Params, Schedule, deriv  # noqa: E402

BIG_T = 1e6  # schedule span -> "infinite" chronic drive


def sched(eps: float) -> Schedule:
    """Chronic inward attention a_hold = eps, held forever."""
    return Schedule({"a_hold": [(0.0, BIG_T, eps)]})


def F(y: np.ndarray, eps: float, p: Params | None = None) -> np.ndarray:
    """RHS at t=inf under chronic drive eps (autonomous by construction)."""
    p = p if p is not None else Params()
    return np.array(deriv(BIG_T * 0.5, y, p, sched(eps)))


def jac(y: np.ndarray, eps: float, p: Params | None = None,
        h: float = 1e-7) -> np.ndarray:
    """Forward-difference Jacobian dF/dy at y (columns = dF/dy_i)."""
    p = p if p is not None else Params()
    y = np.asarray(y, float)
    J = np.zeros((5, 5))
    f0 = F(y, eps, p)
    for i in range(5):
        yp = y.copy()
        yp[i] += h
        J[:, i] = (F(yp, eps, p) - f0) / h
    return J


def eig(y: np.ndarray, eps: float, p: Params | None = None) -> np.ndarray:
    """Eigenvalues of the Jacobian, sorted slowest-first (most positive first).
    Complex eigenvalues return as a python complex when imag is non-negligible
    (the 5-state system is real, so they come in pairs; we keep numpy's)."""
    w = np.linalg.eigvals(jac(y, eps, p))
    return w[np.argsort(-w.real)]


def is_stable(y: np.ndarray, eps: float, p: Params | None = None,
              tol: float = 1e-8) -> bool:
    return bool(np.max(eig(y, eps, p).real) < tol)


def residual(y: np.ndarray, eps: float, p: Params | None = None) -> float:
    return float(np.linalg.norm(F(y, eps, p), ord=2))


def settle(eps: float, y0: np.ndarray, T: float = 60000.0,
           p: Params | None = None) -> np.ndarray:
    """Integrate the chronic-drive system from y0 for T t.u.; return y(T).
    (Used only as an independent check / IC generator; equilibrium analysis
    is root-finding, not settling.)"""
    p = p if p is not None else Params()
    sol = solve_ivp(deriv, (0.0, T), np.asarray(y0, float), args=(p, sched(eps)),
                    method="RK45", rtol=1e-8, atol=1e-10, max_step=5.0)
    return sol.y[:, -1]


def eigdesc(w: np.ndarray) -> str:
    def fmt(x):
        return f"{x.real:+.6f}{x.imag:+.6f}j" if abs(x.imag) > 1e-9 else \
            f"{x.real:+.6f}"
    return "[" + ", ".join(fmt(x) for x in w) + "]"


def eig_real_sorted(w: np.ndarray) -> np.ndarray:
    return np.sort(w.real)


# --------------------------------------------------------------- equilibrium
def equilibrium(eps: float, y0: np.ndarray, p: Params | None = None,
                maxiter: int = 400, tol: float = 1e-12) -> np.ndarray | None:
    """Damped Newton on F(y;eps)=0 from y0.  Returns None on failure.

    Exact finite-difference Jacobian every step (5x5, cheap); step halved
    while ||F|| fails to decrease; gives up when even the full Newton step
    cannot reduce the residual."""
    p = p if p is not None else Params()
    y = np.asarray(y0, float).copy()
    f = F(y, eps, p)
    fn = np.linalg.norm(f)
    for _ in range(maxiter):
        if fn < tol:
            return y
        B = jac(y, eps, p)
        try:
            dy = np.linalg.solve(B, -f)
        except np.linalg.LinAlgError:
            return None
        t, ok = 1.0, False
        for _ in range(40):
            yn = y + t * dy
            fnn = np.linalg.norm(F(yn, eps, p))
            if fnn < fn:
                ok = True
                break
            t *= 0.5
        if not ok:
            return None
        y, f, fn = yn, F(yn, eps, p), fnn
    return None


def bisect_zero(f, lo, hi, xtol: float = 1e-12):
    """Plain bisection on a sign change; returns (x, f(x))."""
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise ValueError(f"no sign change on [{lo}, {hi}]")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0 or (hi - lo) < xtol:
            return mid, fm
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi), f(0.5 * (lo + hi))
