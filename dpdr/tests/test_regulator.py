"""Regression tests for dpdr.regulator (review handoff-selfreg-reg-review).

Covers: the t_engage-on-a-breakpoint crash (finding 3), the frozen-wrap
fidelity, and the scoping-critical always-armed result (finding 1).
"""
import numpy as np
import pytest

from dpdr.events import baseline_schedule, failure_schedule, rescue_schedule
from dpdr.integrate import simulate
from dpdr.model import Params
from dpdr.regulator import RegulatorParams, floor_theta_eff, simulate_reg

T_BASE, T_FAIL, T_RESC = 1000.0, 600.0, 1400.0


@pytest.fixture(scope="module")
def p():
    return Params()


OFF = RegulatorParams(floor=None, k_pull=0.0, c_mon=0.0)


# ------------------------------------------- finding 3: breakpoint crash
class TestEngageOnBreakpoint:
    """t_engage coinciding with a schedule breakpoint used to raise
    TypeError (duplicate edge -> zero-length segment -> sol.y a list)."""

    @pytest.mark.parametrize("t_engage", [100.0, 160.0, 200.0])
    def test_no_crash_on_breakpoint(self, p, t_engage):
        s = simulate_reg(p, RegulatorParams(floor=0.7, t_engage=t_engage),
                         failure_schedule(), 300.0)
        assert s["t"].shape == s["G"].shape
        assert np.all(np.diff(s["t"]) > 0)

    def test_grid_length_exact(self, p):
        # 300 t.u. at dt=0.05 -> 6001 samples, no duplicated edges
        s = simulate_reg(p, RegulatorParams(floor=0.7, t_engage=100.0),
                         failure_schedule(), 300.0)
        assert len(s["t"]) == 6001

    def test_engage_at_breakpoint_is_continuous(self, p):
        # engaging exactly at a breakpoint vs one dt later: the only
        # difference is 0.05 t.u. of floor-off dynamics during the episode
        # (not a discontinuity artifact of the de-duplicated edge)
        s1 = simulate_reg(p, RegulatorParams(floor=0.7, t_engage=100.0),
                          failure_schedule(), 300.0)
        s2 = simulate_reg(p, RegulatorParams(floor=0.7, t_engage=100.05),
                          failure_schedule(), 300.0)
        assert np.max(np.abs(s1["G"] - s2["G"])) < 5e-3


# ------------------------------------------------- wrap fidelity (frozen)
class TestWrapFidelity:
    @pytest.mark.parametrize("name,sch,T",
                             [("baseline", baseline_schedule(), T_BASE),
                              ("failure", failure_schedule(), T_FAIL),
                              ("rescue", rescue_schedule(), T_RESC)])
    def test_off_wrap_is_frozen(self, p, name, sch, T):
        sf = simulate(p, sch, T)
        sr = simulate_reg(p, OFF, sch, T)
        assert np.max(np.abs(sf["G"] - sr["G"])) == 0.0


# --------------------------------- finding 1: always-armed threshold loss
class TestAlwaysArmedScope:
    """In the deployed configuration (t_engage=0) the GATED monitoring-cost
    threshold vanishes: the gate fires during the descent, before capture
    consolidates, so even an 'elaborate' gated introspector escapes."""

    @pytest.mark.parametrize("k,c_mon",
                             [(1.0, 0.0), (1.0, 0.5), (1.0, 1.5),
                              (2.0, 0.8), (2.0, 1.5)])
    def test_gated_cost_never_kills_always_armed(self, p, k, c_mon):
        s = simulate_reg(p, RegulatorParams(floor=None, k_pull=k,
                                            c_mon=c_mon, t_engage=0.0),
                         failure_schedule(), 1500.0)
        assert s["G"][-1] == pytest.approx(0.8855, abs=2e-3)

    def test_frozen_still_fails_always_armed_geometry(self, p):
        s = simulate_reg(p, RegulatorParams(floor=None, k_pull=0.0,
                                            c_mon=0.0, t_engage=0.0),
                         failure_schedule(), 1500.0)
        assert s["G"][-1] < 0.1

    def test_post_collapse_gated_threshold_exists(self, p):
        # the assay geometry (engage t=600, after settling) is where the
        # threshold c_mon_crit ~ 0.5 lives: c_mon=0.8 at k=2 fails there
        s = simulate_reg(p, RegulatorParams(floor=None, k_pull=2.0,
                                            c_mon=0.8, t_engage=600.0),
                         failure_schedule(), 1500.0)
        assert s["G"][-1] < 0.15


# ------------------------------ finding 4: floor + ungated post-collapse
class TestKnowingFloorCell:
    """Holding the floor while continuously re-checking it (ungated
    always-on cost) defeats the floor at c_mon ~ 0.2 — the knowing-floor
    result kc ~ 0.2: the floor must not be re-examined."""

    def test_ungated_cost_defeats_floor_post_collapse(self, p):
        g0 = simulate_reg(p, RegulatorParams(floor=0.7, c_mon=0.0,
                                             mon_gated=False,
                                             t_engage=600.0),
                          failure_schedule(), 1500.0)["G"][-1]
        g2 = simulate_reg(p, RegulatorParams(floor=0.7, c_mon=0.2,
                                             mon_gated=False,
                                             t_engage=600.0),
                          failure_schedule(), 1500.0)["G"][-1]
        assert g0 > 0.85                      # floor alone escapes
        assert 0.15 < g2 < 0.5                # re-examination defeats it

    def test_ungated_cost_monotone_post_collapse(self, p):
        gs = [simulate_reg(p, RegulatorParams(floor=0.7, c_mon=cm,
                                              mon_gated=False,
                                              t_engage=600.0),
                           failure_schedule(), 1500.0)["G"][-1]
              for cm in (0.1, 0.2, 0.3, 0.5)]
        assert gs == sorted(gs, reverse=True)  # monotone degradation


# ------------------------------------------------------------- floor math
class TestFloorBasics:
    def test_floor_clamps_from_below(self, p):
        r = RegulatorParams(floor=0.7)
        assert floor_theta_eff(0.1, p, r) == pytest.approx(0.7)
        # S=S_rest gives the unclamped Theta_eff=0.4 < 0.7: still clamped
        assert floor_theta_eff(p.S_rest, p, r) == pytest.approx(0.7)
        # only S high enough that Theta*S/S_rest > floor escapes the clamp
        assert floor_theta_eff(1.0, p, r) > 0.7

    def test_none_floor_is_frozen(self, p):
        r = RegulatorParams(floor=None)
        assert floor_theta_eff(0.1, p, r) == pytest.approx(
            p.Theta * 0.1 / p.S_rest)
