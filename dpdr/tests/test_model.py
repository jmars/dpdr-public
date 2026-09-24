"""Gate regression tests G1-G2c (plan §4c) on the packaged model."""
import numpy as np
import pytest

from dpdr.events import baseline_schedule, failure_schedule, rescue_schedule
from dpdr.integrate import simulate
from dpdr.metrics import (EPISODE, RESCUE, check_gates, classify,
                          detect_relapse, is_stuck)
from dpdr.model import Params

T_BASE, T_FAIL, T_RESC = 1000.0, 600.0, 1400.0


@pytest.fixture(scope="module")
def p():
    return Params()


@pytest.fixture(scope="module")
def sol_base(p):
    return simulate(p, baseline_schedule(), T_BASE)


@pytest.fixture(scope="module")
def sol_fail(p):
    return simulate(p, failure_schedule(), T_FAIL)


@pytest.fixture(scope="module")
def sol_resc(p):
    return simulate(p, rescue_schedule(), T_resc := T_RESC)


# ------------------------------------------------------------------ G1
class TestG1Baseline:
    def test_g_in_open_unit_interval(self, sol_base):
        assert 0.0 < sol_base["G"][-1] < 1.0

    def test_error_below_threshold(self, sol_base, p):
        assert sol_base["E"][-1] < p.Theta

    def test_cannibalization_off(self, sol_base):
        assert sol_base["c"][-1] < 0.05

    def test_regime_baseline(self, sol_base, p):
        assert classify(sol_base, p) == "baseline"


# ------------------------------------------------------------------ G2/G2b
class TestG2Failure:
    def test_g_collapsed(self, sol_fail):
        assert sol_fail["G"][-1] < 0.1

    def test_error_above_threshold(self, sol_fail, p):
        assert sol_fail["E"][-1] > p.Theta

    def test_cannibalization_on(self, sol_fail):
        assert sol_fail["c"][-1] > 0.5

    def test_stuck_criterion(self, sol_fail, p):
        assert is_stuck(sol_fail, p)

    def test_g2b_no_self_recovery(self, sol_fail, p):
        w = sol_fail["t"] >= EPISODE[1] + 10 * p.tau_G
        assert sol_fail["G"][w].max() < 0.1

    def test_regime_depersonalized(self, sol_fail, p):
        assert classify(sol_fail, p) == "depersonalized"


# ------------------------------------------------------------------ G2c
class TestG2cRescue:
    def test_g_recovered_at_end(self, sol_resc):
        assert sol_resc["G"][-1] > 0.5

    def test_g_stays_recovered(self, sol_resc):
        w = sol_resc["t"] >= RESCUE[1]
        assert sol_resc["G"][w].min() > 0.5

    def test_stuck_before_rescue(self, sol_resc, p):
        i = np.searchsorted(sol_resc["t"], EPISODE[1] + 400)
        assert sol_resc["G"][i] < 0.1

    def test_regime_recovering_or_baseline(self, sol_resc, p):
        assert classify(sol_resc, p) in ("recovering", "baseline")


# ------------------------------------------------------------------ gate set
def test_all_gates():
    gates = check_gates()
    failed = [k for k, v in gates.items() if not v]
    assert not failed, f"gates failed: {failed}"


def test_no_relapse_in_standard_rescue(sol_resc, p):
    relapsed, t_rel = detect_relapse(sol_resc, p)
    assert not relapsed


def test_schedules_have_expected_breakpoints():
    assert failure_schedule().breakpoints() == [0.0, 100.0, 160.0, 200.0]
    r = rescue_schedule()
    assert r.breakpoints() == [0.0, 100.0, 160.0, 200.0, 800.0, 1000.0]
