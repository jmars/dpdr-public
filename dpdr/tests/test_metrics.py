"""Classifier unit tests for dpdr.metrics."""
import numpy as np
import pytest

from dpdr.metrics import classify, detect_relapse, first_above_after, \
    first_below, is_stuck, rescue_success, stuck_duration, summarize
from dpdr.model import Params


def _sol(t, G, E=0.0, c=0.0, Theta_eff=0.4):
    t = np.asarray(t, float)
    G = np.asarray(G, float)
    E = np.full_like(t, E) if np.isscalar(E) else np.asarray(E, float)
    c = np.full_like(t, c) if np.isscalar(c) else np.asarray(c, float)
    Teff = np.full_like(t, Theta_eff)
    return dict(t=t, G=G, E=E, c=c, Theta_eff=Teff, D=G + E, S=Teff / 0.8,
                a=np.zeros_like(t), g=np.full_like(t, 0.5),
                u_loop=G / (G + E + 1e-9), collapsed_at=None, nfev=0)


@pytest.fixture
def p():
    return Params()


class TestPrimitives:
    def test_first_below(self):
        t = np.arange(0.0, 100.0, 0.05)
        G = np.where(t < 50, 0.5, 0.05)
        assert first_below(G, t) == pytest.approx(50.0, abs=0.1)

    def test_first_below_never(self):
        assert first_below(np.full(100, 0.5), np.arange(100.0)) is None

    def test_first_above_after(self):
        t = np.arange(0.0, 100.0, 0.05)
        G = np.where(t > 60, 0.7, 0.05)
        assert first_above_after(G, t, 0.5, 50.0) == pytest.approx(60.0,
                                                                   abs=0.1)


class TestStuck:
    def test_stuck_true(self, p):
        t = np.arange(0.0, 600.0, 0.05)
        sol = _sol(t, np.full_like(t, 0.05), E=0.5, c=0.9)
        assert is_stuck(sol, p)

    def test_stuck_false_when_g_high(self, p):
        t = np.arange(0.0, 600.0, 0.05)
        sol = _sol(t, np.full_like(t, 0.8), E=-0.2, c=0.0)
        assert not is_stuck(sol, p)

    def test_stuck_duration(self, p):
        t = np.arange(0.0, 600.0, 0.05)
        G = np.where(t < 300, 0.6, 0.05)
        assert stuck_duration(_sol(t, G), p) == pytest.approx(300.0, abs=0.1)


class TestClassify:
    def test_baseline_never_collapsed(self, p):
        t = np.arange(0.0, 600.0, 0.05)
        assert classify(_sol(t, np.full_like(t, 0.7), E=-0.1), p) == "baseline"

    def test_depersonalized(self, p):
        t = np.arange(0.0, 600.0, 0.05)
        sol = _sol(t, np.where(t < 100, 0.7, 0.05), E=0.5, c=0.9)
        assert classify(sol, p) == "depersonalized"

    def test_recovering_escaped_collapse(self, p):
        t = np.arange(0.0, 600.0, 0.05)
        G = np.where((t > 100) & (t < 300), 0.05, 0.7)
        assert classify(_sol(t, G, E=-0.1), p) == "recovering"

    def test_recovered_far_past_counts_recovering(self, p):
        # collapse early, recover and stay up for the rest -> recovering
        t = np.arange(0.0, 1000.0, 0.05)
        G = np.where((t > 100) & (t < 200), 0.05, 0.7)
        assert classify(_sol(t, G, E=-0.1), p) == "recovering"

    def test_relapsed(self, p):
        t = np.arange(0.0, 1000.0, 0.05)
        G = (np.where(t < 100, 0.7, 0.05)          # initial healthy
             )
        G = np.where(t < 100, 0.7, 0.05)
        G = np.where((t > 100) & (t < 300), 0.05, G)   # collapse
        G = np.where((t > 300) & (t < 700), 0.7, G)    # sustained recovery
        G = np.where(t > 700, 0.05, G)                 # relapse
        E = np.where(G > 0.4, -0.1, 0.5)
        c = np.where(G > 0.4, 0.0, 0.9)
        assert classify(_sol(t, G, E, c), p) == "relapsed"


class TestDetectRelapse:
    def test_no_relapse_when_stuck_once(self, p):
        t = np.arange(0.0, 600.0, 0.05)
        G = np.where(t < 100, 0.7, 0.05)
        assert not detect_relapse(_sol(t, G, E=0.5, c=0.9), p)[0]

    def test_relapse_after_sustained_recovery(self, p):
        t = np.arange(0.0, 1000.0, 0.05)
        G = np.where(t < 100, 0.7, 0.05)
        G = np.where((t > 100) & (t < 300), 0.05, G)
        G = np.where((t > 300) & (t < 700), 0.7, G)
        G = np.where(t > 700, 0.05, G)
        relapsed, t_rel = detect_relapse(_sol(t, G, E=0.5, c=0.9), p)
        assert relapsed and t_rel == pytest.approx(700.0, abs=0.1)

    def test_brief_lift_is_not_sustained(self, p):
        # G above 0.5 for less than 2*tau_G then back down: not a relapse
        t = np.arange(0.0, 600.0, 0.05)
        G = np.where(t < 100, 0.7, 0.05)
        G = np.where((t > 300) & (t < 330), 0.6, G)   # 30 t.u. < 2*tau_G
        assert not detect_relapse(_sol(t, G, E=0.5, c=0.9), p)[0]


class TestRescueSuccess:
    def test_success(self, p):
        t = np.arange(0.0, 1400.0, 0.05)
        G = np.where((t > 100) & (t < 900), 0.05, 0.7)
        E = np.where(G > 0.4, -0.2, 0.5)
        sol = _sol(t, G, E, c=np.where(G > 0.4, 0.0, 0.9))
        assert rescue_success(sol, p)

    def test_failure_when_never_recovers(self, p):
        t = np.arange(0.0, 1400.0, 0.05)
        G = np.where(t > 100, 0.05, 0.7)
        sol = _sol(t, G, E=0.5, c=0.9)
        assert not rescue_success(sol, p)


class TestSummarize:
    def test_summary_fields(self, p):
        t = np.arange(0.0, 600.0, 0.05)
        sol = _sol(t, np.where(t < 100, 0.7, 0.05), E=0.5, c=0.9)
        s = summarize(sol, p)
        assert s["G_end"] == pytest.approx(0.05)
        assert s["regime"] == 1                    # depersonalized
        assert s["t_collapse"] == pytest.approx(100.0, abs=0.1)
        assert s["T_stuck"] == pytest.approx(500.0, abs=0.1)
        assert np.isnan(s["t_recover"])
