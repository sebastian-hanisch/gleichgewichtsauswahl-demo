"""ggw_chain: Raten, stationäre Verteilung, Massen und Schwelle gegen Handrechnung (Mini-Instanzen in conftest.py)."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import ggw_chain as CH
import ggw_game as G
from conftest import ANDERES_GEFAELLE, GLEICHAUF, HIRSCH


def _normalise(weights):
    total = sum(weights)
    return [w / total for w in weights]


def test_two_player_chain_by_hand(mini2):
    """pi ~ (1, 2·e^(-beta), e^(2·beta)): aus up(0)/down(0) = 2·e^(-beta) und up(1)/down(1) = e^(3·beta)/2."""
    beta = 0.7
    pi = CH.stationary(CH.rates_logit(2, mini2["game"], beta))
    expected = _normalise([1.0, 2 * math.exp(-beta), math.exp(2 * beta)])
    assert pi.tolist() == pytest.approx(expected, rel=1e-12)


def test_two_player_rates_by_hand(mini2):
    """Rate aufwärts von 0: (2 - 0)/2·sigma(-beta); abwärts von 1: (0 + 1)/2·sigma(beta)."""
    beta = 0.7
    r = CH.rates_logit(2, mini2["game"], beta)
    sigma = lambda z: 1 / (1 + math.exp(-z))
    assert math.exp(r.log_up[0]) == pytest.approx(sigma(-beta), rel=1e-12)
    assert math.exp(r.log_down[0]) == pytest.approx(0.5 * sigma(beta), rel=1e-12)
    assert math.exp(r.log_up[1]) == pytest.approx(0.5 * sigma(3 * beta), rel=1e-12)
    assert math.exp(r.log_down[1]) == pytest.approx(sigma(-3 * beta), rel=1e-12)


def test_three_player_distribution_by_hand(mini3):
    """pi ~ (1, 3·e^(-beta), 3, e^(3·beta)): Binomialfaktor (1, 3, 3, 1) mal exp(beta·(Phi(k) - Phi(0)))."""
    beta = 0.5
    expected = _normalise([1.0, 3 * math.exp(-beta), 3.0, math.exp(3 * beta)])
    assert CH.stationary(CH.rates_logit(3, mini3["game"], beta)).tolist() == pytest.approx(expected, rel=1e-12)
    assert CH.stationary_logit_closed(3, mini3["game"], beta).tolist() == pytest.approx(expected, rel=1e-12)


def test_closed_form_equals_the_chain_for_every_game(games):
    for game in games.values():
        for n in (2, 5, 12):
            for beta in (0.3, 1.0, 2.5):
                chain = CH.stationary(CH.rates_logit(n, game, beta))
                assert np.abs(chain - CH.stationary_logit_closed(n, game, beta)).max() < 1e-12


def test_binomial_factor_is_the_reason_for_the_middle():
    """(5, 2, 1, 2), n = 10, beta = 0,5 (x = 2): je Konfiguration ist „alle S“ am wahrscheinlichsten, aber der Zustand k = 5 hat 252 Konfigurationen und mehr Masse."""
    pi = CH.stationary_logit_closed(10, HIRSCH, 0.5)
    assert pi[5] > pi[10]
    assert pi[5] / math.comb(10, 5) < pi[10]      # je Konfiguration ist „alle S“ wahrscheinlicher


def test_invariance_same_lambda_and_p_star_same_distribution():
    """(5, 2, 1, 2), (4, 1, 0, 1), (3, 5, 4, 0): alle a - d = 3, b - c = 1; die Auszahlungsdominanz a - b (3, 3, -2) geht nicht ein."""
    ref = CH.stationary(CH.rates_logit(6, HIRSCH, 1.0))
    for game in ((4, 1, 0, 1), ANDERES_GEFAELLE):
        assert np.abs(CH.stationary(CH.rates_logit(6, game, 1.0)) - ref).max() < 1e-14


def test_p_star_one_half_is_symmetric():
    pi = CH.stationary(CH.rates_logit(4, GLEICHAUF, 1.3))
    assert pi.tolist() == pytest.approx(pi[::-1].tolist(), abs=1e-14)


def test_mirror_game_mirrors_the_distribution():
    """Spiegelung p* -> 1 - p*: S und H vertauschen die Rollen, also pi_spiegel(k) = pi(n - k)."""
    g1, g2 = G.game_from(1, F(1, 4)), G.game_from(1, F(3, 4))
    p1 = CH.stationary(CH.rates_logit(9, g1, 3.0))
    p2 = CH.stationary(CH.rates_logit(9, g2, 3.0))
    assert p1.tolist() == pytest.approx(p2[::-1].tolist(), abs=1e-13)


def test_stationary_is_a_distribution_and_survives_extreme_sharpness():
    """n = 1000 bei x = 40: Wahrscheinlichkeiten unter 1e-300 laufen im Logarithmus nicht über oder unter."""
    pi = CH.stationary(CH.rates_logit(1000, G.game_from(1, F(1, 4)), 40.0))
    assert np.isfinite(pi).all() and pi.sum() == pytest.approx(1.0, abs=1e-12) and pi.min() >= 0 and pi.argmax() == 1000


def test_masses_by_hand():
    """n = 10: „fast alle S“ ist k >= 9, „fast alle H“ k <= 1, Mitte k = 2 .. 8."""
    pi = np.array([0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    m = CH.masses(pi)
    assert (m.high, m.low, m.mid) == pytest.approx((0.2, 0.1, 0.7), abs=1e-15)


def test_masses_boundaries_are_inclusive_and_integer():
    """n = 20: k = 18 (0,9·20) gehört zu „fast alle S“, k = 2 (0,1·20) zu „fast alle H“; n = 30: 27 und 3."""
    for n, hi, lo in ((20, 18, 2), (30, 27, 3), (50, 45, 5)):
        for k in range(n + 1):
            pi = np.zeros(n + 1)
            pi[k] = 1.0
            m = CH.masses(pi)
            assert (m.high == 1.0) == (k >= hi) and (m.low == 1.0) == (k <= lo) and (m.mid == 1.0) == (lo < k < hi), (n, k)


def test_masses_with_two_players():
    """n = 2: 10·k >= 18 heißt k = 2, 10·k <= 2 heißt k = 0; k = 1 ist die Mitte."""
    assert CH.masses(np.array([0.0, 1.0, 0.0])).mid == 1.0
    assert CH.masses(np.array([0.0, 0.0, 1.0])).high == 1.0 and CH.masses(np.array([1.0, 0.0, 0.0])).low == 1.0


def test_middle_dominates_below_the_threshold():
    """Zweig-Test (Nullspalte = Bug-Signal): bei x = 2, n = 100 liegt fast die ganze Masse in der Mitte."""
    pi = CH.stationary(CH.rates_logit(100, G.game_from(1, F(1, 4)), 2.0))
    assert CH.masses(pi).mid > 0.99


def test_uniform_slope_is_the_count_of_positive_minus_negative_states():
    """log pi(n)/pi(0) = (Zahl Delta > 0 - Zahl Delta < 0)·log((1 - eps/2)/(eps/2)): n = 5, (5, 2, 1, 2): 3 - 1 = 2 (Delta = -1, 0, 1, 2, 3)."""
    for eps in (0.3, 1e-3, 1e-7):
        pi = CH.stationary(CH.rates_uniform(5, HIRSCH, eps))
        assert math.log(pi[5] / pi[0]) == pytest.approx(2 * math.log((1 - eps / 2) / (eps / 2)), rel=1e-9)


def test_uniform_tie_state_gives_half_half():
    """Bei Delta = 0 (n = 5, k = 1) wählen beide Formate mit Wahrscheinlichkeit 1/2: aufwärts (5 - 1)/5·1/2, abwärts 2/5·1/2."""
    r = CH.rates_uniform(5, HIRSCH, 0.2)
    assert math.exp(r.log_up[1]) == pytest.approx(4 / 5 * 0.5) and math.exp(r.log_down[1]) == pytest.approx(2 / 5 * 0.5)


def test_uniform_rates_at_a_positive_and_a_negative_state():
    """n = 5, eps = 0,2: Delta_3 > 0: P(S) = 1 - 0,1 = 0,9, aufwärts 2/5·0,9, abwärts 4/5·0,1. Delta_0 < 0: P(S) = 0,1, aufwärts 5/5·0,1, abwärts 1/5·0,9."""
    r = CH.rates_uniform(5, HIRSCH, 0.2)
    assert math.exp(r.log_up[3]) == pytest.approx(0.4 * 0.9) and math.exp(r.log_down[3]) == pytest.approx(0.8 * 0.1)
    assert math.exp(r.log_up[0]) == pytest.approx(1.0 * 0.1) and math.exp(r.log_down[0]) == pytest.approx(0.2 * 0.9)


def test_typed_slope_by_hand():
    """n = 5: 3 positive, 1 negative, 1 Gleichstand: log pi(n)/pi(0) = 3·log((1 - eps_S)/eps_S) - 1·log((1 - eps_H)/eps_H)."""
    eps_s, eps_h = 0.1, 0.02
    pi = CH.stationary(CH.rates_typed(5, HIRSCH, eps_s, eps_h))
    expected = 3 * math.log(0.9 / 0.1) - math.log(0.98 / 0.02)
    assert math.log(pi[5] / pi[0]) == pytest.approx(expected, rel=1e-9)


def test_typed_equals_uniform_when_the_errors_are_the_uniform_ones():
    """Gleichmäßig mit eps ist typabhängig mit eps_S = eps_H = eps/2."""
    a = CH.stationary(CH.rates_uniform(12, HIRSCH, 0.2))
    b = CH.stationary(CH.rates_typed(12, HIRSCH, 0.1, 0.1))
    assert np.abs(a - b).max() < 1e-14


def test_error_models_reject_impossible_error_rates():
    with pytest.raises(ValueError):
        CH.rates_uniform(5, HIRSCH, 0.0)
    with pytest.raises(ValueError):
        CH.rates_uniform(5, HIRSCH, 1.5)
    with pytest.raises(ValueError):
        CH.rates_typed(5, HIRSCH, 0.1, 0.0)
    with pytest.raises(ValueError):
        CH.rates_typed(5, HIRSCH, 0.0, 0.1)


def test_error_models_differ_on_at_least_one_cell():
    """Zweig-Tests: gleichmäßig unterscheidet p* kaum (Masse bei S für p* = 1/4 und 2/5 bei n = 20, eps = 0,2 auf 0,001 gleich, aber nicht identisch), der Logit schon; typabhängig kippt, gleichmäßig nicht."""
    n = 20
    g1, g2 = G.game_from(1, F(1, 4)), G.game_from(1, F(2, 5))
    u1 = CH.masses(CH.stationary(CH.rates_uniform(n, g1, 0.2))).high
    u2 = CH.masses(CH.stationary(CH.rates_uniform(n, g2, 0.2))).high
    assert u1 == pytest.approx(u2, abs=1e-3) and u1 != u2
    l1 = CH.masses(CH.stationary(CH.rates_logit(n, g1, 4.0))).high
    l2 = CH.masses(CH.stationary(CH.rates_logit(n, g2, 4.0))).high
    assert l1 - l2 > 0.2
    g = G.game_from(1, F(1, 4))
    typed = CH.stationary(CH.rates_typed(40, g, 0.1, 1e-4))     # ε_H klein: H hält besser als S
    assert CH.masses(typed).low > 0.99 > CH.masses(CH.stationary(CH.rates_typed(40, g, 0.1, 0.1))).low


def test_transition_matrix_rows_sum_to_one_and_have_the_stationary_vector():
    r = CH.rates_logit(6, HIRSCH, 0.8)
    p = CH.transition_matrix(r)
    assert np.abs(p.sum(axis=1) - 1).max() < 1e-14 and p.min() >= 0
    pi = CH.stationary(r)
    assert np.abs(pi @ p - pi).max() < 1e-14
    assert p[0, 0] == pytest.approx(1 - math.exp(r.log_up[0])) and p[1, 0] == pytest.approx(math.exp(r.log_down[0]))


def test_threshold_is_the_smallest_sharpness_for_99_percent():
    """Bei x* liegt die Masse bei genau 0,99, knapp darunter darunter, knapp darüber darüber."""
    x_star = CH.threshold_x(20, F(1, 4))
    assert CH.mass_high_for(20, F(1, 4), x_star) == pytest.approx(0.99, abs=1e-9)
    assert CH.mass_high_for(20, F(1, 4), x_star - 0.01) < 0.99 < CH.mass_high_for(20, F(1, 4), x_star + 0.01)


def test_threshold_is_never_when_p_star_is_at_least_one_half():
    """Bei p* >= 1/2 hat S nie die Mehrheit der Masse: x* = „nie“ (None). Bei p* = 0,45 und n = 3 ebenfalls, weil die Zustände zu grob sind."""
    assert CH.threshold_x(20, F(1, 2)) is None and CH.threshold_x(20, F(2, 3)) is None and CH.threshold_x(20, F(3, 4)) is None


def test_threshold_rises_with_p_star_and_falls_with_n():
    a = [CH.threshold_x(50, F(p)) for p in (F(1, 10), F(1, 4), F(2, 5))]
    assert a == sorted(a) and len(set(a)) == 3
    b = [CH.threshold_x(n, F(1, 4)) for n in (10, 50, 200)]
    assert b == sorted(b, reverse=True) and len(set(b)) == 3


def test_threshold_respects_the_level_and_the_cap():
    assert CH.threshold_x(20, F(1, 4), level=0.5) < CH.threshold_x(20, F(1, 4), level=0.99)
    assert CH.threshold_x(20, F(1, 4), x_max=1.0) is None


def test_mean_share_and_entropy_curve():
    assert CH.mean_share(np.array([0.5, 0.0, 0.5])) == pytest.approx(0.5)
    assert CH.mean_share(np.array([0.0, 0.0, 1.0])) == pytest.approx(1.0)
    assert np.exp(CH.entropy_curve(4)).round().tolist() == [1, 4, 6, 4, 1]
