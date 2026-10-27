"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12, Mutationstests der Linie Mehrkriterien-Ranking): n = 2, Gleichstand, inklusive Grenzen, entartete Spiele."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import ggw_chain as CH
import ggw_game as G
import ggw_limit as LI
import ggw_sim as SI
import ggw_three as T
from conftest import HIRSCH


def test_exactly_two_firms():
    """n = 2: der kleinste Fall (Division durch n - 1 = 1, keine Zustände dazwischen); Verteilung hat drei Einträge."""
    pi = CH.stationary(CH.rates_logit(2, HIRSCH, 1.0))
    assert len(pi) == 3 and pi.sum() == pytest.approx(1.0) and G.barrier_state(2, HIRSCH) == 1
    assert G.sign_counts(2, HIRSCH) == (1, 1, 0)


def test_a_single_firm_is_rejected():
    for f in (lambda: G.delta(1, HIRSCH), lambda: G.advantage_numerator(1, 0, HIRSCH), lambda: G.potential_state(0, 1, HIRSCH)):
        with pytest.raises(ValueError):
            f()


def test_the_smallest_group_of_the_app_n_four_has_every_part():
    pi = CH.stationary(CH.rates_logit(4, HIRSCH, 1.0))
    assert len(pi) == 5 and CH.masses(pi).high > 0 and CH.masses(pi).mid > 0


def test_all_payoffs_equal_gives_the_binomial_distribution():
    """a = b = c = d: Delta ≡ 0, jede Konfiguration gleich wahrscheinlich: pi(k) = C(n, k)/2^n, für jede Schärfe und jedes Modell (Gleichstand: beide je 1/2)."""
    game = (2, 2, 2, 2)
    binom = np.array([math.comb(10, k) / 2 ** 10 for k in range(11)])
    assert np.abs(CH.stationary(CH.rates_logit(10, game, 7.0)) - binom).max() < 1e-14
    assert np.abs(CH.stationary(CH.rates_uniform(10, game, 0.01)) - binom).max() < 1e-14
    assert np.abs(CH.stationary(CH.rates_typed(10, game, 0.1, 0.01)) - binom).max() < 1e-14
    assert G.scale(game) == 1 and G.beta_from_x(game, 3.0) == 3.0


def test_a_tie_state_is_exactly_one_half_for_the_error_models():
    """n = 4, p* = 1/3: Delta_1 = 0 (Gleichstand). Gleichmäßig mit eps = 0,5: P(S) = 1/2 (nicht 1 - eps/2 und nicht eps/2)."""
    game = G.game_from(1, F(1, 3))
    r = CH.rates_uniform(4, game, 0.5)
    assert math.exp(r.log_up[1]) == pytest.approx(3 / 4 * 0.5) and math.exp(r.log_down[1]) == pytest.approx(2 / 4 * 0.5)


def test_uniform_with_eps_one_is_the_pure_random_choice():
    """eps = 1 (obere Grenze erlaubt): P(S) = 1/2 überall, pi binomial."""
    pi = CH.stationary(CH.rates_uniform(6, HIRSCH, 1.0))
    assert pi.tolist() == pytest.approx([math.comb(6, k) / 64 for k in range(7)])


def test_typed_errors_must_be_smaller_than_one():
    with pytest.raises(ValueError):
        CH.rates_typed(5, HIRSCH, 1.0, 0.1)
    with pytest.raises(ValueError):
        CH.rates_typed(5, HIRSCH, 0.1, 1.0)


def test_mass_boundaries_at_group_sizes_where_ten_percent_is_not_an_integer():
    """n = 15: fast alle S ab k = 14 (13,5 aufgerundet), fast alle H bis k = 1 (1,5 abgerundet)."""
    for k in range(16):
        pi = np.zeros(16)
        pi[k] = 1.0
        m = CH.masses(pi)
        assert (m.high == 1.0) == (k >= 14) and (m.low == 1.0) == (k <= 1)


def test_threshold_with_a_lower_level_and_at_the_smallest_group():
    assert CH.threshold_x(2, F(1, 4), level=0.5) is not None
    assert CH.threshold_x(4, F(1, 4)) > CH.threshold_x(1000, F(1, 4))


def test_symmetric_game_p_star_one_half_has_equal_masses_for_every_group_size():
    for n in (4, 5, 11, 50):
        m = CH.masses(CH.stationary(CH.rates_logit(n, G.game_from(1, F(1, 2)), 9.0)))
        assert m.high == pytest.approx(m.low, abs=1e-12)


def test_sharpness_at_the_slider_limits_does_not_break_the_chain():
    for n, x in ((4, 0.25), (4, 40.0), (1000, 0.25), (1000, 40.0)):
        pi = CH.stationary(CH.rates_logit(n, HIRSCH, G.beta_from_x(HIRSCH, x)))
        assert np.isfinite(pi).all() and pi.sum() == pytest.approx(1.0)


def test_rate_function_at_extreme_sharpness_keeps_the_maximum_inside_the_grid():
    maxima = LI.local_maxima(40.0, 0.25)
    assert maxima and all(0 < p < 1 for p, _ in maxima)


def test_fixed_points_for_zero_sharpness_is_the_single_point_one_half():
    assert LI.fixed_points(0.0, 0.25, 20_001) == [pytest.approx(0.5)]


def test_barrier_state_boundaries():
    """Die Barriere liegt in 0 .. n - 1; bei n = 2 und p* = 1/2 (Delta_0 < 0 = Delta_1 ... Gleichstand) gibt es nur Delta_1 = lam > 0."""
    assert G.barrier_state(2, G.game_from(1, F(1, 2))) == 1
    assert G.barrier_state(2, (3, 1, 0, 0)) == 1


def test_escape_time_with_barrier_at_state_one_is_the_inverse_rate():
    """k* = 1: aus alle H bis zur Barriere ist es genau ein Schritt der Kette von 0 nach 1."""
    game = (5, 2, 1, 2)
    rates = CH.rates_logit(2, game, 1.0)
    assert 10 ** (SI.escape_log_time(rates, game, "H") + math.log10(2)) == pytest.approx(1 / math.exp(rates.log_up[0]))


def test_three_formats_with_two_firms_and_one_firm_per_state():
    states, p = T.chain_m(2, np.eye(3) * 2.0, T.LOGIT, 1.0)
    assert len(states) == 6 and np.abs(p.sum(axis=1) - 1).max() < 1e-14
