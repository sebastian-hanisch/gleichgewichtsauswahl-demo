"""ggw_sim: Austrittszeit von Hand, Spiegelung, Einzelspieler-Simulation (Form, Bestimmtheit, Fehlerfälle)."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import ggw_chain as CH
import ggw_game as G
import ggw_sim as SI
from conftest import HIRSCH


def test_total_variation_by_hand():
    assert SI.total_variation([0.5, 0.5, 0.0], [0.0, 0.5, 0.5]) == 0.5
    assert SI.total_variation([1.0, 0.0], [1.0, 0.0]) == 0.0


def test_hitting_time_without_selection_pressure_by_hand():
    """beta = 0, n = 3: Raten up_0 = 1/2, down_0 = 1/6, up_1 = 1/3; E[T_0 -> 2] = 1/up_0 + (1 + 3)/(3·up_1) = 2 + 4 = 6 Einzelaktualisierungen
    (gleiche Zahl aus den Gleichungen T_0 = 2 + T_1, T_1 = 1 + T_1/2 + T_0/6)."""
    rates = CH.rates_logit(3, HIRSCH, 0.0)
    assert SI.log10_hitting_time(rates, 2) == pytest.approx(math.log10(6))


def test_hitting_time_to_the_first_state_is_the_inverse_up_rate():
    """Ziel 1 von 0 aus: E = 1/up_0 = 1/sigma(-beta·Delta_0·...) ; n = 2, beta = 1: up_0 = sigma(-1), E = 1 + e."""
    rates = CH.rates_logit(2, HIRSCH, 1.0)
    assert SI.log10_hitting_time(rates, 1) == pytest.approx(math.log10(1 + math.e))


def test_escape_time_from_h_and_from_s_for_two_firms_by_hand():
    """n = 2, (5, 2, 1, 2), beta = 1, Barriere k* = 1 (Delta = (-1, 3)): aus alle H: E = 1 + e Einzelaktualisierungen, je Betrieb durch 2;
    aus alle S: Rate 2 -> 1 ist (2/2)·sigma(-3), E = 1 + e^3, je Betrieb durch 2."""
    rates = CH.rates_logit(2, HIRSCH, 1.0)
    assert SI.escape_log_time(rates, HIRSCH, "H") == pytest.approx(math.log10((1 + math.e) / 2))
    assert SI.escape_log_time(rates, HIRSCH, "S") == pytest.approx(math.log10((1 + math.e ** 3) / 2))


def test_escape_time_is_none_without_a_barrier():
    anti = (-1, 2, 4, 0)       # Barriere bei j = 0: kein Austritt aus „alle H“ zu messen
    assert SI.escape_log_time(CH.rates_logit(5, anti, 1.0), anti, "H") is None
    h_dominant = (2, 5, 1, 3)  # S nie beste Antwort
    assert SI.escape_log_time(CH.rates_logit(5, h_dominant, 1.0), h_dominant, "S") is None


def test_escape_time_rejects_an_unknown_side_and_a_target_out_of_range():
    rates = CH.rates_logit(5, HIRSCH, 1.0)
    with pytest.raises(ValueError):
        SI.escape_log_time(rates, HIRSCH, "X")
    with pytest.raises(ValueError):
        SI.log10_hitting_time(rates, 0)
    with pytest.raises(ValueError):
        SI.log10_hitting_time(rates, 6)


def test_reverse_swaps_and_mirrors_the_rates():
    rates = CH.rates_logit(4, HIRSCH, 1.3)
    rev = SI.reverse(rates)
    assert rev.log_up.tolist() == rates.log_down[::-1].tolist() and rev.log_down.tolist() == rates.log_up[::-1].tolist()
    assert SI.reverse(rev).log_up.tolist() == rates.log_up.tolist()
    assert np.abs(CH.stationary(rev) - CH.stationary(rates)[::-1]).max() < 1e-13


def test_times_grow_with_sharpness_and_with_group_size_on_the_s_side():
    game = G.game_from(1, F(1, 4))
    t = [SI.escape_log_time(CH.rates_logit(10, game, x), game, "S") for x in (1.0, 2.0, 4.0)]
    assert t == sorted(t) and len(set(t)) == 3
    assert SI.escape_log_time(CH.rates_logit(20, game, 2.0), game, "S") > SI.escape_log_time(CH.rates_logit(10, game, 2.0), game, "S")


def test_simulate_players_is_a_distribution_and_deterministic():
    h1 = SI.simulate_players(6, HIRSCH, 1.0, 5000, 1000, 3, 7)
    h2 = SI.simulate_players(6, HIRSCH, 1.0, 5000, 1000, 3, 7)
    assert h1.sum() == pytest.approx(1.0) and len(h1) == 7 and h1.tolist() == h2.tolist()
    assert h1.tolist() != SI.simulate_players(6, HIRSCH, 1.0, 5000, 1000, 3, 8).tolist()


def test_simulate_players_with_extreme_sharpness_does_not_overflow():
    """beta·Delta unter -700 (exp würde überlaufen): die Wahl ist dann sicher H, die Simulation bleibt am Rand."""
    hist = SI.simulate_players(10, HIRSCH, 5000.0, 2000, 100, 0, 3)
    assert hist[0] == pytest.approx(1.0)


def test_simulate_players_counts_only_after_the_burn_in():
    """H dominiert (a - d = -1, b - c = 4), beta riesig: die Wahl ist sofort sicher H. Start bei alle S: ohne Einschwingschritte zählt der Anfang mit, nach 1 000 Schritten nur das Ende."""
    h_dominant = (2, 5, 1, 3)
    assert SI.simulate_players(2, h_dominant, 1e6, 1500, 1, 2, 5)[0] < 1.0
    assert SI.simulate_players(2, h_dominant, 1e6, 1500, 1000, 2, 5)[0] == pytest.approx(1.0)


def test_simulate_players_rejects_bad_arguments():
    with pytest.raises(ValueError):
        SI.simulate_players(5, HIRSCH, 1.0, 100, 10, 6, 1)
    with pytest.raises(ValueError):
        SI.simulate_players(5, HIRSCH, 1.0, 100, 100, 2, 1)


def test_simulate_hitting_cap_counts_aborted_runs_and_does_not_average_them():
    game = G.game_from(1, F(1, 4))
    times, capped = SI.simulate_hitting(10, game, 4.0, "S", 5, 1, 50)
    assert capped == 5 and times == []


def test_simulate_hitting_from_h_is_fast_and_reaches_the_barrier():
    game = G.game_from(1, F(1, 4))
    times, capped = SI.simulate_hitting(10, game, 2.0, "H", 20, 3, 100_000)
    assert capped == 0 and len(times) == 20 and min(times) > 0 and all(t * 10 == int(t * 10) for t in times)     # ganze Zahl von Einzelaktualisierungen


def test_simulate_hitting_is_deterministic():
    game = G.game_from(1, F(1, 4))
    assert SI.simulate_hitting(10, game, 2.0, "H", 5, 3, 100_000) == SI.simulate_hitting(10, game, 2.0, "H", 5, 3, 100_000)


def test_simulate_hitting_rejects_a_missing_barrier_and_an_unknown_side():
    game = G.game_from(1, F(1, 4))
    with pytest.raises(ValueError):
        SI.simulate_hitting(5, (-1, 2, 4, 0), 1.0, "H", 2, 1, 1000)
    with pytest.raises(ValueError):
        SI.simulate_hitting(10, game, 1.0, "X", 2, 1, 1000)
