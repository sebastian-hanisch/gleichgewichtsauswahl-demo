"""ggw_game: Spiel, Schwelle, Vorteil Delta_j, exakte Vorzeichenzählung und Potential gegen Handrechnung (Mini-Instanzen in conftest.py)."""

from fractions import Fraction as F

import numpy as np
import pytest

import ggw_game as G

HIRSCH_GAME = (5, 2, 1, 2)


def test_classification_of_the_seven_games(games):
    assert G.classify(games["hirsch"]) == G.KOORDINATION
    assert G.classify(games["buendeln"]) == G.KOORDINATION
    assert G.classify(games["gleichauf"]) == G.KOORDINATION
    assert G.classify(games["anti"]) == G.ANTI
    assert G.classify(games["dominant"]) == G.DOMINANT
    assert G.classify(games["entartet"]) == G.ENTARTET


def test_gaps_lambda_and_p_star_by_hand(games):
    """(5, 2, 1, 2): a - d = 3, b - c = 1, lam = 4, p* = 1/4. (6, 4, 0, 4): 2 und 4, lam = 6, p* = 4/6 = 2/3. (6, 4, 2, 4): 2 und 2, lam = 4, p* = 1/2."""
    assert G.gaps(games["hirsch"]) == (3, 1) and G.lam(games["hirsch"]) == 4 and G.p_star(games["hirsch"]) == F(1, 4)
    assert G.gaps(games["buendeln"]) == (2, 4) and G.lam(games["buendeln"]) == 6 and G.p_star(games["buendeln"]) == F(2, 3)
    assert G.gaps(games["gleichauf"]) == (2, 2) and G.p_star(games["gleichauf"]) == F(1, 2)


def test_p_star_exists_only_for_coordination_games(games):
    for name in ("anti", "dominant", "entartet"):
        assert G.p_star(games[name]) is None


def test_different_payoff_gradient_same_lambda_and_p_star(games):
    """(3, 5, 4, 0) hat dieselben Lücken (3, 1) wie (5, 2, 1, 2), aber a - b = -2 gegen +3."""
    assert G.gaps(games["anderes"]) == G.gaps(games["hirsch"]) and G.p_star(games["anderes"]) == F(1, 4)
    assert games["anderes"][0] - games["anderes"][1] == -2 and games["hirsch"][0] - games["hirsch"][1] == 3


def test_delta_of_the_two_player_game(mini2):
    """n = 2: Delta_0 = c - b = -1, Delta_1 = a - d = 3."""
    assert G.delta(mini2["n"], mini2["game"]).tolist() == list(mini2["delta"])
    assert [G.advantage_numerator(2, j, mini2["game"]) for j in (0, 1)] == [-1, 3]     # Nenner n - 1 = 1


def test_delta_of_the_three_player_game(mini3):
    assert G.delta(mini3["n"], mini3["game"]).tolist() == list(mini3["delta"])
    assert [G.advantage_numerator(3, j, mini3["game"]) for j in (0, 1, 2)] == [-2, 2, 6]    # Nenner 2


def test_delta_equals_lambda_times_share_minus_p_star():
    """Delta_j = lam·(j/(n - 1) - p*): für (5, 2, 1, 2), n = 5 genau 4·(j/4 - 1/4) = j - 1."""
    assert G.delta(5, (5, 2, 1, 2)).tolist() == [-1.0, 0.0, 1.0, 2.0, 3.0]


def test_numerator_route_and_payoff_route_agree(games):
    for game in games.values():
        for n in (2, 3, 7, 20):
            num = np.array([float(G.advantage_numerator(n, j, game)) for j in range(n)]) / (n - 1)
            assert np.allclose(G.delta(n, game), num, rtol=0, atol=1e-12)


def test_sign_counts_by_hand_n5():
    """Delta = (-1, 0, 1, 2, 3): 3 positive, 1 negative, 1 Gleichstand."""
    assert G.sign_counts(5, (5, 2, 1, 2)) == (3, 1, 1)


def test_sign_counts_tie_is_exact_not_floating_point():
    """p* = 1/3 bei n = 4: der Gleichstand liegt genau bei j = 1 (Delta_1 = lam·(1/3 - 1/3)); mit Brüchen gezählt, nicht mit Toleranz."""
    game = G.game_from(1, F(1, 3))
    assert G.sign_counts(4, game) == (2, 1, 1) and G.barrier_state(4, game) == 2


def test_sign_counts_sum_to_n(games):
    for game in games.values():
        assert sum(G.sign_counts(9, game)) == 9


def test_barrier_state_is_first_positive_delta():
    """(5, 2, 1, 2), n = 5: Delta = (-1, 0, 1, 2, 3), der erste positive steht bei j = 2."""
    assert G.barrier_state(5, (5, 2, 1, 2)) == 2
    assert G.barrier_state(5, (1, 4, 0, 0)) == 4          # p* = 4/5: Zähler 5·j - 16 = (-16, -11, -6, -1, 4), erst j = 4 ist positiv
    assert G.barrier_state(5, (-1, 2, 4, 0)) == 0          # Anti-Koordination: Delta_0 = c - b = 4 - 2 > 0, S ist schon ohne andere S besser
    assert G.barrier_state(5, (2, 5, 1, 3)) is None          # H dominiert (a - d = -1, b - c = 4): Delta nie positiv


def test_potential_pair_and_state_by_hand(mini3):
    """Phi(k) = [3·k(k-1)/2 + 1·(3-k)(2-k)/2]/2 für n = 3: Phi(0) = 3/2, Phi(1) = 1/2, Phi(2) = 3/2, Phi(3) = 9/2."""
    assert G.potential_pair(HIRSCH_GAME) == {"SS": 3, "HH": 1, "SH": 0}
    assert [G.potential_state(k, 3, HIRSCH_GAME) for k in range(4)] == [1.5, 0.5, 1.5, 4.5]
    rel = [G.potential_state(k, 3, HIRSCH_GAME) - 1.5 for k in range(4)]
    assert rel == list(mini3["phi_rel"])


def test_potential_difference_is_delta(games):
    """Der Gewinn eines Wechsels ist die Potentialdifferenz: Phi(k + 1) - Phi(k) = Delta_k."""
    for game in games.values():
        for n in (3, 8, 15):
            phi = G.potential_curve(n, game)
            assert np.allclose(np.diff(phi), G.delta(n, game), rtol=0, atol=1e-12)


def test_x_to_beta_uses_lambda_in_coordination_games():
    assert G.beta_from_x((5, 2, 1, 2), 8.0) == pytest.approx(2.0)          # lam = 4
    assert G.beta_from_x((6, 4, 0, 4), 6.0) == pytest.approx(1.0)          # lam = 6


def test_scale_of_a_non_coordination_game_is_the_sum_of_absolute_gaps():
    assert G.scale((-1, 2, 4, 0)) == 3 and G.scale((3, 1, 2, 0)) == 4 and G.scale((2, 2, 2, 2)) == 1


def test_game_from_has_the_requested_lambda_and_p_star():
    game = G.game_from(F(5), F(2, 5))
    assert G.lam(game) == 5 and G.p_star(game) == F(2, 5)

