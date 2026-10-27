"""ggw_limit: Ratenfunktion, Maxima, Logit-Gleichgewichte und kritischer Wert gegen Handrechnung."""

import math

import numpy as np
import pytest

import ggw_limit as LI


def test_entropy_by_hand():
    """H(1/2) = ln 2; H(y) ist symmetrisch."""
    y = np.array([0.5, 0.2, 0.8])
    h = LI.entropy(y)
    assert h[0] == pytest.approx(math.log(2)) and h[1] == pytest.approx(h[2])
    assert h[1] == pytest.approx(-0.2 * math.log(0.2) - 0.8 * math.log(0.8))


def test_rate_function_by_hand():
    """x = 2, p* = 1/4, y = 1/2: 2·(1/8 - 1/8) + ln 2 = ln 2. x = 4, p* = 1/2, y = 1: nicht auf dem Gitter, aber y = 0,5: 4·(1/8 - 1/4) + ln 2 = ln 2 - 1/2."""
    assert LI.rate_function(np.array([0.5]), 2.0, 0.25)[0] == pytest.approx(math.log(2))
    assert LI.rate_function(np.array([0.5]), 4.0, 0.5)[0] == pytest.approx(math.log(2) - 0.5)


def test_cell_midpoints_never_hit_the_boundary_or_one_half():
    y = LI.cell_midpoints(10)
    assert y[0] == 0.05 and y[-1] == 0.95 and 0.5 not in y.tolist()
    assert 0.5 in LI.cell_midpoints(11).tolist()


def test_critical_value_is_four_at_p_star_one_half():
    assert LI.CRITICAL_X == 4.0
    assert LI.critical_value() == pytest.approx(4.0, abs=1e-3)


def test_fixed_point_counts_one_below_and_three_above_the_critical_value():
    for points in (200_000, 200_001):
        assert [LI.count_fixed_points(x, 0.5, points) for x in (1.0, 3.9, 4.1, 8.0)] == [1, 1, 3, 3]


def test_fixed_points_solve_their_equation_and_are_symmetric_at_p_star_one_half():
    roots = LI.fixed_points(6.0, 0.5)
    assert len(roots) == 3
    for y in roots:
        assert y == pytest.approx(1 / (1 + math.exp(-6.0 * (y - 0.5))), abs=1e-9)
    assert roots[0] + roots[2] == pytest.approx(1.0, abs=1e-9) and roots[1] == pytest.approx(0.5, abs=1e-9)


def test_one_fixed_point_for_small_sharpness_at_p_star_one_quarter():
    assert LI.count_fixed_points(2.0, 0.25) == 1


def test_local_maxima_are_the_stable_fixed_points():
    """p* = 1/2, x = 6: zwei Maxima (die äußeren Lösungen), das mittlere (0,5) ist ein Minimum der Ratenfunktion und kein Maximum."""
    maxima = [p for p, _ in LI.local_maxima(6.0, 0.5)]
    roots = LI.fixed_points(6.0, 0.5)
    assert len(maxima) == 2 and maxima[0] == pytest.approx(roots[0], abs=1e-4) and maxima[1] == pytest.approx(roots[2], abs=1e-4)


def test_local_maximum_at_a_plateau_is_counted_once():
    """p* = 1/2 und kleines x: das einzige Maximum liegt bei 1/2 zwischen zwei gleich hohen Zellen (gerades Gitter) und darf nicht verloren gehen."""
    maxima = LI.local_maxima(2.0, 0.5)
    assert len(maxima) == 1 and maxima[0][0] == pytest.approx(0.5, abs=1e-5)


def test_mode_of_rate_matches_the_largest_maximum():
    assert LI.mode_of_rate(6.0, 0.25) == max(LI.local_maxima(6.0, 0.25), key=lambda t: t[1])[0]
    assert LI.mode_of_rate(0.001, 0.25) == pytest.approx(0.5, abs=0.01)    # fast ohne Schärfe liegt der Modus in der Mitte (Entropie allein)


def test_second_branch_start_for_the_two_p_stars():
    """Das zweite Maximum (der unterlegene Ast) entsteht erst bei hoher Schärfe; bei höherem p* früher."""
    low, high = LI.second_branch_start(0.25), LI.second_branch_start(0.4)
    assert high < low
    assert len(LI.local_maxima(low + 0.05, 0.25)) == 2 and len(LI.local_maxima(low - 0.05, 0.25)) == 1


def test_second_branch_start_is_none_when_there_is_none():
    assert LI.second_branch_start(0.25, hi=5.0) is None


def test_middle_mass_boundaries_are_inclusive_and_integer():
    """n = 10: 0,4·n = 4 bis 0,6·n = 6 inklusive."""
    for k in range(11):
        pi = np.zeros(11)
        pi[k] = 1.0
        assert (LI.middle_mass(pi) == 1.0) == (4 <= k <= 6), k


def test_middle_mass_with_other_limits():
    pi = np.full(11, 1 / 11)
    assert LI.middle_mass(pi, 0.0, 1.0) == pytest.approx(1.0) and LI.middle_mass(pi, 0.5, 0.5) == pytest.approx(1 / 11)
