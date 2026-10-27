"""ggw_three: Zusammensetzungen, Wahlwahrscheinlichkeiten, Kette und GTH gegen Handrechnung."""

import math

import numpy as np
import pytest

import ggw_three as T


def test_compositions_of_two_into_three_by_hand():
    assert T.compositions(2, 3) == [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]


def test_composition_counts():
    """C(n + m - 1, m - 1): n = 4, m = 3 gibt 15; n = 12, m = 3 gibt 91; n = 20 gibt 231; n = 5, m = 2 gibt 6."""
    assert len(T.compositions(4, 3)) == 15 and len(T.compositions(12, 3)) == 91 and len(T.compositions(20, 3)) == 231 and len(T.compositions(5, 2)) == 6
    assert all(sum(s) == 7 for s in T.compositions(7, 3))


def test_logit_choice_probabilities_by_hand():
    """u = (0, 0): je 1/2; u = (ln 2, 0) bei beta = 1: 2/3 zu 1/3."""
    assert T.choice_probabilities(np.array([0.0, 0.0]), T.LOGIT, 3.0).tolist() == pytest.approx([0.5, 0.5])
    assert T.choice_probabilities(np.array([math.log(2), 0.0]), T.LOGIT, 1.0).tolist() == pytest.approx([2 / 3, 1 / 3])


def test_uniform_choice_probabilities_by_hand():
    """eps = 0,3, drei Formate, u = (2, 1, 2): beste Antworten 0 und 2: je 0,7/2 + 0,3/3 = 0,45 und für das mittlere 0,1."""
    q = T.choice_probabilities(np.array([2.0, 1.0, 2.0]), T.UNIFORM, 0.3)
    assert q.tolist() == pytest.approx([0.45, 0.1, 0.45]) and q.sum() == pytest.approx(1.0)


def test_unknown_choice_rule_raises():
    with pytest.raises(ValueError):
        T.choice_probabilities(np.array([1.0, 0.0]), "zufall", 1.0)


def test_two_format_chain_for_two_firms_by_hand():
    """n = 2, Spiel (5, 2, 1, 2) als Matrix [[5, 1], [2, 2]] (Zeile = eigenes Format), eps = 0,2.
    Zustand (1, 1): der S-Betrieb (Wkt 1/2) sieht H: u_S = 1, u_H = 2, H beste Antwort: wählt S mit 0,1; der H-Betrieb (1/2) sieht S: u_S = 5, u_H = 2: wählt S mit 0,9.
    Nach (2, 0): H-Betrieb wird S mit 0,5·0,9 = 0,45; nach (0, 2): S-Betrieb wird H mit 0,5·0,9 = 0,45; Rest 0,1."""
    states, p = T.chain_m(2, [[5, 1], [2, 2]], T.UNIFORM, 0.2)
    assert states == [(0, 2), (1, 1), (2, 0)]
    assert p[1].tolist() == pytest.approx([0.45, 0.1, 0.45])
    assert p.sum(axis=1).tolist() == pytest.approx([1, 1, 1])


def test_gth_of_a_two_state_chain():
    """P = [[0,9, 0,1], [0,3, 0,7]]: pi = (0,75, 0,25) (Gleichgewicht 0,1·pi_0 = 0,3·pi_1)."""
    assert T.gth(np.array([[0.9, 0.1], [0.3, 0.7]])).tolist() == pytest.approx([0.75, 0.25])


def test_gth_is_stable_for_tiny_probabilities():
    """Eine Kette mit Übergang 1e-60: pi_1 = 1e-60·pi_0/0,5 (nur GTH ohne Subtraktion trifft diese Größenordnung)."""
    p = np.array([[1 - 1e-60, 1e-60], [0.5, 0.5]])
    pi = T.gth(p)
    assert pi[1] == pytest.approx(2e-60, rel=1e-9) and pi[0] == pytest.approx(1.0)


def test_pure_states_and_masses():
    states = T.compositions(3, 3)
    idx = T.pure_states(states)
    assert [states[i] for i in idx] == [(3, 0, 0), (0, 3, 0), (0, 0, 3)]
    pi = np.zeros(len(states))
    pi[idx[1]] = 0.7
    pi[idx[0]] = 0.3
    assert T.pure_masses(states, pi).tolist() == [0.3, 0.7, 0.0]


def test_selected_format_needs_half_of_the_mass():
    """Genau 0,5 Masse im reinen Zustand reicht (Grenze inklusive), 0,49 nicht; ohne Mindestmasse wird immer gewählt."""
    states = T.compositions(3, 3)
    idx = T.pure_states(states)
    pi = np.zeros(len(states))
    pi[idx[2]], pi[1] = 0.5, 0.5          # Zustand 1 ist (0, 1, 2), kein reiner Zustand
    assert T.selected_format(states, pi) == 2
    pi[idx[2]], pi[1] = 0.49, 0.51
    assert T.selected_format(states, pi) is None and T.selected_format(states, pi, min_mass=0.0) == 2
    assert T.selected_format(states, np.eye(len(states))[idx[1]]) == 1


def test_logit_chain_with_equal_diagonals_is_symmetric():
    states, p = T.chain_m(6, np.eye(3) * 2.0, T.LOGIT, 1.0)
    masses = T.pure_masses(states, T.gth(p))
    assert masses[0] == pytest.approx(masses[1], rel=1e-10) == pytest.approx(masses[2], rel=1e-10)


def test_each_wahl_rule_differs_on_one_game():
    """Zweig-Test: Logit und gleichmäßige Fehler geben auf einem Spiel mit knappen Diagonalen verschiedene Massen."""
    A = np.array([[9.0, 0, 0], [0, 8, 0], [0, 0, 1]])
    s1, p1 = T.chain_m(8, A, T.LOGIT, 2.0)
    s2, p2 = T.chain_m(8, A, T.UNIFORM, 0.05)
    assert not np.allclose(T.pure_masses(s1, T.gth(p1)), T.pure_masses(s2, T.gth(p2)), atol=1e-3)
