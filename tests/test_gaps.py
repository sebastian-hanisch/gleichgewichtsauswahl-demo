"""Lücken, die die erste Mutationsprüfung fand: Grenzfälle der Bisektion, gebrochene Lücken in der Klassifikation, Randmaxima der Ratenfunktion, Zahl der Bisektionsschritte,
Grenzen der Zeitformel und der genaue Zufallsstrom der Simulationen (unabhängig mit dem skalaren SplitMix64 nachgespielt)."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import ggw_chain as CH
import ggw_game as G
import ggw_limit as LI
import ggw_sim as SI
import ggw_three as T
from ggw_rng import SplitMix64

GAME = (5, 2, 1, 2)


# ------------------------------------------------------------------------------------------------------------ Bisektion


def test_threshold_equality_with_the_level_counts_as_reached():
    p = F(1, 4)
    level = CH.mass_high_for(20, p, 3.0)
    x = CH.threshold_x(20, p, level=level, x_max=3.0)
    assert x is not None and x == pytest.approx(3.0, abs=1e-9)        # Masse bei x_max gleich dem Niveau: erreicht, nicht „nie“
    level_mid = CH.mass_high_for(20, p, 200.0)
    assert CH.threshold_x(20, p, level=level_mid, steps=1) == 200.0      # Gleichheit bei der Mitte: die Hälfte zählt als erreicht


def test_threshold_bisection_halves_the_interval_once_per_step():
    assert CH.threshold_x(20, F(1, 4), steps=1) == 200.0 and CH.threshold_x(20, F(1, 4), steps=2) == 100.0
    assert CH.threshold_x(20, F(1, 4), steps=0) == 400.0


def test_critical_value_and_second_branch_halve_the_interval_once_per_step():
    assert LI.critical_value(steps=1) == 5.5               # Mitte von 1 und 10: dort gibt es schon drei Lösungen
    assert LI.second_branch_start(0.25, steps=1) == 30.5   # Mitte von 1 und 60: dort gibt es schon zwei Maxima


# ------------------------------------------------------------------------------------------------------------ Klassifikation mit gebrochenen Lücken


def test_classification_with_fractional_gaps_between_zero_and_one():
    """Lücken wie 0,5 oder -0,5 (Auszahlungen mit Nachkommastellen): das Vorzeichen entscheidet, nicht der Betrag."""
    assert G.classify((0.5, 2, 1, 1)) == G.DOMINANT          # a - d = -0,5, b - c = 1
    assert G.classify((3, 1.5, 2, 0)) == G.DOMINANT          # a - d = 3, b - c = -0,5
    assert G.classify((1.5, 0, 1, 1)) == G.DOMINANT          # a - d = 0,5, b - c = -1
    assert G.classify((0, 1.5, 1, 1)) == G.DOMINANT          # a - d = -1, b - c = 0,5
    assert G.classify((1, 0, 1, 1.5)) == G.ANTI              # a - d = -0,5, b - c = -1
    assert G.classify((0, 1, 1.5, 1)) == G.ANTI              # a - d = -1, b - c = -0,5
    assert G.classify((1.5, 1.5, 1, 1)) == G.KOORDINATION and G.classify((1.5, 1, 1, 1)) == G.ENTARTET


# ------------------------------------------------------------------------------------------------------------ Randmaxima


def test_rate_function_maximum_at_the_last_and_first_grid_cell_for_extreme_sharpness():
    """x = 40, p* = 1/4: die Lösung liegt bei 1 - 1e-13, näher als die letzte Zellmitte: das Maximum steht in der letzten Zelle (die erste Zelle ist hier der unterlegene Ast).
    Bei p* = 3/4 ist es umgekehrt."""
    last = LI.local_maxima(40.0, 0.25, 1000)
    assert [round(p, 6) for p, _ in last] == [0.0005, 0.9995]
    first = LI.local_maxima(40.0, 0.75, 1000)
    assert [round(p, 6) for p, _ in first] == [0.0005, 0.9995] and first[0][1] > first[1][1]          # Hauptast in der ersten Zelle, unterlegener Ast in der letzten


# ------------------------------------------------------------------------------------------------------------ Zeitformel


def test_hitting_time_can_target_the_last_state_and_rejects_values_outside():
    rates = CH.rates_logit(5, GAME, 1.0)
    p = CH.transition_matrix(rates)
    t = np.linalg.solve(np.eye(5) - p[:5, :5], np.ones(5))
    assert SI.log10_hitting_time(rates, 5) == pytest.approx(math.log10(t[0]), abs=1e-9)         # Ziel = n ist erlaubt
    for bad in (0, 6):
        with pytest.raises(ValueError, match="außerhalb"):
            SI.log10_hitting_time(rates, bad)


# ------------------------------------------------------------------------------------------------------------ Simulation: genauer Strom


def reference_players(n, game, beta, steps, burn, start_k, seed):
    """Die Simulation mit dem skalaren SplitMix64-Strom und der Auszahlung aus dem ganzen Zustandsvektor nachgespielt (je Schritt zuerst der Betrieb, dann die Wahl)."""
    a, b, c, d = (float(v) for v in game)
    rng = SplitMix64(seed)
    state = [1] * start_k + [0] * (n - start_k)
    counts = [0] * (n + 1)
    for t in range(steps):
        i = int(rng.uniform() * n)
        coin = rng.uniform()
        others = [state[j] for j in range(n) if j != i]
        u_s = sum(a if o else c for o in others) / (n - 1)
        u_h = sum(d if o else b for o in others) / (n - 1)
        p_s = 1 / (1 + math.exp(-beta * (u_s - u_h)))
        state[i] = 1 if coin < p_s else 0
        if t >= burn:
            counts[sum(state)] += 1
    total = sum(counts)
    return [v / total for v in counts]


def test_simulate_players_replays_exactly_the_documented_stream():
    for n, start_k, steps, burn, seed in ((3, 1, 60, 10, 4), (4, 4, 80, 0, 9), (5, 0, 120, 119, 2)):
        got = SI.simulate_players(n, GAME, 0.9, steps, burn, start_k, seed)
        assert got.tolist() == pytest.approx(reference_players(n, GAME, 0.9, steps, burn, start_k, seed), abs=1e-15), (n, start_k)


def test_simulate_players_rejects_a_negative_start():
    with pytest.raises(ValueError):
        SI.simulate_players(5, GAME, 1.0, 100, 10, -1, 1)


def reference_hitting(n, game, beta, from_side, run, seed, cap):
    """Ein Lauf der Austrittszeit mit dem Strom SplitMix64(seed + 1 000 003·run) (erster Block), Auszahlung aus dem ganzen Vektor; gibt die Zahl der Einzelaktualisierungen oder None bei Abbruch."""
    a, b, c, d = (float(v) for v in game)
    kstar = G.barrier_state(n, game)
    rng = SplitMix64(seed + 1_000_003 * run)
    state = [0] * n if from_side == "H" else [1] * n
    steps = 0
    while steps < cap and (sum(state) < kstar if from_side == "H" else sum(state) > kstar):
        i = int(rng.uniform() * n)
        coin = rng.uniform()
        others = [state[j] for j in range(n) if j != i]
        u_s = sum(a if o else c for o in others) / (n - 1)
        u_h = sum(d if o else b for o in others) / (n - 1)
        state[i] = 1 if coin < 1 / (1 + math.exp(-beta * (u_s - u_h))) else 0
        steps += 1
    done = sum(state) >= kstar if from_side == "H" else sum(state) <= kstar
    return steps if done else None


@pytest.mark.parametrize("side,beta", [("H", 0.7), ("S", 0.5)])
def test_simulate_hitting_replays_exactly_the_documented_stream(side, beta):
    n = 4
    times, capped = SI.simulate_hitting(n, GAME, beta, side, 6, 3, 100_000)
    ref = [reference_hitting(n, GAME, beta, side, run, 3, 100_000) for run in range(6)]
    assert capped == 0 and [t * n for t in times] == pytest.approx(ref)


def test_simulate_hitting_cap_is_exclusive_of_one_more_step():
    """Braucht ein Lauf genau T Einzelaktualisierungen, endet er mit Obergrenze T und wird mit Obergrenze T - 1 abgebrochen."""
    n = 4
    ref = [reference_hitting(n, GAME, 0.7, "H", run, 3, 100_000) for run in range(8)]
    run = int(np.argmax(ref))
    steps = ref[run]
    assert steps > 1
    assert reference_hitting(n, GAME, 0.7, "H", run, 3, steps) == steps and reference_hitting(n, GAME, 0.7, "H", run, 3, steps - 1) is None
    times, capped = SI.simulate_hitting(n, GAME, 0.7, "H", run + 1, 3, steps - 1)
    assert capped >= 1 and len(times) == run + 1 - capped
    times_full, capped_full = SI.simulate_hitting(n, GAME, 0.7, "H", run + 1, 3, steps)
    assert capped_full < capped and steps / n in times_full


# ------------------------------------------------------------------------------------------------------------ Wahlwahrscheinlichkeiten


def test_logit_choice_is_stable_for_large_payoffs():
    """Auszahlungen um 1000: ohne Verschiebung um das Maximum liefe exp(1000) über; mit ihr ist das Verhältnis e^-1."""
    q = T.choice_probabilities(np.array([1000.0, 999.0]), T.LOGIT, 1.0)
    assert q.tolist() == pytest.approx([1 / (1 + math.exp(-1)), math.exp(-1) / (1 + math.exp(-1))])
