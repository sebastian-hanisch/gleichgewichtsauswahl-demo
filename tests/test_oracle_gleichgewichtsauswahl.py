"""Orakel-Tests gleichgewichtsauswahl-demo: jede Kernroutine gegen einen anderen Rechenweg (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §2 „Spieltheorie“, „Warteschlangen“).

1. Stationäre Verteilung der Geburts-Sterbe-Kette gegen GTH an der VOLLEN Übergangsmatrix (hier im Test, eigene Fassung), gegen den numpy-Eigenvektor (Toleranz 1e-5: numpy löst
   Wahrscheinlichkeiten unter 1e-10 schlecht auf), gegen die geschlossene Binomialform und, für Fehlermodelle, gegen die exakte Bruchrechnung.
2. Gibbs-Maß über ALLE 2^n Konfigurationen mit dem Paarpotential (n <= 10) und der Nachweis, dass das Paarpotential ein Potential der Auszahlungen ist (Brüche).
3. Einzelspieler-Simulation, bei der jede Aktualisierung die Auszahlung aus dem ganzen Zustandsvektor rechnet (anderer Zufallsstrom als die Demo).
4. Ratenfunktion: Maximum von g gegen den Modus der exakten Verteilung (n = 2000); Zahl der Logit-Gleichgewichte gegen Iteration der Antwortabbildung.
5. Mittlere Austrittszeit: gegen das lineare Gleichungssystem der vollen Kette und gegen die Simulation.
6. Kette über Zusammensetzungen (zwei Formate) gegen die Geburts-Sterbe-Kette; drei Formate gegen exakte Bruchrechnung.
Die große Fassung (300 Zufallsspiele, n bis 40, 1,5·10^6 Schritte) lief im Scratch-Ordner; hier die kleine, schnelle (< 10 s)."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

import ggw_chain as CH
import ggw_game as G
import ggw_limit as LI
import ggw_sim as SI
import ggw_three as T

# ------------------------------------------------------------------------------------------------------------ Hilfen


def oracle_gth(P):
    """Grassmann–Taksar–Heyman, eigene Fassung im Test: Zustände von hinten entfernen, ohne Subtraktion, mit Schleifen statt Feldoperationen."""
    k = len(P)
    A = [[float(P[i][j]) for j in range(k)] for i in range(k)]
    for m in range(k - 1, 0, -1):
        s = sum(A[m][j] for j in range(m))
        for i in range(m):
            for j in range(m):
                A[i][j] += A[i][m] * A[m][j] / s
    pi = [1.0] + [0.0] * (k - 1)
    for m in range(1, k):
        pi[m] = sum(pi[i] * A[i][m] for i in range(m)) / sum(A[m][j] for j in range(m))
    total = sum(pi)
    return np.array([v / total for v in pi])


def random_game(rng):
    """Zufälliges Koordinationsspiel mit reellen Auszahlungen (a > d, b > c)."""
    while True:
        a, b = rng.uniform(1, 6, 2)
        c, d = rng.uniform(0, 0.9 * min(a, b), 2)
        if a > d and b > c:
            return (float(a), float(b), float(c), float(d))


def enumerate_gibbs(n, game, beta):
    """Verteilung der Zahl k der S-Betriebe aus dem Gibbs-Maß exp(beta·Phi) über alle 2^n Konfigurationen; Phi = Summe über Paare P(s_i, s_j)/(n - 1)."""
    gap_s, gap_h = float(G.gaps(game)[0]), float(G.gaps(game)[1])
    weights = np.zeros(n + 1)
    phis = {}
    for config in itertools.product((0, 1), repeat=n):
        k = sum(config)
        phi = sum((gap_s if (config[i] and config[j]) else gap_h if (not config[i] and not config[j]) else 0.0) for i, j in itertools.combinations(range(n), 2)) / (n - 1)
        phis[config] = (k, phi)
    top = max(p for _, p in phis.values())
    for k, phi in phis.values():
        weights[k] += math.exp(beta * (phi - top))
    return weights / weights.sum()


# ------------------------------------------------------------------------------------------------------------ 1. Stationäre Verteilung


def test_chain_against_gth_and_eigenvector_on_the_full_matrix():
    rng = np.random.default_rng(20261027)
    worst_gth = worst_eig = worst_closed = 0.0
    for _ in range(40):
        n = int(rng.integers(4, 41))
        game = random_game(rng)
        beta = float(rng.uniform(0.2, 12)) / float(G.lam(game))
        rates = CH.rates_logit(n, game, beta)
        pi = CH.stationary(rates)
        p = CH.transition_matrix(rates)
        worst_gth = max(worst_gth, float(np.abs(pi - oracle_gth(p)).max()))
        w, v = np.linalg.eig(p.T)
        x = np.real(v[:, int(np.argmin(np.abs(w - 1)))])
        worst_eig = max(worst_eig, float(np.abs(pi - x / x.sum()).max()))
        worst_closed = max(worst_closed, float(np.abs(pi - CH.stationary_logit_closed(n, game, beta)).max()))
    assert worst_gth < 1e-12 and worst_closed < 1e-12
    assert worst_eig < 1e-5      # numpy-Eigenvektor: nur Näherung (große Stellen), daher die weite Toleranz


@pytest.mark.parametrize("kind", ["uniform", "typed"])
def test_error_model_chains_against_gth_on_the_full_matrix(kind):
    rng = np.random.default_rng(7)
    for _ in range(25):
        n = int(rng.integers(4, 31))
        game = random_game(rng)
        eps = float(10 ** rng.uniform(-6, -0.5))
        rates = CH.rates_uniform(n, game, eps) if kind == "uniform" else CH.rates_typed(n, game, eps, float(10 ** rng.uniform(-6, -0.5)))
        assert np.abs(CH.stationary(rates) - oracle_gth(CH.transition_matrix(rates))).max() < 1e-12


def test_error_models_with_integer_payoffs_against_exact_fractions():
    """Mit ganzzahligen Auszahlungen und rationalem eps sind die Übergangswahrscheinlichkeiten Brüche: das Gleichungssystem pi·P = pi wird exakt gelöst (Gauß mit Brüchen)."""
    for game, n, eps in (((5, 2, 1, 2), 5, F(1, 5)), ((6, 4, 0, 4), 6, F(1, 10)), ((3, 1, 1, 0), 4, F(1, 3))):
        s = [G.advantage_numerator(n, j, game) for j in range(n)]
        up, down = [], []
        for k in range(n):
            p_s = (1 - eps / 2) if s[k] > 0 else eps / 2 if s[k] < 0 else F(1, 2)
            up.append(F(n - k, n) * p_s)
            down.append(F(k + 1, n) * (1 - p_s))
        ratio = [F(1)]
        for k in range(n):
            ratio.append(ratio[-1] * up[k] / down[k])
        total = sum(ratio)
        exact = [float(r / total) for r in ratio]
        assert CH.stationary(CH.rates_uniform(n, game, float(eps))).tolist() == pytest.approx(exact, rel=1e-12)


def test_stationary_vector_is_invariant_under_the_full_matrix():
    for game, n, x in (((5, 2, 1, 2), 30, 3.0), ((6, 4, 0, 4), 25, 5.0)):
        rates = CH.rates_logit(n, game, G.beta_from_x(game, x))
        pi, p = CH.stationary(rates), CH.transition_matrix(rates)
        assert np.abs(pi @ p - pi).max() < 1e-13


# ------------------------------------------------------------------------------------------------------------ 2. Aufzählung aller Konfigurationen


def test_gibbs_over_all_configurations_against_the_chain():
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(12):
        n = int(rng.integers(4, 11))
        game = random_game(rng)
        beta = float(rng.uniform(0.2, 12)) / float(G.lam(game))
        worst = max(worst, float(np.abs(CH.stationary(CH.rates_logit(n, game, beta)) - enumerate_gibbs(n, game, beta)).max()))
    assert worst < 1e-12


def test_pair_potential_is_a_potential_of_the_payoffs_with_fractions():
    """Für jede Konfiguration und jeden Betrieb: der Auszahlungsgewinn eines Wechsels (Mittel gegen die n - 1 anderen) ist die Potentialdifferenz (Paarpotential geteilt durch n - 1)."""
    pay = {("S", "S"): 5, ("H", "H"): 2, ("S", "H"): 1, ("H", "S"): 2}        # (eigenes, anderes) -> a, b, c, d
    gap = {("S", "S"): 3, ("H", "H"): 1, ("S", "H"): 0, ("H", "S"): 0}        # Paarpotential P(S,S) = a - d, P(H,H) = b - c
    n = 5
    for config in itertools.product("SH", repeat=n):
        for i in range(n):
            flipped = config[:i] + ("H" if config[i] == "S" else "S",) + config[i + 1:]
            u = lambda cfg: F(sum(pay[(cfg[i], cfg[j])] for j in range(n) if j != i), n - 1)
            phi = lambda cfg: F(sum(gap[(cfg[a], cfg[b])] for a, b in itertools.combinations(range(n), 2)), n - 1)
            assert u(flipped) - u(config) == phi(flipped) - phi(config)


# ------------------------------------------------------------------------------------------------------------ 3. Einzelspieler-Simulation


def full_vector_simulation(n, game, beta, steps, burn, seed):
    """Jede Aktualisierung rechnet die Auszahlung beider Formate aus dem ganzen Zustandsvektor (Mittel über die n - 1 anderen), ohne Zähl-Abkürzung; numpy-Zufall."""
    a, b, c, d = (float(v) for v in game)
    rng = np.random.default_rng(seed)
    who = rng.integers(0, n, size=steps)
    coin = rng.random(steps)
    state = (rng.random(n) < 0.5).astype(int)
    hist = np.zeros(n + 1)
    for t in range(steps):
        i = who[t]
        others = np.delete(state, i)
        u_s = np.where(others == 1, a, c).mean()
        u_h = np.where(others == 1, d, b).mean()
        state[i] = 1 if coin[t] < 1.0 / (1.0 + math.exp(-beta * (u_s - u_h))) else 0
        if t >= burn:
            hist[int(state.sum())] += 1
    return hist / hist.sum()


@pytest.mark.parametrize("pstar,x", [(F(1, 4), 3.0), (F(1, 2), 3.0)])
def test_player_level_simulation_against_the_exact_distribution(pstar, x):
    n = 8
    game = G.game_from(1, pstar)
    exact = CH.stationary(CH.rates_logit(n, game, x))
    hist = np.mean([full_vector_simulation(n, game, x, 50_000, 10_000, seed) for seed in (1, 2, 3, 4)], axis=0)
    assert SI.total_variation(hist, exact) < 0.02


def test_product_simulation_against_the_exact_distribution():
    """Die Simulation der Demo (SplitMix64, Zähl-Abkürzung) gegen die exakte Verteilung: vier Zufalls-Seeds, Gesamtvariationsabstand des Mittels unter 0,02."""
    game = G.game_from(1, F(1, 4))
    exact = CH.stationary(CH.rates_logit(10, game, 4.0))
    hist = np.mean([SI.simulate_players(10, game, 4.0, 300_000, 50_000, 5, seed) for seed in (11, 12, 13, 14)], axis=0)
    assert SI.total_variation(hist, exact) < 0.02


# ------------------------------------------------------------------------------------------------------------ 4. Ratenfunktion


@pytest.mark.parametrize("pstar", [0.25, 0.4])
def test_mode_of_the_exact_distribution_against_the_maximum_of_the_rate_function(pstar):
    n = 2000
    game = G.game_from(1, F(pstar).limit_denominator(100))
    for x in (1, 3, 6, 12, 20):
        mode = int(np.argmax(CH.stationary(CH.rates_logit(n, game, x)))) / n
        assert abs(mode - LI.mode_of_rate(x, pstar)) < 0.002, (pstar, x)


def test_fixed_points_against_iteration_of_the_response_map():
    """Stabile Lösungen von y = sigma(X·(y - p*)) als Grenzwerte der Iteration von 40 Startwerten; sie müssen genau die lokalen Maxima von g sein."""
    for pstar, x in ((0.25, 6.0), (0.5, 6.0), (0.4, 14.0), (0.5, 3.0)):
        limits = set()
        for y0 in np.linspace(0.011, 0.989, 40):      # gerade Zahl: der instabile Punkt 1/2 ist kein Startwert
            y = y0
            for _ in range(5000):
                y = 1 / (1 + math.exp(-x * (y - pstar)))
            limits.add(round(y, 6))
        maxima = sorted(round(p, 3) for p, _ in LI.local_maxima(x, pstar))
        assert maxima == sorted(round(v, 3) for v in limits), (pstar, x)


def test_fixed_point_count_at_the_grid_point_one_half():
    """Fehler der ersten Fassung: das Vorzeichen null bei y = 1/2 (ungerades Gitter) wurde falsch gezählt. 1 Lösung bei X < 4, 3 bei X > 4, auf geradem und ungeradem Gitter."""
    for points in (200_000, 200_001):
        assert LI.count_fixed_points(3.5, 0.5, points) == 1
        assert LI.count_fixed_points(4.5, 0.5, points) == 3


# ------------------------------------------------------------------------------------------------------------ 5. Austrittszeit


def linear_system_hitting_time(rates, target):
    """Mittlere Zahl der Einzelaktualisierungen von 0 bis target aus dem linearen System (I - Q)t = 1 der vollen Kette (numpy)."""
    p = CH.transition_matrix(rates)
    q = p[:target, :target]
    t = np.linalg.solve(np.eye(target) - q, np.ones(target))
    return float(t[0])


@pytest.mark.parametrize("n,pstar,x", [(10, F(1, 4), 1.0), (10, F(1, 4), 2.0), (20, F(1, 4), 1.5), (30, F(2, 5), 1.0)])
def test_hitting_time_formula_against_the_linear_system(n, pstar, x):
    game = G.game_from(1, pstar)
    rates = CH.rates_logit(n, game, x)
    kstar = G.barrier_state(n, game)
    assert SI.log10_hitting_time(rates, kstar) == pytest.approx(math.log10(linear_system_hitting_time(rates, kstar)), abs=1e-9)
    rev = SI.reverse(rates)
    assert SI.log10_hitting_time(rev, n - kstar) == pytest.approx(math.log10(linear_system_hitting_time(rev, n - kstar)), abs=1e-6)


def test_hitting_time_against_the_simulation_within_four_standard_errors():
    """Formel gegen Einzelspieler-Simulation (SplitMix64-Strom, 120 Läufe), n = 10, p* = 1/4: Austritt aus „alle S“ bei x = 1 und 2, aus „alle H“ bei x = 2."""
    game = G.game_from(1, F(1, 4))
    for side, x in (("S", 1.0), ("S", 2.0), ("H", 2.0)):
        rates = CH.rates_logit(10, game, x)
        predicted = 10 ** SI.escape_log_time(rates, game, side)
        times, capped = SI.simulate_hitting(10, game, x, side, 120, 5, 2_000_000)
        assert capped == 0
        se = np.std(times, ddof=1) / math.sqrt(len(times))
        assert abs(np.mean(times) - predicted) < 4 * se, (side, x, np.mean(times), predicted, se)


# ------------------------------------------------------------------------------------------------------------ 6. Kette über Zusammensetzungen


@pytest.mark.parametrize("mode,par", [("logit", 1.5), ("logit", 4.0), ("uniform", 0.1), ("uniform", 0.01)])
def test_composition_chain_with_two_formats_equals_the_birth_death_chain(mode, par):
    """Die Kette über Zusammensetzungen (GTH, allgemeine Matrix) mit zwei Formaten gegen die Geburts-Sterbe-Kette: ganz anderer Code."""
    for game in ((5, 2, 1, 2), (6, 4, 0, 4), (3, 5, 4, 0)):
        a, b, c, d = game
        n = 9
        states, p = T.chain_m(n, [[a, c], [d, b]], mode, par)
        pi_m = T.gth(p)
        by_k = np.zeros(n + 1)
        for s, v in zip(states, pi_m):
            by_k[s[0]] += v
        if mode == "logit":
            rates = CH.rates_logit(n, game, par)
        else:
            rates = CH.rates_uniform(n, game, par)
        assert np.abs(by_k - CH.stationary(rates)).max() < 1e-11


def test_three_formats_against_exact_fractions():
    """n = 4, drei Formate, gleichmäßige Fehler eps = 1/10, ganzzahlige Auszahlungen: P ist rational, pi wird mit Brüchen gelöst und mit GTH verglichen."""
    A = [[5, 1, 0], [1, 4, 2], [0, 2, 3]]
    n, eps = 4, F(1, 10)
    states = T.compositions(n, 3)
    index = {s: i for i, s in enumerate(states)}
    P = [[F(0)] * len(states) for _ in states]
    for s in states:
        for i in range(3):
            if s[i] == 0:
                continue
            u = [sum(A[i2][j] * ((s[j] - 1) if j == i else s[j]) for j in range(3)) for i2 in range(3)]    # 3·(n - 1) mal Auszahlung, gleicher Nenner
            best = [i2 for i2 in range(3) if u[i2] == max(u)]
            for j in range(3):
                q = (1 - eps) * (F(1, len(best)) if j in best else 0) + eps / 3
                t = list(s)
                t[i] -= 1
                t[j] += 1
                P[index[s]][index[tuple(t)]] += F(s[i], n) * q
    assert all(sum(r) == 1 for r in P)       # jede Ziehung eines Betriebs ist erfasst (auch die, bei der er sein Format behält)
    # pi·P = pi mit Summe 1: Gauß mit Brüchen auf (P^T - I) mit ersetzter letzter Zeile
    m = len(states)
    M = [[P[j][i] - (1 if i == j else 0) for j in range(m)] + [F(0)] for i in range(m)]
    M[-1] = [F(1)] * m + [F(1)]
    for col in range(m):
        piv = next(r for r in range(col, m) if M[r][col] != 0)
        M[col], M[piv] = M[piv], M[col]
        M[col] = [v / M[col][col] for v in M[col]]
        for r in range(m):
            if r != col and M[r][col] != 0:
                f = M[r][col]
                M[r] = [v - f * w for v, w in zip(M[r], M[col])]
    exact = [float(M[i][m]) for i in range(m)]
    states2, p = T.chain_m(n, np.array(A, dtype=float), "uniform", 0.1)
    assert states2 == states
    assert T.gth(p).tolist() == pytest.approx(exact, rel=1e-10)


def test_three_formats_with_equal_diagonals_give_equal_pure_masses():
    """Symmetrie: A = 3·Einheitsmatrix, alle drei Formate gleich: die Massen der drei reinen Zustände sind gleich."""
    states, p = T.chain_m(9, np.eye(3) * 3.0, "logit", 2.0)
    masses = T.pure_masses(states, T.gth(p))
    assert masses[0] == pytest.approx(masses[1], rel=1e-9) and masses[1] == pytest.approx(masses[2], rel=1e-9)


def test_gth_agrees_with_the_test_side_gth_and_the_eigenvector_on_three_formats():
    states, p = T.chain_m(7, np.array([[4.0, 1, 0], [1, 3, 2], [0, 2, 5]]), "logit", 1.0)
    pi = T.gth(p)
    assert np.abs(pi - oracle_gth(p)).max() < 1e-12
    w, v = np.linalg.eig(p.T)
    x = np.real(v[:, int(np.argmin(np.abs(w - 1)))])
    assert np.abs(pi - x / x.sum()).max() < 1e-8


# ------------------------------------------------------------------------------------------------------------ 7. Auswertung und Messreihen gegen GTH


def test_main_view_and_experiment_rows_against_gth_on_the_full_matrix():
    """Die Ansicht der App (Verteilung, Massen, Erwartung) und die Massenzeilen von E1 gegen GTH an der vollen Matrix, für alle drei Wahlregeln und vier Spiele."""
    import ggw_constants as C
    import ggw_evaluation as V
    import ggw_experiments as X
    for game in ((5, 2, 1, 2), (6, 4, 0, 4), (3, 5, 4, 0), (-1, 2, 4, 0)):
        for model in C.MODELS:
            s = V.Settings(a=game[0], b=game[1], c=game[2], d=game[3], n=12, x=5.0, model=model, eps_exp=1.5, eps_s_exp=1.0, eps_h_exp=2.5)
            view = V.analyse_main(s)
            pi = oracle_gth(CH.transition_matrix(V.rates_for(s)))
            assert np.abs(view.pi - pi).max() < 1e-12
            k = np.arange(13)
            assert view.masses.high == pytest.approx(pi[10 * k >= 9 * 12].sum(), abs=1e-12) and view.masses.low == pytest.approx(pi[10 * k <= 12].sum(), abs=1e-12)
            assert view.mean == pytest.approx(float(pi @ k) / 12, abs=1e-12)
    for row in X.e1_mass_rows(F(1, 4), ns=(10,), xs=(2, 6)):
        pi = oracle_gth(CH.transition_matrix(CH.rates_logit(10, G.game_from(1, F(1, 4)), row["x"])))
        k = np.arange(11)
        assert row["high"] == pytest.approx(pi[10 * k >= 90].sum(), abs=1e-12) and row["low"] == pytest.approx(pi[10 * k <= 10].sum(), abs=1e-12)
