"""Jede Zahl der README steht hier, nachgerechnet mit dem Code der Demo (DEMO-PLAYBOOK §4, §11a). Bänder statt exakter Gleitkommawerte, wo die Plattform die letzten Stellen ändert.

Die Zählung der Fälle (k von m Zellen) steht neben jeder Aussage über „alle“ oder „meist“; die Zufallsströme sind die festen SWEEP_SEEDS."""

import math
from fractions import Fraction as F

import numpy as np
import pytest

import ggw_chain as CH
import ggw_constants as C
import ggw_experiments as X
import ggw_game as G
import ggw_limit as LI
import ggw_sim as SI
from test_oracle_gleichgewichtsauswahl import enumerate_gibbs, oracle_gth, random_game


def near(value, expected, digits=3):
    """Gleich der in der README gedruckten, auf `digits` Stellen gerundeten Zahl."""
    return abs(value - expected) < 0.5 * 10 ** -digits + 1e-12


def high(n, p, x):
    return CH.mass_high_for(n, p, x)


# ------------------------------------------------------------------------------------------------------------ H1 stationäre Verteilung


def test_h1_formula_equals_gth_closed_form_and_eigenvector_over_300_random_games():
    rng = np.random.default_rng(20261027)
    gth = eig = closed = 0.0
    for _ in range(300):
        n = int(rng.integers(4, 41))
        game = random_game(rng)
        beta = float(rng.uniform(0.2, 12)) / float(G.lam(game))
        rates = CH.rates_logit(n, game, beta)
        pi, p = CH.stationary(rates), CH.transition_matrix(rates)
        gth = max(gth, float(np.abs(pi - oracle_gth(p)).max()))
        w, v = np.linalg.eig(p.T)
        x = np.real(v[:, int(np.argmin(np.abs(w - 1)))])
        eig = max(eig, float(np.abs(pi - x / x.sum()).max()))
        closed = max(closed, float(np.abs(pi - CH.stationary_logit_closed(n, game, beta)).max()))
    assert gth < 1e-13 and closed < 1e-13 and eig < 1e-5


def test_h1_enumeration_of_all_2_to_the_n_configurations_over_60_random_games():
    rng = np.random.default_rng(99)
    worst = 0.0
    for _ in range(60):
        n = int(rng.integers(4, 11))                    # bis n = 10 im Repo (2^10 Konfigurationen); die Messreihe bis n = 14 lief im Scratch-Ordner
        game = random_game(rng)
        beta = float(rng.uniform(0.2, 12)) / float(G.lam(game))
        worst = max(worst, float(np.abs(CH.stationary(CH.rates_logit(n, game, beta)) - enumerate_gibbs(n, game, beta)).max()))
    assert worst < 1e-13


def test_h1_binomial_factor_n10_x2_state_5_beats_state_10():
    """(5, 2, 1, 2), n = 10, beta = 0,5 (x = 2): Zustand k = 5 hat Masse 0,0978 auf 252 Konfigurationen, Zustand k = 10 nur 0,0760 auf einer; je Konfiguration liegt k = 10 um den Faktor 196 vorn."""
    pi = CH.stationary_logit_closed(10, (5, 2, 1, 2), 0.5)
    assert near(pi[5], 0.0978, 4) and near(pi[10], 0.0760, 4)
    assert math.comb(10, 5) == 252
    assert abs(pi[10] / (pi[5] / 252) - 196) < 0.5


# ------------------------------------------------------------------------------------------------------------ H2 Invarianz


def test_h2_five_games_with_equal_lambda_and_p_star_give_one_distribution():
    rows = X.e2_invariance()
    assert [r["a_minus_b"] for r in rows] == [3, 3, -2, 4, -91]
    assert {r["lam"] for r in rows} == {4} and {r["pstar"] for r in rows} == {F(1, 4)}
    assert max(r["deviation"] for r in rows) < 1e-14
    assert near(rows[0]["high"], 0.869)         # n = 30, beta = 1 (x = 4)


# ------------------------------------------------------------------------------------------------------------ H3 / H4 / H5 Auswahl, Schwelle


def test_h3_risk_dominant_format_has_at_least_99_percent_from_x_8_in_all_18_cells():
    cells = [(p, n) for p in (F(1, 10), F(1, 4), F(2, 5)) for n in (10, 20, 50, 100, 200, 1000)]
    masses = {c: high(c[1], c[0], 8.0) for c in cells}
    assert len(cells) == 18 and all(v >= 0.99 for v in masses.values())
    assert min(masses, key=masses.get) == (F(2, 5), 10) and near(min(masses.values()), 0.9915, 4)


def test_h3_beta_2_is_not_a_general_rule_it_is_x_8_only_for_lambda_4():
    """(5, 2, 1, 2): lam = 4, beta = 2 heißt x = 8. Ein Spiel mit Lücken 2 und 1 (lam = 3, p* = 1/3): beta = 2 ist nur x = 6, und bei n = 10 steht die Konvention dann nicht (Schwelle 6,92)."""
    assert G.beta_from_x((5, 2, 1, 2), 8.0) == 2.0
    game = (2, 1, 0, 0)
    assert G.lam(game) == 3 and G.p_star(game) == F(1, 3) and G.beta_from_x(game, 6.0) == 2.0
    assert near(high(10, F(1, 3), 6.0), 0.970) and CH.threshold_x(10, F(1, 3)) == pytest.approx(6.916, abs=1e-3)


def test_h4_masses_at_p_star_one_quarter_by_group_size_and_sharpness():
    expected = {10: [(0.060, 0.932), (0.246, 0.751), (0.852, 0.148), (0.989, 0.011), (0.999, 0.001), (1.000, 0.000)],
                100: [(0.000, 1.000), (0.000, 1.000), (0.941, 0.059), (1.000, 0.000), (1.000, 0.000), (1.000, 0.000)],
                1000: [(0.000, 1.000), (0.000, 1.000), (1.000, 0.000), (1.000, 0.000), (1.000, 0.000), (1.000, 0.000)]}
    for n, row in expected.items():
        for x, (hi, mid) in zip(C.SURFACE_XS, row):
            m = CH.masses(CH.stationary(CH.rates_logit(n, G.game_from(1, F(1, 4)), x)))
            assert near(m.high, hi) and near(m.mid, mid), (n, x)


def test_h4_below_the_threshold_larger_groups_lose_the_convention():
    """x = 2: Masse „fast alle S“ 0,246 (n = 10), weniger als 0,001 (n = 100), 0 (n = 1 000); Mitte 0,751, mehr als 0,999, 1."""
    masses = [CH.masses(CH.stationary(CH.rates_logit(n, G.game_from(1, F(1, 4)), 2.0))) for n in (10, 100, 1000)]
    assert near(masses[0].high, 0.246) and masses[1].high < 0.001 and masses[2].high < 1e-9
    assert near(masses[0].mid, 0.751) and masses[1].mid > 0.999 and masses[2].mid > 1 - 1e-9


def test_h5_threshold_table_values_and_monotonicity():
    t = X.compute_threshold_table()
    expected = {"1/10": [4.97, 4.50, 3.93, 3.59, 3.35, 3.01], "1/4": [6.06, 5.49, 4.79, 4.39, 4.09, 3.70], "2/5": [7.85, 7.03, 6.14, 5.64, 5.27, 4.78], "9/20": [9.96, 7.81, 6.78, 6.23, 5.83, 5.30]}
    for p, row in zip(t["pstars"], t["x_star"]):
        assert [round(v, 2) for v in row] == expected[p], p
    xs = t["x_star"]
    assert all(xs[i][j] < xs[i + 1][j] for j in range(6) for i in range(3))           # 18 von 18 Paaren steigen mit p*
    assert all(xs[i][j] > xs[i][j + 1] for i in range(4) for j in range(5))           # 20 von 20 Paaren fallen mit n


def test_h5_threshold_is_the_exact_level_of_99_percent():
    assert high(50, F(1, 4), CH.threshold_x(50, F(1, 4))) == pytest.approx(0.99, abs=1e-9)


def test_h5_the_mass_in_x_is_monotone_on_the_whole_grid_so_that_bisection_is_valid():
    for p in (F(1, 10), F(1, 4), F(2, 5), F(9, 20)):
        for n in (10, 100):
            v = [high(n, p, x) for x in np.arange(0.25, 40.01, 0.25)]
            assert all(b >= a - 1e-12 for a, b in zip(v, v[1:])), (p, n)


def test_h5_the_flip_in_p_star_is_abrupt_and_p_star_one_half_splits_exactly():
    rows = X.e1_flip_in_p_star()
    assert near(rows[0]["high"], 0.999) and near(rows[1]["high"], 0.500) and near(rows[2]["high"], 0.000) and rows[2]["low"] > 0.999
    assert abs(rows[1]["high"] - rows[1]["low"]) < 1e-12


def test_h5_payoff_dominant_format_loses_at_p_star_055_to_09():
    rows = X.e2_payoff_dominated()
    assert [float(r["ratio"]) for r in rows] == pytest.approx([1.82, 1.67, 1.33, 1.11], abs=0.005)
    assert [round(r[4.0][1], 3) for r in rows] == [0.138, 0.367, 0.895, 0.993]
    assert all(r[12.0][1] > 0.99999999 and r[40.0][1] == pytest.approx(1.0) for r in rows)
    assert max(r[x][0] for r in rows for x in (4.0, 12.0, 40.0)) < 4e-5


# ------------------------------------------------------------------------------------------------------------ H6 / H7 große Gruppen


def test_h6_critical_value_four_and_fixed_point_counts():
    assert LI.critical_value() == pytest.approx(4.0, abs=1e-3)
    counts = {r["x"]: r["count"] for r in X.e3_fixed_point_rows() if r["pstar"] == 0.5}
    assert counts[3.9] == 1 and counts[4.1] == 3 and all(counts[x] == 1 for x in (1.0, 2.0, 3.0)) and all(counts[x] == 3 for x in (5.0, 6.0, 8.0))
    assert {r["count"] for r in X.e3_fixed_point_rows() if r["pstar"] == 0.25} == {1}


def test_h6_middle_mass_at_p_star_one_half():
    rows = {(r["n"], r["x"]): r["mid"] for r in X.e3_middle_rows()}
    expected = {50: (0.573, 0.311, 0.156), 200: (0.859, 0.434, 0.066), 1000: (0.999, 0.644, 0.000), 5000: (1.000, 0.885, 0.000)}
    for n, vals in expected.items():
        for x, v in zip((3.0, 4.0, 4.5), vals):
            assert near(rows[(n, x)], v), (n, x)


def test_h7_mode_of_the_exact_distribution_equals_the_maximum_of_the_rate_function():
    rows = X.e3_mode_rows()
    assert len(rows) == 18 and max(r["gap"] for r in rows) < 0.0006 and max(r["gap"] for r in rows) > 0.0005
    row = next(r for r in rows if r["pstar"] == 0.4 and r["x"] == 6)
    assert near(row["mode"], 0.9685, 4) and near(row["rate"], 0.9679, 4)


def test_h7_second_branch_starts_at_x_14_36_and_6_89():
    rows = {r["pstar"]: r["x"] for r in X.e3_second_branch_rows()}
    assert near(rows[0.25], 14.36, 2) and near(rows[0.4], 6.89, 2)


# ------------------------------------------------------------------------------------------------------------ H8 / H9 Fehlermodelle


def test_h8_slope_equals_the_count_in_5_of_5_cells():
    rows = X.e4_count_rows()
    assert [r["count"] for r in rows] == [10, 24, 4, 10, 4] and all(abs(r["slope"] - r["count"]) < 0.002 for r in rows)
    assert [(r["npos"], r["nneg"]) for r in rows] == [(15, 5), (37, 13), (12, 8), (30, 20), (27, 23)]


def test_h8_uniform_and_logit_choose_the_same_side_in_60_of_63_cells_the_exceptions_are_count_ties():
    agree = tot = 0
    exceptions = []
    for p in (F(1, 10), F(1, 4), F(2, 5), F(9, 20), F(11, 20), F(3, 5), F(3, 4)):
        for n in (10, 20, 50):
            game = G.game_from(1, p)
            lg = CH.masses(CH.stationary(CH.rates_logit(n, game, 30.0)))
            npos, nneg, _ = G.sign_counts(n, game)
            for eps in (0.1, 0.01, 0.001):
                un = CH.masses(CH.stationary(CH.rates_uniform(n, game, eps)))
                tot += 1
                same = (lg.high > lg.low) == (un.high > un.low)
                agree += same
                if not same:
                    exceptions.append((p, n, eps, npos, nneg))
    assert (agree, tot) == (60, 63)
    assert exceptions == [(F(9, 20), 10, 0.1, 5, 5), (F(9, 20), 10, 0.01, 5, 5), (F(9, 20), 10, 0.001, 5, 5)]
    for eps, value in ((0.1, 0.457), (0.01, 0.499), (0.001, 0.500)):          # Gleichstand der Zählung: beide Seiten gleich groß
        tie = CH.masses(CH.stationary(CH.rates_uniform(10, G.game_from(1, F(9, 20)), eps)))
        assert near(tie.high, value) and near(tie.low, value)


def test_h8_finite_group_uniform_errors_barely_distinguish_p_star_the_logit_does():
    rows = X.e4_finite_rows()
    assert [round(v, 3) for v in rows[0]["uniform"]] == [0.405, 0.677, 0.925] and [round(v, 3) for v in rows[1]["uniform"]] == [0.405, 0.677, 0.925]
    assert max(abs(a - b) for a, b in zip(rows[0]["uniform"], rows[1]["uniform"])) < 0.0005 and rows[0]["uniform"] != rows[1]["uniform"]
    assert near(rows[0]["logit"], 0.857) and near(rows[1]["logit"], 0.487)


def test_h9_typed_errors_tip_the_selection_exactly_at_the_counted_value():
    tip = X.e4_tipping_eps_h(40, F(1, 4), 0.1)
    assert tip == pytest.approx(1 / 730) and near(tip, 1.4e-3, 4)
    assert G.sign_counts(40, G.game_from(1, F(1, 4))) == (30, 10, 0)
    rows = {r["eps_h"]: r for r in X.e4_typed_rows()}
    assert near(rows[0.1]["high"], 0.629) and near(rows[0.001]["high"], 0.463) and near(rows[0.001]["low"], 0.264) and near(rows[0.001]["mid"], 0.273)
    assert rows[0.003]["high"] > 0.62 and rows[0.0003]["low"] > 0.9999 and rows[1e-7]["low"] > 1 - 1e-12
    assert all(r["high"] > r["low"] for e, r in rows.items() if e >= 0.003) and all(r["low"] > r["high"] for e, r in rows.items() if e <= 3e-4)


# ------------------------------------------------------------------------------------------------------------ H10 drei Formate


def test_h10_three_formats_160_games():
    r = X.e5_three_formats()
    t = r["total"]
    assert (t["games"], t["logit_clear"], t["uniform_clear"], t["both_clear"]) == (160, 160, 160, 160)
    assert t["logit_eq_potential"] == 160 and t["logit_eq_uniform"] == 160 and r["dissenters"] == []
    assert (t["logit_eq_payoff"], t["payoff_comparable"], t["uniform_eq_payoff"]) == (74, 145, 74)
    assert [s["games"] for s in r["streams"]] == [40] * 4


def test_h10_search_over_2400_games_finds_12_with_different_choices_and_2_unclear():
    r = X.e5_dissenter_scan(count=600)
    assert (r["games"], r["differ"], r["unclear"]) == (2400, 12, 2)
    assert all(tuple(c) in r["found"] for c in C.DISSENTER_EXAMPLES)


def test_h10_the_dissenters_keep_their_choice_in_the_limit_or_not():
    """Die zwei Beispiele: Logit wählt das Potentialformat auch bei Schärfe 160, gleichmäßige Fehler bleiben bei eps = 1e-8 bei ihrem (anderen) Format."""
    import ggw_scenario as S
    for seed, index in C.DISSENTER_EXAMPLES:
        g = S.draw_three_format_games(seed, index + 1)[index]
        lim = X.e5_limit_check(g)
        pot = S.potential_format(g)
        assert int(np.argmax(lim["logit"][160.0])) == pot and lim["logit"][160.0][pot] > 0.999
        assert int(np.argmax(lim["uniform"][1e-8])) != pot


# ------------------------------------------------------------------------------------------------------------ H11 Zeit


def test_h11_escape_times_in_powers_of_ten():
    rows = {(r["n"], r["x"]): r for r in X.e6_time_rows()}
    s_side = {10: (1.31, 2.85, 7.52), 20: (2.36, 5.55, 14.98), 100: (10.68, 26.60, 73.36)}
    for n, vals in s_side.items():
        assert [round(rows[(n, x)]["from_s"], 2) for x in (2.0, 4.0, 8.0)] == list(vals), n
    h_side = [r["from_h"] for r in rows.values()]
    assert min(h_side) > -0.12 and max(h_side) < 0.35 and near(min(h_side), -0.116, 3) and near(max(h_side), 0.348, 3)


def test_h11_formula_against_player_level_simulation_within_three_standard_errors():
    rows = X.e6_simulation_check(runs=100)
    assert len(rows) == 8 and all(r["capped"] == 0 and r["runs"] == 400 for r in rows)
    z = [abs(r["sim"] - r["formula"]) / r["se"] for r in rows]
    assert all(v < 3 for v in z) and max(z) > 2.0
    by = {(r["n"], r["x"], r["side"]): r for r in rows}
    assert near(by[(10, 2.0, "S")]["formula"], 20.48, 2) and near(by[(10, 4.0, "S")]["formula"], 702.5, 1) and near(by[(20, 2.0, "S")]["formula"], 231.0, 1)
    assert near(by[(10, 1.0, "H")]["formula"], 0.824, 3)


def test_h11_a_deep_basin_cannot_be_left_within_the_largest_budget():
    """Austritt aus „alle S“ bei n = 20, x = 4: 10^5,55 Aktualisierungen je Betrieb (rund 350 000), das größte Budget von 3·10^6 Aktualisierungen sind 150 000 je Betrieb."""
    game = G.game_from(1, F(1, 4))
    log_t = SI.escape_log_time(CH.rates_logit(20, game, 4.0), game, "S")
    assert round(log_t, 2) == 5.55 and 10 ** log_t > C.SIM_STEPS[-1] / 20 and C.SIM_STEPS[-1] / 20 == 150_000


def test_h11_default_settings_the_simulation_stays_in_the_s_basin_and_still_matches_the_distribution():
    """Standard (5, 2, 1, 2), n = 50, x = 6: Austritt aus „alle S“ nach 10^24,5 Aktualisierungen je Betrieb (aus „alle H“ 1,3), die Masse im S-Becken ist mehr als 99,9 %;
    die Simulation (Start bei 25 S-Betrieben, 100 000 bzw. 3 000 000 Aktualisierungen) hat Gesamtvariationsabstand 0,020 bzw. 0,007 und ist nicht vom Start abhängig."""
    game = (5, 2, 1, 2)
    beta = G.beta_from_x(game, 6.0)
    rates = CH.rates_logit(50, game, beta)
    assert near(SI.escape_log_time(rates, game, "S"), 24.48, 2) and near(10 ** SI.escape_log_time(rates, game, "H"), 1.31, 2)
    pi = CH.stationary(rates)
    assert CH.masses(pi).high > 0.999
    tv_small = SI.total_variation(SI.simulate_players(50, game, beta, 100_000, 20_000, 25, 35), pi)
    assert near(tv_small, 0.020) and SI.total_variation(SI.simulate_players(50, game, beta, 100_000, 20_000, 0, 35), pi) == pytest.approx(tv_small, abs=1e-4)
    assert near(SI.total_variation(SI.simulate_players(50, game, beta, 3_000_000, 600_000, 25, 35), pi), 0.007)
