"""ggw_experiments: Form der Messreihen, vorgerechnete Schwellentabelle (Hash, Zellen), Beispiele des Drei-Formate-Experiments."""

import json
import math
from fractions import Fraction as F

import numpy as np
import pytest

import ggw_chain as CH
import ggw_constants as C
import ggw_experiments as X
import ggw_game as G
import ggw_scenario as S


def test_load_threshold_table_recomputes_when_the_hash_is_stale(tmp_path):
    stale = tmp_path / "t.json"
    stale.write_text(json.dumps({"level": 0.99, "generator_hash": "veraltet", "pstars": ["1/4"], "ns": [10], "x_star": [[1.0]]}), encoding="utf-8")
    table = X.load_threshold_table(stale)
    assert table["generator_hash"] == X.generator_hash() and table["x_star"] != [[1.0]] and len(table["pstars"]) == 4
    assert X.load_threshold_table(tmp_path / "gibt_es_nicht.json")["pstars"] == list(C.THRESHOLD_PSTARS)


def test_generator_hash_changes_with_the_hashed_sources(monkeypatch):
    h = X.generator_hash()
    assert len(h) == 16 and h == X.generator_hash()
    monkeypatch.setattr(X, "HASHED_SOURCES", ("ggw_game.py",))
    assert X.generator_hash() != h


def test_threshold_table_cells_are_the_documented_thresholds():
    """Zelle p* = 1/4, n = 10: 6,06; p* = 1/10, n = 1000: 3,01; Spalten wie in den Konstanten."""
    table = X.compute_threshold_table()
    assert table["x_star"][1][0] == pytest.approx(6.0635, abs=1e-3) and table["x_star"][0][-1] == pytest.approx(3.0140, abs=1e-3)
    assert [len(r) for r in table["x_star"]] == [len(C.THRESHOLD_NS)] * len(C.THRESHOLD_PSTARS)


def test_e1_mass_rows_have_one_row_per_cell_and_sum_to_one():
    rows = X.e1_mass_rows(F(1, 4))
    assert len(rows) == 3 * len(C.SURFACE_XS)
    for r in rows:
        assert r["high"] + r["low"] + r["mid"] == pytest.approx(1.0, abs=1e-12)


def test_e1_surface_matrix_shape_and_monotone_in_x():
    pstars = [F(1, 10), F(1, 4), F(2, 5)]
    xs = [1.0, 3.0, 6.0, 12.0]
    m = X.e1_surface(pstars, xs, 50)
    assert m.shape == (3, 4) and (np.diff(m, axis=1) >= -1e-12).all() and (np.diff(m, axis=0) <= 1e-12).all()


def test_e1_flip_is_symmetric_at_p_star_one_half():
    rows = X.e1_flip_in_p_star()
    assert rows[1]["high"] == pytest.approx(rows[1]["low"], abs=1e-12)
    assert rows[0]["high"] == pytest.approx(rows[2]["low"], abs=1e-12)


def test_e2_invariance_rows():
    rows = X.e2_invariance()
    assert len(rows) == len(C.INVARIANCE_GAMES) and {r["lam"] for r in rows} == {4} and {r["pstar"] for r in rows} == {F(1, 4)}
    assert sorted(r["a_minus_b"] for r in rows) == [-91, -2, 3, 3, 4]
    assert max(r["deviation"] for r in rows) < 1e-14


def test_e2_payoff_dominated_games_have_s_paying_one_over_p_star_times():
    rows = X.e2_payoff_dominated()
    for r in rows:
        assert r["ratio"] * r["pstar"] == 1


def test_e3_tables_have_the_documented_shape():
    mode = X.e3_mode_rows()
    assert len(mode) == 2 * 9 and all(r["gap"] < 0.002 for r in mode)
    mid = X.e3_middle_rows()
    assert len(mid) == len(C.PHASE_NS) * 3
    assert {r["count"] for r in X.e3_fixed_point_rows() if r["pstar"] == 0.5 and r["x"] < 4} == {1}
    assert {r["count"] for r in X.e3_fixed_point_rows() if r["pstar"] == 0.5 and r["x"] > 4} == {3}
    assert [r["x"] > 0 for r in X.e3_second_branch_rows()] == [True, True]


def test_e4_count_rows_slope_equals_count():
    for r in X.e4_count_rows():
        assert r["slope"] == pytest.approx(r["count"], abs=0.01)
        assert r["npos"] + r["nneg"] + r["ties"] == r["n"]


def test_e4_log_ratio_is_exact_for_the_count():
    """Für jedes eps: log10 pi(n)/pi(0) = (N+ - N-)·log10((2 - eps)/eps)."""
    row = X.e4_count_rows(cells=((F(1, 4), 20),), eps_list=(1e-3,))[0]
    assert row["logs"][0] == pytest.approx(row["count"] * math.log10((2 - 1e-3) / 1e-3), rel=1e-9)


def test_e4_tipping_value_by_hand():
    """n = 40, p* = 1/4: Delta_j < 0 für j = 0..9 (10 Zustände), Delta_j > 0 für j = 10..39 (30), also npos/nneg = 3; eps_S = 0,1: odds 9, eps_H = 1/(1 + 9^3) = 1/730."""
    assert X.e4_tipping_eps_h(40, F(1, 4), 0.1) == pytest.approx(1 / 730)
    assert X.e4_tipping_eps_h(40, F(1, 4), 0.5) == pytest.approx(0.5)      # odds 1: 1/(1 + 1)


def test_e4_typed_selection_changes_sides_around_the_tipping_value():
    tip = X.e4_tipping_eps_h(40, F(1, 4), 0.1)
    above = CH.masses(CH.stationary(CH.rates_typed(40, G.game_from(1, F(1, 4)), 0.1, 3 * tip)))
    below = CH.masses(CH.stationary(CH.rates_typed(40, G.game_from(1, F(1, 4)), 0.1, tip / 3)))
    assert above.high > above.low and below.low > below.high
    at = CH.stationary(CH.rates_typed(40, G.game_from(1, F(1, 4)), 0.1, tip))
    assert at[40] == pytest.approx(at[0], rel=1e-9)         # genau an der Schwelle sind „alle S“ und „alle H“ gleich wahrscheinlich


def test_e4_tipping_value_is_none_without_negative_or_positive_states():
    """p* = 0: kein Zustand mit Delta < 0 (nur Delta_0 = 0); p* = 1: kein Zustand mit Delta > 0 (nur Delta_n-1 = 0): es gibt nichts zu kippen."""
    assert X.e4_tipping_eps_h(10, F(0), 0.1) is None and X.e4_tipping_eps_h(10, F(1), 0.1) is None


def test_e5_example_games_exist_and_dissent():
    """Die beiden Beispiele der App: Logit und gleichmäßige Fehler wählen verschieden, beide eindeutig; der Logit folgt dem Potential."""
    for seed, index in C.DISSENTER_EXAMPLES:
        g = S.draw_three_format_games(seed, index + 1)[index]
        ex = X.e5_example(g)
        assert ex["logit"] is not None and ex["uniform"] is not None and ex["logit"] != ex["uniform"]
        assert ex["logit"] == S.potential_format(g)


def test_e5_stream_counts_are_consistent():
    r = X.e5_stream(C.SWEEP_SEEDS[0], count=6)
    assert r["games"] == 6 and r["both_clear"] <= min(r["logit_clear"], r["uniform_clear"]) and r["logit_eq_uniform"] + len(r["dissenters"]) == r["both_clear"]
    assert r["payoff_comparable"] <= r["logit_clear"] and r["logit_eq_payoff"] <= r["payoff_comparable"]


def test_e5_limit_check_returns_masses_for_every_setting():
    g = S.draw_three_format_games(*C.DISSENTER_EXAMPLES[0][:1], C.DISSENTER_EXAMPLES[0][1] + 1)[-1]
    lim = X.e5_limit_check(g)
    assert set(lim["uniform"]) == {1e-4, 1e-6, 1e-8} and set(lim["logit"]) == {20.0, 40.0, 80.0, 160.0}
    for masses in list(lim["uniform"].values()) + list(lim["logit"].values()):
        assert len(masses) == 3 and sum(masses) <= 1 + 1e-9


def test_e6_time_rows_by_cell():
    rows = X.e6_time_rows()
    assert len(rows) == 9 and {r["kstar"] for r in rows if r["n"] == 10} == {3}
    assert all(r["from_s"] > r["from_h"] for r in rows)


def test_e6_simulation_check_reports_capped_runs_instead_of_averaging_them():
    rows = X.e6_simulation_check(cells=((10, 4.0, "S"),), runs=2, seeds=(1, 2), cap_steps=100)
    assert rows[0]["capped"] == 4 and rows[0]["sim"] is None and rows[0]["se"] is None and rows[0]["runs"] == 4


# ------------------------------------------------------------------------------------------------------------ Grenzen (zweite Mutationsrunde)


def test_logit_beta_divides_by_the_largest_diagonal_entry():
    game = {"sym": [[3, 0, 0], [0, 5, 0], [0, 0, 8]]}
    assert X.logit_beta(game, 20.0) == 2.5 and X.logit_beta({"sym": [[9, 0, 0], [0, 5, 0], [0, 0, 8]]}, 18.0) == 2.0 and X.logit_beta({"sym": [[3, 0, 0], [0, 5, 0], [0, 0, 1]]}, 10.0) == 2.0


def test_game_outcome_names_all_four_choices():
    g = S.draw_three_format_games(100003, 107)[106]
    o = X.e5_game_outcome(g)
    assert o["logit"] == 0 and o["uniform"] == 1 and o["potential"] == S.potential_format(g) == 0 and o["payoff"] == S.payoff_dominant_format(g)


def test_a_game_where_the_uniform_rule_does_not_choose_clearly():
    g = S.draw_three_format_games(100002, 1374)[1373]
    o = X.e5_game_outcome(g)
    assert o["logit"] == 0 and o["uniform"] is None


def test_tally_counts_unclear_choices_and_dissenters_separately():
    g = {"sym": [[1]], "payoff": [[1]]}
    outcomes = [
        (g, {"logit": 0, "uniform": 0, "potential": 0, "payoff": 0}),       # alles einig
        (g, {"logit": 0, "uniform": 1, "potential": 0, "payoff": 1}),       # Abweichler
        (g, {"logit": 1, "uniform": None, "potential": 1, "payoff": 2}),    # uniform unklar
        (g, {"logit": None, "uniform": 2, "potential": 0, "payoff": None}),  # logit unklar, kein Auszahlungsfavorit
        (g, {"logit": None, "uniform": None, "potential": 0, "payoff": 0}),
    ]
    t = X.tally_outcomes(outcomes)
    assert (t["logit_clear"], t["uniform_clear"], t["both_clear"]) == (3, 3, 2)
    assert (t["logit_eq_potential"], t["logit_eq_uniform"], len(t["dissenters"])) == (3, 1, 1)
    assert (t["payoff_comparable"], t["logit_eq_payoff"]) == (3, 1)               # logit klar und Favorit da: Spiele 1, 2, 3
    assert (t["uniform_comparable"], t["uniform_eq_payoff"]) == (2, 2)           # uniform klar und Favorit da: Spiele 1 und 2 (Spiel 4 hat keinen Favoriten)
    assert t["dissenters"][0]["logit"] == 0 and t["dissenters"][0]["uniform"] == 1


def test_e6_check_with_a_single_stream_has_no_spread():
    rows = X.e6_simulation_check(cells=((10, 1.0, "H"),), runs=3, seeds=(1,), cap_steps=100_000)
    assert rows[0]["sim"] is not None and rows[0]["std"] is None and rows[0]["se"] is None
    rows2 = X.e6_simulation_check(cells=((10, 1.0, "H"),), runs=3, seeds=(1, 2), cap_steps=100_000)
    assert rows2[0]["std"] is not None and rows2[0]["se"] == pytest.approx(rows2[0]["std"] / math.sqrt(2))


def test_load_threshold_table_discards_a_table_of_another_level(tmp_path):
    other = tmp_path / "t.json"
    other.write_text(json.dumps({"level": 0.5, "generator_hash": X.generator_hash(), "pstars": ["1/4"], "ns": [10], "x_star": [[1.0]]}), encoding="utf-8")
    table = X.load_threshold_table(other)
    assert table["level"] == C.THRESHOLD_LEVEL and table["x_star"] != [[1.0]]
