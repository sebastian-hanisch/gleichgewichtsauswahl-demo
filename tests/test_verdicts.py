"""Verdikte und Formate an von Hand gesetzten Werten (DEMO-PLAYBOOK §12, §11a Punkt 7): jede Schwelle der Art und jede Zahl im Text steht hier."""

from dataclasses import replace
from fractions import Fraction as F

import math

import pytest

import ggw_constants as C
import ggw_evaluation as V


def settings(**kw):
    base = dict(C.BASE)
    base.update(kw)
    return V.Settings(**base)


def view(**kw):
    return V.analyse_main(settings(**kw))


# ------------------------------------------------------------------------------------------------------------ Formate


def test_formats():
    assert V.de(1.23456, 2) == "1,23" and V.de(0.5) == "0,500"
    assert V.frac(F(2, 3)) == "2/3" and V.frac(F(4, 2)) == "2"


def test_pct_does_not_round_edges_to_exact_values():
    assert V.pct(0.5) == "50,0 %" and V.pct(0.9996) == "mehr als 99,9 %" and V.pct(0.0001) == "weniger als 0,1 %" and V.pct(0.0) == "0,0 %" and V.pct(1.0) == "100,0 %"
    assert V.pct(0.99876, 2) == "99,88 %"


def test_pct_short_form_for_narrow_metric_fields():
    assert V.pct(0.9996, short=True) == "> 99,9 %" and V.pct(0.0001, short=True) == "< 0,1 %" and V.pct(0.5, short=True) == "50,0 %" and V.pct(0.99995, 2, short=True) == "> 99,99 %"


def test_pct_thresholds_of_the_edge_text_are_exclusive():
    assert V.pct(0.9995) == "mehr als 99,9 %" and V.pct(0.9994) == "99,9 %" and V.pct(0.0004) == "weniger als 0,1 %" and V.pct(0.0006) == "0,1 %"


def test_power_of_ten_switches_at_one_hundred():
    assert V.power_of_ten(596.4) == "10^596,40"      # 10 ** 596,4 liefe über den Gleitkommabereich
    assert V.power_of_ten(0.0) == "1,00" and V.power_of_ten(1.0) == "10,0" and V.power_of_ten(1.9) == "79,4"
    assert V.power_of_ten(2.0) == "10^2,00" and V.power_of_ten(5.5477) == "10^5,55"


def test_scientific_notation():
    assert V.sci(0.0) == "0" and V.sci(6.7e-16) == "6,7·10^-16" and V.sci(1.0e-3) == "1,0·10^-3" and V.sci(0.1) == "1,0·10^-1"


# ------------------------------------------------------------------------------------------------------------ Hauptverdikt


def test_main_verdict_success_when_the_risk_dominant_format_wins_and_pays_more():
    kind, text = V.verdict_main(view())            # (5, 2, 1, 2), n = 50, x = 6
    assert kind == "success" and "das neue Format (S)" in text and "mehr als 99,9 %" in text and "x = 6,00, n = 50, p* = 1/4" in text and text.endswith("Es ist auch das risikodominante.")


def test_main_verdict_warning_when_the_payoff_dominant_format_loses():
    kind, text = V.verdict_main(view(a=6, b=4, c=0, d=4))
    assert kind == "warning" and "das alte Format (H) mit 99,8 %" in text and "Das neue Format (S) ist auszahlungsdominant (a = 6 gegen b = 4), gewinnt aber nicht" in text


def test_main_verdict_warning_names_the_other_payoff_dominant_format():
    kind, text = V.verdict_main(view(a=3, b=5, c=4, d=0))
    assert kind == "warning" and "Das alte Format (H) ist auszahlungsdominant (a = 3 gegen b = 5)" in text


def test_main_verdict_success_without_payoff_dominance_when_a_equals_b():
    kind, text = V.verdict_main(view(a=4, b=4, c=3, d=1))
    assert kind == "success" and "auszahlungsdominant" not in text and text.endswith("Es ist auch das risikodominante.")


def test_main_verdict_h_wins_without_the_risk_sentence_when_not_risk_dominant():
    """Ein Koordinationsspiel, in dem H gewinnt und H auch risikodominant ist, aber auszahlungsdominant ist H ebenfalls: Erfolg mit Hinweis."""
    kind, text = V.verdict_main(view(a=2, b=5, c=0, d=1))      # a - d = 1, b - c = 5: p* = 5/6
    assert kind == "success" and "das alte Format (H)" in text and text.endswith("Es ist auch das risikodominante.")


def test_main_verdict_one_side_prevails_but_the_convention_is_not_yet_standing():
    kind, text = V.verdict_main(view(x=4.0))
    assert kind == "info" and text.startswith("**Eine Seite überwiegt") and "99 %" in text


def test_main_verdict_no_convention_in_the_middle_names_the_threshold():
    kind, text = V.verdict_main(view(x=2.0))
    assert kind == "info" and text.startswith("**Keine Konvention:**") and "Mitte 99,1 %" in text and "braucht die Konvention x ≥ 4,79" in text


def test_main_verdict_two_sides_at_p_star_one_half():
    kind, text = V.verdict_main(view(a=6, b=4, c=2, d=4))
    assert kind == "info" and text.startswith("**Keine Konvention, zwei Seiten:**") and "fast alle S 39,4 %, fast alle H 39,4 %" in text and "x ≥" not in text


def test_main_verdict_other_game_classes_are_not_coordination_games():
    kind, text = V.verdict_main(view(a=-1, b=2, c=4, d=0))
    assert kind == "info" and "Kein Koordinationsspiel" in text and "Anti-Koordination" in text and "Summe der Lücken (3)" in text
    assert "dominant" in V.verdict_main(view(a=3, b=1, c=2, d=0))[1] and "entartet" in V.verdict_main(view(a=3, b=2, c=1, d=3))[1]


def test_main_verdict_error_models_point_to_experiment_e4():
    for model, word in ((C.MODELS[1], "gleichmäßigen Fehlern"), (C.MODELS[2], "typabhängigen Fehlern")):
        kind, text = V.verdict_main(view(model=model))
        assert kind == "info" and word in text and "E4" in text


def test_risk_and_payoff_dominance_functions():
    assert V.risk_dominant((5, 2, 1, 2)) == "S" and V.risk_dominant((6, 4, 0, 4)) == "H" and V.risk_dominant((6, 4, 2, 4)) is None and V.risk_dominant((-1, 2, 4, 0)) is None
    assert V.payoff_dominant((6, 4, 0, 4)) == "S" and V.payoff_dominant((3, 5, 4, 0)) == "H" and V.payoff_dominant((4, 4, 3, 1)) is None


def test_threshold_verdict_branches():
    v = view()
    assert V.verdict_threshold_for(v)[0] == "success" and "x* = 4,79" in V.verdict_threshold_for(v)[1] and "erreicht" in V.verdict_threshold_for(v)[1]
    kind, text = V.verdict_threshold_for(view(x=4.5))
    assert kind == "info" and "noch nicht erreicht" in text
    assert V.verdict_threshold_for(view(x=4.75))[0] == "info" and V.verdict_threshold_for(view(x=4.8))[0] == "success"
    exact = view().threshold                               # genau die Schwelle: erreicht (Grenze inklusive)
    assert V.verdict_threshold_for(view(x=exact))[0] == "success" and V.verdict_threshold_for(view(x=exact - 1e-6))[0] == "info"
    assert "nie" in V.verdict_threshold_for(view(a=6, b=4, c=0, d=4))[1]
    assert "nur im Koordinationsspiel" in V.verdict_threshold_for(view(a=-1, b=2, c=4, d=0))[1]


def test_threshold_verdict_when_even_a_huge_sharpness_does_not_reach_99_percent():
    """Im Reglerbereich gibt es dafür kein Spiel (x* bleibt unter 400); der Zweig wird mit einer Ansicht ohne Schwelle geprüft."""
    kind, text = V.verdict_threshold_for(replace(view(), threshold=None))
    assert kind == "info" and "p* = 1/4 und n = 50" in text and "auch bei x = 400 keine 99 %" in text


def test_rates_for_rejects_an_unknown_model():
    with pytest.raises(ValueError):
        V.rates_for(settings(), "Zufall")


# ------------------------------------------------------------------------------------------------------------ Experimente


def _table(rows, ns=(10, 100)):
    return {"level": 0.99, "pstars": ["1/10", "1/4", "2/5"][: len(rows)], "ns": list(ns), "x_star": rows}


def test_verdict_e1_counts_both_monotonicities():
    kind, text = V.verdict_e1(_table([[5.0, 3.0], [6.0, 4.0], [8.0, 5.0]]))
    assert kind == "success" and "in 4 von 4 Paaren" in text and "in 3 von 3" in text
    assert "von 6,00 (n = 10) auf 4,00 (n = 100)" in text


def test_verdict_e1_warns_when_a_monotonicity_fails():
    kind, text = V.verdict_e1(_table([[5.0, 3.0], [6.0, 7.0], [8.0, 5.0]]))
    assert kind == "warning" and "fällt mit n in 2 von 3" in text
    kind, text = V.verdict_e1(_table([[5.0, 3.0], [4.0, 4.0], [8.0, 5.0]]))
    assert kind == "warning" and "steigt mit p* in 3 von 4" in text


def test_verdict_e2_thresholds_and_counts():
    rows = [{"a_minus_b": -91, "deviation": 1e-16}, {"a_minus_b": 4, "deviation": 3e-16}]
    dominated = [{12.0: (0.0, 0.999)}, {12.0: (0.0, 0.995)}, {12.0: (0.0, 0.5)}]
    kind, text = V.verdict_e2(rows, dominated, 50, 12.0)
    assert kind == "success" and "a − b von -91 bis 4" in text and "3,0·10^-16" in text and "In 2 von 3 Spielen" in text and "n = 50" in text
    assert V.verdict_e2([{"a_minus_b": 0, "deviation": 1e-9}], dominated, 50, 12.0)[0] == "warning"


def test_verdict_e2_threshold_is_just_above_99_percent():
    assert "In 1 von 1 Spielen" in V.verdict_e2([{"a_minus_b": 0, "deviation": 0.0}], [{12.0: (0.0, 0.9901)}], 50, 12.0)[1]
    assert "In 0 von 1 Spielen" in V.verdict_e2([{"a_minus_b": 0, "deviation": 0.0}], [{12.0: (0.0, 0.99)}], 50, 12.0)[1]


def test_verdict_e3_uses_the_given_rows():
    mode = [{"gap": 0.0004}, {"gap": 0.0006}]
    middle = [{"n": 50, "x": 3.0, "mid": 0.5}, {"n": 5000, "x": 3.0, "mid": 0.9999}, {"n": 5000, "x": 4.5, "mid": 1e-18}, {"n": 50, "x": 4.5, "mid": 0.1}]
    fixed = [{"pstar": 0.5, "x": 3.9, "count": 1}, {"pstar": 0.5, "x": 4.1, "count": 3}, {"pstar": 0.5, "x": 1.0, "count": 1}, {"pstar": 0.25, "x": 6.0, "count": 1}]
    kind, text = V.verdict_e3(mode, middle, fixed, [])
    assert kind == "info" and "größte Abweichung 0,0006 bei n = 2000" in text and "für x ≤ 3,9 eine Lösung, für x ≥ 4,1 drei" in text and "kritischer Wert x = 4" in text
    assert "Masse in der Mitte bei n = 5000: mehr als 99,9 % bei x = 3,0, weniger als 0,1 % bei x = 4,5." in text


def test_verdict_e4_counts_the_cells_and_names_the_tipping_value():
    count = [{"slope": 10.0005, "count": 10}, {"slope": 24.02, "count": 24}, {"slope": 4.0, "count": 4}]
    finite = [{"uniform": [0.4049, 0.6769, 0.9245], "logit": 0.857}, {"uniform": [0.4045, 0.6768, 0.9245], "logit": 0.487}]
    typed = [{"eps_h": 0.1, "high": 0.629, "low": 0.0}, {"eps_h": 1e-7, "high": 0.0, "low": 1.0}]
    kind, text = V.verdict_e4(count, finite, typed, 1 / 730)
    assert kind == "info" and "in 2 von 3 Zellen" in text and "0,0004" in text and "0,370" in text and "Masse bei S 62,9 %" in text and "Masse bei H 100,0 %" in text and "1,4·10^-3" in text
    assert "Kippschwelle" not in V.verdict_e4(count, finite, typed, None)[1]


def test_verdict_e5_reports_the_totals():
    total = {"games": 160, "logit_clear": 160, "logit_eq_potential": 150, "both_clear": 158, "logit_eq_uniform": 150, "payoff_comparable": 140, "logit_eq_payoff": 70}
    kind, text = V.verdict_e5({"total": total})
    assert kind == "info" and "160 Spiele" in text and "in 150 von 160 eindeutigen Fällen" in text and "dasselbe in 150 von 158" in text and "in 70 von 140" in text and "weicht sie in 70 ab" in text


def test_verdict_e5_dissenter_branches():
    sym = [[9, 2, 1], [2, 8, 6], [1, 6, 7]]
    same = V.verdict_e5_dissenter(sym, 0, 0)
    none = V.verdict_e5_dissenter(sym, 0, None)
    assert same[0] == "info" and none[0] == "info"
    kind, text = V.verdict_e5_dissenter(sym, 0, 1)
    assert kind == "warning" and "Logit wählt Format 1, gleichmäßige Fehler wählen Format 2" in text and "Format 1: Der Logit folgt dem Potential, die gleichmäßigen Fehler weichen vom Potential ab" in text
    kind, text = V.verdict_e5_dissenter(sym, 1, 0)
    assert "Der Logit weicht vom Potential ab, die gleichmäßigen Fehler folgen dem Potential" in text


def test_verdict_e6_live_branches():
    assert "nur im Koordinationsspiel" in V.verdict_e6_live(view(a=-1, b=2, c=4, d=0))[1]
    assert "nur im Koordinationsspiel" in V.verdict_e6_live(view(a=2, b=5, c=1, d=3))[1]              # H dominiert: keine Barriere
    text = V.verdict_e6_live(view(n=10, x=2.0, sim_steps=100_000))[1]
    assert "Barriere k* = 3 von n = 10" in text and "Aus „alle H“ dauert es im Mittel 0,91" in text and "aus „alle S“ 20,5" in text and "es reicht" in text and "es reicht nicht" not in text
    assert "es reicht nicht" in V.verdict_e6_live(view(n=20, x=8.0, sim_steps=10_000))[1]
    text_big = V.verdict_e6_live(view(n=1000, x=40.0, sim_steps=3_000_000))[1]                       # Zeiten jenseits von 10^308: kein Überlauf
    assert "es reicht nicht" in text_big and "aus „alle S“ 10^4642,35" in text_big and "im Mittel 10^308,16" in text_big


def test_verdict_e6_table_counts_cells_within_three_standard_errors():
    rows = [{"sim": 10.0, "formula": 10.5, "se": 0.3, "capped": 0}, {"sim": 10.0, "formula": 11.0, "se": 0.3, "capped": 1}, {"sim": None, "formula": 5.0, "se": None, "capped": 2}]
    text = V.verdict_e6_table(rows)[1]
    assert "1 von 2 Zellen" in text and "abgebrochene Läufe: 3" in text
    assert "2 von 2" in V.verdict_e6_table([{"sim": 10.0, "formula": 10.89, "se": 0.3, "capped": 0}, {"sim": 10.0, "formula": 10.0, "se": 0.1, "capped": 0}])[1]


def test_verdict_simulation_branches():
    assert V.verdict_simulation(0.01, None, 100.0)[0] == "info"
    assert V.verdict_simulation(0.01, math.log10(50), 100.0)[0] == "success"
    assert V.verdict_simulation(0.05, math.log10(50), 100.0)[0] == "info"
    kind, text = V.verdict_simulation(0.3, 5.0, 100.0)
    assert kind == "warning" and "reicht nicht" in text and "10^5,00" in text and "0,300" in text
    assert V.verdict_simulation(0.01, 2.0, 100.0)[0] == "warning"      # gleich lang reicht nicht (strikt größer)
    assert V.verdict_simulation(0.01, 600.5, 100.0)[0] == "warning"    # n = 1 000: die Zeit ist größer als jede Gleitkommazahl, kein Überlauf


# ------------------------------------------------------------------------------------------------------------ Grenzen der Schwellen (zweite Mutationsrunde)


def test_power_of_ten_digits_change_at_ten():
    assert V.power_of_ten(math.log10(9.5)) == "9,50" and V.power_of_ten(math.log10(10.0)) == "10,0" and V.power_of_ten(math.log10(99.0)) == "99,0"


def test_verdict_e1_requires_strict_monotonicity_in_both_directions():
    equal_p = _table([[5.0, 3.0], [5.0, 4.0], [8.0, 5.0]])           # erste Spalte: 5,0 = 5,0 steigt nicht
    assert V.verdict_e1(equal_p)[0] == "warning" and "steigt mit p* in 3 von 4" in V.verdict_e1(equal_p)[1]
    equal_n = _table([[5.0, 5.0], [6.0, 4.0], [8.0, 5.0]])           # erste Zeile: 5,0 = 5,0 fällt nicht
    assert V.verdict_e1(equal_n)[0] == "warning" and "fällt mit n in 2 von 3" in V.verdict_e1(equal_n)[1]


def test_verdict_e1_needs_both_directions_to_hold():
    only_p = _table([[5.0, 3.0], [6.0, 7.0], [8.0, 9.0]])            # steigt mit p* in 4 von 4, fällt mit n nur in 1 von 3
    assert V.verdict_e1(only_p)[0] == "warning"
    only_n = _table([[5.0, 3.0], [4.0, 2.0], [8.0, 1.0]])            # fällt mit n in 3 von 3, steigt mit p* nur in 2 von 4
    assert V.verdict_e1(only_n)[0] == "warning"


def test_verdict_e3_takes_the_smallest_x_with_three_solutions_and_the_largest_with_one():
    mode = [{"gap": 0.0001}]
    middle = [{"n": 50, "x": 3.0, "mid": 0.5}, {"n": 50, "x": 4.5, "mid": 0.1}]
    fixed = [{"pstar": 0.5, "x": x, "count": 1 if x < 4 else 3} for x in (1.0, 2.0, 3.9, 4.1, 5.0, 8.0)]
    text = V.verdict_e3(mode, middle, fixed, [])[1]
    assert "für x ≤ 3,9 eine Lösung, für x ≥ 4,1 drei" in text


def test_verdict_e4_counts_slopes_strictly_and_uses_every_column_of_the_finite_table():
    typed = [{"eps_h": 0.1, "high": 0.6, "low": 0.0}, {"eps_h": 1e-7, "high": 0.0, "low": 1.0}]
    finite = [{"uniform": [0.40, 0.60, 0.900], "logit": 0.9}, {"uniform": [0.40, 0.60, 0.901], "logit": 0.5}]
    text = V.verdict_e4([{"slope": 0.01, "count": 0}, {"slope": 4.0, "count": 4}], finite, typed, None)[1]
    assert "in 1 von 2 Zellen" in text and "größter Unterschied der Masse 0,0010" in text


def test_verdict_e5_dissenter_looks_at_all_three_diagonal_entries():
    sym = [[7, 0, 0], [0, 8, 0], [0, 0, 9]]
    text = V.verdict_e5_dissenter(sym, 2, 0)[1]
    assert "hat Format 3" in text and "Der Logit folgt dem Potential, die gleichmäßigen Fehler weichen vom Potential ab" in text


def test_verdict_e6_live_for_a_degenerate_game_with_a_barrier_at_state_one():
    """b = c, a > d: Delta_0 = 0, Delta_1 > 0, also Barriere k* = 1, aber kein Koordinationsspiel."""
    v = view(a=3, b=2, c=2, d=0)
    assert v.klass == "entartet" and v.barrier == 1
    assert "nur im Koordinationsspiel" in V.verdict_e6_live(v)[1]


def test_verdict_e6_table_includes_exactly_three_standard_errors():
    rows = [{"sim": 13.0, "formula": 10.0, "se": 1.0, "capped": 0}, {"sim": 13.5, "formula": 10.0, "se": 1.0, "capped": 0}]
    assert "1 von 2 Zellen" in V.verdict_e6_table(rows)[1]


def test_main_verdict_thresholds_at_exactly_the_limits():
    base = view()
    from ggw_chain import Masses
    assert V.verdict_main(replace(base, masses=Masses(0.99, 0.0, 0.01)))[1].startswith("**Eine Konvention steht:**")
    assert V.verdict_main(replace(base, masses=Masses(0.98, 0.0, 0.02)))[1].startswith("**Eine Seite überwiegt")
    assert V.verdict_main(replace(base, masses=Masses(0.5, 0.0, 0.5)))[1].startswith("**Eine Seite überwiegt")
    assert V.verdict_main(replace(base, masses=Masses(0.4, 0.1, 0.5)))[1].startswith("**Keine Konvention:**")
    assert V.verdict_main(replace(base, masses=Masses(0.4, 0.1, 0.49)))[1].startswith("**Keine Konvention, zwei Seiten:**")
    assert V.verdict_main(replace(base, masses=Masses(0.0, 0.99, 0.01)))[1].startswith("**Eine Konvention steht:** das alte Format (H)")


def test_threshold_verdict_at_p_star_exactly_one_half_says_never():
    kind, text = V.verdict_threshold_for(view(a=6, b=4, c=2, d=4))
    assert kind == "info" and "p* = 1/2" in text and "nie" in text
