"""Presets: Vollständigkeit, gültige Werte, erstes Preset = Standardeinstellungen, gerechnete Aussagen der Hilfetexte."""

from fractions import Fraction as F

import ggw_chain as CH
import ggw_constants as C
import ggw_evaluation as V
import ggw_game as G
import ggw_presets as P


def _view(name):
    return V.analyse_main(V.Settings(**C.PRESETS[name]))


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.choices is not None:
                assert preset[key] in spec.choices, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert V.Settings(**C.PRESETS[first]) == V.Settings()
    assert first.startswith("Neues Palettenformat")


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_no_two_presets_are_identical():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_permalink_params_are_unique():
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_help_text_names_the_payoffs_and_p_star_of_each_game():
    """Die Spiele der Presets stehen im Namen; p* und Dominanz der Hilfetexte gegen die Rechnung."""
    g = {name: (p["a"], p["b"], p["c"], p["d"]) for name, p in C.PRESETS.items()}
    for name, game in g.items():
        if "(" in name and "x = 2" not in name:
            assert name.endswith(f"({game[0]}, {game[1]}, {game[2]}, {game[3]})"), name
    assert G.p_star(g["Neues Palettenformat (5, 2, 1, 2)"]) == F(1, 4) and V.risk_dominant(g["Neues Palettenformat (5, 2, 1, 2)"]) == "S" and V.payoff_dominant(g["Neues Palettenformat (5, 2, 1, 2)"]) == "S"
    assert G.p_star(g["Bündeln am Hub (6, 4, 0, 4)"]) == F(2, 3) and V.risk_dominant(g["Bündeln am Hub (6, 4, 0, 4)"]) == "H" and V.payoff_dominant(g["Bündeln am Hub (6, 4, 0, 4)"]) == "S"
    assert G.p_star(g["Gleichauf (6, 4, 2, 4)"]) == F(1, 2) and V.payoff_dominant(g["Gleichauf (6, 4, 2, 4)"]) == "S"
    assert G.p_star(g["Gleiches Spiel, anderes Gefälle (3, 5, 4, 0)"]) == F(1, 4) and V.payoff_dominant(g["Gleiches Spiel, anderes Gefälle (3, 5, 4, 0)"]) == "H"


def test_default_game_help_statement_convention_stands_at_x_6_n_50():
    v = _view("Neues Palettenformat (5, 2, 1, 2)")
    assert v.masses.high >= 0.99 and (v.settings.x, v.settings.n) == (6.0, 50)


def test_hub_preset_help_statement_old_format_stands():
    v = _view("Bündeln am Hub (6, 4, 0, 4)")
    assert v.masses.low >= 0.99 and (v.settings.x, v.settings.n) == (6.0, 50)


def test_equal_risk_preset_has_mass_on_both_sides_and_the_critical_value_is_four():
    v = _view("Gleichauf (6, 4, 2, 4)")
    assert v.masses.high > 0.3 and v.masses.low > 0.3 and abs(v.masses.high - v.masses.low) < 1e-12 and v.settings.x > 4


def test_different_gradient_preset_has_the_same_distribution_as_the_first():
    a, b = _view("Neues Palettenformat (5, 2, 1, 2)"), _view("Gleiches Spiel, anderes Gefälle (3, 5, 4, 0)")
    assert abs(a.pi - b.pi).max() < 1e-14 and G.gaps(a.settings.game) == G.gaps(b.settings.game)
    assert a.settings.a - a.settings.b == 3 and b.settings.a - b.settings.b == -2


def test_low_sharpness_preset_has_almost_all_mass_in_the_middle():
    v = _view("Zu wenig Schärfe (x = 2)")
    assert v.masses.mid > 0.99 and v.settings.x == 2.0 and v.settings.x < v.threshold
    other = {k: v2 for k, v2 in C.PRESETS["Zu wenig Schärfe (x = 2)"].items() if k != "x"}
    assert other == {k: v2 for k, v2 in C.PRESETS[next(iter(C.PRESETS))].items() if k != "x"}


def test_threshold_of_the_default_game_is_below_the_default_sharpness():
    v = _view(next(iter(C.PRESETS)))
    assert v.threshold < C.DEFAULT_X and CH.mass_high_for(C.DEFAULT_N, F(1, 4), C.DEFAULT_X) >= 0.99
