"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an seine
Grenzen und prüft, dass kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key=)
und `StreamlitAPIException` bei Reglern mit berechneten Grenzen (min == max). Die ▶️-Knöpfe laufen einzeln (die Schleife lässt sie aus)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import ggw_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # ▶: Knöpfe mit Rechenlast, einzeln getestet
TAB_LABELS = ["1 Schwellenfläche", "2 Gleiches Spiel, anderes Gefälle", "3 Große Gruppen", "4 Fehlermodelle", "5 Drei Formate", "6 Kann man es sehen?"]


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([w.value for w in at.warning] + [s.value for s in at.success] + [i.value for i in at.info])


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run()
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run()
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("model", C.MODELS)
def test_all_sliders_at_their_minima_and_all_at_their_maxima_do_not_raise(model):
    """Eckfall-Matrix (DEMO-PLAYBOOK §1d): alle Regler zugleich am Minimum, dann alle am Maximum (n = 1 000, x = 40, ε bis 10^-8, ...), je Fehlermodell."""
    for edge in ("min", "max"):
        at = _fresh_app()
        for slider in at.slider:
            value = slider.min if edge == "min" else slider.max
            slider.set_value(int(value) if isinstance(slider.value, int) else value)
        at.selectbox(key="model_select").set_value(model)
        at.select_slider(key="steps_select").set_value(C.SIM_STEPS[0] if edge == "min" else C.SIM_STEPS[-1])
        at.run()
        assert not at.exception, (edge, model, [str(e) for e in at.exception])


def test_every_game_class_renders():
    for game in ((5, 2, 1, 2), (-1, 2, 4, 0), (3, 1, 2, 0), (3, 2, 1, 3), (2, 2, 2, 2)):
        at = _fresh_app()
        for key, value in zip(("a_slider", "b_slider", "c_slider", "d_slider"), game):
            at.slider(key=key).set_value(value)
        at.run()
        assert not at.exception, (game, [str(e) for e in at.exception])


def test_the_three_play_buttons_work_individually_and_show_their_tables():
    """Jeder ▶️-Knopf einzeln: erst nach dem Klick erscheint die Tabelle samt Verdikt (die Smoke-Schleife oben lässt sie aus)."""
    at = _fresh_app()
    assert not any("Logit = Potential" in df.value.columns.tolist() for df in at.dataframe), "vor dem Klick darf die Tabelle nicht stehen"
    at.button(key="e5_button").click().run()
    assert not at.exception and at.session_state["e5_done"] is True
    assert any("Logit = Potential" in df.value.columns.tolist() for df in at.dataframe) and "Spiele mit drei Formaten" in _texts(at)

    at = _fresh_app()
    at.slider(key="runs_slider").set_value(10)
    at.run()
    assert not any("Simulation (Mittel über Ströme)" in df.value.columns.tolist() for df in at.dataframe)
    at.button(key="e6_button").click().run()
    assert not at.exception and at.session_state["e6_done"] is True
    assert any("Simulation (Mittel über Ströme)" in df.value.columns.tolist() for df in at.dataframe) and "Formel gegen Einzelspieler-Simulation" in _texts(at)

    at = _fresh_app()
    assert "Gesamtvariationsabstand" not in _texts(at)
    at.button(key="sim_button").click().run()
    assert not at.exception and at.session_state["sim_signature"] is not None
    assert "Gesamtvariationsabstand" in _texts(at)


def test_simulation_is_hidden_when_the_settings_change_and_for_other_models():
    at = _fresh_app()
    at.button(key="sim_button").click().run()
    assert "Gesamtvariationsabstand" in _texts(at)
    at.slider(key="n_slider").set_value(40).run()
    assert not at.exception and "Gesamtvariationsabstand" not in _texts(at)
    at = _fresh_app()
    at.selectbox(key="model_select").set_value(C.MODELS[1]).run()
    assert not at.exception and not [b for b in at.button if b.key == "sim_button"]
    assert any("Wahlregel „Logit“" in i.value for i in at.info)


def test_simulation_at_the_worst_case_budget_and_group_size():
    """Worst-Case-Test am Reglermaximum (n = 1 000, 3·10^6 Aktualisierungen): kein Hänger, das Ergebnis erscheint."""
    at = _fresh_app()
    at.slider(key="n_slider").set_value(C.N_MAX)
    at.select_slider(key="steps_select").set_value(C.SIM_STEPS[-1])
    at.run()
    at.button(key="sim_button").click().run()
    assert not at.exception and "Gesamtvariationsabstand" in _texts(at)


def test_play_button_result_survives_the_next_rerun():
    at = _fresh_app()
    at.button(key="e5_button").click().run()
    at.slider(key="n_slider").set_value(60).run()
    assert not at.exception and any("Logit = Potential" in df.value.columns.tolist() for df in at.dataframe)


def test_model_selectbox_every_option_renders_its_own_verdict():
    for model, word in zip(C.MODELS, ("Konvention", "gleichmäßigen Fehlern", "typabhängigen Fehlern")):
        at = _fresh_app()
        at.selectbox(key="model_select").set_value(model).run()
        assert not at.exception, (model, [str(e) for e in at.exception])
        assert word in _texts(at), model


def test_log_toggle_renders():
    at = _fresh_app()
    at.toggle(key="log_toggle").set_value(True).run()
    assert not at.exception


def test_default_state_shows_the_six_tabs_and_the_main_verdict():
    at = _fresh_app()
    assert [t.label for t in at.tabs] == TAB_LABELS
    assert "**Eine Konvention steht:** das neue Format (S)" in " ".join(s.value for s in at.success)


def test_permalink_sets_the_game_and_is_limited_to_the_bounds():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["a"] = "99"
    at.query_params["model"] = "Typabhängige Fehler"
    at.query_params["n"] = "0"
    at.query_params["x"] = "1000"
    at.query_params["steps"] = "300000"
    at.run()
    assert not at.exception
    assert at.slider(key="a_slider").value == 10 and at.selectbox(key="model_select").value == "Typabhängige Fehler" and at.slider(key="n_slider").value == 4
    assert at.slider(key="x_slider").value == 40.0 and at.select_slider(key="steps_select").value == 300000


def test_permalink_ignores_unknown_values_and_garbage():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["model"] = "Zufall"
    at.query_params["b"] = "viel"
    at.query_params["steps"] = "12345"
    at.query_params["x"] = "nan"
    at.run()
    assert not at.exception
    assert at.selectbox(key="model_select").value == "Logit" and at.slider(key="b_slider").value == 2 and at.select_slider(key="steps_select").value == C.DEFAULT_SIM_STEPS
    assert at.slider(key="x_slider").value == C.DEFAULT_X


def test_pdf_button_creates_a_pdf_and_offers_the_download():
    at = _fresh_app()
    at.button(key="pdf_button").click().run()
    assert not at.exception
    assert at.session_state["pdf_bytes"][:5] == b"%PDF-"


def test_every_preset_shows_its_main_verdict_of_the_expected_kind():
    expected = {"Neues Palettenformat (5, 2, 1, 2)": ("success", "Eine Konvention steht"), "Bündeln am Hub (6, 4, 0, 4)": ("warning", "gewinnt aber nicht"),
                "Gleichauf (6, 4, 2, 4)": ("info", "zwei Seiten"), "Gleiches Spiel, anderes Gefälle (3, 5, 4, 0)": ("warning", "gewinnt aber nicht"),
                "Zu wenig Schärfe (x = 2)": ("info", "Keine Konvention")}
    for name, (kind, word) in expected.items():
        at = _fresh_app()
        at.button(key=f"preset_{name}").click().run()
        assert not at.exception, name
        texts = {"warning": [w.value for w in at.warning], "success": [s.value for s in at.success], "info": [i.value for i in at.info]}[kind]
        assert any(word in t for t in texts), (name, kind)
