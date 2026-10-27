"""Welche Konvention überlebt? Auswahl durch Rauschen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zweites Stück der Reihe Koordinationsspiele. Viele Betriebe entscheiden sich immer wieder neu und liegen gelegentlich daneben; die Demo rechnet exakt, auf welche Konvention das System
auf Dauer läuft, wie scharf die Wahl sein muss, damit überhaupt eine entsteht, und wann eine Simulation die Antwort nicht mehr erreicht.

Lauffähig mit: streamlit run app.py
"""

from fractions import Fraction

import streamlit as st

import ggw_chain as CH
import ggw_constants as C
import ggw_evaluation as V
import ggw_experiments as X
import ggw_game as G
import ggw_limit as LI
import ggw_scenario as S
import ggw_sim as SI
import ggw_three as T
import ggw_visualization as Z
from ggw_layout import dunkles_design, kopf_css
from ggw_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)

st.set_page_config(page_title="Welche Konvention überlebt? – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _main(a, b, c, d, n, x, model, eps_exp, eps_s_exp, eps_h_exp):
    return V.analyse_main(V.Settings(a, b, c, d, n, x, model, eps_exp, eps_s_exp, eps_h_exp))


@st.cache_data(show_spinner=False)
def _surface(n):
    pstars = [i / 20 for i in range(1, 20)]
    xs = [0.5 * i for i in range(1, 41)]
    return pstars, xs, X.e1_surface([Fraction(i, 20) for i in range(1, 20)], xs, n)


@st.cache_data(show_spinner=False)
def _flip():
    return X.e1_flip_in_p_star()


@st.cache_data(show_spinner=False)
def _e1_mass_rows():
    return X.e1_mass_rows(Fraction(1, 4))


@st.cache_data(show_spinner=False)
def _e2():
    return X.e2_invariance(), X.e2_payoff_dominated()


@st.cache_data(show_spinner=False)
def _e2_overlay():
    n = 30
    games = C.INVARIANCE_GAMES
    return [CH.stationary(CH.rates_logit(n, g, 1.0)) for g in games]


@st.cache_data(show_spinner=False)
def _e3():
    return X.e3_mode_rows(), X.e3_middle_rows(), X.e3_fixed_point_rows(), X.e3_second_branch_rows()


@st.cache_data(show_spinner=False)
def _e4():
    return X.e4_count_rows(), X.e4_finite_rows(), X.e4_typed_rows(), X.e4_tipping_eps_h(40, Fraction(1, 4), 0.1)


@st.cache_data(show_spinner=False)
def _e5():
    return X.e5_three_formats()


@st.cache_data(show_spinner=False)
def _e5_examples():
    out = []
    for seed, index in C.DISSENTER_EXAMPLES:
        g = S.draw_three_format_games(seed, index + 1)[index]
        out.append({"seed": seed, "index": index, **X.e5_example(g)})
    return out


@st.cache_data(show_spinner=False)
def _e6_rows():
    return X.e6_time_rows()


@st.cache_data(show_spinner=False)
def _e6_check(runs):
    return X.e6_simulation_check(runs=runs, cap_steps=C.HITTING_CAP)


@st.cache_data(show_spinner=False)
def _simulate(a, b, c, d, n, beta, steps, start_k, seed):
    burn = int(steps * C.SIM_BURN_FRACTION)
    return SI.simulate_players(n, (a, b, c, d), beta, steps, burn, start_k, seed)


def show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


def table(rows):
    st.dataframe(rows, hide_index=True, width="stretch")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in ggw_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🎲 Welche Konvention überlebt? Auswahl durch Rauschen")
st.markdown(
    """
**Zweites Stück der Reihe Koordinationsspiele.** Das erste Stück (`koordinationsspiel-demo`) zeigt die Gleichgewichte eines Koordinationsspiels; hier geht es um die Frage danach:
Viele Betriebe wählen ein **Palettenformat**, das neue (S) oder das alte (H), und es lohnt sich nur, dasselbe zu wählen wie die anderen. Jeder entscheidet sich immer wieder neu und liegt gelegentlich daneben.
**Auf welche Konvention läuft das System auf Dauer, und wie scharf muss die Wahl sein, damit überhaupt eine entsteht?**

**Eigennutz und die Lernregel (Logit: die bessere Wahl ist wahrscheinlicher, nie sicher) sind Modellannahmen**, keine Aussagen über Menschen oder Terminals. Die Auszahlungen (Zeitersparnis je Sendung in Minuten) sind gewählt, nicht gemessen.

Weiter unten: sechs Tabs mit je einem **🔬 Experiment**, die Abschnitte **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Wahl mit Schärfe", expanded=True):
    st.markdown(
        r"""
1. **Das Spiel hängt nur an zwei Lücken.** Ein Betrieb vergleicht, was ihn S gegenüber H kostet oder bringt, wenn der andere S wählt ($a - d$) bzw. H wählt ($b - c$). Die Skala ist $\lambda = (a-d)+(b-c)$, die Schwelle $p^* = (b-c)/\lambda$: Ab diesem Anteil der S-Betriebe unter den anderen ist S besser.

$$\Delta_j = \frac{(a-d)\,j - (b-c)\,(n-1-j)}{n-1} = \lambda\left(\frac{j}{n-1} - p^*\right)$$

2. **Logit: gut ist wahrscheinlicher, nie sicher.** Ein gezogener Betrieb wählt S mit der Wahrscheinlichkeit

$$\Pr(S) = \frac{1}{1 + e^{-\beta\,\Delta}}$$

Die **Schärfe in Einheiten der Auszahlungsskala** ist $x = \beta\lambda$. Der Regler **x** stellt sie direkt ein, $\beta = x/\lambda$ rechnet die App aus. Zwei Spiele mit demselben $(\lambda, p^*)$ und derselben Schärfe $x$ geben dieselbe Verteilung, auch wenn ihre Auszahlungen verschieden sind (Experiment E2).

3. **Die Verteilung, auf die das System läuft.** Gezählt wird die Zahl $k$ der S-Betriebe. Die Kette hat eine stationäre Verteilung $\pi$, die die App exakt rechnet (nicht simuliert). Sie ist das **Gibbs-Maß des Potentials mal einem Binomialfaktor**:

$$\pi(k) \propto \binom{n}{k}\, e^{\beta\,\Phi(k)}$$

Der Binomialfaktor zählt die Konfigurationen, die zu $k$ gehören. Er erklärt, warum bei kleiner Schärfe **keine** Konvention entsteht: Eine gemischte Konfiguration ist weniger wahrscheinlich als „alle gleich“, aber es gibt sehr viel mehr gemischte.
        """
    )

with st.expander("Das Potential"):
    st.markdown(
        r"""
Jedes symmetrische 2×2-Koordinationsspiel ist ein **Potentialspiel**: Mit dem Paarpotential

$$P(S,S) = a - d, \qquad P(H,H) = b - c, \qquad P(S,H) = 0$$

ist der Auszahlungsgewinn eines Wechsels gleich der Potentialdifferenz. Summiert über alle Paare, wenn $k$ von $n$ Betrieben S nutzen:

$$\Phi(k) = \frac{(a-d)\binom{k}{2} + (b-c)\binom{n-k}{2}}{n-1}$$

Logit-Lernen ist dann das Gibbs-Verfahren zum Potential. Auf **Konfigurationen** ist die stationäre Verteilung $\propto e^{\beta\Phi}$, auf dem **Zustand** $k$ kommt der Binomialfaktor dazu (siehe oben). Das Potential ist beim **risikodominanten** Gleichgewicht am größten: Es vergleicht $a - d$ mit $b - c$, nicht $a$ mit $b$.
        """
    )

with st.expander("Erklärtabelle: die Begriffe"):
    st.markdown(
        """
| Begriff | Bedeutung | Wo gezeigt |
|---|---|---|
| **Auszahlungsdominanz** | Das Gleichgewicht, in dem beide am meisten bekommen (a gegen b). | Hauptansicht, E2 |
| **Risikodominanz** | Das Gleichgewicht mit der größeren Lücke: Weicht der andere ab, verliere ich dort weniger (a − d gegen b − c). Gleichbedeutend: p* < 1/2 heißt S risikodominant. | Hauptansicht, E1, E2 |
| **Stationäre Verteilung** | Der Anteil der Zeit, den die Gruppe auf Dauer in jedem Zustand verbringt, wenn sie sich immer wieder neu entscheidet. Hier exakt gerechnet. | Hauptansicht |
| **Konvention steht** | Mindestens 99 % der Masse liegt bei „fast alle S“ (mindestens 90 % der Betriebe nutzen S) oder bei „fast alle H“ (höchstens 10 %). | Hauptansicht, E1 |
| **Schärfe x = β·λ** | Wie sicher die bessere Wahl getroffen wird, in Einheiten der Auszahlungsskala. | Regler x, alle Tabs |
| **Barriere k\\*** | Erster Zustand, in dem S die beste Antwort ist; die Zeit bis dorthin ist die Austrittszeit. | E6 |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🎮 Das Spiel")
    st.caption("Zeitersparnis je Sendung in Minuten. S = neues Format, H = altes Format.")
    a = st.slider("a: beide S", *bounds("a_slider"), key="a_slider", help="Meine Auszahlung, wenn beide das neue Format nutzen.")
    b = st.slider("b: beide H", *bounds("b_slider"), key="b_slider", help="Meine Auszahlung, wenn beide das alte Format nutzen.")
    c = st.slider("c: ich S, der andere H", *bounds("c_slider"), key="c_slider", help="Wer das neue Format allein nutzt, bekommt oft wenig.")
    d = st.slider("d: ich H, der andere S", *bounds("d_slider"), key="d_slider", help="Meine Auszahlung, wenn ich das alte Format nutze und der andere das neue.")
    st.header("👥 Gruppe und Schärfe")
    n = st.slider("Betriebe n", *bounds("n_slider"), key="n_slider", help="Wie viele Betriebe sich vollständig durchmischt begegnen (jeder trifft jeden gleich oft).")
    x = st.slider("Schärfe x = β·λ", *bounds("x_slider"), key="x_slider", step=0.25,
                  help="Wie sicher die bessere Wahl getroffen wird, in Einheiten der Auszahlungsskala λ = (a − d) + (b − c). β = x/λ rechnet die App aus. Nur x zählt, nicht β und λ einzeln.")
    st.header("⚠️ Fehlermodell")
    model = st.selectbox("Wahlregel der Hauptansicht", C.MODELS, key="model_select",
                         help="Logit: gute Wahl wahrscheinlicher, nie sicher. Gleichmäßige Fehler: beste Antwort, mit Wahrscheinlichkeit ε zufällig. Typabhängig: je nach bester Antwort eigene Fehlerrate.")
    eps_exp = st.slider("ε = 10^(−e), gleichmäßige Fehler: e", *bounds("eps_slider"), key="eps_slider", step=0.5, help="Fehlerrate ε = 10^(−e). Wirkt auf die gleichmäßigen Fehler (Hauptansicht, wenn gewählt, und Experiment E4).")
    eps_s_exp = st.slider("ε_S = 10^(−e): S ist beste Antwort, H wird gewählt", *bounds("eps_s_slider"), key="eps_s_slider", step=0.5, help="Fehlerrate ε_S = 10^(−e) der typabhängigen Fehler. Wirkt in Experiment E4 und in der Hauptansicht, wenn dort „Typabhängige Fehler“ gewählt ist.")
    eps_h_exp = st.slider("ε_H = 10^(−e): H ist beste Antwort, S wird gewählt", *bounds("eps_h_slider"), key="eps_h_slider", step=0.5, help="Fehlerrate ε_H = 10^(−e) der typabhängigen Fehler. Wirkt in Experiment E4 und in der Hauptansicht, wenn dort „Typabhängige Fehler“ gewählt ist.")
    st.header("🖥️ Simulation (Experiment E6)")
    start = st.slider("Anfangsanteil der S-Betriebe (%)", *bounds("start_slider"), key="start_slider", help="So viele Betriebe starten in der Simulation mit dem neuen Format (auf ganze Betriebe gerundet).")
    sim_steps = st.select_slider("Budget: Aktualisierungen", options=C.SIM_STEPS, key="steps_select", help="So viele Einzelaktualisierungen darf die Simulation rechnen (3 Millionen sind die Obergrenze).")
    runs = st.slider("Läufe der Austrittszeit-Simulation", *bounds("runs_slider"), key="runs_slider", help="Wiederholungen je Zufallsstrom bei „Formel gegen Simulation der Austrittszeit“ (vier Ströme).")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"a_slider": int(a), "b_slider": int(b), "c_slider": int(c), "d_slider": int(d), "n_slider": int(n), "x_slider": float(x), "model_select": model, "eps_slider": float(eps_exp),
                   "eps_s_slider": float(eps_s_exp), "eps_h_slider": float(eps_h_exp), "start_slider": int(start), "steps_select": int(sim_steps), "runs_slider": int(runs), "seed_input": int(seed)})

settings = V.Settings(int(a), int(b), int(c), int(d), int(n), float(x), model, float(eps_exp), float(eps_s_exp), float(eps_h_exp), int(start), int(sim_steps), int(runs), int(seed))
view = _main(settings.a, settings.b, settings.c, settings.d, settings.n, settings.x, settings.model, settings.eps_exp, settings.eps_s_exp, settings.eps_h_exp)
verdicts = []   # Verdikte für den PDF-Export

# --- Hauptansicht --------------------------------------------------------------------------------------------------------------------------
st.markdown("## 📊 Auf welche Konvention läuft die Gruppe?")
m1, m2, m3, m4 = st.columns([1.5, 1, 1, 1.2])
m1.metric("Klasse", view.klass)
m2.metric("p*", "–" if view.p_star is None else f"{V.frac(view.p_star)} ({V.de(float(view.p_star), 3)})")
m3.metric("Skala λ", str(view.scale))
m4.metric("β = x/λ (je Minute)", V.de(view.beta, 3))
log_scale = st.toggle("Logarithmusachse (zeigt auch winzige Massen)", key="log_toggle", value=False)
st.plotly_chart(Z.build_distribution(view.pi, log_scale), width="stretch", key="distribution_chart")
st.caption(f"Stationäre Verteilung der Zahl der S-Betriebe bei n = {settings.n}, Wahlregel {settings.model}. Grün: fast alle S (mindestens 90 % der Betriebe), orange: fast alle H (höchstens 10 %), grau: Mitte.")
mt1, mt2, mt3, mt4 = st.columns(4)
mt1.metric("Masse „fast alle S“", V.pct(view.masses.high, short=True))
mt2.metric("Masse „fast alle H“", V.pct(view.masses.low, short=True))
mt3.metric("Masse in der Mitte", V.pct(view.masses.mid, short=True))
mt4.metric("Erwarteter Anteil S", V.pct(view.mean, short=True))
st.markdown("#### ❓ Kernfrage: Steht eine Konvention, und ist es die risikodominante?")
v_main = V.verdict_main(view)
show(v_main)
verdicts.append(v_main)
v_thr = V.verdict_threshold_for(view)
show(v_thr)
verdicts.append(v_thr)
if view.klass == G.KOORDINATION:
    with st.expander("Potential und Binomialfaktor getrennt (Logit)"):
        st.plotly_chart(Z.build_energy_entropy(settings.n, view.beta, view.potential, view.log_binom), width="stretch", key="energy_chart")
        st.caption("Blau: wie wahrscheinlich eine einzelne Konfiguration ist (Potential, Logit-Schärfe β), gestrichelt: wie viele Konfigurationen zu k gehören, rot: die Summe, also die Masse des Zustands. "
                   "Bei kleiner Schärfe gewinnt die gestrichelte Kurve, bei großer die blaue.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["1 Schwellenfläche", "2 Gleiches Spiel, anderes Gefälle", "3 Große Gruppen", "4 Fehlermodelle", "5 Drei Formate", "6 Kann man es sehen?"])

# --- Tab 1 ---------------------------------------------------------------------------------------------------------------------------------
with tab1:
    st.subheader("🔬 Experiment E1: Wie scharf muss die Wahl sein?")
    st.caption("Schwelle x*: die kleinste Schärfe x = β·λ, bei der mindestens 99 % der Masse bei „fast alle S“ liegen (Bisektion, vorgerechnet in data/threshold_table.json; „nie“ steht für: bis x = 400 nicht erreicht, geprüft bei p* = 1/2, 2/3 und 3/4).")
    tbl = X.load_threshold_table()
    table([{"p*": p, **{f"n = {nn}": ("nie" if val is None else V.de(val, 2)) for nn, val in zip(tbl["ns"], row)}} for p, row in zip(tbl["pstars"], tbl["x_star"])])
    v1 = V.verdict_e1(tbl)
    show(v1)
    verdicts.append(v1)
    st.plotly_chart(Z.build_threshold_curves(tbl), width="stretch", key="e1_curves")
    st.markdown("**Die Fläche für Ihre Gruppengröße.** Masse „fast alle S“ über Schärfe und Schwelle p*, n wie im Regler; die weiße Linie begrenzt 99 %.")
    pstars, xs, surf = _surface(settings.n)
    mark = (settings.x, float(view.p_star)) if view.p_star is not None else None
    st.plotly_chart(Z.build_surface(pstars, xs, surf, settings.n, mark), width="stretch", key="e1_surface")
    st.caption("Die Raute ist das gewählte Spiel. Bei p* > 1/2 ist H das risikodominante Format; die Masse bei „fast alle H“ ist dort die gespiegelte Fläche (p* ↔ 1 − p*).")
    st.markdown("**Der Umschlag in p\\* ist schroff** (n = 100, x = 8):")
    table([{"p*": V.frac(r["pstar"]), "fast alle S": V.pct(r["high"], 2), "fast alle H": V.pct(r["low"], 2), "Mitte": V.pct(r["mid"], 2)} for r in _flip()])
    st.markdown("**Unterhalb der Schwelle entsteht keine Konvention, und größere Gruppen machen es schlimmer** (p\\* = 1/4):")
    rows1 = _e1_mass_rows()
    table([{"n": r["n"], "x": V.de(r["x"], 0), "fast alle S": V.pct(r["high"], 1), "fast alle H": V.pct(r["low"], 1), "Mitte": V.pct(r["mid"], 1)} for r in rows1])

# --- Tab 2 ---------------------------------------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("🔬 Experiment E2: Zählt die Auszahlungsdominanz?")
    inv_rows, dom_rows = _e2()
    st.caption("Fünf Spiele mit derselben Lücke (a − d = 3, b − c = 1, also λ = 4, p* = 1/4), aber ganz verschiedenem a − b. n = 30, β = 1. Die Verteilungen liegen aufeinander.")
    st.plotly_chart(Z.build_overlay(_e2_overlay(), [f"{g}" for g in C.INVARIANCE_GAMES]), width="stretch", key="e2_overlay")
    table([{"Spiel (a, b, c, d)": str(r["game"]), "a − b": r["a_minus_b"], "p*": V.frac(r["pstar"]), "Masse fast alle S": V.pct(r["high"], 2), "Abweichung zur ersten Zeile": V.sci(r["deviation"])} for r in inv_rows])
    st.markdown("**Gegenprobe: das auszahlungsdominante Format verliert.** Spiel (1, p\\*, 0, p\\*): S zahlt gegen S das 1/p\\*-fache des sicheren H, gegen H nichts. n = 50.")
    xs_dom = [k for k in dom_rows[0] if isinstance(k, float)]
    table([{"p*": V.frac(r["pstar"]), "S zahlt das …-fache von H": V.de(float(r["ratio"]), 2), **{f"Masse bei H, x = {V.de(xx, 0)}": V.pct(r[xx][1], 2) for xx in xs_dom}} for r in dom_rows])
    v2 = V.verdict_e2(inv_rows, dom_rows, 50, 12.0)
    show(v2)
    verdicts.append(v2)

# --- Tab 3 ---------------------------------------------------------------------------------------------------------------------------------
with tab3:
    st.subheader("🔬 Experiment E3: Große Gruppen und der kritische Wert")
    mode_rows, mid_rows, fp_rows, branch_rows = _e3()
    ps_f = float(view.p_star) if view.p_star is not None else 0.25
    st.markdown(f"**Die Ratenfunktion** für p\\* = {V.de(ps_f, 3)} (aus dem gewählten Spiel, sonst 1/4) und x = {V.de(settings.x, 2)}: Für große n ist log π(k)/n ≈ g(k/n), g(y) = x·(y²/2 − p\\*·y) + H(y) mit der Entropie H. Die Maxima von g sind die Logit-Gleichgewichte y = σ(x·(y − p\\*)).")
    st.plotly_chart(Z.build_rate(settings.x, ps_f), width="stretch", key="e3_rate")
    maxima = LI.local_maxima(settings.x, ps_f)
    st.caption("Lokale Maxima von g: " + "; ".join(f"y = {V.de(p, 3)}" for p, _ in maxima) + f". Zahl der Lösungen von y = σ(x·(y − p*)): {LI.count_fixed_points(settings.x, ps_f)}.")
    table([{"p*": V.de(r["pstar"], 2), "x": V.de(r["x"], 0), f"Modus bei n = {C.PHASE_N}": V.de(r["mode"], 4), "Maximum von g": V.de(r["rate"], 4), "Abstand": V.de(r["gap"], 4)} for r in mode_rows])
    st.markdown("**Bei p\\* = 1/2 gibt es einen kritischen Wert.** Unterhalb hat die Ratenfunktion nur ein Maximum in der Mitte, also auch für sehr große Gruppen keine Konvention, oberhalb zwei Maxima:")
    st.plotly_chart(Z.build_middle_mass(mid_rows), width="stretch", key="e3_middle")
    table([{"p*": V.de(r["pstar"], 2), "x": V.de(r["x"], 1), "Zahl der Lösungen": r["count"]} for r in fp_rows if r["pstar"] == 0.5])
    table([{"p*": V.de(r["pstar"], 2), "zweites lokales Maximum ab x": ("keines" if r["x"] is None else V.de(r["x"], 2))} for r in branch_rows])
    v3 = V.verdict_e3(mode_rows, mid_rows, fp_rows, branch_rows)
    show(v3)
    verdicts.append(v3)

# --- Tab 4 ---------------------------------------------------------------------------------------------------------------------------------
with tab4:
    st.subheader("🔬 Experiment E4: Folgt die Auswahl dem Spiel oder dem Fehlermodell?")
    cnt_rows, fin_rows, typ_rows, tipping = _e4()
    st.markdown("**Ihr Spiel mit den drei Wahlregeln** (Schärfe x und Fehlerraten wie in der Seitenleiste; logarithmische Achse):")
    pis = {C.MODELS[0]: CH.stationary(V.rates_for(settings, C.MODELS[0])), f"{C.MODELS[1]} (ε = {V.sci(settings.eps)})": CH.stationary(V.rates_for(settings, C.MODELS[1])),
           f"{C.MODELS[2]} (ε_S = {V.sci(settings.eps_s)}, ε_H = {V.sci(settings.eps_h)})": CH.stationary(V.rates_for(settings, C.MODELS[2]))}
    st.plotly_chart(Z.build_models(pis), width="stretch", key="e4_models")
    table([{"Wahlregel": name, "fast alle S": V.pct(CH.masses(pi).high, 2), "fast alle H": V.pct(CH.masses(pi).low, 2), "Mitte": V.pct(CH.masses(pi).mid, 2)} for name, pi in pis.items()])
    st.markdown("**Gleichmäßige Fehler: die Steigung ist eine Zählung.** log₁₀(π(alle S)/π(alle H)) wächst je Dekade von 1/ε um (Zahl der Zustände mit Δ > 0) − (Zahl mit Δ < 0):")
    table([{"p*": V.frac(r["pstar"]), "n": r["n"], "Δ > 0": r["npos"], "Δ < 0": r["nneg"], "Gleichstand": r["ties"], "Zählung": r["count"], "gemessene Steigung": V.de(r["slope"], 2)} for r in cnt_rows])
    st.markdown("**In der endlichen Gruppe** (n = 20, Masse „fast alle S“): gleichmäßige Fehler unterscheiden p\\* kaum, der Logit (x = 4) schon:")
    table([{"p*": V.frac(r["pstar"]), **{f"ε = {e}": V.pct(val, 2) for e, val in zip((0.3, 0.2, 0.1), r["uniform"])}, "Logit, x = 4": V.pct(r["logit"], 2)} for r in fin_rows])
    st.markdown("**Typabhängige Fehler kippen die Auswahl** (p\\* = 1/4, n = 40, ε_S = 0,1; ε_H sinkt von links nach rechts):")
    st.plotly_chart(Z.build_typed_tipping(typ_rows, tipping), width="stretch", key="e4_typed")
    table([{"ε_H": V.sci(r["eps_h"]), "fast alle S": V.pct(r["high"], 2), "fast alle H": V.pct(r["low"], 2), "Mitte": V.pct(r["mid"], 2)} for r in typ_rows])
    v4 = V.verdict_e4(cnt_rows, fin_rows, typ_rows, tipping)
    show(v4)
    verdicts.append(v4)

# --- Tab 5 ---------------------------------------------------------------------------------------------------------------------------------
with tab5:
    st.subheader("🔬 Experiment E5: Drei Formate")
    st.caption(f"Symmetrische Zufallsspiele mit drei Formaten, n = {C.THREE_N} Betriebe ({len(T.compositions(C.THREE_N))} Zustände), ganzzahlige Einträge 0 bis 9, strenge Gleichgewichte, "
               f"{len(C.SWEEP_SEEDS)} feste Ströme zu je {C.THREE_GAMES_PER_STREAM} Spielen; Logit mit Schärfe {V.de(C.THREE_X, 0)} (β mal größte Diagonale), gleichmäßige Fehler ε = {V.sci(C.THREE_EPS)}. Rechnet einige Sekunden.")
    if st.button("▶️ Drei-Formate-Experiment rechnen", key="e5_button"):
        st.session_state["e5_done"] = True
    if st.session_state.get("e5_done"):
        with st.spinner("Rechne 160 Ketten mit je 91 Zuständen …"):
            e5 = _e5()
        t5 = e5["total"]
        table([{"Strom": r["seed"], "Spiele": r["games"], "Logit = Potential": f"{r['logit_eq_potential']} von {r['logit_clear']}", "Logit = gleichmäßig": f"{r['logit_eq_uniform']} von {r['both_clear']}",
                "Logit = Auszahlungsdominanz": f"{r['logit_eq_payoff']} von {r['payoff_comparable']}"} for r in e5["streams"]]
              + [{"Strom": "alle", "Spiele": t5["games"], "Logit = Potential": f"{t5['logit_eq_potential']} von {t5['logit_clear']}", "Logit = gleichmäßig": f"{t5['logit_eq_uniform']} von {t5['both_clear']}",
                  "Logit = Auszahlungsdominanz": f"{t5['logit_eq_payoff']} von {t5['payoff_comparable']}"}])
        v5 = V.verdict_e5(e5)
        show(v5)
        verdicts.append(v5)
    st.markdown("**Ein Spiel, in dem die Wahlregeln verschieden wählen** (Existenznachweis, kein Häufigkeitsurteil): zwei Spiele aus längeren Strömen, in denen Logit und gleichmäßige Fehler verschiedene Formate wählen.")
    for ex in _e5_examples():
        st.markdown(f"Strom {ex['seed']}, Spiel Nr. {ex['index'] + 1}: Sym = {ex['sym']}, Spaltenverschiebung g = {ex['shift']}")
        st.plotly_chart(Z.build_pure_masses(["Logit (Schärfe 20)", f"gleichmäßig (ε = {V.sci(C.THREE_EPS)})"], [ex["logit_masses"], ex["uniform_masses"]], ["Format 1", "Format 2", "Format 3"]),
                        width="stretch", key=f"e5_chart_{ex['seed']}_{ex['index']}")
        show(V.verdict_e5_dissenter(ex["sym"], ex["logit"], ex["uniform"]))

# --- Tab 6 ---------------------------------------------------------------------------------------------------------------------------------
with tab6:
    st.subheader("🔬 Experiment E6: Kann man es sehen?")
    st.caption("Die stationäre Verteilung ist exakt rechenbar, aber nicht immer beobachtbar: Die mittlere Zeit, ein Becken zu verlassen, wächst mit n und x; die Zahlen unten sind Zehnerpotenzen (Aktualisierungen je Betrieb).")
    v6 = V.verdict_e6_live(view)
    show(v6)
    verdicts.append(v6)
    st.plotly_chart(Z.build_times(_e6_rows()), width="stretch", key="e6_times")
    st.caption("Normiertes Spiel mit p* = 1/4. Die Zeit ist der Erwartungswert bis zur Barriere k* (erster Zustand, in dem S die beste Antwort ist), im Logarithmus gerechnet.")

    st.markdown("**Simulation der Verteilung für Ihr Spiel** (Logit, Einzelspieler, Budget aus der Seitenleiste; Rechenzeit wächst mit den Schritten).")
    if view.klass != G.KOORDINATION or settings.model != C.MODELS[0]:
        st.info("Die Simulation läuft für den Logit im Koordinationsspiel (Wahlregel „Logit“ in der Seitenleiste).")
    elif st.button("▶️ Simulation der Verteilung starten", key="sim_button"):
        st.session_state["sim_signature"] = (settings.game, settings.n, settings.x, settings.sim_steps, settings.start, settings.seed)
    if st.session_state.get("sim_signature") == (settings.game, settings.n, settings.x, settings.sim_steps, settings.start, settings.seed) and view.klass == G.KOORDINATION and settings.model == C.MODELS[0]:
        with st.spinner(f"Simuliere {settings.sim_steps:,} Aktualisierungen …".replace(",", ".")):
            hist = _simulate(settings.a, settings.b, settings.c, settings.d, settings.n, settings.beta, settings.sim_steps, S.start_state(settings.n, settings.start), settings.seed)
        st.plotly_chart(Z.build_sim_vs_exact(view.pi, hist), width="stretch", key="e6_sim_chart")
        tv = SI.total_variation(hist, view.pi)
        rates_l = CH.rates_logit(settings.n, settings.game, settings.beta)
        times = [t for t in (SI.escape_log_time(rates_l, settings.game, "H"), SI.escape_log_time(rates_l, settings.game, "S")) if t is not None]
        expected = max(times) if times else None
        v_sim = V.verdict_simulation(tv, expected, settings.sim_steps / settings.n)
        show(v_sim)
        verdicts.append(v_sim)

    st.markdown("**Formel gegen Simulation der Austrittszeit** (normiertes Spiel p\\* = 1/4, acht Zellen, vier Ströme, Läufe aus der Seitenleiste). Rechnet einige Sekunden bis eine halbe Minute.")
    if st.button("▶️ Formel gegen Simulation rechnen", key="e6_button"):
        st.session_state["e6_done"] = True
    if st.session_state.get("e6_done"):
        with st.spinner("Simuliere die Austrittszeiten …"):
            rows6 = _e6_check(settings.runs)
        table([{"n": r["n"], "x": V.de(r["x"], 0), "aus": "alle " + r["side"], "Formel": V.de(r["formula"], 2), "Simulation (Mittel über Ströme)": "–" if r["sim"] is None else V.de(r["sim"], 2),
                "Standardfehler": "–" if r["se"] is None else V.de(r["se"], 2), "abgebrochene Läufe": r["capped"]} for r in rows6])
        v6t = V.verdict_e6_table(rows6)
        show(v6t)
        verdicts.append(v6t)

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **2×2 und vollständige Durchmischung:** jeder Betrieb trifft jeden anderen gleich oft | In Netzen und Nachbarschaften entscheidet, wer wen sieht; die Schwelle p* gilt dann nicht mehr als Einzugsbereich der ganzen Gruppe, und die Wechselzeit der Konvention hängt am Netz. | komplexe-ansteckung-demo (Schwellen auf Netzen) |
| **Ein Betrieb je Schritt, alle gleich schnell** | Bei gleichzeitiger Aktualisierung aller Betriebe oder bei verschiedenen Lerngeschwindigkeiten kann die Verteilung anders ausfallen; gemessen ist nur die asynchrone Aktualisierung. | – |
| **Auszahlungen gewählt, nicht gemessen** | Alle Zahlen hängen an der Wahl von a, b, c, d. Der Befund „die Risikodominanz entscheidet“ gilt für diese Lernregel, nicht für das, was Menschen oder Terminals tun. | – |
| **Eine Lernregel für alle** (gleiche Schärfe, gleiche Fehlerraten) | Wie E4 zeigt, folgt die Auswahl dem Fehlermodell: Mit typabhängigen Fehlern kippt sie (gemessen bei p* = 1/4, n = 40). Heterogene Betriebe sind nicht gerechnet. | – |
| **Zeit nur als Erwartungswert** | Die stationäre Verteilung ist die Antwort für unendlich lange Zeit. Wie lange das System in einem Becken bleibt, zeigt E6 nur als mittlere Austrittszeit, nicht als Wechselzeit auf Netzen oder mit Gedächtnis. | – |
| **Zwei Formate (Hauptansicht)** | Drei Formate rechnet nur E5 mit n = 12; ob die Aussagen dort für andere Spielklassen oder größere Gruppen gelten, ist nicht gemessen. | – |
    """
)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist ein symmetrisches 2×2-Spiel mit den Auszahlungen $a, b, c, d$ und $n$ Betriebe, die sich immer wieder neu entscheiden; gesucht ist die Verteilung der Zahl $k$ der S-Betriebe auf Dauer.

| Symbol | Bedeutung |
|---|---|
| $a, b, c, d$ | Auszahlung: beide S, beide H, ich S und der andere H, ich H und der andere S |
| $\lambda, p^*$ | Skala $(a-d)+(b-c)$ und Schwelle $(b-c)/\lambda$ |
| $n,\ k,\ j$ | Zahl der Betriebe, der S-Betriebe, der anderen S-Betriebe |
| $\beta,\ x$ | Schärfe der Wahl je Minute und in Einheiten der Skala, $x = \beta\lambda$ |
| $\Delta_j$ | Vorteil von S gegenüber H bei $j$ anderen S-Betrieben |
| $\Phi$ | Potential |
| $\pi$ | stationäre Verteilung |

**Vorteil von S gegenüber H** bei $j$ anderen S-Betrieben (Mittel gegen die $n-1$ anderen):

$$\Delta_j = \frac{(a-d)\,j - (b-c)\,(n-1-j)}{n-1}$$

**Aufwärts- und Abwärtsrate** (ein gleichverteilt gezogener Betrieb entscheidet neu; Kante zwischen $k$ und $k+1$):

$$r^{+}_k = \frac{n-k}{n}\,\Pr(S \mid \Delta_k), \qquad r^{-}_k = \frac{k+1}{n}\,\Pr(H \mid \Delta_k)$$

**Stationäre Verteilung** (Detailed Balance der Geburts-Sterbe-Kette):

$$\frac{\pi(k+1)}{\pi(k)} = \frac{r^{+}_k}{r^{-}_k}$$

**Logit** und seine stationäre Verteilung (Gibbs-Maß mal Binomialfaktor):

$$\Pr(S \mid \Delta) = \frac{1}{1+e^{-\beta\Delta}}, \qquad \pi(k) \propto \binom{n}{k}\, e^{\beta\,\Phi(k)}$$

**Potential** (Summe des Paarpotentials über alle Paare):

$$\Phi(k) = \frac{(a-d)\binom{k}{2} + (b-c)\binom{n-k}{2}}{n-1}$$

**Große Gruppen.** Mit $y = k/n$ und $x = \beta\lambda$ gilt $\log \pi(k) \approx n\,g(y)$ mit der Ratenfunktion ($H$ die Entropie):

$$g(y) = x\left(\tfrac{y^2}{2} - p^* y\right) + H(y), \qquad H(y) = -y\ln y - (1-y)\ln(1-y)$$

Die Maxima von $g$ lösen die Logit-Gleichgewichts-Bedingung; bei $p^* = \tfrac12$ gibt es einen kritischen Wert:

$$y = \frac{1}{1 + e^{-x\,(y - p^*)}}, \qquad x_{\mathrm{krit}} = 4 \text{ bei } p^* = \tfrac12$$

**Gleichmäßige Fehler** (mit Wahrscheinlichkeit $\varepsilon$ zufällige Wahl): Mit $N_+$ Zuständen $\Delta > 0$ und $N_-$ Zuständen $\Delta < 0$ gilt exakt

$$\ln\frac{\pi(n)}{\pi(0)} = (N_+ - N_-)\,\ln\frac{2-\varepsilon}{\varepsilon}$$

**Typabhängige Fehler** ($\varepsilon_S$: S ist beste Antwort, H wird gewählt; $\varepsilon_H$ umgekehrt):

$$\ln\frac{\pi(n)}{\pi(0)} = N_+\ln\frac{1-\varepsilon_S}{\varepsilon_S} - N_-\ln\frac{1-\varepsilon_H}{\varepsilon_H}$$

**Mittlere Zeit bis zur Barriere** $m$ von Zustand 0 aus (reflektierend in 0), in Einzelaktualisierungen:

$$E\,T_{0\to m} = \sum_{k=0}^{m-1} \frac{1}{\pi_k\, r^{+}_k}\sum_{j=0}^{k} \pi_j$$

**Literatur.** Die Quellen sind in der README mit Fundstelle genannt; Formeln und Zahlen dieser Demo sind eigene Rechnung, durch Aufzählung aller $2^n$ Konfigurationen und die volle Übergangsmatrix bestätigt.
        """
    )

st.markdown("---")
st.subheader("📄 Ergebnisse als PDF")
if st.button("PDF erstellen", key="pdf_button"):
    from ggw_pdf_export import build_pdf
    st.session_state["pdf_bytes"] = build_pdf(view, verdicts)
if st.session_state.get("pdf_bytes"):
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="gleichgewichtsauswahl.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
