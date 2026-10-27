"""Auswertungen für die App: Einstellungen, Ansichten und die Verdikte als reine Funktionen (Art, Text). Die Zahlen im Text stammen aus der Rechnung, nicht aus Konstanten."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

import ggw_chain as CH
import ggw_constants as C
import ggw_game as G
import ggw_limit as LI
import ggw_sim as SI

NAMES = {"S": "das neue Format (S)", "H": "das alte Format (H)"}


# ---------------------------------------------------------------------------------------------------------------------------- Formate


def de(x: float, digits: int = 3) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 1, short: bool = False) -> str:
    """Anteil als Prozentzahl mit Komma. Ein Anteil knapp unter 1 (knapp über 0) wird nicht auf 100,0 % (0,0 %) gerundet, sondern als „mehr als 99,9 %“ („weniger als 0,1 %“) gezeigt;
    `short` kürzt das für schmale Kennzahlfelder zu „> 99,9 %“ und „< 0,1 %“."""
    s = f"{100 * x:.{digits}f}"
    if x < 1 and float(s) >= 100:
        s = ("> " if short else "mehr als ") + f"{100 - 10 ** -digits:.{digits}f}"
    elif x > 0 and float(s) == 0:
        s = ("< " if short else "weniger als ") + f"{10 ** -digits:.{digits}f}"
    return s.replace(".", ",") + " %"


def frac(x: Fraction) -> str:
    """Bruch als 2/3, ganze Zahlen ohne Nenner."""
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def power_of_ten(log10_value: float) -> str:
    """Zeit als Zehnerpotenz, z. B. „10^5,55“; unter 100 Aktualisierungen je Betrieb als Zahl."""
    if log10_value >= 2:
        return f"10^{de(log10_value, 2)}"          # nie 10 ** log10_value ausrechnen: bei n = 1 000 läuft die Zahl über den Gleitkommabereich
    value = 10 ** log10_value
    return de(value, 2 if value < 10 else 1)


def sci(x: float) -> str:
    """Kleine Zahl als 6,7·10^-16."""
    if x == 0:
        return "0"
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant.replace('.', ',')}·10^{int(exp)}"


# ---------------------------------------------------------------------------------------------------------------------------- Einstellungen


@dataclass(frozen=True)
class Settings:
    a: int = C.BASE["a"]
    b: int = C.BASE["b"]
    c: int = C.BASE["c"]
    d: int = C.BASE["d"]
    n: int = C.BASE["n"]
    x: float = C.BASE["x"]
    model: str = C.BASE["model"]
    eps_exp: float = C.BASE["eps_exp"]
    eps_s_exp: float = C.BASE["eps_s_exp"]
    eps_h_exp: float = C.BASE["eps_h_exp"]
    start: int = C.BASE["start"]
    sim_steps: int = C.BASE["sim_steps"]
    runs: int = C.BASE["runs"]
    seed: int = C.BASE["seed"]

    @property
    def game(self) -> tuple:
        return (self.a, self.b, self.c, self.d)

    @property
    def beta(self) -> float:
        return G.beta_from_x(self.game, self.x)

    @property
    def eps(self) -> float:
        return 10.0 ** -self.eps_exp

    @property
    def eps_s(self) -> float:
        return 10.0 ** -self.eps_s_exp

    @property
    def eps_h(self) -> float:
        return 10.0 ** -self.eps_h_exp


def rates_for(settings: Settings, model: str | None = None) -> CH.Rates:
    """Raten der gewählten Wahlregel (Logit mit beta = x/Skala; gleichmäßige Fehler mit eps; typabhängig mit eps_S und eps_H)."""
    model = model or settings.model
    if model == C.MODELS[0]:
        return CH.rates_logit(settings.n, settings.game, settings.beta)
    if model == C.MODELS[1]:
        return CH.rates_uniform(settings.n, settings.game, settings.eps)
    if model == C.MODELS[2]:
        return CH.rates_typed(settings.n, settings.game, settings.eps_s, settings.eps_h)
    raise ValueError(f"unbekanntes Fehlermodell: {model}")


# ---------------------------------------------------------------------------------------------------------------------------- Hauptansicht


@dataclass(frozen=True)
class MainView:
    settings: Settings
    klass: str
    lam: object
    scale: object
    p_star: Fraction | None
    beta: float
    pi: np.ndarray
    masses: CH.Masses
    mean: float
    potential: np.ndarray
    log_binom: np.ndarray
    risk: str | None            # risikodominantes Format ("S", "H" oder None bei p* = 1/2 bzw. keinem Koordinationsspiel)
    payoff: str | None          # auszahlungsdominantes Format ("S", "H" oder None bei a = b)
    threshold: float | None     # x*(p*, n) für das gewählte Spiel (nur Koordinationsspiel mit p* < 1/2), sonst None
    barrier: int | None


def risk_dominant(game) -> str | None:
    """Risikodominant ist das Format mit der größeren Lücke: S bei a - d > b - c (p* < 1/2), H bei a - d < b - c; None bei Gleichheit oder wenn es kein Koordinationsspiel ist."""
    if not G.is_coordination(game):
        return None
    gap_s, gap_h = G.gaps(game)
    return "S" if gap_s > gap_h else "H" if gap_s < gap_h else None


def payoff_dominant(game) -> str | None:
    """Auszahlungsdominant ist das Format mit der größeren Diagonale (a gegen b); None bei a = b."""
    a, b, _, _ = game
    return "S" if a > b else "H" if a < b else None


def analyse_main(settings: Settings) -> MainView:
    game, n = settings.game, settings.n
    pi = CH.stationary(rates_for(settings))
    ps = G.p_star(game)
    threshold = CH.threshold_x(n, ps) if ps is not None and ps < Fraction(1, 2) else None
    return MainView(settings, G.classify(game), G.lam(game), G.scale(game), ps, settings.beta, pi, CH.masses(pi), CH.mean_share(pi), G.potential_curve(n, game),
                    CH.entropy_curve(n), risk_dominant(game), payoff_dominant(game), threshold, G.barrier_state(n, game))


def verdict_main(v: MainView) -> tuple[str, str]:
    """Hauptverdikt: steht eine Konvention (Masse bei „fast alle S“ oder „fast alle H“ mindestens 99 %), und ist es die risikodominante?"""
    s, m = v.settings, v.masses
    n = s.n
    shares = f"fast alle S {pct(m.high)}, fast alle H {pct(m.low)}, Mitte {pct(m.mid)}"
    if v.klass != G.KOORDINATION:
        return "info", (f"**Kein Koordinationsspiel** (a > d und b > c wären nötig, hier gilt {v.klass}): Es gibt nichts auszuwählen, und die Skala der Schärfe ist die Summe der Lücken ({v.scale}). "
                        f"Masse bei n = {n}: {shares}.")
    where = f"p* = {frac(v.p_star)}"
    if s.model != C.MODELS[0]:
        what = "gleichmäßigen Fehlern" if s.model == C.MODELS[1] else "typabhängigen Fehlern"
        return "info", f"Wahlregel mit {what}, {where}, n = {n}: {shares}. Der Logit-Vergleich steht in Experiment E4."
    if max(m.high, m.low) >= 0.99:
        winner = "S" if m.high >= m.low else "H"
        base = f"**Eine Konvention steht:** {NAMES[winner]} mit {pct(max(m.high, m.low))} der Masse bei x = {de(s.x, 2)}, n = {n}, {where}."
        if v.payoff is not None and v.payoff != winner:
            lost = NAMES[v.payoff][0].upper() + NAMES[v.payoff][1:]
            return "warning", base + f" {lost} ist auszahlungsdominant (a = {s.a} gegen b = {s.b}), gewinnt aber nicht: Die Wahl folgt der Risikodominanz, nicht der Auszahlungsdominanz."
        return "success", base + (" Es ist auch das risikodominante." if v.risk == winner else "")
    if max(m.high, m.low) >= 0.5:
        return "info", f"**Eine Seite überwiegt, die Konvention steht noch nicht:** {shares} ({where}, x = {de(s.x, 2)}, n = {n}). Die Schärfe x ist zu klein, um {pct(0.99, 0)} zu erreichen."
    thr = ""
    if v.threshold is not None:
        thr = f" Für dieses Spiel und n = {n} braucht die Konvention x ≥ {de(v.threshold, 2)}."
    if m.mid >= 0.5:
        return "info", f"**Keine Konvention:** {shares} ({where}, x = {de(s.x, 2)}, n = {n}). Die meiste Masse liegt in gemischten Zuständen (es gibt von ihnen viel mehr Konfigurationen).{thr}"
    return "info", f"**Keine Konvention, zwei Seiten:** {shares} ({where}, x = {de(s.x, 2)}, n = {n}). Keine Seite hat die Mehrheit der Masse.{thr}"


# ---------------------------------------------------------------------------------------------------------------------------- E1


def verdict_e1(table: dict) -> tuple[str, str]:
    """Aus der Schwellentabelle: x* steigt mit p* (je Spalte) und fällt mit n (je Zeile)? Die Zählung nennt die Zellen."""
    xs = table["x_star"]
    rows, cols = len(xs), len(xs[0])
    rising = sum(1 for j in range(cols) for i in range(rows - 1) if xs[i][j] < xs[i + 1][j])
    falling = sum(1 for i in range(rows) for j in range(cols - 1) if xs[i][j] > xs[i][j + 1])
    pairs_p, pairs_n = (rows - 1) * cols, rows * (cols - 1)
    ok = rising == pairs_p and falling == pairs_n
    first = xs[1][0], xs[1][-1]
    text = (f"x* steigt mit p* in {rising} von {pairs_p} Paaren benachbarter Zellen und fällt mit n in {falling} von {pairs_n}. Beispiel p* = {table['pstars'][1]}: "
            f"von {de(first[0], 2)} (n = {table['ns'][0]}) auf {de(first[1], 2)} (n = {table['ns'][-1]}). Je knapper die Risikodominanz, desto schärfer muss gewählt werden; je größer die Gruppe, desto weniger.")
    return ("success" if ok else "warning"), text


def verdict_threshold_for(v: MainView) -> tuple[str, str]:
    """Die Schwelle für das gewählte Spiel gegen die gewählte Schärfe."""
    s = v.settings
    if v.klass != G.KOORDINATION:
        return "info", "Die Schwelle x* gibt es nur im Koordinationsspiel."
    if v.p_star >= Fraction(1, 2):
        return "info", (f"Bei p* = {frac(v.p_star)} erreicht „fast alle S“ die 99 % der Masse nie (bis x = 400 geprüft): x* gibt es nicht. "
                        f"Für „fast alle H“ gilt dieselbe Schwelle mit p* ↔ 1 − p*; die Tabelle zeigt p* < 1/2.")
    if v.threshold is None:
        return "info", f"Für p* = {frac(v.p_star)} und n = {s.n} erreicht die Masse bei „fast alle S“ auch bei x = 400 keine 99 %."
    reached = s.x >= v.threshold
    return ("success" if reached else "info"), (f"Für p* = {frac(v.p_star)} und n = {s.n} braucht „fast alle S“ (Masse ≥ 99 %) die Schärfe x* = {de(v.threshold, 2)}. "
                                                f"Gewählt ist x = {de(s.x, 2)}: {'erreicht' if reached else 'noch nicht erreicht'} (Masse {pct(v.masses.high)}).")


# ---------------------------------------------------------------------------------------------------------------------------- E2


def verdict_e2(rows: list[dict], dominated: list[dict], n: int, x_high: float) -> tuple[str, str]:
    dev = max(r["deviation"] for r in rows)
    lo, hi = min(r["a_minus_b"] for r in rows), max(r["a_minus_b"] for r in rows)
    flipped = sum(1 for r in dominated if r[x_high][1] > 0.99)
    text = (f"Fünf Spiele mit demselben (λ, p*), aber a − b von {lo} bis {hi}: größte Abweichung der Verteilungen {sci(dev)}. Im Logit zählt nur die Lücke (λ, p*), die Auszahlungsdominanz nicht. "
            f"Gegenprobe: In {flipped} von {len(dominated)} Spielen, in denen S das 1,1- bis 1,8-fache zahlt, liegt bei x = {de(x_high, 0)} und n = {n} mehr als 99 % der Masse bei H.")
    return ("success" if dev < 1e-12 else "warning"), text


# ---------------------------------------------------------------------------------------------------------------------------- E3


def verdict_e3(mode_rows: list[dict], middle_rows: list[dict], fixed_rows: list[dict], branch_rows: list[dict]) -> tuple[str, str]:
    gap = max(r["gap"] for r in mode_rows)
    f_low = max(r["x"] for r in fixed_rows if r["pstar"] == 0.5 and r["count"] == 1)
    f_high = min(r["x"] for r in fixed_rows if r["pstar"] == 0.5 and r["count"] == 3)
    big = max(r["n"] for r in middle_rows)
    mid_hi = next(r["mid"] for r in middle_rows if r["n"] == big and r["x"] == max(q["x"] for q in middle_rows))
    mid_lo = next(r["mid"] for r in middle_rows if r["n"] == big and r["x"] == min(q["x"] for q in middle_rows))
    return "info", (f"Modus der exakten Verteilung gegen das Maximum der Ratenfunktion: größte Abweichung {de(gap, 4)} bei n = {C.PHASE_N}. Bei p* = 1/2 gibt es für x ≤ {de(f_low, 1)} eine Lösung, für x ≥ {de(f_high, 1)} drei: kritischer Wert x = {de(LI.CRITICAL_X, 0)}. "
                    f"Masse in der Mitte bei n = {big}: {pct(mid_lo)} bei x = {de(min(q['x'] for q in middle_rows), 1)}, {pct(mid_hi)} bei x = {de(max(q['x'] for q in middle_rows), 1)}.")



# ---------------------------------------------------------------------------------------------------------------------------- E4


def verdict_e4(count_rows: list[dict], finite_rows: list[dict], typed_rows: list[dict], tipping: float | None) -> tuple[str, str]:
    exact = sum(1 for r in count_rows if abs(r["slope"] - r["count"]) < 0.01)
    first_diff = max(abs(finite_rows[0]["uniform"][i] - finite_rows[1]["uniform"][i]) for i in range(len(finite_rows[0]["uniform"])))
    logit_diff = finite_rows[0]["logit"] - finite_rows[1]["logit"]
    high_hi = typed_rows[0]["high"]
    low_lo = typed_rows[-1]["low"]
    tip = f" Gezählte Kippschwelle ε_H = {sci(tipping)}." if tipping is not None else ""
    return "info", (f"Gleichmäßige Fehler: Die Steigung von log₁₀(π(alle S)/π(alle H)) je Dekade von ε stimmt in {exact} von {len(count_rows)} Zellen mit der Zählung (Zahl Δ > 0) − (Zahl Δ < 0) überein; "
                    f"in der endlichen Gruppe unterscheiden sie p* kaum (größter Unterschied der Masse {de(first_diff, 4)}), der Logit schon ({de(logit_diff, 3)}). "
                    f"Typabhängige Fehler kippen die Auswahl: Masse bei S {pct(high_hi)} bei ε_H = {sci(typed_rows[0]['eps_h'])}, Masse bei H {pct(low_lo)} bei ε_H = {sci(typed_rows[-1]['eps_h'])}.{tip}")


# ---------------------------------------------------------------------------------------------------------------------------- E5


def verdict_e5(result: dict) -> tuple[str, str]:
    t = result["total"]
    return "info", (f"{t['games']} Spiele mit drei Formaten (n = {C.THREE_N}): Logit wählt das Format mit dem größten Potential in {t['logit_eq_potential']} von {t['logit_clear']} eindeutigen Fällen; "
                    f"Logit und gleichmäßige Fehler wählen dasselbe in {t['logit_eq_uniform']} von {t['both_clear']}; die Auszahlungsdominanz stimmt mit dem Logit in {t['logit_eq_payoff']} von {t['payoff_comparable']} überein "
                    f"(also weicht sie in {t['payoff_comparable'] - t['logit_eq_payoff']} ab).")


def verdict_e5_dissenter(sym: list, logit_sel: int | None, uniform_sel: int | None) -> tuple[str, str]:
    """Ein einzelnes Spiel: wählen Logit und gleichmäßige Fehler verschieden, und wem folgt jede Regel (dem Potential = größte Diagonale von Sym)?"""
    if logit_sel is None or uniform_sel is None or logit_sel == uniform_sel:
        return "info", "In diesem Spiel wählen Logit und gleichmäßige Fehler dasselbe, oder eine der beiden Regeln wählt nicht eindeutig."
    diag = [sym[i][i] for i in range(3)]
    pot = diag.index(max(diag))
    return "warning", (f"Logit wählt Format {logit_sel + 1}, gleichmäßige Fehler wählen Format {uniform_sel + 1}. Die größte Diagonale von Sym (das Potential) hat Format {pot + 1}: "
                       f"Der Logit {'folgt dem Potential' if logit_sel == pot else 'weicht vom Potential ab'}, die gleichmäßigen Fehler {'folgen dem Potential' if uniform_sel == pot else 'weichen vom Potential ab'}.")


# ---------------------------------------------------------------------------------------------------------------------------- E6


def verdict_e6_live(v: MainView) -> tuple[str, str]:
    """Austrittszeiten für das gewählte Spiel (Logit): aus „alle H“ und aus „alle S“ bis zur Barriere k*."""
    s = v.settings
    if v.klass != G.KOORDINATION or v.barrier in (None, 0):
        return "info", "Austrittszeiten gibt es nur im Koordinationsspiel mit einer Barriere k* > 0."
    rates = CH.rates_logit(s.n, s.game, v.beta)
    h = SI.escape_log_time(rates, s.game, "H")
    sd = SI.escape_log_time(rates, s.game, "S")
    return "info", (f"Barriere k* = {v.barrier} von n = {s.n} Betrieben. Aus „alle H“ dauert es im Mittel {power_of_ten(h)} Aktualisierungen je Betrieb, aus „alle S“ {power_of_ten(sd)}. "
                    f"Eine Simulation sieht die stationäre Verteilung nur, wenn ihr Budget ({s.sim_steps / s.n:.0f} Aktualisierungen je Betrieb) die längere dieser Zeiten übersteigt: "
                    f"{'es reicht' if max(h, sd) < math.log10(s.sim_steps / s.n) else 'es reicht nicht'}.")


def verdict_e6_table(rows: list[dict]) -> tuple[str, str]:
    done = [r for r in rows if r["sim"] is not None]
    inside = sum(1 for r in done if r["se"] is not None and abs(r["sim"] - r["formula"]) <= 3 * r["se"])
    return "info", f"Formel gegen Einzelspieler-Simulation: {inside} von {len(done)} Zellen liegen innerhalb von 3 Standardfehlern der Mittel über die Zufallsströme; abgebrochene Läufe: {sum(r['capped'] for r in rows)}."


def verdict_simulation(tv: float, expected_log10: float | None, steps_per_player: float) -> tuple[str, str]:
    """Simulation gegen exakte Verteilung: Gesamtvariationsabstand und der Vergleich Budget gegen Austrittszeit (als Zehnerlogarithmus der Aktualisierungen je Betrieb)."""
    if expected_log10 is None:
        return "info", f"Gesamtvariationsabstand zur exakten Verteilung {de(tv, 3)}. Für dieses Spiel gibt es keine Barriere, an der die Zeit gemessen werden kann."
    ok = math.log10(steps_per_player) > expected_log10
    return ("success" if ok and tv < 0.05 else "warning" if not ok else "info"), (
        f"Gesamtvariationsabstand zur exakten Verteilung {de(tv, 3)} nach {steps_per_player:.0f} Aktualisierungen je Betrieb. Die mittlere Austrittszeit des längeren Beckens ist {power_of_ten(expected_log10)}: "
        f"{'Das Budget reicht, die Verteilung kann sich einstellen.' if ok else 'Das Budget reicht nicht: Die Simulation bleibt in einem Becken und zeigt nicht die stationäre Verteilung (ein kleiner Abstand sagt dann nur, dass die Masse im erreichten Becken liegt).'}")
