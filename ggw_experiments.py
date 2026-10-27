"""Die Messreihen der sechs Experimente E1 bis E6 als reine Funktionen (kein Streamlit, aller Zufall aus SplitMix64).

E1 Schwellenfläche x*(p*, n) (die Tabelle liegt vorgerechnet in data/threshold_table.json mit Erzeuger-Hash), E2 gleiches Spiel mit anderem Gefälle (Invarianz),
E3 große Gruppen (Ratenfunktion, kritischer Wert), E4 Fehlermodelle, E5 drei Formate, E6 Austrittszeiten."""

from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction as F
from pathlib import Path

import numpy as np

import ggw_chain as CH
import ggw_constants as C
import ggw_game as G
import ggw_limit as LI
import ggw_scenario as S
import ggw_sim as SI
import ggw_three as T

DATA_FILE = Path(__file__).resolve().parent / "data" / "threshold_table.json"
HASHED_SOURCES = ("ggw_game.py", "ggw_chain.py")


def frac(text: str) -> F:
    return F(text)


# ------------------------------------------------------------------------------------------------------------------------------ E1


def generator_hash() -> str:
    """Kurzer Hash der Quellen, aus denen die Schwellentabelle entsteht (Zeilenenden normalisiert): ändert sich der Rechencode, ist die vorgerechnete Tabelle veraltet."""
    h = hashlib.sha256()
    for name in HASHED_SOURCES:
        text = (Path(__file__).resolve().parent / name).read_bytes().replace(b"\r\n", b"\n")
        h.update(name.encode() + b"\0" + text)
    return h.hexdigest()[:16]


def compute_threshold_table(pstars=C.THRESHOLD_PSTARS, ns=C.THRESHOLD_NS, level: float = C.THRESHOLD_LEVEL) -> dict:
    """x*(p*, n) für alle Zellen: kleinste Schärfe x = beta·lam mit Masse „fast alle S“ >= level (Bisektion); None heißt „nie“ (bis x = 400)."""
    return {"level": level, "pstars": list(pstars), "ns": list(ns), "x_star": [[CH.threshold_x(n, frac(p), level) for n in ns] for p in pstars]}


def write_threshold_table(path: Path = DATA_FILE) -> dict:
    """Schreibt die vorgerechnete Tabelle samt Erzeuger-Hash (Aufruf: tools/make_threshold_table.py)."""
    table = compute_threshold_table()
    table["generator_hash"] = generator_hash()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return table


def load_threshold_table(path: Path = DATA_FILE) -> dict:
    """Liest die vorgerechnete Tabelle; ist der Erzeuger-Hash veraltet, wird sie neu gerechnet (nie eine veraltete Tabelle zeigen)."""
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
        if table.get("generator_hash") == generator_hash() and table.get("level") == C.THRESHOLD_LEVEL:
            return table
    except (OSError, ValueError):
        pass
    table = compute_threshold_table()
    table["generator_hash"] = generator_hash()
    return table


def e1_mass_rows(pstar: F, ns=(10, 100, 1000), xs=C.SURFACE_XS) -> list[dict]:
    """Massen (fast alle S, fast alle H, Mitte) im normierten Spiel mit Schwelle pstar: eine Zeile je (n, x)."""
    rows = []
    game = G.game_from(1, pstar)
    for n in ns:
        for x in xs:
            m = CH.masses(CH.stationary(CH.rates_logit(n, game, x)))
            rows.append({"n": n, "x": x, "high": m.high, "low": m.low, "mid": m.mid})
    return rows


def e1_surface(pstars: list[F], xs: list[float], n: int) -> np.ndarray:
    """Matrix der Masse „fast alle S“: Zeile = p*, Spalte = x, für die Gruppengröße n."""
    out = np.zeros((len(pstars), len(xs)))
    for i, p in enumerate(pstars):
        game = G.game_from(1, p)
        for j, x in enumerate(xs):
            out[i, j] = CH.masses(CH.stationary(CH.rates_logit(n, game, x))).high
    return out


def e1_flip_in_p_star(n: int = 100, x: float = 8.0, pstars=(F(49, 100), F(1, 2), F(51, 100))) -> list[dict]:
    """Umschlag in p*: Masse „fast alle S“ und „fast alle H“ knapp unter, bei und knapp über p* = 1/2."""
    rows = []
    for p in pstars:
        m = CH.masses(CH.stationary(CH.rates_logit(n, G.game_from(1, p), x)))
        rows.append({"pstar": p, "high": m.high, "low": m.low, "mid": m.mid})
    return rows


# ------------------------------------------------------------------------------------------------------------------------------ E2


def e2_invariance(n: int = 30, beta: float = 1.0, games=C.INVARIANCE_GAMES) -> list[dict]:
    """Spiele mit gleichem (lam, p*) und verschiedenem a - b: größte Abweichung der Verteilung zur ersten und die Masse „fast alle S“."""
    ref = CH.stationary(CH.rates_logit(n, games[0], beta))
    rows = []
    for game in games:
        pi = CH.stationary(CH.rates_logit(n, game, beta))
        rows.append({"game": game, "a_minus_b": game[0] - game[1], "lam": G.lam(game), "pstar": G.p_star(game), "deviation": float(np.abs(pi - ref).max()), "high": CH.masses(pi).high})
    return rows


def e2_payoff_dominated(n: int = 50, pstars=(F(11, 20), F(3, 5), F(3, 4), F(9, 10)), xs=(4.0, 12.0, 40.0)) -> list[dict]:
    """Auszahlungsdominanz gegen Risikodominanz: Spiel (a, b, c, d) = (1, p*, 0, p*) (H zahlt sicher p*, S zahlt 1 gegen S und 0 gegen H, also das 1/p*-fache): Masse bei H gegen S."""
    rows = []
    for p in pstars:
        game = (F(1), p, F(0), p)
        row = {"pstar": p, "ratio": 1 / p}
        for x in xs:
            m = CH.masses(CH.stationary(CH.rates_logit(n, game, x)))      # lam = 1: x = beta
            row[x] = (m.high, m.low)
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------------------------------------ E3


def e3_mode_rows(pstars=(0.25, 0.4), xs=(1, 2, 3, 4, 5, 6, 8, 12, 20), n: int = C.PHASE_N) -> list[dict]:
    """Modus der exakten Verteilung (Anteil k/n) gegen das Maximum der Ratenfunktion g; ein Eintrag je (p*, x)."""
    rows = []
    for p in pstars:
        game = G.game_from(1, F(p).limit_denominator(1000))
        for x in xs:
            mode = int(np.argmax(CH.stationary(CH.rates_logit(n, game, x)))) / n
            rate = LI.mode_of_rate(x, p)
            rows.append({"pstar": p, "x": x, "mode": mode, "rate": rate, "gap": abs(mode - rate), "maxima": len(LI.local_maxima(x, p))})
    return rows


def e3_middle_rows(ns=C.PHASE_NS, xs=(3.0, 4.0, 4.5)) -> list[dict]:
    """p* = 1/2: Masse in der Mitte (0,4·n bis 0,6·n) nach Gruppengröße und Schärfe."""
    game = G.game_from(1, F(1, 2))
    return [{"n": n, "x": x, "mid": LI.middle_mass(CH.stationary(CH.rates_logit(n, game, x)))} for n in ns for x in xs]


def e3_fixed_point_rows(xs=(1.0, 2.0, 3.0, 3.9, 4.1, 5.0, 6.0, 8.0), pstars=(0.25, 0.5)) -> list[dict]:
    return [{"pstar": p, "x": x, "count": LI.count_fixed_points(x, p)} for p in pstars for x in xs]


def e3_second_branch_rows(pstars=(0.25, 0.4)) -> list[dict]:
    """Ab welcher Schärfe hat die Ratenfunktion ein zweites lokales Maximum (den unterlegenen Ast)?"""
    return [{"pstar": p, "x": LI.second_branch_start(p)} for p in pstars]


# ------------------------------------------------------------------------------------------------------------------------------ E4


def e4_count_rows(cells=((F(1, 4), 20), (F(1, 4), 50), (F(2, 5), 20), (F(2, 5), 50), (F(9, 20), 50)), eps_list=(1e-3, 1e-5, 1e-7)) -> list[dict]:
    """Gleichmäßige Fehler: log10(pi(alle S)/pi(alle H)) für verschiedene eps, die Steigung je Dekade und die Zählung (Zahl Delta > 0) - (Zahl Delta < 0)."""
    rows = []
    for p, n in cells:
        game = G.game_from(1, p)
        npos, nneg, ties = G.sign_counts(n, game)
        logs = []
        for eps in eps_list:
            pi = CH.stationary(CH.rates_uniform(n, game, eps))
            logs.append(math.log10(pi[n]) - math.log10(pi[0]))
        slope = (logs[-1] - logs[0]) / (math.log10(eps_list[0]) - math.log10(eps_list[-1])) if eps_list[0] != eps_list[-1] else None      # Dekaden von 1/eps; ein einzelnes eps hat keine Steigung
        rows.append({"pstar": p, "n": n, "npos": npos, "nneg": nneg, "ties": ties, "logs": logs, "slope": slope, "count": npos - nneg})
    return rows


def e4_finite_rows(n: int = 20, pstars=(F(1, 4), F(2, 5)), eps_list=(0.3, 0.2, 0.1), x: float = 4.0) -> list[dict]:
    """Endliche Gruppe: gleichmäßige Fehler unterscheiden p* kaum, der Logit schon. Masse „fast alle S“."""
    rows = []
    for p in pstars:
        game = G.game_from(1, p)
        rows.append({"pstar": p, "uniform": [CH.masses(CH.stationary(CH.rates_uniform(n, game, e))).high for e in eps_list],
                     "logit": CH.masses(CH.stationary(CH.rates_logit(n, game, x))).high})
    return rows


def e4_tipping_eps_h(n: int, pstar: F, eps_s: float) -> float | None:
    """Fehlerrate eps_H, bei der die typabhängige Auswahl kippt: (Zahl Delta > 0)·log((1 - eps_S)/eps_S) = (Zahl Delta < 0)·log((1 - eps_H)/eps_H), also
    eps_H = 1/(1 + odds_S^(npos/nneg)) mit odds_S = (1 - eps_S)/eps_S; None ohne negative Zustände."""
    npos, nneg, _ = G.sign_counts(n, G.game_from(1, pstar))
    if nneg == 0 or npos == 0:
        return None
    return 1.0 / (1.0 + ((1 - eps_s) / eps_s) ** (npos / nneg))


def e4_typed_rows(n: int = 40, pstar: F = F(1, 4), eps_s: float = 0.1, eps_h_list=(1e-1, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-7)) -> list[dict]:
    game = G.game_from(1, pstar)
    rows = []
    for eps_h in eps_h_list:
        m = CH.masses(CH.stationary(CH.rates_typed(n, game, eps_s, eps_h)))
        rows.append({"eps_h": eps_h, "high": m.high, "low": m.low, "mid": m.mid})
    return rows


# ------------------------------------------------------------------------------------------------------------------------------ E5


def _chain_selection(game: dict, mode: str, par: float, n: int):
    states, P = T.chain_m(n, game["payoff"], mode, par)
    return states, T.gth(P)


def logit_beta(game: dict, x: float) -> float:
    """beta der Drei-Formate-Experimente: die Schärfe x geteilt durch die größte Diagonale von Sym."""
    return x / max(game["sym"][i][i] for i in range(3))


def e5_game_outcome(game: dict, n: int = C.THREE_N, x: float = C.THREE_X, eps: float = C.THREE_EPS) -> dict:
    """Auswahl je Regel für ein Spiel: Logit, gleichmäßige Fehler (None, wenn keine reine Konvention mindestens die halbe Masse hat), Potentialformat, auszahlungsdominantes Format (None bei Gleichstand)."""
    return {"logit": T.selected_format(*_chain_selection(game, T.LOGIT, logit_beta(game, x), n)), "uniform": T.selected_format(*_chain_selection(game, T.UNIFORM, eps, n)),
            "potential": S.potential_format(game), "payoff": S.payoff_dominant_format(game)}


def tally_outcomes(pairs) -> dict:
    """Zählt die Auswahlen über Paare (Spiel, Ergebnis von `e5_game_outcome`): eindeutige Fälle, Übereinstimmungen und die Abweichler (beide eindeutig, verschieden)."""
    out = {"logit_clear": 0, "uniform_clear": 0, "both_clear": 0, "logit_eq_potential": 0, "logit_eq_uniform": 0, "logit_eq_payoff": 0, "payoff_comparable": 0, "uniform_eq_payoff": 0,
           "uniform_comparable": 0, "dissenters": []}
    for g, o in pairs:
        sel_l, sel_u, pay = o["logit"], o["uniform"], o["payoff"]
        out["logit_clear"] += sel_l is not None
        out["uniform_clear"] += sel_u is not None
        if sel_l is not None:
            out["logit_eq_potential"] += sel_l == o["potential"]
            if pay is not None:
                out["payoff_comparable"] += 1
                out["logit_eq_payoff"] += sel_l == pay
        if sel_u is not None and pay is not None:
            out["uniform_comparable"] += 1
            out["uniform_eq_payoff"] += sel_u == pay
        if sel_l is not None and sel_u is not None:
            out["both_clear"] += 1
            if sel_l == sel_u:
                out["logit_eq_uniform"] += 1
            else:
                out["dissenters"].append({"sym": g["sym"], "payoff": g["payoff"], "logit": sel_l, "uniform": sel_u})
    return out


def e5_stream(seed: int, count: int = C.THREE_GAMES_PER_STREAM, n: int = C.THREE_N, x: float = C.THREE_X, eps: float = C.THREE_EPS) -> dict:
    """Ein Strom von Spielen mit drei Formaten: Auswahl des Logit, der gleichmäßigen Fehler, des Potentials (größte Diagonale von Sym) und der Auszahlungsdominanz (größte Diagonale von A)."""
    games = S.draw_three_format_games(seed, count)
    return {"seed": seed, "games": count, **tally_outcomes((g, e5_game_outcome(g, n, x, eps)) for g in games)}


def e5_dissenter_scan(seeds=C.SWEEP_SEEDS, count: int = 600, n: int = C.THREE_N, x: float = C.THREE_X, eps: float = C.THREE_EPS) -> dict:
    """Suche nach Spielen, in denen Logit und gleichmäßige Fehler verschieden wählen: je Strom die ersten `count` Spiele. Gibt die Zahl der Spiele, die der Fälle mit zwei eindeutigen verschiedenen
    Auswahlen (`differ`), die der Fälle, in denen eine der beiden Regeln nicht eindeutig wählt (`unclear`), und die Fundstellen (Strom, Index ab 0) der verschiedenen."""
    total = differ = unclear = 0
    found = []
    for seed in seeds:
        for index, g in enumerate(S.draw_three_format_games(seed, count)):
            o = e5_game_outcome(g, n, x, eps)
            total += 1
            if o["logit"] is None or o["uniform"] is None:
                unclear += 1
            elif o["logit"] != o["uniform"]:
                differ += 1
                found.append((seed, index))
    return {"games": total, "differ": differ, "unclear": unclear, "found": found}


def e5_example(game: dict, n: int = C.THREE_N, x: float = C.THREE_X, eps: float = C.THREE_EPS) -> dict:
    """Ein einzelnes Spiel: Auswahl und Massen der reinen Zustände für Logit und gleichmäßige Fehler."""
    st_l, pi_l = _chain_selection(game, T.LOGIT, logit_beta(game, x), n)
    st_u, pi_u = _chain_selection(game, T.UNIFORM, eps, n)
    return {"sym": game["sym"], "shift": game["shift"], "payoff": game["payoff"], "logit": T.selected_format(st_l, pi_l), "uniform": T.selected_format(st_u, pi_u),
            "logit_masses": T.pure_masses(st_l, pi_l).tolist(), "uniform_masses": T.pure_masses(st_u, pi_u).tolist()}


def e5_limit_check(game: dict, n: int = C.THREE_N) -> dict:
    """Wirkt eine Abweichung im Grenzfall fort? Reine Zustände bei gleichmäßigen Fehlern eps = 1e-4, 1e-6, 1e-8 und beim Logit mit Schärfe x = 20, 40, 80, 160."""
    uniform = {eps: T.pure_masses(*_chain_selection(game, T.UNIFORM, eps, n)).tolist() for eps in (1e-4, 1e-6, 1e-8)}
    logit = {x: T.pure_masses(*_chain_selection(game, T.LOGIT, logit_beta(game, x), n)).tolist() for x in (20.0, 40.0, 80.0, 160.0)}
    return {"uniform": uniform, "logit": logit}


def e5_three_formats(seeds=C.SWEEP_SEEDS) -> dict:
    streams = [e5_stream(s) for s in seeds]
    keys = ("games", "logit_clear", "uniform_clear", "both_clear", "logit_eq_potential", "logit_eq_uniform", "logit_eq_payoff", "payoff_comparable", "uniform_eq_payoff", "uniform_comparable")
    total = {k: sum(s[k] for s in streams) for k in keys}
    dissenters = [d for s in streams for d in s["dissenters"]]
    return {"streams": streams, "total": total, "dissenters": dissenters, "limits": [e5_limit_check(d) for d in dissenters]}


# ------------------------------------------------------------------------------------------------------------------------------ E6


def e6_time_rows(pstar: F = F(1, 4), ns=(10, 20, 100), xs=(2.0, 4.0, 8.0)) -> list[dict]:
    """log10 der mittleren Zeit (Aktualisierungen je Betrieb) bis zur Barriere k*, aus „alle H“ und aus „alle S“, im normierten Spiel mit Schwelle pstar."""
    game = G.game_from(1, pstar)
    rows = []
    for n in ns:
        for x in xs:
            rates = CH.rates_logit(n, game, x)
            rows.append({"n": n, "x": x, "kstar": G.barrier_state(n, game), "from_h": SI.escape_log_time(rates, game, "H"), "from_s": SI.escape_log_time(rates, game, "S")})
    return rows


def e6_simulation_check(cells=((10, 1.0, "H"), (10, 2.0, "H"), (10, 4.0, "H"), (20, 2.0, "H"), (10, 1.0, "S"), (10, 2.0, "S"), (10, 4.0, "S"), (20, 2.0, "S")),
                        pstar: F = F(1, 4), runs: int = 100, seeds=C.SWEEP_SEEDS, cap_steps: int = 2_000_000) -> list[dict]:
    """Formel gegen Einzelspieler-Simulation: je Zelle Mittel über die Ströme (Std über Ströme); Zellen, deren Zeit die Obergrenze übersteigt, werden als nicht beobachtbar gemeldet."""
    game = G.game_from(1, pstar)
    rows = []
    for n, x, side in cells:
        rates = CH.rates_logit(n, game, x)
        predicted = 10 ** SI.escape_log_time(rates, game, side)
        means, capped = [], 0
        for seed in seeds:
            times, c = SI.simulate_hitting(n, game, x, side, runs, seed, cap_steps)
            capped += c
            if times:
                means.append(float(np.mean(times)))
        std = float(np.std(means, ddof=1)) if len(means) > 1 else None
        rows.append({"n": n, "x": x, "side": side, "formula": predicted, "sim": float(np.mean(means)) if means else None, "std": std,
                     "se": std / math.sqrt(len(means)) if std is not None else None, "capped": capped, "runs": runs * len(seeds)})
    return rows
