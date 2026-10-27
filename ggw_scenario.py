"""Szenario: Zufallsspiele mit drei Formaten (E5) und der Startzustand der Simulation. Aller Zufall kommt aus dem SplitMix64-Strom (ggw_rng.py): gleiche Eingabe, gleiche Zahlen auf jeder Plattform."""

from __future__ import annotations

import numpy as np

from ggw_rng import SplitMix64

ENTRY_LIMIT = 10         # Einträge der Matrix: ganze Zahlen 0 bis 9
SHIFT_LIMIT = 10         # Spaltenverschiebung g_j: ganze Zahlen 0 bis 9


def start_state(n: int, percent: int) -> int:
    """Zahl der S-Betriebe beim Start der Simulation: n·percent/100, auf eine ganze Zahl gerundet (ab 0,5 aufwärts), ganzzahlig gerechnet."""
    if not 0 <= percent <= 100:
        raise ValueError(f"percent = {percent} liegt außerhalb von 0 bis 100")
    return (2 * n * percent + 100) // 200


def _draw_int(rng: SplitMix64, limit: int) -> int:
    return int(rng.uniform() * limit)


def draw_three_format_games(seed: int, count: int) -> list[dict]:
    """`count` Spiele mit drei Formaten aus dem Strom SplitMix64(seed). Ein Spiel ist eine symmetrische Matrix Sym mit ganzzahligen Einträgen 0 bis 9, deren Diagonale paarweise verschieden
    ist und in jeder Spalte strikt über allen anderen Einträgen liegt (drei strenge Gleichgewichte „alle wählen i“), plus eine Spaltenverschiebung g_j in 0 bis 9: die Auszahlung von Format i
    gegen Format j ist A[i][j] = Sym[i][j] + g_j. Die Verschiebung ändert keine beste Antwort und keine Logit-Wahl, wohl aber die Diagonale (die Auszahlungsdominanz).
    Verworfene Ziehungen werden gezählt (`rejected`). Gibt Wörterbücher mit 'sym', 'shift', 'payoff' und 'rejected' (Zahl der bis zu diesem Spiel verworfenen Ziehungen)."""
    rng = SplitMix64(seed)
    games, rejected = [], 0
    while len(games) < count:
        raw = [[_draw_int(rng, ENTRY_LIMIT) for _ in range(3)] for _ in range(3)]
        sym = [[raw[min(i, j)][max(i, j)] for j in range(3)] for i in range(3)]
        diag = [sym[i][i] for i in range(3)]
        strict = all(sym[i][i] > sym[j][i] for i in range(3) for j in range(3) if j != i)
        if len(set(diag)) < 3 or not strict:
            rejected += 1
            continue
        shift = [_draw_int(rng, SHIFT_LIMIT) for _ in range(3)]
        payoff = [[sym[i][j] + shift[j] for j in range(3)] for i in range(3)]
        games.append({"sym": sym, "shift": shift, "payoff": payoff, "rejected": rejected})
    return games


def potential_format(game: dict) -> int:
    """Das Format mit dem größten Potential im reinen Zustand: die größte Diagonale von Sym (das Potential von „alle wählen i“ ist n·Sym[i][i]/2)."""
    diag = [game["sym"][i][i] for i in range(3)]
    return int(np.argmax(diag))


def payoff_dominant_format(game: dict) -> int | None:
    """Das Format mit der größten Diagonale der Auszahlungsmatrix A (Auszahlungsdominanz: alle wählen i und bekommen A[i][i]); None bei einem Gleichstand an der Spitze."""
    diag = [game["payoff"][i][i] for i in range(3)]
    best = max(diag)
    return diag.index(best) if diag.count(best) == 1 else None
