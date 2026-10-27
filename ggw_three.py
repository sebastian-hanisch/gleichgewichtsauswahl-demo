"""Drei (oder mehr) Formate: die Kette über Zusammensetzungen und die stationäre Verteilung nach Grassmann–Taksar–Heyman.

Ein gleichverteilt gezogener Betrieb entscheidet neu; die Auszahlungsmatrix A hat die Zeile = eigenes Format, die Spalte = Format des anderen. Im Zustand
s = (s_1, ..., s_m) (Zahl der Betriebe je Format) bekommt ein Betrieb mit Format i gegen die n - 1 anderen die mittlere Auszahlung u = A·(s - e_i)/(n - 1). Wahlregeln:

* Logit: Format j mit Wahrscheinlichkeit exp(beta·u_j)/Summe.
* Gleichmäßige Fehler: mit Wahrscheinlichkeit 1 - eps die beste Antwort (Gleichstände gleich verteilt), mit eps gleichverteilt über alle Formate.

Die Zustandszahl ist C(n + m - 1, m - 1) (231 für n = 20, m = 3); die volle Matrix ist dicht und GTH rechnet in O(Zustände³)."""

from __future__ import annotations

from itertools import combinations

import numpy as np

LOGIT = "logit"
UNIFORM = "uniform"


def compositions(n: int, m: int = 3) -> list[tuple[int, ...]]:
    """Alle Zusammensetzungen von n in m nichtnegative Summanden (Sterne und Striche), lexikografisch."""
    out = []
    for cuts in combinations(range(n + m - 1), m - 1):
        prev, parts = -1, []
        for c in cuts:
            parts.append(c - prev - 1)
            prev = c
        parts.append(n + m - 2 - prev)
        out.append(tuple(parts))
    return sorted(out)


def choice_probabilities(u: np.ndarray, mode: str, par: float) -> np.ndarray:
    """Wahlwahrscheinlichkeiten bei den Auszahlungen u: Logit mit Schärfe par, oder beste Antwort mit Fehler par (Gleichstände gleich verteilt, exakt auf 1e-12)."""
    m = len(u)
    if mode == LOGIT:
        z = par * (u - u.max())
        q = np.exp(z)
        return q / q.sum()
    if mode == UNIFORM:
        best = (u > u.max() - 1e-12).astype(float)
        return (1 - par) * best / best.sum() + par / m
    raise ValueError(f"unbekannte Wahlregel: {mode}")


def chain_m(n: int, payoff, mode: str, par: float):
    """Zustände und Übergangsmatrix P der Kette über Zusammensetzungen; `payoff[i][j]` ist die Auszahlung von Format i gegen Format j."""
    A = np.asarray(payoff, dtype=float)
    m = A.shape[0]
    states = compositions(n, m)
    index = {s: i for i, s in enumerate(states)}
    P = np.zeros((len(states), len(states)))
    for s in states:
        for i in range(m):
            if s[i] == 0:
                continue
            others = np.array(s, dtype=float)
            others[i] -= 1
            q = choice_probabilities(A @ others / (n - 1), mode, par)
            for j in range(m):
                t = list(s)
                t[i] -= 1
                t[j] += 1
                P[index[s], index[tuple(t)]] += s[i] / n * q[j]
    return states, P


def gth(P: np.ndarray) -> np.ndarray:
    """Stationäre Verteilung nach Grassmann–Taksar–Heyman: Zustände von hinten eliminieren, ohne Subtraktion (stabil auch bei Wahrscheinlichkeiten unter 1e-60)."""
    k = P.shape[0]
    A = P.astype(float).copy()
    for i in range(k - 1, 0, -1):
        s = A[i, :i].sum()
        A[:i, :i] += np.outer(A[:i, i], A[i, :i]) / s
    pi = np.zeros(k)
    pi[0] = 1.0
    for i in range(1, k):
        pi[i] = pi[:i] @ A[:i, i] / A[i, :i].sum()
    return pi / pi.sum()


def pure_states(states: list[tuple[int, ...]]) -> list[int]:
    """Indizes der Zustände „alle Betriebe nutzen Format i“, i = 0 .. m - 1."""
    m = len(states[0])
    n = sum(states[0])
    return [states.index(tuple(n if j == i else 0 for j in range(m))) for i in range(m)]


def pure_masses(states, pi) -> np.ndarray:
    """Stationäre Masse der m reinen Zustände."""
    return np.array([pi[i] for i in pure_states(states)])


def selected_format(states, pi, min_mass: float = 0.5) -> int | None:
    """Das Format mit der größten Masse im reinen Zustand, wenn diese Masse mindestens `min_mass` beträgt, sonst None (keine eindeutige Auswahl)."""
    masses = pure_masses(states, pi)
    best = int(np.argmax(masses))
    return best if masses[best] >= min_mass else None
