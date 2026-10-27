"""Große Gruppen: Ratenfunktion, Logit-Gleichgewichte und der kritische Wert (Phasenstruktur).

Mit der skala-freien Schärfe X = beta·lam und dem Anteil y = k/n der S-Betriebe gilt für große n  log pi(k) ≈ n·g(y) + const  mit

    g(y) = X·(y²/2 - p*·y) + H(y),   H(y) = -y·ln y - (1 - y)·ln(1 - y)  (Entropie).

Die Maxima von g sind die stabilen Lösungen von y = sigma(X·(y - p*)) (Logit-Gleichgewichte, „quantal response“). Bei p* = 1/2 gibt es für X <= 4 genau eine, für X > 4 drei Lösungen:
der kritische Wert X = 4 ist 1/sigma'(0). Alle Gitter bestehen aus Zellmitten (i + 1/2)/M; das Vorzeichen null am Gitterpunkt 1/2 (ungerade M) zählt als eine Lösung."""

from __future__ import annotations

import numpy as np

CRITICAL_X = 4.0       # 1/sigma'(0) = 4: Steigung der Logit-Antwort im Mittelpunkt gleich 1
DEFAULT_POINTS = 200_000


def cell_midpoints(points: int = DEFAULT_POINTS) -> np.ndarray:
    """Zellmitten (i + 1/2)/M, i = 0 .. M - 1: weder 0 noch 1 liegen auf dem Gitter."""
    return (np.arange(points) + 0.5) / points


def entropy(y: np.ndarray) -> np.ndarray:
    """H(y) = -y·ln y - (1 - y)·ln(1 - y) für y in (0, 1)."""
    return -y * np.log(y) - (1 - y) * np.log1p(-y)


def rate_function(y: np.ndarray, x: float, pstar: float) -> np.ndarray:
    """g(y) = X·(y²/2 - p*·y) + H(y)."""
    return x * (y ** 2 / 2 - float(pstar) * y) + entropy(y)


def local_maxima(x: float, pstar: float, points: int = DEFAULT_POINTS) -> list[tuple[float, float]]:
    """Lokale Maxima (y, g(y)) von g auf dem Zellmitten-Gitter, einschließlich des ersten und letzten Gitterpunkts, wenn dort ein Maximum liegt."""
    y = cell_midpoints(points)
    v = rate_function(y, x, pstar)
    idx = [int(i) for i in np.where((v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:]))[0] + 1]     # >=: bei p* = 1/2 liegen zwei gleich hohe Zellen nebeneinander, das Maximum zählt einmal (links)
    if v[0] >= v[1]:
        idx = [0] + idx
    if v[-1] > v[-2]:
        idx = idx + [points - 1]
    return [(float(y[i]), float(v[i])) for i in idx]


def mode_of_rate(x: float, pstar: float, points: int = DEFAULT_POINTS) -> float:
    """Ort des größten Werts von g (Modus der Verteilung für n -> unendlich)."""
    y = cell_midpoints(points)
    return float(y[int(np.argmax(rate_function(y, x, pstar)))])


def _best_response_gap(y: np.ndarray, x: float, pstar: float) -> np.ndarray:
    """f(y) = y - sigma(X·(y - p*)): die Nullstellen sind die Logit-Gleichgewichte."""
    return y - 1.0 / (1.0 + np.exp(-x * (y - float(pstar))))


def fixed_points(x: float, pstar: float, points: int = DEFAULT_POINTS) -> list[float]:
    """Lösungen von y = sigma(X·(y - p*)) auf (0, 1): Vorzeichenwechsel von f zwischen benachbarten Gitterpunkten (verfeinert durch Bisektion) und exakte Nullstellen auf dem Gitter,
    jede Lösung genau einmal (ein Gitterpunkt mit f = 0 zählt, aber nicht die beiden Wechsel daneben)."""
    y = cell_midpoints(points)
    f = _best_response_gap(y, x, pstar)
    roots = []
    for i in np.where(f == 0.0)[0]:
        roots.append(float(y[i]))
    s = np.sign(f)
    for i in np.where(s[:-1] * s[1:] < 0)[0]:
        lo, hi = float(y[i]), float(y[i + 1])
        flo = float(_best_response_gap(np.array([lo]), x, pstar)[0])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            fm = float(_best_response_gap(np.array([mid]), x, pstar)[0])
            if (fm < 0) == (flo < 0):
                lo, flo = mid, fm
            else:
                hi = mid
        roots.append(0.5 * (lo + hi))
    return sorted(roots)


def count_fixed_points(x: float, pstar: float, points: int = DEFAULT_POINTS) -> int:
    return len(fixed_points(x, pstar, points))


def critical_value(pstar: float = 0.5, lo: float = 1.0, hi: float = 10.0, steps: int = 40, points: int = DEFAULT_POINTS) -> float:
    """Kleinstes X, ab dem es bei p* (hier 1/2) mehr als eine Lösung gibt (Bisektion über die Zahl der Lösungen); Ergebnis 4 bis auf Gitterfeinheit."""
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if count_fixed_points(mid, pstar, points) > 1:
            hi = mid
        else:
            lo = mid
    return hi


def second_branch_start(pstar: float, lo: float = 1.0, hi: float = 60.0, steps: int = 40, points: int = DEFAULT_POINTS) -> float | None:
    """Kleinstes X, ab dem g zwei lokale Maxima hat (das zweite ist der unterlegene Ast); None, wenn es bei hi keines gibt."""
    if len(local_maxima(hi, pstar, points)) < 2:
        return None
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if len(local_maxima(mid, pstar, points)) >= 2:
            hi = mid
        else:
            lo = mid
    return hi


def middle_mass(pi: np.ndarray, lo: float = 0.4, hi: float = 0.6) -> float:
    """Masse der Zustände mit Anteil k/n zwischen lo und hi (beide eingeschlossen; ganzzahlig auf Promille verglichen: 1000·k gegen 1000·lo·n)."""
    n = len(pi) - 1
    k = np.arange(n + 1)
    return float(pi[(1000 * k >= round(lo * 1000) * n) & (1000 * k <= round(hi * 1000) * n)].sum())
