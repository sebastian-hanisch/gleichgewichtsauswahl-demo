"""Das 2×2-Koordinationsspiel und sein Vorteilsmaß: eine Einheit je Regel (DEMO-PLAYBOOK §1b).

Spiel (a, b, c, d) in Minuten Zeitersparnis je Sendung: a beide wählen S (neues Format), b beide wählen H (altes Format), c ich S und der andere H, d ich H und der andere S.
In einer Gruppe von n Betrieben, von denen j der anderen n - 1 das Format S nutzen, ist der Vorteil von S gegenüber H

    Delta_j = u_S - u_H = ((a - d)·j - (b - c)·(n - 1 - j)) / (n - 1).

Das Spiel geht nur über die zwei Lücken A = a - d und B = b - c ein; lam = A + B ist die Skala, p* = B / lam die Schwelle (Anteil der S-Nutzer, ab dem S besser ist)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

KOORDINATION = "Koordination"
ANTI = "Anti-Koordination"
DOMINANT = "dominant"
ENTARTET = "entartet"


def _check_group(n: int) -> None:
    """Eine Gruppe braucht mindestens zwei Betriebe (der Vorteil teilt durch n - 1)."""
    if n < 2:
        raise ValueError(f"n = {n}: eine Gruppe braucht mindestens zwei Betriebe")


def gaps(game) -> tuple:
    """Die zwei Lücken (a - d, b - c): S gegenüber H, wenn der andere S bzw. H wählt (das Vorzeichen von b - c ist das Gegenteil des Gewinns von H)."""
    a, b, c, d = game
    return a - d, b - c


def lam(game):
    """Skala der Auszahlungen: (a - d) + (b - c)."""
    gap_s, gap_h = gaps(game)
    return gap_s + gap_h


def classify(game) -> str:
    """Koordination (a > d und b > c), Anti-Koordination (a < d und b < c), entartet (a = d oder b = c), sonst dominant."""
    gap_s, gap_h = gaps(game)
    if gap_s == 0 or gap_h == 0:
        return ENTARTET
    if gap_s > 0 and gap_h > 0:
        return KOORDINATION
    if gap_s < 0 and gap_h < 0:
        return ANTI
    return DOMINANT


def is_coordination(game) -> bool:
    return classify(game) == KOORDINATION


def p_star(game) -> Fraction | None:
    """Schwelle p* = (b - c) / ((a - d) + (b - c)) als Bruch; nur im Koordinationsspiel definiert (dort liegt sie in (0, 1))."""
    if not is_coordination(game):
        return None
    gap_s, gap_h = gaps(game)
    return Fraction(gap_h) / (Fraction(gap_s) + Fraction(gap_h))


def scale(game):
    """Skala für die skala-freie Schärfe x = beta·lam: lam im Koordinationsspiel, sonst |a - d| + |b - c| (und 1 bei lauter Gleichständen), damit beta endlich bleibt."""
    if is_coordination(game):
        return lam(game)
    gap_s, gap_h = gaps(game)
    return abs(gap_s) + abs(gap_h) or 1


def beta_from_x(game, x: float) -> float:
    """beta (je Minute) aus der skala-freien Schärfe x = beta·lam."""
    return float(x) / float(scale(game))


def game_from(lam_: float | Fraction, pstar: float | Fraction) -> tuple:
    """Ein Spiel mit Skala lam_ und Schwelle pstar: (a, b, c, d) = (lam·(1 - p*), lam·p*, 0, 0), also a - d = lam·(1 - p*) und b - c = lam·p*."""
    lam_ = Fraction(lam_)
    pstar = Fraction(pstar)
    return lam_ * (1 - pstar), lam_ * pstar, Fraction(0), Fraction(0)


def advantage_numerator(n: int, j: int, game):
    """Zähler von Delta_j (Nenner n - 1): (a - d)·j - (b - c)·(n - 1 - j). Exakt für ganze Zahlen und Brüche, daher für das Zählen von Vorzeichen."""
    _check_group(n)
    gap_s, gap_h = gaps(game)
    return gap_s * j - gap_h * (n - 1 - j)


def delta(n: int, game) -> np.ndarray:
    """Delta_j für j = 0 .. n - 1 anderer S-Betriebe, aus den Auszahlungen beider Formate gerechnet: u_S - u_H mit u_S = (j·a + (n - 1 - j)·c)/(n - 1), u_H = (j·d + (n - 1 - j)·b)/(n - 1)."""
    _check_group(n)
    a, b, c, d = (float(v) for v in game)
    j = np.arange(n, dtype=float)
    u_s = (j * a + (n - 1 - j) * c) / (n - 1)
    u_h = (j * d + (n - 1 - j) * b) / (n - 1)
    return u_s - u_h


def sign_counts(n: int, game) -> tuple[int, int, int]:
    """(Zahl der Zustände j mit Delta_j > 0, mit Delta_j < 0, mit Delta_j = 0): aus den exakten Zählern, ohne Gleitkommavergleich."""
    nums = [advantage_numerator(n, j, game) for j in range(n)]
    return sum(1 for v in nums if v > 0), sum(1 for v in nums if v < 0), sum(1 for v in nums if v == 0)


def barrier_state(n: int, game) -> int | None:
    """Erster Zustand j = 0 .. n - 1, in dem S die beste Antwort ist (Delta_j > 0), sonst None."""
    for j in range(n):
        if advantage_numerator(n, j, game) > 0:
            return j
    return None


def potential_pair(game) -> dict:
    """Paarpotential: P(S, S) = a - d, P(H, H) = b - c, P(S, H) = 0 (der Gewinn eines Wechsels ist die Potentialdifferenz)."""
    gap_s, gap_h = gaps(game)
    return {"SS": gap_s, "HH": gap_h, "SH": 0}


def potential_state(k: int, n: int, game) -> float:
    """Phi(k) = [(a - d)·k(k - 1)/2 + (b - c)·(n - k)(n - k - 1)/2] / (n - 1): Summe des Paarpotentials über alle Paare, wenn k Betriebe S nutzen."""
    _check_group(n)
    gap_s, gap_h = gaps(game)
    return float(gap_s * k * (k - 1) / 2 + gap_h * (n - k) * (n - k - 1) / 2) / (n - 1)


def potential_curve(n: int, game) -> np.ndarray:
    """Phi(0 .. n) als Feld."""
    return np.array([potential_state(k, n, game) for k in range(n + 1)])
