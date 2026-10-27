"""Die Geburts-Sterbe-Kette über die Zahl k der S-Betriebe und ihre stationäre Verteilung (alles im Logarithmus).

Ein gleichverteilt gezogener Betrieb entscheidet neu. Im Zustand k (k Betriebe nutzen S) hat ein H-Betrieb k andere S-Betriebe um sich, ein S-Betrieb k - 1; die Kante
zwischen k und k + 1 gehört deshalb zu Delta_k: aufwärts mit Rate (n - k)/n · P(S | Delta_k), abwärts mit Rate (k + 1)/n · P(H | Delta_k). Die Raten stehen als Logarithmen
(`log_up[k]` von k nach k + 1, `log_down[k]` von k + 1 nach k), damit Wahrscheinlichkeiten von 10^-60 nicht unterlaufen. Drei Wahlregeln:

* Logit (Blume): P(S | Delta) = 1/(1 + exp(-beta·Delta)).
* Gleichmäßige Fehler (Kandori, Mailath, Rob; Young): beste Antwort, mit Wahrscheinlichkeit eps wird gleichverteilt zwischen S und H gewählt.
* Typabhängige Fehler: ist S die beste Antwort, wird mit eps_S trotzdem H gewählt; ist H die beste Antwort, mit eps_H trotzdem S.

Bei Delta = 0 (Gleichstand) wählen beide Fehlermodelle S und H je zur Hälfte. Das Vorzeichen von Delta kommt aus dem exakten Zähler (`ggw_game.advantage_numerator`)."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

import ggw_game as G


class Rates(NamedTuple):
    n: int
    log_up: np.ndarray      # Länge n, Kante k: k -> k + 1
    log_down: np.ndarray    # Länge n, Kante k: k + 1 -> k


class Masses(NamedTuple):
    high: float     # fast alle S: k >= 0,9·n
    low: float      # fast alle H: k <= 0,1·n
    mid: float      # alle übrigen Zustände


def _log_sigma(z: np.ndarray) -> np.ndarray:
    """log sigma(z) = -log(1 + exp(-z)), stabil für große |z|."""
    return -np.logaddexp(0.0, -z)


def rates_logit(n: int, game, beta: float) -> Rates:
    """Logit-Kette: aufwärts (n - k)/n·sigma(beta·Delta_k), abwärts (k + 1)/n·sigma(-beta·Delta_k)."""
    z = beta * G.delta(n, game)
    k = np.arange(n)
    return Rates(n, np.log((n - k) / n) + _log_sigma(z), np.log((k + 1) / n) + _log_sigma(-z))


def _signs(n: int, game) -> np.ndarray:
    """Vorzeichen von Delta_j (+1, 0, -1) aus den exakten Zählern."""
    out = np.zeros(n, dtype=int)
    for j in range(n):
        v = G.advantage_numerator(n, j, game)
        out[j] = 1 if v > 0 else -1 if v < 0 else 0
    return out


def _rates_from_probability(n: int, log_p_s: np.ndarray, log_p_h: np.ndarray) -> Rates:
    k = np.arange(n)
    return Rates(n, np.log((n - k) / n) + log_p_s, np.log((k + 1) / n) + log_p_h)


def rates_uniform(n: int, game, eps: float) -> Rates:
    """Gleichmäßige Fehler: P(S) = 1 - eps/2 bei Delta > 0, eps/2 bei Delta < 0, 1/2 bei Delta = 0."""
    if not 0 < eps <= 1:
        raise ValueError(f"eps = {eps} liegt außerhalb von (0, 1]")
    s = _signs(n, game)
    p_s = np.where(s > 0, 1 - eps / 2, np.where(s < 0, eps / 2, 0.5))
    p_h = np.where(s > 0, eps / 2, np.where(s < 0, 1 - eps / 2, 0.5))
    return _rates_from_probability(n, np.log(p_s), np.log(p_h))


def rates_typed(n: int, game, eps_s: float, eps_h: float) -> Rates:
    """Typabhängige Fehler: P(S) = 1 - eps_S bei Delta > 0, eps_H bei Delta < 0, 1/2 bei Delta = 0."""
    for e in (eps_s, eps_h):
        if not 0 < e < 1:
            raise ValueError(f"eps = {e} liegt außerhalb von (0, 1): bei 1 wählt dieses Format nie")
    s = _signs(n, game)
    p_s = np.where(s > 0, 1 - eps_s, np.where(s < 0, eps_h, 0.5))
    p_h = np.where(s > 0, eps_s, np.where(s < 0, 1 - eps_h, 0.5))
    return _rates_from_probability(n, np.log(p_s), np.log(p_h))


def stationary(rates: Rates) -> np.ndarray:
    """Stationäre Verteilung pi(0 .. n) der Geburts-Sterbe-Kette: log pi(k + 1) - log pi(k) = log_up[k] - log_down[k] (Detailed Balance)."""
    log_p = np.concatenate([[0.0], np.cumsum(rates.log_up - rates.log_down)])
    log_p -= log_p.max()
    p = np.exp(log_p)
    return p / p.sum()


def stationary_logit_closed(n: int, game, beta: float) -> np.ndarray:
    """Geschlossene Form pi(k) ~ C(n, k)·exp(beta·Phi(k)): Gibbs-Maß je Konfiguration mal Binomialfaktor (Binomialkoeffizient über lgamma)."""
    k = np.arange(n + 1)
    log_binom = np.array([math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) for i in k])
    log_p = log_binom + beta * G.potential_curve(n, game)
    log_p -= log_p.max()
    p = np.exp(log_p)
    return p / p.sum()


def transition_matrix(rates: Rates) -> np.ndarray:
    """Volle Übergangsmatrix (n + 1) × (n + 1) der Kette (für Orakel und kleine n)."""
    n = rates.n
    p = np.zeros((n + 1, n + 1))
    up, down = np.exp(rates.log_up), np.exp(rates.log_down)
    for k in range(n):
        p[k, k + 1] = up[k]
        p[k + 1, k] = down[k]
    p[np.arange(n + 1), np.arange(n + 1)] = 1.0 - p.sum(axis=1)
    return p


def masses(pi: np.ndarray) -> Masses:
    """Masse bei „fast alle S“ (k >= 0,9·n), „fast alle H“ (k <= 0,1·n) und in der Mitte; die Grenzen werden ganzzahlig verglichen (10·k gegen 9·n bzw. n)."""
    n = len(pi) - 1
    k = np.arange(n + 1)
    high = float(pi[10 * k >= 9 * n].sum())
    low = float(pi[10 * k <= n].sum())
    mid = float(pi[(10 * k < 9 * n) & (10 * k > n)].sum())
    return Masses(high, low, mid)


def mass_high_for(n: int, pstar, x: float) -> float:
    """Masse „fast alle S“ der Logit-Kette im normierten Spiel mit lam = 1 und Schwelle pstar bei der Schärfe x = beta·lam."""
    game = G.game_from(1, pstar)
    return masses(stationary(rates_logit(n, game, x))).high


def threshold_x(n: int, pstar, level: float = 0.99, x_max: float = 400.0, steps: int = 40) -> float | None:
    """Kleinste Schärfe x = beta·lam, bei der die Masse „fast alle S“ mindestens `level` erreicht (Bisektion, `steps` Schritte); None („nie“), wenn sie bei x_max nicht erreicht wird.
    Sinnvoll für p* < 1/2 (bei p* >= 1/2 hat S nie die Mehrheit der Masse)."""
    if mass_high_for(n, pstar, x_max) < level:
        return None
    lo, hi = 0.0, float(x_max)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if mass_high_for(n, pstar, mid) >= level:
            hi = mid
        else:
            lo = mid
    return hi


def mean_share(pi: np.ndarray) -> float:
    """Erwarteter Anteil der S-Betriebe unter pi."""
    n = len(pi) - 1
    return float(np.dot(pi, np.arange(n + 1)) / n)


def entropy_curve(n: int) -> np.ndarray:
    """Logarithmus des Binomialkoeffizienten log C(n, k) für k = 0 .. n (Zahl der Konfigurationen je Zustand)."""
    return np.array([math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) for k in range(n + 1)])
