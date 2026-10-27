"""Einzelspieler-Simulation und mittlere Austrittszeiten (Kann man die stationäre Verteilung sehen?).

Zeit wird in Aktualisierungen JE BETRIEB gemessen (Einzelaktualisierungen geteilt durch n). `simulate_players` führt die Betriebe einzeln: jede Aktualisierung zieht einen Betrieb,
rechnet seinen Vorteil aus der Zahl der anderen S-Betriebe und wählt per Logit. `log10_hitting_time` rechnet die mittlere Zeit der Geburts-Sterbe-Kette im Logarithmus
(die Zeiten reichen bis 10^73 und lassen sich nicht ausschreiben); `escape_log_time` wendet sie auf den Austritt aus „alle H“ bzw. „alle S“ bis zur Barriere k* an."""

from __future__ import annotations

import math

import numpy as np

import ggw_chain as CH
import ggw_game as G
from ggw_rng import uniform_block


def simulate_players(n: int, game, beta: float, steps: int, burn: int, start_k: int, seed: int) -> np.ndarray:
    """Relative Häufigkeit jedes Zustands k = 0 .. n nach `burn` Einschwingschritten; Start mit start_k S-Betrieben (die ersten start_k Betriebe), Zufall aus SplitMix64(seed).
    Je Schritt zwei Zahlen: Betrieb und Wahl."""
    if not 0 <= start_k <= n:
        raise ValueError(f"start_k = {start_k} liegt außerhalb von 0 bis {n}")
    if steps <= burn:
        raise ValueError("steps muss größer sein als burn")
    gap_s, gap_h = G.gaps(game)
    gap_s, gap_h = float(gap_s), float(gap_h)
    u = uniform_block(seed, 2 * steps)
    who = (u[0::2] * n).astype(np.int64).tolist()
    coin = u[1::2].tolist()
    state = [1] * start_k + [0] * (n - start_k)
    k = start_k
    hist = [0] * (n + 1)
    exp = math.exp
    denom = n - 1
    for t in range(steps):
        i = who[t]
        j = k - state[i]
        z = beta * (gap_s * j - gap_h * (denom - j)) / denom
        p_s = 1.0 / (1.0 + exp(-z)) if z > -700 else 0.0
        new = 1 if coin[t] < p_s else 0
        k += new - state[i]
        state[i] = new
        if t >= burn:
            hist[k] += 1
    h = np.array(hist, dtype=float)
    return h / h.sum()


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """Gesamtvariationsabstand 0,5·Summe|p - q|."""
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def _cumulative_logsumexp(v: np.ndarray) -> np.ndarray:
    return np.logaddexp.accumulate(v)


def log10_hitting_time(rates: CH.Rates, target: int) -> float:
    """log10 der mittleren Zahl der Einzelaktualisierungen, bis die Kette von Zustand 0 aus (dort reflektierend) erstmals den Zustand `target` erreicht.
    E = Summe_{k<target} (1/(pi_k·up_k))·Summe_{j<=k} pi_j mit pi_j ~ Produkt up/down, im Logarithmus aufsummiert."""
    if not 1 <= target <= rates.n:
        raise ValueError(f"target = {target} liegt außerhalb von 1 bis {rates.n}")
    log_pi = np.concatenate([[0.0], np.cumsum(rates.log_up[: target - 1] - rates.log_down[: target - 1])])   # Länge target, Zustände 0 .. target - 1
    cum = _cumulative_logsumexp(log_pi)
    terms = cum - log_pi - rates.log_up[:target]
    return float(np.logaddexp.reduce(terms)) / math.log(10)


def reverse(rates: CH.Rates) -> CH.Rates:
    """Spiegelung k -> n - k: die Kette von „alle S“ aus gesehen."""
    return CH.Rates(rates.n, rates.log_down[::-1].copy(), rates.log_up[::-1].copy())


def escape_log_time(rates: CH.Rates, game, from_side: str) -> float | None:
    """log10 der mittleren Zeit in Aktualisierungen JE BETRIEB, bis die Kette von „alle H“ (from_side = 'H') bzw. „alle S“ ('S') erstmals die Barriere k* (erster Zustand mit Delta > 0) erreicht;
    None, wenn es keine Barriere gibt (S nie die beste Antwort) oder k* = 0."""
    n = rates.n
    kstar = G.barrier_state(n, game)
    if kstar is None or kstar == 0:
        return None
    if from_side == "H":
        return log10_hitting_time(rates, kstar) - math.log10(n)
    if from_side == "S":
        return log10_hitting_time(reverse(rates), n - kstar) - math.log10(n)
    raise ValueError(f"unbekannte Seite: {from_side}")


def simulate_hitting(n: int, game, beta: float, from_side: str, runs: int, seed: int, cap_steps: int) -> tuple[list[float], int]:
    """Einzelspieler-Simulation: Zeiten (Aktualisierungen je Betrieb) bis zur Barriere k*, ausgehend von „alle H“ bzw. „alle S“, je Lauf mit Obergrenze `cap_steps` Einzelaktualisierungen.
    Gibt (Zeiten der beendeten Läufe, Zahl der abgebrochenen Läufe): abgebrochene Läufe zählen nicht als Zeit (nie still mitteln)."""
    if from_side not in ("H", "S"):
        raise ValueError(f"unbekannte Seite: {from_side}")
    kstar = G.barrier_state(n, game)
    if kstar is None or kstar == 0:
        raise ValueError("keine Barriere")
    gap_s, gap_h = (float(v) for v in G.gaps(game))
    block = 65536
    times, capped = [], 0
    exp = math.exp
    denom = n - 1
    for run in range(runs):
        k = 0 if from_side == "H" else n
        state = [0] * n if from_side == "H" else [1] * n
        steps = 0
        pos = block
        who = coin = None
        while steps < cap_steps and (k < kstar if from_side == "H" else k > kstar):
            if pos == block:
                u = uniform_block(seed + 1_000_003 * run + steps, 2 * block)
                who = (u[0::2] * n).astype(np.int64).tolist()
                coin = u[1::2].tolist()
                pos = 0
            i = who[pos]
            j = k - state[i]
            z = beta * (gap_s * j - gap_h * (denom - j)) / denom
            p_s = 1.0 / (1.0 + exp(-z)) if z > -700 else 0.0
            new = 1 if coin[pos] < p_s else 0
            k += new - state[i]
            state[i] = new
            pos += 1
            steps += 1
        if (k < kstar if from_side == "H" else k > kstar):
            capped += 1
        else:
            times.append(steps / n)
    return times, capped
