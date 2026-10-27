"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import plotly.graph_objects as go

import ggw_chain as CH
import ggw_constants as C
import ggw_game as G
import ggw_limit as LI

LOG_FLOOR = 1e-40


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=70 if legend else 10), plot_bgcolor="rgba(0,0,0,0)", showlegend=legend, legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def region_colors(n: int) -> list[str]:
    """Farbe je Zustand k: fast alle S (10·k >= 9·n), fast alle H (10·k <= n), Mitte (dieselben ganzzahligen Grenzen wie `ggw_chain.masses`)."""
    return [C.COLOR_S if 10 * k >= 9 * n else C.COLOR_H if 10 * k <= n else C.COLOR_MID for k in range(n + 1)]


# ---------------------------------------------------------------------------------------------------------------------------- Hauptansicht


def build_distribution(pi: np.ndarray, log_scale: bool = False) -> go.Figure:
    """Stationäre Verteilung über den Anteil der S-Betriebe k/n; grün: fast alle S, orange: fast alle H, grau: Mitte. Mit Logarithmusachse (von 10^-40 bis 1) sieht man auch kleine Massen."""
    n = len(pi) - 1
    y = np.maximum(pi, LOG_FLOOR) if log_scale else pi
    fig = go.Figure(go.Bar(x=np.arange(n + 1) / n, y=y, marker=dict(color=region_colors(n), line=dict(width=0)), hovertemplate="Anteil %{x:.3f}: Masse %{y:.3g}<extra></extra>"))
    fig.update_xaxes(title_text="Anteil der Betriebe mit dem neuen Format S", range=[-0.5 / n, 1 + 0.5 / n])
    if log_scale:
        fig.update_yaxes(title_text="Masse (logarithmisch)", type="log", range=[-40, 0.1])
    else:
        fig.update_yaxes(title_text="Masse", range=[0, max(0.02, float(pi.max()) * 1.08)])
    return _base(fig, legend=False)


def build_energy_entropy(n: int, beta: float, potential: np.ndarray, log_binom: np.ndarray) -> go.Figure:
    """log π(k) = β·Φ(k) + log C(n, k) + const: Potential und Zahl der Konfigurationen getrennt (jeweils auf den Höchstwert 0 verschoben)."""
    share = np.arange(n + 1) / n
    energy = beta * potential
    total = energy + log_binom
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=share, y=energy - energy.max(), mode="lines", name="β·Φ(k): je Konfiguration", line=dict(color=C.COLOR_LINE, width=3)))
    fig.add_trace(go.Scatter(x=share, y=log_binom - log_binom.max(), mode="lines", name="log C(n, k): Zahl der Konfigurationen", line=dict(color=C.COLOR_MID, width=3, dash="dash")))
    fig.add_trace(go.Scatter(x=share, y=total - total.max(), mode="lines", name="Summe: log der Masse des Zustands", line=dict(color=C.COLOR_RED, width=3)))
    fig.update_xaxes(title_text="Anteil der Betriebe mit S", range=[0, 1])
    fig.update_yaxes(title_text="Logarithmus, auf den Höchstwert verschoben")
    return _base(fig)


# ---------------------------------------------------------------------------------------------------------------------------- E1


def build_threshold_curves(table: dict) -> go.Figure:
    """Schwelle x*(p*, n) gegen die Gruppengröße, eine Linie je p*; „nie“ (None) fehlt."""
    fig = go.Figure()
    for p, row in zip(table["pstars"], table["x_star"]):
        pts = [(n, v) for n, v in zip(table["ns"], row) if v is not None]
        fig.add_trace(go.Scatter(x=[n for n, _ in pts], y=[v for _, v in pts], mode="lines+markers", name=f"p* = {p}", hovertemplate="n = %{x}: x* = %{y:.2f}<extra>p* = " + p + "</extra>"))
    fig.update_xaxes(title_text="Zahl der Betriebe n", type="log")
    fig.update_yaxes(title_text="Schwelle x* (Masse fast alle S ≥ 99 %)")
    return _base(fig)


def build_surface(pstars: list[float], xs: list[float], matrix: np.ndarray, n: int, mark: tuple[float, float] | None = None) -> go.Figure:
    """Masse „fast alle S“ über (Schärfe x, Schwelle p*) für die Gruppengröße n; die weiße Linie begrenzt 99 %; Raute: gewähltes Spiel."""
    fig = go.Figure(go.Heatmap(x=xs, y=pstars, z=matrix, zmin=0, zmax=1, colorscale="Viridis", colorbar=dict(title=dict(text="Masse", side="top")), hovertemplate="x = %{x:.2f}, p* = %{y:.3f}: %{z:.3f}<extra></extra>"))
    fig.add_trace(go.Contour(x=xs, y=pstars, z=matrix, contours=dict(start=C.THRESHOLD_LEVEL, end=C.THRESHOLD_LEVEL, size=1, coloring="none"), line=dict(color="white", width=2), showscale=False, hoverinfo="skip"))
    if mark is not None:
        fig.add_trace(go.Scatter(x=[mark[0]], y=[mark[1]], mode="markers", marker=dict(symbol="diamond", size=14, color=C.COLOR_RED, line=dict(color="white", width=1)), showlegend=False, hovertemplate="gewählt<extra></extra>"))
    fig.update_xaxes(title_text="Schärfe x = β·λ")
    fig.update_yaxes(title_text="Schwelle p*")
    return _base(fig, legend=False)


# ---------------------------------------------------------------------------------------------------------------------------- E2


def build_overlay(distributions: list[np.ndarray], labels: list[str]) -> go.Figure:
    """Mehrere Verteilungen übereinander: bei gleichem (λ, p*) liegen sie aufeinander."""
    fig = go.Figure()
    n = len(distributions[0]) - 1
    for i, (pi, label) in enumerate(zip(distributions, labels)):
        fig.add_trace(go.Scatter(x=np.arange(n + 1) / n, y=pi, mode="lines", name=label, line=dict(width=6 - i if i < 5 else 1, dash="solid" if i == 0 else "dot")))
    fig.update_xaxes(title_text="Anteil der Betriebe mit S", range=[0, 1])
    fig.update_yaxes(title_text="Masse")
    return _base(fig)


# ---------------------------------------------------------------------------------------------------------------------------- E3


def build_rate(x: float, pstar: float, ns: tuple[int, ...] = (50, 200, 1000), points: int = 2000) -> go.Figure:
    """Ratenfunktion g(y) (schwarz, gestrichelt) und die Größe (log π(k) − Höchstwert)/n der exakten Verteilung für wachsendes n; Dreiecke: lokale Maxima von g."""
    y = LI.cell_midpoints(points)
    g = LI.rate_function(y, x, pstar)
    fig = go.Figure()
    for n in ns:
        pi = CH.stationary(CH.rates_logit(n, G.game_from(1, Fraction(pstar).limit_denominator(1000)), x))
        with np.errstate(divide="ignore"):
            lp = np.log(np.maximum(pi, 1e-300))
        fig.add_trace(go.Scatter(x=np.arange(n + 1) / n, y=(lp - lp.max()) / n, mode="lines", name=f"exakt, n = {n}"))
    fig.add_trace(go.Scatter(x=y, y=g - g.max(), mode="lines", name="Ratenfunktion g(y)", line=dict(color="black", dash="dash", width=2)))
    maxima = LI.local_maxima(x, pstar)
    fig.add_trace(go.Scatter(x=[m[0] for m in maxima], y=[m[1] - g.max() for m in maxima], mode="markers", name="lokale Maxima", marker=dict(symbol="triangle-up", size=12, color=C.COLOR_RED)))
    fig.update_xaxes(title_text="Anteil der Betriebe mit S", range=[0, 1])
    fig.update_yaxes(title_text="log π(k) / n, auf den Höchstwert verschoben", range=[-0.35, 0.02])
    return _base(fig)


def build_middle_mass(rows: list[dict]) -> go.Figure:
    """p* = 1/2: Masse in der Mitte gegen die Schärfe, eine Linie je Gruppengröße; senkrecht der kritische Wert."""
    fig = go.Figure()
    for n in sorted({r["n"] for r in rows}):
        sel = [r for r in rows if r["n"] == n]
        fig.add_trace(go.Scatter(x=[r["x"] for r in sel], y=[r["mid"] for r in sel], mode="lines+markers", name=f"n = {n}"))
    fig.add_vline(x=LI.CRITICAL_X, line=dict(color=C.COLOR_RED, dash="dash"), annotation_text="kritischer Wert 4", annotation_position="top right")
    fig.update_xaxes(title_text="Schärfe x = β·λ")
    fig.update_yaxes(title_text="Masse in der Mitte (0,4·n bis 0,6·n)", range=[-0.02, 1.02])
    return _base(fig)


# ---------------------------------------------------------------------------------------------------------------------------- E4


def build_models(pis: dict[str, np.ndarray]) -> go.Figure:
    """Die Verteilungen der drei Wahlregeln für dasselbe Spiel (logarithmisch)."""
    fig = go.Figure()
    for name, pi in pis.items():
        n = len(pi) - 1
        fig.add_trace(go.Scatter(x=np.arange(n + 1) / n, y=np.maximum(pi, LOG_FLOOR), mode="lines", name=name, line=dict(width=3)))
    fig.update_xaxes(title_text="Anteil der Betriebe mit S", range=[0, 1])
    fig.update_yaxes(title_text="Masse (logarithmisch)", type="log", range=[-40, 0.1])
    return _base(fig)


def build_typed_tipping(rows: list[dict], tipping: float | None) -> go.Figure:
    """Typabhängige Fehler: Massen gegen ε_H (logarithmisch); gestrichelt die gezählte Kippschwelle."""
    fig = go.Figure()
    xs = [r["eps_h"] for r in rows]
    fig.add_trace(go.Scatter(x=xs, y=[r["high"] for r in rows], mode="lines+markers", name="fast alle S", line=dict(color=C.COLOR_S)))
    fig.add_trace(go.Scatter(x=xs, y=[r["low"] for r in rows], mode="lines+markers", name="fast alle H", line=dict(color=C.COLOR_H)))
    fig.add_trace(go.Scatter(x=xs, y=[r["mid"] for r in rows], mode="lines+markers", name="Mitte", line=dict(color=C.COLOR_MID)))
    if tipping is not None:
        fig.add_trace(go.Scatter(x=[tipping, tipping], y=[0, 1], mode="lines", name="gezählte Kippschwelle", line=dict(color=C.COLOR_RED, dash="dash")))
    fig.update_xaxes(title_text="Fehlerrate ε_H (H ist beste Antwort, S wird trotzdem gewählt)", type="log", autorange="reversed")
    fig.update_yaxes(title_text="Masse", range=[-0.02, 1.02])
    return _base(fig)


# ---------------------------------------------------------------------------------------------------------------------------- E5


def build_pure_masses(labels: list[str], values: list[list[float]], names: list[str]) -> go.Figure:
    """Masse der drei reinen Zustände (alle wählen Format i) je Wahlregel."""
    fig = go.Figure()
    for label, vals in zip(labels, values):
        fig.add_trace(go.Bar(x=names, y=vals, name=label))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Masse", range=[0, 1.02])
    return _base(fig)


# ---------------------------------------------------------------------------------------------------------------------------- E6


def build_times(rows: list[dict]) -> go.Figure:
    """log10 der mittleren Zeit bis zur Barriere (Aktualisierungen je Betrieb) gegen die Schärfe; durchgezogen: Austritt aus „alle S“, gestrichelt: aus „alle H“."""
    fig = go.Figure()
    for n in sorted({r["n"] for r in rows}):
        sel = [r for r in rows if r["n"] == n]
        fig.add_trace(go.Scatter(x=[r["x"] for r in sel], y=[r["from_s"] for r in sel], mode="lines+markers", name=f"aus alle S, n = {n}"))
        fig.add_trace(go.Scatter(x=[r["x"] for r in sel], y=[r["from_h"] for r in sel], mode="lines+markers", name=f"aus alle H, n = {n}", line=dict(dash="dash")))
    fig.update_xaxes(title_text="Schärfe x = β·λ")
    fig.update_yaxes(title_text="log₁₀ der Zeit (Aktualisierungen je Betrieb)")
    return _base(fig, height=380)


def build_sim_vs_exact(exact: np.ndarray, sim: np.ndarray) -> go.Figure:
    """Exakte Verteilung (Balken) gegen die Häufigkeiten der Simulation (Punkte)."""
    n = len(exact) - 1
    share = np.arange(n + 1) / n
    fig = go.Figure()
    fig.add_trace(go.Bar(x=share, y=exact, name="exakt", marker=dict(color=C.COLOR_MID, line=dict(width=0))))
    fig.add_trace(go.Scatter(x=share, y=sim, mode="markers", name="Simulation", marker=dict(color=C.COLOR_RED, size=7)))
    fig.update_xaxes(title_text="Anteil der Betriebe mit S", range=[-0.5 / n, 1 + 0.5 / n])
    fig.update_yaxes(title_text="Masse", range=[0, max(0.02, float(max(exact.max(), sim.max())) * 1.08)])
    return _base(fig)
