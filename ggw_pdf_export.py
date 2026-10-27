"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Spiel, Massen und die Verdikte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

from fpdf import FPDF

import ggw_evaluation as V

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "≠": "!=", "√": "Wurzel", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"', "∞": "unendlich",
            "→": "->", "ε": "eps", "β": "beta", "λ": "lambda", "σ": "sigma", "π": "pi", "Φ": "Phi", "Δ": "Delta", "×": "x", "₁": "1", "₂": "2", "₁₀": "10", "↔": "<->", "≈": "~", "∝": "~"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1: Sonderzeichen ersetzen, Markdown-Sterne entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def build_pdf(view: V.MainView, verdicts: list[tuple[str, str]]) -> bytes:
    s, m = view.settings, view.masses
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Welche Konvention überlebt? Auswahl durch Rauschen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    ps = "-" if view.p_star is None else V.frac(view.p_star)
    pdf.multi_cell(0, 5, _latin(f"Spiel (a, b, c, d) = {s.game}, Klasse {view.klass}, p* = {ps}, Skala lambda = {view.scale}, n = {s.n}, Schärfe x = {V.de(s.x, 2)} (beta = {V.de(view.beta, 3)} je Minute), "
                                f"Wahlregel {s.model}, Zufalls-Seed {s.seed}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Masse der stationären Verteilung", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _latin(f"fast alle S (mindestens 90 % der Betriebe): {V.pct(m.high)}; fast alle H (höchstens 10 %): {V.pct(m.low)}; Mitte: {V.pct(m.mid)}; erwarteter Anteil S: {V.pct(view.mean)}."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Verdikte", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Eigennutz und Lernregel (Logit) sind Modellannahmen, keine Aussagen über Menschen oder Terminals. Auszahlungen: Zeitersparnis je Sendung in Minuten, gewählt, nicht gemessen."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
