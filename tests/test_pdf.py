"""PDF-Export: gültiges PDF, Umlaute und Sonderzeichen, Latin-1-Ersatz."""

import ggw_constants as C
import ggw_evaluation as V
import ggw_pdf_export as PDF


def _view(**kw):
    return V.analyse_main(V.Settings(**{**C.BASE, **kw}))


def test_pdf_is_a_pdf_for_every_game_class():
    for game in ((5, 2, 1, 2), (-1, 2, 4, 0), (3, 1, 2, 0), (2, 2, 2, 2)):
        v = _view(a=game[0], b=game[1], c=game[2], d=game[3])
        data = PDF.build_pdf(v, [V.verdict_main(v)])
        assert data[:5] == b"%PDF-" and len(data) > 1000


def test_latin_replaces_special_characters_and_removes_markdown():
    text = PDF._latin("**Konvention** β·λ ≥ 4 → „ja“ – ε ≈ 10⁻")
    assert "**" not in text and "beta*lambda >= 4 -> \"ja\" - eps ~ 10" in text
    assert PDF._latin("Größe über Äpfel") == "Größe über Äpfel"


def test_pdf_with_all_verdict_kinds():
    v = _view()
    verdicts = [V.verdict_main(v), V.verdict_threshold_for(v), ("warning", "Warnung mit ≤ und Δ"), ("info", "Info")]
    assert PDF.build_pdf(v, verdicts)[:5] == b"%PDF-"
