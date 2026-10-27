"""Die vorgerechnete Schwellentabelle (data/threshold_table.json) gehört zum aktuellen Rechencode.

Eigene Datei, weil jede Änderung an ggw_game.py oder ggw_chain.py den Erzeuger-Hash ändert: bei einer Mutationsprüfung würde dieser Test jeden Mutanten fangen, ohne etwas
über die Rechnung zu sagen (die Mutationsprüfung lässt diese Datei aus, siehe README)."""

import json

import pytest

import ggw_experiments as X


def test_threshold_table_file_matches_the_generator_hash_and_the_live_computation():
    """Die vorgerechnete Tabelle gehört zum aktuellen Rechencode (Hash) und stimmt Zelle für Zelle mit der Neuberechnung überein."""
    stored = json.loads(X.DATA_FILE.read_text(encoding="utf-8"))
    assert stored["generator_hash"] == X.generator_hash(), "ggw_game.py/ggw_chain.py wurde geändert: python tools/make_threshold_table.py ausführen"
    live = X.compute_threshold_table()
    assert stored["pstars"] == live["pstars"] and stored["ns"] == live["ns"] and stored["level"] == live["level"]
    for row_s, row_l in zip(stored["x_star"], live["x_star"]):
        assert row_s == pytest.approx(row_l, abs=1e-9)


def test_load_threshold_table_reads_the_file_when_the_hash_fits():
    assert X.load_threshold_table() == json.loads(X.DATA_FILE.read_text(encoding="utf-8"))
