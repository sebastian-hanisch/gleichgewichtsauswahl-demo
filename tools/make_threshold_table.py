"""Rechnet die Schwellentabelle x*(p*, n) neu und schreibt data/threshold_table.json samt Erzeuger-Hash.

Aufruf (im Demo-Ordner): python tools/make_threshold_table.py
Ändert sich ggw_game.py oder ggw_chain.py, stimmt der Hash nicht mehr (tests/test_experiments.py schlägt an, die App rechnet die Tabelle dann live neu);
danach diesen Aufruf wiederholen."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ggw_experiments as X  # noqa: E402

if __name__ == "__main__":
    table = X.write_threshold_table()
    print(f"geschrieben: {X.DATA_FILE}  (Hash {table['generator_hash']}, {len(table['pstars'])} x {len(table['ns'])} Zellen)")
