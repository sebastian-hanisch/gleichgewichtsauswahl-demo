"""Szenario: Drei-Formate-Spiele, Startzustand, SplitMix64-Kopie und Referenzwerte."""

import ast
from pathlib import Path

import numpy as np
import pytest

import ggw_scenario as S
from ggw_rng import SplitMix64, uniform_block

ROOT = Path(__file__).resolve().parent.parent
RNG_SOURCE = ("ab-test-demo", "abt_rng.py")


def test_rng_copy_is_verbatim_apart_from_the_header():
    folder, original = RNG_SOURCE
    source = ROOT.parent / folder / original
    mine = (ROOT / "ggw_rng.py").read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith(f"# Wörtliche Kopie von {folder}/{original}")
    if not source.exists():
        pytest.skip("Original liegt nicht im Portfolio-Ordner")
    assert mine[1:] == source.read_text(encoding="utf-8").splitlines(), "die Kopie weicht vom Original ab: Änderungen gehören ins Original, danach neu kopieren"


def test_copied_generator_has_the_documented_stream():
    """SplitMix64 mit Seed 0: die bekannten ersten Werte der Konvention (Vigna)."""
    rng = SplitMix64(0)
    assert [rng.next() for _ in range(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_block_equals_the_scalar_stream():
    for seed in (0, 1, 35, 2 ** 63 + 5):
        rng = SplitMix64(seed)
        assert np.array_equal(uniform_block(seed, 100), np.array([rng.uniform() for _ in range(100)]))


def test_rng_module_has_no_other_imports_than_numpy_and_math():
    tree = ast.parse((ROOT / "ggw_rng.py").read_text(encoding="utf-8"))
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert names == {"math", "numpy"}


def test_start_state_rounds_half_up_with_integers():
    """n·percent/100 auf ganze Betriebe: 50 % von 5 = 2,5 -> 3; 50 % von 4 = 2; 10 % von 15 = 1,5 -> 2; 0 und 100 Prozent exakt."""
    assert S.start_state(5, 50) == 3 and S.start_state(4, 50) == 2 and S.start_state(15, 10) == 2
    assert S.start_state(7, 0) == 0 and S.start_state(7, 100) == 7 and S.start_state(1000, 33) == 330


def test_start_state_rejects_percentages_outside_zero_to_hundred():
    for bad in (-1, 101):
        with pytest.raises(ValueError):
            S.start_state(10, bad)


def test_three_format_games_are_deterministic_and_have_the_stated_structure():
    games = S.draw_three_format_games(7, 25)
    assert games == S.draw_three_format_games(7, 25) and len(games) == 25
    assert games != S.draw_three_format_games(8, 25)
    for g in games:
        sym, shift, pay = g["sym"], g["shift"], g["payoff"]
        assert all(sym[i][j] == sym[j][i] for i in range(3) for j in range(3))                                 # symmetrisch
        assert all(0 <= sym[i][j] <= 9 for i in range(3) for j in range(3)) and all(0 <= v <= 9 for v in shift)
        assert len({sym[i][i] for i in range(3)}) == 3                                                         # paarweise verschiedene Diagonale
        assert all(sym[i][i] > sym[j][i] for i in range(3) for j in range(3) if j != i)                        # strenge Gleichgewichte
        assert all(pay[i][j] == sym[i][j] + shift[j] for i in range(3) for j in range(3))


def test_three_format_games_first_values_come_from_the_stream():
    """Die erste Ziehung nimmt die ersten neun Zufallszahlen als Einträge 0 bis 9 (Obere Dreiecksmatrix wird gespiegelt)."""
    u = uniform_block(11, 9)
    raw = [int(v * 10) for v in u]
    games = S.draw_three_format_games(11, 1)
    first = games[0]
    # die erste angenommene Ziehung kann später kommen, wenn die erste verworfen wird: dann ist rejected > 0
    if first["rejected"] == 0:
        m = [raw[0:3], raw[3:6], raw[6:9]]
        assert first["sym"] == [[m[min(i, j)][max(i, j)] for j in range(3)] for i in range(3)]


def test_rejected_draws_are_counted():
    games = S.draw_three_format_games(3, 30)
    counts = [g["rejected"] for g in games]
    assert counts == sorted(counts) and counts[-1] > 0


def test_potential_and_payoff_dominant_format():
    g = {"sym": [[9, 2, 1], [2, 8, 3], [1, 3, 7]], "shift": [0, 5, 5], "payoff": [[9, 7, 6], [2, 13, 8], [1, 8, 12]]}
    assert S.potential_format(g) == 0                      # größte Diagonale von Sym: 9
    assert S.payoff_dominant_format(g) == 1                # Diagonale von A: 9, 13, 12


def test_payoff_dominant_format_is_none_on_a_tie_at_the_top():
    g = {"sym": [[9, 0, 0], [0, 8, 0], [0, 0, 7]], "payoff": [[9, 0, 0], [0, 9, 0], [0, 0, 5]]}
    assert S.payoff_dominant_format(g) is None
