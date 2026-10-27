"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Alle Erwartungswerte stehen als Rechnung dabei."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Spiele (a, b, c, d): a beide S (neues Format), b beide H (altes Format), c ich S und der andere H, d ich H und der andere S.
HIRSCH = (5, 2, 1, 2)       # a - d = 3, b - c = 1: lam = 4, p* = 1/4; S auszahlungs- und risikodominant
BUENDELN = (6, 4, 0, 4)     # a - d = 2, b - c = 4: lam = 6, p* = 2/3; a > b (S auszahlungsdominant), aber b - c > a - d (H risikodominant)
GLEICHAUF = (6, 4, 2, 4)    # a - d = 2 = b - c: lam = 4, p* = 1/2
ANDERES_GEFAELLE = (3, 5, 4, 0)   # a - d = 3, b - c = 1 wie HIRSCH, aber a - b = -2 statt +3
ANTI = (-1, 2, 4, 0)        # a - d = -1, b - c = -2: Anti-Koordination
DOMINANT = (3, 1, 2, 0)     # a - d = 3 > 0, b - c = -1 < 0: S dominiert
ENTARTET = (3, 2, 1, 3)     # a = d


@pytest.fixture
def games():
    return {"hirsch": HIRSCH, "buendeln": BUENDELN, "gleichauf": GLEICHAUF, "anderes": ANDERES_GEFAELLE, "anti": ANTI, "dominant": DOMINANT, "entartet": ENTARTET}


@pytest.fixture
def mini2():
    """n = 2, Spiel (5, 2, 1, 2): Delta_0 = u_S - u_H bei j = 0 anderen S = c - b = 1 - 2 = -1; Delta_1 = a - d = 5 - 2 = 3.
    pi(1)/pi(0) = up(0)/down(0) = [sigma(-beta)] / [1/2·sigma(beta)] = 2·e^(-beta); pi(2)/pi(1) = [1/2·sigma(3·beta)] / [sigma(-3·beta)] = e^(3·beta)/2."""
    return {"game": HIRSCH, "n": 2, "delta": (-1.0, 3.0)}


@pytest.fixture
def mini3():
    """n = 3, Spiel (5, 2, 1, 2): Delta_j = (3·j - 1·(2 - j))/2 = 2·j - 1, also (-1, 1, 3). Phi(k+1) - Phi(k) = Delta_k, also Phi - Phi(0) = (0, -1, 0, 3).
    pi(k) ~ C(3, k)·exp(beta·(Phi(k) - Phi(0))) = (1, 3·e^(-beta), 3, e^(3·beta))."""
    return {"game": HIRSCH, "n": 3, "delta": (-1.0, 1.0, 3.0), "phi_rel": (0.0, -1.0, 0.0, 3.0), "binom": (1, 3, 3, 1)}
