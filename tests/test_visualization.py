"""Abbildungen: alle Achsen gesperrt, Inhalte (Farben, Achsentyp, Spuren) gegen die Daten."""

from fractions import Fraction as F

import numpy as np
import pytest

import ggw_chain as CH
import ggw_constants as C
import ggw_experiments as X
import ggw_game as G
import ggw_visualization as Z
from conftest import HIRSCH


def locked(fig):
    return all(ax.fixedrange for ax in fig.select_xaxes()) and all(ax.fixedrange for ax in fig.select_yaxes())


def _pi(n=20, x=6.0):
    return CH.stationary(CH.rates_logit(n, HIRSCH, G.beta_from_x(HIRSCH, x)))


def test_region_colors_follow_the_mass_boundaries():
    """n = 20: k = 0 bis 2 orange (fast alle H), 3 bis 17 grau, 18 bis 20 grün (fast alle S): dieselben ganzzahligen Grenzen wie ggw_chain.masses."""
    cols = Z.region_colors(20)
    assert cols[:3] == [C.COLOR_H] * 3 and cols[3:18] == [C.COLOR_MID] * 15 and cols[18:] == [C.COLOR_S] * 3


def test_distribution_figure_linear_and_logarithmic():
    pi = _pi()
    lin, log = Z.build_distribution(pi), Z.build_distribution(pi, True)
    assert locked(lin) and locked(log)
    assert list(lin.data[0].y) == pytest.approx(pi.tolist()) and lin.layout.yaxis.type is None
    assert log.layout.yaxis.type == "log" and min(log.data[0].y) >= Z.LOG_FLOOR and log.layout.yaxis.range == (-40, 0.1)


def test_distribution_figure_has_one_bar_per_state_at_the_right_share():
    fig = Z.build_distribution(_pi(8))
    assert list(fig.data[0].x) == pytest.approx([k / 8 for k in range(9)]) and len(fig.data[0].marker.color) == 9


def test_energy_entropy_figure_has_three_curves_with_maximum_zero():
    n = 20
    fig = Z.build_energy_entropy(n, 1.0, G.potential_curve(n, HIRSCH), CH.entropy_curve(n))
    assert locked(fig) and len(fig.data) == 3 and all(max(t.y) == pytest.approx(0.0, abs=1e-12) for t in fig.data)


def test_threshold_curves_skip_never_cells():
    table = {"pstars": ["1/4", "1/2"], "ns": [10, 20], "x_star": [[6.0, 5.0], [None, None]]}
    fig = Z.build_threshold_curves(table)
    assert locked(fig) and len(fig.data) == 2 and list(fig.data[0].y) == [6.0, 5.0] and len(fig.data[1].x) == 0


def test_surface_figure_marks_the_chosen_game_only_when_given():
    pstars, xs = [0.25, 0.4], [1.0, 6.0]
    m = X.e1_surface([F(1, 4), F(2, 5)], xs, 20)
    plain = Z.build_surface(pstars, xs, m, 20)
    marked = Z.build_surface(pstars, xs, m, 20, (6.0, 0.25))
    assert locked(plain) and len(plain.data) == 2 and len(marked.data) == 3 and marked.data[2].x == (6.0,) and marked.data[2].y == (0.25,)


def test_overlay_figure_has_one_trace_per_distribution():
    pis = [_pi(10), _pi(10, 2.0)]
    fig = Z.build_overlay(pis, ["a", "b"])
    assert locked(fig) and [t.name for t in fig.data] == ["a", "b"]


def test_rate_figure_has_exact_curves_the_rate_function_and_the_maxima():
    fig = Z.build_rate(6.0, 0.25, ns=(20, 40), points=500)
    assert locked(fig) and [t.name for t in fig.data] == ["exakt, n = 20", "exakt, n = 40", "Ratenfunktion g(y)", "lokale Maxima"]
    assert max(fig.data[2].y) == pytest.approx(0.0, abs=1e-12) and len(fig.data[3].x) == 1


def test_middle_mass_figure_has_a_line_per_group_size_and_the_critical_line():
    fig = Z.build_middle_mass(X.e3_middle_rows(ns=(50, 200)))
    assert locked(fig) and len(fig.data) == 2 and fig.layout.shapes[0].x0 == 4.0


def test_models_figure_is_logarithmic():
    fig = Z.build_models({"a": _pi(10), "b": _pi(10, 2.0)})
    assert locked(fig) and fig.layout.yaxis.type == "log" and len(fig.data) == 2


def test_typed_tipping_figure_draws_the_tipping_line_only_when_given():
    rows = X.e4_typed_rows()
    assert len(Z.build_typed_tipping(rows, None).data) == 3
    fig = Z.build_typed_tipping(rows, 1 / 730)
    assert locked(fig) and len(fig.data) == 4 and fig.data[3].x == (1 / 730, 1 / 730) and fig.layout.xaxis.type == "log"


def test_pure_masses_figure_groups_the_bars():
    fig = Z.build_pure_masses(["Logit", "gleichmäßig"], [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1]], ["F1", "F2", "F3"])
    assert locked(fig) and fig.layout.barmode == "group" and len(fig.data) == 2


def test_times_figure_has_two_lines_per_group_size():
    fig = Z.build_times(X.e6_time_rows(ns=(10, 20), xs=(2.0, 4.0)))
    assert locked(fig) and len(fig.data) == 4


def test_simulation_figure_shows_bars_and_points():
    exact = _pi(10)
    fig = Z.build_sim_vs_exact(exact, np.full(11, 1 / 11))
    assert locked(fig) and [t.name for t in fig.data] == ["exakt", "Simulation"]
