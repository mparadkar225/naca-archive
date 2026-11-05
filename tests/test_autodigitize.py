"""Calibration and curve tracking, on synthetic scans with known truth."""
import numpy as np
import pytest

import synth
from nacaarchive import autodigitize as A
from nacaarchive import figures as F


@pytest.mark.parametrize("t,v", [(".02", 0.02), ("-.10", -0.1), ("1.2", 1.2), ("O.4", 0.4), ("l0", 10.0),
                                 ("1,5", 1.5), ("abc", None), ("", None), ("—.05", -0.05)])
def test_parse_number(t, v):
    assert A.parse_number(t) == v


@pytest.mark.parametrize("s,n", [(0.0199, 0.02), (0.1003, 0.1), (24.9, 25), (-0.0101, -0.01), (0.03, 0.03)])
def test_nice(s, n):
    assert A.nice(s) == pytest.approx(n)


def test_fit_axis_rejects_misread_label():
    lines = 100 + 50.0 * np.arange(11)                     # 11 lines, 0 .. 1.0
    labels = [(100, 0.0), (200, 0.2), (300, 0.4), (400, 6.0), (500, 0.8), (600, 1.0)]   # ".6" read as "6"
    ax = A.fit_axis(lines, labels)
    assert ax.ok and ax.inliers == 5 and ax.step == pytest.approx(0.1)
    assert ax.to_val(350) == pytest.approx(0.5) and ax.to_px(0.7) == pytest.approx(450)


def test_fit_axis_piecewise_follows_uneven_stretch():
    lines = np.array([100, 150, 200, 252, 304, 356, 406, 456.0])   # one stretched stretch
    ax = A.fit_axis(lines, [(100, 0.0), (200, 0.2), (304, 0.4), (406, 0.6)])
    assert ax.to_val(278) == pytest.approx(0.35, abs=1e-9)       # midway between lines 200..252..304


def test_fit_axis_downward_y():
    lines = 100 + 80.0 * np.arange(9)
    ax = A.fit_axis(lines, [(100, 0.16), (260, 0.12), (420, 0.08), (580, 0.04), (740, 0.0)])
    assert ax.ok and ax.step == pytest.approx(-0.02) and ax.to_val(180) == pytest.approx(0.14)


def _digitize_synthetic(seed, skew, stretch=1.004):
    chart = synth.default_chart(seed)
    img, truth = synth.render_page(chart, seed=seed, skew_deg=skew, stretch=stretch)
    a = np.asarray(img)
    ang = F.deskew_angle(a)
    a = F.rotate(a, ang)
    grids = F.find_grids(a)
    assert len(grids) == 1
    return A.digitize(a, grids[0]), chart


@pytest.mark.parametrize("seed,skew,stretch", [(0, 0.35, 1.004), (1, -0.6, 1.0), (2, 0.0, 1.012), (3, 1.0, 0.995)])
def test_synthetic_chart_accuracy(seed, skew, stretch):
    res, chart = _digitize_synthetic(seed, skew, stretch)
    assert res.x_axis.ok and res.y_axis.ok
    assert res.x_axis.step == pytest.approx(0.1) and res.y_axis.step == pytest.approx(-0.01)
    assert res.status == "auto", res.reasons
    assert len(res.curves) == len(chart.curves)
    full_scale = chart.y[1] - chart.y[0]
    for (lab, (x, y)) in chart.curves.items():
        # the digitized curve that overlaps this one best
        errs = []
        for c in res.curves:
            cx, cy = np.array(c["x"]), np.array(c["y"])
            m = (cx >= x[0]) & (cx <= x[-1])
            if m.sum() > 20:
                errs.append((np.mean(np.abs(np.interp(cx[m], x, y) - cy[m])), m.sum(), cx))
        err, n, cx = min(errs, key=lambda e: e[0])
        assert err < 0.002 * full_scale, (lab, err)
        # covers the curve's range within a grid step
        assert cx.min() < x[0] + 0.1 and cx.max() > x[-1] - 0.1, (lab, cx.min(), cx.max(), x[0], x[-1])


def test_labels_read_for_most_curves():
    res, chart = _digitize_synthetic(2, 0.0)
    labs = {c["label"] for c in res.curves} - {None}
    assert len(labs & set(chart.curves)) >= 3


def test_share_axes_between_stacked_charts():
    """Charts stacked in a column: only the lowest has x labels; the others borrow its x."""
    import numpy as np
    from nacaarchive.figures import Grid

    xs = np.arange(100.0, 1100.0, 100.0)
    xa = A.Axis(xs, np.arange(10) * 0.2, [(100.0, 0.0, True)] * 4, 4, 0.2, 0.0, "bottom")
    ya = A.Axis(np.arange(50.0, 450.0, 50.0), np.arange(8)[::-1] * 0.1, [], 4, -0.1, 0.0, "left")
    bad_x = A.Axis(xs, np.zeros(10), [], 0, 0.0, 1.0, "bottom")
    top = A.Result(Grid((100, 50, 1000, 400), xs + 2, np.arange(50.0, 450.0, 50.0)), {}, bad_x, ya,
                    [dict(label=None, px=[300.0, 500.0], py=[100.0, 150.0])], 0.9, "needs_review",
                    ["x axis not calibrated from tick labels"],
                    points=[dict(symbol="circle", n=1, px=[700.0], py=[200.0])])
    low = A.Result(Grid((100, 500, 1000, 900), xs, np.arange(500.0, 900.0, 50.0)), {}, xa, ya, [], 0.9, "auto", [])
    A.share_axes([top, low])
    assert top.status == "auto" and top.x_axis is xa
    assert top.curves[0]["x"] == pytest.approx([0.4, 0.8]) and top.points[0]["x"] == pytest.approx([1.2])
    assert any("taken from the chart" in n for n in top.grid.notes)
