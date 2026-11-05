"""Data-point symbols: found, grouped by shape and read as values (synthetic, known truth)."""
import numpy as np
import pytest

import synth
from nacaarchive import autodigitize as ad
from nacaarchive import figures as F


def symbol_chart(seed=0, ink="k"):
    rng = np.random.default_rng(seed)
    c = synth.Chart(grid_color="0.1", labels=False, ink=ink)
    for k, marker in enumerate(("o", "s", "^")):
        j = np.linspace(0.3 + 0.25 * k, 0.9 + 0.3 * k, 60)
        f = lambda x: (0.05 + 0.03 * k) * (1 - ((x - j[0]) / (j[-1] - j[0] + 0.15)) ** 2)
        c.curves[f"c{k}"] = (j, f(j))
        xs = np.sort(rng.uniform(j[0] + 0.02, j[-1] - 0.02, 8))
        c.points[marker] = (xs, f(xs) + rng.normal(0, 0.0015, 8), marker)
    return c


@pytest.mark.parametrize("seed,skew", [(0, 0.35), (1, -0.6)])
def test_symbol_tables(seed, skew):
    chart = symbol_chart(seed)
    page, truth = synth.render_page(chart, seed=seed, skew_deg=skew)
    img = np.asarray(page)
    img = F.rotate(img, F.deskew_angle(img))
    g = F.find_grids(img)[0]
    res = ad.digitize(img, g)
    assert res.x_axis.ok and res.y_axis.ok
    names = {p["symbol"]: p for p in res.points}
    want = {"o": "circle with dot", "s": "square with dot", "^": "triangle up with dot"}
    for marker, name in want.items():
        assert name in names, (name, list(names))
        got, (tx, ty, _) = names[name], chart.points[marker]
        assert got["n"] >= 5          # random points: two can overlap and be skipped
        for x, y in zip(got["x"], got["y"]):
            i = np.argmin(np.hypot((tx - x) / 1.6, (ty - y) / 0.16))     # nearest in chart scale
            assert abs(tx[i] - x) < 0.006 * 1.6 and abs(ty[i] - y) < 0.006 * 0.16, (name, x, y, tx[i], ty[i])
    # odd ones out (a symbol overlapping another) stay few
    assert 18 <= sum(p["n"] for p in res.points) <= 26
    assert res.status == "auto", res.reasons
    assert len(res.curves) >= 3            # faired curves are traced too (best effort)


def test_faded_pencil_symbols_are_found():
    """A faded scan whose pencilled symbols are lighter than the inked grid and whose darkest
    ink is mid-gray (WR L-177): the points are found where they are. Telling circles from
    squares at this contrast is not asked for here."""
    chart = symbol_chart(2, ink="0.3")
    page, truth = synth.render_page(chart, seed=2, skew_deg=0.2)
    img = (255 - (255 - np.asarray(page).astype(float)) * 0.85).astype(np.uint8)
    img = F.rotate(img, F.deskew_angle(img))
    res = ad.digitize(img, F.find_grids(img)[0])
    tx = np.concatenate([p[0] for p in chart.points.values()])
    ty = np.concatenate([p[1] for p in chart.points.values()])
    got = [(x, y) for p in res.points for x, y in zip(p["x"], p["y"])]
    assert len(got) >= 18
    for x, y in got:
        i = np.argmin(np.hypot((tx - x) / 1.6, (ty - y) / 0.16))
        assert abs(tx[i] - x) < 0.006 * 1.6 and abs(ty[i] - y) < 0.006 * 0.16, (x, y, tx[i], ty[i])
