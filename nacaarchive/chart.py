"""Digitize curves from scanned charts (used for NACA Report 747's figures).

    g = Chart.from_image("page.png", box=(x0, y0, x1, y1))       # pixel box around the grid
    g.calibrate(x=(x_first_line, x_step), y=(y_first_line, y_step))  # values of grid lines
    xs, ys = g.trace(seed=(1.2, 0.10), x_range=(0.4, 1.6))         # follow one curve
    g.check("out.png", curves)                                     # overlay for inspection

Grid lines are found as pixel rows/columns that are dark over most of the box; values are
assigned to them from the first line's value and the line spacing, so calibration uses
every line (not two hand-picked points) and scan skew/stretch shows up as a fit residual.
Curves are traced along dark pixels with the grid lines masked out, following the slope
from one column (or row) to the next, which keeps a trace on its own curve where curves
cross at shallow angles. Every digitized curve is drawn back onto the scan for checking.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


def _lines(profile: np.ndarray, frac: float, min_gap: int = 6) -> np.ndarray:
    """Centers of runs where profile > frac (grid lines), merging runs closer than min_gap."""
    on = profile > frac
    idx = np.flatnonzero(on)
    if idx.size == 0:
        return idx
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > min_gap) + 1)
    return np.array([g.mean() for g in groups])


@dataclass
class Chart:
    img: np.ndarray                 # grayscale, 0 = black
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 of the plot area in img

    @classmethod
    def from_image(cls, path, box, ink: int = 140, grid: int = 215, grid_frac: float = 0.45):
        """ink: gray level below which a pixel is curve ink; grid: the (lighter) level used
        to find grid lines, which on these scans are mid-gray and thinner than curves."""
        a = np.asarray(Image.open(path).convert("L"), dtype=np.int16)
        c = cls(a, tuple(int(v) for v in box))
        c.dark = a < ink
        c.gridpix = a < grid
        c.find_grid(grid_frac)
        return c

    # ------------------------------------------------------------------ grid
    def find_grid(self, frac: float = 0.45):
        x0, y0, x1, y1 = self.box
        d = self.gridpix[y0:y1, x0:x1]
        self.vlines = _lines(d.mean(axis=0), frac) + x0
        self.hlines = _lines(d.mean(axis=1), frac) + y0
        # mask: grid-line pixels (+-2) inside the box
        m = np.zeros_like(self.dark)
        for v in self.vlines:
            m[y0:y1, max(int(round(v)) - 2, 0): int(round(v)) + 3] = True
        for h in self.hlines:
            m[max(int(round(h)) - 2, 0): int(round(h)) + 3, x0:x1] = True
        self.gridmask = m
        self.ink = self.dark & ~m

    def calibrate(self, x, y, xlines=None, ylines=None):
        """x = (value, step[, anchor]) for the vertical grid lines, y likewise for the
        horizontal ones (top to bottom); anchor 'first' (default) or 'last' says which line
        has that value. Each detected line gets its grid index from the median spacing, so
        a missed or spurious line doesn't shift the rest (spurious ones far from an
        integer index are dropped). Returns the largest fit residuals (x, y)."""
        def fit(lines, spec):
            value, step = spec[0], spec[1]
            anchor = spec[2] if len(spec) > 2 else "first"
            lines = np.asarray(lines, float)
            gap = float(np.median(np.diff(lines)))
            ref = lines[0] if anchor == "first" else lines[-1]
            k = (lines - ref) / gap
            good = np.abs(k - np.round(k)) < 0.25
            k = np.round(k[good])
            vals = value + step * k
            coef = np.polyfit(lines[good], vals, 1)
            return coef, float(np.max(np.abs(np.polyval(coef, lines[good]) - vals)))
        vl = self.vlines if xlines is None else self.vlines[xlines]
        hl = self.hlines if ylines is None else self.hlines[ylines]
        self.px, self.xres = fit(vl, x)
        self.py, self.yres = fit(hl, y)
        return self.xres, self.yres

    def to_px(self, xv, yv):
        return ((np.asarray(xv) - self.px[1]) / self.px[0], (np.asarray(yv) - self.py[1]) / self.py[0])

    def to_val(self, xp, yp):
        return np.polyval(self.px, xp), np.polyval(self.py, yp)

    # ----------------------------------------------------------------- trace
    def _runs(self, line: np.ndarray):
        idx = np.flatnonzero(line)
        if idx.size == 0:
            return []
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        return [(g.mean(), len(g)) for g in groups]

    def trace(self, seed, x_range, step_px: int = 2, window: float = 12.0, along: str = "x",
              max_gap: int = 40):
        """Follow the curve through the seed point (data units) across x_range (or y_range
        when along='y', for steep curves). Returns (x, y) arrays in data units, sorted."""
        sx, sy = self.to_px(*seed)
        sx, sy = float(sx), float(sy)
        if along == "x":
            ends = [float(self.to_px(v, seed[1])[0]) for v in x_range]
        else:
            ends = [float(self.to_px(seed[0], v)[1]) for v in x_range]
        lo, hi = min(ends), max(ends)
        out = []
        for direction in (+1, -1):
            pos, cross = (sx, sy) if along == "x" else (sy, sx)
            slope = 0.0
            last_hit = pos
            hist = []
            while lo <= pos <= hi:
                p = int(round(pos))
                if along == "x":
                    line = self.ink[:, p]
                else:
                    line = self.ink[p, :]
                pred = cross + slope * (step_px * direction if hist else 0.0)
                runs = [(c, n) for c, n in self._runs(line[int(max(pred - window, 0)): int(pred + window) + 1])]
                runs = [(c + int(max(pred - window, 0)), n) for c, n in runs]
                if runs:
                    c = min(runs, key=lambda r: abs(r[0] - pred))[0]
                    if hist:
                        new_slope = (c - cross) / (step_px * direction)
                        slope = 0.6 * slope + 0.4 * new_slope if len(hist) > 2 else new_slope
                    cross = c
                    last_hit = pos
                    hist.append((pos, cross))
                else:
                    cross = pred                        # grid line or gap: coast
                    if abs(pos - last_hit) > max_gap:
                        break
                pos += step_px * direction
            out += hist
        out = sorted(set(out))
        if not out:
            return np.array([]), np.array([])
        a = np.array(out)
        if along == "x":
            xv, yv = self.to_val(a[:, 0], a[:, 1])
        else:
            xv, yv = self.to_val(a[:, 1], a[:, 0])
        order = np.argsort(xv if along == "x" else yv)
        return xv[order], yv[order]

    # ----------------------------------------------------------------- check
    def check(self, path, curves, crop=True):
        im = Image.fromarray(np.clip(self.img, 0, 255).astype(np.uint8)).convert("RGB")
        d = ImageDraw.Draw(im)
        colors = [(230, 0, 0), (0, 150, 0), (0, 0, 230), (200, 0, 200), (0, 170, 170), (230, 120, 0)]
        for k, (xv, yv) in enumerate(curves):
            xp, yp = self.to_px(xv, yv)
            pts = list(zip(xp.tolist(), yp.tolist()))
            d.line(pts, fill=colors[k % len(colors)], width=3)
        if crop:
            x0, y0, x1, y1 = self.box
            im = im.crop((x0 - 60, y0 - 40, x1 + 60, y1 + 60))
        im.save(path)


def crossings(chart: Chart, y_value: float, x_range=None, min_sep_px: int = 8):
    """x values (data units) where curves cross the horizontal line y = y_value, left to
    right (grid lines masked)."""
    _, yp = chart.to_px(0.0, y_value)
    row = chart.ink[int(round(float(yp))), :]
    x0, _, x1, _ = chart.box
    row = row.copy()
    row[:x0] = False
    row[x1:] = False
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return np.array([])
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > min_sep_px) + 1)
    xs = np.array([chart.to_val(g.mean(), 0.0)[0] for g in groups])
    if x_range is not None:
        xs = xs[(xs >= x_range[0]) & (xs <= x_range[1])]
    return xs


def family(chart: Chart, labels, y_probe: float, y_max: float, x_min: float = 0.0,
           x_max: float | None = None, **trace_kw):
    """Trace a family of roughly parallel curves (one per label, left to right where they
    cross y = y_probe), each kept between x_min and where it reaches y_max going left,
    down to y = 0. Returns {label: (x, y)}."""
    xs = crossings(chart, y_probe)
    if len(xs) < len(labels):
        raise ValueError(f"found {len(xs)} crossings at y={y_probe}, need {len(labels)}: {xs}")
    out = {}
    hi = x_max if x_max is not None else float(chart.to_val(chart.box[2], 0)[0])
    for lab, x in zip(labels, xs):
        cx, cy = chart.trace((x, y_probe), (x_min, hi), **trace_kw)
        keep = (cy <= y_max) & (cy >= -0.002)
        # the curve's own branch: contiguous around the seed
        out[lab] = (cx[keep], cy[keep])
    return out


def descending_branch(x, y, seed_x, y_min=0.04, tol=0.002):
    """The contiguous part of a traced curve through seed_x on which y decreases with x
    (within tol), restricted to y >= y_min: drops hops onto crossing curves and labels."""
    o = np.argsort(x)
    x, y = x[o], y[o]
    k = int(np.argmin(np.abs(x - seed_x)))
    lo = k
    while lo > 0 and y[lo - 1] >= y[lo] - tol:
        lo -= 1
    hi = k
    while hi < len(x) - 1 and y[hi + 1] <= y[hi] + tol:
        hi += 1
    xs, ys = x[lo:hi + 1], y[lo:hi + 1]
    keep = ys >= y_min
    return xs[keep], ys[keep]


def zero_crossing(x, y, y_fit=(0.04, 0.07)):
    """x where a nearly straight descending curve reaches y = 0, from a line fit to y_fit."""
    m = (y >= y_fit[0]) & (y <= y_fit[1])
    a, b = np.polyfit(y[m], x[m], 1)
    return float(b)


def snap(chart: Chart, x_guess: float, y_value: float, tol: float, dy: float = None):
    """(x, y) of the curve crossing nearest to x_guess, probing y = y_value and nearby
    heights (a crossing can fall on a masked vertical grid line at one height)."""
    step = abs(float(chart.py[0])) * 3.0 if dy is None else dy      # ~3 px
    best = None
    for k in (0, 1, -1, 2, -2, 3, -3, 4, -4):
        y = y_value + k * step
        xs = crossings(chart, y)
        if len(xs):
            x = float(xs[np.argmin(np.abs(xs - x_guess))])
            if abs(x - x_guess) <= tol and (best is None or abs(x - x_guess) < abs(best[0] - x_guess)):
                best = (x, y)
                if k == 0:
                    break
        if best is not None and k != 0:
            break
    if best is None:
        raise ValueError(f"no curve within {tol} of x={x_guess} near y={y_value}")
    return best


def straight_family(chart: Chart, guesses: dict, y_seed: float, y_max: float, x_lo: float,
                    x_hi: float, tol: float = 0.08, y_min: float = 0.04, y_fit=(0.04, 0.07)):
    """Nearly straight descending curves (CT or CP against V/nD above the stall region):
    each seeded near guesses[label] at y_seed, its descending branch kept for
    y_min <= y <= y_max, and its zero crossing extrapolated. Returns
    {label: (x, y, x_zero)}."""
    out = {}
    for lab, g in guesses.items():
        sx, sy = snap(chart, g, y_seed, tol)
        x, y = chart.trace((sx, sy), (x_lo, x_hi))
        x, y = descending_branch(x, y, sx, y_min=y_min)
        m = y <= y_max
        x, y = x[m], y[m]
        out[lab] = (x, y, zero_crossing(x, y, y_fit))
    return out
