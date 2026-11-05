"""Data-point symbols in charts: find them, tell their shapes apart, and turn their centres
into tables of values.

NACA and NASA test reports plot measured points as small symbols (circle, square, diamond,
triangle, half-round, ...), usually hollow and often with a dot at the exact point, with a
faired curve drawn through them. The points are the data; the curve is easy to redo.

Hollow symbols are enclosed holes in the ink (with the grid removed and the outline bridged
across grid lines) of a consistent size; solid symbols are blobs that survive an opening
that removes lines. Shapes are told apart by the width profile of the symbol's interior.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# width of the interior at 25/50/75% of its height (rows), then height at 25/50/75% of
# its width (columns), relative to the widest row / tallest column
PROTOTYPES = {
    "circle":            (0.87, 1.00, 0.87, 0.87, 1.00, 0.87),
    "square":            (1.00, 1.00, 1.00, 1.00, 1.00, 1.00),
    "diamond":           (0.50, 1.00, 0.50, 0.50, 1.00, 0.50),
    "triangle up":       (0.30, 0.55, 0.80, 0.50, 1.00, 0.50),
    "triangle down":     (0.80, 0.55, 0.30, 0.50, 1.00, 0.50),
    "triangle left":     (0.50, 1.00, 0.50, 0.30, 0.55, 0.80),
    "triangle right":    (0.50, 1.00, 0.50, 0.80, 0.55, 0.30),
    "half-round top":    (0.87, 1.00, 1.00, 0.93, 1.00, 0.93),   # round top, flat bottom
    "half-round bottom": (1.00, 1.00, 0.87, 0.93, 1.00, 0.93),
    "right triangle ◣":  (0.25, 0.50, 0.75, 0.75, 0.50, 0.25),   # right angle at bottom left
    "right triangle ◢":  (0.25, 0.50, 0.75, 0.25, 0.50, 0.75),
    "right triangle ◤":  (0.75, 0.50, 0.25, 0.75, 0.50, 0.25),
    "right triangle ◥":  (0.75, 0.50, 0.25, 0.25, 0.50, 0.75),
}
GLYPH = {"circle": "○", "square": "□", "diamond": "◇", "triangle up": "△", "triangle down": "▽",
         "triangle left": "◁", "triangle right": "▷", "half-round top": "◠", "half-round bottom": "◡",
         "right triangle ◣": "◣", "right triangle ◢": "◢", "right triangle ◤": "◤", "right triangle ◥": "◥"}


@dataclass
class Symbol:
    cx: float          # centre (the dot when there is one), box-local pixels
    cy: float
    size: float        # outer size, pixels
    shape: str
    filled: bool
    dot: bool
    box: tuple         # outer box, box-local pixels
    match: float       # distance to the shape prototype (smaller is better)
    inked: float = 1.0 # fraction of the silhouette that is ink (hollow ~0.4, half-filled ~0.7)
    profile: tuple = (0,) * 6
    hole: tuple = (0, 0, 0, 0)   # interior box

    @property
    def kind(self) -> str:
        return ("filled " if self.filled else "") + self.shape + (" with dot" if self.dot and not self.filled else "")


def profile(region: np.ndarray) -> tuple:
    ys, xs = np.nonzero(region)
    if len(ys) < 6:
        return (0,) * 6
    r = region[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    rows, cols = r.sum(1).astype(float), r.sum(0).astype(float)

    def at(v):
        v = np.convolve(v, np.ones(3) / 3, mode="same") if len(v) >= 5 else v
        m = v.max()
        return [float(v[min(int(f * (len(v) - 1) + 0.5), len(v) - 1)] / m) for f in (0.25, 0.5, 0.75)]

    return tuple(at(rows) + at(cols))


def classify(region: np.ndarray) -> tuple[str, float]:
    p = np.array(profile(region))
    best = min(PROTOTYPES, key=lambda k: np.abs(p - PROTOTYPES[k]).sum())
    return best, float(np.abs(p - PROTOTYPES[best]).sum())


def find_symbols(ink: np.ndarray, text_boxes=(), min_size: int = 14, max_size: int = 70,
                 raw: np.ndarray | None = None, vlines=(), hlines=()) -> list[Symbol]:
    """Symbols in a plot's ink mask (grid removed, curves bridged across grid lines).

    raw: the same area's ink with the grid left in. Symbols are usually opaque, hiding the
    grid behind them, and removing the grid band also removes a symbol edge lying on a grid
    line; so holes are looked for in the raw ink first (grid cells and pockets with two
    sides on grid lines, vlines/hlines in the mask's pixels, are not symbols), then in the
    grid-removed ink for symbols the grid shows through.
    text_boxes: (x0, y0, x1, y1, is_text) in the mask's pixels; a hole inside a box whose
    is_text is true is a character, not a symbol."""
    H, W = ink.shape
    syms = []
    if raw is not None:
        # a hole with two sides on grid lines is a grid cell or a pocket between the grid and
        # a curve, unless it has a centre dot
        syms = [s for s in _hollow(raw, min_size, max_size) if s.dot or _grid_sides(s, vlines, hlines) < 2]
    for s in _hollow(ink, min_size, max_size):
        if not any(abs(s.cx - t.cx) < t.size / 2 and abs(s.cy - t.cy) < t.size / 2 for t in syms):
            syms.append(s)
    syms += _filled(ink, syms, min_size, max_size)
    # characters with holes (0, o, D, 6, 8, 9, B) inside words that were read as text
    out = []
    for s in syms:
        inside = [b for b in text_boxes if b[0] - 2 <= s.cx <= b[2] + 2 and b[1] - 2 <= s.cy <= b[3] + 2]
        if any(b[4] for b in inside):
            continue
        out.append(s)
    syms = _not_lettering(ink, out)
    if len(syms) < 3 and not any(s.dot for s in syms):     # a stray pocket or letter, not data
        return []
    # one chart uses one symbol size: drop holes far from the typical size
    med = float(np.median([s.size for s in syms]))
    syms = [s for s in syms if 0.6 * med <= s.size <= 1.6 * med]
    # when most hollow symbols carry a centre dot, holes without one are curve/grid pockets
    hollow = [s for s in syms if not s.filled]
    if len(hollow) >= 4 and sum(s.dot for s in hollow) >= 0.6 * len(hollow):
        syms = [s for s in syms if s.filled or s.dot]
    return syms


def _not_lettering(ink: np.ndarray, syms: list[Symbol]) -> list[Symbol]:
    """Drop holes that are letters of hand lettering OCR did not read ("Low-speed tests",
    the NACA stamp): a letter has letter-sized neighbours on its baseline, close beside it.
    A data point's neighbours are the curve (long and thin) or other symbols."""
    import cv2

    if not syms:
        return syms
    n, _, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    comps = st[1:, :4]
    keep = []
    for s in syms:
        x0, y0, x1, y1 = s.box
        h = max(y1 - y0, 1)
        cy = (y0 + y1) / 2
        near = 0
        for cx_, cy0, cw, ch in comps:
            if cw > 2.5 * h or not (0.45 * h <= ch <= 1.6 * h):      # a curve, a speck or a big shape
                continue
            if cx_ <= x1 and cx_ + cw >= x0 and cy0 <= y1 and cy0 + ch >= y0:   # the symbol itself
                continue
            if abs(cy0 + ch / 2 - cy) > 0.35 * h:
                continue
            gap = max(x0 - (cx_ + cw), cx_ - x1)
            if gap <= 0.7 * h and not any(t is not s and t.box[0] <= cx_ + cw / 2 <= t.box[2]
                                          and t.box[1] <= cy0 + ch / 2 <= t.box[3] for t in syms):
                near += 1
        if near < 2:
            keep.append(s)
    return keep


def _grid_sides(s: Symbol, vlines, hlines, tol: float = 4.0) -> int:
    """How many sides of a hole lie on grid lines (a grid cell has four)."""
    x0, y0, x1, y1 = s.hole
    near = lambda v, lines: any(abs(v - l) <= tol for l in lines)
    return near(x0, vlines) + near(x1, vlines) + near(y0, hlines) + near(y1, hlines)


def _hollow(ink: np.ndarray, min_size: int, max_size: int) -> list[Symbol]:
    bg = (~ink).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(bg, connectivity=4)
    H, W = ink.shape
    out = []
    for i in range(1, n):
        bx, by, bw, bh, area = st[i]
        if bx == 0 or by == 0 or bx + bw >= W or by + bh >= H:
            continue
        if not (min_size * 0.5 <= bw <= max_size and min_size * 0.5 <= bh <= max_size):
            continue
        if not (0.45 <= bw / bh <= 2.2):
            continue
        hole = lab[by:by + bh, bx:bx + bw] == i
        filled = _fill(hole)
        if filled.sum() / (bw * bh) < 0.4:
            continue
        # the outline: a thin closed ring of ink right around the hole
        t = 3
        fp = np.pad(filled, t).astype(np.uint8)
        ring = cv2.dilate(fp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1))).astype(bool) & ~fp.astype(bool)
        pad = _pad(ink, bx, by, bw, bh, t)
        if pad[ring].mean() < 0.65:
            continue
        dot_px = filled & ink[by:by + bh, bx:bx + bw]
        dot = 2 <= dot_px.sum() <= 0.25 * filled.sum()
        centres = []
        if dot:
            nd, dl, dst, dcen = cv2.connectedComponentsWithStats(dot_px.astype(np.uint8))
            dots = [dcen[k] for k in range(1, nd) if dst[k][4] >= 2]
            # two overlapping symbols share one interior: one point per dot
            centres = [(bx + c[0], by + c[1]) for c in dots] if len(dots) > 1 else []
            ys, xs = np.nonzero(dot_px)
            cx, cy = bx + xs.mean(), by + ys.mean()
        else:
            cx, cy = bx + (bw - 1) / 2, by + (bh - 1) / 2
        # the shape is read from the interior (clean: the faired curve and flags are outside
        # it); the outer silhouette gives the size and how much of the symbol is inked, which
        # tells a half-shaded symbol from a hollow one
        shape, m = classify(filled)
        if m > 1.3:
            continue
        sil, (sx0, sy0) = _silhouette(ink, lab == i, bx, by, bw, bh)
        if sil is None:
            continue
        ys, xs = np.nonzero(sil)
        box = (sx0 + xs.min(), sy0 + ys.min(), sx0 + xs.max() + 1, sy0 + ys.max() + 1)
        size = max(bw, bh) + 2 * _thickness(pad, ring)
        if size < min_size:                     # a degree sign, a small "o"
            continue
        if not dot:   # the outer shape's centre: a curve through the symbol can cut the hole
            cx, cy = (box[0] + box[2] - 1) / 2, (box[1] + box[3] - 1) / 2
        inked = float(ink[sy0:sy0 + sil.shape[0], sx0:sx0 + sil.shape[1]][sil].mean())
        for cx, cy in centres or [(cx, cy)]:
            out.append(Symbol(float(cx), float(cy), float(size), shape, False, bool(dot), box, m,
                              round(inked, 2), profile(filled), (bx, by, bx + bw, by + bh)))
    return out


def _silhouette(ink, hole_full, bx, by, bw, bh):
    """The symbol's outer shape: ink near the hole plus the hole, filled, with thin lines
    (the faired curve, flags) opened away. Classifying this rather than the hole keeps
    half-filled symbols (a triangle shaded on one side) in their own shape."""
    H, W = ink.shape
    r = max(bw, bh) // 2 + 3
    x0, y0, x1, y1 = max(bx - r, 0), max(by - r, 0), min(bx + bw + r, W), min(by + bh + r, H)
    hole = hole_full[y0:y1, x0:x1]
    near = cv2.dilate(hole.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    reg = (ink[y0:y1, x0:x1] & near) | hole
    reg = _fill(reg)
    reg = cv2.morphologyEx(reg.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab = cv2.connectedComponents(reg)
    ys, xs = np.nonzero(hole)
    k = lab[int(np.median(ys)), int(np.median(xs))]
    if k == 0:
        return None, (x0, y0)
    sil = lab == k
    if sil.sum() > 4 * hole.sum() + 400:      # merged with neighbours or a text blob
        return None, (x0, y0)
    return sil, (x0, y0)


def _filled(ink: np.ndarray, hollow: list[Symbol], min_size: int, max_size: int) -> list[Symbol]:
    """Solid symbols: blobs that survive an opening which removes lines (curves are up to
    about 8 px at 300 dpi, thicker where they cross a bridged grid line)."""
    k = 13
    op = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lab, st, cen = cv2.connectedComponentsWithStats(op)
    out = []
    for i in range(1, n):
        bx, by, bw, bh, area = st[i]
        if not (min_size <= bw <= max_size and min_size <= bh <= max_size and 0.5 <= bw / bh <= 2.0):
            continue
        if any(h.box[0] <= cen[i][0] <= h.box[2] and h.box[1] <= cen[i][1] <= h.box[3] for h in hollow):
            continue
        region = lab[by:by + bh, bx:bx + bw] == i
        if region.sum() / (bw * bh) < 0.45:
            continue
        shape, m = classify(region)
        if m > 1.3:
            continue
        out.append(Symbol(float(cen[i][0]), float(cen[i][1]), float(max(bw, bh)), shape, True, False,
                          (bx, by, bx + bw, by + bh), m, 1.0, profile(region)))
    return out


def _fill(hole: np.ndarray) -> np.ndarray:
    """The hole with anything it encloses (the centre dot) filled in."""
    h, w = hole.shape
    canvas = np.zeros((h + 2, w + 2), np.uint8)
    canvas[1:-1, 1:-1] = hole
    outside = 1 - canvas
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(outside, mask, (0, 0), 2)
    return (outside[1:-1, 1:-1] != 2)


def _pad(ink, bx, by, bw, bh, t):
    H, W = ink.shape
    out = np.zeros((bh + 2 * t, bw + 2 * t), bool)
    ys0, xs0 = max(by - t, 0), max(bx - t, 0)
    ys1, xs1 = min(by + bh + t, H), min(bx + bw + t, W)
    out[ys0 - (by - t):ys1 - (by - t), xs0 - (bx - t):xs1 - (bx - t)] = ink[ys0:ys1, xs0:xs1]
    return out


def _thickness(pad, ring) -> float:
    return 3.0 if ring.sum() == 0 else float(min(3.0, 3.0 * pad[ring].mean()))


def erase(ink: np.ndarray, syms: list[Symbol], grow: int = 3) -> np.ndarray:
    """The ink with the symbols removed, so the faired curve can be traced on its own."""
    out = ink.copy()
    H, W = ink.shape
    for s in syms:
        x0, y0, x1, y1 = s.box
        out[max(int(y0) - grow, 0):min(int(y1) + grow + 1, H), max(int(x0) - grow, 0):min(int(x1) + grow + 1, W)] = False
    return out


def group(syms: list[Symbol], tol: float = 0.55) -> list[list[Symbol]]:
    """Symbols of one kind, by clustering their shape profiles (average linkage): a chart's
    symbol set isn't known in advance, and a prototype name alone would merge two kinds
    that resemble the same prototype or split one kind that falls between two."""
    if not syms:
        return []
    f = np.array([list(profile_of(s)) + [2.0 * s.inked, 1.5 * s.filled] for s in syms])
    clusters = [[i] for i in range(len(syms))]

    def dist(a, b):
        return float(np.mean([np.abs(f[i] - f[j]).sum() for i in a for j in b]))

    while len(clusters) > 1:
        best, pair = None, None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = dist(clusters[i], clusters[j])
                if best is None or d < best:
                    best, pair = d, (i, j)
        if best > tol:
            break
        i, j = pair
        clusters[i] += clusters.pop(j)
    return [[syms[i] for i in c] for c in sorted(clusters, key=len, reverse=True)]


def profile_of(s: Symbol) -> tuple:
    return s.profile


def series(syms: list[Symbol], to_x, to_y, ox: float = 0.0, oy: float = 0.0) -> list[dict]:
    """One table per symbol kind: x, y values sorted by x, plus pixel centres. Kinds are
    named after the nearest standard shape (by majority); two kinds with the same name are
    told apart by a letter."""
    out, seen = [], {}
    for ss in group(syms):
        names = [s.kind for s in ss]
        name = max(set(names), key=names.count)
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name} ({chr(ord('A') + seen[name] - 1)})"
        ss = sorted(ss, key=lambda s: s.cx)
        px = np.array([s.cx + ox for s in ss])
        py = np.array([s.cy + oy for s in ss])
        base = max(set(s.shape for s in ss), key=[s.shape for s in ss].count)
        d = dict(symbol=name, glyph=("●" if ss[0].filled and base == "circle" else GLYPH.get(base, "?")),
                 n=len(ss), px=px.round(1).tolist(), py=py.round(1).tolist())
        if to_x is not None and to_y is not None:
            d["x"] = [round(float(v), 6) for v in to_x(px)]
            d["y"] = [round(float(v), 6) for v in to_y(py)]
        out.append(d)
    return out
