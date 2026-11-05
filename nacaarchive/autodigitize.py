"""Automatic digitizing of a gridded chart: axis calibration from OCR'd tick labels,
then curve tracking, with a confidence verdict and an overlay image for review.

Calibration. Tick labels are OCR'd in strips beside the plot box and matched to the
detected grid lines. A robust fit (all label pairs tried, most inliers wins) gives the
value step per grid line, snapped to a "nice" step (1, 2, 2.5, 5 x 10^k) when close.
Every detected grid line then gets a value, and pixel <-> value conversion interpolates
piecewise-linearly between lines, so uneven scan stretch doesn't bias the readings
(NACA 747 fig. 13c was stretched by up to 4 px between lines).

Curves. Grid lines and label text are masked; ink runs in each pixel column are linked
into tracks by predicted position and slope (so tracks pass through crossings and dashes
by coasting), short tracks are dropped, broken tracks are rejoined, and duplicates are
removed. Labels inside the plot ("20°") are attached to the nearest track.

Verdict: ``auto`` when both axes calibrate with at least 3 consistent labels and the
tracks cover most of the curve ink; otherwise ``needs_review`` with reasons.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

from .figures import Grid
from .pages import Word, tesseract

NUMERIC = "0123456789.-"


# ---------------------------------------------------------------- numbers
def parse_number(t: str) -> float | None:
    t = t.strip().replace("−", "-").replace("—", "-").replace("–", "-").replace(",", ".")
    t = t.translate(str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1"}))
    t = t.rstrip(".")
    if not re.fullmatch(r"-?(\d+\.?\d*|\.\d+)", t):
        return None
    return float(t)


def nice(step: float, tol: float = 0.04) -> float:
    """Snap a step to 1, 2, 2.5 or 5 x 10^k when within tol (relative)."""
    if step == 0:
        return step
    e = math.floor(math.log10(abs(step)))
    m = abs(step) / 10 ** e
    for c in (1, 2, 2.5, 5, 10):
        if abs(m - c) / c < tol:
            return math.copysign(c * 10 ** e, step)
    return step


@dataclass
class Axis:
    lines: np.ndarray          # pixel positions of grid lines
    values: np.ndarray         # value of each line
    labels: list               # (pixel, value, inlier) for every parsed label
    inliers: int
    step: float                # value per grid gap
    residual: float            # max label misfit, in grid gaps
    side: str = ""

    @property
    def ok(self) -> bool:
        return self.inliers >= 3 and self.residual < 0.2 and self.inliers >= 0.6 * len(self.labels)

    def to_val(self, p):
        return _interp(np.asarray(p, float), self.lines, self.values)

    def to_px(self, v):
        o = np.argsort(self.values)
        return _interp(np.asarray(v, float), self.values[o], self.lines[o])

    def as_dict(self):
        return dict(side=self.side, lines=self.lines.round(2).tolist(), values=self.values.tolist(),
                    step=self.step, inliers=self.inliers, residual=round(self.residual, 4),
                    labels=[(round(p, 1), v, bool(i)) for p, v, i in self.labels], ok=self.ok)


def _interp(x, xp, fp):
    """np.interp with linear extrapolation at both ends."""
    y = np.interp(x, xp, fp)
    if len(xp) >= 2:
        lo, hi = x < xp[0], x > xp[-1]
        y = np.where(lo, fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]), y)
        y = np.where(hi, fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y)
    return y


def fit_axis(lines: np.ndarray, labels: list[tuple[float, float]], side: str = "") -> Axis | None:
    """Values for grid lines from (pixel, value) tick labels."""
    lines = np.asarray(lines, float)
    if len(lines) < 2 or len(labels) < 2:
        return None
    gap = float(np.median(np.diff(lines)))
    # snap each label to its grid line when close
    pts = []
    for p, v in labels:
        j = int(np.argmin(np.abs(lines - p)))
        pts.append((lines[j] if abs(lines[j] - p) < 0.4 * gap else p, v))
    pts = sorted(set(pts))
    best = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            (p1, v1), (p2, v2) = pts[i], pts[j]
            if abs(p2 - p1) < 0.5 * gap or v1 == v2:
                continue
            a = (v2 - v1) / (p2 - p1)
            b = v1 - a * p1
            err = np.array([abs(a * p + b - v) / abs(a * gap) for p, v in pts])
            n = int((err < 0.2).sum())
            score = (n, -float(np.sum(np.minimum(err, 0.2))))
            if best is None or score > best[0]:
                best = (score, a, b, err)
    if best is None:
        return None
    _, a, b, err = best
    inl = err < 0.2
    P = np.array([p for p, _ in pts])[inl]
    V = np.array([v for _, v in pts])[inl]
    if len(P) >= 2:
        a, b = np.polyfit(P, V, 1)
    step = nice(a * gap)
    # values on grid lines: index each line from the inlier labels' lines
    k = np.round((lines - lines[0]) / gap)
    v_first = np.median([v - step * np.round((p - lines[0]) / gap) for p, v in zip(P, V)]) if len(P) else b
    values = v_first + step * k
    values = np.round(values / abs(step) * 1000) / 1000 * abs(step) if step else values
    resid = float(np.max(np.abs(_interp(P, lines, values) - V)) / abs(step)) if len(P) and step else 9.9
    labs = [(p, v, bool(e < 0.2)) for (p, v), e in zip(pts, err)]
    return Axis(lines, values, labs, int(inl.sum()), float(step), resid, side)


def axis_labels(img: np.ndarray, g: Grid, side: str) -> tuple[list[tuple[float, float]], list[Word]]:
    """Parsed tick labels in a strip beside the plot box: (pixel along axis, value)."""
    H, W = img.shape
    x0, y0, x1, y1 = g.box
    gap_y = float(np.median(np.diff(g.hlines)))
    gap_x = float(np.median(np.diff(g.vlines)))
    if side == "left":
        r = (max(x0 - 260, 0), max(int(y0 - gap_y), 0), max(x0 - 3, 1), min(int(y1 + gap_y), H))
    elif side == "right":
        r = (min(x1 + 3, W - 1), max(int(y0 - gap_y), 0), min(x1 + 260, W), min(int(y1 + gap_y), H))
    elif side == "bottom":
        r = (max(int(x0 - gap_x), 0), min(y1 + 3, H - 1), min(int(x1 + gap_x), W), min(y1 + 110, H))
    else:  # top
        r = (max(int(x0 - gap_x), 0), max(y0 - 110, 0), min(int(x1 + gap_x), W), max(y0 - 3, 1))
    if r[2] - r[0] < 10 or r[3] - r[1] < 10:
        return [], []
    crop = img[r[1]:r[3], r[0]:r[2]]
    words = tesseract(crop, psm=11, whitelist=NUMERIC)
    for w in words:
        w.x0, w.x1, w.y0, w.y1 = w.x0 + r[0], w.x1 + r[0], w.y0 + r[1], w.y1 + r[1]
    out = _parse_labels(img, words, side)
    if len(out) < 4:
        # sparse OCR misses short labels (".5") and labels touching the frame:
        # read each label on its own instead
        words2 = strip_words(img, r)
        out2 = _parse_labels(img, words2, side)
        if len(out2) > len(out):
            out, words = out2, words2
    return out, words


def strip_words(img: np.ndarray, r) -> list[Word]:
    """Numbers in a strip beside the plot, OCR'd one at a time: long lines (the frame, axis
    rules) are dropped, small ink components grouped into words, each word read with psm 7."""
    import cv2

    x0, y0, x1, y1 = r
    sub = img[y0:y1, x0:x1]
    dark = (sub < 150).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
    keep = np.zeros_like(dark)
    for i in range(1, n):
        bx, by, bw, bh, area = st[i]
        if bh <= 80 and bw <= 80 and area >= 10:
            keep[lab == i] = 1
    groups = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(groups)
    words = []
    for i in range(1, n):
        bx, by, bw, bh, _ = st[i]
        if bh < 16 or bh > 110 or bw > 260:
            continue
        pad = 12
        cx0, cy0 = max(bx - pad, 0), max(by - pad, 0)
        piece = np.where(keep[cy0:by + bh + pad, cx0:bx + bw + pad] > 0, 0, 255).astype(np.uint8)
        piece = np.pad(piece, 12, constant_values=255)
        found = tesseract(piece, psm=7, whitelist=NUMERIC) or tesseract(piece, psm=10, whitelist=NUMERIC)
        for w in found:
            ox, oy = x0 + cx0 - 12, y0 + cy0 - 12
            words.append(Word(w.text, w.x0 + ox, w.y0 + oy, w.x1 + ox, w.y1 + oy, w.conf, 2_000_000 + i))
    return words


def _parse_labels(img: np.ndarray, words: list[Word], side: str) -> list[tuple[float, float]]:
    out = []
    for w in words:
        if w.conf < 30:
            continue
        t = w.text
        # NACA labels are written ".02", "-.10": Tesseract usually drops a leading point
        if "." not in t and re.fullmatch(r"-?\d+", t) and leading_dot(img, w):
            t = t.replace("-", "-.") if t.startswith("-") else "." + t
        v = parse_number(t)
        if v is None:
            continue
        out.append(((w.cy if side in ("left", "right") else w.cx), v))
    return out


def leading_dot(img: np.ndarray, w: Word) -> bool:
    """A decimal point just left of a number: a small compact blob near the baseline. In
    italic NACA lettering the point sits under the slanted digit's foot, so the search
    reaches a little under the digit too."""
    import cv2

    h = w.y1 - w.y0
    if h < 6:
        return False
    # a window larger than the search box, so a blob cut by the box edge is seen whole
    X0, X1 = int(w.x0 - 1.1 * h), int(w.x0 + 0.5 * h)
    Y0, Y1 = int(w.y0 + 0.3 * h), int(w.y1 + 0.35 * h)
    if X0 < 0 or Y0 < 0 or Y1 > img.shape[0]:
        return False
    sub = (img[Y0:Y1, X0:X1] < 140).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(sub)
    for i in range(1, n):
        bx, by, bw, bh, area = st[i]
        cx, cy = cen[i][0] + X0, cen[i][1] + Y0
        if not (0.07 * h <= bw <= 0.35 * h and 0.07 * h <= bh <= 0.35 * h and area >= 0.5 * bw * bh):
            continue
        if bx == 0 or by == 0 or bx + bw >= sub.shape[1] or by + bh >= sub.shape[0]:
            continue
        if w.x0 - 0.9 * h <= cx <= w.x0 + 0.3 * h and cy >= w.y0 + 0.6 * h:
            return True
    return False


def calibrate(img: np.ndarray, g: Grid) -> dict[str, Axis]:
    axes = {}
    for side in ("left", "bottom", "right", "top"):
        labs, _ = axis_labels(img, g, side)
        lines = g.hlines if side in ("left", "right") else g.vlines
        ax = fit_axis(lines, labs, side)
        if ax is not None:
            axes[side] = ax
    return axes


# ----------------------------------------------------------------- curves
@dataclass
class Track:
    pts: list = field(default_factory=list)   # (x px, y px)
    slope: float = 0.0
    last_hit: float = 0.0
    misses: int = 0
    label: str | None = None

    def predict(self, x):
        px, py = self.pts[-1]
        return py + self.slope * (x - px)


def _runs(col: np.ndarray, max_len: int):
    idx = np.flatnonzero(col)
    if idx.size == 0:
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [float(gr.mean()) for gr in groups if len(gr) <= max_len]


def ink_mask(img: np.ndarray, g: Grid, text_boxes=(), level: int | None = None, grid_halfwidth: int = 3):
    x0, y0, x1, y1 = g.box
    sub = img[y0:y1 + 1, x0:x1 + 1]
    if level is None:
        gl = [img[int(round(h)), x0:x1 + 1] for h in g.hlines]
        grid_gray = float(np.median(np.concatenate(gl))) if gl else 150.0
        level = int(np.clip(grid_gray - 25, 90, 175))
    ink = sub < level
    for v in g.vlines:
        c = int(round(v)) - x0
        ink[:, max(c - grid_halfwidth, 0):c + grid_halfwidth + 1] = False
    for h in g.hlines:
        c = int(round(h)) - y0
        ink[max(c - grid_halfwidth, 0):c + grid_halfwidth + 1, :] = False
    _bridge(ink, [int(round(v)) - x0 for v in g.vlines], [int(round(h)) - y0 for h in g.hlines],
            grid_halfwidth)
    for bx0, by0, bx1, by1 in text_boxes:
        ink[max(int(by0) - y0 - 3, 0):max(int(by1) - y0 + 4, 0), max(int(bx0) - x0 - 3, 0):max(int(bx1) - x0 + 4, 0)] = False
    return ink, level


def _bridge(ink: np.ndarray, vcols, hrows, hw: int, reach: int = 7):
    """Restore curve ink inside the masked grid-line bands: a band pixel is ink when there
    is ink just outside the band on both sides (within ``reach`` along the band, so a
    steep curve that shifts while crossing still bridges)."""
    import cv2

    H, W = ink.shape
    k_h = np.ones((1, 2 * reach + 1), np.uint8)
    for c in hrows:
        a, b = c - hw - 1, c + hw + 1
        if a < 0 or b >= H:
            continue
        top = cv2.dilate(ink[a].astype(np.uint8)[None, :], k_h)[0].astype(bool)
        bot = cv2.dilate(ink[b].astype(np.uint8)[None, :], k_h)[0].astype(bool)
        ink[a + 1:b, :] |= (top & bot)[None, :]
    for c in vcols:
        a, b = c - hw - 1, c + hw + 1
        if a < 0 or b >= W:
            continue
        lft = cv2.dilate(ink[:, a].astype(np.uint8)[:, None], k_h.T)[:, 0].astype(bool)
        rgt = cv2.dilate(ink[:, b].astype(np.uint8)[:, None], k_h.T)[:, 0].astype(bool)
        ink[:, a + 1:b] |= (lft & rgt)[:, None]


def track_curves(ink: np.ndarray, step: int = 2, window: float = 10.0, max_gap: int = 45,
                 max_run: int = 70, min_span: float = 0.05) -> list[Track]:
    """Link ink runs column by column into tracks (pixel coordinates of ``ink``)."""
    H, W = ink.shape
    live: list[Track] = []
    done: list[Track] = []
    for x in range(0, W, step):
        runs = _runs(ink[:, x], max_run)
        used = set()
        # nearest-first assignment of runs to predicted tracks
        pairs = []
        for ti, t in enumerate(live):
            pred = t.predict(x)
            win = window + 1.5 * abs(t.slope) * step     # steep curves move more per column
            for ri, r in enumerate(runs):
                d = abs(r - pred)
                if d <= win:
                    pairs.append((d, ti, ri))
        pairs.sort()
        got_t = set()
        for d, ti, ri in pairs:
            if ti in got_t or ri in used:
                continue
            t = live[ti]
            px, py = t.pts[-1]
            new_slope = (runs[ri] - py) / max(x - px, 1)
            t.slope = new_slope if len(t.pts) < 3 else 0.7 * t.slope + 0.3 * new_slope
            t.pts.append((float(x), runs[ri]))
            t.last_hit = x
            got_t.add(ti)
            used.add(ri)
        # unclaimed runs near a claimed one (thick or crossing) are skipped; others seed tracks
        for ri, r in enumerate(runs):
            if ri in used:
                continue
            if any(abs(r - t.predict(x)) <= window for t in live):
                continue
            live.append(Track([(float(x), r)], 0.0, x))
        keep = []
        for t in live:
            if x - t.last_hit > max_gap:
                done.append(t)
            else:
                keep.append(t)
        live = keep
    done += live
    min_pts = max(8, int(min_span * W / step * 0.5))
    tracks = [t for t in done if len(t.pts) >= min_pts and (t.pts[-1][0] - t.pts[0][0]) >= min_span * W]
    for t in tracks:                      # end points are often junk from a crossing
        t.pts = t.pts[3:-3]
    tracks = _join(tracks, max_join=3 * max_gap)
    return _dedupe(tracks)


def _join(tracks: list[Track], max_join: int, tol: float = 6.0) -> list[Track]:
    tracks = sorted(tracks, key=lambda t: t.pts[0][0])
    merged = True
    while merged:
        merged = False
        for a in tracks:
            ax, ay = a.pts[-1]
            sl = _end_slope(a.pts, -1)
            cands = []
            for b in tracks:
                if b is a:
                    continue
                bx, by = b.pts[0]
                if -12 < bx - ax <= max_join:
                    bs = _end_slope(b.pts, 0)
                    mid = 0.5 * (sl + bs)
                    if abs(ay + mid * (bx - ax) - by) < tol + 5 * abs(mid) + 0.02 * abs(bx - ax):
                        cands.append((bx - ax, b))
            if cands:
                b = min(cands, key=lambda c: c[0])[1]
                a.pts += [p for p in b.pts if p[0] > ax]
                tracks.remove(b)
                merged = True
                break
    return tracks


def _end_slope(pts, end, n=14):
    p = np.array(pts[-n:] if end == -1 else pts[:n])
    if len(p) < 2:
        return 0.0
    return float(np.polyfit(p[:, 0], p[:, 1], 1)[0])


def _dedupe(tracks: list[Track], tol: float = 3.0) -> list[Track]:
    tracks = sorted(tracks, key=lambda t: -len(t.pts))
    keep: list[Track] = []
    for t in tracks:
        a = np.array(t.pts)
        dup = False
        for k in keep:
            b = np.array(k.pts)
            lo, hi = max(a[0, 0], b[0, 0]), min(a[-1, 0], b[-1, 0])
            if hi <= lo:
                continue
            m = (a[:, 0] >= lo) & (a[:, 0] <= hi)
            if m.sum() < 0.7 * len(a):
                continue
            d = np.abs(a[m, 1] - np.interp(a[m, 0], b[:, 0], b[:, 1]))
            if np.mean(d < tol) > 0.7:
                dup = True
                break
        if not dup:
            keep.append(t)
    return sorted(keep, key=lambda t: t.pts[0][0])


def coverage(ink: np.ndarray, tracks: list[Track], radius: int = 4) -> float:
    if not ink.any():
        return 1.0
    cov = np.zeros_like(ink)
    H, W = ink.shape
    for t in tracks:
        p = np.array(t.pts)
        xs = np.arange(int(p[0, 0]), int(p[-1, 0]) + 1)
        ys = np.interp(xs, p[:, 0], p[:, 1])
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                xx = np.clip(xs + dx, 0, W - 1)
                yy = np.clip(np.round(ys).astype(int) + dy, 0, H - 1)
                cov[yy, xx] = True
    return float((ink & cov).sum() / ink.sum())


# ------------------------------------------------------------------ figure
@dataclass
class Result:
    grid: Grid
    axes: dict
    x_axis: Axis | None
    y_axis: Axis | None
    curves: list            # dicts: label, x[], y[], px[], py[]
    coverage: float
    status: str
    reasons: list
    ink_level: int = 0

    def as_dict(self):
        return dict(box=list(self.grid.box), label=self.grid.label, caption=self.grid.caption,
                    axes={k: a.as_dict() for k, a in self.axes.items()},
                    x_side=self.x_axis.side if self.x_axis else None,
                    y_side=self.y_axis.side if self.y_axis else None,
                    curves=self.curves, coverage=round(self.coverage, 3), status=self.status,
                    reasons=self.reasons, notes=self.grid.notes)


def digitize(img: np.ndarray, g: Grid, min_points: int = 15) -> Result:
    reasons = []
    axes = calibrate(img, g)
    xa = axes.get("bottom") if axes.get("bottom") and axes["bottom"].ok else axes.get("top") or axes.get("bottom")
    ya = axes.get("left") if axes.get("left") and axes["left"].ok else axes.get("right") or axes.get("left")
    if xa is None or not xa.ok:
        reasons.append("x axis not calibrated from tick labels")
    if ya is None or not ya.ok:
        reasons.append("y axis not calibrated from tick labels")
    if axes.get("left") and axes.get("right") and axes["left"].ok and axes["right"].ok \
            and abs(axes["left"].step - axes["right"].step) > 1e-12:
        reasons.append("two y scales (left and right): curves not assigned to an axis")
    # labels and other text inside the plot
    x0, y0, x1, y1 = g.box
    inner = inner_text(img, g)
    ink, level = ink_mask(img, g, [(w.x0, w.y0, w.x1, w.y1) for w in inner])
    tracks = track_curves(ink)
    cov = coverage(ink, tracks)
    if not tracks:
        reasons.append("no curves found")
    elif cov < 0.85:
        reasons.append(f"tracks cover {cov:.0%} of the ink (steep curves, symbols or text left over)")
    _attach_labels(tracks, inner, x0, y0)
    unlabeled = sum(t.label is None for t in tracks)
    if tracks and unlabeled:
        g.notes.append(f"{unlabeled} of {len(tracks)} curves have no label read")
    curves = []
    for t in tracks:
        p = np.array(t.pts)
        px, py = p[:, 0] + x0, _running_median(p[:, 1]) + y0     # removes 1-2 px steps at grid lines
        c = dict(label=t.label, px=px.round(1).tolist(), py=py.round(1).tolist(), n=len(p))
        if xa is not None and ya is not None:
            c["x"] = [round(float(v), 6) for v in xa.to_val(px)]
            c["y"] = [round(float(v), 6) for v in ya.to_val(py)]
        curves.append(c)
    status = "auto" if not reasons else "needs_review"
    return Result(g, axes, xa, ya, curves, cov, status, reasons, level)


def inner_text(img: np.ndarray, g: Grid, max_h: int = 75, max_w: int = 90) -> list[Word]:
    """Text inside the plot box (curve labels, legends). OCR on the whole box fails on the
    grid, so: whiten the grid lines, take small ink components (characters), group them
    into words, and OCR each word on its own."""
    import cv2

    x0, y0, x1, y1 = g.box
    sub = img[y0:y1 + 1, x0:x1 + 1].copy()
    for v in g.vlines:
        c = int(round(v)) - x0
        sub[:, max(c - 3, 0):c + 4] = 255
    for h in g.hlines:
        c = int(round(h)) - y0
        sub[max(c - 3, 0):c + 4, :] = 255
    dark = (sub < 150).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark)
    small = np.zeros_like(dark)
    for i in range(1, n):
        bx, by, bw, bh, area = st[i]
        if 4 <= bh <= max_h and bw <= max_w and area >= 12:
            small[lab == i] = 1
    groups = cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(groups)
    words = []
    for i in range(1, n):
        bx, by, bw, bh, _ = st[i]
        if bh < 14 or bh > 2 * max_h or bw > 6 * max_w:
            continue
        pad = 10
        cx0, cy0 = max(bx - pad, 0), max(by - pad, 0)
        crop = sub[cy0:by + bh + pad, cx0:bx + bw + pad]
        crop = np.where(crop < 150, 0, 255).astype(np.uint8)
        for w in tesseract(crop, psm=7):
            if w.conf < 60 or not any(ch.isalnum() for ch in w.text):
                continue
            text = re.sub(r'(\d)["”“\'’]+$', r"\1°", w.text)     # 30" -> 30°
            words.append(Word(text, w.x0 + cx0 + x0, w.y0 + cy0 + y0, w.x1 + cx0 + x0,
                              w.y1 + cy0 + y0, w.conf, 1_000_000 + i))
    return words


def _running_median(y: np.ndarray, k: int = 5) -> np.ndarray:
    if len(y) < k:
        return y
    pad = np.pad(y, k // 2, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)


_LABEL_RE = re.compile(r"^-?\d+(\.\d+)?°?$|°")


def _attach_labels(tracks, words, x0, y0, max_d: float = 70.0):
    cands = [w for w in words if _LABEL_RE.search(w.text)]
    used = set()
    for w in sorted(cands, key=lambda w: w.cy):
        best = None
        for t in tracks:
            if t.label is not None:
                continue
            p = np.array(t.pts) + (x0, y0)
            d = float(np.min(np.hypot(p[:, 0] - w.cx, p[:, 1] - w.cy)))
            if d < max_d and (best is None or d < best[0]):
                best = (d, t)
        if best and id(w) not in used:
            best[1].label = w.text.replace("°°", "°")
            used.add(id(w))


COLORS = [(220, 0, 0), (0, 140, 0), (0, 0, 220), (200, 0, 200), (0, 160, 160), (230, 120, 0)]


def overlay(img: np.ndarray, res: Result, width: int = 1400) -> Image.Image:
    """The chart region with detected grid, axis labels (green inlier / red outlier) and
    traced curves drawn on, for checking by eye."""
    from PIL import ImageFont

    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=30)
    except TypeError:                      # Pillow < 10.1
        font = ImageFont.load_default()
    x0, y0, x1, y1 = res.grid.box
    d.rectangle(res.grid.box, outline=(0, 120, 255), width=2)
    for side, ax in res.axes.items():
        for p, v, ok in ax.labels:
            col = (0, 170, 0) if ok else (230, 0, 0)
            if side in ("left", "right"):
                xx = x0 - 8 if side == "left" else x1 + 8
                d.ellipse((xx - 5, p - 5, xx + 5, p + 5), outline=col, width=2)
            else:
                yy = y1 + 8 if side == "bottom" else y0 - 8
                d.ellipse((p - 5, yy - 5, p + 5, yy + 5), outline=col, width=2)
    for k, c in enumerate(res.curves):
        col = COLORS[k % len(COLORS)]
        d.line(list(zip(c["px"], c["py"])), fill=col, width=3)
        if c.get("label"):
            d.text((c["px"][0] + 4, c["py"][0] - 36), c["label"], fill=col, font=font)
    rx0, ry0, rx1, ry1 = res.grid.region
    im = im.crop((rx0, ry0, rx1, ry1))
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    return im
