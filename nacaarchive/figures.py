"""Find figures on a page: gridded charts (digitizable) and captions.

Charts are found as clusters of long horizontal and vertical lines; the plot box is the
outermost detected grid lines. Captions ("FIGURE 12.-...", "Fig. 3", "TABLE II.-...")
come from the page words and are attached to the nearest chart above them, or to the
nearest one below for captions printed above the figure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .pages import Word

CAPTION_RE = re.compile(r"^(FIGURE|Figure|FIG\.?|Fig\.?)\s*(\d+)\s*[\.,:\-—–]?\s*(\([a-z]\))?", re.I)
TABLE_RE = re.compile(r"^TABLE\s+([IVXLC]+|\d+)\b", re.I)


@dataclass
class Grid:
    box: tuple[int, int, int, int]     # x0, y0, x1, y1: outermost grid lines (plot area)
    vlines: np.ndarray                 # x pixel centers of vertical grid lines
    hlines: np.ndarray                 # y pixel centers of horizontal grid lines
    region: tuple[int, int, int, int] = (0, 0, 0, 0)   # box padded to include labels
    label: str | None = None           # "12", "12(a)"
    caption: str | None = None
    notes: list = field(default_factory=list)


def deskew_angle(img: np.ndarray, max_deg: float = 2.0) -> float:
    """Page skew (degrees, PIL rotate convention) from long near-horizontal lines."""
    mask = (img < 200).astype(np.uint8) * 255
    w = img.shape[1]
    lines = cv2.HoughLinesP(mask, 1, np.pi / 3600, threshold=200, minLineLength=w // 5, maxLineGap=6)
    if lines is None:
        return 0.0
    ang = []
    for x0, y0, x1, y1 in lines.reshape(-1, 4):
        a = np.degrees(np.arctan2(y1 - y0, x1 - x0))
        if abs(a) < max_deg:
            ang.append(a)
    return float(np.median(ang)) if len(ang) >= 3 else 0.0


def rotate(img: np.ndarray, deg: float) -> np.ndarray:
    """Rotate about the center by -deg (undoing a skew measured by deskew_angle)."""
    if abs(deg) < 0.02:
        return img
    h, w = img.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)


def _centers(profile: np.ndarray, frac: float, min_gap: int = 6) -> np.ndarray:
    idx = np.flatnonzero(profile > frac)
    if idx.size == 0:
        return idx.astype(float)
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > min_gap) + 1)
    return np.array([g.mean() for g in groups])


def line_masks(img: np.ndarray, level: int = 215, frac_len: float = 0.06):
    mask = (img < level).astype(np.uint8)
    L = max(40, int(img.shape[1] * frac_len))
    h = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    v = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    return h.astype(bool), v.astype(bool)


def find_grids(img: np.ndarray, min_lines: int = 4, min_size: float = 0.12, cover: float = 0.35,
               level: int = 215) -> list[Grid]:
    """Gridded charts on a (deskewed) page image."""
    H, W = img.shape
    hm, vm = line_masks(img, level)
    both = (hm | vm).astype(np.uint8)
    both = cv2.dilate(both, np.ones((25, 25), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(both)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        if w < min_size * W or h < min_size * W * 0.5:
            continue
        sub_h, sub_v = hm[y:y + h, x:x + w], vm[y:y + h, x:x + w]
        hl = _centers(sub_h.mean(axis=1), cover) + y
        vl = _centers(sub_v.mean(axis=0), cover) + x
        if len(hl) < min_lines or len(vl) < min_lines:
            continue
        hl, vl = _regular(hl), _regular(vl)
        if len(hl) < min_lines or len(vl) < min_lines:
            continue
        box = (int(round(vl[0])), int(round(hl[0])), int(round(vl[-1])), int(round(hl[-1])))
        pad_x, pad_y = int(0.12 * W), int(0.06 * H)
        region = (max(box[0] - pad_x, 0), max(box[1] - pad_y // 2, 0),
                  min(box[2] + pad_x // 2, W), min(box[3] + pad_y, H))
        out.append(Grid(box, vl, hl, region))
    return sorted(out, key=lambda g: (g.box[1], g.box[0]))


def _regular(lines: np.ndarray) -> np.ndarray:
    """Drop lines far from the regular spacing (a curve's straight stretch, a frame line
    off the grid); keeps the longest run of near-integer multiples of the median gap."""
    if len(lines) < 3:
        return lines
    gap = float(np.median(np.diff(lines)))
    k = (lines - lines[0]) / gap
    ok = np.abs(k - np.round(k)) < 0.3
    # a missing first line shifts k: retry from each line, keep the best
    best = lines[ok]
    for ref in lines[1:4]:
        k = (lines - ref) / gap
        ok2 = np.abs(k - np.round(k)) < 0.3
        if ok2.sum() > len(best):
            best = lines[ok2]
    return best


def captions(words: list[Word]) -> list[dict]:
    """Figure and table captions on a page: {kind, label, text, y0, y1, x0, x1}."""
    lines: dict[int, list[Word]] = {}
    for w in words:
        lines.setdefault(w.line, []).append(w)
    out = []
    ordered = sorted(lines.values(), key=lambda ws: min(w.y0 for w in ws))
    for i, ws in enumerate(ordered):
        ws = sorted(ws, key=lambda w: w.x0)
        text = " ".join(w.text for w in ws)
        m, t = CAPTION_RE.match(text), TABLE_RE.match(text)
        if not (m or t):
            continue
        # a caption may continue on the next line(s), if they're close and not another caption
        y1 = max(w.y1 for w in ws)
        h = y1 - min(w.y0 for w in ws)
        for nxt in ordered[i + 1:i + 4]:
            ny0 = min(w.y0 for w in nxt)
            ntext = " ".join(w.text for w in sorted(nxt, key=lambda w: w.x0))
            if ny0 - y1 > 0.8 * h or CAPTION_RE.match(ntext) or TABLE_RE.match(ntext):
                break
            text += " " + ntext
            y1 = max(w.y1 for w in nxt)
        if m:
            label = m.group(2) + (m.group(3) or "")
            out.append(dict(kind="figure", label=label, text=text, y0=min(w.y0 for w in ws), y1=y1,
                            x0=min(w.x0 for w in ws), x1=max(w.x1 for w in ws)))
        else:
            out.append(dict(kind="table", label=t.group(1).upper(), text=text, y0=min(w.y0 for w in ws),
                            y1=y1, x0=min(w.x0 for w in ws), x1=max(w.x1 for w in ws)))
    return out


def attach_captions(grids: list[Grid], caps: list[dict]):
    """Each figure caption goes to the nearest unclaimed chart above it (or below, for a
    caption printed above). Subfigure labels '(a)' printed under a chart and a figure
    caption lower on the page are combined."""
    figs = [c for c in caps if c["kind"] == "figure"]
    free = list(grids)
    for c in sorted(figs, key=lambda c: c["y0"]):
        above = [g for g in free if g.box[3] <= c["y0"] + 5]
        below = [g for g in free if g.box[1] >= c["y1"] - 5]
        g = max(above, key=lambda g: g.box[3]) if above else (min(below, key=lambda g: g.box[1]) if below else None)
        if g is None:
            continue
        g.label, g.caption = c["label"], c["text"]
        free.remove(g)
    # charts left without a caption take the next caption below on the page (shared caption
    # for subfigures (a), (b) ...)
    for g in free:
        lower = [c for c in figs if c["y0"] >= g.box[3]]
        if lower:
            c = min(lower, key=lambda c: c["y0"])
            g.label, g.caption = c["label"], c["text"]
            g.notes.append("caption shared with another chart")
