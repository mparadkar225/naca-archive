"""Tables from page words: rows by vertical position, columns by gaps that stay empty
across most rows. Starts at a "TABLE n.-" caption and runs until a large vertical gap,
another caption, or the end of the page."""
from __future__ import annotations

import numpy as np

from .figures import CAPTION_RE, TABLE_RE
from .pages import Word


def rows_of(words: list[Word]) -> list[list[Word]]:
    if not words:
        return []
    hs = np.median([w.y1 - w.y0 for w in words])
    rows: list[list[Word]] = []
    for w in sorted(words, key=lambda w: w.cy):
        if rows and abs(w.cy - np.mean([v.cy for v in rows[-1]])) < 0.6 * hs:
            rows[-1].append(w)
        else:
            rows.append([w])
    return [sorted(r, key=lambda w: w.x0) for r in rows]


def columns(rows: list[list[Word]], min_gap: float | None = None, frac: float = 0.8) -> list[float]:
    """Column boundaries (x) where the gap between words is empty in >= frac of rows."""
    words = [w for r in rows for w in r]
    if not words:
        return []
    x0, x1 = min(w.x0 for w in words), max(w.x1 for w in words)
    hs = float(np.median([w.y1 - w.y0 for w in words]))
    min_gap = min_gap or 1.2 * hs
    n = int(x1 - x0) + 1
    occ = np.zeros((len(rows), n), bool)
    for i, r in enumerate(rows):
        for w in r:
            occ[i, int(w.x0 - x0):int(w.x1 - x0) + 1] = True
    empty = occ.mean(axis=0) <= (1 - frac)
    bounds, start = [], None
    for i, e in enumerate(empty):
        if e and start is None:
            start = i
        elif not e and start is not None:
            if i - start >= min_gap:
                bounds.append(x0 + 0.5 * (start + i))
            start = None
    return bounds


def table_after(words: list[Word], cap: dict, max_gap_lines: float = 2.8) -> dict | None:
    below = [w for w in words if w.y0 > cap["y1"] + 2]
    rows = rows_of(below)
    if not rows:
        return None
    hs = float(np.median([w.y1 - w.y0 for w in below]))
    keep, last_y = [], cap["y1"]
    for r in rows:
        y0 = min(w.y0 for w in r)
        text = " ".join(w.text for w in r)
        if y0 - last_y > max_gap_lines * hs or CAPTION_RE.match(text) or TABLE_RE.match(text):
            break
        keep.append(r)
        last_y = max(w.y1 for w in r)
    if len(keep) < 2:
        return None
    bounds = columns(keep)
    cells = []
    for r in keep:
        row = [""] * (len(bounds) + 1)
        for w in r:
            k = int(np.searchsorted(bounds, w.cx))
            row[k] = f"{row[k]} {w.text}".strip()
        cells.append(row)
    counts = [sum(bool(c) for c in row) for row in cells]
    mode = max(set(counts), key=counts.count)
    conf = counts.count(mode) / len(counts)
    box = (min(w.x0 for r in keep for w in r), cap["y0"], max(w.x1 for r in keep for w in r),
           max(w.y1 for r in keep for w in r))
    return dict(label=cap["label"], caption=cap["text"], rows=cells, ncols=len(bounds) + 1,
                confidence=round(conf, 3), box=[round(v) for v in box],
                status="auto" if conf >= 0.8 and len(bounds) >= 1 else "needs_review")
