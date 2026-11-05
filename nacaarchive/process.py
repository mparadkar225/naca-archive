"""Process one report: pages -> text, figures (digitized when gridded), tables.

Output in data/reports/{id}/:
  report.json   metadata, figures (with curves), tables, per-page text source
  pages.jsonl   {"page", "text", "source"} per page
  fig/          {key}.jpg (figure crop) and {key}-check.jpg (digitizing overlay)

Review corrections in reviews/{id}/{key}.json (kept in git on main; default data/reviews) are applied on every run, so
reprocessing with a newer pipeline keeps them. A correction may set:
  "status": "accepted" | "rejected"
  "x": [value_of_first_grid_line, step]   "y": [...]   (replaces the OCR calibration)
  "labels": {"0": "20°", ...}   curve index -> label
  "drop": [2, 5]                curve indices to discard
"""
from __future__ import annotations

import json
import re
import signal
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import PIPELINE_VERSION
from . import autodigitize as ad
from . import figures as F
from . import tables as T
from .ntrs import citation_url
from .pages import Word, open_pages, orientation, tesseract


def _rotate_words(words: list[Word], shape, deg: float):
    if abs(deg) < 0.02:
        return
    h, w = shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    for wd in words:
        pts = np.array([[wd.x0, wd.y0, 1], [wd.x1, wd.y1, 1]]) @ m.T
        wd.x0, wd.y0 = pts[0]
        wd.x1, wd.y1 = pts[1]


def figure_region(img: np.ndarray, words: list[Word], cap: dict) -> tuple | None:
    """Approximate box of an ungridded figure above its caption: ink that isn't page
    text, clustered, nearest above the caption."""
    H, W = img.shape
    dark = (img < 170).astype(np.uint8)
    for w in words:
        if w.y0 >= cap["y0"] - 2 and w.y1 <= cap["y1"] + 2:
            continue
        dark[int(max(w.y0 - 2, 0)):int(w.y1 + 3), int(max(w.x0 - 2, 0)):int(w.x1 + 3)] = 0
    dark[int(cap["y0"]) - 2:, :] = 0
    dark = cv2.dilate(dark, np.ones((35, 35), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(dark)
    best = None
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if w < 0.15 * W or h < 0.05 * H or y + h < cap["y0"] - 0.6 * H:
            continue
        if best is None or y + h > best[1] + best[3]:
            best = (x, y, w, h)
    if best is None:
        return None
    x, y, w, h = best
    return (int(x), int(y), int(x + w), int(y + h))


def _apply_review(fig: dict, rev: dict, res: ad.Result | None):
    if not rev:
        return
    fig["review"] = rev
    if res is not None and ("x" in rev or "y" in rev):
        g = res.grid
        for key, lines in (("x", g.vlines), ("y", g.hlines)):
            if key in rev:
                v0, step = rev[key]
                vals = v0 + step * np.round((lines - lines[0]) / np.median(np.diff(lines)))
                ax = ad.Axis(np.asarray(lines, float), vals, [], 99, step, 0.0, "review")
                if key == "x":
                    res.x_axis = ax
                else:
                    res.y_axis = ax
        if res.x_axis is not None and res.y_axis is not None:
            for c in fig["curves"] + (fig.get("points") or []):
                c["x"] = [round(float(v), 6) for v in res.x_axis.to_val(c["px"])]
                c["y"] = [round(float(v), 6) for v in res.y_axis.to_val(c["py"])]
    for k, lab in (rev.get("labels") or {}).items():
        if int(k) < len(fig["curves"]):
            fig["curves"][int(k)]["label"] = lab
    drop = set(rev.get("drop") or [])
    fig["curves"] = [c for i, c in enumerate(fig["curves"]) if i not in drop]
    # data-point tables: name a symbol series (from the chart's legend), merge or drop series
    pts = fig.get("points") or []
    for k, name in (rev.get("symbols") or {}).items():
        if int(k) < len(pts):
            pts[int(k)]["name"] = name
    for a, b in rev.get("merge_points") or []:          # [into, from]
        if a < len(pts) and b < len(pts) and a != b:
            for key in ("px", "py", "x", "y"):
                if key in pts[a] and key in pts[b]:
                    pts[a][key] = pts[a][key] + pts[b][key]
            order = np.argsort(pts[a]["px"])
            for key in ("px", "py", "x", "y"):
                if key in pts[a]:
                    pts[a][key] = [pts[a][key][i] for i in order]
            pts[a]["n"] = len(pts[a]["px"])
            pts[b]["n"] = 0
    dropp = set(rev.get("drop_points") or [])
    fig["points"] = [p for i, p in enumerate(pts) if i not in dropp and p["n"] > 0]
    if rev.get("status") in ("accepted", "rejected"):
        fig["status"] = rev["status"]


CHART_SECONDS = 120   # a chart that takes longer is left for review (one took hours on WR L-594)


class ChartTimeout(Exception):
    pass


def _digitize_limited(img, g, seconds=CHART_SECONDS):
    """ad.digitize with a wall-clock limit (SIGALRM; main thread on Unix only)."""
    if seconds is None or not hasattr(signal, "SIGALRM"):
        return ad.digitize(img, g)

    def stop(*_):
        raise ChartTimeout(f"digitizing took over {seconds} s")
    old = signal.signal(signal.SIGALRM, stop)
    signal.alarm(int(seconds))
    try:
        return ad.digitize(img, g)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def process_report(pdf: Path, record: dict, data: Path, force_ocr: bool = False, log=print,
                   reviews: Path | None = None, deadline: float | None = None) -> dict:
    """deadline (time.time() value): charts reached after it are kept for review, not
    digitized, so a run's time budget also holds inside a long report."""
    rid = str(record["id"])
    out = data / "reports" / rid
    (out / "fig").mkdir(parents=True, exist_ok=True)
    reviews = (reviews or data / "reviews") / rid
    figs, tabs, page_rows = [], [], []
    used: set[str] = set()

    def unique(key: str) -> str:
        """A page can carry two captions with the same label (a caption and a line that
        starts with "Figure 3."): keys must still be unique within the report."""
        k, n = key, 2
        while k in used:
            k, n = f"{key}-{n}", n + 1
        used.add(k)
        return k

    t_report = time.time()
    for page in open_pages(pdf, force_ocr=force_ocr):
        t_page = time.time()
        img, turn = page.image, 0
        ang = F.deskew_angle(img)
        upright = F.rotate(img, ang)
        grids = F.find_grids(upright)
        if grids:
            # a chart page: landscape charts are printed sideways, so turn the page upright
            # and read it again (tick labels and captions must be upright for OCR)
            turn = orientation(img)
            if turn:
                img = np.ascontiguousarray(np.rot90(img, turn))
                page.words, page.source = tesseract(img), "ocr"
                ang = F.deskew_angle(img)
                upright = F.rotate(img, ang)
                grids = F.find_grids(upright)
        img = upright
        _rotate_words(page.words, img.shape, ang)
        page_rows.append({"page": page.number, "text": page.text, "source": page.source,
                          "skew": round(ang, 3), "turn": 90 * turn})
        caps = F.captions(page.words)
        F.attach_captions(grids, caps)
        claimed = set()
        results = []
        for g in grids:
            if deadline is not None and time.time() > deadline:
                results.append((None, "time budget reached before this chart"))
                continue
            try:
                results.append((_digitize_limited(img, g), None))
            except Exception as e:  # keep going: one bad chart must not lose the report
                results.append((None, e))
        ad.share_axes([r for r, _ in results if r is not None])
        for k, (g, (res, err)) in enumerate(zip(grids, results)):
            key = unique(f"p{page.number}-{k}")
            if res is not None:
                fig = res.as_dict()
            else:
                fig = dict(box=list(g.box), label=g.label, caption=g.caption, curves=[],
                           status="needs_review", reasons=[err if isinstance(err, str) else f"digitizer error: {err!r}"])
            fig.update(key=key, page=page.number, kind="chart")
            claimed.add(g.label)
            rev_p = reviews / f"{key}.json"
            _apply_review(fig, json.loads(rev_p.read_text()) if rev_p.exists() else {}, res)
            rx0, ry0, rx1, ry1 = g.region
            Image.fromarray(img[ry0:ry1, rx0:rx1]).save(out / "fig" / f"{key}.jpg", quality=80)
            if res is not None:
                ad.overlay(img, res).save(out / "fig" / f"{key}-check.jpg", quality=80)
            figs.append(fig)
        # figures without a grid: indexed (caption + image), not digitized
        for c in caps:
            if c["kind"] == "figure" and c["label"] not in claimed:
                key = unique(f"p{page.number}-c{c['label']}")
                box = figure_region(img, page.words, c)
                if box:
                    Image.fromarray(img[box[1]:box[3], box[0]:box[2]]).save(out / "fig" / f"{key}.jpg", quality=80)
                figs.append(dict(key=key, page=page.number, kind="figure", label=c["label"], caption=c["text"],
                                 box=list(box) if box else None, curves=[], status="not_digitized",
                                 reasons=["no grid found"]))
            elif c["kind"] == "table":
                t = T.table_after(page.words, c)
                if t:
                    t.update(page=page.number, key=unique(f"p{page.number}-t{c['label']}"))
                    tabs.append(t)
        dt = time.time() - t_page
        if dt > 60:
            log(f"{rid}: page {page.number} took {dt:.0f} s ({len(grids)} grids)")
    report = dict(
        id=rid, number=record.get("_number"), title=record.get("title"), year=record.get("_year"),
        authors=[a.get("meta", {}).get("author", {}).get("name") for a in record.get("authorAffiliations") or []],
        abstract=record.get("abstract"), subjects=record.get("subjectCategories") or [],
        url=citation_url(rid), pipeline=PIPELINE_VERSION, pages=len(page_rows),
        text_sources=sorted({p["source"] for p in page_rows}), figures=figs, tables=tabs)
    (out / "pages.jsonl").write_text("".join(json.dumps(p) + "\n" for p in page_rows))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    n_auto = sum(f["status"] in ("auto", "accepted") for f in figs if f["kind"] == "chart")
    n_pts = sum(p["n"] for f in figs for p in f.get("points") or [] if "x" in p)
    charts = [f for f in figs if f["kind"] == "chart"]
    src = Counter(p["source"] for p in page_rows)
    why = Counter(re.sub(r"[\d.]+", "#", r) for f in charts for r in f.get("reasons") or [])
    log(f"{rid} {report['number'] or ''}: {len(page_rows)} pages ({', '.join(f'{v} {k}' for k, v in src.items())}), "
        f"{len(figs)} figures, {len(charts)} with grids ({n_auto} digitized automatically, {n_pts} data points), {len(tabs)} tables"
        + (f"; review: {'; '.join(f'{v}x {k}' for k, v in why.most_common(3))}" if why else "")
        + f" [{(time.time() - t_report) / 60:.1f} min]")
    return report


def needs_processing(data: Path, rid: str) -> bool:
    p = data / "reports" / rid / "report.json"
    if not p.exists():
        return True
    try:
        r = json.loads(p.read_text())
    except json.JSONDecodeError:
        return True
    if r.get("pipeline", 0) < PIPELINE_VERSION:
        return True
    # figure images are kept out of git (CI cache only): redo the report if they're gone
    fig = p.parent / "fig"
    # (charts that hit the time limit have no overlay either; redoing them won't help)
    return any(f["kind"] == "chart" and not (fig / f"{f['key']}-check.jpg").exists()
               and not any("took over" in x for x in f.get("reasons") or []) for f in r["figures"])
