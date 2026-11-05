"""PDF pages to images and words with pixel boxes.

NTRS scans of NACA reports usually carry an OCR text layer of uneven quality. A page's
text layer is used when it looks like real text; otherwise (or when ``force_ocr``) the
page is OCR'd with Tesseract. Word boxes are in pixels of the page rendered at ``DPI``.
"""
from __future__ import annotations

import csv
import io
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DPI = 300
_WORDLIKE = re.compile(r"^[A-Za-z][a-z]{2,}[.,;:]?$")


@dataclass
class Word:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    conf: float = 100.0
    line: int = 0          # line id, unique within a page

    @property
    def cx(self):
        return 0.5 * (self.x0 + self.x1)

    @property
    def cy(self):
        return 0.5 * (self.y0 + self.y1)


@dataclass
class Page:
    number: int            # 1-based
    image: np.ndarray      # grayscale uint8, 0 = black
    words: list[Word]
    source: str            # "text-layer" or "ocr"

    @property
    def text(self) -> str:
        return words_to_text(self.words)


def words_to_text(words: list[Word]) -> str:
    lines: dict[int, list[Word]] = {}
    for w in words:
        lines.setdefault(w.line, []).append(w)
    rows = sorted(lines.values(), key=lambda ws: (min(w.y0 for w in ws), min(w.x0 for w in ws)))
    return "\n".join(" ".join(w.text for w in sorted(ws, key=lambda w: w.x0)) for ws in rows)


def text_quality(words: list[Word]) -> float:
    """Fraction of alphabetic tokens that look like ordinary words (0 when there are
    fewer than 20 tokens: a figure page, or no text layer)."""
    alpha = [w.text for w in words if any(c.isalpha() for c in w.text)]
    if len(alpha) < 20:
        return 0.0
    return sum(bool(_WORDLIKE.match(t)) for t in alpha) / len(alpha)


def tesseract(img: np.ndarray | Image.Image, psm: int = 3, whitelist: str | None = None,
              line_offset: int = 0) -> list[Word]:
    """Tesseract words (TSV output) with boxes in the image's pixels."""
    im = img if isinstance(img, Image.Image) else Image.fromarray(img)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "in.png"
        im.save(p)
        cmd = ["tesseract", str(p), "stdout", "--psm", str(psm), "-l", "eng", "tsv"]
        if whitelist:
            cmd[-1:-1] = ["-c", f"tessedit_char_whitelist={whitelist}"]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    words = []
    for r in csv.DictReader(io.StringIO(out), delimiter="\t", quoting=csv.QUOTE_NONE):
        t = (r.get("text") or "").strip()
        if not t or float(r["conf"]) < 0:
            continue
        x, y, w, h = (float(r[k]) for k in ("left", "top", "width", "height"))
        line = line_offset + int(r["block_num"]) * 10000 + int(r["par_num"]) * 100 + int(r["line_num"])
        words.append(Word(t, x, y, x + w, y + h, float(r["conf"]), line))
    return words


_DICTWORD = re.compile(r"^[A-Za-z][a-z]{2,}[.,;:]?$")


def orientation(img: np.ndarray) -> int:
    """Quarter turns (counterclockwise, as np.rot90's k) that make a page's text upright:
    0, 1 or 3. Landscape charts are printed sideways on portrait pages; Tesseract's own
    orientation detection is unreliable on figure pages with little text, so sparse OCR
    is tried at each turn of a half-size image and the turn that reads the most ordinary
    words wins (it must clearly beat upright)."""
    small = np.asarray(Image.fromarray(img).reduce(2))

    def score(k):
        ws = tesseract(np.rot90(small, k), psm=11)
        return sum(w.conf for w in ws if w.conf > 60 and _DICTWORD.match(w.text))

    s0 = score(0)
    k, best = max(((k, score(k)) for k in (1, 3)), key=lambda t: t[1])
    return k if best > 150 and best > 2 * s0 else 0


def open_pages(pdf: Path, force_ocr: bool = False, min_quality: float = 0.6, dpi: int = DPI):
    """Yield Page objects for each page of the PDF."""
    import pymupdf as fitz

    doc = fitz.open(pdf)
    scale = dpi / 72.0
    for i, pg in enumerate(doc):
        pix = pg.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
        words, source = [], "ocr"
        if not force_ocr:
            raw = pg.get_text("words")  # x0, y0, x1, y1, text, block, line, word
            words = [Word(t, x0 * scale, y0 * scale, x1 * scale, y1 * scale, 100.0, b * 1000 + ln)
                     for x0, y0, x1, y1, t, b, ln, _ in raw]
            if text_quality(words) >= min_quality:
                source = "text-layer"
            else:
                words = []
        if source == "ocr":
            words = tesseract(img)
        yield Page(i + 1, img, words, source)
    doc.close()
