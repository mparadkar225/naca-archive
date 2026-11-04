"""Synthetic NACA-style scanned pages with known truth, for testing the pipeline.

A page has a text paragraph, one chart (fine gray grid, labels on major lines, thick
curves with angle labels, a "FIGURE n.-" caption) and optionally a table; it is rendered
at 300 dpi and degraded like a scan (skew, stretch, blur, noise, JPEG).
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

DPI = 300
PARA = ("The propellers were tested in the propeller-research tunnel at several blade angles. "
        "Thrust and power coefficients are given against the advance-diameter ratio V/nD, and "
        "the efficiency is computed from them. Corrections for tunnel-wall interference were "
        "found to be small and have not been applied.")


@dataclass
class Chart:
    x: tuple = (0.0, 1.6, 0.1, 0.2)       # min, max, grid step, label step
    y: tuple = (0.0, 0.16, 0.01, 0.02)
    xlabel: str = "V/nD"
    ylabel: str = "CP"
    curves: dict = field(default_factory=dict)  # label -> (x array, y array)
    dashed: tuple = ()


def default_chart(seed=0) -> Chart:
    rng = np.random.default_rng(seed)
    c = Chart()
    for k, beta in enumerate((15, 20, 25, 30, 35)):
        j0 = 0.25 + 0.21 * k
        j1 = min(j0 + 0.55 + 0.05 * k, 1.58)
        j = np.linspace(j0, j1, 60)
        cp0 = 0.04 + 0.022 * k + rng.uniform(-0.003, 0.003)
        y = cp0 * (1 - ((j - j0) / (j1 - j0 + 0.12)) ** 2.2)
        c.curves[f"{beta}°"] = (j, y)
    return c


def render_page(chart: Chart, figure_no=9, table=False, seed=0, skew_deg=0.35, stretch=1.004):
    """Returns (PIL grayscale page image, truth dict). truth['to_px'](x, y) maps data
    coordinates to scanned-page pixels; truth['box'] is the plot area's pixel box."""
    fig = plt.figure(figsize=(8.5, 11), dpi=DPI)
    fig.text(0.1, 0.93, _wrap(PARA, 95), fontsize=10, family="serif", va="top")
    ax = fig.add_axes([0.16, 0.42, 0.74, 0.38])
    (x0, x1, xs, xl), (y0, y1, ys, yl) = chart.x, chart.y
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    for v in np.arange(x0, x1 + xs / 2, xs):
        ax.axvline(v, color="0.55", lw=0.6, zorder=0)
    for v in np.arange(y0, y1 + ys / 2, ys):
        ax.axhline(v, color="0.55", lw=0.6, zorder=0)
    ax.set_xticks(np.arange(x0, x1 + xl / 2, xl))
    ax.set_yticks(np.arange(y0, y1 + yl / 2, yl))
    fmt = lambda s: (lambda v, _: f"{v:.{max(0, -int(np.floor(np.log10(s))))}f}".replace("0.", ".", 1)
                     if 0 < abs(v) < 1 else f"{v:g}")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt(xl)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt(yl)))
    ax.tick_params(length=0, labelsize=10, pad=6)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_family("serif")
    for s in ax.spines.values():
        s.set_color("0.55")
        s.set_linewidth(0.6)
    ax.set_xlabel(chart.xlabel, family="serif", fontsize=11)
    ax.set_ylabel(chart.ylabel, family="serif", fontsize=11)
    for k, (lab, (cx, cy)) in enumerate(chart.curves.items()):
        ax.plot(cx, cy, "k", lw=1.6, ls="--" if lab in chart.dashed else "-")
        i = int(len(cx) * 0.15)
        ax.text(cx[i], cy[i] + 0.035 * (y1 - y0), lab, family="serif", fontsize=10, ha="center")
    fig.text(0.5, 0.36, f"FIGURE {figure_no}.—Power coefficient against V/nD for several blade angles.",
             ha="center", family="serif", fontsize=10)
    if table:
        rows = [("Blade angle", "Peak η", "V/nD"), ("15°", "0.81", "0.62"), ("20°", "0.84", "0.81"),
                ("25°", "0.85", "1.02"), ("30°", "0.86", "1.21")]
        fig.text(0.5, 0.30, "TABLE I.—PEAK EFFICIENCIES", ha="center", family="serif", fontsize=10)
        for r, row in enumerate(rows):
            for c_, cell in enumerate(row):
                fig.text(0.28 + 0.2 * c_, 0.27 - 0.022 * r, cell, family="serif", fontsize=10)
    fig.canvas.draw()
    # data -> rendered pixel (origin top-left)
    H = fig.get_figheight() * DPI
    trans = ax.transData

    def to_render_px(x, y):
        p = trans.transform(np.column_stack([np.ravel(x), np.ravel(y)]))
        return p[:, 0], H - p[:, 1]

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=DPI)
    plt.close(fig)
    img = Image.open(buf).convert("L")
    # scan degradation: stretch, skew about the page center, blur, noise, jpeg
    W0, H0 = img.size
    img = img.resize((W0, int(H0 * stretch)), Image.BICUBIC)
    img = img.rotate(skew_deg, resample=Image.BICUBIC, fillcolor=255)
    cx_, cy_ = img.size[0] / 2, img.size[1] / 2
    th = np.deg2rad(skew_deg)

    def to_px(x, y):
        u, v = to_render_px(x, y)
        v = v * stretch
        du, dv = u - cx_, v - cy_
        # PIL rotates counter-clockwise for positive angles (image y down)
        return cx_ + du * np.cos(th) + dv * np.sin(th), cy_ - du * np.sin(th) + dv * np.cos(th)

    rng = np.random.default_rng(seed)
    a = np.asarray(img.filter(ImageFilter.GaussianBlur(0.9)), float)
    a = a * 0.93 + 12 + rng.normal(0, 7, a.shape)          # yellowed paper, grain
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=70)
    img = Image.open(io.BytesIO(b.getvalue())).convert("L")
    xs_, ys_ = to_px([x0, x1], [y1, y0])
    truth = {"to_px": to_px, "box": (xs_[0], ys_[0], xs_[1], ys_[1]), "chart": chart,
             "figure": figure_no}
    return img, truth


def make_pdf(path, pages):
    """Image-only PDF (no text layer), like a scanned NTRS file."""
    ims = [p.convert("RGB") for p in pages]
    ims[0].save(path, "PDF", resolution=DPI, save_all=True, append_images=ims[1:])


def _wrap(s, n):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > n:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return "\n".join(lines + [cur])
