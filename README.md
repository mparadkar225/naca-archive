# naca-archive

A searchable archive of data extracted from NACA and NASA reports. It finds reports on the
NASA Technical Reports Server (NTRS), downloads them, and turns the scans into:

- **searchable text**: the PDF's own text layer when it reads well, Tesseract OCR otherwise;
- **digitized charts**: gridded figures are calibrated from their tick labels and their
  curves traced into x, y data, with a confidence verdict and an overlay image for checking;
- **tables**: rows and columns recovered from the word layout;
- **one SQLite database** with full-text search over titles, abstracts, page text, figure
  captions and tables, plus a static web page to search and browse it.

The first corpus is propellers (NACA 1915–1958, and NASA propeller and prop-fan work), chosen
to feed [pyxrotor](https://github.com/mparadkar225/pyxrotor) validation. Other topics are a new
query file in `queries/`.

## How it runs

```
GitHub Actions (daily, harvest.yml)
  naca redistributions   apply NTRS withdrawal notices (required at least weekly)
  naca harvest           search NTRS with queries/propellers.toml, keep on-topic records
  naca download          new PDFs into the Actions cache (never into git)
  naca process           text, figures, tables per report -> data branch (JSON)
  naca build-db / site   SQLite + web page -> GitHub Pages
```

The harvester stays under NTRS's documented limits (500 requests per 15 minutes, 100
results per page, 10,000 per query) and links every report to its NTRS citation page.

Locally:

```
pip install -e ".[test]"        # needs tesseract on PATH
naca harvest --limit 20         # data/metadata/records.jsonl
naca download --max-new 5       # data/pdfs/
naca process                    # data/reports/{id}/
naca build-db && naca site      # build/naca.db, build/site/
naca review                     # figures the digitizer wasn't sure of
```

## The database

```python
from nacaarchive.db import connect
con = connect("build/naca.db")
con.execute("""
  SELECT r.number, f.label, f.caption, f.status FROM search s
    JOIN figures f ON s.kind = 'figure' AND f.id = s.ref
    JOIN reports r ON r.id = f.report_id
   WHERE search MATCH 'dual rotating power coefficient'
   ORDER BY rank(matchinfo(search, 'pcx')) DESC""").fetchall()
```

Curves are in `curves(figure_id, label, x, y)` as JSON arrays in the chart's units. The
schema is in `nacaarchive/db.py`. The index is FTS4 because the browser's sql.js has no
FTS5; `connect` registers the same `rank` function the web page uses.

## Automatic digitizing

`nacaarchive/autodigitize.py`, built on the chart tools from pyxrotor's NACA 747 work
(`nacaarchive/chart.py` is pyxrotor's `tools/digitize.py`):

1. Deskew the page from its long lines; find charts as clusters of regular grid lines.
2. OCR the tick labels beside each axis. NACA labels are written ".02": Tesseract drops the
   leading point, so a point is looked for in the pixels just left of each number.
3. Fit labels to grid lines robustly (a misread label is outvoted), snap the step to
   1/2/2.5/5 × 10ⁿ, give every grid line a value, and convert pixels to values by
   interpolating between lines, so uneven scan stretch doesn't bias readings.
4. Mask grid lines and text, bridge curves across the masked bands, link ink runs column
   by column into tracks that coast through crossings and dashes, rejoin broken tracks.
5. Read labels inside the plot ("20°") and attach each to the nearest curve.
6. Find data-point symbols (hollow outlines, usually with a centre dot, or solid blobs), group
   them by shape, and give each kind a table of x, y values: for test-data charts the points
   are the data and the faired curve is secondary. A curve through symbols is traced with
   the symbols cut out.

A figure is `auto` when both axes calibrate from at least three consistent labels and the
tracks cover at least 85% of the curve ink; otherwise `needs_review`, with the reasons.
On synthetic scans (skew up to 1°, stretch up to 1.2%, blur, noise, JPEG) curves come out
within 0.2% of full scale. The real-scan check is `tests/test_naca747.py`: it compares the
automatic curves of NACA Report 747 with pyxrotor's hand-calibrated digitizing, and runs in
the harvest workflow once the report is processed.

Data points are in `points(figure_id, idx, symbol, name, n, x, y)`; the site shows each kind
as a table and as markers on the plot, with a CSV download. On synthetic charts points come
out within 0.1% of full scale; overlapping symbols can be missed or come out as a separate
odd kind, and a chart with many of those goes to review.

Not handled yet: charts without grids (indexed with caption and image, not digitized),
log scales, matching symbols to the chart's legend (name them in a review), solid/dashed pairs (747 fig. 12), two y scales on one chart
(flagged for review).

## Reviewing

Every figure page on the site shows the reading drawn on the scan and a form that writes a
correction: accept or reject, replace an axis calibration, rename or drop curves, name a
kind of data point from the legend (`"symbols": {"0": "Curtiss 714"}`), merge two kinds
(`"merge_points": [[0, 3]]`) or drop one (`"drop_points": [3]`). Save it as
`reviews/{report id}/{figure key}.json` on `main`; corrections are applied every time the
report is processed.

## Data and terms

Source: NASA STI Compliance and Distribution Services (NTRS). PDFs are not stored in this
repository or on the site; each report links to `https://ntrs.nasa.gov/citations/{id}`.
Records withdrawn by NTRS are removed with everything derived from them.

## Setup on GitHub

1. Settings → Pages → Source: GitHub Actions.
2. Run the *harvest* workflow once by hand (Actions → harvest → Run workflow); after that it
   runs daily until the corpus is processed, then keeps it current.

GPL v3 or later (includes code from pyxrotor).
