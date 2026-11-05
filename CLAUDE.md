# Working on naca-archive

## Rules
- NTRS terms: stay under 500 requests / 15 min (RateLimiter default 450), apply
  redistribution notices, never commit PDFs or put them on the site, link to
  ntrs.nasa.gov/citations/{id}. The `data` branch holds JSON only; PDFs and figure images
  live in the Actions cache.
- Bump `PIPELINE_VERSION` in `nacaarchive/__init__.py` whenever processing output changes,
  so older reports are redone.
- Review corrections (`reviews/`) are human decisions: never edit or delete them to make a
  test pass. Processing must keep applying them.
- Test digitizer changes on synthetic scans (`tests/test_autodigitize.py`, known truth) and,
  when data is available, on NACA 747 (`NACA_DATA=data pytest -s tests/test_naca747.py`).
  Look at the overlay images; a passing number can hide a curve on the wrong branch.

## Known pitfalls
- The sandbox where this was written can't reach ntrs.nasa.gov (proxy policy); harvesting
  runs in GitHub Actions or locally. The API field formats (otherReportNumbers, dates) were
  taken from NASA's OpenAPI documentation and not yet seen live: check the first harvest's
  records.jsonl and adjust `harvest.report_number` / `harvest.year` if needed.
- Landscape charts are printed sideways (common in NASA TN D-series): chart pages are turned
  upright by `pages.orientation` (sparse OCR at each quarter turn; Tesseract's OSD guessed
  wrong on figure pages).
- Short tick labels (".5") and labels touching the frame are missed by sparse OCR:
  `autodigitize.strip_words` reads them one at a time. In italic lettering the decimal point
  sits under the digit's foot (`leading_dot`).
- To look at real crops from here, run the debug-figs workflow (pushes crops of chosen
  reports to a temporary `debug-figs` branch); the Actions cache and artifacts can't be
  downloaded through this sandbox's proxy, and the Pages site isn't reachable either.
- Tesseract drops leading decimal points (".02" -> "02"): `autodigitize.leading_dot`.
- OCR over a grid is garbage: text inside a plot is found from small ink components after
  whitening grid lines, then OCR'd word by word.
- Grid detection needs the page deskewed first (a 0.35° skew is 12 px over a chart width).
- sql.js has FTS4 but not FTS5. FTS4's porter tokenizer treats non-ASCII as letters, so
  indexed text goes through `db.fts_text` (and the site normalizes queries the same way).
- Charts that were stretched unevenly: values interpolate between detected grid lines
  (pyxrotor's NACA 747 fig. 13c was off by up to 4 px with a single linear fit).
- Digitizing lessons from pyxrotor: a crop that misses the zero grid line shifts every value
  by one step (calibration here uses labels, not the crop); check CT·J/CP ≤ 1 when both are
  digitized.
