"""Whole pipeline on a two-page synthetic scanned PDF: process, build the database, search."""
import json
import sqlite3

import pytest

import synth
from nacaarchive.db import build, connect
from nacaarchive.process import needs_processing, process_report


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    p1, _ = synth.render_page(synth.default_chart(0), figure_no=9, table=True, seed=0)
    p2, _ = synth.render_page(synth.default_chart(1), figure_no=10, seed=1, skew_deg=-0.6)
    (d / "pdfs").mkdir()
    pdf = d / "pdfs" / "1.pdf"
    synth.make_pdf(pdf, [p1, p2])
    rec = {"id": 1, "title": "Tests of dual-rotating propellers", "_number": "NACA-TR-747", "_year": 1942,
           "abstract": "Wind-tunnel tests.", "authorAffiliations": [{"meta": {"author": {"name": "Biermann, D."}}}]}
    # a review correction is applied on processing
    (d / "reviews" / "1").mkdir(parents=True)
    (d / "reviews" / "1" / "p2-0.json").write_text(json.dumps({"status": "accepted", "labels": {"1": "20°"}}))
    report = process_report(pdf, rec, d, log=lambda *a: None)
    db = d / "naca.db"
    counts = build(d, db)
    return d, report, db, counts


def test_report(built):
    d, report, _, counts = built
    assert counts == {"reports": 1, "pages": 2, "figures": 2, "curves": 10, "points": 0, "tables": 1}
    figs = {f["label"]: f for f in report["figures"]}
    assert set(figs) == {"9", "10"} and figs["9"]["status"] == "auto"
    assert figs["10"]["status"] == "accepted" and figs["10"]["curves"][1]["label"] == "20°"
    assert (d / "reports" / "1" / "fig" / "p1-0-check.jpg").exists()
    assert not needs_processing(d, "1")


def test_search(built):
    _, _, db, _ = built
    con = connect(db)
    hits = con.execute("SELECT kind, title FROM search WHERE search MATCH 'power coefficient' ORDER BY rank(matchinfo(search, 'pcx')) DESC").fetchall()
    assert ("figure", "Figure 9") in hits and ("figure", "Figure 10") in hits
    assert con.execute("SELECT count(*) FROM search WHERE search MATCH 'efficiencies' AND kind='table'").fetchone()[0] == 1
    # the 15° curve, found by position (labels are best-effort OCR and vary with Tesseract version)
    curves = [(json.loads(x), json.loads(y)) for x, y in con.execute(
        "SELECT c.x, c.y FROM curves c JOIN figures f ON f.id=c.figure_id WHERE f.label='9'")]
    x, y = min(curves, key=lambda c: c[0][0])
    assert 0.24 < x[0] < 0.30 and 0.035 < max(y) < 0.047
