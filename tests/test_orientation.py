"""Landscape charts printed sideways (NASA TN D-995 and others) are turned upright."""
import json

import numpy as np
from PIL import Image

import synth
from nacaarchive.pages import orientation
from nacaarchive.process import process_report


def test_sideways_chart(tmp_path):
    page, _ = synth.render_page(synth.default_chart(0), figure_no=9, seed=0)
    page = np.asarray(page)
    assert orientation(page) == 0
    side = np.ascontiguousarray(np.rot90(page, 1))      # text reads bottom to top
    assert orientation(side) == 3
    (tmp_path / "pdfs").mkdir()
    pdf = tmp_path / "pdfs" / "1.pdf"
    synth.make_pdf(pdf, [Image.fromarray(side)])
    report = process_report(pdf, {"id": 1, "title": "t"}, tmp_path, log=lambda *a: None)
    rows = [json.loads(x) for x in (tmp_path / "reports" / "1" / "pages.jsonl").read_text().splitlines()]
    assert rows[0]["turn"] == 270
    charts = [f for f in report["figures"] if f["kind"] == "chart"]
    assert len(charts) == 1 and charts[0]["status"] == "auto" and charts[0]["label"] == "9"
