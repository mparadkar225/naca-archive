"""The automatic digitizer on a real scan: NACA Report 747 against the curves digitized by
hand-calibrated tracing in pyxrotor (validation/naca747/tests.json).

Runs only when the report has been harvested and processed (it is in the propeller
corpus): set NACA_DATA to the data directory, or it looks in ./data.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

DATA = Path(os.environ.get("NACA_DATA", Path(__file__).parents[1] / "data"))
TRUTH = json.loads((Path(__file__).parent / "fixtures" / "naca747_tests.json").read_text())


def _report():
    recs = DATA / "metadata" / "records.jsonl"
    if not recs.exists():
        return None
    for line in recs.read_text().splitlines():
        r = json.loads(line)
        if r.get("_number") == "NACA-TR-747":
            p = DATA / "reports" / str(r["id"]) / "report.json"
            return json.loads(p.read_text()) if p.exists() else None
    return None


REPORT = _report()
pytestmark = pytest.mark.skipif(REPORT is None, reason="NACA Report 747 not processed in NACA_DATA")


def _match(fig_label, xs, ys):
    """Best-matching automatic curve among figures whose label starts with the figure
    number: mean |error| over the overlap, and the overlap fraction."""
    num = fig_label.removeprefix("fig").rstrip("abc")
    best = (np.inf, 0.0, None)
    for f in REPORT["figures"]:
        if (f.get("label") or "").split("(")[0] != num:
            continue
        for c in f.get("curves") or []:
            if "x" not in c:
                continue
            cx, cy = np.array(c["x"]), np.array(c["y"])
            o = np.argsort(cx)
            cx, cy = cx[o], cy[o]
            m = (xs >= cx[0]) & (xs <= cx[-1])
            if m.sum() < 5:
                continue
            err = float(np.mean(np.abs(np.interp(xs[m], cx, cy) - ys[m])))
            if err < best[0]:
                best = (err, m.mean(), f["key"])
    return best


def test_auto_digitizing_matches_hand_digitized_curves():
    rows, good = [], 0
    for c in TRUTH["curves"]:
        if c["figure"] == "fig12":
            continue            # solid/dashed front/rear pairs: needs line-style separation
        xs, ys = np.array(c["J"]), np.array(c["value"])
        err, cover, key = _match(c["figure"], xs, ys)
        step = 0.01 if max(ys) < 0.15 else 0.02
        ok = err < 0.3 * step and cover > 0.6
        good += ok
        rows.append(f"{c['figure']} {c['quantity']} {c['beta_front']:>4}: err {err:.4f} cover {cover:.2f} {key} {'ok' if ok else 'MISS'}")
    print("\n".join(rows))
    assert good >= 0.6 * len(rows), f"{good}/{len(rows)} curves matched"
