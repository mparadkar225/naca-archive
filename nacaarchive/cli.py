"""Command line: naca {harvest,redistributions,download,process,build-db,site,review}."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="naca")
    ap.add_argument("--data", type=Path, default=ROOT / "data", help="data directory")
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="search NTRS and update metadata/records.jsonl")
    h.add_argument("--queries", type=Path, default=ROOT / "queries" / "propellers.toml")
    h.add_argument("--limit", type=int, help="max results per query (testing)")

    sub.add_parser("redistributions", help="apply NTRS redistribution notices (run weekly)")

    d = sub.add_parser("download", help="download PDFs of harvested records")
    d.add_argument("--pdfs", type=Path, default=None, help="PDF cache (default data/pdfs)")
    d.add_argument("--max-new", type=int, help="download at most this many new PDFs")
    d.add_argument("--ids", nargs="*", help="only these NTRS ids")
    d.add_argument("--match", help="only records whose title or abstract matches this regex")

    p = sub.add_parser("process", help="extract text, figures, tables from downloaded PDFs")
    p.add_argument("--pdfs", type=Path, default=None)
    p.add_argument("--ids", nargs="*", help="only these NTRS ids")
    p.add_argument("--match", help="only records whose title or abstract matches this regex")
    p.add_argument("--all", action="store_true", help="reprocess even if up to date")
    p.add_argument("--force-ocr", action="store_true")
    p.add_argument("--reviews", type=Path, default=ROOT / "reviews", help="review corrections")
    p.add_argument("--max", type=int, help="process at most this many reports")
    p.add_argument("--budget-min", type=float, help="stop starting new reports after this many minutes")

    b = sub.add_parser("build-db", help="build the SQLite database")
    b.add_argument("--out", type=Path, default=ROOT / "build" / "naca.db")

    s = sub.add_parser("site", help="assemble the static web site (database + figure images)")
    s.add_argument("--db", type=Path, default=ROOT / "build" / "naca.db")
    s.add_argument("--out", type=Path, default=ROOT / "build" / "site")

    r = sub.add_parser("review", help="list figures that need review")
    r.add_argument("--limit", type=int, default=50)

    a = ap.parse_args(argv)
    data: Path = a.data

    if a.cmd == "harvest":
        from .harvest import harvest
        harvest(a.queries, data, limit=a.limit)
    elif a.cmd == "redistributions":
        from .harvest import check_redistributions
        check_redistributions(data)
    elif a.cmd == "download":
        from .harvest import download_all
        download_all(data, a.pdfs or data / "pdfs", max_new=a.max_new, ids=a.ids, match=a.match)
    elif a.cmd == "process":
        from .harvest import _read_jsonl, select
        from .process import needs_processing, process_report
        pdfs = a.pdfs or data / "pdfs"
        recs = _read_jsonl(data / "metadata" / "records.jsonl")
        ids = sorted(select(recs, a.ids, a.match))
        t0, done, failed = time.time(), 0, []
        for rid in ids:
            pdf = pdfs / f"{rid}.pdf"
            if rid not in recs or not pdf.exists() or not (a.all or needs_processing(data, rid)):
                continue
            if a.max is not None and done >= a.max:
                break
            if a.budget_min and (time.time() - t0) / 60 > a.budget_min:
                print(f"time budget reached after {done} reports")
                break
            try:
                process_report(pdf, recs[rid], data, force_ocr=a.force_ocr, reviews=a.reviews,
                               deadline=t0 + 60 * (a.budget_min + 20) if a.budget_min else None)
                done += 1
            except Exception as e:
                failed.append(rid)
                print(f"{rid}: FAILED: {e!r}", file=sys.stderr)
        print(f"processed {done}, failed {len(failed)}")
    elif a.cmd == "build-db":
        from .db import build
        n = build(data, a.out)
        print(json.dumps(n))
    elif a.cmd == "site":
        out: Path = a.out
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "site" / "index.html", out / "index.html")
        shutil.copy(a.db, out / "naca.db")
        for rep in (data / "reports").glob("*"):
            if (rep / "fig").is_dir():
                dst = out / "reports" / rep.name / "fig"
                shutil.copytree(rep / "fig", dst, dirs_exist_ok=True)
        print(f"site in {out}")
    elif a.cmd == "review":
        rows = []
        for rp in sorted((data / "reports").glob("*/report.json")):
            r = json.loads(rp.read_text())
            for f in r["figures"]:
                if f["status"] == "needs_review":
                    rows.append((r.get("number") or r["id"], f["key"], f.get("label"), "; ".join(f.get("reasons") or [])))
        for row in rows[: a.limit]:
            print(" | ".join(str(x) for x in row))
        print(f"{len(rows)} figures need review")


if __name__ == "__main__":
    main()
