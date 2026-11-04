"""Build the SQLite database (with FTS4 full-text search) from data/.

Tables
  reports(id, number, title, year, authors, abstract, subjects, url, pages, pipeline)
  pages(report_id, page, text, source)
  figures(id, report_id, key, page, kind, label, caption, status, reasons, x_axis,
          y_axis, coverage, image, check_image)
  curves(id, figure_id, idx, label, n, x, y)          x, y: JSON arrays in data units
  tables(id, report_id, key, page, label, caption, ncols, rows, status, confidence)
  search  FTS4 (porter stemming) over (kind, report_id, ref, title, body): reports, pages,
          captions, tables. FTS4 rather than FTS5 because the browser's sql.js has only FTS4.
          Rank with rank(matchinfo(search, 'pcx')), registered by ``connect`` (and by the site).

Example
  con = connect("naca.db")
  con.execute('''SELECT r.number, f.label, f.caption FROM search s
    JOIN figures f ON s.kind='figure' AND f.id = s.ref
    JOIN reports r ON r.id = f.report_id
   WHERE search MATCH 'dual rotating power coefficient'
   ORDER BY rank(matchinfo(search, 'pcx')) DESC''')
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE reports(id TEXT PRIMARY KEY, number TEXT, title TEXT, year INTEGER, authors TEXT,
  abstract TEXT, subjects TEXT, url TEXT, pages INTEGER, pipeline INTEGER);
CREATE TABLE pages(report_id TEXT, page INTEGER, text TEXT, source TEXT, PRIMARY KEY(report_id, page));
CREATE TABLE figures(id INTEGER PRIMARY KEY, report_id TEXT, key TEXT, page INTEGER, kind TEXT,
  label TEXT, caption TEXT, status TEXT, reasons TEXT, x_axis TEXT, y_axis TEXT, coverage REAL,
  image TEXT, check_image TEXT, UNIQUE(report_id, key));
CREATE TABLE curves(id INTEGER PRIMARY KEY, figure_id INTEGER, idx INTEGER, label TEXT, n INTEGER,
  x TEXT, y TEXT);
CREATE TABLE tables(id INTEGER PRIMARY KEY, report_id TEXT, key TEXT, page INTEGER, label TEXT,
  caption TEXT, ncols INTEGER, rows TEXT, status TEXT, confidence REAL);
CREATE VIRTUAL TABLE search USING fts4(kind, report_id, ref, title, body, notindexed=kind,
  notindexed=report_id, notindexed=ref, tokenize=porter);
CREATE INDEX figures_report ON figures(report_id);
CREATE INDEX curves_figure ON curves(figure_id);
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
"""


def fts_text(s: str | None) -> str:
    """Text as indexed: FTS4's porter tokenizer treats non-ASCII characters as letters, so
    "9.—Power" would be one token. Accents are dropped, degree signs become "deg",
    other non-ASCII punctuation and symbols become spaces. The site applies the same
    rules to queries."""
    import unicodedata

    if not s:
        return ""
    s = s.replace("°", " deg ")
    out = []
    for ch in unicodedata.normalize("NFKD", s):
        if ord(ch) < 128:
            out.append(ch)
        elif unicodedata.category(ch) == "Mn":
            continue
        elif unicodedata.category(ch)[0] in "PSZC":
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def rank(blob: bytes) -> float:
    """Relevance from matchinfo(search, 'pcx'): for each phrase and column, the row's hits
    divided by the hits in all rows (the example ranking in the SQLite FTS4 docs). Title
    hits count double."""
    import struct

    v = struct.unpack(f"{len(blob) // 4}I", blob)
    p, c = v[0], v[1]
    score = 0.0
    for i in range(p):
        for j in range(c):
            here, total = v[2 + 3 * (i * c + j)], v[2 + 3 * (i * c + j) + 1]
            if here:
                score += (2.0 if j == 3 else 1.0) * here / total
    return score


def connect(path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.create_function("rank", 1, rank, deterministic=True)
    return con


def build(data: Path, out: Path, include_page_text: bool = True) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)
    n = dict(reports=0, pages=0, figures=0, curves=0, tables=0)
    for rp in sorted((data / "reports").glob("*/report.json")):
        r = json.loads(rp.read_text())
        rid = r["id"]
        con.execute("INSERT INTO reports VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (rid, r.get("number"), r.get("title"), r.get("year"), "; ".join(a for a in r.get("authors") or [] if a),
                     r.get("abstract"), "; ".join(r.get("subjects") or []), r.get("url"), r.get("pages"), r.get("pipeline")))
        con.execute("INSERT INTO search VALUES ('report',?,?,?,?)",
                    (rid, rid, fts_text(f"{r.get('number') or ''} {r.get('title') or ''}"), fts_text(r.get("abstract"))))
        n["reports"] += 1
        pj = rp.parent / "pages.jsonl"
        if include_page_text and pj.exists():
            for line in pj.read_text().splitlines():
                p = json.loads(line)
                con.execute("INSERT INTO pages VALUES (?,?,?,?)", (rid, p["page"], p["text"], p["source"]))
                con.execute("INSERT INTO search VALUES ('page',?,?,?,?)", (rid, p["page"], fts_text(r.get("title")), fts_text(p["text"])))
                n["pages"] += 1
        for f in r.get("figures", []):
            xs, ys = f.get("x_side"), f.get("y_side")
            axes = f.get("axes") or {}
            cur = con.execute(
                "INSERT INTO figures(report_id,key,page,kind,label,caption,status,reasons,x_axis,y_axis,coverage,image,check_image)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rid, f["key"], f["page"], f["kind"], f.get("label"), f.get("caption"), f["status"],
                 json.dumps((f.get("reasons") or []) + (f.get("notes") or [])),
                 json.dumps(axes.get(xs)) if xs else None, json.dumps(axes.get(ys)) if ys else None,
                 f.get("coverage"), f"{rid}/fig/{f['key']}.jpg",
                 f"{rid}/fig/{f['key']}-check.jpg" if f["kind"] == "chart" else None))
            fid = cur.lastrowid
            con.execute("INSERT INTO search VALUES ('figure',?,?,?,?)",
                        (rid, fid, f"Figure {f.get('label') or ''}", fts_text(f.get("caption"))))
            n["figures"] += 1
            for i, c in enumerate(f.get("curves") or []):
                if "x" not in c:
                    continue
                con.execute("INSERT INTO curves(figure_id,idx,label,n,x,y) VALUES (?,?,?,?,?,?)",
                            (fid, i, c.get("label"), len(c["x"]), json.dumps(c["x"]), json.dumps(c["y"])))
                n["curves"] += 1
        for t in r.get("tables", []):
            cur = con.execute(
                "INSERT INTO tables(report_id,key,page,label,caption,ncols,rows,status,confidence) VALUES (?,?,?,?,?,?,?,?,?)",
                (rid, t["key"], t["page"], t["label"], t["caption"], t["ncols"], json.dumps(t["rows"]), t["status"], t["confidence"]))
            body = t["caption"] + "\n" + "\n".join(" | ".join(row) for row in t["rows"])
            con.execute("INSERT INTO search VALUES ('table',?,?,?,?)", (rid, cur.lastrowid, f"Table {t['label']}", fts_text(body)))
            n["tables"] += 1
    con.execute("INSERT INTO meta VALUES ('counts', ?)", (json.dumps(n),))
    con.execute("INSERT INTO meta VALUES ('source', ?)",
                ("NASA STI Compliance and Distribution Services (NTRS); data extracted by naca-archive",))
    con.commit()
    con.execute("INSERT INTO search(search) VALUES ('optimize')")
    con.commit()
    con.execute("VACUUM")
    con.close()
    tmp.replace(out)
    return n
